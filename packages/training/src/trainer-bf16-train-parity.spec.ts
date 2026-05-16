import { createParameterRegistry } from '@llm-transformer-diy/autograd';
import {
  GpuContext,
  KernelRegistry,
  probeWebGpu,
  registerCoreKernels,
} from '@llm-transformer-diy/gpu';
import {
  collectGptParameters,
  createGptModel,
  type GptConfig,
  type GptModel,
} from '@llm-transformer-diy/model';
import { destroy, type TensorRuntime } from '@llm-transformer-diy/tensor';
import { beforeAll, describe, expect, it } from 'vitest';

import { AdamW } from './adamw';
import { createLmBatch } from './lm-batch';
import { registerModelParameters, Trainer } from './trainer';

const webgpuAvailable = await probeWebGpu();

/** Matches `tiny` preset scale; vocab bounded for fast steps. */
const tinyCfg: GptConfig = {
  vocabSize: 12,
  maxSeqLen: 16,
  dModel: 32,
  nHeads: 4,
  dFf: 64,
  nLayers: 1,
};

const PARITY_STEPS = 20;

/**
 * OpenSpec add-16bit-dtype-support §7.4: same architecture and optimizer setup as
 * `trainer.spec.ts`, fixed synthetic token stream (indices-only batching). Compares final
 * cross-entropy after **N** steps for **f32** vs **bf16** compute. Shakespeare-sized prose is
 * omitted here for CI speed; the CLI path exercises real corpora end-to-end.
 *
 * **Envelope:** `|loss_f32 − loss_bf16| < 2.5` after **N = 20** — loose enough for rounded
 * bf16 weights at init plus forward/back numeric drift; tighten only if we see stable hardware.
 */
describe.skipIf(!webgpuAvailable)('Trainer f32 vs bf16 parity (GPU)', () => {
  let rt: TensorRuntime;
  let tokenStream: Uint32Array;

  beforeAll(async () => {
    const ctx = await GpuContext.create();
    const registryK = new KernelRegistry();
    registerCoreKernels(ctx.device, registryK);
    rt = { ctx, registry: registryK };
    tokenStream = new Uint32Array(48);
    for (let i = 0; i < tokenStream.length; i++) {
      tokenStream[i] = i % tinyCfg.vocabSize;
    }
  });

  async function lossTraceAfterSteps(
    dtype: 'f32' | 'bf16',
  ): Promise<{ first: number; last: number }> {
    const model: GptModel = createGptModel(rt, tinyCfg, { dtype });
    const reg = createParameterRegistry();
    registerModelParameters(reg, collectGptParameters(model));
    const opt = new AdamW(rt, reg, { lr: 4e-2, weightDecay: 0 });
    const trainer = new Trainer(rt, model, reg, opt, {
      maxGradNorm: 1.0,
      schedule: {
        lrMax: 4e-2,
        lrMin: 1e-3,
        warmupSteps: 3,
        totalSteps: PARITY_STEPS,
      },
    });
    const seqLen = 8;
    const start = 0;
    const { input, targets } = createLmBatch(rt, tokenStream, start, seqLen);
    let first = 0;
    let last = 0;
    for (let s = 0; s < PARITY_STEPS; s++) {
      last = await trainer.trainStep(input, targets, s);
      if (s === 0) {
        first = last;
      }
    }
    destroy(input);
    destroy(targets);
    opt.dispose();
    return { first, last };
  }

  it(`bf16 vs f32 final loss within envelope after N=${String(PARITY_STEPS)} steps`, async () => {
    const f32 = await lossTraceAfterSteps('f32');
    const bf16 = await lossTraceAfterSteps('bf16');
    expect(f32.last).toBeGreaterThan(0);
    expect(bf16.last).toBeGreaterThan(0);
    expect(Math.abs(f32.last - bf16.last)).toBeLessThan(2.5);
  });

  /**
   * Regression: a `bf16`-only bug in `broadcast-fill-scalar.bf16.frag.wgsl` zeroed the upstream
   * scalar gradient out of `sumFlat` backward, freezing the bf16 loss at `~log(vocab)`. Asserting
   * a meaningful drop catches that class of bug — both losses must move down by a clear margin
   * and bf16 must not stall while f32 makes progress.
   */
  it(`bf16 loss drops meaningfully after N=${String(PARITY_STEPS)} steps`, async () => {
    const bf16 = await lossTraceAfterSteps('bf16');
    expect(bf16.first).toBeGreaterThan(0);
    expect(bf16.last).toBeLessThan(bf16.first - 0.25);
  });
});
