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
  gptParameterTensorNames,
} from '@llm-transformer-diy/model';
import { destroy, type TensorRuntime } from '@llm-transformer-diy/tensor';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { beforeAll, describe, expect, it } from 'vitest';

import { AdamW } from './adamw';
import { loadGptCheckpointTensors, saveGptCheckpoint } from './checkpoint';
import { createLmBatch } from './lm-batch';
import { registerModelParameters, Trainer } from './trainer';

const webgpuAvailable = await probeWebGpu();

const tinyCfg: GptConfig = {
  vocabSize: 12,
  maxSeqLen: 16,
  dModel: 32,
  nHeads: 4,
  dFf: 64,
  nLayers: 1,
};

describe.skipIf(!webgpuAvailable)('Trainer (GPU)', () => {
  let rt: TensorRuntime;
  let model: GptModel;
  let tokenStream: Uint32Array;

  beforeAll(async () => {
    const ctx = await GpuContext.create();
    const registryK = new KernelRegistry();
    registerCoreKernels(ctx.device, registryK);
    rt = { ctx, registry: registryK };
    model = createGptModel(rt, tinyCfg);
    tokenStream = new Uint32Array(48);
    for (let i = 0; i < tokenStream.length; i++) {
      tokenStream[i] = i % tinyCfg.vocabSize;
    }
  });

  it('overfits a repeating batch: loss drops', async () => {
    const reg = createParameterRegistry();
    const params = collectGptParameters(model);
    registerModelParameters(reg, params);
    const opt = new AdamW(rt, reg, { lr: 4e-2, weightDecay: 0 });
    const trainer = new Trainer(rt, model, reg, opt, {
      maxGradNorm: 1.0,
      schedule: {
        lrMax: 4e-2,
        lrMin: 1e-3,
        warmupSteps: 3,
        totalSteps: 40,
      },
    });

    const seqLen = 8;
    const start = 0;
    const { input, targets } = createLmBatch(rt, tokenStream, start, seqLen);

    const losses: number[] = [];
    const steps = 25;
    for (let s = 0; s < steps; s++) {
      const L = await trainer.trainStep(input, targets, s);
      losses.push(L);
    }

    destroy(input);
    destroy(targets);

    expect(losses[0]).toBeGreaterThan(0);
    const last = losses[losses.length - 1];
    const first = losses[0];
    if (last === undefined || first === undefined) {
      throw new Error('trainer spec: loss history empty');
    }
    expect(last).toBeLessThan(first);
    opt.dispose();
  });

  it('saveGptCheckpoint round-trips tensor count', async () => {
    const dir = await mkdtemp(path.join(os.tmpdir(), 'llm-diy-ckpt-'));
    try {
      const binPath = path.join(dir, 'w.bin');
      const jsonPath = path.join(dir, 'w.json');
      await saveGptCheckpoint({
        rt,
        model,
        binPath,
        jsonPath,
        config: tinyCfg,
        trainingStep: 7,
      });
      const { sidecar, tensors } = await loadGptCheckpointTensors(binPath, jsonPath);
      const names = gptParameterTensorNames(model);
      expect(Object.keys(sidecar.tensors).length).toBe(names.length);
      for (const n of names) {
        expect(tensors[n]).toBeDefined();
      }
      const metaRaw = await readFile(jsonPath, 'utf8');
      expect(metaRaw).toContain('"step"');
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });

  it('Trainer calls onStep after trainStep', async () => {
    const reg = createParameterRegistry();
    const params = collectGptParameters(model);
    registerModelParameters(reg, params);
    const opt = new AdamW(rt, reg, { lr: 1e-2, weightDecay: 0 });
    const recorded: { loss: number; lr: number }[] = [];
    const trainer = new Trainer(rt, model, reg, opt, {
      maxGradNorm: 1.0,
      schedule: {
        lrMax: 1e-2,
        lrMin: 1e-3,
        warmupSteps: 1,
        totalSteps: 5,
      },
      onStep: (info) => {
        recorded.push({ loss: info.loss, lr: info.lr });
      },
    });
    const { input, targets } = createLmBatch(rt, tokenStream, 0, 8);
    await trainer.trainStep(input, targets, 0);
    destroy(input);
    destroy(targets);
    expect(recorded.length).toBe(1);
    expect(Number.isFinite(recorded[0]?.loss)).toBe(true);
    opt.dispose();
  });
});
