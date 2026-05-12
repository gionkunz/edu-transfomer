import {
  GpuContext,
  KernelRegistry,
  probeWebGpu,
  registerCoreKernels,
} from '@llm-transformer-diy/gpu';
import {
  createGptKvCache,
  createGptModel,
  destroyGptKvCache,
  forwardGpt,
  forwardGptDecodeStep,
  forwardGptPrefill,
  type GptConfig,
  type GptModel,
} from '@llm-transformer-diy/model';
import { createIndices, destroy, toArray, type TensorRuntime } from '@llm-transformer-diy/tensor';
import { beforeAll, describe, expect, it } from 'vitest';

const webgpuAvailable = await probeWebGpu();

/* eslint-disable @typescript-eslint/no-non-null-assertion --
 * Float32Array compares use indices bounded by shared `.length`.
 */

function expectCloseVec(a: Float32Array, b: Float32Array, digits = 3): void {
  expect(a.length).toBe(b.length);
  for (let i = 0; i < a.length; i++) {
    expect(a[i]).toBeCloseTo(b[i]!, digits);
  }
}

describe.skipIf(!webgpuAvailable)('KV decode vs full forward', () => {
  let rt: TensorRuntime;
  let model: GptModel;
  const cfg: GptConfig = {
    vocabSize: 11,
    maxSeqLen: 16,
    dModel: 32,
    nHeads: 4,
    dFf: 64,
    nLayers: 1,
  };

  beforeAll(async () => {
    const ctx = await GpuContext.create();
    const registry = new KernelRegistry();
    registerCoreKernels(ctx.device, registry);
    rt = { ctx, registry };
    model = createGptModel(rt, cfg);
  });

  it('prefill + one decode matches a single full forward on three tokens', async () => {
    const ids = new Uint32Array([3, 7, 2]);
    const V = cfg.vocabSize;

    const tokFull = createIndices(rt, ids, [3]);
    const logitsFull = forwardGpt(rt, model, tokFull);
    const full = await toArray(rt, logitsFull.tensor);
    const wantLast = full.subarray(2 * V, 3 * V);
    destroy(tokFull);

    const cache = createGptKvCache(rt, model);
    const tokPre = createIndices(rt, new Uint32Array([3, 7]), [2]);
    forwardGptPrefill(rt, model, tokPre, cache);
    destroy(tokPre);

    const logitsDec = forwardGptDecodeStep(rt, model, 2, cache);
    const dec = await toArray(rt, logitsDec.tensor);

    expect(cache.seqLen).toBe(3);
    expectCloseVec(dec, wantLast, 3);

    destroyGptKvCache(cache);
  });

  it('single-token decode matches full forward length 1', async () => {
    const tokFull = createIndices(rt, new Uint32Array([5]), [1]);
    const logitsFull = forwardGpt(rt, model, tokFull);
    const want = await toArray(rt, logitsFull.tensor);
    destroy(tokFull);

    const cache = createGptKvCache(rt, model);
    const logitsDec = forwardGptDecodeStep(rt, model, 5, cache);
    const dec = await toArray(rt, logitsDec.tensor);

    expect(cache.seqLen).toBe(1);
    expectCloseVec(dec, want, 3);

    destroyGptKvCache(cache);
  });
});
