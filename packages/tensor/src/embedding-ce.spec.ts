import {
  cpuCrossEntropyFromLogits,
  cpuCrossEntropyLogitsGradFromProbs,
  cpuCrossEntropyLogitsGradFromProbsBf16,
  cpuCrossEntropyLogitsGradFromProbsFp16,
  cpuEmbeddingGather,
  cpuEmbeddingScatterAccum,
  cpuEmbeddingScatterAccumBf16,
  cpuEmbeddingScatterAccumFp16,
  cpuSoftmax,
  cpuSoftmaxBf16,
  cpuSoftmaxFp16,
  GpuContext,
  KernelRegistry,
  probeWebGpu,
  registerCoreKernels,
  supportsDtype,
} from '@llm-transformer-diy/gpu';
import { afterEach, beforeAll, describe, expect, it } from 'vitest';

import {
  broadcastFillFromScalar,
  createFromArray,
  createIndices,
  crossEntropyLogitsGrad,
  crossEntropyRowsFromLogits,
  destroy,
  embedding,
  scatterEmbeddingAccum,
  sumFlat,
  toArray,
} from '..';
import type { Tensor, TensorRuntime } from '..';

const webgpuAvailable = await probeWebGpu();

function dtypeDecimalPlaces(d: 'bf16' | 'fp16'): number {
  return d === 'bf16' ? 3 : 2;
}

function expectCloseVec(actual: Float32Array, expected: Float32Array, places = 5): void {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < actual.length; i++) {
    expect(actual[i]).toBeCloseTo(expected[i], places);
  }
}

describe.skipIf(!webgpuAvailable)('tensor embedding + CE (GPU)', () => {
  let rt: TensorRuntime;
  const tensors: Tensor[] = [];

  function track<T extends Tensor>(t: T): T {
    tensors.push(t);
    return t;
  }

  beforeAll(async () => {
    const ctx = await GpuContext.create();
    const registry = new KernelRegistry();
    registerCoreKernels(ctx.device, registry);
    rt = { ctx, registry };
  });

  afterEach(() => {
    while (tensors.length > 0) {
      const t = tensors.pop();
      if (t) {
        try {
          destroy(t);
        } catch {
          // ignore duplicate destroy
        }
      }
    }
  });

  it('embedding matches CPU gather', async () => {
    const vocab = 7;
    const embed = 4;
    const tokens = 5;
    const w = Float32Array.from({ length: vocab * embed }, (_, i) => (i % 17) * 0.07 - 0.3);
    const weights = track(createFromArray(rt, w, [vocab, embed]));
    const idx = Uint32Array.from({ length: tokens }, (_, i) => (i * 3) % vocab);
    const indices = track(createIndices(rt, idx, [tokens]));
    const gpuOut = track(embedding(rt, weights, indices));
    const expected = cpuEmbeddingGather(w, idx, vocab, embed);
    expectCloseVec(await toArray(rt, gpuOut), expected);
  });

  it('scatterEmbeddingAccum matches CPU (duplicate ids)', async () => {
    const vocab = 5;
    const embed = 3;
    const tokens = 4;
    const idx = Uint32Array.from([1, 2, 1, 1]);
    const gradTok = Float32Array.from({ length: tokens * embed }, (_, i) => (i + 1) * 0.03);
    const indices = track(createIndices(rt, idx, [tokens]));
    const gTokens = track(createFromArray(rt, gradTok, [tokens, embed]));
    const scattered = track(scatterEmbeddingAccum(rt, indices, gTokens, vocab));
    expect(scattered.shape).toEqual([vocab, embed]);
    const cpu = cpuEmbeddingScatterAccum(gradTok, idx, vocab, embed);
    expectCloseVec(await toArray(rt, scattered), cpu);
  });

  it('cross-entropy rows + logits grad match CPU', async () => {
    const rows = 3;
    const cols = 5;
    const logitsData = Float32Array.from({ length: rows * cols }, (_, i) => (i - 8) * 0.09);
    const logits = track(createFromArray(rt, logitsData, [rows, cols]));
    const targ = Uint32Array.from([0, 4, 2]);
    const targets = track(createIndices(rt, targ, [rows]));

    const ce = track(crossEntropyRowsFromLogits(rt, logits, targets));
    expect(ce.shape).toEqual([rows]);
    const cpuCe = cpuCrossEntropyFromLogits(logitsData, targ, rows, cols);
    expectCloseVec(await toArray(rt, ce), cpuCe);

    const probsCpu = cpuSoftmax(logitsData, rows, cols);
    const probs = track(createFromArray(rt, probsCpu, [rows, cols]));
    const upstreamData = new Float32Array([0.5, 1.0, -0.25]);
    const upstreamT = track(createFromArray(rt, upstreamData, [rows]));
    const logitsGrad = track(crossEntropyLogitsGrad(rt, probs, targets, upstreamT));
    const cpuGrad = cpuCrossEntropyLogitsGradFromProbs(probsCpu, targ, upstreamData, rows, cols);
    expect(logitsGrad.shape).toEqual([rows, cols]);
    expectCloseVec(await toArray(rt, logitsGrad), cpuGrad, 5);
  });

  const dtypes16 = ['bf16', 'fp16'] as const;
  for (const d16 of dtypes16) {
    it(`scatterEmbeddingAccum matches CPU (${d16}, duplicate ids)`, async () => {
      if (!supportsDtype(rt.ctx, d16)) {
        return;
      }
      const places = dtypeDecimalPlaces(d16);
      const vocab = 5;
      const embed = 3;
      const tokens = 4;
      const idx = Uint32Array.from([1, 2, 1, 1]);
      const gradTok = Float32Array.from({ length: tokens * embed }, (_, i) => (i + 1) * 0.03);
      const indices = track(createIndices(rt, idx, [tokens]));
      const gTokens = track(createFromArray(rt, gradTok, [tokens, embed], d16));
      const scattered = track(scatterEmbeddingAccum(rt, indices, gTokens, vocab));
      expect(scattered.shape).toEqual([vocab, embed]);
      const cpu =
        d16 === 'bf16'
          ? cpuEmbeddingScatterAccumBf16(gradTok, idx, vocab, embed)
          : cpuEmbeddingScatterAccumFp16(gradTok, idx, vocab, embed);
      const scatterPlaces = d16 === 'bf16' ? 2 : places;
      expectCloseVec(await toArray(rt, scattered), cpu, scatterPlaces);
    });

    it(`cross-entropy logits grad matches CPU (${d16})`, async () => {
      if (!supportsDtype(rt.ctx, d16)) {
        return;
      }
      const places = dtypeDecimalPlaces(d16);
      const rows = 3;
      const cols = 5;
      const logitsData = Float32Array.from({ length: rows * cols }, (_, i) => (i - 8) * 0.09);
      const targ = Uint32Array.from([0, 4, 2]);
      const targets = track(createIndices(rt, targ, [rows]));
      const probsCpu =
        d16 === 'bf16'
          ? cpuSoftmaxBf16(logitsData, rows, cols)
          : cpuSoftmaxFp16(logitsData, rows, cols);
      const probs = track(createFromArray(rt, probsCpu, [rows, cols], d16));
      const upstreamData = new Float32Array([0.5, 1.0, -0.25]);
      const upstreamT = track(createFromArray(rt, upstreamData, [rows], d16));
      const logitsGrad = track(crossEntropyLogitsGrad(rt, probs, targets, upstreamT));
      const cpuGrad =
        d16 === 'bf16'
          ? cpuCrossEntropyLogitsGradFromProbsBf16(probsCpu, targ, upstreamData, rows, cols)
          : cpuCrossEntropyLogitsGradFromProbsFp16(probsCpu, targ, upstreamData, rows, cols);
      expect(logitsGrad.shape).toEqual([rows, cols]);
      expectCloseVec(await toArray(rt, logitsGrad), cpuGrad, places);
    });
  }

  it('sumFlat + broadcastFillFromScalar', async () => {
    const a = track(createFromArray(rt, new Float32Array([1, -2, 3.5]), [3]));
    const s = track(sumFlat(rt, a));
    expect(s.shape).toEqual([1]);
    expect(await toArray(rt, s)).toEqual(new Float32Array([2.5]));

    const g = Float32Array.from([4]);
    const up = track(createFromArray(rt, g, [1]));
    const b = track(broadcastFillFromScalar(rt, up, 3));
    expectCloseVec(await toArray(rt, b), new Float32Array([4, 4, 4]));
  });

  /**
   * Regression: previously the `bf16` / `fp16-packed` `broadcast-fill-scalar` shaders bound the
   * scalar input as `array<f32>`, but {@link broadcastFillFromScalar} hands them a 16-bit packed
   * buffer (low 16 bits hold the value, high 16 are zero). Reading those bytes as `f32` produced
   * a subnormal ≈ 0, which silently zeroed the upstream gradient that flows out of `sumFlat`'s
   * backward — meaning the cross-entropy mean gradient died and the model never learned in
   * `bf16` / `fp16` training. These checks pin the value broadcast from a 16-bit `[1]` tensor.
   */
  for (const d16 of ['bf16', 'fp16'] as const) {
    it(`broadcastFillFromScalar preserves ${d16} scalar (regression for sumFlat-backward)`, async () => {
      if (!supportsDtype(rt.ctx, d16)) {
        return;
      }
      const places = dtypeDecimalPlaces(d16);
      const value = 1 / 32;
      const up = track(createFromArray(rt, Float32Array.from([value]), [1], d16));
      const b = track(broadcastFillFromScalar(rt, up, 5));
      expect(b.dtype).toBe(d16);
      const host = await toArray(rt, b);
      expect(host.length).toBe(5);
      expectCloseVec(host, new Float32Array(5).fill(value), places);
      for (let i = 0; i < host.length; i++) {
        expect(host[i]).toBeGreaterThan(value * 0.9);
      }
    });
  }
});
