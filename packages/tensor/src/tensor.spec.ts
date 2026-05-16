import {
  cpuAdd,
  cpuGelu,
  cpuGeluBackward,
  cpuMatMul,
  cpuMul,
  cpuScale,
  cpuSoftmax,
  cpuSoftmaxBackward,
  cpuTranspose,
  GpuContext,
  KernelRegistry,
  packBf16FromFloat32,
  probeWebGpu,
  registerCoreKernels,
  unpackBf16ToFloat32,
} from '@llm-transformer-diy/gpu';
import { afterEach, beforeAll, describe, expect, it } from 'vitest';

import {
  add,
  copyF32Elems,
  gelu,
  geluBackward,
  matmul,
  mul,
  reshape,
  scale,
  softmax,
  softmaxBackward,
  transpose,
  view2dRowPrefix,
} from './ops';
import { cast, createFromArray, createZeros, destroy, toArray } from './tensor';
import type { Tensor, TensorRuntime } from './types';

const webgpuAvailable = await probeWebGpu();

function expectCloseVec(actual: Float32Array, expected: Float32Array, places = 5): void {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < actual.length; i++) {
    expect(actual[i]).toBeCloseTo(expected[i], places);
  }
}

describe.skipIf(!webgpuAvailable)('tensor (GPU-backed)', () => {
  let rt: TensorRuntime;
  /** Tensors registered here are destroyed in afterEach; `null` slots indicate a tensor whose buffer is shared. */
  const allocatedThisTest: Tensor[] = [];

  function track<T extends Tensor>(t: T): T {
    allocatedThisTest.push(t);
    return t;
  }

  beforeAll(async () => {
    const ctx = await GpuContext.create();
    const registry = new KernelRegistry();
    registerCoreKernels(ctx.device, registry);
    rt = { ctx, registry };
  });

  afterEach(() => {
    while (allocatedThisTest.length > 0) {
      const t = allocatedThisTest.pop();
      if (t) {
        try {
          destroy(t);
        } catch {
          // Buffer may already be destroyed via reshape sharing — ignore.
        }
      }
    }
  });

  it('createFromArray + toArray round-trips data', async () => {
    const data = new Float32Array([1, 2, 3, 4, 5, 6]);
    const t = track(createFromArray(rt, data, [2, 3]));
    expect(t.dtype).toBe('f32');
    expect(t.shape).toEqual([2, 3]);
    expect(t.strides).toEqual([3, 1]);
    expect(t.length).toBe(6);
    expectCloseVec(await toArray(rt, t), data);
  });

  it('createZeros allocates a zero-filled buffer of the requested shape', async () => {
    const t = track(createZeros(rt, [2, 4]));
    expect(t.length).toBe(8);
    expectCloseVec(await toArray(rt, t), new Float32Array(8));
  });

  it('matmul matches the CPU reference for non-square shapes', async () => {
    const M = 3;
    const K = 2;
    const N = 4;
    const aData = Float32Array.from({ length: M * K }, (_, i) => i + 1);
    const bData = Float32Array.from({ length: K * N }, (_, i) => i * 0.5 - 1);

    const a = track(createFromArray(rt, aData, [M, K]));
    const b = track(createFromArray(rt, bData, [K, N]));
    const c = track(matmul(rt, a, b));

    expect(c.shape).toEqual([M, N]);
    expectCloseVec(await toArray(rt, c), cpuMatMul(aData, bData, M, N, K));
  });

  it('add / mul / scale match CPU references on a non-trivial vector', async () => {
    const n = 64;
    const aData = Float32Array.from({ length: n }, (_, i) => Math.sin(i * 0.1));
    const bData = Float32Array.from({ length: n }, (_, i) => Math.cos(i * 0.07));

    const a = track(createFromArray(rt, aData, [n]));
    const b = track(createFromArray(rt, bData, [n]));

    const sumT = track(add(rt, a, b));
    const prodT = track(mul(rt, a, b));
    const scaledT = track(scale(rt, a, 2.5));

    expectCloseVec(await toArray(rt, sumT), cpuAdd(aData, bData, n));
    expectCloseVec(await toArray(rt, prodT), cpuMul(aData, bData, n));
    expectCloseVec(await toArray(rt, scaledT), cpuScale(aData, 2.5, n));
  });

  it('gelu matches the CPU reference', async () => {
    const n = 32;
    const data = Float32Array.from({ length: n }, (_, i) => (i - n / 2) * 0.15);
    const t = track(createFromArray(rt, data, [n]));
    const out = track(gelu(rt, t));
    expectCloseVec(await toArray(rt, out), cpuGelu(data, n));
  });

  it('geluBackward matches the CPU reference', async () => {
    const n = 32;
    const inp = Float32Array.from({ length: n }, (_, i) => (i - n / 2) * 0.15);
    const gradOutData = Float32Array.from({ length: n }, (_, i) => Math.cos(i * 0.11));
    const gradT = track(createFromArray(rt, gradOutData, [n]));
    const inpT = track(createFromArray(rt, inp, [n]));
    const out = track(geluBackward(rt, gradT, inpT));
    expectCloseVec(await toArray(rt, out), cpuGeluBackward(gradOutData, inp, n));
  });

  it('softmax matches the CPU reference for [2, 4]', async () => {
    const rows = 2;
    const cols = 4;
    const data = Float32Array.from({ length: rows * cols }, (_, i) => Math.sin(i * 0.4));
    const t = track(createFromArray(rt, data, [rows, cols]));
    const out = track(softmax(rt, t));
    expectCloseVec(await toArray(rt, out), cpuSoftmax(data, rows, cols));
  });

  it('softmaxBackward matches the CPU reference for [2, 4]', async () => {
    const rows = 2;
    const cols = 4;
    const logits = Float32Array.from({ length: rows * cols }, (_, i) => Math.cos(i * 0.3));
    const y = cpuSoftmax(logits, rows, cols);
    const gradOutData = Float32Array.from({ length: rows * cols }, (_, i) => (i % 4) * 0.1 - 0.15);
    const gradT = track(createFromArray(rt, gradOutData, [rows, cols]));
    const yT = track(createFromArray(rt, y, [rows, cols]));
    const out = track(softmaxBackward(rt, gradT, yT));
    expectCloseVec(await toArray(rt, out), cpuSoftmaxBackward(gradOutData, y, rows, cols));
  });

  it('transpose matches the CPU reference for a non-square matrix', async () => {
    const rows = 3;
    const cols = 5;
    const data = Float32Array.from({ length: rows * cols }, (_, i) => i * 0.25 - 2);
    const t = track(createFromArray(rt, data, [rows, cols]));
    const tT = track(transpose(rt, t));
    expect(tT.shape).toEqual([cols, rows]);
    expectCloseVec(await toArray(rt, tT), cpuTranspose(data, rows, cols));
  });

  it('reshape preserves data on a GPU round-trip and shares the underlying buffer', async () => {
    const data = Float32Array.from({ length: 6 }, (_, i) => i + 1);
    const a = track(createFromArray(rt, data, [2, 3]));
    const b = reshape(a, [3, 2]);
    // Don't track `b` — it shares `a.buffer`; destroying `a` releases both.
    expect(b.buffer).toBe(a.buffer);
    expect(b.shape).toEqual([3, 2]);
    expectCloseVec(await toArray(rt, b), data);
  });

  it('matmul throws on inner-dim mismatch before any dispatch', () => {
    const a = track(createFromArray(rt, new Float32Array(6), [2, 3]));
    const b = track(createFromArray(rt, new Float32Array(8), [4, 2]));
    expect(() => matmul(rt, a, b)).toThrow(/inner dimensions must match/);
  });

  it('add throws on shape mismatch before any dispatch', () => {
    const a = track(createFromArray(rt, new Float32Array(6), [2, 3]));
    const b = track(createFromArray(rt, new Float32Array(4), [2, 2]));
    expect(() => add(rt, a, b)).toThrow(/shapes differ/);
  });

  it('transpose throws on rank ≠ 2 before any dispatch', () => {
    const a = track(createFromArray(rt, new Float32Array(8), [2, 2, 2]));
    expect(() => transpose(rt, a)).toThrow(/expected rank 2/);
  });

  it('copyF32Elems copies a prefix between tensors', async () => {
    const a = track(createFromArray(rt, new Float32Array([1, 2, 3, 4, 5, 6]), [2, 3]));
    const b = track(createZeros(rt, [2, 3]));
    copyF32Elems(rt, a, 0, b, 0, 4);
    expectCloseVec(await toArray(rt, b), new Float32Array([1, 2, 3, 4, 0, 0]));
  });

  it('view2dRowPrefix + matmul uses only the prefix rows', async () => {
    const full = track(createFromArray(rt, new Float32Array([1, 2, 3, 4, 9, 9]), [3, 2]));
    const pref = view2dRowPrefix(full, 2);
    const rhs = track(createFromArray(rt, new Float32Array([1, 0, 0, 1]), [2, 2]));
    const out = track(matmul(rt, pref, rhs));
    expectCloseVec(await toArray(rt, out), new Float32Array([1, 2, 3, 4]));
  });

  it('softmax throws on rank ≠ 2 before any dispatch', () => {
    const a = track(createFromArray(rt, new Float32Array(4), [4]));
    expect(() => softmax(rt, a)).toThrow(/expected rank 2/);
  });

  it('createFromArray bf16 + toArray matches CPU pack/unpack rounding', async () => {
    const data = Float32Array.from([1, 2, -0.5, 3.5]);
    const t = track(createFromArray(rt, data, [2, 2], 'bf16'));
    expect(t.dtype).toBe('bf16');
    const expected = unpackBf16ToFloat32(packBf16FromFloat32(data));
    expectCloseVec(await toArray(rt, t), expected, 3);
  });

  it('matmul rejects mixed float dtypes', () => {
    const a = track(createFromArray(rt, new Float32Array(4), [2, 2], 'f32'));
    const b = track(createFromArray(rt, new Float32Array(4), [2, 2], 'bf16'));
    expect(() => matmul(rt, a, b)).toThrow(/dtype mismatch/);
  });

  it('cast to same dtype uses a new buffer (no aliasing)', async () => {
    const data = new Float32Array([1, 2, 3]);
    const a = track(createFromArray(rt, data, [3], 'bf16'));
    const b = track(cast(rt, a, 'bf16'));
    expect(b.buffer).not.toBe(a.buffer);
    expect(b.dtype).toBe('bf16');
    expectCloseVec(await toArray(rt, b), await toArray(rt, a), 3);
  });

  it('cast f32 → bf16 → f32 matches bf16 rounding reference', async () => {
    const data = new Float32Array([1.25, -2, 0.125]);
    const f = track(createFromArray(rt, data, [3], 'f32'));
    const q = track(cast(rt, f, 'bf16'));
    const r = track(cast(rt, q, 'f32'));
    const expected = unpackBf16ToFloat32(packBf16FromFloat32(data));
    expectCloseVec(await toArray(rt, r), expected, 3);
  });
});
