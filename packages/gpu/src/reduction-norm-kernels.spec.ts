/* eslint-disable @typescript-eslint/no-non-null-assertion --
 * Float32Array reads are typed `number | undefined` under
 * `noUncheckedIndexedAccess`; lengths are checked before iterating.
 */
import './webgpu-globals';

import { beforeAll, describe, expect, it } from 'vitest';

import {
  dispatchReduceMax,
  dispatchReduceSum,
  dispatchRmsNorm,
  dispatchSoftmax,
  dispatchSoftmaxBackward,
  registerCoreKernels,
} from './core-kernels';
import {
  smokeReduceMax,
  smokeReduceSum,
  smokeRmsNorm,
  smokeSoftmax,
  smokeSoftmaxBackward,
} from './core-kernels.test-helpers';
import {
  cpuReduceMax,
  cpuReduceSum,
  cpuRmsNorm,
  cpuSoftmax,
  cpuSoftmaxBackward,
} from './cpu-reference';
import { GpuContext, probeWebGpu } from './device';
import { KernelRegistry } from './kernel-registry';

const webgpuAvailable = await probeWebGpu();

function expectCloseVec(actual: Float32Array, expected: Float32Array, places = 5): void {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < actual.length; i++) {
    expect(actual[i]).toBeCloseTo(expected[i]!, places);
  }
}

describe.skipIf(!webgpuAvailable)('reduction-norm-kernels (GPU)', () => {
  let ctx: GpuContext;
  let registry: KernelRegistry;

  beforeAll(async () => {
    ctx = await GpuContext.create();
    registry = new KernelRegistry();
    registerCoreKernels(ctx.device, registry);
  });

  it('reduce_sum matches CPU for [3, 5] and [1, 16]', async () => {
    const a = Float32Array.from({ length: 3 * 5 }, (_, i) => i * 0.5 - 3);
    expectCloseVec(await smokeReduceSum(ctx, registry, a, 3, 5), cpuReduceSum(a, 3, 5));

    const b = Float32Array.from({ length: 16 }, (_, i) => Math.sin(i * 0.3));
    expectCloseVec(await smokeReduceSum(ctx, registry, b, 1, 16), cpuReduceSum(b, 1, 16));
  });

  it('reduce_max matches CPU for [3, 5] and [1, 16]', async () => {
    const a = Float32Array.from({ length: 3 * 5 }, (_, i) => Math.sin(i * 0.7) * 4);
    expectCloseVec(await smokeReduceMax(ctx, registry, a, 3, 5), cpuReduceMax(a, 3, 5));

    const b = Float32Array.from({ length: 16 }, (_, i) => (i % 2 === 0 ? -i : i));
    expectCloseVec(await smokeReduceMax(ctx, registry, b, 1, 16), cpuReduceMax(b, 1, 16));
  });

  it('softmax matches CPU for a typical [2, 4] row and rows sum to 1', async () => {
    const data = Float32Array.from({ length: 2 * 4 }, (_, i) => Math.sin(i * 0.4));
    const got = await smokeSoftmax(ctx, registry, data, 2, 4);
    expectCloseVec(got, cpuSoftmax(data, 2, 4));
    for (let i = 0; i < 2; i++) {
      let s = 0;
      for (let j = 0; j < 4; j++) s += got[i * 4 + j]!;
      expect(s).toBeCloseTo(1, 5);
    }
  });

  it('softmax stays finite on a large-magnitude row', async () => {
    const data = new Float32Array([1000, 1001, 999]);
    const got = await smokeSoftmax(ctx, registry, data, 1, 3);
    for (let i = 0; i < got.length; i++) {
      expect(Number.isFinite(got[i]!)).toBe(true);
    }
    expectCloseVec(got, cpuSoftmax(data, 1, 3));
  });

  it('softmax_backward matches CPU for [2, 4]', async () => {
    const rows = 2;
    const cols = 4;
    const logits = Float32Array.from({ length: rows * cols }, (_, i) => Math.sin(i * 0.5));
    const y = cpuSoftmax(logits, rows, cols);
    const gradOut = Float32Array.from({ length: rows * cols }, (_, i) => (i % 3) * 0.25 - 0.12);
    const got = await smokeSoftmaxBackward(ctx, registry, gradOut, y, rows, cols);
    expectCloseVec(got, cpuSoftmaxBackward(gradOut, y, rows, cols));
  });

  it('rmsnorm matches CPU for [2, 4] with non-trivial gamma', async () => {
    const data = Float32Array.from({ length: 2 * 4 }, (_, i) => (i + 1) * 0.25);
    const gamma = new Float32Array([0.5, 1, 1.5, 2]);
    const epsilon = 1e-6;
    expectCloseVec(
      await smokeRmsNorm(ctx, registry, data, gamma, 2, 4, epsilon),
      cpuRmsNorm(data, gamma, 2, 4, epsilon),
    );
  });
});

describe('reduction-norm-kernels argument validation (no GPU required)', () => {
  // Validation runs synchronously inside each dispatch helper before any
  // property access on ctx / registry, so we can safely pass nulls cast through
  // unknown — the helpers throw before either object is touched.
  const ctx = null as unknown as GpuContext;
  const registry = null as unknown as KernelRegistry;
  const buf = null as unknown as GPUBuffer;

  it('dispatchReduceSum rejects non-positive dims', () => {
    expect(() => {
      dispatchReduceSum(ctx, registry, buf, buf, 0, 4);
    }).toThrow(/rows/);
    expect(() => {
      dispatchReduceSum(ctx, registry, buf, buf, 2, 0);
    }).toThrow(/cols/);
    expect(() => {
      dispatchReduceSum(ctx, registry, buf, buf, -1, 4);
    }).toThrow(/rows/);
  });

  it('dispatchReduceMax rejects non-positive dims', () => {
    expect(() => {
      dispatchReduceMax(ctx, registry, buf, buf, 0, 4);
    }).toThrow(/rows/);
    expect(() => {
      dispatchReduceMax(ctx, registry, buf, buf, 2, -3);
    }).toThrow(/cols/);
  });

  it('dispatchSoftmax rejects non-positive dims', () => {
    expect(() => {
      dispatchSoftmax(ctx, registry, buf, buf, 0, 4);
    }).toThrow(/rows/);
    expect(() => {
      dispatchSoftmax(ctx, registry, buf, buf, 2, 0);
    }).toThrow(/cols/);
  });

  it('dispatchSoftmaxBackward rejects non-positive dims', () => {
    expect(() => {
      dispatchSoftmaxBackward(ctx, registry, buf, buf, buf, 0, 4);
    }).toThrow(/rows/);
    expect(() => {
      dispatchSoftmaxBackward(ctx, registry, buf, buf, buf, 2, 0);
    }).toThrow(/cols/);
  });

  it('dispatchRmsNorm rejects non-positive dims', () => {
    expect(() => {
      dispatchRmsNorm(ctx, registry, buf, buf, 4, buf, 0, 4);
    }).toThrow(/rows/);
    expect(() => {
      dispatchRmsNorm(ctx, registry, buf, buf, 4, buf, 2, 0);
    }).toThrow(/cols/);
  });

  it('dispatchRmsNorm rejects gamma length mismatch', () => {
    expect(() => {
      dispatchRmsNorm(ctx, registry, buf, buf, 5, buf, 2, 4);
    }).toThrow(/gamma/);
    expect(() => {
      dispatchRmsNorm(ctx, registry, buf, buf, 3, buf, 2, 4);
    }).toThrow(/gamma/);
  });
});
