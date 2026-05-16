import './webgpu-globals';

import { beforeAll, describe, expect, it } from 'vitest';

import {
  dispatchReduceMax,
  dispatchReduceSum,
  dispatchRmsNorm,
  dispatchRmsNormGradGamma,
  dispatchRmsNormGradInput,
  dispatchSoftmax,
  dispatchSoftmaxBackward,
  registerCoreKernels,
} from './core-kernels';
import {
  smokeReduceMax,
  smokeReduceMax16,
  smokeReduceSum,
  smokeReduceSum16,
  smokeRmsNorm,
  smokeRmsNorm16,
  smokeRmsNormGradGamma16,
  smokeRmsNormGradInput16,
  smokeSoftmax,
  smokeSoftmax16,
  smokeSoftmaxBackward,
  smokeSoftmaxBackward16,
} from './core-kernels.test-helpers';
import {
  cpuReduceMax,
  cpuReduceMaxBf16,
  cpuReduceSum,
  cpuReduceSumBf16,
  cpuRmsNorm,
  cpuRmsNormBf16,
  cpuRmsNormGradGamma,
  cpuRmsNormGradGammaBf16,
  cpuRmsNormGradInput,
  cpuRmsNormGradInputBf16,
  cpuSoftmax,
  cpuSoftmaxBf16,
  cpuSoftmaxBackward,
  cpuSoftmaxBackwardBf16,
} from './cpu-reference';
import { createFloat32StorageBuffer, readFloat32Buffer, writeFloat32Buffer } from './buffer';
import { GpuContext, probeWebGpu, supportsDtype } from './device';
import { KernelRegistry } from './kernel-registry';

const webgpuAvailable = await probeWebGpu();

function expectCloseVec(actual: Float32Array, expected: Float32Array, places = 5): void {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < actual.length; i++) {
    expect(actual[i]).toBeCloseTo(expected[i], places);
  }
}

function dtypeDecimalPlaces(_d: 'bf16'): number {
  return 3;
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
      for (let j = 0; j < 4; j++) s += got[i * 4 + j];
      expect(s).toBeCloseTo(1, 5);
    }
  });

  it('softmax stays finite on a large-magnitude row', async () => {
    const data = new Float32Array([1000, 1001, 999]);
    const got = await smokeSoftmax(ctx, registry, data, 1, 3);
    for (let i = 0; i < got.length; i++) {
      expect(Number.isFinite(got[i])).toBe(true);
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

  it('rmsnorm backward matches CPU for [2, 4]', async () => {
    const rows = 2;
    const cols = 4;
    const epsilon = 1e-6;
    const data = Float32Array.from({ length: rows * cols }, (_, i) => (i + 1) * 0.2 - 1);
    const gamma = Float32Array.from({ length: cols }, (_, j) => 0.8 + j * 0.1);
    const gradOut = Float32Array.from({ length: rows * cols }, (_, i) => Math.sin(i * 0.31));

    const gxi = cpuRmsNormGradInput(data, gamma, gradOut, rows, cols, epsilon);

    const bIn = createFloat32StorageBuffer(ctx.device, rows * cols);
    const bGam = createFloat32StorageBuffer(ctx.device, cols);
    const bGradOut = createFloat32StorageBuffer(ctx.device, rows * cols);
    const bGxi = createFloat32StorageBuffer(ctx.device, rows * cols);
    const bGgm = createFloat32StorageBuffer(ctx.device, cols);
    writeFloat32Buffer(ctx.queue, bIn, data);
    writeFloat32Buffer(ctx.queue, bGam, gamma);
    writeFloat32Buffer(ctx.queue, bGradOut, gradOut);

    dispatchRmsNormGradInput(ctx, registry, bIn, bGam, cols, bGradOut, bGxi, rows, cols, epsilon);
    const gpuGxi = await readFloat32Buffer(ctx.device, ctx.queue, bGxi, rows * cols * 4);
    expectCloseVec(gpuGxi, gxi);

    const ggc = cpuRmsNormGradGamma(data, gradOut, rows, cols, epsilon);
    dispatchRmsNormGradGamma(ctx, registry, bIn, bGradOut, bGgm, rows, cols, epsilon);
    const gpuGgc = await readFloat32Buffer(ctx.device, ctx.queue, bGgm, cols * 4);
    expectCloseVec(gpuGgc, ggc);

    bIn.destroy();
    bGam.destroy();
    bGradOut.destroy();
    bGxi.destroy();
    bGgm.destroy();
  });

  const dtypes16 = ['bf16'] as const;
  for (const d16 of dtypes16) {
    it(`reduce_sum matches CPU (${d16})`, async () => {
      if (!supportsDtype(ctx, d16)) {
        return;
      }
      const places = dtypeDecimalPlaces(d16);
      const a = Float32Array.from({ length: 3 * 5 }, (_, i) => i * 0.5 - 3);
      const cpuA = cpuReduceSumBf16(a, 3, 5);
      expectCloseVec(await smokeReduceSum16(ctx, registry, a, 3, 5, d16), cpuA, places);

      const b = Float32Array.from({ length: 16 }, (_, i) => Math.sin(i * 0.3));
      const cpuB = cpuReduceSumBf16(b, 1, 16);
      expectCloseVec(await smokeReduceSum16(ctx, registry, b, 1, 16, d16), cpuB, places);
    });

    it(`reduce_max matches CPU (${d16})`, async () => {
      if (!supportsDtype(ctx, d16)) {
        return;
      }
      const places = dtypeDecimalPlaces(d16);
      const a = Float32Array.from({ length: 3 * 5 }, (_, i) => Math.sin(i * 0.7) * 4);
      const cpuA = cpuReduceMaxBf16(a, 3, 5);
      expectCloseVec(await smokeReduceMax16(ctx, registry, a, 3, 5, d16), cpuA, places);

      const b = Float32Array.from({ length: 16 }, (_, i) => (i % 2 === 0 ? -i : i));
      const cpuB = cpuReduceMaxBf16(b, 1, 16);
      expectCloseVec(await smokeReduceMax16(ctx, registry, b, 1, 16, d16), cpuB, places);
    });

    it(`softmax matches CPU (${d16}) and rows sum to 1`, async () => {
      if (!supportsDtype(ctx, d16)) {
        return;
      }
      const places = dtypeDecimalPlaces(d16);
      const sumPlaces = Math.max(0, places - 2);
      const data = Float32Array.from({ length: 2 * 4 }, (_, i) => Math.sin(i * 0.4));
      const cpuRef = cpuSoftmaxBf16(data, 2, 4);
      const got = await smokeSoftmax16(ctx, registry, data, 2, 4, d16);
      const eltPlaces = Math.min(places, 2);
      expectCloseVec(got, cpuRef, eltPlaces);
      for (let i = 0; i < 2; i++) {
        let s = 0;
        for (let j = 0; j < 4; j++) {
          s += got[i * 4 + j];
        }
        expect(s).toBeCloseTo(1, sumPlaces);
      }
    });

    it(`softmax stays finite (${d16}) on a large-magnitude row`, async () => {
      if (!supportsDtype(ctx, d16)) {
        return;
      }
      const places = dtypeDecimalPlaces(d16);
      const data = new Float32Array([1000, 1001, 999]);
      const got = await smokeSoftmax16(ctx, registry, data, 1, 3, d16);
      for (let i = 0; i < got.length; i++) {
        expect(Number.isFinite(got[i])).toBe(true);
      }
      const cpuRef = cpuSoftmaxBf16(data, 1, 3);
      expectCloseVec(got, cpuRef, places);
    });

    it(`rmsnorm matches CPU (${d16}) for [2, 4] with non-trivial gamma`, async () => {
      if (!supportsDtype(ctx, d16)) {
        return;
      }
      const places = dtypeDecimalPlaces(d16);
      const data = Float32Array.from({ length: 2 * 4 }, (_, i) => (i + 1) * 0.25);
      const gamma = new Float32Array([0.5, 1, 1.5, 2]);
      const epsilon = 1e-6;
      const cpuRef = cpuRmsNormBf16(data, gamma, 2, 4, epsilon);
      expectCloseVec(
        await smokeRmsNorm16(ctx, registry, data, gamma, 2, 4, d16, epsilon),
        cpuRef,
        places,
      );
    });

    it(`softmax_backward matches CPU (${d16}) for [2, 4]`, async () => {
      if (!supportsDtype(ctx, d16)) {
        return;
      }
      const places = dtypeDecimalPlaces(d16);
      const rows = 2;
      const cols = 4;
      const logits = Float32Array.from({ length: rows * cols }, (_, i) => Math.sin(i * 0.5));
      const yCpu = cpuSoftmaxBf16(logits, rows, cols);
      const gradOut = Float32Array.from({ length: rows * cols }, (_, i) => (i % 3) * 0.25 - 0.12);
      const cpuRef = cpuSoftmaxBackwardBf16(gradOut, yCpu, rows, cols);
      expectCloseVec(
        await smokeSoftmaxBackward16(ctx, registry, gradOut, yCpu, rows, cols, d16),
        cpuRef,
        places,
      );
    });

    it(`rmsnorm backward matches CPU (${d16}) for [2, 4]`, async () => {
      if (!supportsDtype(ctx, d16)) {
        return;
      }
      const places = dtypeDecimalPlaces(d16);
      const rows = 2;
      const cols = 4;
      const epsilon = 1e-6;
      const data = Float32Array.from({ length: rows * cols }, (_, i) => (i + 1) * 0.2 - 1);
      const gamma = Float32Array.from({ length: cols }, (_, j) => 0.8 + j * 0.1);
      const gradOut = Float32Array.from({ length: rows * cols }, (_, i) => Math.sin(i * 0.31));
      const gxiRef = cpuRmsNormGradInputBf16(data, gamma, gradOut, rows, cols, epsilon);
      expectCloseVec(
        await smokeRmsNormGradInput16(
          ctx,
          registry,
          data,
          gamma,
          gradOut,
          rows,
          cols,
          d16,
          epsilon,
        ),
        gxiRef,
        places,
      );
      const ggmRef = cpuRmsNormGradGammaBf16(data, gradOut, rows, cols, epsilon);
      expectCloseVec(
        await smokeRmsNormGradGamma16(ctx, registry, data, gradOut, rows, cols, d16, epsilon),
        ggmRef,
        places,
      );
    });
  }
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
