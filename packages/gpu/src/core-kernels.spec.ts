import './webgpu-globals';

import { beforeAll, describe, expect, it } from 'vitest';

import { createFloat32StorageBuffer, readFloat32Buffer, writeFloat32Buffer } from './buffer';
import { CORE_KERNEL_IDS, dispatchAdamWStep, registerCoreKernels } from './core-kernels';
import {
  smokeAdd,
  smokeAdd16,
  smokeGelu,
  smokeGelu16,
  smokeGeluBackward,
  smokeGeluBackward16,
  smokeMatMul,
  smokeMatMul16,
  smokeMul,
  smokeMul16,
  smokeScale,
  smokeScale16,
  smokeTranspose,
  smokeTranspose16,
} from './core-kernels.test-helpers';
import {
  cpuAdamWStep,
  cpuAdd,
  cpuAddBf16,
  cpuGelu,
  cpuGeluBf16,
  cpuGeluBackward,
  cpuGeluBackwardBf16,
  cpuMatMul,
  cpuMatMulBf16,
  cpuMul,
  cpuMulBf16,
  cpuScale,
  cpuScaleBf16,
  cpuTranspose,
  cpuTransposeBf16,
} from './cpu-reference';
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

describe.skipIf(!webgpuAvailable)('core-kernels', () => {
  let ctx: GpuContext;
  let registry: KernelRegistry;

  beforeAll(async () => {
    ctx = await GpuContext.create();
    registry = new KernelRegistry();
    registerCoreKernels(ctx.device, registry);
  });

  it('registers every core kernel id', () => {
    for (const id of CORE_KERNEL_IDS) {
      expect(registry.get(id)).toBeDefined();
    }
  });

  it('matmul matches CPU for square and non-square shapes', async () => {
    const M = 3;
    const N = 4;
    const K = 2;
    const a = Float32Array.from({ length: M * K }, (_, i) => i + 1);
    const b = Float32Array.from({ length: K * N }, (_, i) => i * 0.5 - 1);

    expectCloseVec(await smokeMatMul(ctx, registry, a, b, M, N, K), cpuMatMul(a, b, M, N, K));

    const Ms = 4;
    const sa = Float32Array.from({ length: Ms * Ms }, () => 1);
    const sb = Float32Array.from({ length: Ms * Ms }, (_, i) => i);
    expectCloseVec(
      await smokeMatMul(ctx, registry, sa, sb, Ms, Ms, Ms),
      cpuMatMul(sa, sb, Ms, Ms, Ms),
    );
  });

  it('transpose matches CPU for a non-square matrix', async () => {
    const rows = 3;
    const cols = 5;
    const src = Float32Array.from({ length: rows * cols }, (_, i) => i * 0.25 - 2);
    expectCloseVec(
      await smokeTranspose(ctx, registry, src, rows, cols),
      cpuTranspose(src, rows, cols),
    );
  });

  it('add / mul / scale match CPU references', async () => {
    const n = 127;
    const a = Float32Array.from({ length: n }, (_, i) => Math.sin(i * 0.1));
    const b = Float32Array.from({ length: n }, (_, i) => Math.cos(i * 0.07));

    expectCloseVec(await smokeAdd(ctx, registry, a, b, n), cpuAdd(a, b, n));
    expectCloseVec(await smokeMul(ctx, registry, a, b, n), cpuMul(a, b, n));
    expectCloseVec(await smokeScale(ctx, registry, a, n, 2.5), cpuScale(a, 2.5, n));
  });

  it('gelu matches CPU reference', async () => {
    const n = 64;
    const a = Float32Array.from({ length: n }, (_, i) => (i - n / 2) * 0.15);
    expectCloseVec(await smokeGelu(ctx, registry, a, n), cpuGelu(a, n));
  });

  it('gelu backward matches CPU reference', async () => {
    const n = 64;
    const inp = Float32Array.from({ length: n }, (_, i) => (i - n / 2) * 0.15);
    const gradOut = Float32Array.from({ length: n }, (_, i) => Math.sin(i * 0.13) + 0.5);
    expectCloseVec(
      await smokeGeluBackward(ctx, registry, gradOut, inp, n),
      cpuGeluBackward(gradOut, inp, n),
    );
  });

  it('adamw step matches CPU reference', async () => {
    const n = 111;
    const theta0 = Float32Array.from({ length: n }, (_, i) => Math.sin(i * 0.03));
    const grad0 = Float32Array.from({ length: n }, (_, i) => (i % 7) * 0.01 - 0.02);
    const m0 = new Float32Array(n);
    const v0 = new Float32Array(n);

    const thetaCpu = new Float32Array(theta0);
    const gradCpu = new Float32Array(grad0);
    const mCpu = new Float32Array(m0);
    const vCpu = new Float32Array(v0);

    const t = 3;
    const beta1 = 0.9;
    const beta2 = 0.95;
    const hyper = {
      lr: 0.01,
      beta1,
      beta2,
      eps: 1e-8,
      weightDecay: 0.01,
      biasCorrection1: 1 - beta1 ** t,
      biasCorrection2: 1 - beta2 ** t,
    };

    cpuAdamWStep(
      thetaCpu,
      gradCpu,
      mCpu,
      vCpu,
      n,
      hyper.lr,
      hyper.beta1,
      hyper.beta2,
      hyper.eps,
      hyper.weightDecay,
      hyper.biasCorrection1,
      hyper.biasCorrection2,
    );

    const bTheta = createFloat32StorageBuffer(ctx.device, n);
    const bGrad = createFloat32StorageBuffer(ctx.device, n);
    const bM = createFloat32StorageBuffer(ctx.device, n);
    const bV = createFloat32StorageBuffer(ctx.device, n);
    writeFloat32Buffer(ctx.queue, bTheta, new Float32Array(theta0));
    writeFloat32Buffer(ctx.queue, bGrad, new Float32Array(grad0));
    writeFloat32Buffer(ctx.queue, bM, new Float32Array(m0));
    writeFloat32Buffer(ctx.queue, bV, new Float32Array(v0));

    dispatchAdamWStep(ctx, registry, bTheta, bGrad, bM, bV, n, hyper);

    const thetaGpu = await readFloat32Buffer(ctx.device, ctx.queue, bTheta, n * 4);
    const mGpu = await readFloat32Buffer(ctx.device, ctx.queue, bM, n * 4);
    const vGpu = await readFloat32Buffer(ctx.device, ctx.queue, bV, n * 4);

    expectCloseVec(thetaGpu, thetaCpu);
    expectCloseVec(mGpu, mCpu);
    expectCloseVec(vGpu, vCpu);

    bTheta.destroy();
    bGrad.destroy();
    bM.destroy();
    bV.destroy();
  });

  const dtypes16 = ['bf16'] as const;
  for (const d16 of dtypes16) {
    it(`matmul matches CPU reference (${d16})`, async () => {
      if (!supportsDtype(ctx, d16)) {
        return;
      }
      const places = dtypeDecimalPlaces(d16);
      const M = 3;
      const N = 4;
      const K = 2;
      const a = Float32Array.from({ length: M * K }, (_, i) => i + 1);
      const b = Float32Array.from({ length: K * N }, (_, i) => i * 0.5 - 1);
      const cpuRef = cpuMatMulBf16(a, b, M, N, K);
      expectCloseVec(await smokeMatMul16(ctx, registry, a, b, M, N, K, d16), cpuRef, places);
    });

    it(`transpose matches CPU (${d16})`, async () => {
      if (!supportsDtype(ctx, d16)) {
        return;
      }
      const places = dtypeDecimalPlaces(d16);
      const rows = 3;
      const cols = 5;
      const src = Float32Array.from({ length: rows * cols }, (_, i) => i * 0.25 - 2);
      const cpuRef = cpuTransposeBf16(src, rows, cols);
      expectCloseVec(await smokeTranspose16(ctx, registry, src, rows, cols, d16), cpuRef, places);
    });

    it(`add/mul/scale match CPU (${d16})`, async () => {
      if (!supportsDtype(ctx, d16)) {
        return;
      }
      const places = dtypeDecimalPlaces(d16);
      const n = 127;
      const va = Float32Array.from({ length: n }, (_, i) => Math.sin(i * 0.1));
      const vb = Float32Array.from({ length: n }, (_, i) => Math.cos(i * 0.07));
      expectCloseVec(
        await smokeAdd16(ctx, registry, va, vb, n, d16),
        cpuAddBf16(va, vb, n),
        places,
      );
      expectCloseVec(
        await smokeMul16(ctx, registry, va, vb, n, d16),
        cpuMulBf16(va, vb, n),
        places,
      );
      expectCloseVec(
        await smokeScale16(ctx, registry, va, n, 2.5, d16),
        cpuScaleBf16(va, 2.5, n),
        places,
      );
    });

    it(`gelu matches CPU (${d16})`, async () => {
      if (!supportsDtype(ctx, d16)) {
        return;
      }
      const places = dtypeDecimalPlaces(d16);
      const n = 64;
      const a = Float32Array.from({ length: n }, (_, i) => (i - n / 2) * 0.15);
      const cpuRef = cpuGeluBf16(a, n);
      expectCloseVec(await smokeGelu16(ctx, registry, a, n, d16), cpuRef, places);
    });

    it(`gelu backward matches CPU (${d16})`, async () => {
      if (!supportsDtype(ctx, d16)) {
        return;
      }
      const places = dtypeDecimalPlaces(d16);
      const n = 64;
      const inp = Float32Array.from({ length: n }, (_, i) => (i - n / 2) * 0.15);
      const gradOut = Float32Array.from({ length: n }, (_, i) => Math.sin(i * 0.13) + 0.5);
      const cpuRef = cpuGeluBackwardBf16(gradOut, inp, n);
      expectCloseVec(
        await smokeGeluBackward16(ctx, registry, gradOut, inp, n, d16),
        cpuRef,
        places,
      );
    });
  }
});
