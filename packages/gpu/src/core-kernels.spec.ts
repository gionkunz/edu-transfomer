/* eslint-disable @typescript-eslint/no-non-null-assertion --
 * Float32Array reads are typed `number | undefined` under
 * `noUncheckedIndexedAccess`; lengths are checked before iterating.
 */
import './webgpu-globals';

import { beforeAll, describe, expect, it } from 'vitest';

import { CORE_KERNEL_IDS, registerCoreKernels } from './core-kernels';
import {
  smokeAdd,
  smokeGelu,
  smokeMatMul,
  smokeMul,
  smokeScale,
  smokeTranspose,
} from './core-kernels.test-helpers';
import { cpuAdd, cpuGelu, cpuMatMul, cpuMul, cpuScale, cpuTranspose } from './cpu-reference';
import { GpuContext, probeWebGpu } from './device';
import { KernelRegistry } from './kernel-registry';

const webgpuAvailable = await probeWebGpu();

function expectCloseVec(actual: Float32Array, expected: Float32Array): void {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < actual.length; i++) {
    expect(actual[i]).toBeCloseTo(expected[i]!, 5);
  }
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
});
