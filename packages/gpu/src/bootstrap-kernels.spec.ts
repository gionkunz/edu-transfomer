import './webgpu-globals';

import { describe, expect, it } from 'vitest';

import {
  BOOTSTRAP_KERNEL_FILL_PI,
  registerBootstrapKernels,
  smokeFillPiBuffer,
} from './bootstrap-kernels';
import { GpuContext, probeWebGpu } from './device';
import { KernelRegistry } from './kernel-registry';

const webgpuAvailable = await probeWebGpu();

describe.skipIf(!webgpuAvailable)('bootstrap-kernels', () => {
  it('runs fill-π kernel and matches CPU reference', async () => {
    const ctx = await GpuContext.create();
    const registry = new KernelRegistry();
    registerBootstrapKernels(ctx.device, registry);

    const n = 1024;
    const result = await smokeFillPiBuffer(ctx, registry, n);
    expect(result.length).toBe(n);
    const pi = 3.14159265;
    for (let i = 0; i < n; i++) {
      expect(result[i]).toBeCloseTo(pi, 5);
    }

    expect(registry.get(BOOTSTRAP_KERNEL_FILL_PI)).toBeDefined();
  });
});
