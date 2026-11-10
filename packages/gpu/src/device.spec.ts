import { describe, expect, it } from 'vitest';

import { GpuContext, gpuAdapterLikelyCpuOnly, probeWebGpu } from './device';

describe('gpuAdapterLikelyCpuOnly', () => {
  it('detects llvmpipe / swiftshader / lavapipe / software labels', () => {
    expect(gpuAdapterLikelyCpuOnly('llvmpipe: Mesa (LLVM 20)')).toBe(true);
    expect(gpuAdapterLikelyCpuOnly('SwiftShader blah')).toBe(true);
    expect(gpuAdapterLikelyCpuOnly('foo lavapipe bar')).toBe(true);
    expect(gpuAdapterLikelyCpuOnly('foo · software · bar')).toBe(true);
    expect(gpuAdapterLikelyCpuOnly('NVIDIA GeForce RTX')).toBe(false);
    expect(gpuAdapterLikelyCpuOnly('ANGLE (AMD)')).toBe(false);
  });
});

describe('probeWebGpu', () => {
  it('resolves to a boolean', async () => {
    const v = await probeWebGpu();
    expect(typeof v).toBe('boolean');
  });
});

const webgpuAvailable = await probeWebGpu();

describe.skipIf(!webgpuAvailable)('GpuContext', () => {
  it('create exposes device, queue, retained gpu handle, and adapter summary', async () => {
    const ctx = await GpuContext.create();
    expect(ctx.device).toBeDefined();
    expect(ctx.immediateQueue).toBe(ctx.device.queue);
    expect(ctx.gpu).toBeDefined();
    expect(typeof ctx.adapterLabel).toBe('string');
    expect(ctx.adapterLabel.trim().length).toBeGreaterThan(0);
    expect(typeof ctx.isLikelyCpuSoftwareAdapter).toBe('boolean');
  });
});
