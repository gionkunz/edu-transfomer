import { describe, expect, it } from 'vitest';

import { GpuContext, probeWebGpu } from './device';

describe('probeWebGpu', () => {
  it('resolves to a boolean', async () => {
    const v = await probeWebGpu();
    expect(typeof v).toBe('boolean');
  });
});

const webgpuAvailable = await probeWebGpu();

describe.skipIf(!webgpuAvailable)('GpuContext', () => {
  it('create exposes device, queue, and retained gpu handle', async () => {
    const ctx = await GpuContext.create();
    expect(ctx.device).toBeDefined();
    expect(ctx.queue).toBe(ctx.device.queue);
    expect(ctx.gpu).toBeDefined();
  });
});
