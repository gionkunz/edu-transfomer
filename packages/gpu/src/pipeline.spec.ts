import './webgpu-globals';

import { describe, expect, it } from 'vitest';

import { GpuContext, probeWebGpu } from './device';
import { createSimpleComputePipeline } from './pipeline';

const MINIMAL_COMPUTE_WGSL = `
@group(0) @binding(0) var<storage, read_write> data: array<f32>;
@compute @workgroup_size(1)
fn main() {}
`;

const webgpuAvailable = await probeWebGpu();

describe.skipIf(!webgpuAvailable)('createSimpleComputePipeline', () => {
  it('compiles a minimal storage compute pipeline', async () => {
    const ctx = await GpuContext.create();
    const { pipeline, bindGroupLayout } = createSimpleComputePipeline(ctx.device, {
      code: MINIMAL_COMPUTE_WGSL,
      entryPoint: 'main',
      bindings: [
        {
          binding: 0,
          visibility: GPUShaderStage.COMPUTE,
          buffer: { type: 'storage' },
        },
      ],
    });
    expect(pipeline).toBeDefined();
    expect(bindGroupLayout).toBeDefined();
  });
});
