import { createFloat32StorageBuffer, readFloat32Buffer, writeFloat32Buffer } from './buffer';
import type { GpuContext } from './device';
import type { KernelRegistry } from './kernel-registry';
import { loadTrivialFillWgsl } from './kernels/load-trivial-fill';
import { createSimpleComputePipeline } from './pipeline';

/** Stable ID for the bundled fill kernel (π in every lane). */
export const BOOTSTRAP_KERNEL_FILL_PI = 'bootstrap.fill-pi';

const WORKGROUP_SIZE = 256;

export function registerBootstrapKernels(device: GPUDevice, registry: KernelRegistry): void {
  const code = loadTrivialFillWgsl();
  const { pipeline } = createSimpleComputePipeline(device, {
    code,
    entryPoint: 'main',
    bindings: [
      {
        binding: 0,
        visibility: GPUShaderStage.COMPUTE,
        buffer: { type: 'storage' },
      },
    ],
  });
  registry.register(BOOTSTRAP_KERNEL_FILL_PI, pipeline);
}

export function dispatchFillPi(
  ctx: GpuContext,
  registry: KernelRegistry,
  buffer: GPUBuffer,
  elementCount: number,
): void {
  const pipeline = registry.get(BOOTSTRAP_KERNEL_FILL_PI);
  const bindGroup = ctx.device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      {
        binding: 0,
        resource: { buffer },
      },
    ],
  });

  const encoder = ctx.device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bindGroup);
  const wg = Math.ceil(elementCount / WORKGROUP_SIZE);
  const MAX_PER_DIM = 65535;
  if (wg <= MAX_PER_DIM) {
    pass.dispatchWorkgroups(wg);
  } else {
    pass.dispatchWorkgroups(MAX_PER_DIM, Math.ceil(wg / MAX_PER_DIM), 1);
  }
  pass.end();
  ctx.queue.submit([encoder.finish()]);
}

/** Convenience: allocate, optionally zero upload, run fill-π, return host floats. */
export async function smokeFillPiBuffer(
  ctx: GpuContext,
  registry: KernelRegistry,
  elementCount: number,
): Promise<Float32Array> {
  const buffer = createFloat32StorageBuffer(ctx.device, elementCount);
  writeFloat32Buffer(ctx.queue, buffer, new Float32Array(elementCount));
  dispatchFillPi(ctx, registry, buffer, elementCount);
  const bytes = elementCount * 4;
  const out = await readFloat32Buffer(ctx.device, ctx.queue, buffer, bytes);
  buffer.destroy();
  return out;
}
