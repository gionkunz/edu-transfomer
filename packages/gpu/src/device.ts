import { create } from 'webgpu';

export type GpuBootstrapOptions = {
  /** Options forwarded to Dawn (`webgpu` / `create`), e.g. `['backend=vulkan']`. */
  dawnOptions?: string[];
};

/**
 * Holds a live Dawn `GPU` handle (keeps the native binding alive), a {@link GPUDevice},
 * and its default queue.
 */
export class GpuContext {
  /** Retain reference so the Dawn Node binding outlives this context. */
  readonly gpu: GPU;

  readonly device: GPUDevice;
  readonly queue: GPUQueue;

  private constructor(gpu: GPU, device: GPUDevice) {
    this.gpu = gpu;
    this.device = device;
    this.queue = device.queue;
  }

  /**
   * Initialize WebGPU: request an adapter and device. Fails loudly if bootstrap cannot proceed.
   */
  static async create(options?: GpuBootstrapOptions): Promise<GpuContext> {
    const dawnOptions = options?.dawnOptions ?? [];
    const gpu = create(dawnOptions);
    const adapter = await gpu.requestAdapter();
    if (!adapter) {
      throw new Error(
        'GpuContext.create: requestAdapter returned null — no suitable WebGPU adapter is available.',
      );
    }

    let device: GPUDevice;
    try {
      device = await adapter.requestDevice();
    } catch (cause) {
      const message = cause instanceof Error ? cause.message : String(cause);
      throw new Error(`GpuContext.create: requestDevice failed (${message})`);
    }

    return new GpuContext(gpu, device);
  }
}

/**
 * Returns whether an adapter can be obtained (without creating a full {@link GpuContext}).
 * Intended for tests and optional GPU paths; {@link GpuContext.create} still throws on failure.
 */
export async function probeWebGpu(dawnOptions: string[] = []): Promise<boolean> {
  try {
    const gpu = create(dawnOptions);
    const adapter = await gpu.requestAdapter();
    return adapter !== null;
  } catch {
    return false;
  }
}
