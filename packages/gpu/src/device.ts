import { create } from 'webgpu';

export type GpuBootstrapOptions = {
  /** Options forwarded to Dawn (`webgpu` / `create`), e.g. `['backend=vulkan']`. */
  dawnOptions?: string[];
  /**
   * Forwarded to **`GPU.requestAdapter`**. Defaults include **`powerPreference: 'high-performance'`**
   * so discrete GPUs are preferred when the runtime exposes more than one adapter.
   */
  requestAdapterOptions?: GPURequestAdapterOptions;
};

/** Heuristic: adapter label / info often marks LLVMpipe, SwiftShader, or explicit “software”. */
export function gpuAdapterLikelyCpuOnly(adapterLabel: string): boolean {
  const l = adapterLabel.toLowerCase();
  return (
    l.includes('llvmpipe') ||
    l.includes('swiftshader') ||
    l.includes('lavapipe') ||
    /\bsoftware\b/.test(l)
  );
}

/**
 * Best-effort human-readable adapter line from **`GPUAdapter.info`** (vendor / GPU name),
 * falling back when the backend omits strings.
 */
export function formatGpuAdapterLabel(adapter: GPUAdapter): string {
  const chunks: string[] = [];
  try {
    if ('info' in adapter && adapter.info !== null && typeof adapter.info === 'object') {
      const i = adapter.info as Partial<GPUAdapterInfo>;
      for (const k of ['description', 'device', 'architecture', 'vendor'] as const) {
        const v = i[k];
        if (typeof v === 'string' && v.trim().length > 0) {
          chunks.push(v.trim());
        }
      }
    }
  } catch {
    /* ignore malformed adapter.info */
  }
  return chunks.length > 0 ? chunks.join(' · ') : 'WebGPU adapter (no description from runtime)';
}

/**
 * Holds a live Dawn `GPU` handle (keeps the native binding alive), a {@link GPUDevice},
 * and its default queue.
 */
export class GpuContext {
  /** Retain reference so the Dawn Node binding outlives this context. */
  readonly gpu: GPU;

  readonly device: GPUDevice;
  readonly queue: GPUQueue;

  /** One-line **`GPUAdapter.info`** summary — see **`formatGpuAdapterLabel`**. */
  readonly adapterLabel: string;

  /** True when **`adapterLabel`** matches known CPU software backends (e.g. LLVMpipe). */
  readonly isLikelyCpuSoftwareAdapter: boolean;

  private constructor(
    gpu: GPU,
    device: GPUDevice,
    adapterLabel: string,
    isLikelyCpuSoftwareAdapter: boolean,
  ) {
    this.gpu = gpu;
    this.device = device;
    this.queue = device.queue;
    this.adapterLabel = adapterLabel;
    this.isLikelyCpuSoftwareAdapter = isLikelyCpuSoftwareAdapter;
  }

  /**
   * Initialize WebGPU: request an adapter and device. Fails loudly if bootstrap cannot proceed.
   */
  static async create(options?: GpuBootstrapOptions): Promise<GpuContext> {
    const dawnOptions = options?.dawnOptions ?? [];
    const gpu = create(dawnOptions);
    const requestOpts: GPURequestAdapterOptions = {
      powerPreference: 'high-performance',
      ...options?.requestAdapterOptions,
    };
    const adapter = await gpu.requestAdapter(requestOpts);
    if (!adapter) {
      throw new Error(
        'GpuContext.create: requestAdapter returned null — no suitable WebGPU adapter is available.',
      );
    }

    const adapterLabel = formatGpuAdapterLabel(adapter);
    const isLikelyCpuSoftwareAdapter = gpuAdapterLikelyCpuOnly(adapterLabel);

    let device: GPUDevice;
    try {
      device = await adapter.requestDevice();
    } catch (cause) {
      const message = cause instanceof Error ? cause.message : String(cause);
      throw new Error(`GpuContext.create: requestDevice failed (${message})`);
    }

    return new GpuContext(gpu, device, adapterLabel, isLikelyCpuSoftwareAdapter);
  }
}

/**
 * Returns whether an adapter can be obtained (without creating a full {@link GpuContext}).
 * Intended for tests and optional GPU paths; {@link GpuContext.create} still throws on failure.
 */
export async function probeWebGpu(dawnOptions: string[] = []): Promise<boolean> {
  try {
    const gpu = create(dawnOptions);
    const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
    return adapter !== null;
  } catch {
    return false;
  }
}
