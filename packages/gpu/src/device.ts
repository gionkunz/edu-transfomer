import { clearPendingGpuWork, notePendingGpuWork } from './pending-work';
import { create } from 'webgpu';

import { BufferPool } from './buffer-pool';

export type GpuBootstrapOptions = {
  /** Options forwarded to Dawn (`webgpu` / `create`), e.g. `['backend=vulkan']`. */
  dawnOptions?: string[];
  /**
   * Forwarded to **`GPU.requestAdapter`**. Defaults include **`powerPreference: 'high-performance'`**
   * so discrete GPUs are preferred when the runtime exposes more than one adapter.
   */
  requestAdapterOptions?: GPURequestAdapterOptions;
  /**
   * Features that **`GPUAdapter.features`** must advertise; **`GpuContext.create`** throws when any
   * entry is missing. These are always passed to **`GPUAdapter.requestDevice`**.
   */
  requiredFeatures?: GPUFeatureName[];
  /**
   * Optional features requested when supported by the adapter (intersected with **`adapter.features`**).
   * Defaults to **`[]`**. Unsupported optional features are silently omitted — the created
   * **`GPUDevice.features`** is the source of truth for what was granted.
   */
  optionalFeatures?: GPUFeatureName[];
};

/** Dtypes with packed GPU storage handled by this repo (mirrors tensor **`DType`** without a package edge). */
export type GpuStorageDType = 'f32' | 'bf16' | 'u32';

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
 * Whether **`dtype`** can be used with **`GpuContext`**: all current storage dtypes are universal
 * — **`bf16`** uses two-scalars-per-`u32` packing in `array<u32>`, **`f32`** / **`u32`** are native.
 *
 * The `ctx` parameter is unused today; it is kept so callers can switch on adapter capabilities
 * in the future (e.g. a future int8 storage path conditional on `subgroups` or similar features).
 */
export function supportsDtype(_ctx: GpuContext, dtype: GpuStorageDType): boolean {
  switch (dtype) {
    case 'f32':
    case 'u32':
    case 'bf16':
      return true;
    default: {
      const _e: never = dtype;
      return _e;
    }
  }
}

function collectDeviceFeatures(
  adapter: GPUAdapter,
  required: readonly GPUFeatureName[],
  optional: readonly GPUFeatureName[],
): GPUFeatureName[] {
  for (const f of required) {
    if (!adapter.features.has(f)) {
      const label = formatGpuAdapterLabel(adapter);
      throw new Error(
        `GpuContext.create: required WebGPU feature '${String(f)}' is not available on this adapter ` +
          `(${label}).`,
      );
    }
  }
  const granted: GPUFeatureName[] = [...required];
  for (const f of optional) {
    if (adapter.features.has(f)) {
      granted.push(f);
    }
  }
  return granted;
}

/**
 * Holds a live Dawn `GPU` handle (keeps the native binding alive), a {@link GPUDevice},
 * and its default queue.
 */
export class GpuContext {
  /** Retain reference so the Dawn Node binding outlives this context. */
  readonly gpu: GPU;

  readonly device: GPUDevice;
  /**
   * The device queue, wrapped so that **`submit`** and **`writeBuffer`** first
   * submit any dispatches recorded by {@link encodeComputeDispatch}. Everything
   * that goes through this queue therefore observes dispatches in program
   * order, exactly as if each had been submitted on its own.
   */
  readonly queue: GPUQueue;
  /**
   * The unwrapped device queue. Only safe for writes to a buffer that no
   * recorded dispatch can reference yet — i.e. one created just now (the
   * per-dispatch uniform buffers). Anything else must use {@link queue}.
   */
  readonly immediateQueue: GPUQueue;

  private pendingEncoder: GPUCommandEncoder | null = null;
  private pendingPass: GPUComputePassEncoder | null = null;
  private pendingDispatches = 0;
  private readonly deferredDestroys: GPUBuffer[] = [];

  /** One-line **`GPUAdapter.info`** summary — see **`formatGpuAdapterLabel`**. */
  readonly adapterLabel: string;

  /** True when **`adapterLabel`** matches known CPU software backends (e.g. LLVMpipe). */
  readonly isLikelyCpuSoftwareAdapter: boolean;

  /** Subset of **`GPUFeatureName`** values actually enabled on **`device`**. */
  readonly grantedFeatures: ReadonlySet<string>;

  /**
   * Per-context buffer pool. All compute storage and readback allocations made
   * through this context's helpers route through here, so a steady-state
   * training step recycles every intermediate buffer instead of creating new
   * `GPUBuffer` objects each iteration.
   */
  readonly bufferPool: BufferPool;

  private constructor(
    gpu: GPU,
    device: GPUDevice,
    adapterLabel: string,
    isLikelyCpuSoftwareAdapter: boolean,
    grantedFeatures: ReadonlySet<string>,
  ) {
    this.gpu = gpu;
    this.device = device;
    this.immediateQueue = device.queue;
    const realQueue = device.queue;
    const flushPending = (): void => {
      this.flush();
    };
    this.queue = new Proxy(realQueue, {
      get(target, prop): unknown {
        if (prop === 'submit') {
          return (commandBuffers: Iterable<GPUCommandBuffer>): void => {
            flushPending();
            target.submit(commandBuffers);
          };
        }
        if (prop === 'writeBuffer') {
          return (...args: Parameters<GPUQueue['writeBuffer']>): void => {
            flushPending();
            target.writeBuffer(...args);
          };
        }
        const value: unknown = Reflect.get(target, prop, target);
        // Native methods need the real queue as receiver, not the proxy.
        return typeof value === 'function'
          ? (value as (...a: unknown[]) => unknown).bind(target)
          : value;
      },
    });
    this.adapterLabel = adapterLabel;
    this.isLikelyCpuSoftwareAdapter = isLikelyCpuSoftwareAdapter;
    this.grantedFeatures = grantedFeatures;
    this.bufferPool = new BufferPool(device);
  }

  /**
   * Record one compute dispatch into the shared command encoder instead of
   * submitting it on its own.
   *
   * Why: a training step issues thousands of tiny dispatches, and on D3D12 /
   * Vulkan each `queue.submit` costs far more than the dispatch it carries.
   * Recording them into one compute pass and submitting once per
   * {@link MAX_PENDING_DISPATCHES} (or whenever the host needs a result) keeps
   * the GPU work identical while removing most of the per-op overhead.
   *
   * Ordering is preserved: dispatches inside a pass run in recording order, and
   * {@link queue} flushes before any other submit or buffer write.
   */
  encodeComputeDispatch(
    pipeline: GPUComputePipeline,
    bindGroup: GPUBindGroup,
    workgroups: readonly [number, number?, number?],
  ): void {
    if (this.pendingEncoder === null || this.pendingPass === null) {
      this.pendingEncoder = this.device.createCommandEncoder();
      this.pendingPass = this.pendingEncoder.beginComputePass();
      notePendingGpuWork(this);
    }
    this.pendingPass.setPipeline(pipeline);
    this.pendingPass.setBindGroup(0, bindGroup);
    this.pendingPass.dispatchWorkgroups(...workgroups);
    this.pendingDispatches += 1;
    if (this.pendingDispatches >= GpuContext.MAX_PENDING_DISPATCHES) {
      this.flush();
    }
  }

  /** Upper bound on recorded dispatches before an automatic {@link flush}. */
  static readonly MAX_PENDING_DISPATCHES = 512;

  /** Submit every recorded dispatch, then destroy buffers parked by {@link destroyBuffer}. */
  flush(): void {
    const encoder = this.pendingEncoder;
    const pass = this.pendingPass;
    if (encoder !== null && pass !== null) {
      this.pendingEncoder = null;
      this.pendingPass = null;
      this.pendingDispatches = 0;
      clearPendingGpuWork(this);
      pass.end();
      this.immediateQueue.submit([encoder.finish()]);
    }
    if (this.deferredDestroys.length > 0) {
      for (const b of this.deferredDestroys) {
        b.destroy();
      }
      this.deferredDestroys.length = 0;
    }
  }

  /**
   * Destroy **`buffer`** once no recorded dispatch can reference it: right away
   * when nothing is pending, otherwise at the next {@link flush}. (Destroying a
   * buffer used by a not-yet-submitted command buffer invalidates the submit.)
   */
  destroyBuffer(buffer: GPUBuffer): void {
    if (this.pendingEncoder === null) {
      buffer.destroy();
    } else {
      this.deferredDestroys.push(buffer);
    }
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

    const required = options?.requiredFeatures ?? [];
    const optional = options?.optionalFeatures ?? [];
    const deviceFeatures = collectDeviceFeatures(adapter, required, optional);

    let device: GPUDevice;
    try {
      device = await adapter.requestDevice({
        requiredFeatures: deviceFeatures.length > 0 ? deviceFeatures : undefined,
      });
    } catch (cause) {
      const message = cause instanceof Error ? cause.message : String(cause);
      throw new Error(`GpuContext.create: requestDevice failed (${message})`);
    }

    return new GpuContext(
      gpu,
      device,
      adapterLabel,
      isLikelyCpuSoftwareAdapter,
      new Set<string>(device.features),
    );
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
