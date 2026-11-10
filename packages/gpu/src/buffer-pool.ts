/**
 * Buffer pool: keep destroyed-but-reusable GPU buffers alive in a per-size /
 * per-usage free-list so a steady-state training step can recycle every
 * intermediate buffer instead of paying for `GPUDevice.createBuffer` and the
 * implicit V8-GC-driven Dawn-pool churn on every iteration.
 *
 * The training loop hits the same shape pattern every step, so this pool
 * stabilizes in a handful of warm-up iterations and from then on every
 * allocation is a `pop()` off an in-process array.
 *
 * Lifecycle:
 *   const buf = acquireStorageBuffer(ctx, byteSize);
 *   // ... use buf ...
 *   releasePooledBuffer(buf);   // returns to the pool (no `destroy()` call)
 *
 * Buffers not registered with the pool fall through to a plain `buf.destroy()`
 * via {@link releasePooledOrDestroy}, so callers in non-trainer code paths
 * (tests, one-shot scripts) don't have to know about the pool.
 */

import './webgpu-globals';
import { flushAllPendingGpuWork } from './pending-work';

/** Standard usage flags for compute-shader storage buffers in this codebase. */
export const STORAGE_BUFFER_USAGE =
  GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;

/** Standard usage flags for one-shot mappable readback buffers. */
export const READBACK_BUFFER_USAGE = GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ;

/**
 * Per-buffer book-keeping. `byteSize` is the **exact** allocated size — we do not
 * round up to power-of-two because the trainer reuses identical shapes every step,
 * so exact-size buckets give us a 100% hit rate without wasting VRAM.
 *
 * `state` tracks whether the buffer is currently checked out (`'in-use'`) or
 * sitting in the free-list (`'free'`). We mutate this in place when acquire /
 * release transitions occur, so double-release calls (which match the
 * historical idempotency of `GPUBuffer.destroy()`) become explicit no-ops
 * rather than corrupting the bucket with duplicates.
 */
type PoolEntry = {
  readonly pool: BufferPool;
  readonly byteSize: number;
  readonly usage: number;
  state: 'free' | 'in-use';
};

/**
 * Module-level registry. Used by {@link releasePooledOrDestroy} so callers can
 * route any tensor's buffer through the right pool without threading a
 * `BufferPool` reference through every API.
 *
 * `WeakMap` keys hold no strong reference: once a buffer is genuinely destroyed
 * (manual `destroy()` or pool eviction) and no longer referenced from JS, the
 * entry is collected automatically.
 */
const POOL_REGISTRY: WeakMap<GPUBuffer, PoolEntry> = new WeakMap();

type BucketKey = string;

function bucketKey(byteSize: number, usage: number): BucketKey {
  return `${String(byteSize)}:${String(usage)}`;
}

export type BufferPoolStats = {
  /** Free buffers currently sitting in the pool waiting for reuse. */
  readonly idleBuffers: number;
  /** Distinct `(byteSize, usage)` buckets present. */
  readonly buckets: number;
  /** Total bytes of GPU memory pinned by idle pool entries. */
  readonly idleBytes: number;
  readonly acquireHits: number;
  readonly acquireMisses: number;
  readonly releases: number;
  /**
   * Releases that bypassed the pool because the bucket was already at
   * {@link BufferPoolOptions.maxIdlePerBucket}. The buffer is destroyed instead.
   */
  readonly overflowDestroys: number;
};

export type BufferPoolOptions = {
  /**
   * Maximum idle buffers retained per `(byteSize, usage)` bucket. Releases
   * beyond this destroy the incoming buffer instead of pooling it. Generous
   * default because a single forward / backward pass can produce dozens of
   * same-sized intermediates that all release together.
   */
  readonly maxIdlePerBucket: number;
};

/**
 * Default cap on idle buffers per `(byteSize, usage)` bucket.
 *
 * Sized for the realistic transformer-step access pattern observed on the
 * `larger-longctx` config (6 layers, 1024 ctx): each training iteration
 * allocates ~2000 buffers spread across ~13 distinct sizes — meaning each
 * bucket sees ~150 same-size buffers in flight before they all release at
 * end-of-step. A cap of 64 here destroyed ~1400 buffers/step via the
 * overflow path, defeating the whole purpose of the pool.
 *
 * `4096` is effectively "no per-bucket limit" for any realistic workload
 * (worst-case idle memory still equals the trainer's natural working set,
 * which we have to fit in VRAM anyway). Memory safety is preserved at the
 * device level: the pool can never hold more bytes than the trainer's
 * non-pooled equivalent.
 */
const DEFAULT_OPTIONS: BufferPoolOptions = {
  maxIdlePerBucket: 4096,
};

export class BufferPool {
  private readonly device: GPUDevice;
  private readonly options: BufferPoolOptions;
  private readonly buckets: Map<BucketKey, GPUBuffer[]> = new Map();

  private hits = 0;
  private misses = 0;
  private releases = 0;
  private overflowDestroys = 0;

  constructor(device: GPUDevice, options: Partial<BufferPoolOptions> = {}) {
    this.device = device;
    this.options = { ...DEFAULT_OPTIONS, ...options };
  }

  /**
   * Acquire a buffer of exactly **`byteSize`** bytes with **`usage`** flags.
   * Returns a cached buffer when one is available, otherwise creates a fresh
   * one via `GPUDevice.createBuffer`.
   */
  acquire(byteSize: number, usage: number): GPUBuffer {
    const key = bucketKey(byteSize, usage);
    const bucket = this.buckets.get(key);
    if (bucket !== undefined && bucket.length > 0) {
      const reused = bucket.pop();
      if (reused !== undefined) {
        this.hits += 1;
        const entry = POOL_REGISTRY.get(reused);
        if (entry !== undefined) {
          entry.state = 'in-use';
        }
        return reused;
      }
    }
    this.misses += 1;
    const fresh = this.device.createBuffer({ size: byteSize, usage });
    POOL_REGISTRY.set(fresh, { pool: this, byteSize, usage, state: 'in-use' });
    return fresh;
  }

  /**
   * Return a buffer to the pool. Caller MUST guarantee the buffer is no longer
   * referenced by any pending GPU command — same lifetime contract as a manual
   * `destroy()` call.
   */
  release(buf: GPUBuffer): void {
    const entry = POOL_REGISTRY.get(buf);
    if (entry === undefined) {
      // Foreign buffer (not tracked by us). Fall back to destroy so we don't
      // leak it, but don't count it in pool stats.
      flushAllPendingGpuWork();
      buf.destroy();
      return;
    }
    if (entry.pool !== this) {
      // Wrong pool: route to the correct one to keep accounting sane.
      entry.pool.release(buf);
      return;
    }
    if (entry.state === 'free') {
      // Idempotent double-release: `GPUBuffer.destroy()` is spec'd to be a
      // no-op the second time, and several call sites rely on that (shared
      // reshape buffers, double-tracked tensors in tests, …). Match that
      // contract instead of corrupting the bucket with duplicates.
      return;
    }
    entry.state = 'free';
    const key = bucketKey(entry.byteSize, entry.usage);
    let bucket = this.buckets.get(key);
    if (bucket === undefined) {
      bucket = [];
      this.buckets.set(key, bucket);
    }
    if (bucket.length >= this.options.maxIdlePerBucket) {
      this.overflowDestroys += 1;
      POOL_REGISTRY.delete(buf);
      flushAllPendingGpuWork();
      buf.destroy();
      return;
    }
    this.releases += 1;
    bucket.push(buf);
  }

  /**
   * Destroy every idle buffer and clear the pool. Counters are preserved so
   * callers can correlate trim events with subsequent miss spikes.
   */
  flush(): void {
    flushAllPendingGpuWork();
    for (const bucket of this.buckets.values()) {
      for (const b of bucket) {
        POOL_REGISTRY.delete(b);
        b.destroy();
      }
      bucket.length = 0;
    }
    this.buckets.clear();
  }

  stats(): BufferPoolStats {
    let idleBuffers = 0;
    let idleBytes = 0;
    let buckets = 0;
    for (const [key, bucket] of this.buckets) {
      if (bucket.length === 0) continue;
      buckets += 1;
      idleBuffers += bucket.length;
      const sepIdx = key.indexOf(':');
      const byteSize = sepIdx > 0 ? Number(key.slice(0, sepIdx)) : 0;
      idleBytes += byteSize * bucket.length;
    }
    return {
      idleBuffers,
      buckets,
      idleBytes,
      acquireHits: this.hits,
      acquireMisses: this.misses,
      releases: this.releases,
      overflowDestroys: this.overflowDestroys,
    };
  }
}

/**
 * Acquire a buffer with the codebase's standard storage-buffer usage flags
 * (`STORAGE | COPY_DST | COPY_SRC`). All compute kernels can bind it.
 */
export function acquireStorageBuffer(pool: BufferPool, byteSize: number): GPUBuffer {
  return pool.acquire(byteSize, STORAGE_BUFFER_USAGE);
}

/** Acquire a mappable readback buffer (`COPY_DST | MAP_READ`). */
export function acquireReadbackBuffer(pool: BufferPool, byteSize: number): GPUBuffer {
  return pool.acquire(byteSize, READBACK_BUFFER_USAGE);
}

/**
 * Return a buffer to its originating pool. Buffers not registered with any
 * pool are destroyed directly so callers don't have to know whether the
 * buffer came from a pool. This is the **single canonical entrypoint** for
 * releasing a buffer in pool-aware code paths.
 */
export function releasePooledOrDestroy(buf: GPUBuffer): void {
  const entry = POOL_REGISTRY.get(buf);
  if (entry === undefined) {
    flushAllPendingGpuWork();
    buf.destroy();
    return;
  }
  entry.pool.release(buf);
}

/** Visible for tests / diagnostics: is this buffer tracked by some pool? */
export function isPoolTracked(buf: GPUBuffer): boolean {
  return POOL_REGISTRY.has(buf);
}
