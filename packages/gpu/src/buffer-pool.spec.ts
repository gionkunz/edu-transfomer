/// <reference types="@webgpu/types" />

import { afterEach, beforeAll, describe, expect, it } from 'vitest';

import {
  acquireStorageBuffer,
  BufferPool,
  isPoolTracked,
  releasePooledOrDestroy,
  STORAGE_BUFFER_USAGE,
} from './buffer-pool';
import { GpuContext, probeWebGpu } from './device';

const webgpuAvailable = await probeWebGpu();

describe.skipIf(!webgpuAvailable)('BufferPool (GPU)', () => {
  let ctx: GpuContext;
  let pool: BufferPool;
  const created: GPUBuffer[] = [];

  beforeAll(async () => {
    ctx = await GpuContext.create();
    pool = new BufferPool(ctx.device, { maxIdlePerBucket: 4 });
  });

  afterEach(() => {
    pool.flush();
    while (created.length > 0) {
      const b = created.pop();
      if (b !== undefined) {
        try {
          b.destroy();
        } catch {
          /* ignore — pool may already have destroyed it */
        }
      }
    }
  });

  it('returns a fresh buffer on first acquire (miss)', () => {
    const buf = acquireStorageBuffer(pool, 64);
    created.push(buf);
    expect(isPoolTracked(buf)).toBe(true);
    const stats = pool.stats();
    expect(stats.acquireHits).toBe(0);
    expect(stats.acquireMisses).toBe(1);
  });

  it('recycles a released buffer on next acquire of same size', () => {
    const a = acquireStorageBuffer(pool, 64);
    releasePooledOrDestroy(a);
    const b = acquireStorageBuffer(pool, 64);
    created.push(b);
    expect(b).toBe(a);
    const stats = pool.stats();
    expect(stats.acquireHits).toBe(1);
  });

  it('keeps different sizes in different buckets', () => {
    const a = acquireStorageBuffer(pool, 64);
    const b = acquireStorageBuffer(pool, 128);
    releasePooledOrDestroy(a);
    releasePooledOrDestroy(b);
    const reused = acquireStorageBuffer(pool, 64);
    created.push(reused);
    expect(reused).toBe(a);
    expect(reused).not.toBe(b);
  });

  it('treats a double-release as a no-op (matches GPUBuffer.destroy() idempotency)', () => {
    const a = acquireStorageBuffer(pool, 64);
    releasePooledOrDestroy(a);
    releasePooledOrDestroy(a);
    const first = acquireStorageBuffer(pool, 64);
    created.push(first);
    expect(first).toBe(a);
    // Second acquire must NOT return the same buffer; it should be a fresh one,
    // because the duplicate release was ignored.
    const second = acquireStorageBuffer(pool, 64);
    created.push(second);
    expect(second).not.toBe(a);
  });

  it('destroys overflow when bucket is full instead of pooling', () => {
    const buffers: GPUBuffer[] = [];
    for (let i = 0; i < 5; i++) {
      buffers.push(acquireStorageBuffer(pool, 64));
    }
    for (const b of buffers) {
      releasePooledOrDestroy(b);
    }
    const stats = pool.stats();
    expect(stats.overflowDestroys).toBe(1);
    expect(stats.idleBuffers).toBe(4);
  });

  it('untracked buffers fall through to plain destroy()', () => {
    const plain = ctx.device.createBuffer({ size: 64, usage: STORAGE_BUFFER_USAGE });
    expect(isPoolTracked(plain)).toBe(false);
    releasePooledOrDestroy(plain);
    // No throw: the wrapper degenerated to plain destroy(). We can't observe
    // post-destroy state safely beyond this without invoking the device.
  });

  it('flush empties every bucket', () => {
    const a = acquireStorageBuffer(pool, 64);
    const b = acquireStorageBuffer(pool, 128);
    releasePooledOrDestroy(a);
    releasePooledOrDestroy(b);
    pool.flush();
    expect(pool.stats().idleBuffers).toBe(0);
    expect(pool.stats().buckets).toBe(0);
  });
});
