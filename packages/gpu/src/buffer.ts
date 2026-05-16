import {
  acquireReadbackBuffer,
  acquireStorageBuffer,
  releasePooledOrDestroy,
  type BufferPool,
} from './buffer-pool';

/**
 * Number of **`u32`** words used to store **`elementCount`** packed bf16 scalars.
 * Two scalars are packed per word (low / high 16 bits); odd counts round up with a
 * zero-padded high half so the trailing word is still safely addressable.
 */
export function bf16WordCount(elementCount: number): number {
  return (elementCount + 1) >>> 1;
}

/**
 * Byte length of a **bf16** GPU storage vector of **`elementCount`** scalars.
 * Two scalars pack into one **`u32`** word ⇒ `ceil(N/2) * 4` bytes. This is the genuine
 * 16-bit-per-element storage we exploit for VRAM and bandwidth savings; safe concurrent
 * writes are achieved by dispatching one thread per pair (each thread owns one whole word)
 * for ordinary kernels, or by half-word `atomicCompareExchangeWeak` RMW for accumulators
 * (see `embedding-scatter-accum.bf16.frag.wgsl`).
 */
export function bf16ElementsByteLength(elementCount: number): number {
  return bf16WordCount(elementCount) * 4;
}

/** Same byte length as {@link bf16ElementsByteLength}; alias for call-site clarity. */
export function bf16StorageBufferByteLength(elementCount: number): number {
  return bf16ElementsByteLength(elementCount);
}

/** Byte length for a float32 vector of `elementCount` elements. */
export function float32ByteLength(elementCount: number): number {
  return elementCount * 4;
}

/** Byte length for a `u32` vector of `elementCount` elements. */
export function uint32ByteLength(elementCount: number): number {
  return elementCount * 4;
}

/**
 * Storage buffer for f32 compute: read/write in shaders, copy src/dst for I/O.
 *
 * When **`pool`** is provided the allocation routes through the pool (recycled
 * after a previous {@link releasePooledOrDestroy}); otherwise we create a
 * fresh buffer. Tests and one-shot callers can keep passing the device alone.
 */
export function createFloat32StorageBuffer(
  device: GPUDevice,
  elementCount: number,
  pool?: BufferPool,
): GPUBuffer {
  const size = float32ByteLength(elementCount);
  if (pool !== undefined) {
    return acquireStorageBuffer(pool, size);
  }
  return device.createBuffer({
    size,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
}

export function writeFloat32Buffer(
  queue: GPUQueue,
  buffer: GPUBuffer,
  data: Float32Array,
  bufferOffset = 0,
): void {
  queue.writeBuffer(buffer, bufferOffset, data.buffer, data.byteOffset, data.byteLength);
}

/**
 * Copies GPU buffer bytes into a new {@link Float32Array} using a dedicated readback buffer.
 * `source` must have been created with {@link GPUBufferUsage.COPY_SRC}.
 *
 * Previously this function leaked the temporary readback buffer (only
 * `unmap()`ed it, which keeps the GPU allocation alive until V8 collects the
 * `GPUBuffer` wrapper). The buffer is now either released to **`pool`** for
 * reuse, or destroyed directly when no pool is provided.
 */
export async function readFloat32Buffer(
  device: GPUDevice,
  queue: GPUQueue,
  source: GPUBuffer,
  byteLength: number,
  pool?: BufferPool,
): Promise<Float32Array> {
  const readback =
    pool !== undefined
      ? acquireReadbackBuffer(pool, byteLength)
      : device.createBuffer({
          size: byteLength,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });

  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(source, 0, readback, 0, byteLength);
  queue.submit([encoder.finish()]);

  await readback.mapAsync(GPUMapMode.READ);
  const mapped = readback.getMappedRange();
  const copy = new Uint8Array(mapped.slice(0));
  readback.unmap();
  releasePooledOrDestroy(readback);

  return new Float32Array(copy.buffer, copy.byteOffset, copy.byteLength / 4);
}

/** Storage buffer for `array<u32>` WGSL layouts (embedding indices, class labels). */
export function createUint32StorageBuffer(
  device: GPUDevice,
  elementCount: number,
  pool?: BufferPool,
): GPUBuffer {
  const size = uint32ByteLength(elementCount);
  if (pool !== undefined) {
    return acquireStorageBuffer(pool, size);
  }
  return device.createBuffer({
    size,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
}

export function writeUint32Buffer(
  queue: GPUQueue,
  buffer: GPUBuffer,
  data: Uint32Array,
  bufferOffset = 0,
): void {
  queue.writeBuffer(buffer, bufferOffset, data.buffer, data.byteOffset, data.byteLength);
}

/** Readback helper mirroring {@link readFloat32Buffer}; `source` must allow `COPY_SRC`. */
export async function readUint32Buffer(
  device: GPUDevice,
  queue: GPUQueue,
  source: GPUBuffer,
  byteLength: number,
  pool?: BufferPool,
): Promise<Uint32Array> {
  const readback =
    pool !== undefined
      ? acquireReadbackBuffer(pool, byteLength)
      : device.createBuffer({
          size: byteLength,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });

  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(source, 0, readback, 0, byteLength);
  queue.submit([encoder.finish()]);

  await readback.mapAsync(GPUMapMode.READ);
  const mapped = readback.getMappedRange();
  const copy = new Uint8Array(mapped.slice(0));
  readback.unmap();
  releasePooledOrDestroy(readback);

  return new Uint32Array(copy.buffer, copy.byteOffset, copy.byteLength / 4);
}

/** Storage buffer sized with {@link bf16StorageBufferByteLength}. */
export function createBf16StorageBuffer(
  device: GPUDevice,
  elementCount: number,
  pool?: BufferPool,
): GPUBuffer {
  const size = bf16StorageBufferByteLength(elementCount);
  if (pool !== undefined) {
    return acquireStorageBuffer(pool, size);
  }
  return device.createBuffer({
    size,
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
}

/**
 * Upload **`data`** as **bf16** storage (low/high halves of each **`u32`** word).
 *
 * **`data[2k]`** → low 16 bits of word `k`, **`data[2k+1]`** → high 16 bits. Odd lengths
 * leave the trailing high half zero so callers can size buffers with
 * {@link bf16StorageBufferByteLength} without worrying about the parity of N.
 */
export function writeBf16Buffer(
  queue: GPUQueue,
  buffer: GPUBuffer,
  data: Uint16Array,
  bufferOffset = 0,
): void {
  const wordCount = bf16WordCount(data.length);
  const words = new Uint32Array(wordCount);
  const lastFull = data.length & ~1;
  for (let i = 0; i < lastFull; i += 2) {
    words[i >>> 1] = (data[i] & 0xffff) | ((data[i + 1] & 0xffff) << 16);
  }
  if (data.length & 1) {
    words[wordCount - 1] = data[lastFull] & 0xffff;
  }
  queue.writeBuffer(buffer, bufferOffset, words.buffer, words.byteOffset, words.byteLength);
}

/**
 * Read back **`elementCount`** packed bf16 scalars. Returns a **`Uint16Array`** of length
 * **`elementCount`** containing the raw bf16 bit patterns; trailing zero-padding of the
 * underlying buffer (when **`elementCount`** is odd) is hidden from the caller.
 */
export async function readBf16Buffer(
  device: GPUDevice,
  queue: GPUQueue,
  source: GPUBuffer,
  elementCount: number,
  pool?: BufferPool,
): Promise<Uint16Array> {
  const byteLength = bf16ElementsByteLength(elementCount);
  const readback =
    pool !== undefined
      ? acquireReadbackBuffer(pool, byteLength)
      : device.createBuffer({
          size: byteLength,
          usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ,
        });

  const encoder = device.createCommandEncoder();
  encoder.copyBufferToBuffer(source, 0, readback, 0, byteLength);
  queue.submit([encoder.finish()]);

  await readback.mapAsync(GPUMapMode.READ);
  const mapped = readback.getMappedRange();
  const words = new Uint32Array(mapped.slice(0, byteLength));
  readback.unmap();
  releasePooledOrDestroy(readback);

  const out = new Uint16Array(elementCount);
  const lastFull = elementCount & ~1;
  for (let i = 0; i < lastFull; i += 2) {
    const w = words[i >>> 1];
    out[i] = w & 0xffff;
    out[i + 1] = (w >>> 16) & 0xffff;
  }
  if (elementCount & 1) {
    out[lastFull] = words[words.length - 1] & 0xffff;
  }
  return out;
}
