/**
 * Byte length of a **bf16** / **fp16** GPU storage vector of **`elementCount`** scalars.
 * Each scalar occupies one **`u32`** (low 16 bits) so parallel kernels never RMW the same word.
 */
export function bf16Fp16ElementsByteLength(elementCount: number): number {
  return elementCount * 4;
}

/** Same as {@link bf16Fp16ElementsByteLength} — buffers are already 4-byte aligned per element. */
export function bf16Fp16StorageBufferByteLength(elementCount: number): number {
  return elementCount * 4;
}

/** Byte length for a float32 vector of `elementCount` elements. */
export function float32ByteLength(elementCount: number): number {
  return elementCount * 4;
}

/** Byte length for a `u32` vector of `elementCount` elements. */
export function uint32ByteLength(elementCount: number): number {
  return elementCount * 4;
}

/** Storage buffer for f32 compute: read/write in shaders, copy src/dst for I/O. */
export function createFloat32StorageBuffer(device: GPUDevice, elementCount: number): GPUBuffer {
  return device.createBuffer({
    size: float32ByteLength(elementCount),
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
 */
export async function readFloat32Buffer(
  device: GPUDevice,
  queue: GPUQueue,
  source: GPUBuffer,
  byteLength: number,
): Promise<Float32Array> {
  const readback = device.createBuffer({
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

  return new Float32Array(copy.buffer, copy.byteOffset, copy.byteLength / 4);
}

/** Storage buffer for `array<u32>` WGSL layouts (embedding indices, class labels). */
export function createUint32StorageBuffer(device: GPUDevice, elementCount: number): GPUBuffer {
  return device.createBuffer({
    size: uint32ByteLength(elementCount),
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
): Promise<Uint32Array> {
  const readback = device.createBuffer({
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

  return new Uint32Array(copy.buffer, copy.byteOffset, copy.byteLength / 4);
}

/** Storage buffer sized with {@link bf16Fp16StorageBufferByteLength}. */
export function createBf16StorageBuffer(device: GPUDevice, elementCount: number): GPUBuffer {
  return device.createBuffer({
    size: bf16Fp16StorageBufferByteLength(elementCount),
    usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC,
  });
}

/** Same layout as {@link createBf16StorageBuffer} — distinct helpers for call-site clarity. */
export function createFp16StorageBuffer(device: GPUDevice, elementCount: number): GPUBuffer {
  return createBf16StorageBuffer(device, elementCount);
}

export function writeBf16Buffer(
  queue: GPUQueue,
  buffer: GPUBuffer,
  data: Uint16Array,
  bufferOffset = 0,
): void {
  const words = new Uint32Array(data.length);
  for (let i = 0; i < data.length; i++) {
    words[i] = data[i] & 0xffff;
  }
  queue.writeBuffer(buffer, bufferOffset, words.buffer, words.byteOffset, words.byteLength);
}

export function writeFp16Buffer(
  queue: GPUQueue,
  buffer: GPUBuffer,
  data: Uint16Array,
  bufferOffset = 0,
): void {
  writeBf16Buffer(queue, buffer, data, bufferOffset);
}

export async function readBf16Buffer(
  device: GPUDevice,
  queue: GPUQueue,
  source: GPUBuffer,
  elementCount: number,
): Promise<Uint16Array> {
  const byteLength = bf16Fp16ElementsByteLength(elementCount);
  const readback = device.createBuffer({
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

  const out = new Uint16Array(elementCount);
  for (let i = 0; i < elementCount; i++) {
    out[i] = words[i] & 0xffff;
  }
  return out;
}

export async function readFp16Buffer(
  device: GPUDevice,
  queue: GPUQueue,
  source: GPUBuffer,
  elementCount: number,
): Promise<Uint16Array> {
  return readBf16Buffer(device, queue, source, elementCount);
}
