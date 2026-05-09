/** Byte length for a float32 vector of `elementCount` elements. */
export function float32ByteLength(elementCount: number): number {
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
