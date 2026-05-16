import {
  bf16StorageBufferByteLength,
  createBf16StorageBuffer,
  createFloat32StorageBuffer,
  createUint32StorageBuffer,
  dispatchCastBf16ToF32,
  dispatchCastF32ToBf16,
  float32ByteLength,
  packBf16FromFloat32,
  readBf16Buffer,
  readFloat32Buffer,
  readUint32Buffer,
  releasePooledOrDestroy,
  uint32ByteLength,
  writeBf16Buffer,
  writeFloat32Buffer,
  writeUint32Buffer,
  unpackBf16ToFloat32,
} from '@llm-transformer-diy/gpu';

import {
  assertDtype,
  assertSameDtype,
  assertSameShape,
  computeContiguousStrides,
  isContiguous,
  productOfShape,
  type FloatDType,
} from './shape';
import type { DType, Tensor, TensorRuntime } from './types';

/** Byte length of a contiguous GPU buffer backing **`length`** scalars at **`dtype`**. */
export function gpuStorageByteLength(length: number, dtype: DType): number {
  switch (dtype) {
    case 'f32':
      return float32ByteLength(length);
    case 'u32':
      return uint32ByteLength(length);
    case 'bf16':
      return bf16StorageBufferByteLength(length);
    default: {
      const _e: never = dtype;
      return _e;
    }
  }
}

/**
 * Internal helper: build a contiguous {@link Tensor} record around an existing
 * GPU buffer. Not exported from the package index; ops and constructors share
 * this so the `Tensor` shape stays consistent.
 */
export function makeContiguousTensor(
  shape: readonly number[],
  buffer: GPUBuffer,
  dtype: FloatDType = 'f32',
): Tensor {
  return {
    shape,
    strides: computeContiguousStrides(shape),
    dtype,
    device: 'gpu',
    buffer,
    length: productOfShape(shape),
  };
}

/** Internal contiguous builder for **`dtype: 'u32'`** tensors. */
export function makeIndicesTensor(shape: readonly number[], buffer: GPUBuffer): Tensor {
  return {
    shape,
    strides: computeContiguousStrides(shape),
    dtype: 'u32',
    device: 'gpu',
    buffer,
    length: productOfShape(shape),
  };
}

/**
 * Allocates a fresh storage buffer (uninitialized) for **`length`** scalars at **`dtype`**.
 * Internal helper — exported for `ops.ts` / autograd; not re-exported via index.
 */
export function allocateContiguousBuffer(
  rt: TensorRuntime,
  length: number,
  dtype: FloatDType,
): GPUBuffer {
  const pool = rt.ctx.bufferPool;
  switch (dtype) {
    case 'f32':
      return createFloat32StorageBuffer(rt.ctx.device, length, pool);
    case 'bf16':
      return createBf16StorageBuffer(rt.ctx.device, length, pool);
    default: {
      const _e: never = dtype;
      return _e;
    }
  }
}

/**
 * Allocate a tensor of the given shape, zero-fill the underlying buffer, and
 * return the immutable record. Caller owns the buffer and must call
 * {@link destroy} when done.
 */
export function createZeros(
  rt: TensorRuntime,
  shape: readonly number[],
  dtype: FloatDType = 'f32',
): Tensor {
  const length = productOfShape(shape);
  const buffer = allocateContiguousBuffer(rt, length, dtype);
  const zeroBytes = new Uint8Array(gpuStorageByteLength(length, dtype));
  rt.ctx.queue.writeBuffer(buffer, 0, zeroBytes);
  return makeContiguousTensor(shape, buffer, dtype);
}

/**
 * Allocate a tensor of the given shape and upload **`data`** after rounding to **`dtype`**.
 * Throws when `data.length` does not match the product of `shape`.
 */
export function createFromArray(
  rt: TensorRuntime,
  data: Float32Array,
  shape: readonly number[],
  dtype: FloatDType = 'f32',
): Tensor {
  const length = productOfShape(shape);
  if (data.length !== length) {
    throw new Error(
      `createFromArray: data.length=${String(data.length)} does not match ` +
        `productOfShape([${shape.join(',')}])=${String(length)}`,
    );
  }
  const buffer = allocateContiguousBuffer(rt, length, dtype);
  if (dtype === 'f32') {
    writeFloat32Buffer(rt.ctx.queue, buffer, data);
  } else {
    writeBf16Buffer(rt.ctx.queue, buffer, packBf16FromFloat32(data));
  }
  return makeContiguousTensor(shape, buffer, dtype);
}

/**
 * Immutable u32 tensor for discrete indices (**embedding lookup**, softmax targets, …).
 *
 * WGSL binds it as readonly **`array<u32>`** (`STORAGE` buffer backed by **`createIndices`** upload).
 */
export function createIndices(
  rt: TensorRuntime,
  data: Uint32Array,
  shape: readonly number[],
): Tensor {
  const length = productOfShape(shape);
  if (data.length !== length) {
    throw new Error(
      `createIndices: data.length=${String(data.length)} does not match ` +
        `productOfShape([${shape.join(',')}])=${String(length)}`,
    );
  }
  const buf = createUint32StorageBuffer(rt.ctx.device, length, rt.ctx.bufferPool);
  writeUint32Buffer(rt.ctx.queue, buf, data);
  return makeIndicesTensor(shape, buf);
}

/**
 * Read tensor contents back to host as a fresh {@link Float32Array}.
 *
 * `async` because WebGPU's `mapAsync` is async; this is the only async surface
 * in the package. Use sparingly in hot paths.
 */
export async function toArray(rt: TensorRuntime, t: Tensor): Promise<Float32Array> {
  if (t.dtype === 'u32') {
    throw new Error("toArray: use readIndices for dtype 'u32'");
  }
  const pool = rt.ctx.bufferPool;
  if (t.dtype === 'f32') {
    return readFloat32Buffer(
      rt.ctx.device,
      rt.ctx.queue,
      t.buffer,
      float32ByteLength(t.length),
      pool,
    );
  }
  return unpackBf16ToFloat32(
    await readBf16Buffer(rt.ctx.device, rt.ctx.queue, t.buffer, t.length, pool),
  );
}

/**
 * Copy **`t`** to **`dtype`** (new buffer). Same dtype performs a full GPU buffer copy
 * (non-aliasing). **`u32`** is not supported.
 */
export function cast(rt: TensorRuntime, t: Tensor, dtype: FloatDType): Tensor {
  if (t.dtype === 'u32') {
    throw new Error("cast: dtype 'u32' is not supported");
  }
  const n = t.length;
  if (t.dtype === dtype) {
    const buf = allocateContiguousBuffer(rt, n, dtype);
    const bytes = gpuStorageByteLength(n, dtype);
    const enc = rt.ctx.device.createCommandEncoder();
    enc.copyBufferToBuffer(t.buffer, 0, buf, 0, bytes);
    rt.ctx.queue.submit([enc.finish()]);
    return makeContiguousTensor(t.shape, buf, dtype);
  }

  const out = allocateContiguousBuffer(rt, n, dtype);
  if (t.dtype === 'f32' && dtype === 'bf16') {
    dispatchCastF32ToBf16(rt.ctx, rt.registry, t.buffer, out, n);
  } else if (t.dtype === 'bf16' && dtype === 'f32') {
    dispatchCastBf16ToF32(rt.ctx, rt.registry, t.buffer, out, n);
  } else {
    throw new Error(`cast: unsupported conversion ${t.dtype} → ${dtype}`);
  }
  return makeContiguousTensor(t.shape, out, dtype);
}

/**
 * Full-buffer GPU copy between contiguous tensors with identical **shape** and **dtype**.
 * Both buffers must support **COPY_SRC** / **COPY_DST** (standard storage allocation).
 */
export function copyContiguousTensorBytes(rt: TensorRuntime, src: Tensor, dst: Tensor): void {
  assertSameShape(src, dst);
  assertSameDtype(src, dst);
  if (!isContiguous(src) || !isContiguous(dst)) {
    throw new Error('copyContiguousTensorBytes: non-contiguous tensors not supported');
  }
  const bytes = gpuStorageByteLength(dst.length, dst.dtype);
  const enc = rt.ctx.device.createCommandEncoder();
  enc.copyBufferToBuffer(src.buffer, 0, dst.buffer, 0, bytes);
  rt.ctx.queue.submit([enc.finish()]);
}

/**
 * GPU copy of **`elemCount`** contiguous scalars (**same dtype**). Offsets are scalar indices.
 */
export function copyContiguousElems(
  rt: TensorRuntime,
  src: Tensor,
  srcElemOffset: number,
  dst: Tensor,
  dstElemOffset: number,
  elemCount: number,
): void {
  assertSameDtype(src, dst);
  if (!isContiguous(src) || !isContiguous(dst)) {
    throw new Error('copyContiguousElems: non-contiguous tensors not supported');
  }
  if (!Number.isInteger(elemCount) || elemCount < 0) {
    throw new Error(
      `copyContiguousElems: elemCount must be a non-negative integer (got ${String(elemCount)})`,
    );
  }
  if (
    !Number.isInteger(srcElemOffset) ||
    !Number.isInteger(dstElemOffset) ||
    srcElemOffset < 0 ||
    dstElemOffset < 0
  ) {
    throw new Error('copyContiguousElems: offsets must be non-negative integers');
  }
  if (srcElemOffset + elemCount > src.length || dstElemOffset + elemCount > dst.length) {
    throw new Error(
      `copyContiguousElems: range out of bounds (src ${String(srcElemOffset)}+${String(elemCount)} > ${String(
        src.length,
      )}, dst ${String(dstElemOffset)}+${String(elemCount)} > ${String(dst.length)})`,
    );
  }
  if (elemCount === 0) {
    return;
  }
  const bps = gpuStorageByteLength(1, src.dtype);
  const enc = rt.ctx.device.createCommandEncoder();
  enc.copyBufferToBuffer(
    src.buffer,
    srcElemOffset * bps,
    dst.buffer,
    dstElemOffset * bps,
    elemCount * bps,
  );
  rt.ctx.queue.submit([enc.finish()]);
}

/**
 * Pull u32 indices (**`dtype: 'u32'`**) back into a {@link Uint32Array}.
 */
export async function readIndices(rt: TensorRuntime, t: Tensor): Promise<Uint32Array> {
  assertDtype(t, 'u32');
  return readUint32Buffer(
    rt.ctx.device,
    rt.ctx.queue,
    t.buffer,
    uint32ByteLength(t.length),
    rt.ctx.bufferPool,
  );
}

/**
 * Release the GPU buffer backing this tensor.
 *
 * No internal alive-flag is tracked: subsequent {@link toArray} or op calls on
 * the destroyed tensor will surface as WebGPU validation errors. When a tensor
 * shares its buffer with another (see {@link import('./ops').reshape}),
 * destroying either invalidates both.
 */
export function destroy(t: Tensor): void {
  releasePooledOrDestroy(t.buffer);
}
