import {
  createFloat32StorageBuffer,
  createUint32StorageBuffer,
  float32ByteLength,
  readFloat32Buffer,
  readUint32Buffer,
  uint32ByteLength,
  writeFloat32Buffer,
  writeUint32Buffer,
} from '@llm-transformer-diy/gpu';

import { assertDtype, computeContiguousStrides, productOfShape } from './shape';
import type { Tensor, TensorRuntime } from './types';

/**
 * Internal helper: build a contiguous {@link Tensor} record around an existing
 * GPU buffer. Not exported from the package index; ops and constructors share
 * this so the `Tensor` shape stays consistent.
 */
export function makeContiguousTensor(shape: readonly number[], buffer: GPUBuffer): Tensor {
  return {
    shape,
    strides: computeContiguousStrides(shape),
    dtype: 'f32',
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
 * Allocates a fresh f32 storage buffer (uninitialized) sized for `length` elements.
 * Internal helper — exported for `ops.ts` / autograd layer; not re-exported via index.
 */
export function allocateContiguousBuffer(rt: TensorRuntime, length: number): GPUBuffer {
  return createFloat32StorageBuffer(rt.ctx.device, length);
}

/**
 * Allocate a tensor of the given shape, zero-fill the underlying buffer, and
 * return the immutable record. Caller owns the buffer and must call
 * {@link destroy} when done.
 */
export function createZeros(rt: TensorRuntime, shape: readonly number[]): Tensor {
  const length = productOfShape(shape);
  const buffer = allocateContiguousBuffer(rt, length);
  // Per WebGPU spec, buffer contents are undefined post-creation; zero-fill explicitly.
  writeFloat32Buffer(rt.ctx.queue, buffer, new Float32Array(length));
  return makeContiguousTensor(shape, buffer);
}

/**
 * Allocate a tensor of the given shape and upload `data` into it. Throws when
 * `data.length` does not match the product of `shape`.
 */
export function createFromArray(
  rt: TensorRuntime,
  data: Float32Array,
  shape: readonly number[],
): Tensor {
  const length = productOfShape(shape);
  if (data.length !== length) {
    throw new Error(
      `createFromArray: data.length=${String(data.length)} does not match ` +
        `productOfShape([${shape.join(',')}])=${String(length)}`,
    );
  }
  const buffer = allocateContiguousBuffer(rt, length);
  writeFloat32Buffer(rt.ctx.queue, buffer, data);
  return makeContiguousTensor(shape, buffer);
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
  const buf = createUint32StorageBuffer(rt.ctx.device, length);
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
  assertDtype(t, 'f32');
  return readFloat32Buffer(rt.ctx.device, rt.ctx.queue, t.buffer, float32ByteLength(t.length));
}

/**
 * Pull u32 indices (**`dtype: 'u32'`**) back into a {@link Uint32Array}.
 */
export async function readIndices(rt: TensorRuntime, t: Tensor): Promise<Uint32Array> {
  assertDtype(t, 'u32');
  return readUint32Buffer(rt.ctx.device, rt.ctx.queue, t.buffer, uint32ByteLength(t.length));
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
  t.buffer.destroy();
}
