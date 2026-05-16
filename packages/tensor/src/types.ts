/// <reference types="@webgpu/types" />

import type { GpuContext, KernelRegistry } from '@llm-transformer-diy/gpu';

/** Scalar element dtypes for **`Tensor`** GPU storage (`bf16` / `fp16` use 16-bit packed buffers). */
export type DType = 'f32' | 'bf16' | 'fp16' | 'u32';

/** Single supported storage backend for now — see `tensor-bootstrap` proposal. */
export type Device = 'gpu';

/**
 * Immutable view of a GPU-resident tensor.
 *
 * Owned by the caller: lifecycle is managed via {@link import('./tensor').destroy}
 * (which calls `buffer.destroy()`). Op functions never mutate `Tensor` records or
 * their backing buffers; each op allocates a fresh output buffer.
 *
 * `strides` is row-major and contiguous in the current scope; non-contiguous
 * views are deferred to a future change.
 */
export interface Tensor {
  readonly shape: readonly number[];
  readonly strides: readonly number[];
  readonly dtype: DType;
  readonly device: Device;
  /** GPU storage buffer; treat as opaque outside the tensor / autograd layers. */
  readonly buffer: GPUBuffer;
  /** Total element count (`productOfShape(shape)`) cached for hot paths. */
  readonly length: number;
}

/**
 * Threaded explicitly into every constructor / op that allocates or dispatches.
 *
 * No module-level default exists — callers (tests, future autograd / training
 * code) hold a single `TensorRuntime` and pass it as the first argument.
 */
export interface TensorRuntime {
  readonly ctx: GpuContext;
  readonly registry: KernelRegistry;
}
