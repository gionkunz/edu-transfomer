/* eslint-disable @typescript-eslint/no-non-null-assertion --
 * Index reads on `t.shape` are guarded by `assertRank` immediately above each use,
 * so the non-null assertions are safe under `noUncheckedIndexedAccess`.
 */

import {
  dispatchAdd,
  dispatchGelu,
  dispatchGeluBackward,
  dispatchMatMul,
  dispatchMul,
  dispatchScale,
  dispatchTranspose,
} from '@llm-transformer-diy/gpu';

import {
  assertRank,
  assertSameShape,
  computeContiguousStrides,
  isContiguous,
  productOfShape,
} from './shape';
import { allocateContiguousBuffer, makeContiguousTensor } from './tensor';
import type { Tensor, TensorRuntime } from './types';

/**
 * Matrix multiply for **rank-2** row-major tensors: `a` is `[M, K]`, `b` is
 * `[K, N]`, output is `[M, N]`. Throws on rank or inner-dim mismatch.
 */
export function matmul(rt: TensorRuntime, a: Tensor, b: Tensor): Tensor {
  assertRank(a, 2);
  assertRank(b, 2);
  const M = a.shape[0]!;
  const K = a.shape[1]!;
  const N = b.shape[1]!;
  if (b.shape[0] !== K) {
    throw new Error(
      `matmul: inner dimensions must match (a=[${a.shape.join(',')}], b=[${b.shape.join(',')}])`,
    );
  }
  const out = allocateContiguousBuffer(rt, M * N);
  dispatchMatMul(rt.ctx, rt.registry, a.buffer, b.buffer, out, M, N, K);
  return makeContiguousTensor([M, N], out);
}

/** Elementwise `out[i] = a[i] + b[i]`. Requires equal shapes (no broadcasting). */
export function add(rt: TensorRuntime, a: Tensor, b: Tensor): Tensor {
  assertSameShape(a, b);
  const out = allocateContiguousBuffer(rt, a.length);
  dispatchAdd(rt.ctx, rt.registry, a.buffer, b.buffer, out, a.length);
  return makeContiguousTensor(a.shape, out);
}

/** Elementwise `out[i] = a[i] * b[i]`. Requires equal shapes (no broadcasting). */
export function mul(rt: TensorRuntime, a: Tensor, b: Tensor): Tensor {
  assertSameShape(a, b);
  const out = allocateContiguousBuffer(rt, a.length);
  dispatchMul(rt.ctx, rt.registry, a.buffer, b.buffer, out, a.length);
  return makeContiguousTensor(a.shape, out);
}

/** Elementwise `out[i] = a[i] * k`. */
export function scale(rt: TensorRuntime, a: Tensor, k: number): Tensor {
  const out = allocateContiguousBuffer(rt, a.length);
  dispatchScale(rt.ctx, rt.registry, a.buffer, out, a.length, k);
  return makeContiguousTensor(a.shape, out);
}

/** Elementwise GELU activation (tanh formulation; matches `geluScalar` in `@llm-transformer-diy/gpu`). */
export function gelu(rt: TensorRuntime, a: Tensor): Tensor {
  const out = allocateContiguousBuffer(rt, a.length);
  dispatchGelu(rt.ctx, rt.registry, a.buffer, out, a.length);
  return makeContiguousTensor(a.shape, out);
}

/**
 * GELU backward: `grad_in[i] = grad_out[i] * gelu′(input[i])`. Shapes must
 * match elementwise (no broadcasting).
 */
export function geluBackward(rt: TensorRuntime, gradOutput: Tensor, input: Tensor): Tensor {
  assertSameShape(gradOutput, input);
  const out = allocateContiguousBuffer(rt, input.length);
  dispatchGeluBackward(rt.ctx, rt.registry, gradOutput.buffer, input.buffer, out, input.length);
  return makeContiguousTensor(input.shape, out);
}

/**
 * Transpose a **rank-2** row-major tensor. Input `[rows, cols]` → output
 * `[cols, rows]`. Throws on rank ≠ 2.
 */
export function transpose(rt: TensorRuntime, a: Tensor): Tensor {
  assertRank(a, 2);
  const rows = a.shape[0]!;
  const cols = a.shape[1]!;
  const out = allocateContiguousBuffer(rt, rows * cols);
  dispatchTranspose(rt.ctx, rt.registry, a.buffer, out, rows, cols);
  return makeContiguousTensor([cols, rows], out);
}

/**
 * Pure metadata reshape: the returned tensor **shares** `a.buffer`. No GPU
 * runtime needed — this neither allocates nor dispatches.
 *
 * Throws when the new shape's product differs from `a.length` or when `a` has
 * non-contiguous strides (views beyond reshape are out of scope for
 * `tensor-bootstrap`). Because the buffer is shared, calling
 * {@link import('./tensor').destroy} on either tensor invalidates both.
 */
export function reshape(a: Tensor, newShape: readonly number[]): Tensor {
  const newLength = productOfShape(newShape);
  if (newLength !== a.length) {
    throw new Error(
      `reshape: element count mismatch ` +
        `(current=${String(a.length)} from [${a.shape.join(',')}], ` +
        `requested=${String(newLength)} from [${newShape.join(',')}])`,
    );
  }
  if (!isContiguous(a)) {
    throw new Error('reshape: non-contiguous strides not supported in tensor-bootstrap');
  }
  return {
    shape: newShape,
    strides: computeContiguousStrides(newShape),
    dtype: a.dtype,
    device: a.device,
    buffer: a.buffer,
    length: a.length,
  };
}
