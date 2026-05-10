/* eslint-disable @typescript-eslint/no-non-null-assertion --
 * Indices are bounded by the freshly-allocated arrays' known lengths under
 * `noUncheckedIndexedAccess`, so the assertions are safe.
 */

import type { DType, Tensor } from './types';

/**
 * Row-major contiguous strides for `shape`. For `shape = [a, b, c]` returns
 * `[b*c, c, 1]`. Returns `[]` for the scalar shape `[]`.
 */
export function computeContiguousStrides(shape: readonly number[]): readonly number[] {
  const n = shape.length;
  if (n === 0) return [];
  const strides = new Array<number>(n);
  let stride = 1;
  for (let i = n - 1; i >= 0; i--) {
    strides[i] = stride;
    stride *= shape[i]!;
  }
  return strides;
}

/**
 * Product of dimensions; returns `1` for the scalar shape `[]`. Throws on any
 * negative dimension; zero dimensions are allowed (length will be `0`).
 */
export function productOfShape(shape: readonly number[]): number {
  let product = 1;
  for (const dim of shape) {
    if (dim < 0) {
      throw new Error(`productOfShape: negative dimension ${String(dim)} in [${shape.join(',')}]`);
    }
    product *= dim;
  }
  return product;
}

/** True when `t.strides` equals `computeContiguousStrides(t.shape)`. */
export function isContiguous(t: Tensor): boolean {
  const expected = computeContiguousStrides(t.shape);
  if (t.strides.length !== expected.length) return false;
  for (let i = 0; i < expected.length; i++) {
    if (t.strides[i] !== expected[i]) return false;
  }
  return true;
}

/** True when `a.shape` and `b.shape` are element-wise equal. */
export function sameShape(a: Tensor, b: Tensor): boolean {
  if (a.shape.length !== b.shape.length) return false;
  for (let i = 0; i < a.shape.length; i++) {
    if (a.shape[i] !== b.shape[i]) return false;
  }
  return true;
}

/** Throws unless `t.shape.length === expectedRank`. */
export function assertRank(t: Tensor, expectedRank: number): void {
  if (t.shape.length !== expectedRank) {
    throw new Error(
      `assertRank: expected rank ${String(expectedRank)}, got rank ${String(t.shape.length)} ` +
        `(shape=[${t.shape.join(',')}])`,
    );
  }
}

/** Throws unless {@link sameShape} holds. */
export function assertSameShape(a: Tensor, b: Tensor): void {
  if (!sameShape(a, b)) {
    throw new Error(
      `assertSameShape: shapes differ (a=[${a.shape.join(',')}], b=[${b.shape.join(',')}])`,
    );
  }
}

/** Throws unless `t.dtype === expected`. */
export function assertDtype(t: Tensor, expected: DType): void {
  if (t.dtype !== expected) {
    throw new Error(
      `assertDtype: expected '${expected}', got '${t.dtype}' (shape=[${t.shape.join(',')}])`,
    );
  }
}
