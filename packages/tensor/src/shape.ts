import type { DType, Tensor } from './types';

/** Float compute dtypes (`bf16` / `f32`); excludes index tensors (`u32`). */
export type FloatDType = Exclude<DType, 'u32'>;

/** Byte size of one scalar in **`dtype`** for **host** packing / logical element size (`2` for **`bf16`**, `4` for **`f32`** / **`u32`**). */
export function bytesPerElement(dtype: DType): number {
  switch (dtype) {
    case 'f32':
    case 'u32':
      return 4;
    case 'bf16':
      return 2;
    default: {
      const _exhaustive: never = dtype;
      return _exhaustive;
    }
  }
}

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
    stride *= shape[i];
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

/** Throws unless `t.dtype` is a float dtype (not **`u32`**). */
export function assertFloatDtype(t: Tensor): void {
  if (t.dtype === 'u32') {
    throw new Error(
      `assertFloatDtype: expected float dtype, got 'u32' (shape=[${t.shape.join(',')}])`,
    );
  }
}

/**
 * Throws unless **`a`** and **`b`** share the same **`dtype`** (including **`u32`**).
 * For GPU math ops, pair with **`assertFloatDtype`** on each operand when indices must be excluded.
 */
export function assertSameDtype(a: Tensor, b: Tensor): void {
  if (a.dtype !== b.dtype) {
    throw new Error(
      `assertSameDtype: dtype mismatch '${a.dtype}' vs '${b.dtype}' ` +
        `(a=[${a.shape.join(',')}], b=[${b.shape.join(',')}])`,
    );
  }
}

/** After {@link assertFloatDtype}, returns **`t.dtype`** narrowed to {@link FloatDType}. */
export function floatDtypeOf(t: Tensor): FloatDType {
  assertFloatDtype(t);
  return t.dtype as FloatDType;
}

/** Throws unless `t.dtype === expected`. */
export function assertDtype(t: Tensor, expected: DType): void {
  if (t.dtype !== expected) {
    throw new Error(
      `assertDtype: expected '${expected}', got '${t.dtype}' (shape=[${t.shape.join(',')}])`,
    );
  }
}
