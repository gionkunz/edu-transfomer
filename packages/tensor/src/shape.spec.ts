import { describe, expect, it } from 'vitest';

import {
  assertFloatDtype,
  assertRank,
  assertSameDtype,
  assertSameShape,
  computeContiguousStrides,
  isContiguous,
  productOfShape,
  sameShape,
} from './shape';
import type { Tensor } from './types';

/**
 * Build a fake `Tensor` literal for pure-shape tests. The `buffer` field is
 * type-erased here because shape helpers never touch GPU resources.
 */
function fakeTensor(shape: readonly number[], strides?: readonly number[]): Tensor {
  return {
    shape,
    strides: strides ?? computeContiguousStrides(shape),
    dtype: 'f32',
    device: 'gpu',
    buffer: {} as Tensor['buffer'],
    length: productOfShape(shape),
  };
}

describe('computeContiguousStrides', () => {
  it('returns row-major strides for rank-3 shapes', () => {
    expect(computeContiguousStrides([2, 3, 4])).toEqual([12, 4, 1]);
  });

  it('returns [1] for rank-1 shapes', () => {
    expect(computeContiguousStrides([7])).toEqual([1]);
  });

  it('returns [] for the scalar shape', () => {
    expect(computeContiguousStrides([])).toEqual([]);
  });
});

describe('productOfShape', () => {
  it('multiplies all dimensions', () => {
    expect(productOfShape([2, 3, 4])).toBe(24);
  });

  it('returns 1 for the scalar shape', () => {
    expect(productOfShape([])).toBe(1);
  });

  it('returns 0 when any dimension is 0', () => {
    expect(productOfShape([3, 0, 5])).toBe(0);
  });

  it('throws on negative dimensions', () => {
    expect(() => productOfShape([2, -1, 4])).toThrow(/negative dimension/);
  });
});

describe('isContiguous', () => {
  it('is true for tensors built with computeContiguousStrides', () => {
    expect(isContiguous(fakeTensor([2, 3, 4]))).toBe(true);
  });

  it('is false when strides do not match the contiguous layout', () => {
    expect(isContiguous(fakeTensor([2, 3], [1, 2]))).toBe(false);
  });
});

describe('sameShape', () => {
  it('is true for identical shapes', () => {
    expect(sameShape(fakeTensor([2, 3]), fakeTensor([2, 3]))).toBe(true);
  });

  it('is false when ranks differ', () => {
    expect(sameShape(fakeTensor([6]), fakeTensor([2, 3]))).toBe(false);
  });

  it('is false when any dimension differs', () => {
    expect(sameShape(fakeTensor([2, 3]), fakeTensor([2, 4]))).toBe(false);
  });
});

describe('assertRank', () => {
  it('passes when the rank matches', () => {
    expect(() => {
      assertRank(fakeTensor([2, 3]), 2);
    }).not.toThrow();
  });

  it('throws with a descriptive error when ranks differ', () => {
    expect(() => {
      assertRank(fakeTensor([2, 3]), 3);
    }).toThrow(/expected rank 3.*got rank 2/);
  });
});

describe('assertSameShape', () => {
  it('passes for identical shapes', () => {
    expect(() => {
      assertSameShape(fakeTensor([2, 3]), fakeTensor([2, 3]));
    }).not.toThrow();
  });

  it('throws with both shapes when they differ', () => {
    expect(() => {
      assertSameShape(fakeTensor([2, 3]), fakeTensor([2, 4]));
    }).toThrow(/a=\[2,3\].*b=\[2,4\]/);
  });
});

function fakeTensorWithDtype(shape: readonly number[], dtype: Tensor['dtype']): Tensor {
  return {
    shape,
    strides: computeContiguousStrides(shape),
    dtype,
    device: 'gpu',
    buffer: {} as Tensor['buffer'],
    length: productOfShape(shape),
  };
}

describe('assertFloatDtype', () => {
  it('throws on u32', () => {
    expect(() => {
      assertFloatDtype(fakeTensorWithDtype([3], 'u32'));
    }).toThrow(/float dtype/);
  });

  it('accepts bf16', () => {
    expect(() => {
      assertFloatDtype(fakeTensorWithDtype([2], 'bf16'));
    }).not.toThrow();
  });
});

describe('assertSameDtype', () => {
  it('throws when dtypes differ', () => {
    expect(() => {
      assertSameDtype(fakeTensorWithDtype([2], 'f32'), fakeTensorWithDtype([2], 'bf16'));
    }).toThrow(/dtype mismatch/);
  });
});
