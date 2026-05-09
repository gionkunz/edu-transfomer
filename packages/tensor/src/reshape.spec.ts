import { describe, expect, it } from 'vitest';

import { reshape } from './ops';
import { computeContiguousStrides, productOfShape } from './shape';
import type { Tensor } from './types';

/** Fake tensor for pure metadata tests; `buffer` is a sentinel object reused across cases. */
function fakeTensor(
  shape: readonly number[],
  bufferSentinel: object,
  strides?: readonly number[],
): Tensor {
  return {
    shape,
    strides: strides ?? computeContiguousStrides(shape),
    dtype: 'f32',
    device: 'gpu',
    buffer: bufferSentinel as Tensor['buffer'],
    length: productOfShape(shape),
  };
}

describe('reshape (pure metadata)', () => {
  it('returns a new Tensor sharing the input buffer when shapes are compatible', () => {
    const buffer = {};
    const a = fakeTensor([2, 3], buffer);
    const b = reshape(a, [3, 2]);

    expect(b.shape).toEqual([3, 2]);
    expect(b.strides).toEqual([2, 1]);
    expect(b.length).toBe(6);
    expect(b.dtype).toBe('f32');
    expect(b.device).toBe('gpu');
    expect(b.buffer).toBe(a.buffer);
  });

  it('does not mutate the input tensor', () => {
    const a = fakeTensor([6], {});
    const beforeShape = a.shape;
    const beforeStrides = a.strides;

    reshape(a, [2, 3]);

    expect(a.shape).toBe(beforeShape);
    expect(a.strides).toBe(beforeStrides);
  });

  it('supports the scalar shape on either side when lengths agree', () => {
    const a = fakeTensor([1], {});
    const b = reshape(a, []);
    expect(b.shape).toEqual([]);
    expect(b.strides).toEqual([]);
    expect(b.length).toBe(1);
  });

  it('throws when the element count does not match', () => {
    const a = fakeTensor([2, 3], {});
    expect(() => reshape(a, [2, 4])).toThrow(/element count mismatch/);
  });

  it('throws when the input has non-contiguous strides', () => {
    // Build a tensor whose strides do not match the row-major contiguous layout.
    const a = fakeTensor([2, 3], {}, [1, 2]);
    expect(() => reshape(a, [3, 2])).toThrow(/non-contiguous strides not supported/);
  });
});
