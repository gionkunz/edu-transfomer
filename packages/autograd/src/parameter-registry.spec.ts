/// <reference types="@webgpu/types" />

import type { Tensor } from '@llm-transformer-diy/tensor';
import { describe, expect, it } from 'vitest';

import {
  createParameterRegistry,
  parameters,
  registerParameter,
  totalParams,
  zeroGrad,
} from './parameter-registry';
import type { Parameter } from './types';

function fakeTensor(length: number): Tensor {
  return {
    shape: [length],
    strides: [1],
    dtype: 'f32',
    device: 'gpu',
    buffer: {} as Tensor['buffer'],
    length,
  };
}

let nextFakeId = 0;

function fakeParameter(length: number, grad: Tensor | null = null): Parameter {
  const id = nextFakeId;
  nextFakeId += 1;
  return {
    id,
    tensor: fakeTensor(length),
    requiresGrad: true,
    creator: null,
    parents: [],
    grad,
  };
}

describe('parameter registry helpers (pure)', () => {
  it('createParameterRegistry returns an empty registry', () => {
    const reg = createParameterRegistry();
    expect(parameters(reg)).toHaveLength(0);
    expect(totalParams(reg)).toBe(0);
  });

  it('registerParameter preserves registration order', () => {
    const reg = createParameterRegistry();
    const a = fakeParameter(4);
    const b = fakeParameter(8);
    const c = fakeParameter(2);
    registerParameter(reg, a);
    registerParameter(reg, b);
    registerParameter(reg, c);

    const snap = parameters(reg);
    expect(snap.map((p) => p.id)).toEqual([a.id, b.id, c.id]);
  });

  it('parameters returns a defensive copy', () => {
    const reg = createParameterRegistry();
    const a = fakeParameter(3);
    registerParameter(reg, a);

    const snap = parameters(reg);
    expect(snap).toHaveLength(1);

    // Registering after the snapshot must not retroactively affect it.
    registerParameter(reg, fakeParameter(5));
    expect(snap).toHaveLength(1);
    expect(parameters(reg)).toHaveLength(2);
  });

  it('zeroGrad clears every registered parameter grad', () => {
    const reg = createParameterRegistry();
    const a = fakeParameter(3, fakeTensor(3));
    const b = fakeParameter(7, fakeTensor(7));
    registerParameter(reg, a);
    registerParameter(reg, b);

    expect(a.grad).not.toBeNull();
    expect(b.grad).not.toBeNull();

    zeroGrad(reg);

    expect(a.grad).toBeNull();
    expect(b.grad).toBeNull();
  });

  it('totalParams sums tensor lengths across registrations', () => {
    const reg = createParameterRegistry();
    expect(totalParams(reg)).toBe(0);

    registerParameter(reg, fakeParameter(4));
    registerParameter(reg, fakeParameter(8));
    registerParameter(reg, fakeParameter(15));

    expect(totalParams(reg)).toBe(27);
  });

  it('does not deduplicate when the same parameter is registered twice', () => {
    const reg = createParameterRegistry();
    const a = fakeParameter(5);
    registerParameter(reg, a);
    registerParameter(reg, a);

    expect(parameters(reg)).toHaveLength(2);
    // totalParams reflects both registrations because we documented no dedup.
    expect(totalParams(reg)).toBe(10);
  });
});
