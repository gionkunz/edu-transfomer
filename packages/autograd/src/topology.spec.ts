/// <reference types="@webgpu/types" />

import type { Tensor } from '@llm-transformer-diy/tensor';
import { describe, expect, it } from 'vitest';

import { topologicalOrder } from './backward';
import type { Operation, Variable } from './types';

/**
 * Build a fake `Tensor` literal for pure-topology tests. The `buffer` field is
 * type-erased here because the topo walk never touches GPU resources.
 */
function fakeTensor(shape: readonly number[]): Tensor {
  return {
    shape,
    strides: shape.map(() => 1),
    dtype: 'f32',
    device: 'gpu',
    buffer: {} as Tensor['buffer'],
    length: shape.reduce((p, d) => p * d, 1),
  };
}

let nextFakeId = 0;

function fakeLeaf(shape: readonly number[] = []): Variable {
  const id = nextFakeId;
  nextFakeId += 1;
  return {
    id,
    tensor: fakeTensor(shape),
    requiresGrad: true,
    creator: null,
    parents: [],
    grad: null,
  };
}

function fakeOp(name: string, parents: readonly Variable[]): Operation {
  return {
    name,
    inputs: parents,
    backward: () => parents.map(() => null),
  };
}

function fakeInternal(
  name: string,
  parents: readonly Variable[],
  shape: readonly number[] = [],
): Variable {
  const id = nextFakeId;
  nextFakeId += 1;
  const op = fakeOp(name, parents);
  return {
    id,
    tensor: fakeTensor(shape),
    requiresGrad: parents.some((p) => p.requiresGrad),
    creator: op,
    parents,
    grad: null,
  };
}

function indexOf(order: readonly Variable[], v: Variable): number {
  return order.findIndex((x) => x.id === v.id);
}

describe('topologicalOrder', () => {
  it('returns a single-element list for an isolated leaf', () => {
    const x = fakeLeaf();
    const order = topologicalOrder(x);
    expect(order).toHaveLength(1);
    expect(order[0]?.id).toBe(x.id);
  });

  it('places parents before their children on a linear chain', () => {
    const a = fakeLeaf();
    const b = fakeInternal('op1', [a]);
    const c = fakeInternal('op2', [b]);
    const order = topologicalOrder(c);
    expect(order).toHaveLength(3);
    expect(indexOf(order, a)).toBeLessThan(indexOf(order, b));
    expect(indexOf(order, b)).toBeLessThan(indexOf(order, c));
  });

  it('visits each diamond node exactly once', () => {
    // a is a single source; b and c both depend on a; d depends on both b and c.
    const a = fakeLeaf();
    const b = fakeInternal('left', [a]);
    const c = fakeInternal('right', [a]);
    const d = fakeInternal('sink', [b, c]);
    const order = topologicalOrder(d);

    expect(order).toHaveLength(4);
    const ids = order.map((v) => v.id);
    expect(new Set(ids).size).toBe(4);

    expect(indexOf(order, a)).toBeLessThan(indexOf(order, b));
    expect(indexOf(order, a)).toBeLessThan(indexOf(order, c));
    expect(indexOf(order, b)).toBeLessThan(indexOf(order, d));
    expect(indexOf(order, c)).toBeLessThan(indexOf(order, d));
  });

  it('de-duplicates a node referenced by multiple parents', () => {
    // a self-feeds an internal node twice (e.g. mul(a, a)); the topo walk must
    // still emit `a` exactly once.
    const a = fakeLeaf();
    const square = fakeInternal('mul', [a, a]);
    const order = topologicalOrder(square);
    expect(order).toHaveLength(2);
    expect(indexOf(order, a)).toBe(0);
    expect(indexOf(order, square)).toBe(1);
  });

  it('places every variable after all of its parents in a deeper graph', () => {
    const x = fakeLeaf();
    const y = fakeLeaf();
    const z = fakeLeaf();
    const xy = fakeInternal('add', [x, y]);
    const xyz = fakeInternal('add', [xy, z]);
    const head = fakeInternal('scale', [xyz]);

    const order = topologicalOrder(head);
    expect(order).toHaveLength(6);

    for (const v of order) {
      const childIdx = indexOf(order, v);
      for (const parent of v.parents) {
        expect(indexOf(order, parent)).toBeLessThan(childIdx);
      }
    }
  });
});
