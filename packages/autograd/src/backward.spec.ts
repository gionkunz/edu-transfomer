import {
  GpuContext,
  KernelRegistry,
  probeWebGpu,
  registerCoreKernels,
} from '@llm-transformer-diy/gpu';
import { createFromArray, destroy, toArray } from '@llm-transformer-diy/tensor';
import type { Tensor, TensorRuntime } from '@llm-transformer-diy/tensor';
import { afterEach, beforeAll, describe, expect, it } from 'vitest';

import { backward } from './backward';
import { add, mul } from './ops';
import { createLeafVariable, createParameter } from './variable';

const webgpuAvailable = await probeWebGpu();

function expectCloseVec(actual: Float32Array, expected: Float32Array): void {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < actual.length; i++) {
    expect(actual[i]).toBeCloseTo(expected[i], 5);
  }
}

describe.skipIf(!webgpuAvailable)('backward (GPU-backed)', () => {
  let rt: TensorRuntime;
  const allocatedThisTest: Tensor[] = [];

  function track<T extends Tensor>(t: T): T {
    allocatedThisTest.push(t);
    return t;
  }

  beforeAll(async () => {
    const ctx = await GpuContext.create();
    const registry = new KernelRegistry();
    registerCoreKernels(ctx.device, registry);
    rt = { ctx, registry };
  });

  afterEach(() => {
    while (allocatedThisTest.length > 0) {
      const t = allocatedThisTest.pop();
      if (t) {
        try {
          destroy(t);
        } catch {
          // Buffer may already be destroyed via reshape sharing — ignore.
        }
      }
    }
  });

  it('initializes output.grad to ones when null', async () => {
    const aData = new Float32Array([1, 2, 3, 4]);
    const bData = new Float32Array([5, 6, 7, 8]);
    const a = createParameter(track(createFromArray(rt, aData, [4])));
    const b = createParameter(track(createFromArray(rt, bData, [4])));

    const out = add(rt, a, b);
    track(out.tensor);

    expect(out.grad).toBeNull();
    backward(rt, out);

    expect(out.grad).not.toBeNull();
    track(out.grad as Tensor);
    const ones = await toArray(rt, out.grad as Tensor);
    expectCloseVec(ones, new Float32Array([1, 1, 1, 1]));
  });

  it('respects a caller-supplied output.grad', async () => {
    const aData = new Float32Array([1, 2, 3, 4]);
    const bData = new Float32Array([5, 6, 7, 8]);
    const a = createParameter(track(createFromArray(rt, aData, [4])));
    const b = createParameter(track(createFromArray(rt, bData, [4])));

    const out = add(rt, a, b);
    track(out.tensor);

    const customGradData = new Float32Array([10, 20, 30, 40]);
    const customGrad = track(createFromArray(rt, customGradData, [4]));
    out.grad = customGrad;

    backward(rt, out);

    // For an `add`, each leaf's grad is value-equal to the upstream but
    // exclusively owned (the old "return the same gradOutput twice" path was
    // intentionally replaced by a clone — see `ops.ts::add` comment — so
    // `accumulateGrad` can safely free orphaned intermediates).
    expect(a.grad).not.toBeNull();
    expect(b.grad).not.toBeNull();
    expect(a.grad).not.toBe(customGrad);
    expect(b.grad).not.toBe(customGrad);
    expect(a.grad).not.toBe(b.grad);
    expectCloseVec(await toArray(rt, a.grad as Tensor), customGradData);
    expectCloseVec(await toArray(rt, b.grad as Tensor), customGradData);
  });

  it('accumulates contributions on a diamond graph (mul(a, a) ⇒ grad = 2a)', async () => {
    const aData = new Float32Array([1, 2, 3, 4]);
    const a = createParameter(track(createFromArray(rt, aData, [4])));

    // square = mul(a, a) — `a` appears twice in the operation's inputs.
    const square = mul(rt, a, a);
    track(square.tensor);

    // Seed output.grad to ones implicitly via backward.
    backward(rt, square);
    track(square.grad as Tensor);

    expect(a.grad).not.toBeNull();
    track(a.grad as Tensor);
    // d/da [a*a] = 2a element-wise; with upstream = ones, accumulated grad = 2a.
    expectCloseVec(await toArray(rt, a.grad as Tensor), new Float32Array([2, 4, 6, 8]));
  });

  it('leaves grad === null on inputs whose requiresGrad is false', async () => {
    const aData = new Float32Array([1, 2, 3, 4]);
    const bData = new Float32Array([5, 6, 7, 8]);
    const a = createParameter(track(createFromArray(rt, aData, [4])));
    const b = createLeafVariable(track(createFromArray(rt, bData, [4])), false);

    const out = mul(rt, a, b);
    track(out.tensor);

    backward(rt, out);
    track(out.grad as Tensor);

    expect(a.grad).not.toBeNull();
    track(a.grad as Tensor);
    // d/da [a*b] = b
    expectCloseVec(await toArray(rt, a.grad as Tensor), bData);

    expect(b.grad).toBeNull();
  });

  it('skips graph recording when no input requires grad — backward is a no-op', () => {
    const aData = new Float32Array([1, 2, 3, 4]);
    const bData = new Float32Array([5, 6, 7, 8]);
    const a = createLeafVariable(track(createFromArray(rt, aData, [4])), false);
    const b = createLeafVariable(track(createFromArray(rt, bData, [4])), false);

    const out = add(rt, a, b);
    track(out.tensor);

    expect(out.requiresGrad).toBe(false);
    expect(out.creator).toBeNull();

    // Calling backward on a leaf-style output must not throw and must leave
    // both inputs' grads as null.
    backward(rt, out);
    track(out.grad as Tensor); // backward still seeded out.grad to ones.

    expect(a.grad).toBeNull();
    expect(b.grad).toBeNull();
  });
});
