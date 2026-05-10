/* eslint-disable @typescript-eslint/no-non-null-assertion --
 * Float32Array reads are typed `number | undefined` under
 * `noUncheckedIndexedAccess`; lengths are checked before iterating.
 */
import {
  GpuContext,
  KernelRegistry,
  probeWebGpu,
  registerCoreKernels,
} from '@llm-transformer-diy/gpu';
import {
  add as tensorAdd,
  createFromArray,
  destroy,
  mul as tensorMul,
  scale as tensorScale,
  toArray,
  transpose as tensorTranspose,
} from '@llm-transformer-diy/tensor';
import type { Tensor, TensorRuntime } from '@llm-transformer-diy/tensor';
import { afterEach, beforeAll, describe, expect, it } from 'vitest';

import { add, mul, reshape, scale, transpose } from './ops';
import type { Variable } from './types';
import { createLeafVariable, createParameter } from './variable';

const webgpuAvailable = await probeWebGpu();

function expectCloseVec(actual: Float32Array, expected: Float32Array): void {
  expect(actual.length).toBe(expected.length);
  for (let i = 0; i < actual.length; i++) {
    expect(actual[i]).toBeCloseTo(expected[i]!, 5);
  }
}

describe.skipIf(!webgpuAvailable)('autograd ops (GPU-backed)', () => {
  let rt: TensorRuntime;
  const allocatedThisTest: Tensor[] = [];

  function track<T extends Tensor>(t: T): T {
    allocatedThisTest.push(t);
    return t;
  }

  function trackV(v: Variable): Variable {
    track(v.tensor);
    return v;
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

  it('add forward matches tensor.add and backward emits upstream to each input', async () => {
    const aData = new Float32Array([1, 2, 3, 4]);
    const bData = new Float32Array([10, 20, 30, 40]);
    const a = trackV(createParameter(track(createFromArray(rt, aData, [2, 2]))));
    const b = trackV(createParameter(track(createFromArray(rt, bData, [2, 2]))));

    const out = trackV(add(rt, a, b));
    expect(out.requiresGrad).toBe(true);
    expect(out.creator?.name).toBe('add');

    const expectedSum = track(tensorAdd(rt, a.tensor, b.tensor));
    expectCloseVec(await toArray(rt, out.tensor), await toArray(rt, expectedSum));

    const upstreamData = new Float32Array([100, 200, 300, 400]);
    const upstream = track(createFromArray(rt, upstreamData, [2, 2]));
    const grads = out.creator!.backward(rt, upstream);
    expect(grads).toHaveLength(2);
    // For add, both grads are the upstream itself (reference equal).
    expect(grads[0]).toBe(upstream);
    expect(grads[1]).toBe(upstream);
  });

  it('mul forward matches tensor.mul and backward emits gradOutput * other-input', async () => {
    const aData = new Float32Array([1, 2, 3, 4]);
    const bData = new Float32Array([5, 6, 7, 8]);
    const a = trackV(createParameter(track(createFromArray(rt, aData, [4]))));
    const b = trackV(createParameter(track(createFromArray(rt, bData, [4]))));

    const out = trackV(mul(rt, a, b));
    expect(out.creator?.name).toBe('mul');

    const expectedProd = track(tensorMul(rt, a.tensor, b.tensor));
    expectCloseVec(await toArray(rt, out.tensor), await toArray(rt, expectedProd));

    const upstreamData = new Float32Array([10, 11, 12, 13]);
    const upstream = track(createFromArray(rt, upstreamData, [4]));
    const grads = out.creator!.backward(rt, upstream);
    expect(grads).toHaveLength(2);

    const gradA = grads[0];
    const gradB = grads[1];
    expect(gradA).not.toBeNull();
    expect(gradB).not.toBeNull();
    track(gradA as Tensor);
    track(gradB as Tensor);

    expectCloseVec(
      await toArray(rt, gradA as Tensor),
      new Float32Array([10 * 5, 11 * 6, 12 * 7, 13 * 8]),
    );
    expectCloseVec(
      await toArray(rt, gradB as Tensor),
      new Float32Array([10 * 1, 11 * 2, 12 * 3, 13 * 4]),
    );
  });

  it('scale forward matches tensor.scale and backward emits gradOutput * k', async () => {
    const data = new Float32Array([1, 2, 3, 4, 5, 6]);
    const a = trackV(createParameter(track(createFromArray(rt, data, [2, 3]))));

    const k = 2.5;
    const out = trackV(scale(rt, a, k));
    expect(out.creator?.name).toBe('scale');

    const expectedScale = track(tensorScale(rt, a.tensor, k));
    expectCloseVec(await toArray(rt, out.tensor), await toArray(rt, expectedScale));

    const upstreamData = new Float32Array([1, 1, 1, 1, 1, 1]);
    const upstream = track(createFromArray(rt, upstreamData, [2, 3]));
    const grads = out.creator!.backward(rt, upstream);
    expect(grads).toHaveLength(1);

    const gradA = grads[0];
    expect(gradA).not.toBeNull();
    track(gradA as Tensor);
    expectCloseVec(await toArray(rt, gradA as Tensor), new Float32Array([k, k, k, k, k, k]));
  });

  it('transpose forward matches tensor.transpose and backward transposes the gradient', async () => {
    const rows = 2;
    const cols = 3;
    const data = Float32Array.from({ length: rows * cols }, (_, i) => i + 1);
    const a = trackV(createParameter(track(createFromArray(rt, data, [rows, cols]))));

    const out = trackV(transpose(rt, a));
    expect(out.tensor.shape).toEqual([cols, rows]);
    expect(out.creator?.name).toBe('transpose');

    const expectedT = track(tensorTranspose(rt, a.tensor));
    expectCloseVec(await toArray(rt, out.tensor), await toArray(rt, expectedT));

    // Upstream gradient lives in the transposed shape [cols, rows].
    const upstreamData = Float32Array.from({ length: rows * cols }, (_, i) => (i + 1) * 10);
    const upstream = track(createFromArray(rt, upstreamData, [cols, rows]));
    const grads = out.creator!.backward(rt, upstream);
    expect(grads).toHaveLength(1);

    const gradA = grads[0];
    expect(gradA).not.toBeNull();
    track(gradA as Tensor);
    expect((gradA as Tensor).shape).toEqual([rows, cols]);
    const expectedGradA = track(tensorTranspose(rt, upstream));
    expectCloseVec(await toArray(rt, gradA as Tensor), await toArray(rt, expectedGradA));
  });

  it('reshape forward shares the input buffer and backward reshapes back', async () => {
    const data = new Float32Array([1, 2, 3, 4, 5, 6]);
    const a = trackV(createParameter(track(createFromArray(rt, data, [2, 3]))));

    const out = reshape(a, [3, 2]);
    // Buffer is shared with `a.tensor.buffer`; do NOT track for separate destroy.
    expect(out.tensor.buffer).toBe(a.tensor.buffer);
    expect(out.tensor.shape).toEqual([3, 2]);
    expect(out.creator?.name).toBe('reshape');
    expectCloseVec(await toArray(rt, out.tensor), data);

    // Upstream gradient lives in the new shape [3, 2]; backward must hand back
    // a [2, 3] tensor.
    const upstreamData = new Float32Array([10, 20, 30, 40, 50, 60]);
    const upstream = track(createFromArray(rt, upstreamData, [3, 2]));
    const grads = out.creator!.backward(rt, upstream);
    expect(grads).toHaveLength(1);

    const gradA = grads[0];
    expect(gradA).not.toBeNull();
    expect((gradA as Tensor).shape).toEqual([2, 3]);
    // tensor.reshape shares the buffer of `upstream`, so destroying upstream
    // would invalidate gradA. We don't track gradA separately.
    expectCloseVec(await toArray(rt, gradA as Tensor), upstreamData);
  });

  it('does not record an Operation when no input requires grad', async () => {
    const aData = new Float32Array([1, 2, 3, 4]);
    const bData = new Float32Array([5, 6, 7, 8]);
    const a = trackV(createLeafVariable(track(createFromArray(rt, aData, [4])), false));
    const b = trackV(createLeafVariable(track(createFromArray(rt, bData, [4])), false));

    const out = trackV(add(rt, a, b));
    expect(out.requiresGrad).toBe(false);
    expect(out.creator).toBeNull();
    expect(out.parents).toHaveLength(0);

    // Forward result is still correct.
    const expectedSum = track(tensorAdd(rt, a.tensor, b.tensor));
    expectCloseVec(await toArray(rt, out.tensor), await toArray(rt, expectedSum));
  });

  it('reshape forward shares buffer regardless of the leaf-vs-recorded branch', () => {
    const data = new Float32Array([1, 2, 3, 4, 5, 6]);
    const a = trackV(createLeafVariable(track(createFromArray(rt, data, [2, 3])), false));

    const out = reshape(a, [6]);
    expect(out.requiresGrad).toBe(false);
    expect(out.creator).toBeNull();
    // Even on the leaf branch, the underlying buffer is shared (tensor.reshape
    // semantics) — destroying `a` invalidates `out`.
    expect(out.tensor.buffer).toBe(a.tensor.buffer);
  });
});
