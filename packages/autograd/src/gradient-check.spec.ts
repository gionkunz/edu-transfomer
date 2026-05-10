/* eslint-disable @typescript-eslint/no-non-null-assertion --
 * After a successful backward call, parameter `.grad` fields are non-null by
 * contract; non-null assertions are checked first via expect().not.toBeNull().
 */
import {
  GpuContext,
  KernelRegistry,
  probeWebGpu,
  registerCoreKernels,
} from '@llm-transformer-diy/gpu';
import { createFromArray, destroy, toArray } from '@llm-transformer-diy/tensor';
import type { Tensor, TensorRuntime } from '@llm-transformer-diy/tensor';
import { beforeAll, describe, expect, it } from 'vitest';

import { backward } from './backward';
import { add, gelu, matmul, mul, reshape, scale } from './ops';
import { createParameter } from './variable';

const webgpuAvailable = await probeWebGpu();

interface ForwardResult {
  readonly output: Float32Array;
  readonly grads: {
    readonly a: Float32Array;
    readonly b: Float32Array;
    readonly c: Float32Array;
  } | null;
}

/**
 * Run the documented composed graph once and return the readback of the output
 * (always) and of each parameter's analytic gradient (when `computeGrads` is
 * true). Allocates and destroys all GPU buffers it owns; the returned
 * Float32Arrays live entirely on the host.
 *
 * Composition (every op is in the trivially-derivable scope of `autograd-bootstrap`):
 *   s = add(rt, a, b)            // [2,3]
 *   m = mul(rt, s, c)            // [2,3]
 *   r = reshape(m, [6])          // shares m.buffer
 *   output = scale(rt, r, 2.0)   // [6]
 *
 * The "loss" is implicitly `Σ output`; backward seeds `output.grad = ones`,
 * so each `param.grad` matches `∂(Σ output)/∂param`.
 */
async function runComposedGraph(
  rt: TensorRuntime,
  aData: Float32Array,
  bData: Float32Array,
  cData: Float32Array,
  computeGrads: boolean,
): Promise<ForwardResult> {
  const shape = [2, 3] as const;

  const aT = createFromArray(rt, aData, shape);
  const bT = createFromArray(rt, bData, shape);
  const cT = createFromArray(rt, cData, shape);

  const a = createParameter(aT);
  const b = createParameter(bT);
  const c = createParameter(cT);

  const s = add(rt, a, b);
  const m = mul(rt, s, c);
  // r shares m.tensor.buffer (reshape is metadata-only).
  const r = reshape(m, [6]);
  const out = scale(rt, r, 2.0);

  const outputArray = await toArray(rt, out.tensor);

  let grads: ForwardResult['grads'] = null;
  if (computeGrads) {
    backward(rt, out);
    const gradA = await toArray(rt, a.grad as Tensor);
    const gradB = await toArray(rt, b.grad as Tensor);
    const gradC = await toArray(rt, c.grad as Tensor);
    grads = { a: gradA, b: gradB, c: gradC };
  }

  // Destroy uniquely-owned GPU buffers. `r.tensor.buffer === m.tensor.buffer`
  // (reshape sharing), so we destroy `m` only.
  // Param tensors (a/b/c) are owned by the createFromArray call above; the
  // Variable wraps them but does not duplicate the buffer.
  destroy(aT);
  destroy(bT);
  destroy(cT);
  destroy(s.tensor);
  destroy(m.tensor);
  destroy(out.tensor);

  if (computeGrads) {
    // out.grad is a fresh ones tensor allocated by backward.
    if (out.grad !== null) destroy(out.grad);
    // For `add`, accumulateGrad's direct-assign branch makes a.grad and b.grad
    // share the same buffer (the `gradS` tensor produced by mul backward).
    // Destroy it once via a.grad.
    if (a.grad !== null) destroy(a.grad);
    // c.grad is the other mul-backward tensor.
    if (c.grad !== null) destroy(c.grad);
    // The reshape-backward tensor shares the buffer of the scale-backward
    // tensor; we have no direct handle to destroy it. It is reclaimed when its
    // GPUBuffer is garbage-collected; for a 6-element f32 buffer this is
    // negligible memory churn.
  }

  return { output: outputArray, grads };
}

function sum(arr: Float32Array): number {
  let total = 0;
  for (let i = 0; i < arr.length; i++) {
    total += arr[i] as number;
  }
  return total;
}

describe.skipIf(!webgpuAvailable)('finite-difference gradient check (GPU-backed)', () => {
  let rt: TensorRuntime;

  beforeAll(async () => {
    const ctx = await GpuContext.create();
    const registry = new KernelRegistry();
    registerCoreKernels(ctx.device, registry);
    rt = { ctx, registry };
  });

  it('analytic gradients on a tiny composed graph match central-difference numerical gradients', async () => {
    // Small but non-trivial parameter values; chosen to avoid zeros so element-
    // wise products do not collapse cases that would mask sign errors.
    const aBase = new Float32Array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
    const bBase = new Float32Array([0.7, -0.4, 0.2, -0.1, 0.3, -0.5]);
    const cBase = new Float32Array([1.5, -0.5, 2.0, 0.5, -1.0, 0.25]);

    const { grads } = await runComposedGraph(rt, aBase, bBase, cBase, true);
    expect(grads).not.toBeNull();

    const epsilon = 1e-3;
    const tolerance = 5e-3;

    type ParamSlot = { readonly name: 'a' | 'b' | 'c'; readonly base: Float32Array };
    const slots: readonly ParamSlot[] = [
      { name: 'a', base: aBase },
      { name: 'b', base: bBase },
      { name: 'c', base: cBase },
    ];

    for (const slot of slots) {
      const analytic = grads![slot.name];
      for (let i = 0; i < slot.base.length; i++) {
        // Central difference: f(x + ε) - f(x - ε) over 2ε.
        const plus = new Float32Array(slot.base);
        plus[i] = (slot.base[i] as number) + epsilon;
        const minus = new Float32Array(slot.base);
        minus[i] = (slot.base[i] as number) - epsilon;

        const aP = slot.name === 'a' ? plus : aBase;
        const bP = slot.name === 'b' ? plus : bBase;
        const cP = slot.name === 'c' ? plus : cBase;

        const aM = slot.name === 'a' ? minus : aBase;
        const bM = slot.name === 'b' ? minus : bBase;
        const cM = slot.name === 'c' ? minus : cBase;

        const fPlus = sum((await runComposedGraph(rt, aP, bP, cP, false)).output);
        const fMinus = sum((await runComposedGraph(rt, aM, bM, cM, false)).output);
        const numerical = (fPlus - fMinus) / (2 * epsilon);

        expect(
          Math.abs((analytic[i] as number) - numerical),
          `${slot.name}[${String(i)}]: analytic=${String(analytic[i])} numerical=${String(numerical)}`,
        ).toBeLessThanOrEqual(tolerance);
      }
    }
  }, 30000);

  it('matmul gradients on a non-square composed graph match central-difference numerical gradients', async () => {
    const xBase = new Float32Array([0.1, -0.3, 0.5, 0.2, 0.4, -0.6]);
    const wBase = new Float32Array([
      0.7, -0.2, 0.3, 0.1, -0.4, 0.5, -0.1, 0.8, 0.6, -0.3, 0.2, 0.9,
    ]);

    interface MatmulResult {
      readonly output: Float32Array;
      readonly grads: { readonly x: Float32Array; readonly w: Float32Array } | null;
    }

    async function runMatmulGraph(
      rtInner: TensorRuntime,
      xData: Float32Array,
      wData: Float32Array,
      computeGrads: boolean,
    ): Promise<MatmulResult> {
      const xT = createFromArray(rtInner, xData, [2, 3]);
      const wT = createFromArray(rtInner, wData, [3, 4]);

      const x = createParameter(xT);
      const w = createParameter(wT);

      const product = matmul(rtInner, x, w);
      const flat = reshape(product, [8]);
      const out = scale(rtInner, flat, 2.0);

      const outputArray = await toArray(rtInner, out.tensor);

      let grads: MatmulResult['grads'] = null;
      if (computeGrads) {
        backward(rtInner, out);
        const gradX = await toArray(rtInner, x.grad as Tensor);
        const gradW = await toArray(rtInner, w.grad as Tensor);
        grads = { x: gradX, w: gradW };
      }

      destroy(xT);
      destroy(wT);
      destroy(product.tensor);
      destroy(out.tensor);

      if (computeGrads) {
        if (out.grad !== null) destroy(out.grad);
        if (x.grad !== null) destroy(x.grad);
        if (w.grad !== null) destroy(w.grad);
      }

      return { output: outputArray, grads };
    }

    const { grads } = await runMatmulGraph(rt, xBase, wBase, true);
    expect(grads).not.toBeNull();

    const epsilon = 1e-3;
    const tolerance = 5e-3;

    type ParamSlot = { readonly name: 'x' | 'w'; readonly base: Float32Array };
    const slots: readonly ParamSlot[] = [
      { name: 'x', base: xBase },
      { name: 'w', base: wBase },
    ];

    for (const slot of slots) {
      const analytic = grads![slot.name];
      for (let i = 0; i < slot.base.length; i++) {
        const plus = new Float32Array(slot.base);
        plus[i] = (slot.base[i] as number) + epsilon;
        const minus = new Float32Array(slot.base);
        minus[i] = (slot.base[i] as number) - epsilon;

        const xP = slot.name === 'x' ? plus : xBase;
        const wP = slot.name === 'w' ? plus : wBase;
        const xM = slot.name === 'x' ? minus : xBase;
        const wM = slot.name === 'w' ? minus : wBase;

        const fPlus = sum((await runMatmulGraph(rt, xP, wP, false)).output);
        const fMinus = sum((await runMatmulGraph(rt, xM, wM, false)).output);
        const numerical = (fPlus - fMinus) / (2 * epsilon);

        expect(
          Math.abs((analytic[i] as number) - numerical),
          `${slot.name}[${String(i)}]: analytic=${String(analytic[i])} numerical=${String(numerical)}`,
        ).toBeLessThanOrEqual(tolerance);
      }
    }
  }, 30000);

  it('gelu in a composed graph: analytic grads match central-difference', async () => {
    const aBase = new Float32Array([0.15, -0.4, 0.7, -0.2, 0.55, -0.35]);
    const bBase = new Float32Array([0.2, 0.3, -0.5, 0.1, -0.25, 0.45]);

    interface GeluGraphResult {
      readonly output: Float32Array;
      readonly grads: { readonly a: Float32Array; readonly b: Float32Array } | null;
    }

    async function runGeluGraph(
      rtInner: TensorRuntime,
      aData: Float32Array,
      bData: Float32Array,
      computeGrads: boolean,
    ): Promise<GeluGraphResult> {
      const shape = [2, 3] as const;
      const aT = createFromArray(rtInner, aData, shape);
      const bT = createFromArray(rtInner, bData, shape);

      const a = createParameter(aT);
      const b = createParameter(bT);

      const s = add(rtInner, a, b);
      const g = gelu(rtInner, s);
      const out = scale(rtInner, g, 2.0);

      const outputArray = await toArray(rtInner, out.tensor);

      let grads: GeluGraphResult['grads'] = null;
      if (computeGrads) {
        backward(rtInner, out);
        const gradA = await toArray(rtInner, a.grad as Tensor);
        const gradB = await toArray(rtInner, b.grad as Tensor);
        grads = { a: gradA, b: gradB };
      }

      destroy(aT);
      destroy(bT);
      destroy(s.tensor);
      destroy(g.tensor);
      destroy(out.tensor);

      if (computeGrads) {
        if (out.grad !== null) destroy(out.grad);
        if (a.grad !== null) destroy(a.grad);
        if (b.grad !== null) destroy(b.grad);
      }

      return { output: outputArray, grads };
    }

    const { grads } = await runGeluGraph(rt, aBase, bBase, true);
    expect(grads).not.toBeNull();

    const epsilon = 1e-3;
    const tolerance = 5e-3;

    for (const slot of [
      { name: 'a' as const, base: aBase },
      { name: 'b' as const, base: bBase },
    ]) {
      const analytic = grads![slot.name];
      for (let i = 0; i < slot.base.length; i++) {
        const plus = new Float32Array(slot.base);
        plus[i] = (slot.base[i] as number) + epsilon;
        const minus = new Float32Array(slot.base);
        minus[i] = (slot.base[i] as number) - epsilon;

        const aP = slot.name === 'a' ? plus : aBase;
        const bP = slot.name === 'b' ? plus : bBase;
        const aM = slot.name === 'a' ? minus : aBase;
        const bM = slot.name === 'b' ? minus : bBase;

        const fPlus = sum((await runGeluGraph(rt, aP, bP, false)).output);
        const fMinus = sum((await runGeluGraph(rt, aM, bM, false)).output);
        const numerical = (fPlus - fMinus) / (2 * epsilon);

        expect(
          Math.abs((analytic[i] as number) - numerical),
          `${slot.name}[${String(i)}]: analytic=${String(analytic[i])} numerical=${String(numerical)}`,
        ).toBeLessThanOrEqual(tolerance);
      }
    }
  }, 30000);
});
