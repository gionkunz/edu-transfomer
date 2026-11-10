import { createFromArray, floatDtypeOf } from '@llm-transformer-diy/tensor';
import type { TensorRuntime } from '@llm-transformer-diy/tensor';

import type { Variable } from './types';
import { accumulateGrad } from './variable';

/**
 * Post-order DFS topological sort. The returned list contains every variable
 * reachable from `output` via `parents`, with each variable placed after **all**
 * of its parents. De-duplication is keyed by `Variable.id`, so a variable
 * reached via multiple paths appears exactly once.
 *
 * Pure helper — no GPU access. Exposed primarily for testing the topo
 * invariant; callers in the package use it indirectly through {@link backward}.
 */
export function topologicalOrder(output: Variable): readonly Variable[] {
  const order: Variable[] = [];
  const visited = new Set<number>();

  function visit(v: Variable): void {
    if (visited.has(v.id)) return;
    visited.add(v.id);
    for (const parent of v.parents) {
      visit(parent);
    }
    order.push(v);
  }

  visit(output);
  return order;
}

/**
 * Run the backward pass.
 *
 * If `output.grad === null`, allocates a ones-filled tensor of shape
 * `output.tensor.shape` and assigns it; this matches the standard
 * `loss.backward()` convention (and lets the caller compare to the gradient
 * of `Σ output` per parameter element). If `output.grad` is already set, the
 * supplied gradient is used as-is.
 *
 * `seedScale` multiplies that implicit seed (default `1`). Gradient
 * accumulation uses it: calling `backward(rt, loss_i, 1 / n)` for `n`
 * micro-batches without an intervening `zeroGrad` leaves every parameter with
 * the gradient of the **mean** loss, exactly as one batch of `n` sequences
 * would. Ignored when `output.grad` is already set.
 *
 * The function then walks the topo order in reverse: for each non-leaf
 * variable, it calls `creator.backward(rt, v.grad)` and accumulates each
 * non-`null` returned tensor onto the corresponding input via
 * {@link accumulateGrad}.
 *
 * No automatic `zeroGrad` happens here — calling `backward` twice on the same
 * graph without an intervening
 * {@link import('./parameter-registry').zeroGrad} accumulates contributions
 * (this matches PyTorch's default behavior).
 */
export function backward(rt: TensorRuntime, output: Variable, seedScale = 1): void {
  if (output.grad === null) {
    const ones = new Float32Array(output.tensor.length);
    ones.fill(seedScale);
    const onesTensor = createFromArray(rt, ones, output.tensor.shape, floatDtypeOf(output.tensor));
    // Route through accumulateGrad so the direct-assignment branch keeps the
    // grep audit clean: only accumulateGrad and zeroGrad write Variable.grad.
    accumulateGrad(rt, output, onesTensor);
  }

  const order = topologicalOrder(output);

  for (let i = order.length - 1; i >= 0; i--) {
    const v = order[i];
    if (v === undefined) continue;
    if (v.creator === null) continue;
    const upstream = v.grad;
    if (upstream === null) continue;

    const grads = v.creator.backward(rt, upstream);
    const inputs = v.creator.inputs;
    if (grads.length !== inputs.length) {
      throw new Error(
        `backward: operation '${v.creator.name}' returned ${String(grads.length)} ` +
          `gradients for ${String(inputs.length)} inputs`,
      );
    }
    for (let j = 0; j < inputs.length; j++) {
      const contribution = grads[j];
      const target = inputs[j];
      if (contribution === null || contribution === undefined) continue;
      if (target === undefined) continue;
      if (!target.requiresGrad) continue;
      accumulateGrad(rt, target, contribution);
    }
  }
}
