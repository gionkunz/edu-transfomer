import {
  add as tensorAdd,
  mul as tensorMul,
  reshape as tensorReshape,
  scale as tensorScale,
  transpose as tensorTranspose,
} from '@llm-transformer-diy/tensor';
import type { Tensor, TensorRuntime } from '@llm-transformer-diy/tensor';

import type { Operation, Variable } from './types';
import { createInternalVariable, createLeafVariable } from './variable';

/**
 * Helper: wrap a forward `Tensor` as a `Variable`. When at least one input
 * requires grad, build the recorded-graph `Variable` with a creator pointing
 * at `buildOp()`. Otherwise return a leaf-style `Variable` so read-only
 * inference paths do not record a graph node.
 */
function liftOp(
  forwardOutput: Tensor,
  inputs: readonly Variable[],
  buildOp: () => Operation,
): Variable {
  const requiresGrad = inputs.some((v) => v.requiresGrad);
  if (!requiresGrad) {
    return createLeafVariable(forwardOutput, false);
  }
  return createInternalVariable(forwardOutput, buildOp(), true);
}

/**
 * Elementwise `out = a + b` with autograd. Forward routes to `tensor.add`;
 * backward emits the upstream gradient unchanged to each input that requires
 * grad (∂(a+b)/∂a = 1, ∂(a+b)/∂b = 1).
 */
export function add(rt: TensorRuntime, a: Variable, b: Variable): Variable {
  const out = tensorAdd(rt, a.tensor, b.tensor);
  return liftOp(out, [a, b], () => ({
    name: 'add',
    inputs: [a, b],
    backward: (_rt, gradOutput) => [
      a.requiresGrad ? gradOutput : null,
      b.requiresGrad ? gradOutput : null,
    ],
  }));
}

/**
 * Elementwise `out = a * b` with autograd. Forward routes to `tensor.mul`;
 * backward captures the input tensors and emits `gradOutput * other-input`
 * for each input that requires grad.
 */
export function mul(rt: TensorRuntime, a: Variable, b: Variable): Variable {
  const out = tensorMul(rt, a.tensor, b.tensor);
  return liftOp(out, [a, b], () => ({
    name: 'mul',
    inputs: [a, b],
    backward: (rtBackward, gradOutput) => [
      a.requiresGrad ? tensorMul(rtBackward, gradOutput, b.tensor) : null,
      b.requiresGrad ? tensorMul(rtBackward, gradOutput, a.tensor) : null,
    ],
  }));
}

/**
 * Elementwise `out = a * k` (scalar multiply) with autograd. Backward emits
 * `gradOutput * k`.
 */
export function scale(rt: TensorRuntime, a: Variable, k: number): Variable {
  const out = tensorScale(rt, a.tensor, k);
  return liftOp(out, [a], () => ({
    name: 'scale',
    inputs: [a],
    backward: (rtBackward, gradOutput) => [
      a.requiresGrad ? tensorScale(rtBackward, gradOutput, k) : null,
    ],
  }));
}

/**
 * Rank-2 transpose with autograd. Forward `[rows, cols] → [cols, rows]`;
 * backward transposes the upstream gradient back.
 */
export function transpose(rt: TensorRuntime, a: Variable): Variable {
  const out = tensorTranspose(rt, a.tensor);
  return liftOp(out, [a], () => ({
    name: 'transpose',
    inputs: [a],
    backward: (rtBackward, gradOutput) => [
      a.requiresGrad ? tensorTranspose(rtBackward, gradOutput) : null,
    ],
  }));
}

/**
 * Pure metadata reshape with autograd — **no `rt`** because forward neither
 * allocates nor dispatches (it shares the input buffer, per `tensor.reshape`).
 * Backward reshapes the upstream gradient back to the input's original shape;
 * since `tensor.reshape` shares buffers, both the forward output and the
 * backward gradient share buffers with their inputs.
 */
export function reshape(a: Variable, newShape: readonly number[]): Variable {
  const originalShape = a.tensor.shape;
  const out = tensorReshape(a.tensor, newShape);
  return liftOp(out, [a], () => ({
    name: 'reshape',
    inputs: [a],
    backward: (_rt, gradOutput) => [
      a.requiresGrad ? tensorReshape(gradOutput, originalShape) : null,
    ],
  }));
}
