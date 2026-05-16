import {
  add as tensorAdd,
  broadcastFillFromScalar as tensorBroadcastFillFromScalar,
  crossEntropyLogitsGrad as tensorCrossEntropyLogitsGrad,
  crossEntropyRowsFromLogits as tensorCrossEntropyRowsFromLogits,
  destroy,
  embedding as tensorEmbedding,
  gelu as tensorGelu,
  geluBackward as tensorGeluBackward,
  matmul as tensorMatmul,
  mul as tensorMul,
  reshape as tensorReshape,
  rmsnorm as tensorRmsnorm,
  rmsnormGradGamma as tensorRmsnormGradGamma,
  rmsnormGradInput as tensorRmsnormGradInput,
  scale as tensorScale,
  scatterEmbeddingAccum as tensorScatterEmbeddingAccum,
  softmax as tensorSoftmax,
  softmaxBackward as tensorSoftmaxBackward,
  sumFlat as tensorSumFlat,
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
 * Rank-2 matrix multiply with autograd. Forward `[M, K] @ [K, N] → [M, N]`;
 * backward emits `dA = dC @ B^T` and `dB = A^T @ dC`, composed from
 * `tensor.matmul` + `tensor.transpose`.
 */
export function matmul(rt: TensorRuntime, a: Variable, b: Variable): Variable {
  const out = tensorMatmul(rt, a.tensor, b.tensor);
  return liftOp(out, [a, b], () => ({
    name: 'matmul',
    inputs: [a, b],
    backward: (rtBackward, gradOutput) => [
      a.requiresGrad
        ? tensorMatmul(rtBackward, gradOutput, tensorTranspose(rtBackward, b.tensor))
        : null,
      b.requiresGrad
        ? tensorMatmul(rtBackward, tensorTranspose(rtBackward, a.tensor), gradOutput)
        : null,
    ],
  }));
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
 * Elementwise GELU with autograd (tanh approximation). Forward routes to
 * `tensor.gelu`; backward uses `tensor.geluBackward` with the saved pre-
 * activation input.
 */
export function gelu(rt: TensorRuntime, x: Variable): Variable {
  const out = tensorGelu(rt, x.tensor);
  return liftOp(out, [x], () => ({
    name: 'gelu',
    inputs: [x],
    backward: (rtBackward, gradOutput) => [
      x.requiresGrad ? tensorGeluBackward(rtBackward, gradOutput, x.tensor) : null,
    ],
  }));
}

/**
 * Row-wise RMSNorm (`gamma: [cols]`) with autograd. Uses dedicated backward kernels.
 */
export function rmsnorm(rt: TensorRuntime, x: Variable, gamma: Variable, epsilon = 1e-6): Variable {
  const out = tensorRmsnorm(rt, x.tensor, gamma.tensor, epsilon);
  return liftOp(out, [x, gamma], () => ({
    name: 'rmsnorm',
    inputs: [x, gamma],
    backward: (rtBackward, gradOutput) => [
      x.requiresGrad
        ? tensorRmsnormGradInput(rtBackward, gradOutput, x.tensor, gamma.tensor, epsilon)
        : null,
      gamma.requiresGrad ? tensorRmsnormGradGamma(rtBackward, gradOutput, x.tensor, epsilon) : null,
    ],
  }));
}

/**
 * Row-wise softmax on **rank-2** logits. Forward uses `tensor.softmax`;
 * backward uses `tensor.softmaxBackward` with the saved probability tensor `y`.
 */
export function softmax(rt: TensorRuntime, x: Variable): Variable {
  const y = tensorSoftmax(rt, x.tensor);
  return liftOp(y, [x], () => ({
    name: 'softmax',
    inputs: [x],
    backward: (rtBackward, gradOutput) => [
      x.requiresGrad ? tensorSoftmaxBackward(rtBackward, gradOutput, y) : null,
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

/**
 * Embedding gather (**`weights: [vocab, embed]`**, **`indices: [tokens]`** `u32`) with grads on **`weights`**.
 */
export function embedding(rt: TensorRuntime, weights: Variable, indices: Variable): Variable {
  if (indices.requiresGrad) {
    throw new Error('embedding: integer indices cannot require gradients');
  }
  if (indices.tensor.dtype !== 'u32') {
    throw new Error('embedding: indices tensor dtype must be u32 (use tensor.createIndices)');
  }
  const out = tensorEmbedding(rt, weights.tensor, indices.tensor);
  const vocabDim = weights.tensor.shape[0];
  if (vocabDim === undefined) {
    throw new Error('embedding: weight tensor must expose `shape[0]` (vocabulary size)');
  }
  const vocab = vocabDim;
  return liftOp(out, [weights, indices], () => ({
    name: 'embedding',
    inputs: [weights, indices],
    backward: (rtBackward, gradOutput) => [
      weights.requiresGrad
        ? tensorScatterEmbeddingAccum(rtBackward, indices.tensor, gradOutput, vocab)
        : null,
      null,
    ],
  }));
}

/**
 * Per‑row softmax cross‑entropy (**`targets`** **[rows]** `u32`). Saves **`tensor.softmax(logits)`** for logits gradients.
 */
export function crossEntropyRowsFromLogits(
  rt: TensorRuntime,
  logits: Variable,
  targets: Variable,
): Variable {
  if (targets.requiresGrad) {
    throw new Error('crossEntropyRowsFromLogits: class indices cannot require gradients');
  }
  if (targets.tensor.dtype !== 'u32') {
    throw new Error('crossEntropyRowsFromLogits: targets dtype must be u32');
  }
  const probsTensor = tensorSoftmax(rt, logits.tensor);
  const losses = tensorCrossEntropyRowsFromLogits(rt, logits.tensor, targets.tensor);
  return liftOp(losses, [logits, targets], () => ({
    name: 'cross_entropy_rows_logits',
    inputs: [logits, targets],
    backward: (rtBackward, gradOutput) => {
      try {
        return [
          logits.requiresGrad
            ? tensorCrossEntropyLogitsGrad(rtBackward, probsTensor, targets.tensor, gradOutput)
            : null,
          null,
        ];
      } finally {
        destroy(probsTensor);
      }
    },
  }));
}

/**
 * Sum **`rank‑1`** axis into **`[1]`**. Backward spreads the **`[1]`** upstream gradient uniformly.
 */
export function sumFlat(rt: TensorRuntime, x: Variable): Variable {
  const out = tensorSumFlat(rt, x.tensor);
  const len = x.tensor.shape[0];
  if (len === undefined) {
    throw new Error('sumFlat: rank-1 tensors must expose `shape[0]`');
  }
  const length = len;
  return liftOp(out, [x], () => ({
    name: 'sum_flat',
    inputs: [x],
    backward: (rtBackward, gradOutput) => [
      x.requiresGrad ? tensorBroadcastFillFromScalar(rtBackward, gradOutput, length) : null,
    ],
  }));
}

/** Mean softmax cross‑entropy (**`Σ rowLoss / rows`**) built from **`crossEntropyRowsFromLogits` + **`sum_flat`**. */
export function crossEntropyMeanFromLogits(
  rt: TensorRuntime,
  logits: Variable,
  targets: Variable,
): Variable {
  const rowCount = logits.tensor.shape[0];
  if (rowCount === undefined) {
    throw new Error('crossEntropyMeanFromLogits: logits must expose `shape[0]` row count');
  }
  const rows = rowCount;
  if (!(rows > 0)) {
    throw new Error('crossEntropyMeanFromLogits: logits.rows must be positive');
  }
  const rowsLoss = crossEntropyRowsFromLogits(rt, logits, targets);
  const pooled = sumFlat(rt, rowsLoss);
  return scale(rt, pooled, 1 / rows);
}
