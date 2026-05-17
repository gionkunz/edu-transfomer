import { add as tensorAdd, createFromArray, destroy } from '@llm-transformer-diy/tensor';
import type { FloatDType, Tensor, TensorRuntime } from '@llm-transformer-diy/tensor';

import type { Operation, Parameter, Variable } from './types';

/**
 * Process-monotonic counter for {@link Variable.id}. This is the **only**
 * module-level mutable state in `@llm-transformer-diy/autograd`. Reset only on
 * Node.js process restart; cross-process identity is undefined.
 */
let nextVariableId = 0;

/** Allocate the next id. Internal helper, not re-exported. */
function nextId(): number {
  const id = nextVariableId;
  nextVariableId += 1;
  return id;
}

/**
 * Build a leaf {@link Variable} that has no creator. Use for external inputs
 * (training batches, mask tensors). For trainable leaves, prefer
 * {@link createParameter}, which always sets `requiresGrad: true`.
 */
export function createLeafVariable(tensor: Tensor, requiresGrad: boolean): Variable {
  return {
    id: nextId(),
    tensor,
    requiresGrad,
    creator: null,
    parents: [],
    grad: null,
  };
}

/**
 * Build a trainable leaf — a {@link Parameter} with `requiresGrad: true` and
 * `creator: null`. This is the canonical constructor for any tensor that
 * should accumulate gradients during backward and be tracked by a
 * {@link import('./types').ParameterRegistry}.
 */
export function createParameter(tensor: Tensor): Parameter {
  return {
    id: nextId(),
    tensor,
    masterTensor: null,
    requiresGrad: true,
    creator: null,
    parents: [],
    grad: null,
  };
}

/**
 * Trainable weights at **`computeDtype`** (bf16) with a parallel **`f32`**
 * {@link Parameter.masterTensor} for optimizer state. Forward / backward use
 * {@link Parameter.tensor}; {@link import('@llm-transformer-diy/training').AdamW}
 * updates `masterTensor` then casts into `tensor`.
 *
 * @throws If **`computeDtype === 'f32'`** — use {@link createParameter} instead.
 */
export function createParameterWithMaster(
  rt: TensorRuntime,
  data: Float32Array,
  shape: readonly number[],
  computeDtype: FloatDType,
): Parameter {
  if (computeDtype === 'f32') {
    throw new Error(
      'createParameterWithMaster: for f32 compute weights use createParameter(createFromArray(...))',
    );
  }
  const tensor = createFromArray(rt, data, shape, computeDtype);
  const masterTensor = createFromArray(rt, data, shape, 'f32');
  return {
    id: nextId(),
    tensor,
    masterTensor,
    requiresGrad: true,
    creator: null,
    parents: [],
    grad: null,
  };
}

/**
 * Build an internal graph node — a {@link Variable} produced by an op wrapper.
 * The caller passes the freshly-computed forward tensor and the bound
 * {@link Operation}; `parents` mirrors `op.inputs` so the topo walk avoids a
 * null-check per node.
 *
 * Not re-exported from the package index — only `./ops` constructs internal
 * nodes.
 */
export function createInternalVariable(
  tensor: Tensor,
  op: Operation,
  requiresGrad: boolean,
): Variable {
  return {
    id: nextId(),
    tensor,
    requiresGrad,
    creator: op,
    parents: op.inputs,
    grad: null,
  };
}

/**
 * Accumulate `contribution` onto `target.grad`.
 *
 * - When `target.grad === null`, assigns the contribution directly (no
 *   `tensor.add` dispatch — saves a buffer allocation on the common single-
 *   path case). Ownership of `contribution` transfers to `target`.
 * - Otherwise, allocates a fresh sum via `tensor.add(rt, target.grad,
 *   contribution)`, reassigns, and **releases the previous `target.grad`
 *   and the consumed `contribution`** back to the pool. This is the
 *   leak-free path for residual junctions (a single `x` feeding rmsnorm AND
 *   the residual `add` produces two backward contributions; the first
 *   assigns, the second sums and must free the now-stale intermediate).
 *
 * Ownership contract: callers MUST pass a `contribution` they exclusively
 * own (i.e., not aliased with any other live tensor / variable's `.grad`).
 * Every op `backward` in `./ops.ts` honors this — see the `add` op comment
 * for the historical violation and why it's now cloned.
 *
 * This helper and {@link import('./parameter-registry').zeroGrad} are the
 * **only** writes to {@link Variable.grad} in `@llm-transformer-diy/autograd`.
 */
export function accumulateGrad(rt: TensorRuntime, target: Variable, contribution: Tensor): void {
  if (target.grad === null) {
    target.grad = contribution;
    return;
  }
  const previous = target.grad;
  const sum = tensorAdd(rt, previous, contribution);
  target.grad = sum;
  destroy(previous);
  destroy(contribution);
}
