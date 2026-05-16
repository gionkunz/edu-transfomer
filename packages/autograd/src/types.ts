import type { Tensor, TensorRuntime } from '@llm-transformer-diy/tensor';

/**
 * Define-by-run computation-graph node.
 *
 * Every field except {@link Variable.grad} is `readonly`. `grad` is the **only
 * mutable** field on a `Variable` and is written in exactly two places in this
 * package: {@link import('./variable').accumulateGrad} (during backward
 * traversal) and {@link import('./parameter-registry').zeroGrad} (between
 * training steps). No other code path reassigns `Variable.grad`.
 *
 * Construct via {@link import('./variable').createLeafVariable},
 * {@link import('./variable').createParameter}, or one of the autograd op
 * wrappers in `./ops`. The internal-node constructor is not exported from the
 * package index.
 */
export interface Variable {
  /** Process-monotonic id; supports topological-sort de-duplication. */
  readonly id: number;
  /** Forward value for this node. Borrowed from the producing op or leaf. */
  readonly tensor: Tensor;
  /** When `false`, no gradient is accumulated onto this variable's `grad`. */
  readonly requiresGrad: boolean;
  /** `null` for leaves (parameters or external inputs); set for op outputs. */
  readonly creator: Operation | null;
  /** Cached `creator?.inputs ?? []`. Stored so the topo walk avoids null-checks. */
  readonly parents: readonly Variable[];
  /** Accumulated gradient. `null` until backward reaches this node. */
  grad: Tensor | null;
}

/**
 * The local backward formula for a single forward op, bound at construction
 * time to its concrete `inputs`. Each op wrapper in `./ops` builds one of these
 * inside the forward call (capturing whatever input tensors backward needs in
 * the closure).
 *
 * `backward` MUST return one entry per `inputs` position. Entries MAY be `null`
 * at positions whose corresponding input has `requiresGrad === false`; the
 * topo traversal skips accumulation for `null` entries.
 *
 * `backward` MUST NOT mutate `gradOutput` and MUST NOT mutate any input's
 * fields. It MAY allocate fresh tensors via the supplied `TensorRuntime`.
 */
export interface Operation {
  readonly name: string;
  readonly inputs: readonly Variable[];
  readonly backward: (rt: TensorRuntime, gradOutput: Tensor) => readonly (Tensor | null)[];
}

/**
 * A trainable leaf — a {@link Variable} with `requiresGrad: true` and
 * `creator: null`. Structurally identical to `Variable`; the alias exists for
 * documentation and grep-friendliness inside the parameter registry.
 *
 * When {@link masterTensor} is non-null, it holds an **`f32`** view updated by
 * {@link import('@llm-transformer-diy/training').AdamW}; the optimizer casts
 * into {@link Variable.tensor} (bf16 compute weights) after each step.
 */
export interface Parameter extends Variable {
  readonly requiresGrad: true;
  readonly creator: null;
  /**
   * **`f32`** master weights for mixed-precision training, or `null` when
   * {@link Variable.tensor} is the sole **`f32`** buffer (legacy path).
   */
  readonly masterTensor: Tensor | null;
}

/**
 * Append-only collection of trainable {@link Parameter}s.
 *
 * `params` is a mutable array and is written **only** through
 * {@link import('./parameter-registry').registerParameter}. The registry does
 * not deduplicate registrations and does not support removal in this scope.
 */
export interface ParameterRegistry {
  readonly params: Parameter[];
}
