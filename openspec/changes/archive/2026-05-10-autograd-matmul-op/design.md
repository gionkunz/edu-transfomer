## Context

`packages/autograd` ships a define-by-run engine (from `2026-05-10-autograd-bootstrap`) with `Variable`, `Operation`, topological `backward`, `ParameterRegistry`, and autograd wrappers for `add` / `mul` / `scale` / `transpose` / `reshape`. Each op wrapper follows the pattern in `ops.ts`: forward via the corresponding `tensor.*` function, backward via a closure that captures input tensors and computes `∂output/∂input · gradOutput` using the same tensor surface.

`packages/tensor` already exposes `matmul(rt, a, b)` (rank-2 `[M, K] @ [K, N] → [M, N]`) and `transpose(rt, a)` (rank-2 `[R, C] → [C, R]`). The matmul backward formula composes entirely from these two primitives:

- `dA = gradOutput @ B^T` — `[M, N] @ [N, K] → [M, K]`, matching `A`'s shape.
- `dB = A^T @ gradOutput` — `[K, M] @ [M, N] → [K, N]`, matching `B`'s shape.

No new WGSL kernels, no new tensor ops, no changes to `packages/tensor` or `packages/gpu`.

Constraints carried forward from `autograd-bootstrap`:

- **Functional style**: no classes, no methods on records. The new `matmul` wrapper is a free function in `ops.ts` using the existing `liftOp` helper.
- **Layering**: `autograd → tensor → gpu`. Autograd does not import `@llm-transformer-diy/gpu` directly.
- **GPU-only**: tests gate behind `describe.skipIf(!webgpuAvailable)`.
- **`Variable.grad` mutability audit**: only `accumulateGrad` and `zeroGrad` write `.grad`. The new op's backward allocates fresh tensors via `tensor.matmul` / `tensor.transpose`; it never mutates inputs or `gradOutput`.

## Goals / Non-Goals

**Goals:**

- Add `matmul(rt: TensorRuntime, a: Variable, b: Variable): Variable` to `packages/autograd/src/ops.ts`, following the exact same `liftOp` pattern as `add` / `mul` / `scale` / `transpose`.
- Forward: `tensor.matmul(rt, a.tensor, b.tensor)`.
- Backward: `dA = tensor.matmul(rt, gradOutput, tensor.transpose(rt, b.tensor))`, `dB = tensor.matmul(rt, tensor.transpose(rt, a.tensor), gradOutput)`. Return `null` at positions where the input does not require grad.
- Re-export `matmul` from the package index.
- Unit-test forward correctness and backward analytic values (non-square shapes).
- Finite-difference gradient check on a composed graph including matmul with asymmetric `[M=2, K=3] @ [K=3, N=4]` shapes, verifying both `dA` (`[2, 3]`) and `dB` (`[3, 4]`) paths.

**Non-Goals:**

- Batched / broadcast matmul (rank > 2).
- `gelu` / `softmax` / `rmsnorm` backward (separate pairwise changes).
- `embedding`, `cross_entropy`, optimizer kernels, `clip_norm`.
- Edits to `packages/tensor` or `packages/gpu`.
- Fused `matmul-backward` WGSL kernel — backward is composed from existing `matmul` + `transpose` dispatches.
- Performance optimization of the backward (two separate matmul + one or two transpose dispatches per backward call is acceptable in the correctness-first phase).

## Decisions

| Decision                                                                                                                                   | Rationale                                                                                                                                                                                                                                                                                                         |
| ------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Backward composes `tensor.matmul` + `tensor.transpose` rather than dispatching a fused matmul-backward kernel.**                         | No new WGSL kernels in this scope. The composed path is correct and matches the "own the stack" educational goal — the student sees exactly how `dA = dC @ B^T` maps to existing primitives. A future perf pass can fuse if profiling shows the two-dispatch overhead matters.                                    |
| **`matmul` wrapper captures `a.tensor` and `b.tensor` in the backward closure.**                                                           | Same pattern as `mul` (which captures both inputs). The captured tensors are borrowed; backward does not mutate them. Lifecycle note: if the caller destroys `a.tensor` before backward runs, backward will read a destroyed GPUBuffer — same caveat as all other ops; documented in autograd-bootstrap's design. |
| **The backward closure allocates two intermediate `transpose` tensors** (`B^T` and `A^T`) inside each backward call. These are not cached. | Caching `B^T` during forward would save one `transpose` dispatch during backward but would double the memory per matmul node for a gain that only matters during training. Correctness-first; cache in a later perf pass if needed.                                                                               |
| **Non-square test shape `[2, 3] @ [3, 4]` is chosen for the gradient check.**                                                              | The asymmetry forces `dA` to be `[2, 3]` and `dB` to be `[3, 4]` — different shapes from each other and from `gradOutput` `[2, 4]`. This catches transposition bugs that a square shape would mask.                                                                                                               |
| **The gradient-check test composes `matmul` with `scale` (and optionally `add`) to exercise a multi-node backward chain.**                 | A single-node matmul test only validates the local Jacobian; composing with another op tests that `accumulateGrad` threads the chain correctly through a matmul node sitting in the middle of a graph.                                                                                                            |
| **Tests are added to the existing `ops.spec.ts` and `gradient-check.spec.ts` rather than new files.**                                      | Matches the per-module test organization from autograd-bootstrap. The files remain at manageable line counts (~260 and ~220 respectively after additions).                                                                                                                                                        |

**Alternatives considered:**

- **Cache `B^T` during forward for reuse in backward.** Rejected: increases forward memory; the two-dispatch backward is clear and correct. Revisit if training profiling shows transpose is a bottleneck.
- **Separate `matmul.spec.ts` file.** Rejected: the existing `ops.spec.ts` already tests all autograd ops in a single describe block with shared setup; splitting matmul out would duplicate the `beforeAll` / `afterEach` boilerplate for no structural benefit.
- **Test only square shapes in the gradient check.** Rejected: square shapes hide shape bugs in the backward formula — `dA` and `dB` would have the same dimensions, and transposition bugs could cancel out.

## Risks / Trade-offs

- **[Risk] Two extra `transpose` dispatches per matmul backward.** Mitigation: acceptable for correctness-first; the dispatch cost is dominated by matmul itself for any non-trivial shape. Future optimization can cache transposes or fuse the backward kernel.
- **[Risk] Captured tensors in the backward closure may be destroyed before backward runs.** Mitigation: same risk and same mitigation as all existing autograd ops (documented in autograd-bootstrap). Callers must keep input tensors alive through the backward call. No automatic reference counting in this scope.
- **[Risk] `f32` precision in the finite-difference check for matmul.** Mitigation: matmul accumulates K inner products; for `K=3` the accumulation error is negligible. Central difference with `ε = 1e-3` and tolerance `5e-3` is the same proven regime as the existing gradient-check spec.

## Migration Plan

- **Deploy**: Add `matmul` to `packages/autograd/src/ops.ts`, re-export from `index.ts`, extend `ops.spec.ts` and `gradient-check.spec.ts`. No edits to any other package.
- **Rollback**: Revert the additions to `ops.ts`, `index.ts`, `ops.spec.ts`, `gradient-check.spec.ts`. No downstream consumers exist yet (model layer is still a stub).

## Open Questions

_(none — the design is fully determined by the existing op-wrapper pattern and the textbook backward formula.)_
