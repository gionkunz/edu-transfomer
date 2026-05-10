## Context

Forward row-wise softmax is implemented in `softmax.wgsl` and exposed as `core.softmax` / `dispatchSoftmax` with CPU twin `cpuSoftmax` in `packages/gpu`. `packages/tensor` routes other ops through `dispatch*` helpers but **does not yet expose softmax**; `packages/autograd` follows the rule: **forward and backward call `tensor.*` only** (no direct GPU imports in autograd). Softmax backward is **not** a short composition of existing tensor primitives without either multiple temporaries and reductions or a dedicated row-wise kernel; a single **`core.softmax_backward`** dispatch matches the educational stack and the **GPU-first training** rule from `docs/project.md`.

## Goals / Non-Goals

**Goals:**

- **`tensor.softmax(rt, a)`** for rank-2 `[rows, cols]` contiguous tensors, same contract as `dispatchSoftmax` (reject wrong rank / non-contiguous before dispatch, matching `transpose` / `matmul` style guards).
- **`tensor.softmaxBackward(rt, gradOutput, y)`** where `y` is the **forward softmax output** (probabilities), not raw logits — required for the standard JVP formula above.
- **`core.softmax_backward`** WGSL: one thread per row (same dispatch geometry as `core.softmax`), read `grad_out` and `y`, write `grad_x`.
- **`autograd.softmax(rt, x)`** captures the **forward output tensor** in the `Operation` closure for backward (same lifecycle as needing saved activations; here the math needs **`y`**, not **`x`**).
- Tests: forward vs `cpuSoftmax`; backward vs **`cpuSoftmaxBackward`**; autograd round-trip and finite-difference on small `[rows, cols]`.

**Non-Goals:**

- Backprop w.r.t. logits recomputed from `y` alone (ill-posed / unused here).
- Masked softmax, dtype generalization, or softmax over arbitrary axes.
- Using CPU backward + `writeBuffer` for the training path in autograd.

## Decisions

| Decision                                                                   | Rationale                                                                                                                                              |
| -------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Kernel ID `core.softmax_backward`**                                      | Parallel to `core.gelu_backward` / `core.softmax`; registry remains the single ID table.                                                               |
| **Bind layout: `uniform(rows,cols)`, `grad_out` RO, `y` RO, `grad_in` RW** | Three rank-2-shaped buffers + same 16-byte uniform as forward softmax; `y` must be the normalized probabilities.                                       |
| **Per-row algorithm in WGSL**                                              | First pass per row: `dot = Σ_j y[j]*grad_out[j]`; second pass: `grad_in[j] = y[j]*(grad_out[j]-dot)`. Matches the Jacobian–vector product for softmax. |
| **`cpuSoftmaxBackward` in `cpu-reference.ts`**                             | Same ground-truth style as `cpuSoftmax` / `cpuGeluBackward` for Vitest.                                                                                |
| **Autograd captures `y = tensor.softmax(...)` in closure**                 | Backward requires forward output probabilities; capturing the forward `Tensor` avoids recomputing softmax in backward.                                 |

**Alternatives considered:**

- **Compose backward from `mul`, `reduce_sum`, etc.** — Would require row-wise dot products and broadcasts not yet exposed as ergonomic tensor ops; more dispatches and temporaries than one dedicated kernel.
- **CPU JVP + upload** — Rejected for training path per project GPU-first stance.

## Risks / Trade-offs

- **Numerical tolerance**: Products involving small probabilities can widen float error vs forward-only tests; align tolerances with existing `gelu` / `gradient-check` specs.
- **Buffer lifetime**: Callers must keep forward-output `y` consistent with the logits that produced it; autograd saves the tensor handle from the forward pass like other ops.

## Migration Plan

- Land **gpu → tensor → autograd** in bottom-up commit order within one change/PR.
- **Additive** public API only.
- **Rollback**: Remove kernel, tensor ops, autograd wrapper, and tests.

## Open Questions

_(none)_
