## Context

Forward GELU uses `elementwise-gelu.wgsl` and `dispatchGelu` under `core.gelu`. `packages/autograd` follows a single pattern: forward through `tensor.*`, backward through `tensor.*` only (no direct `@llm-transformer-diy/gpu` imports in autograd). The GELU backward is **not** expressible as a short composition of existing tensor ops (unlike `matmul` backward), so this change introduces one new elementwise kernel and a matching tensor op.

## Goals / Non-Goals

**Goals:**

- Implement `gelu′(x)` for the exact same scalar definition as `gelu_f32` / `geluScalar` (constants `GELU_TANH_COEF`, `GELU_CUBE_COEF` unchanged).
- Kernel computes `grad_in = grad_out ⊙ gelu′(input)` with read-only `grad_out` and `input` buffers.
- `tensor.geluBackward(rt, gradOutput, input)` mirrors other binary elementwise dispatches (shape check, fresh output buffer).
- `autograd.gelu(rt, x)` captures `x.tensor` in the `Operation` closure and calls `geluBackward` when `x.requiresGrad`.
- GPU-gated tests: kernel vs CPU reference; autograd forward/backward and finite-difference on a composed graph.

**Non-Goals:**

- Custom approximate GELU (QuickGELU, etc.).
- In-place backward or fusing with forward.
- Optimizing away the extra read of `input` in backward (correctness and clarity first).

## Decisions

| Decision                                                                               | Rationale                                                                                                                                                                                                                       |
| -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Kernel ID `core.gelu_backward`**                                                     | Keeps naming parallel to `core.gelu`; registry remains the single lookup table for training-related elementwise grads.                                                                                                          |
| **Bind layout: `uniform(n)`, `grad_out` RO, `inp` RO, `grad_in` RW**                   | Matches the pattern of elementwise binary ops with an extra uniform; three storage buffers, all `f32`.                                                                                                                          |
| **Derivative closed form (tanh GELU)**                                                 | Let `c = GELU_TANH_COEF`, `d = GELU_CUBE_COEF`, `u = c·(x + d·x³)`, `t = tanh(u)`. Then `gelu′(x) = ½(1+t) + ½·x·(1−t²)·c·(1+3d·x²)`. Matches the standard derivation for this parameterization and is straightforward in WGSL. |
| **`cpuGeluBackward` (or fused helper) in `cpu-reference.ts`**                          | Same source of truth as `cpuGelu` for Vitest; avoids duplicating constants in tests.                                                                                                                                            |
| **Autograd backward uses saved forward input `x.tensor`, not recomputing from output** | GELU is not bijective; backward must use pre-activation `x`. Capturing `x.tensor` in the closure is the same lifecycle contract as `mul`.                                                                                       |

**Alternatives considered:**

- **Express backward with existing ops only** — Would require exposing `tanh` and many temps on the tensor surface; rejects educational clarity and adds dispatch overhead.
- **CPU derivative + upload** — Violates GPU-first posture documented in `docs/project.md` for trainable paths.

## Risks / Trade-offs

- **Numerical tolerance**: Elementwise product of `grad_out` and `gelu′` can amplify error vs forward-only GELU tests; use the same float tolerances family as existing `mul` / gradient-check specs, relax slightly if Kahan-level accuracy is unnecessary.
- **Buffer lifetime**: Callers must keep `input` alive through backward, identical to other autograd ops.

## Migration Plan

- Land gpu → tensor → autograd in one change (or in bottom-up commit order within the same PR).
- No public API breaks: only additive exports.
- **Rollback**: Remove the new kernel, tensor op, autograd wrapper, and tests.

## Open Questions

_(none)_
