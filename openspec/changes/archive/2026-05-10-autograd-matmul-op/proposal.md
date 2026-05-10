## Why

`packages/autograd` ships a define-by-run engine with `Variable`, `Operation`, topological `backward`, `ParameterRegistry`, and autograd wrappers for `add` / `mul` / `scale` / `transpose` / `reshape` (landed in `2026-05-10-autograd-bootstrap`). Every layer above — `model` (Linear, Attention, FFN), `training` — requires **matmul with gradient support** before a single trainable forward pass can execute. `tensor.matmul` and `tensor.transpose` already exist; the backward formula is textbook (`dA = dC @ B^T`, `dB = A^T @ dC`) and composes entirely from those two primitives — no new WGSL kernels, no new tensor ops. This change lands **matmul** as the next autograd op, following the pairwise strategy from `docs/sequential-change-proposals.md` step 5: _"add backward coverage pairwise with forward kernels."_

## What Changes

- Add a `matmul(rt, a, b)` autograd op wrapper in `packages/autograd/src/ops.ts` alongside the existing `add` / `mul` / `scale` / `transpose` / `reshape`. Forward routes to `tensor.matmul(rt, a.tensor, b.tensor)`. When at least one input requires grad, an `Operation` is recorded whose backward computes:
  - `dA = tensor.matmul(rt, gradOutput, tensor.transpose(rt, b.tensor))` — shape `[M, K]`
  - `dB = tensor.matmul(rt, tensor.transpose(rt, a.tensor), gradOutput)` — shape `[K, N]`
    …returning `null` at positions whose corresponding input has `requiresGrad === false`.
- Re-export `matmul` from `packages/autograd/src/index.ts`.
- Add **GPU-gated Vitest tests** in the existing `ops.spec.ts`:
  - Forward output of `autograd.matmul` matches `tensor.matmul` element-wise.
  - One-step `Operation.backward` against the hand-derived analytic formula for non-square shapes.
  - Leaf-fast-path scenario (no input requires grad ⇒ `creator === null`).
- Add a **finite-difference gradient check** in `gradient-check.spec.ts` on a composed graph that includes matmul with a **non-square shape** (`[M=2, K=3] @ [K=3, N=4]`) so both `dA` (`[2, 3]`) and `dB` (`[3, 4]`) shape paths are exercised. The graph composes matmul with at least one other autograd op (e.g. `scale`) to exercise the backward through a multi-node chain.

## Capabilities

### New Capabilities

- `autograd-matmul-op`: `matmul(rt, a, b)` autograd wrapper — forward via `tensor.matmul`, backward via `dA = dC @ B^T` and `dB = A^T @ dC` composed from `tensor.matmul` + `tensor.transpose`. Finite-difference validated on a non-square composed graph.

### Modified Capabilities

- _(none — the autograd-bootstrap engine, tensor surface, and GPU kernels are unchanged. This change only adds a new op wrapper and its tests.)_

## Impact

- **Primary**: `packages/autograd/src/ops.ts` (new `matmul` wrapper), `packages/autograd/src/index.ts` (re-export), `packages/autograd/src/ops.spec.ts` (forward + backward unit tests), `packages/autograd/src/gradient-check.spec.ts` (finite-difference test with matmul in the graph).
- **Dependencies**: No new npm or workspace dependencies. The existing `@llm-transformer-diy/tensor` runtime edge and `@llm-transformer-diy/gpu` / `@webgpu/types` dev-dependencies remain unchanged.
- **Downstream**: Unblocks `packages/model`'s `Linear` layer (which is fundamentally `matmul(x, W) + bias`). Subsequent pairwise changes wrap `gelu`, `softmax`, and `rmsnorm` backward.
- **Explicitly out of scope**: `gelu` / `softmax` / `rmsnorm` backward, `embedding`, `cross_entropy`, optimizer kernels (`adamw`, `clip_norm`), broadcasting in autograd, any edits to `packages/tensor` or `packages/gpu`.
