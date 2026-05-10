## Why

`docs/sequential-change-proposals.md` step 5 sequences autograd as **pairwise** backward coverage alongside existing forward kernels. `core.gelu` and `tensor.gelu` already exist; `packages/autograd` still has no `gelu` wrapper because the backward needs the derivative of the tanh-form GELU. A dedicated **WGSL backward kernel** keeps the GPU-first stack honest (no CPU-derivative upload shim) and gives `tensor` a single `geluBackward` primitive that autograd can call — the same layering as forward: `gpu` → `tensor` → `autograd`.

## What Changes

- **`packages/gpu`**: New WGSL shader `elementwise-gelu-backward.wgsl` implementing `grad_in[i] = grad_out[i] * gelu′(x[i])` for the same tanh approximation as `elementwise-gelu.wgsl`. Register as `core.gelu_backward` with dispatch helper `dispatchGeluBackward(ctx, registry, gradOut, inp, gradIn, elementCount)`. Extend `cpu-reference.ts` with `geluDerivativeScalar` / `cpuGeluBackward` (or equivalent) for numerical tests. Export new symbols from the package index; extend GPU tests / smoke helpers beside existing GELU coverage.
- **`packages/tensor`**: New op `geluBackward(rt, gradOutput, input)` — requires `gradOutput` and `input` to be same-shaped contiguous tensors, allocates output, dispatches `dispatchGeluBackward`. Re-export from `index.ts`. Add `toArray` round-trip test vs CPU reference when WebGPU is available (mirror `gelu` test style).
- **`packages/autograd`**: New `gelu(rt, x)` in `ops.ts` using `liftOp`: forward `tensor.gelu`, backward `tensor.geluBackward(rt, gradOutput, x.tensor)` when `x.requiresGrad`. Re-export from `index.ts`. GPU-gated `ops.spec.ts` tests: forward vs `tensor.gelu`, one-step `backward` vs reference, leaf fast-path when `requiresGrad === false`. Extend `gradient-check.spec.ts` with a small composed graph (e.g. `scale` ∘ `gelu` ∘ `add`) and central-difference check on a parameter.

## Capabilities

### New Capabilities

- **`autograd-gelu-op`**: End-to-end pairwise slice — `core.gelu_backward` WGSL kernel, `tensor.geluBackward`, `autograd.gelu` with tests and finite-difference validation.

### Modified Capabilities

- **`core-wgsl-kernels`** (conceptual): One new registered kernel ID under the `core.*` namespace for GELU backward; CPU reference extended for verification. On apply/archive, align `openspec/specs/core-wgsl-kernels/spec.md` if maintainers want the canonical spec list to mention `core.gelu_backward` explicitly.

## Impact

- **Primary**: `packages/gpu` (new WGSL + registration + dispatch + CPU reference + tests), `packages/tensor/src/ops.ts` + exports + spec test, `packages/autograd/src/ops.ts` + exports + `ops.spec.ts` + `gradient-check.spec.ts`.
- **Dependencies**: No new npm packages.
- **Downstream**: FFN blocks can use `autograd.gelu` in trainable graphs; softmax / rmsnorm backward remain future pairwise changes.
- **Explicitly out of scope**: `softmax` / `rmsnorm` backward, `embedding`, `cross_entropy`, optimizers, broadcasting, GELU variants other than the existing tanh approximation, second-order derivatives, fusing forward+backward kernels.
