## Why

`docs/sequential-change-proposals.md` step 5 sequences **pairwise** autograd coverage next to existing forward kernels. `core.softmax`, `dispatchSoftmax`, and `cpuSoftmax` already exist in `packages/gpu`, but **`packages/tensor` has no softmax op** — model and training code cannot route row-wise softmax through the tensor surface yet, and autograd has nothing stable to wrap. This change completes the vertical slice **gpu → tensor → autograd** for **rank-2 row-major** row-wise softmax (`[rows, cols]`, same layout contract as the WGSL kernel), including a **GPU backward path** (no “derivative on CPU then upload” for training).

## What Changes

- **`packages/gpu`**: New WGSL shader (e.g. `softmax-backward.wgsl`) implementing the Jacobian–vector product for row-wise softmax:

  \[
  \frac{\partial L}{\partial x*{ij}} = y*{ij}\left(\frac{\partial L}{\partial y*{ij}} - \sum_k y*{ik}\frac{\partial L}{\partial y\_{ik}}\right)
  \]

  where `y = softmax(x)` is the **forward output** (probabilities). Register as `core.softmax_backward` with the same uniform `(rows, cols)` pattern as `core.softmax` and binding layout **`uniform`, read-only `grad_out`, read-only `y`, storage `grad_in`**. Add `dispatchSoftmaxBackward(ctx, registry, gradOut, y, gradIn, rows, cols)` and extend **`cpu-reference.ts`** with `cpuSoftmaxBackward` (or equivalent) for tests. Export new symbols from `packages/gpu/index.ts`; add GPU-gated tests beside existing softmax coverage.

- **`packages/tensor`**: Add **`softmax(rt, a)`** — **rank 2 only**, contiguous row-major `[rows, cols]`, dispatches `dispatchSoftmax` (matches kernel contract). Add **`softmaxBackward(rt, gradOutput, softmaxOutput)`** where **`softmaxOutput` is the saved forward softmax tensor `y`** (same shape as `gradOutput`). Re-export from `index.ts`. WebGPU tests: forward vs `cpuSoftmax`; backward vs `cpuSoftmaxBackward`.

- **`packages/autograd`**: Add **`softmax(rt, x)`** via `liftOp`: forward **`tensor.softmax`**, backward **`tensor.softmaxBackward(rt, gradOutput, yTensor)`** where **`yTensor` is the forward output tensor** (capture in the `Operation` closure, analogous to saving activations — here the JVP is expressed in terms of `y`, not logits). Re-export from `index.ts`. GPU-gated **`ops.spec.ts`** (forward vs tensor, backward vs tensor) and **`gradient-check.spec.ts`** (analytic vs central difference on a small rank-2 graph).

## Capabilities

### New Capabilities

- **`autograd-softmax-op`**: End-to-end pairwise slice — `core.softmax_backward`, `tensor.softmax` / `tensor.softmaxBackward`, `autograd.softmax`, with CPU reference checks and backward validation.

### Modified Capabilities

- **`reduction-norm-kernels`** (conceptual): One new registered kernel ID under `core.*` for softmax backward; CPU reference extended. On apply/archive, optionally align `openspec/specs/reduction-norm-kernels/spec.md` if the project keeps live specs in lockstep with the registry.

## Impact

- **Primary**: `packages/gpu` (WGSL + registration + dispatch + CPU reference + tests), `packages/tensor/src/ops.ts` + exports + spec tests, `packages/autograd/src/ops.ts` + exports + `ops.spec.ts` + `gradient-check.spec.ts`.
- **Dependencies**: No new npm packages; `autograd` continues to import **`@llm-transformer-diy/tensor` only** (no direct `@llm-transformer-diy/gpu` in autograd).
- **Downstream**: Attention-style blocks can depend on `autograd.softmax` once the model layer lands; **`cross_entropy`** and masking remain separate proposals.
- **Explicitly out of scope**: `rmsnorm`, **`cross_entropy`**, attention / **causal masking**, sliding-window softmax, dtypes other than **f32**, **N-D softmax** or rank other than **2**, standalone **`reduce_*` wrappers** in tensor (unchanged unless already present), fusing forward+backward into one kernel, **in-place** softmax or backward.
