## 1. GPU — softmax backward kernel and dispatch

- [x] 1.1 Add `packages/gpu/src/kernels/softmax-backward.wgsl`: for each row `i`, with `base = i * cols`, compute `dot = Σ_j y[base+j]*grad_out[base+j]`, then `grad_in[base+j] = y[base+j] * (grad_out[base+j] - dot)`. Reuse `SoftmaxDims`-style uniform `(rows, cols)` and the same row-indexing contract as `softmax.wgsl`.
- [x] 1.2 Add `loadSoftmaxBackwardWgsl()` in `load-core-wgsl.ts`.
- [x] 1.3 In `core-kernels.ts`: define `CORE_KERNEL_SOFTMAX_BACKWARD = 'core.softmax_backward'`, append to `CORE_KERNEL_IDS`, register with bindings `['uniform', 'read-only-storage', 'read-only-storage', 'storage']`, implement `dispatchSoftmaxBackward(ctx, registry, gradOut, y, gradIn, rows, cols)` mirroring `dispatchSoftmax` uniform fill and `dispatchKernel(..., rows)` thread grid.
- [x] 1.4 Extend `cpu-reference.ts` with `cpuSoftmaxBackward(gradOut, y, rows, cols)` (or row-wise helper) consistent with the JVP formula above.
- [x] 1.5 Export new `CORE_KERNEL_*`, `dispatchSoftmaxBackward`, and CPU helper(s) from `packages/gpu/src/index.ts`.
- [x] 1.6 Tests: GPU-gated check that `dispatchSoftmaxBackward` matches `cpuSoftmaxBackward` for small `[rows, cols]` (reuse patterns from `reduction-norm-kernels.spec.ts`).

## 2. Tensor — `softmax` and `softmaxBackward`

- [x] 2.1 Import `dispatchSoftmax` and `dispatchSoftmaxBackward` in `packages/tensor/src/ops.ts`.
- [x] 2.2 Add `softmax(rt, a: Tensor)`: `assertRank(a, 2)`, assert contiguity consistent with other rank-2 ops, `rows`/`cols` from `a.shape`, allocate `rows*cols` output, `dispatchSoftmax`, return `makeContiguousTensor`.
- [x] 2.3 Add `softmaxBackward(rt, gradOutput, softmaxOutput)`: `assertSameShape(gradOutput, softmaxOutput)`, rank-2 checks matching forward, allocate output, `dispatchSoftmaxBackward`.
- [x] 2.4 Re-export `softmax` and `softmaxBackward` from `packages/tensor/src/index.ts`.
- [x] 2.5 `tensor.spec.ts`: WebGPU — `softmax` vs `cpuSoftmax`; `softmaxBackward` vs `cpuSoftmaxBackward` (chain: logits → `cpuSoftmax` for `y`, random `gradOut`).

## 3. Autograd — `softmax` wrapper

- [x] 3.1 In `packages/autograd/src/ops.ts`, import `softmax as tensorSoftmax` and `softmaxBackward as tensorSoftmaxBackward`; add `softmax(rt, x)` with forward `tensorSoftmax(rt, x.tensor)`, capture forward output tensor `y` in closure, `backward`: `[ x.requiresGrad ? tensorSoftmaxBackward(rtBackward, gradOutput, y) : null ]`, `name: 'softmax'`.
- [x] 3.2 Re-export `softmax` from `packages/autograd/src/index.ts`.
- [x] 3.3 `ops.spec.ts`: forward matches `tensor.softmax`; backward matches `tensor.softmaxBackward`; no `Operation` when `requiresGrad === false`.
- [x] 3.4 `gradient-check.spec.ts`: rank-2 graph including `softmax` — analytic `backward` vs central-difference on logits (e.g. parameter or variable shaped `[rows, cols]`).

## 4. Verification

- [x] 4.1 `nx build` for `@llm-transformer-diy/gpu`, `tensor`, `autograd` — clean TypeScript.
- [x] 4.2 `nx run` tests for `gpu`, `tensor`, `autograd` — pass (GPU-gated cases skip cleanly without adapter).
- [x] 4.3 Confirm `packages/autograd` source does not import `@llm-transformer-diy/gpu`.
- [x] 4.4 Optionally update `openspec/specs/reduction-norm-kernels/spec.md` (or `core-wgsl-kernels` if that spec owns the `core.*` inventory) to list `core.softmax_backward` if maintainers keep live specs in lockstep.
