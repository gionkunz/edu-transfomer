## 1. GPU — GELU backward kernel and dispatch

- [x] 1.1 Add `packages/gpu/src/kernels/elementwise-gelu-backward.wgsl`: `grad_in[i] = grad_out * gelu′(inp[i])` with `gelu′` matching the tanh formulation and constants in `elementwise-gelu.wgsl` / `cpu-reference.ts`.
- [x] 1.2 Add `loadElementwiseGeluBackwardWgsl()` in `load-core-wgsl.ts`.
- [x] 1.3 In `core-kernels.ts`: define `CORE_KERNEL_GELU_BACKWARD = 'core.gelu_backward'`, append to `CORE_KERNEL_IDS`, register pipeline with binding kinds `['uniform', 'read-only-storage', 'read-only-storage', 'storage']`, implement `dispatchGeluBackward(ctx, registry, gradOut, inp, gradIn, elementCount)` mirroring `dispatchGelu` / binary elementwise uniform layout.
- [x] 1.4 Extend `cpu-reference.ts` with `geluDerivativeScalar(x)` and `cpuGeluBackward(gradOut, inp, n)` (or one helper that fuses multiply) — keep constants shared with `geluScalar`.
- [x] 1.5 Export `CORE_KERNEL_GELU_BACKWARD`, `dispatchGeluBackward`, and new CPU helpers from `packages/gpu/src/index.ts`.
- [x] 1.6 Tests: GPU-gated numerical check that `dispatchGeluBackward` output matches `cpuGeluBackward` for small random-ish vectors (reuse patterns from `smokeGelu` / `core-kernels` tests).

## 2. Tensor — `geluBackward` op

- [x] 2.1 Import `dispatchGeluBackward` in `packages/tensor/src/ops.ts`; add `geluBackward(rt, gradOutput, input)` with `assertSameShape(gradOutput, input)`, allocate output, dispatch, return `makeContiguousTensor`.
- [x] 2.2 Re-export `geluBackward` from `packages/tensor/src/index.ts`.
- [x] 2.3 In `tensor.spec.ts`, add WebGPU test: `geluBackward` readback matches `cpuGeluBackward` (same tolerance style as `gelu` test).

## 3. Autograd — `gelu` wrapper

- [x] 3.1 In `packages/autograd/src/ops.ts`, import `gelu as tensorGelu` and `geluBackward as tensorGeluBackward` from tensor; add `gelu(rt, x)` with forward `tensorGelu(rt, x.tensor)` and `liftOp` / `Operation` backward: `[ x.requiresGrad ? tensorGeluBackward(rtBackward, gradOutput, x.tensor) : null ]`, `name: 'gelu'`.
- [x] 3.2 Re-export `gelu` from `packages/autograd/src/index.ts` (alongside existing ops).
- [x] 3.3 `ops.spec.ts`: forward matches `tensor.gelu`; `backward` matches `tensor.geluBackward` for a crafted upstream grad; no `Operation` when `requiresGrad === false`.
- [x] 3.4 `gradient-check.spec.ts`: composed graph touching `gelu` (e.g. parameter → `add`/`scale` chain through `gelu`) with central-difference vs analytic grad on at least one full tensor.

## 4. Verification

- [x] 4.1 `nx build` for `@llm-transformer-diy/gpu`, `tensor`, `autograd` — clean TypeScript.
- [x] 4.2 `nx run @llm-transformer-diy/gpu:test`, `tensor:test`, `autograd:test` — all pass (GPU-gated cases skip cleanly without adapter).
- [x] 4.3 Confirm `packages/autograd` source (non-spec) does not import `@llm-transformer-diy/gpu`.
- [x] 4.4 Optionally update `openspec/specs/core-wgsl-kernels/spec.md` to document `core.gelu_backward` if project convention is to keep live specs in lockstep with the registry.
