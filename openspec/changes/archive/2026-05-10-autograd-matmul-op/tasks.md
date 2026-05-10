## 1. Matmul autograd op wrapper

- [x] 1.1 Add `matmul(rt: TensorRuntime, a: Variable, b: Variable): Variable` to `packages/autograd/src/ops.ts`. Import `matmul as tensorMatmul` and `transpose as tensorTranspose` from `@llm-transformer-diy/tensor`. Forward: `tensorMatmul(rt, a.tensor, b.tensor)`. Use the existing `liftOp` helper. When at least one input requires grad, build an `Operation` with `name: 'matmul'`, `inputs: [a, b]`, and backward:

  ```
  backward(rtBackward, gradOutput) => [
    a.requiresGrad
      ? tensorMatmul(rtBackward, gradOutput, tensorTranspose(rtBackward, b.tensor))
      : null,
    b.requiresGrad
      ? tensorMatmul(rtBackward, tensorTranspose(rtBackward, a.tensor), gradOutput)
      : null,
  ]
  ```

  JSDoc: "Rank-2 matrix multiply with autograd. Forward `[M, K] @ [K, N] → [M, N]`; backward emits `dA = dC @ B^T` and `dB = A^T @ dC`."

- [x] 1.2 Re-export `matmul` from `packages/autograd/src/index.ts` alongside the existing `add, mul, reshape, scale, transpose` in the `ops` export line.

## 2. Unit tests — forward + backward

- [x] 2.1 In `packages/autograd/src/ops.spec.ts`, add a GPU-gated test: **"matmul forward matches tensor.matmul for non-square shapes"**. Use `A` of shape `[2, 3]` and `B` of shape `[3, 4]`. Create both as parameters (`createParameter`). Call `autograd.matmul(rt, a, b)`. Assert: `out.requiresGrad === true`, `out.creator?.name === 'matmul'`, `out.tensor.shape === [2, 4]`. Read back output and compare element-wise (within float tolerance) to `tensor.matmul(rt, a.tensor, b.tensor)`.

- [x] 2.2 In the same file, add a GPU-gated test: **"matmul backward emits dA = dC @ B^T and dB = A^T @ dC"**. Use `A [2, 3]`, `B [3, 4]`, upstream gradient `dC [2, 4]` with known values. Call `out.creator!.backward(rt, dC)`. Read back `grads[0]` (dA) and `grads[1]` (dB). Assert shapes: `dA` is `[2, 3]`, `dB` is `[3, 4]`. Compute expected values via `tensor.matmul(rt, dC, tensor.transpose(rt, B))` and `tensor.matmul(rt, tensor.transpose(rt, A), dC)` and compare element-wise.

- [x] 2.3 In the same file, add a GPU-gated test: **"matmul does not record Operation when no input requires grad"**. Create both inputs via `createLeafVariable(..., false)`. Call `autograd.matmul(rt, a, b)`. Assert `out.requiresGrad === false` and `out.creator === null`. Forward result still correct.

## 3. Finite-difference gradient check

- [x] 3.1 In `packages/autograd/src/gradient-check.spec.ts`, add a new GPU-gated test: **"matmul gradients on a non-square composed graph match central-difference numerical gradients"**. Graph:
  - `W` is a parameter of shape `[K=3, N=4]`.
  - `X` is a parameter of shape `[M=2, K=3]`.
  - `product = matmul(rt, X, W)` — shape `[2, 4]`.
  - `output = scale(rt, reshape(product, [8]), 2.0)` — shape `[8]`.
  - Loss = `Σ output` (backward seeds ones of shape `[8]`).

  Add a `runMatmulGraph` helper (analogous to the existing `runComposedGraph`) that:
  1. Creates tensors from `Float32Array` data, wraps as parameters.
  2. Runs the forward graph.
  3. Optionally runs `backward(rt, output)` and reads back `X.grad` and `W.grad`.
  4. Destroys all uniquely-owned GPU buffers.

  For each element of `X` and `W`, perturb by `±ε` (ε = 1e-3), run two forward passes, compute `(f_plus - f_minus) / (2ε)`, and compare to the analytic gradient. Tolerance: `|analytic - numerical| ≤ 5e-3`.

  The test must exercise both:
  - `dX` path: `dX = dC @ W^T` — shape `[2, 3]` (different from `gradOutput` `[2, 4]`).
  - `dW` path: `dW = X^T @ dC` — shape `[3, 4]` (different from `X` `[2, 3]`).

## 4. Verification

- [x] 4.1 `nx build @llm-transformer-diy/autograd` — clean build, no TS errors.
- [x] 4.2 `nx run @llm-transformer-diy/autograd:test` — 28/28 tests pass (5 topology + 6 registry pure, 10 ops + 5 backward + 2 gradient-check GPU-gated). All pre-existing tests unchanged.
- [x] 4.3 Grep audit:
  - `^import .* from ['"]@llm-transformer-diy/gpu['"]` in non-spec autograd source — **no matches** (autograd does not import gpu directly).
  - `\.grad\s*=` in non-spec autograd source — matches only `accumulateGrad` (`variable.ts:91,94`) and `zeroGrad` (`parameter-registry.ts:40`). Matmul backward does not write `.grad`.
- [x] 4.4 No edits under `packages/tensor/` or `packages/gpu/` confirmed via `git diff`.
