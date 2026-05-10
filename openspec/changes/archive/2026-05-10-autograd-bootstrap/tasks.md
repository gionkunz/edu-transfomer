## 1. Package layout and types

- [x] 1.1 Add new source modules under `packages/autograd/src/`: `types.ts` (`Variable`, `Operation`, `Parameter`, `ParameterRegistry` `readonly` interfaces; only `Variable.grad` is non-`readonly`), `variable.ts` (id allocator + leaf / parameter / internal-node constructors + `accumulateGrad` helper), `backward.ts` (topo-sort + `backward(rt, output)`), `parameter-registry.ts` (registry helpers), `ops.ts` (autograd op wrappers). **No class declarations** anywhere in the package.
- [x] 1.2 Replace the stub-only `packages/autograd/src/index.ts` with real exports: types (`Variable`, `Operation`, `Parameter`, `ParameterRegistry`), constructors (`createLeafVariable`, `createParameter`, `createParameterRegistry`), op wrappers (`add`, `mul`, `scale`, `transpose`, `reshape`), backward (`backward`, `accumulateGrad`, `topologicalOrder`), and registry helpers (`registerParameter`, `parameters`, `zeroGrad`, `totalParams`). Keep `AUTOGRAD_LAYER_MARKER` and `__autogradDependsOnTensor` exports retained until a downstream consumer is migrated off them — `packages/model/src/index.ts` still imports `AUTOGRAD_LAYER_MARKER`. _(Also re-exports `topologicalOrder` so the pure topo spec can exercise the helper directly.)_
- [x] 1.3 Confirm `packages/autograd/package.json` keeps `@llm-transformer-diy/tensor` as the only **runtime** workspace dependency. _(Added `@llm-transformer-diy/gpu` and `@webgpu/types` as `devDependencies` for test-setup only — production code does not import either, per the layering audit in 7.4. Spec/audit refined to scope the gpu-import audit to non-spec source.)_

## 2. Variable, Operation, and constructors

- [x] 2.1 Define `Variable` in `types.ts` as `{ readonly id: number; readonly tensor: Tensor; readonly requiresGrad: boolean; readonly creator: Operation | null; readonly parents: readonly Variable[]; grad: Tensor | null; }`. Document inline that `grad` is the only non-`readonly` field and is mutated only via `accumulateGrad` and `zeroGrad`.
- [x] 2.2 Define `Operation` in `types.ts` as `readonly { name: string; inputs: readonly Variable[]; backward: (rt: TensorRuntime, gradOutput: Tensor) => readonly (Tensor | null)[]; }`. JSDoc the contract: `backward` must return one entry per input position and may emit `null` at positions whose input has `requiresGrad === false`.
- [x] 2.3 Define `Parameter` as a `Variable` interface with `requiresGrad: true` and `creator: null` for documentation and grep-friendliness. Define `ParameterRegistry` as `readonly { params: Parameter[] }` (the array itself is mutated via `registerParameter`).
- [x] 2.4 Implement a module-private monotonic id counter in `variable.ts` (`let nextVariableId = 0`) and a `nextId()` helper. Document it as the only module-level mutable state in the package.
- [x] 2.5 Implement `createLeafVariable(tensor: Tensor, requiresGrad: boolean): Variable` in `variable.ts`: returns `{ id: nextId(), tensor, requiresGrad, creator: null, parents: [], grad: null }`.
- [x] 2.6 Implement `createParameter(tensor: Tensor): Parameter` in `variable.ts`: returns the same record shape with `requiresGrad: true`.
- [x] 2.7 Implement an internal `createInternalVariable(tensor: Tensor, op: Operation, requiresGrad: boolean): Variable` shared by op wrappers; **not re-exported** from `index.ts`.
- [x] 2.8 Implement `accumulateGrad(rt: TensorRuntime, target: Variable, contribution: Tensor): void` in `variable.ts`: when `target.grad === null`, assign `target.grad = contribution` directly; otherwise assign `target.grad = tensor.add(rt, target.grad, contribution)`. JSDoc: this helper and `zeroGrad` are the only writes to `Variable.grad`.

## 3. Op wrappers (forward + backward via existing tensor surface)

- [x] 3.1 `add(rt: TensorRuntime, a: Variable, b: Variable): Variable` in `ops.ts` — forward via `tensor.add(rt, a.tensor, b.tensor)`. When `a.requiresGrad || b.requiresGrad`, build an `Operation` with `name: 'add'`, `inputs: [a, b]`, and `backward(_rt, gradOutput) => [a.requiresGrad ? gradOutput : null, b.requiresGrad ? gradOutput : null]`. Otherwise return a leaf-style `Variable` (`creator: null`, `requiresGrad: false`).
- [x] 3.2 `mul(rt, a, b): Variable` — forward via `tensor.mul(rt, a.tensor, b.tensor)`. Capture `a.tensor` and `b.tensor` in the closure for backward. When at least one input requires grad: `backward(rt, gradOutput) => [a.requiresGrad ? tensor.mul(rt, gradOutput, b.tensor) : null, b.requiresGrad ? tensor.mul(rt, gradOutput, a.tensor) : null]`. Otherwise return a leaf-style `Variable`.
- [x] 3.3 `scale(rt, a, k): Variable` — forward via `tensor.scale(rt, a.tensor, k)`. When `a.requiresGrad`: `backward(rt, gradOutput) => [tensor.scale(rt, gradOutput, k)]`. Otherwise return a leaf-style `Variable`.
- [x] 3.4 `transpose(rt, a): Variable` — forward via `tensor.transpose(rt, a.tensor)` (rank-2 only; `tensor.transpose` already enforces). When `a.requiresGrad`: `backward(rt, gradOutput) => [tensor.transpose(rt, gradOutput)]`. Otherwise return a leaf-style `Variable`.
- [x] 3.5 `reshape(a: Variable, newShape: readonly number[]): Variable` — **no `rt` parameter**. Forward via `tensor.reshape(a.tensor, newShape)`; capture `a.tensor.shape` for backward. When `a.requiresGrad`: `backward(_rt, gradOutput) => [tensor.reshape(gradOutput, originalShape)]`. Otherwise return a leaf-style `Variable`. JSDoc the buffer-sharing consequence inherited from `tensor.reshape`.
- [x] 3.6 Audit `ops.ts` to confirm **no `this`**, **no `class`**, **no module-level mutable variables**; ESLint-friendly `function` declarations only. Every op uses a shared `liftOp` helper that picks `createInternalVariable` (recorded-graph branch) or `createLeafVariable` (no-grad branch) — `liftOp` defers `Operation` construction so the no-grad branch never allocates the closure.

## 4. Backward traversal

- [x] 4.1 Implement a pure topological-order helper `topologicalOrder(output)` in `backward.ts` (post-order DFS keyed by `Variable.id`, returning a `readonly Variable[]` in dependency order). Reachability is walked via `v.parents` (cached `creator?.inputs`). Each `Variable` appears exactly once even if reached via multiple paths.
- [x] 4.2 Implement `backward(rt: TensorRuntime, output: Variable): void` in `backward.ts`:
  - If `output.grad === null`, allocate a ones-filled `Tensor` of shape `output.tensor.shape` via `tensor.createFromArray(rt, new Float32Array(output.tensor.length).fill(1), output.tensor.shape)` and route through `accumulateGrad(rt, output, ones)` so the only writes to `Variable.grad` in production code remain inside `accumulateGrad` and `zeroGrad`.
  - Compute the topo order from `output`, walk it in reverse, and for each non-leaf `v` (i.e. `v.creator !== null`):
    - Call `grads = v.creator.backward(rt, v.grad)`.
    - For each `(input, grad)` pair where `grad !== null` and `input.requiresGrad`, call `accumulateGrad(rt, input, grad)`.
- [x] 4.3 Document in JSDoc that `backward` does not zero existing gradients; calling it twice on the same graph without `zeroGrad(reg)` accumulates contributions (matches PyTorch default).

## 5. ParameterRegistry helpers

- [x] 5.1 `createParameterRegistry(): ParameterRegistry` — returns `{ params: [] }`.
- [x] 5.2 `registerParameter(reg: ParameterRegistry, p: Parameter): void` — appends to `reg.params`. **No deduplication** in this scope (documented).
- [x] 5.3 `parameters(reg: ParameterRegistry): readonly Parameter[]` — returns a defensive `[...reg.params]` copy, with JSDoc noting the cost trade-off versus a direct `readonly` view.
- [x] 5.4 `zeroGrad(reg: ParameterRegistry): void` — sets every `p.grad = null` for `p` in `reg.params`. JSDoc confirms this is the only function besides `accumulateGrad` that writes `Variable.grad`.
- [x] 5.5 `totalParams(reg: ParameterRegistry): number` — sums `p.tensor.length` across all registered parameters.

## 6. Tests (Vitest)

- [x] 6.1 Added `packages/autograd/src/topology.spec.ts` with **pure** unit tests for `topologicalOrder` using fake `Variable` records. Covers: isolated leaf, linear chain, diamond graph, de-duplication when a node is referenced twice via the same parent, and the parent-before-child invariant on a deeper graph. _(5 tests, no GPU.)_
- [x] 6.2 Added `packages/autograd/src/parameter-registry.spec.ts` with **pure** unit tests for `createParameterRegistry`, `registerParameter`, `parameters`, `zeroGrad`, and `totalParams` using fake `Parameter` records. Covers: empty registry totals, append-order preservation, defensive-copy semantics of `parameters`, `zeroGrad` clearing pre-set fake `grad` values, total-element-count summation, and explicit no-deduplication behavior. _(6 tests, no GPU.)_
- [x] 6.3 Added `packages/autograd/src/ops.spec.ts` (GPU-gated via `describe.skipIf(!webgpuAvailable)`):
  - `add` / `mul` / `scale` / `transpose` / `reshape` forward outputs match the corresponding `tensor.*` direct call element-wise (within float tolerance).
  - One-step `Operation.backward` for each op against the hand-derived analytic formula.
  - Leaf-fast-path scenarios (no input requires grad ⇒ `creator === null` and forward still correct).
  - Build the test `TensorRuntime` once per `describe` block; tear down in `afterEach`. _(7 tests.)_
- [x] 6.4 Added `packages/autograd/src/backward.spec.ts` (GPU-gated):
  - `backward(rt, output)` initializes `output.grad` to ones when `null`.
  - `backward` respects a caller-supplied `output.grad`.
  - Diamond-graph parameter (`mul(a, a)`) accumulates `2a` correctly.
  - `requiresGrad: false` leaves end with `grad === null` after backward.
  - `backward` is a no-op when no input requires grad. _(5 tests.)_
- [x] 6.5 Added `packages/autograd/src/gradient-check.spec.ts` (GPU-gated): finite-difference gradient check on the documented composed graph `output = scale(rt, reshape(mul(rt, add(rt, a, b), c), [6]), 2.0)` over parameters of shape `[2, 3]`. Procedure:
  1. Run forward + backward; read back `output` and each parameter's `grad`.
  2. For each parameter element, compute the central-difference numerical gradient `(f_plus - f_minus) / (2 * ε)` of `Σ output` with `ε = 1e-3` via fresh forward passes on perturbed parameters.
  3. Assert `|analytic - numerical| ≤ 5e-3` per element.
     Test passes within tolerance for all 18 parameter elements. _(1 test.)_

## 7. Verification

- [x] 7.1 `nx build @llm-transformer-diy/autograd` — clean build, no TS errors.
- [x] 7.2 Root `pnpm lint` — clean. `pnpm test` succeeds for all 9 projects; autograd suite is **24/24** (5 topology + 6 registry pure tests, 7 ops + 5 backward + 1 gradient-check GPU-gated tests). Pre-existing tensor / gpu suites unchanged. _(GPU was available in this environment so the gated specs ran for real; they skip cleanly when no adapter is present, per the standard policy.)_
- [x] 7.3 `pnpm list --filter @llm-transformer-diy/autograd` shows `@llm-transformer-diy/tensor` as the only **runtime** dep; `@llm-transformer-diy/gpu` and `@webgpu/types` are **devDependencies** (test-setup only). `git status` reports no edits under `packages/tensor/src/` or `packages/gpu/src/`.
- [x] 7.4 Grep audit:
  - `^class\s|^export class\s` in `packages/autograd/src` — **no matches** (structural guard for the functional style).
  - `\.grad\s*=` in non-spec autograd source — matches **only** `accumulateGrad` (`variable.ts:95, 98`) and `zeroGrad` (`parameter-registry.ts:40`); other matches are read-comparisons (`=== null`) or JSDoc.
  - `^import .* from ['"]@llm-transformer-diy/gpu['"]` in non-spec autograd source — **no matches** (autograd does not import the gpu package directly; the only gpu reference in production code is a JSDoc comment in `index.ts`).
  - `\.destroy\(\)|\.mapAsync\(|\.getMappedRange\(|\.unmap\(` in `packages/autograd/src` — **no matches** (autograd does not call any `GPUBuffer` method directly).
- [x] 7.5 Public-surface smoke: `gradient-check.spec.ts` and `backward.spec.ts` collectively import every newly-exported symbol (`createParameter`, `createLeafVariable`, `add`, `mul`, `scale`, `reshape`, `backward`) and exercise them end-to-end; the pure spec exercises `createParameterRegistry`, `registerParameter`, `parameters`, `zeroGrad`, `totalParams`. The "New surface is importable" spec scenario is covered.
