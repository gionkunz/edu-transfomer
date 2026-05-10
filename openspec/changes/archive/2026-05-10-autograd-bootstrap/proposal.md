## Why

`packages/gpu` ships forward kernels for `matmul`, `add`, `mul`, `scale`, `gelu`, `transpose`, `reduce_sum`, `reduce_max`, `softmax`, and `rmsnorm`. `packages/tensor` wraps the elementwise / matmul / transpose / reshape subset as a functional op surface (`add`, `mul`, `scale`, `gelu`, `matmul`, `transpose`, `reshape`). `packages/autograd` is still a single-symbol stub. Every layer above (`model`, `training`, `inference`) needs a **define-by-run computation graph**, a **Variable / Operation** record pair, a **topological backward** traversal, and a **Parameter / ParameterRegistry** abstraction before we can wire `Linear`, `RMSNorm`, attention, FFN, or a training step. This change lands the **first vertical slice** of the autograd engine — narrowed to the ops whose backward is **trivially derivable** from existing tensor surfaces — so subsequent changes can attach `matmul`, `gelu`, `softmax`, `rmsnorm`, `embedding`, and `cross_entropy` backward in their own pairwise diffs (per `docs/sequential-change-proposals.md` step 5).

## What Changes

- Replace the stub `packages/autograd/src/index.ts` with a real **functional** define-by-run engine. **No classes**, **no methods on records**, **no module-level mutable state**. The single intentional mutability is the `grad` slot on a `Variable` — accumulated during `backward` — documented as the only mutable field in the package.
- Add **data types** under `packages/autograd/src/`:
  - `Variable` — `{ id, tensor, requiresGrad, creator, parents, grad }` where `id` is a monotonic number for topo de-duplication, `tensor` is a `@llm-transformer-diy/tensor` `Tensor`, `creator` is the `Operation` that produced it (or `null` for leaves), `parents` is the same as `creator.inputs` (cached for traversal), and `grad: Tensor | null` is the accumulator.
  - `Operation` — `readonly { name, inputs, backward(rt, gradOutput): readonly (Tensor | null)[] }`. Each forward op constructs an `Operation` instance bound to its concrete inputs (closure captures whatever forward values backward needs, e.g. `mul` captures both input tensors).
  - `Parameter` — `Variable` with `requiresGrad === true` and `creator === null` (leaf). A `Parameter` is structurally a `Variable`; the alias exists for documentation and grep-friendliness.
  - `ParameterRegistry` — a `readonly` record `{ params: Parameter[] }` whose array is mutated only via `registerParameter`.
- Add **graph builders** as free functions: `createLeafVariable(tensor, requiresGrad)` and `createParameter(tensor)` for leaves; an internal `createInternalVariable(tensor, op, requiresGrad)` is shared by op wrappers and not exported from the package index.
- Add **autograd op wrappers** in `ops.ts`, each `(rt, ...inputs) => Variable`, **routing through the existing `@llm-transformer-diy/tensor` functional surface** for forward and constructing an `Operation` whose `backward` uses the **same tensor functions** (no new GPU kernels). Output `requiresGrad` is `true` iff any input has `requiresGrad === true`; otherwise the wrapper returns a leaf-style Variable with `creator: null` to avoid building a graph for read-only paths:
  - `add(rt, a, b)` → `tensor.add`. Backward: `(gradOut, gradOut)` (identity to each input).
  - `mul(rt, a, b)` → `tensor.mul`. Backward: `(tensor.mul(gradOut, b.tensor), tensor.mul(gradOut, a.tensor))`.
  - `scale(rt, a, k)` → `tensor.scale(rt, a.tensor, k)`. Backward: `(tensor.scale(rt, gradOut, k))`.
  - `transpose(rt, a)` → `tensor.transpose(rt, a.tensor)`. Backward: `(tensor.transpose(rt, gradOut))`.
  - `reshape(a, newShape)` → `tensor.reshape(a.tensor, newShape)` (no `rt`). Backward captures the original shape and emits `(tensor.reshape(gradOut, originalShape))`. The `Variable` is treated as `requiresGrad: a.requiresGrad`; the resulting `Variable.tensor.buffer` is shared with `a.tensor.buffer` (consistent with `tensor.reshape` semantics).
- Add **backward traversal** in `backward.ts`:
  - `backward(rt, output, options?)` — when `output.grad === null`, allocate a ones-filled tensor of `output.tensor.shape` via `@llm-transformer-diy/tensor` (`createFromArray(rt, new Float32Array(length).fill(1), shape)`) and assign it. Topo-sort all reachable variables from `output` via `creator.inputs`, traversing each variable exactly once (de-duplicated by `Variable.id`). Walk in reverse: for each non-leaf `v`, call `v.creator.backward(rt, v.grad)` and **accumulate** each returned tensor onto its corresponding input's `grad` field via `tensor.add` (or assign directly when the input's `grad` is `null`).
  - `accumulateGrad(rt, target, contribution)` — the helper used by traversal. Documented as the only function that mutates `Variable.grad`.
- Add **parameter management** in `parameter-registry.ts`:
  - `createParameterRegistry(): ParameterRegistry` — fresh empty registry.
  - `registerParameter(reg, p)` — appends; **no automatic deduplication** in this scope (caller is responsible for not double-registering).
  - `parameters(reg): readonly Parameter[]` — returns a defensive copy or a `readonly` view.
  - `zeroGrad(reg)` — sets every parameter's `.grad = null` (does **not** allocate a zero tensor; null is the explicit "no contribution yet" state used by accumulation).
  - `totalParams(reg): number` — sum of `p.tensor.length` over all registered parameters (for display / sanity).
  - `clipGradNorm(maxNorm)` is **explicitly out of scope** and lands with the optimizer change.
- Add **Vitest specs** that run on the existing `describe.skipIf(!webgpuAvailable)` gate (mirroring `tensor-bootstrap` / `core-kernels.spec.ts`):
  - **Pure unit specs (no GPU)** for topological order on a fake `Variable` graph (in-memory records using throwaway numeric IDs) and for `ParameterRegistry` ergonomics (`registerParameter`, `zeroGrad`, `totalParams`) using fake leaves whose `tensor` carries only a `length` field that the registry needs.
  - **GPU-gated specs** for each forward op wrapper: forward output equals the corresponding `tensor.*` direct call (no autograd-induced math drift), and backward gradients on a single-step graph match a hand-derived analytic value (`add` ⇒ `(1,1)`, `mul` ⇒ `(b, a)`, `scale` ⇒ `k`, `transpose` ⇒ `transpose(grad)`, `reshape` ⇒ `reshape(grad, originalShape)`).
  - **GPU-gated finite-difference gradient check** on a tiny composed graph that uses **only the trivially-derivable ops** (e.g. `loss = scale(reshape(mul(add(a, b), c), [a.length]), 2.0)`) over inputs of small shape (e.g. `[2, 3]`). Compare analytic gradients (`backward` result for each parameter) to a central-difference numerical gradient `(f(p + ε) - f(p - ε)) / (2ε)` of `Σ output` per parameter element. Tolerance: max `|analytic - numerical|` ≤ `5e-3` for `ε = 1e-3`.
- **Explicitly omit**: `matmul` backward (separate change once it routes back through `tensor.matmul` + `tensor.transpose` end-to-end), `gelu` / `softmax` / `rmsnorm` backward (each its own pairwise change with the corresponding gradient kernel or fused tensor-side derivative), `embedding`, `cross_entropy`, `optimizer kernels` (AdamW), `clip_norm`, broadcasting in autograd, N-D ops beyond what tensor exposes, non-`f32` dtypes, automatic graph-side memory reclamation (no `retain_graph` flag, no automatic `destroy` of intermediate tensors), higher-order derivatives, **and any edits to `packages/tensor`** (this change consumes the existing tensor surface unchanged).

## Capabilities

### New Capabilities

- `autograd-bootstrap`: Requirements for the first real `packages/autograd` surface — a `Variable` data record with a single mutable `grad` slot, an `Operation` interface bound to its inputs at forward time, a topological `backward` traversal that accumulates gradients via `tensor.add`, autograd op wrappers for `add` / `mul` / `scale` / `transpose` / `reshape` routing through the existing `@llm-transformer-diy/tensor` functional surface, a `Parameter` leaf type, a `ParameterRegistry` with `registerParameter` / `parameters` / `zeroGrad` / `totalParams`, and finite-difference validation on a tiny composed graph.

### Modified Capabilities

- _(none — `gpu-bootstrap`, `core-wgsl-kernels`, `reduction-norm-kernels`, and `tensor-bootstrap` behavior is unchanged. This change consumes their public surface without altering it.)_

## Impact

- **Primary**: `packages/autograd` — replace stub `src/index.ts` with real exports across new modules (`types.ts`, `variable.ts`, `backward.ts`, `parameter-registry.ts`, `ops.ts`); add Vitest coverage in `packages/autograd/src/`.
- **Dependencies**: No new npm dependencies. `@llm-transformer-diy/tensor` remains the only runtime workspace edge. `@webgpu/types` may be added as a `devDependency` (types-only) if and only if the new public API references `GPUBuffer` directly; current design keeps `GPUBuffer` opaque behind `Tensor`, so no `@webgpu/types` dep is anticipated. Layering stays `autograd → tensor → gpu` (never the inverse).
- **Downstream**: A future change wraps `tensor.matmul` with autograd backward (using `tensor.matmul` + `tensor.transpose`) in a focused diff. Subsequent pairwise changes wrap `gelu`, `softmax`, and `rmsnorm` backward as their gradient formulas / kernels land. `packages/model` / `training` / `inference` remain untouched in this change.
- **Docs**: `docs/project.md` and `docs/sequential-change-proposals.md` remain authoritative. `docs/project.md` sketches `Variable` / `Operation` / `Parameter` as TypeScript class shapes; this change deliberately picks a **functional** record-and-free-function surface consistent with `packages/tensor` (the layer immediately below). No doc edits in this change; if maintainers want the project doc updated to reflect the functional choice, that is a separate cosmetic change.
