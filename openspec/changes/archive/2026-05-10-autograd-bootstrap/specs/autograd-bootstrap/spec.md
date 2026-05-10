# autograd-bootstrap

## Purpose

Define requirements for the first real **`packages/autograd`** layer: a functional define-by-run engine consisting of a `Variable` data record (with a single mutable `grad` slot), an `Operation` interface bound to its inputs at forward time, a topological `backward` traversal that accumulates gradients via the existing `@llm-transformer-diy/tensor` functional surface, autograd op wrappers for the trivially-derivable ops `add` / `mul` / `scale` / `transpose` / `reshape`, a `Parameter` leaf type, and a `ParameterRegistry` with `registerParameter` / `parameters` / `zeroGrad` / `totalParams`. Builds on `gpu-bootstrap`, `core-wgsl-kernels`, `reduction-norm-kernels`, and `tensor-bootstrap`; does not own `matmul` backward, `gelu` / `softmax` / `rmsnorm` backward, `embedding`, `cross_entropy`, optimizer kernels, or `clip_norm`.

## Requirements

### Requirement: Variable record carries a Tensor, a creator, parents, and the only mutable grad slot

The implementation SHALL expose a `Variable` type whose runtime value is a record carrying a stable numeric `id`, an immutable `tensor` (a `@llm-transformer-diy/tensor` `Tensor`), an immutable `requiresGrad` boolean, an immutable `creator` (an `Operation` or `null` for leaves), an immutable `parents` array equivalent to `creator?.inputs ?? []`, and a `grad` field of type `Tensor | null`. Every field other than `grad` SHALL be `readonly` at the type level; `grad` SHALL be the only mutable field on the record. The `Variable` SHALL NOT be a class instance and SHALL NOT carry methods.

#### Scenario: Variable record fields are populated by leaf and op constructors

- **WHEN** a consumer constructs a leaf via `createLeafVariable(tensor, requiresGrad)` or an internal node via an op wrapper
- **THEN** the returned record has a numeric `id`, the supplied or computed `tensor`, the supplied or propagated `requiresGrad`, `creator` set to either `null` (leaves) or the constructed `Operation` (op outputs), `parents` equal to `creator?.inputs ?? []`, and `grad === null`

#### Scenario: Variable id is unique within a process

- **WHEN** two `Variable` records are constructed in the same Node.js process via the documented constructors or op wrappers
- **THEN** their `id` fields are not equal

#### Scenario: All non-grad fields are readonly

- **WHEN** the public surface of `@llm-transformer-diy/autograd` is inspected
- **THEN** every field on `Variable` other than `grad` is declared `readonly`

---

### Requirement: Operation captures the local backward formula bound to its inputs

The implementation SHALL expose an `Operation` type whose runtime value is a `readonly` record carrying a string `name`, a `readonly Variable[]` `inputs`, and a `backward(rt, gradOutput)` function returning a `readonly (Tensor | null)[]` of the same length as `inputs`. The `backward` function SHALL compute one gradient contribution per input position using only the supplied `TensorRuntime` and `gradOutput` plus values captured from the enclosing forward call (e.g. the input tensors themselves). The `Operation` SHALL NOT be a class instance, SHALL NOT mutate `gradOutput`, and SHALL NOT mutate any of its `inputs`' fields.

#### Scenario: Operation backward returns one entry per input position

- **WHEN** `backward(rt, gradOutput)` is called on an `Operation` whose `inputs.length === n`
- **THEN** the returned array has length `n`

#### Scenario: Operation backward emits null for inputs that do not require grad

- **WHEN** an op wrapper builds an `Operation` whose `i`-th input has `requiresGrad === false`
- **THEN** the returned array's `i`-th entry is permitted to be `null` (and the topo traversal SHALL skip accumulation at that input position)

---

### Requirement: Leaf and parameter constructors produce trainable and non-trainable variables

The implementation SHALL expose `createLeafVariable(tensor, requiresGrad)` returning a `Variable` with `creator: null`, `parents: []`, `grad: null`, and the supplied `requiresGrad`. The implementation SHALL expose `createParameter(tensor)` returning a `Parameter` (a `Variable` with `requiresGrad: true` and `creator: null`). Both functions SHALL allocate a fresh numeric `id` per call and SHALL NOT mutate the input `tensor`.

#### Scenario: createParameter sets requiresGrad to true and creator to null

- **WHEN** a consumer calls `createParameter(tensor)`
- **THEN** the returned record has `requiresGrad === true`, `creator === null`, `parents.length === 0`, and `grad === null`

#### Scenario: createLeafVariable honors the supplied requiresGrad flag

- **WHEN** a consumer calls `createLeafVariable(tensor, false)`
- **THEN** the returned record has `requiresGrad === false`, `creator === null`, `parents.length === 0`, and `grad === null`

---

### Requirement: Op wrappers route forward through the existing tensor surface and record an Operation when any input requires grad

The implementation SHALL expose autograd op wrappers `add(rt, a, b)`, `mul(rt, a, b)`, `scale(rt, a, k)`, `transpose(rt, a)`, and `reshape(a, newShape)` accepting `Variable` inputs and returning a fresh `Variable`. Forward computation SHALL route through the matching function in `@llm-transformer-diy/tensor` (`tensor.add`, `tensor.mul`, `tensor.scale`, `tensor.transpose`, `tensor.reshape`) without re-implementing any GPU kernel call. When at least one input has `requiresGrad === true`, the wrapper SHALL construct an `Operation` whose `backward` matches the local Jacobian formula and SHALL set the output's `requiresGrad` to `true`. When no input requires grad, the wrapper MAY skip recording an `Operation` and MAY return a leaf-style `Variable` (`creator: null`, `requiresGrad: false`); the forward computation SHALL still occur and produce a correct output tensor in either branch.

#### Scenario: add wrapper forward matches tensor.add and backward is identity per input

- **WHEN** `add(rt, a, b)` is called on `Variable` inputs whose tensors have equal shapes
- **THEN** the output's `tensor` matches `tensor.add(rt, a.tensor, b.tensor)` element-wise within floating-point tolerance, and (if either input requires grad) the recorded `Operation`'s `backward(rt, gradOutput)` returns `[gradOutput, gradOutput]` (each entry suppressible to `null` per the per-input requires-grad rule)

#### Scenario: mul wrapper backward emits gradOutput \* other-input

- **WHEN** `mul(rt, a, b)` is called on `Variable` inputs whose tensors have equal shapes and at least one has `requiresGrad === true`
- **THEN** the recorded `Operation`'s `backward(rt, gradOutput)` returns gradients matching `[tensor.mul(rt, gradOutput, b.tensor), tensor.mul(rt, gradOutput, a.tensor)]` (each entry suppressible to `null` per the per-input requires-grad rule)

#### Scenario: scale wrapper backward applies the same scalar

- **WHEN** `scale(rt, a, k)` is called on a `Variable` input with `requiresGrad === true`
- **THEN** the recorded `Operation`'s `backward(rt, gradOutput)` returns `[tensor.scale(rt, gradOutput, k)]`

#### Scenario: transpose wrapper backward transposes the gradient

- **WHEN** `transpose(rt, a)` is called on a rank-2 `Variable` input with `requiresGrad === true`
- **THEN** the output has shape `[a.tensor.shape[1], a.tensor.shape[0]]` and the recorded `Operation`'s `backward(rt, gradOutput)` returns `[tensor.transpose(rt, gradOutput)]`

#### Scenario: reshape wrapper backward reshapes back to the original shape and shares buffer at forward

- **WHEN** `reshape(a, newShape)` is called on a contiguous `Variable` input whose `tensor.length === productOfShape(newShape)` and `requiresGrad === true`
- **THEN** the output's `tensor.buffer` is reference-equal to `a.tensor.buffer`, the output's `tensor.shape` equals `newShape`, and the recorded `Operation`'s `backward(rt, gradOutput)` returns `[tensor.reshape(gradOutput, a.tensor.shape)]`

#### Scenario: Op wrapper outputs do not require grad when no input requires grad

- **WHEN** an op wrapper is called with all inputs having `requiresGrad === false`
- **THEN** the returned `Variable` has `requiresGrad === false`, `creator === null`, and a forward `tensor` matching the corresponding `tensor.*` call

---

### Requirement: backward seeds the output gradient and traverses the graph in topological order

The implementation SHALL expose `backward(rt, output)` that, when `output.grad === null`, allocates a ones-filled `Tensor` of shape `output.tensor.shape` and assigns it to `output.grad`. The function SHALL compute a topological order of all `Variable`s reachable from `output` via `creator?.inputs`, de-duplicated by `Variable.id`, and walk that order in reverse. For each non-leaf `v` in the reverse walk, the function SHALL call `v.creator.backward(rt, v.grad)` and accumulate each non-`null` returned gradient onto the corresponding input's `grad` field via the `accumulateGrad(rt, target, contribution)` helper. The traversal SHALL visit each reachable `Variable` exactly once even when reached via multiple graph paths.

#### Scenario: backward initializes output.grad to ones when null

- **WHEN** `backward(rt, output)` is called on a `Variable` whose `grad` is `null` and whose underlying tensor has `n` elements with WebGPU available
- **THEN** after the call returns, `output.grad` is non-`null` and an immediately-prior readback would have produced an array of `n` ones (within floating-point tolerance)

#### Scenario: backward respects a caller-supplied output.grad

- **WHEN** `backward(rt, output)` is called on a `Variable` whose `grad` is already set to a non-`null` `Tensor` of matching shape
- **THEN** the traversal uses the caller-supplied `grad` as the upstream gradient and does not overwrite `output.grad` with a ones tensor

#### Scenario: Diamond graph dispatches backward exactly once per node

- **WHEN** `backward(rt, output)` is called on a graph where one parameter feeds two op nodes that both feed `output` (a "diamond" topology)
- **THEN** each `Operation.backward` along the graph is invoked exactly once during the traversal, and the parameter's accumulated `grad` reflects the sum of contributions from both branches

#### Scenario: Leaf variables receive accumulated gradients on their grad field

- **WHEN** `backward(rt, output)` completes on a graph whose leaves have `requiresGrad === true`
- **THEN** every such leaf's `grad` field is non-`null` and contains the accumulated gradient, while leaves with `requiresGrad === false` retain `grad === null`

---

### Requirement: accumulateGrad is the only function that mutates Variable.grad

The implementation SHALL expose (or use internally and document) `accumulateGrad(rt, target, contribution)` that, when `target.grad === null`, assigns `target.grad = contribution` directly, and when `target.grad !== null`, assigns `target.grad = tensor.add(rt, target.grad, contribution)`. No other exported function in `@llm-transformer-diy/autograd` SHALL write to `Variable.grad` except `zeroGrad(reg)` (which sets it to `null`) and `accumulateGrad`. A grep audit on the package source for assignments to `.grad` SHALL match only these two helpers.

#### Scenario: First contribution is assigned without a redundant add

- **WHEN** `accumulateGrad(rt, target, contribution)` is called on a `Variable` whose `grad === null`
- **THEN** after the call, `target.grad === contribution` (reference equal) and no `tensor.add` dispatch has occurred for this accumulation step

#### Scenario: Subsequent contributions are summed via tensor.add

- **WHEN** `accumulateGrad(rt, target, contribution)` is called on a `Variable` whose `grad` is already a non-`null` `Tensor` of matching shape
- **THEN** `target.grad` is reassigned to a fresh `Tensor` whose readback equals the element-wise sum of the previous `grad` and `contribution` within floating-point tolerance

---

### Requirement: ParameterRegistry exposes registerParameter, parameters, zeroGrad, and totalParams

The implementation SHALL expose `createParameterRegistry()` returning a fresh registry record, `registerParameter(reg, p)` appending a parameter to the registry, `parameters(reg)` returning a `readonly Parameter[]` view (or defensive copy) of all registered parameters, `zeroGrad(reg)` setting every registered parameter's `.grad` to `null`, and `totalParams(reg)` returning the sum of `p.tensor.length` across all registered parameters. The registry SHALL NOT deduplicate registrations and SHALL NOT support removal in this scope.

#### Scenario: registerParameter and parameters round-trip the registered set

- **WHEN** a consumer creates a fresh registry and calls `registerParameter(reg, p)` for each of `n` distinct parameters
- **THEN** `parameters(reg)` returns an array of length `n` containing the same `Parameter` references in registration order

#### Scenario: zeroGrad clears every parameter's grad slot

- **WHEN** a consumer calls `zeroGrad(reg)` on a registry whose registered parameters have non-`null` `grad` fields
- **THEN** every registered parameter's `grad` is `null` after the call

#### Scenario: totalParams sums element counts across registered parameters

- **WHEN** a consumer registers parameters whose underlying tensors have lengths summing to `N`
- **THEN** `totalParams(reg)` returns `N`

---

### Requirement: Public surface exposes the autograd types, constructors, op wrappers, backward, and registry without breaking pre-existing exports

The implementation SHALL re-export the new types (`Variable`, `Operation`, `Parameter`, `ParameterRegistry`), the new constructors (`createLeafVariable`, `createParameter`, `createParameterRegistry`), the new op wrappers (`add`, `mul`, `scale`, `transpose`, `reshape`), the backward entry point (`backward`), the accumulator (`accumulateGrad`), and the registry helpers (`registerParameter`, `parameters`, `zeroGrad`, `totalParams`) from `@llm-transformer-diy/autograd`. Pre-existing exports of `AUTOGRAD_LAYER_MARKER` and `__autogradDependsOnTensor` MAY be retained for backward compatibility with consumers that still reference them; if retained, their values SHALL be unchanged. No new runtime npm dependency SHALL be added in this change.

#### Scenario: New surface is importable

- **WHEN** a consumer imports `Variable`, `Operation`, `Parameter`, `ParameterRegistry`, `createLeafVariable`, `createParameter`, `createParameterRegistry`, `add`, `mul`, `scale`, `transpose`, `reshape`, `backward`, `accumulateGrad`, `registerParameter`, `parameters`, `zeroGrad`, and `totalParams` from `@llm-transformer-diy/autograd`
- **THEN** every named import resolves to a defined value or type

#### Scenario: Pre-existing layer-marker exports remain importable when retained

- **WHEN** a consumer imports `AUTOGRAD_LAYER_MARKER` and `__autogradDependsOnTensor` from `@llm-transformer-diy/autograd` after this change is applied
- **THEN** the imports resolve successfully unless a follow-up cleanup change has removed them, and the values (when present) are unchanged from the prior implementation

---

### Requirement: Autograd does not import @llm-transformer-diy/gpu directly and does not allocate or destroy GPUBuffers

The implementation SHALL route every GPU allocation, dispatch, and readback through `@llm-transformer-diy/tensor`'s exported functions. The implementation SHALL NOT import any symbol from `@llm-transformer-diy/gpu` and SHALL NOT call `GPUBuffer.destroy()` or any other `GPUBuffer` method directly. Layering stays `autograd → tensor → gpu`.

#### Scenario: No direct gpu imports in the autograd package source

- **WHEN** the autograd package source is scanned for imports
- **THEN** no file under `packages/autograd/src/` imports from `@llm-transformer-diy/gpu`

#### Scenario: No direct GPUBuffer method calls in the autograd package source

- **WHEN** the autograd package source is scanned for calls on `GPUBuffer` instances
- **THEN** no file under `packages/autograd/src/` invokes `.destroy()`, `.mapAsync(...)`, `.getMappedRange(...)`, or `.unmap()` on a `GPUBuffer`

---

### Requirement: Finite-difference gradient check validates trivially-derivable ops on a composed graph

When WebGPU is available, automated tests SHALL build a tiny composed graph using only the in-scope op wrappers (`add`, `mul`, `scale`, `transpose`, `reshape`) over rank-2 input tensors of small shape (each tensor has at most `12` elements), call `backward(rt, output)` to populate analytic gradients on each input parameter, and compare those analytic gradients to a central-difference numerical gradient of `Σ output` with respect to each input element. The maximum absolute difference between analytic and numerical gradients SHALL be at most `5e-3` for `ε = 1e-3`.

#### Scenario: Analytic gradients match central-difference numerical gradients within tolerance

- **WHEN** the gradient-check test builds the documented composed graph on parameters of shape `[2, 3]`, calls `backward`, reads back each parameter's `grad`, and computes the central-difference numerical gradient of `Σ output` per parameter element with `ε = 1e-3` using fresh forward passes
- **THEN** for every parameter element, `|analytic - numerical| ≤ 5e-3`

#### Scenario: Composed graph uses only in-scope ops

- **WHEN** the gradient-check test source is scanned for op-wrapper calls
- **THEN** every call site uses one of `add`, `mul`, `scale`, `transpose`, or `reshape` from `@llm-transformer-diy/autograd` (no `matmul`, `gelu`, `softmax`, `rmsnorm`, `embedding`, or `cross_entropy` calls)

---

### Requirement: Pure unit tests cover topological order and registry ergonomics without GPU access

Pure-logic specs SHALL exercise topological-order de-duplication and `ParameterRegistry` ergonomics using fake `Variable` records whose `tensor` field is a shape stub sufficient for the helpers under test (e.g. `{ shape, length }` with no GPU buffer). These tests SHALL NOT call any function that allocates a `GPUBuffer` or dispatches a kernel.

#### Scenario: Topological order visits diamond nodes once

- **WHEN** the topo-order helper is called on a hand-built diamond graph of fake `Variable` records (one source feeding two intermediates that both feed a single sink)
- **THEN** the returned order contains every reachable `Variable` exactly once and places each `Variable` after all of its `parents`

#### Scenario: Registry helpers operate on fake parameters without GPU access

- **WHEN** the registry helpers (`registerParameter`, `parameters`, `zeroGrad`, `totalParams`) are exercised with fake `Parameter` records carrying only the fields the helpers read
- **THEN** the helpers behave per their contract (registration order preserved, `zeroGrad` clears `.grad`, `totalParams` sums `.tensor.length`) without invoking any GPU function

---

### Requirement: WebGPU-gated tests remain non-fatal without an adapter

Automated tests that execute autograd op wrappers, `backward`, or `accumulateGrad` against real GPU-backed `Tensor` records SHALL follow the same policy as `gpu-bootstrap`, `core-wgsl-kernels`, `reduction-norm-kernels`, and `tensor-bootstrap`: if no usable WebGPU adapter is available in the environment, GPU-specific tests SHALL be skipped or otherwise gated so the package test command still exits successfully.

#### Scenario: Suite passes without WebGPU

- **WHEN** the autograd package test suite runs in an environment without a usable WebGPU adapter
- **THEN** GPU-dependent autograd tests do not fail the suite solely due to missing GPU hardware or stack
