# autograd-gelu-op

## Purpose

Specify the **pairwise** vertical slice for GELU autograd: a new `core.gelu_backward` WGSL kernel and dispatch in `packages/gpu`, a `geluBackward` functional op in `packages/tensor`, and a `gelu` wrapper in `packages/autograd`, with CPU-reference validation and finite-difference tests. Forward `core.gelu` / `tensor.gelu` are unchanged.

## Requirements

### Requirement: GELU backward kernel and dispatch

The implementation SHALL register a WGSL compute pipeline under the stable identifier `core.gelu_backward`. The shader SHALL consume, per invocation index `i` in range, upstream gradient `grad_out[i]` and forward input `x[i]`, and SHALL write `grad_in[i] = grad_out[i] * gelu′(x[i])` where `gelu′` is the derivative of the same tanh-based GELU scalar function used by `core.gelu` (matching constants in `elementwise-gelu.wgsl` and `cpu-reference.ts`). The implementation SHALL expose `dispatchGeluBackward(ctx, registry, gradOut, inp, gradIn, elementCount)` encoding one dispatch with the documented bind-group layout.

#### Scenario: GPU output matches CPU reference

- **WHEN** tests run `dispatchGeluBackward` on equal-length buffers against `cpuGeluBackward` (or equivalent fused CPU helper) for small `n` with WebGPU available
- **THEN** the readback of `grad_in` matches the CPU reference within floating-point tolerance

---

### Requirement: Tensor op `geluBackward`

The implementation SHALL expose `geluBackward(rt, gradOutput, input)` requiring `gradOutput` and `input` to have equal shapes and to follow the same contiguity rules as other elementwise ops in `tensor-bootstrap`. The function SHALL allocate a fresh output buffer, dispatch `core.gelu_backward`, and return a new `Tensor` without mutating inputs.

#### Scenario: Op matches CPU reference

- **WHEN** `geluBackward` is called after `createFromArray` with WebGPU available
- **THEN** `toArray` of the result matches the CPU backward reference for the same inputs within tolerance

---

### Requirement: Autograd wrapper `gelu`

The implementation SHALL expose `gelu(rt, x: Variable): Variable` whose forward tensor equals `tensor.gelu(rt, x.tensor)`. When `x.requiresGrad === true`, the recorded `Operation` SHALL have `name: 'gelu'`, `inputs: [x]`, and `backward(rt, gradOutput)` returning `[tensor.geluBackward(rt, gradOutput, x.tensor)]`. When `x.requiresGrad === false`, the wrapper MAY return a leaf variable with `creator: null` and MAY omit recording an `Operation`.

#### Scenario: Backward uses input activation

- **WHEN** `backward` is invoked on the `gelu` operation with a non-zero `gradOutput`
- **THEN** the returned gradient tensor equals `tensor.geluBackward(rt, gradOutput, x.tensor)` element-wise within tolerance

#### Scenario: Finite-difference agreement on a small graph

- **WHEN** a test builds a graph that includes `gelu` on at least one `Parameter` and compares analytic gradients from `backward` to central-difference numerical gradients
- **THEN** the error is within the test’s documented tolerance
