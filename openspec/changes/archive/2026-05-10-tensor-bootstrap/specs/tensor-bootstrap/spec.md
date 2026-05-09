# tensor-bootstrap

## Purpose

Define requirements for the first real **`packages/tensor`** layer: a functional `Tensor` data record holding GPU-resident `f32` storage, a `TensorRuntime` data record threaded explicitly to op functions, pure shape helpers, free-function constructors / lifecycle / ops for `matmul` / `add` / `mul` / `scale` / `gelu` / `transpose`, a pure metadata `reshape`, and CPU-reference validation for small shapes when WebGPU is available. Builds on `gpu-bootstrap` and `core-wgsl-kernels`; does not own autograd, broadcasting, reductions, softmax, or random init.

## Requirements

### Requirement: Tensor record holds GPU-resident f32 storage with row-major contiguous strides

The implementation SHALL expose a `Tensor` type whose runtime value is a plain immutable record carrying `shape`, `strides`, `dtype = 'f32'`, `device = 'gpu'`, a GPU storage `buffer`, and a cached element `length`. `strides` SHALL be row-major and contiguous in this scope (matching `computeContiguousStrides(shape)`); non-contiguous views are out of scope. The `Tensor` SHALL NOT be a class instance, SHALL NOT carry methods, and SHALL NOT carry mutable internal state.

#### Scenario: Tensor record fields are populated by constructors

- **WHEN** a consumer constructs a tensor via the documented constructor functions on a valid `TensorRuntime`
- **THEN** the returned record has `dtype === 'f32'`, `device === 'gpu'`, `length === productOfShape(shape)`, and `strides` matching `computeContiguousStrides(shape)`

---

### Requirement: TensorRuntime is threaded explicitly to allocating and dispatching functions

The implementation SHALL expose a `TensorRuntime` type carrying a `GpuContext` and a `KernelRegistry`. Every function that allocates a GPU buffer or dispatches a compute kernel SHALL accept a `TensorRuntime` as its first parameter. The implementation SHALL NOT export any module-level mutable default-runtime state, setter, or getter; consumers SHALL hold and pass a `TensorRuntime` value explicitly.

#### Scenario: Constructors and ops require a runtime parameter

- **WHEN** a consumer calls a constructor (`createZeros`, `createFromArray`) or an op function (`matmul`, `add`, `mul`, `scale`, `gelu`, `transpose`)
- **THEN** the function signature requires a `TensorRuntime` as its first argument

#### Scenario: No global default runtime exists

- **WHEN** the public surface of `@llm-transformer-diy/tensor` is inspected
- **THEN** no `setDefaultTensorRuntime`, `getDefaultTensorRuntime`, or equivalent global-state helper is exported

---

### Requirement: Pure shape helpers compute and validate shape and strides without GPU access

The implementation SHALL expose pure helper functions for computing row-major contiguous strides, computing the product of a shape, testing whether a tensor is contiguous, comparing two tensor shapes for equality, and asserting rank or shape invariants. These helpers SHALL NOT touch the GPU and SHALL NOT mutate their inputs. Assertion helpers SHALL throw an `Error` with a descriptive message when the invariant fails.

#### Scenario: Contiguous strides are row-major

- **WHEN** `computeContiguousStrides([2, 3, 4])` is evaluated
- **THEN** the result equals `[12, 4, 1]`

#### Scenario: Product of an empty shape is one

- **WHEN** `productOfShape([])` is evaluated
- **THEN** the result equals `1`

#### Scenario: Negative dimensions are rejected

- **WHEN** `productOfShape` is called with any dimension less than zero
- **THEN** the call throws an `Error` whose message identifies the offending dimension

#### Scenario: Shape and rank assertions throw with both shapes on mismatch

- **WHEN** `assertSameShape` is called on tensors of differing shapes (or `assertRank` with the wrong expected rank)
- **THEN** the call throws an `Error` whose message includes both shapes (or both ranks)

---

### Requirement: Constructors allocate GPU buffers and upload data deterministically

The implementation SHALL expose `createZeros(rt, shape)` and `createFromArray(rt, data, shape)` constructor functions that allocate a fresh `GPUBuffer` of the appropriate byte length and return a `Tensor` record. `createZeros` SHALL ensure the underlying buffer is zero-filled before returning. `createFromArray` SHALL throw when the input `Float32Array` length does not match the product of the requested shape, before allocating any GPU resources.

#### Scenario: createFromArray round-trips data through the GPU

- **WHEN** a consumer calls `createFromArray(rt, data, shape)` followed by `toArray(rt, t)` with WebGPU available
- **THEN** the readback `Float32Array` is element-wise equal (within float tolerance) to the original `data`

#### Scenario: createZeros yields a zero buffer

- **WHEN** a consumer calls `createZeros(rt, shape)` followed by `toArray(rt, t)` with WebGPU available
- **THEN** every element of the readback equals `0`

#### Scenario: createFromArray rejects length mismatch

- **WHEN** `createFromArray` is called with `data.length !== productOfShape(shape)`
- **THEN** the call throws an `Error` referencing both lengths and does not allocate a GPU buffer for the requested shape

---

### Requirement: Lifecycle uses a free destroy function and an async readback

The implementation SHALL expose `destroy(t)` as a free function that releases the underlying `GPUBuffer` and `toArray(rt, t)` as the only async surface in the package, returning a fresh `Float32Array` populated with the tensor's contents. Neither `destroy` nor `toArray` SHALL be a method on `Tensor`. The implementation SHALL NOT track a custom alive flag; subsequent use of a destroyed tensor SHALL surface as a WebGPU validation error rather than a tensor-layer custom error.

#### Scenario: destroy releases the underlying buffer

- **WHEN** a consumer calls `destroy(t)` once on a previously allocated tensor
- **THEN** the underlying `GPUBuffer.destroy()` has been invoked exactly once

#### Scenario: toArray is the only async surface

- **WHEN** the public surface of `@llm-transformer-diy/tensor` is inspected
- **THEN** `toArray` is the only exported function whose return type is `Promise<...>`

---

### Requirement: Op functions allocate fresh outputs and route to core kernels

The implementation SHALL expose op functions `matmul(rt, a, b)`, `add(rt, a, b)`, `mul(rt, a, b)`, `scale(rt, a, k)`, `gelu(rt, a)`, and `transpose(rt, a)`. Each op SHALL allocate a freshly created `GPUBuffer` for its output, dispatch to the matching kernel registered under the `core.*` ID in `KernelRegistry`, and return a new `Tensor` record without mutating its inputs. `matmul` and `transpose` SHALL accept only rank-2 inputs and SHALL throw on any other rank. `add` and `mul` SHALL require equal-shape inputs (no broadcasting in this scope) and SHALL throw on mismatch synchronously, before any kernel dispatch.

#### Scenario: matmul matches the CPU reference for a non-square shape

- **WHEN** `matmul(rt, a, b)` is dispatched on rank-2 tensors of shapes `[M, K]` and `[K, N]` with WebGPU available
- **THEN** the readback of the result equals `cpuMatMul(aData, bData, M, N, K)` from `@llm-transformer-diy/gpu` within floating-point tolerance

#### Scenario: Elementwise ops match CPU references

- **WHEN** `add`, `mul`, `scale`, or `gelu` is dispatched on equal-shape inputs with WebGPU available
- **THEN** the readback of the result equals the corresponding `cpuAdd` / `cpuMul` / `cpuScale` / `cpuGelu` reference within floating-point tolerance

#### Scenario: transpose matches the CPU reference for a non-square matrix

- **WHEN** `transpose(rt, a)` is dispatched on a rank-2 tensor of shape `[rows, cols]` with WebGPU available
- **THEN** the result has shape `[cols, rows]` and the readback equals `cpuTranspose(aData, rows, cols)` within floating-point tolerance

#### Scenario: matmul throws on inner-dimension mismatch before dispatch

- **WHEN** `matmul(rt, a, b)` is called with rank-2 inputs whose inner dimensions do not match
- **THEN** the call throws an `Error` synchronously, before any kernel is dispatched

#### Scenario: add and mul throw on shape mismatch before dispatch

- **WHEN** `add(rt, a, b)` or `mul(rt, a, b)` is called with tensors of differing shapes
- **THEN** the call throws an `Error` synchronously, before any kernel is dispatched

#### Scenario: transpose throws on rank ≠ 2 before dispatch

- **WHEN** `transpose(rt, a)` is called on a tensor of any rank other than `2`
- **THEN** the call throws an `Error` synchronously, before any kernel is dispatched

---

### Requirement: Reshape is a pure metadata function that shares the input buffer

The implementation SHALL expose `reshape(a, newShape)` as a free function with no `TensorRuntime` parameter. The function SHALL return a new `Tensor` record whose `buffer` field is reference-equal to the input's `buffer`, with `shape` set to `newShape` and `strides` set to `computeContiguousStrides(newShape)`. The function SHALL throw when `productOfShape(newShape)` differs from the input's `length` and SHALL throw when the input is not contiguous (with a message identifying the limitation as scoped to `tensor-bootstrap`).

#### Scenario: Reshape shares the underlying buffer

- **WHEN** `b = reshape(a, newShape)` is called on a contiguous tensor `a` whose shape product matches `newShape`
- **THEN** `b.buffer === a.buffer`, `b.shape` equals `newShape`, and `b.length === a.length`

#### Scenario: Reshape rejects element-count mismatch

- **WHEN** `reshape(a, newShape)` is called with `productOfShape(newShape) !== a.length`
- **THEN** the call throws an `Error` whose message identifies both lengths

#### Scenario: Reshape rejects non-contiguous inputs

- **WHEN** `reshape(a, newShape)` is called on a tensor whose `strides` differ from `computeContiguousStrides(a.shape)`
- **THEN** the call throws an `Error` indicating that non-contiguous strides are not supported in this scope

---

### Requirement: WebGPU-gated tests remain non-fatal without an adapter

Automated tests that execute tensor constructors or op functions on the GPU SHALL follow the same policy as `gpu-bootstrap` and `core-wgsl-kernels`: if no usable WebGPU adapter is available in the environment, GPU-specific tests SHALL be skipped or otherwise gated so the package test command still exits successfully.

#### Scenario: Suite passes without WebGPU

- **WHEN** the package test suite runs in an environment without a usable WebGPU adapter
- **THEN** GPU-dependent tensor tests do not fail the suite solely due to missing GPU hardware or stack
