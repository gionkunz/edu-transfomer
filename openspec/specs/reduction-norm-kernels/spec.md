# reduction-norm-kernels

## Purpose

Define requirements for the **forward** reduction and normalization WGSL kernels in `packages/gpu`: row-wise `reduce_sum`, row-wise `reduce_max`, numerically-stable `softmax`, and `rmsnorm` — registered under stable IDs and validated against CPU references for small shapes. Builds on `gpu-bootstrap` and `core-wgsl-kernels`; does not own backward kernels, `causal_mask`, tensor-layer wrappers, or any consumer above `packages/gpu`.

## Requirements

### Requirement: Row-wise reduce_sum kernel is registered and numerically verified

The implementation SHALL provide a WGSL compute kernel that computes the row-wise sum of a row-major `f32` matrix of shape `[rows, cols]` (length `rows * cols`), producing an `f32` vector of length `rows`. The kernel SHALL be registered in `KernelRegistry` under the stable identifier `core.reduce_sum`. The accompanying TypeScript dispatch helper SHALL accept `rows` and `cols` as explicit parameters and SHALL throw synchronously if either is non-positive, before any kernel dispatch. When WebGPU is available, automated tests SHALL compare GPU readback to a CPU reference for at least one non-square shape.

#### Scenario: Registry lookup succeeds after registration

- **WHEN** the existing core-kernel registration runs on a valid `GPUDevice`
- **THEN** `KernelRegistry.get("core.reduce_sum")` returns a compute pipeline usable for dispatch

#### Scenario: Numeric agreement with CPU reference when GPU tests run

- **WHEN** tests dispatch `core.reduce_sum` on a row-major `[rows, cols]` input with WebGPU available
- **THEN** the readback length-`rows` vector matches `cpuReduceSum(data, rows, cols)` within floating-point tolerance

#### Scenario: Dispatch helper rejects non-positive dimensions

- **WHEN** the `reduce_sum` dispatch helper is called with `rows <= 0` or `cols <= 0`
- **THEN** the call throws an `Error` synchronously, before any GPU kernel is dispatched

---

### Requirement: Row-wise reduce_max kernel is registered and numerically verified

The implementation SHALL provide a WGSL compute kernel that computes the row-wise maximum of a row-major `f32` matrix of shape `[rows, cols]`, producing an `f32` vector of length `rows`. The kernel SHALL be registered under the stable identifier `core.reduce_max`. For any non-empty row, the kernel SHALL produce the true maximum across all `cols` elements. The dispatch helper SHALL throw synchronously on non-positive dimensions, so the kernel is never invoked with `cols === 0`.

#### Scenario: Registry lookup succeeds after registration

- **WHEN** core-kernel registration runs on a valid `GPUDevice`
- **THEN** `KernelRegistry.get("core.reduce_max")` returns a compute pipeline usable for dispatch

#### Scenario: Numeric agreement with CPU reference when GPU tests run

- **WHEN** tests dispatch `core.reduce_max` on a row-major `[rows, cols]` input with WebGPU available
- **THEN** the readback length-`rows` vector matches `cpuReduceMax(data, rows, cols)` within floating-point tolerance

#### Scenario: Dispatch helper rejects non-positive dimensions

- **WHEN** the `reduce_max` dispatch helper is called with `rows <= 0` or `cols <= 0`
- **THEN** the call throws an `Error` synchronously, before any GPU kernel is dispatched

---

### Requirement: Softmax kernel is numerically stable and matches the CPU reference

The implementation SHALL provide a WGSL compute kernel that computes a numerically-stable row-wise softmax of a row-major `f32` matrix `[rows, cols]`, producing a same-shape output. The kernel SHALL be registered under the stable identifier `core.softmax`. The kernel SHALL implement the textbook two-pass algorithm per row — subtract the per-row maximum before exponentiating, then divide by the per-row sum of exponentials — so that rows containing large-magnitude values do not overflow `f32` and produce `Infinity` or `NaN`. When WebGPU is available, tests SHALL verify both numerical agreement with a CPU reference on a typical row and finite-output stability on a deliberately-large-magnitude row.

#### Scenario: Registry lookup succeeds after registration

- **WHEN** core-kernel registration runs on a valid `GPUDevice`
- **THEN** `KernelRegistry.get("core.softmax")` returns a compute pipeline usable for dispatch

#### Scenario: Softmax matches CPU reference for a typical row

- **WHEN** tests dispatch `core.softmax` on a `[rows, cols]` input whose values fit comfortably in `f32` with WebGPU available
- **THEN** the readback matches `cpuSoftmax(data, rows, cols)` within floating-point tolerance and every output row sums to `1` within tolerance

#### Scenario: Softmax stays finite on a row with large-magnitude values

- **WHEN** tests dispatch `core.softmax` on a single-row input whose values include magnitudes near `1000` (e.g. `[1000, 1001, 999]`)
- **THEN** every element of the readback is finite (`Number.isFinite` returns true) and matches `cpuSoftmax` within floating-point tolerance

---

### Requirement: RMSNorm kernel is registered, takes a learned gamma, and is parameterized by epsilon

The implementation SHALL provide a WGSL compute kernel that computes row-wise RMSNorm of a row-major `f32` matrix `[rows, cols]` with a learned scale vector `gamma` of length `cols`, producing a same-shape output. The kernel SHALL be registered under the stable identifier `core.rmsnorm`. For each row `i` the kernel SHALL compute `meanSq = (sum over j of input[i,j]²) / cols`, `inv = 1 / sqrt(meanSq + epsilon)`, and `out[i,j] = input[i,j] * inv * gamma[j]`. The dispatch helper SHALL accept `epsilon` as a parameter (default `1e-6`) and SHALL throw synchronously when `gamma.length !== cols` or when `rows <= 0` or `cols <= 0`.

#### Scenario: Registry lookup succeeds after registration

- **WHEN** core-kernel registration runs on a valid `GPUDevice`
- **THEN** `KernelRegistry.get("core.rmsnorm")` returns a compute pipeline usable for dispatch

#### Scenario: RMSNorm matches CPU reference for a non-trivial gamma

- **WHEN** tests dispatch `core.rmsnorm` on a `[rows, cols]` input with a non-trivial `gamma` of length `cols` and `epsilon = 1e-6` with WebGPU available
- **THEN** the readback matches `cpuRmsNorm(data, gamma, rows, cols, 1e-6)` within floating-point tolerance

#### Scenario: Dispatch helper rejects mismatched gamma length

- **WHEN** the `rmsnorm` dispatch helper is called with a `gamma` buffer whose declared length differs from `cols`
- **THEN** the call throws an `Error` synchronously, before any GPU kernel is dispatched

#### Scenario: Default epsilon is 1e-6

- **WHEN** the `rmsnorm` dispatch helper is called without explicitly passing `epsilon`
- **THEN** the kernel uses `epsilon === 1e-6`, matching `docs/project.md`'s RMSNorm definition

---

### Requirement: Public surface exposes new kernel IDs and dispatch helpers without breaking existing exports

The implementation SHALL re-export the four new kernel identifiers (`CORE_KERNEL_REDUCE_SUM`, `CORE_KERNEL_REDUCE_MAX`, `CORE_KERNEL_SOFTMAX`, `CORE_KERNEL_RMSNORM`), the four new dispatch helpers (`dispatchReduceSum`, `dispatchReduceMax`, `dispatchSoftmax`, `dispatchRmsNorm`), and the four new CPU references (`cpuReduceSum`, `cpuReduceMax`, `cpuSoftmax`, `cpuRmsNorm`) from `@llm-transformer-diy/gpu`. The new identifiers SHALL appear in the `CORE_KERNEL_IDS` array. No existing export SHALL be renamed or removed by this change.

#### Scenario: New IDs are present in CORE_KERNEL_IDS

- **WHEN** the public surface of `@llm-transformer-diy/gpu` is inspected
- **THEN** `CORE_KERNEL_IDS` includes `'core.reduce_sum'`, `'core.reduce_max'`, `'core.softmax'`, and `'core.rmsnorm'` in addition to the previously-shipped IDs

#### Scenario: Existing exports are unchanged

- **WHEN** consumers import any previously-shipped symbol (`CORE_KERNEL_MATMUL`, `dispatchMatMul`, `cpuMatMul`, `registerCoreKernels`, etc.) from `@llm-transformer-diy/gpu`
- **THEN** the import resolves to the same value with the same call signature as before this change

---

### Requirement: Single registerCoreKernels call wires every core ID

The implementation SHALL extend the existing `registerCoreKernels(device, registry)` function so that, after a single call, `KernelRegistry.get(id)` resolves successfully for every identifier in `CORE_KERNEL_IDS`, including the four new IDs added by this change. The duplicate-registration policy from `gpu-bootstrap` SHALL apply unchanged (a second call against the same registry surfaces as the registry's existing duplicate-key behavior).

#### Scenario: All core IDs resolve after a single registration call

- **WHEN** `registerCoreKernels(device, registry)` is called once on a fresh registry with a valid `GPUDevice`
- **THEN** `KernelRegistry.get(id)` returns a compute pipeline for every `id` in `CORE_KERNEL_IDS`

---

### Requirement: WebGPU-gated tests remain non-fatal without an adapter

Automated tests that execute reduction or normalization kernels on the GPU SHALL follow the same policy as `gpu-bootstrap`, `core-wgsl-kernels`, and `tensor-bootstrap`: if no usable WebGPU adapter is available in the environment, GPU-specific tests SHALL be skipped or otherwise gated so the package test command still exits successfully.

#### Scenario: Suite passes without WebGPU

- **WHEN** the package test suite runs in an environment without a usable WebGPU adapter
- **THEN** GPU-dependent reduction / softmax / rmsnorm tests do not fail the suite solely due to missing GPU hardware or stack
