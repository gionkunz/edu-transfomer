## ADDED Requirements

### Requirement: Core matmul kernel is registered and numerically verified

The implementation SHALL provide a WGSL compute kernel that computes the matrix product `C = A * B` for row-major `f32` matrices where dimensions are provided via the dispatch API (height/width/inner dimension). The kernel SHALL be registered in `KernelRegistry` under the stable identifier `core.matmul`. When WebGPU is available, automated tests SHALL compare GPU results to a CPU reference implementation for at least one small non-square shape and one square shape within trivial size limits.

#### Scenario: Registry lookup succeeds after registration

- **WHEN** bootstrap code registers core kernels including matmul on a valid `GPUDevice`
- **THEN** `KernelRegistry.get("core.matmul")` returns a compute pipeline usable for dispatch

#### Scenario: Numeric agreement with CPU reference when GPU tests run

- **WHEN** tests run with a usable WebGPU adapter and dispatch matmul with small integer-valued inputs
- **THEN** readback of `C` matches the CPU reference within a documented floating-point tolerance for `f32`

---

### Requirement: Core elementwise kernels cover add, multiply, scale, and GELU

The implementation SHALL provide WGSL compute kernels registered under stable identifiers `core.add`, `core.mul`, `core.scale`, and `core.gelu`. Each kernel SHALL consume and produce `f32` data in equal-length 1D buffers (pairwise inputs for `add` and `mul`; single input plus scalar scale factor for `scale`; single in-place-style input/output buffers for `gelu` as defined by the implementation’s documented layout). The scalar formula used for GELU SHALL be identical between WGSL and the CPU reference used in tests.

#### Scenario: Pairwise add matches CPU reference

- **WHEN** tests dispatch `core.add` on two length-`n` buffers with WebGPU available
- **THEN** the output buffer matches elementwise CPU addition within floating-point tolerance

#### Scenario: Pairwise multiply matches CPU reference

- **WHEN** tests dispatch `core.mul` on two length-`n` buffers with WebGPU available
- **THEN** the output buffer matches elementwise CPU multiplication within floating-point tolerance

#### Scenario: Scale matches CPU reference

- **WHEN** tests dispatch `core.scale` with a length-`n` buffer and a supplied scalar factor
- **THEN** the output buffer matches the CPU reference scaled vector within floating-point tolerance

#### Scenario: GELU matches CPU reference

- **WHEN** tests dispatch `core.gelu` on a length-`n` input buffer with WebGPU available
- **THEN** the output buffer matches the CPU reference GELU evaluated per element within floating-point tolerance

---

### Requirement: Core 2D transpose kernel is registered and numerically verified

The implementation SHALL provide a WGSL compute kernel that transposes an `m × n` row-major `f32` matrix stored as length `m*n`, producing length `m*n` representing `n × m`. The kernel SHALL be registered under the stable identifier `core.transpose`. When WebGPU is available, tests SHALL verify against a CPU transpose for at least one non-square `(m, n)` pair.

#### Scenario: Transpose registry and correctness

- **WHEN** core kernels are registered and tests dispatch transpose for a small matrix with WebGPU available
- **THEN** `KernelRegistry.get("core.transpose")` is defined before dispatch and the readback equals the CPU transpose reference within floating-point tolerance

---

### Requirement: WebGPU-gated tests remain non-fatal without an adapter

Automated tests that execute core kernels on the GPU SHALL follow the same policy as `gpu-bootstrap`: if no usable WebGPU adapter is available in the environment, GPU-specific tests SHALL be skipped or otherwise gated so the package test command still exits successfully.

#### Scenario: Suite passes without WebGPU

- **WHEN** the package test suite runs in an environment without a usable WebGPU adapter
- **THEN** GPU-dependent core kernel tests do not fail the suite solely due to missing GPU hardware or stack
