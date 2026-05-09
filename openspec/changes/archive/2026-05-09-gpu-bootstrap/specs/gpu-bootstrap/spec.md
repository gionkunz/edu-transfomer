# gpu-bootstrap

## Purpose

Define requirements for the first real **`packages/gpu`** layer: WebGPU device access, buffer and pipeline utilities, a kernel registry, and one trivial compute shader proving dispatch—without owning tensor math or a full kernel catalog.

## ADDED Requirements

### Requirement: GPU bootstrap exposes device and default queue

The `gpu-bootstrap` implementation SHALL provide an API to obtain a WebGPU adapter and device suitable for compute, along with the default command queue for that device. When no suitable adapter or device can be created, the API SHALL fail with a clear error (throw or rejected promise) that indicates GPU initialization failed rather than returning a silent `null` device.

#### Scenario: Successful initialization

- **WHEN** a consumer calls the documented initialization API in an environment where WebGPU reports an available adapter and `requestDevice` succeeds
- **THEN** the consumer receives a device handle and its default queue ready to submit compute work

#### Scenario: Missing GPU

- **WHEN** a consumer calls the documented initialization API and no adapter can be requested or device creation fails
- **THEN** the API SHALL surface an explicit failure indicating that GPU bootstrap is unavailable

---

### Requirement: Buffer utilities support allocation and host-visible transfer

The implementation SHALL provide helpers to create `GPUBuffer` instances sized for a given byte length and to copy data between host `ArrayBuffer` / `TypedArray` views and device buffers in a way appropriate for compute shaders (including readback paths needed for tests).

#### Scenario: Upload and read back

- **WHEN** a test or caller allocates a buffer of known size, writes Float32 (or chosen dtype) host data into it, dispatches a kernel that mutates the buffer, and invokes the documented readback path
- **THEN** the host observes updated bytes consistent with the kernel (validated numerically for the bundled trivial kernel)

---

### Requirement: Compute pipeline and bind group construction are centralized

The implementation SHALL expose shared helpers to create compute pipelines from WGSL and to build bind groups from pipeline layouts, so future kernels do not duplicate bind layout boilerplate.

#### Scenario: Pipeline creation from WGSL

- **WHEN** caller provides WGSL source defining a valid `@compute` entry point matching the helper’s contract
- **THEN** the helper returns a `GPUComputePipeline` usable for dispatch on the same device

---

### Requirement: Kernel registry maps stable IDs to compiled pipelines

The implementation SHALL include a registry that maps deterministic string identifiers to compiled compute pipelines (and any associated metadata required for dispatch, documented alongside the registry). Registering the same ID twice SHALL be treated as an error or explicitly documented as replace semantics; the default SHALL be to throw on duplicate registration.

#### Scenario: Lookup after registration

- **WHEN** a kernel is registered under a given ID and the registry is queried for that ID
- **THEN** the caller receives the pipeline associated with that ID

#### Scenario: Duplicate registration

- **WHEN** a second registration is attempted for an ID that is already registered under the default policy
- **THEN** the implementation SHALL reject the operation in a predictable way (throw or return error) as documented

---

### Requirement: Bundled trivial compute kernel proves dispatch

The implementation SHALL ship at least one WGSL compute shader (e.g. buffer fill or pairwise add) registered in the kernel registry and dispatched from TypeScript such that numerical output can be verified against a CPU reference for small inputs.

#### Scenario: End-to-end numeric check when WebGPU is available

- **WHEN** tests run in an environment where WebGPU initialization succeeds
- **THEN** executing the trivial kernel produces outputs that match the chosen reference implementation within floating-point tolerance appropriate for `f32`

---

### Requirement: Default workspace tests remain passable without WebGPU

Tests executed as part of the normal workspace test command for this package SHALL NOT fail solely because WebGPU is absent. WebGPU-dependent integration tests SHALL be skipped or otherwise gated so that maintainers without a GPU stack still get a green run; this does not relax the “fail loudly” rule for non-test bootstrap APIs.

#### Scenario: No adapter in test environment

- **WHEN** the package test suite runs in an environment without a usable WebGPU adapter
- **THEN** WebGPU-specific tests are skipped or not scheduled, and the suite still exits successfully
