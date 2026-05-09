## Why

`packages/gpu` now ships a usable device, kernel registry, and core numeric kernels (`core.matmul`, `core.add`, `core.mul`, `core.scale`, `core.gelu`, `core.transpose`), but `packages/tensor` is still a single-symbol stub. Every layer above (autograd, model, training, inference) needs a **stable `Tensor` primitive** that owns shape/strides, holds GPU-resident `f32` data, and routes high-level ops to the existing kernel registry — without forcing each call site to touch `GPUBuffer`, uniforms, or workgroup math directly. This change lands the **first vertical slice** of `packages/tensor` so subsequent autograd work can attach to a stable surface and `gpu` stays a leaf concern.

## What Changes

- Implement **`packages/tensor`** in a **functional style**: plain immutable data structures plus free functions, no classes, no methods, no global state.
- Add **data types** under `packages/tensor/src/`:
  - `Tensor` — a `readonly` record `{ shape, strides, dtype: 'f32', device: 'gpu', buffer: GPUBuffer, length }` (matches the `TensorData` sketch in `docs/project.md`).
  - `TensorRuntime` — a `readonly` record `{ ctx: GpuContext, registry: KernelRegistry }` threaded explicitly into every op that allocates or dispatches; **no global default**, no setter / getter pair.
  - `DType = 'f32'`, `Device = 'gpu'` literal-union types in `types.ts`.
- Add **pure shape helpers** in `shape.ts` (no GPU, no I/O): `computeContiguousStrides(shape)`, `productOfShape(shape)`, `isContiguous(t)`, `sameShape(a, b)`, `assertRank(t, n)`, `assertSameShape(a, b)`.
- Add **constructors** as free functions in `tensor.ts`: `createZeros(rt, shape)` and `createFromArray(rt, data, shape)` (each returns a fresh `Tensor` whose `GPUBuffer` is owned by the caller).
- Add **lifecycle** functions: `toArray(rt, t)` (async readback) and `destroy(t)` (calls `GPUBuffer.destroy()`); both are free functions, not methods.
- Add **op functions** in `ops.ts`, each `(rt, ...inputs) => Tensor`, returning a freshly-allocated output and never mutating inputs:
  - `matmul(rt, a, b)` → `core.matmul` (rank-2 only; rejects mismatched inner dims).
  - `add(rt, a, b)` and `mul(rt, a, b)` → `core.add` / `core.mul` (equal shape; no broadcasting yet).
  - `scale(rt, a, k)` → `core.scale`.
  - `gelu(rt, a)` → `core.gelu`.
  - `transpose(rt, a)` → `core.transpose` (rank-2 only).
- Add **`reshape(a, newShape)`** as a **pure metadata-only** function that shares the input `GPUBuffer`; it takes **no runtime** because it neither allocates nor dispatches. It rejects element-count mismatches and non-contiguous strides.
- Add **Vitest specs** that compare GPU readbacks to small CPU references (reusing `cpuMatMul`, `cpuAdd`, etc. from `@llm-transformer-diy/gpu`). GPU-dependent tests SHALL skip when no adapter is available, matching `gpu-bootstrap` / `core-wgsl-kernels` policy.
- **Explicitly omit**: classes / methods / `this`, any global default-runtime singleton, autograd / `Variable` / backward, broadcasting, N-D generality beyond rank-2 ops, non-`f32` dtypes, CPU device backend, random init kernels (`randn`, `xavier`), reductions, softmax/RMSNorm/embedding/cross-entropy ops, training utilities, tokenizer, model, and CLI surface.

## Capabilities

### New Capabilities

- `tensor-bootstrap`: Requirements for the first real `packages/tensor` surface — a `Tensor` data record with GPU-resident `f32` storage, a `TensorRuntime` data record threaded explicitly to ops, pure shape helpers, free-function constructors / lifecycle / ops for `matmul` / `add` / `mul` / `scale` / `gelu` / `transpose`, a pure `reshape`, and CPU-reference validation for small shapes when WebGPU is available.

### Modified Capabilities

- _(none — `gpu-bootstrap` and `core-wgsl-kernels` behavior is unchanged; this change consumes their public surface without altering it.)_

## Impact

- **Primary**: `packages/tensor` — replace stub `src/index.ts` with real exports across new modules (`types.ts`, `shape.ts`, `tensor.ts`, `ops.ts`); add Vitest coverage in `packages/tensor/src/`.
- **Dependencies**: Adds **`@webgpu/types`** as a tensor `devDependency` (types-only, no runtime code) so the global `GPUBuffer` type resolves in tensor's compilation context — required because the `Tensor` interface holds a `GPUBuffer` field. No new runtime dependencies; `@llm-transformer-diy/gpu` remains the only production workspace edge, layering stays `tensor → gpu` (never the inverse). `@webgpu/types` fits the `docs/project.md` allowlist spirit ("infrastructure or ergonomics only").
- **Downstream**: `packages/autograd` can build `Variable` / `Operation` records around the new `Tensor` data and ops in a later change. `packages/model` and `packages/training` remain untouched.
- **Docs**: `docs/project.md` and `docs/sequential-change-proposals.md` remain authoritative. Note that `docs/project.md` sketches a `Tensor.zeros(shape)` / `tensor.matmul(other)` style; this change deliberately picks a **functional** surface (`createZeros(rt, shape)` / `matmul(rt, a, b)`) instead. No doc edits in this change; if maintainers want the project doc updated to reflect the functional choice, that is a separate cosmetic change.
