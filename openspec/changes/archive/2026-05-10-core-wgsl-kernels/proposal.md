## Why

GPU bootstrap proves dispatch with one trivial kernel, but nothing implements the numeric primitives that dense layers and simple graphs need. This change adds **matmul**, **elementwise** ops, and **transpose** as WGSL compute kernels in `packages/gpu`, registered and testable against CPU references for small shapes—unlocking the tensor layer proposal next without expanding scope into reductions or attention.

## What Changes

- Add WGSL compute shaders and TypeScript dispatch glue for:
  - **Matrix multiply** (`matmul`): naive or straightforward algorithm acceptable; **no** tiled/shared-memory optimization in this change (defer to a performance pass per sequencing guidance).
  - **Elementwise**: `add`, `mul`, `scale` (uniform scalar multiply), and **`gelu`** (Gaussian Error Linear Unit forward), matching the sequencing note “as needed” for upcoming FFN paths.
  - **Transpose** for 2D matrices (layout sufficient for upcoming `Linear`-style usage).
- Register each kernel under **deterministic IDs** in the existing kernel registry (same duplicate-registration policy as `gpu-bootstrap`).
- Add **numerical tests** comparing GPU outputs to a small CPU reference implementation for bounded shapes; gate WebGPU-dependent tests like existing gpu package tests when no adapter is available.
- **Explicitly omit**: `packages/tensor` types and op routing, autograd, softmax/reductions/RMSNorm, optimizer kernels, CLI changes, and performance-focused matmul variants—per `docs/sequential-change-proposals.md` ordering.

## Capabilities

### New Capabilities

- `core-wgsl-kernels`: Requirements for matmul, elementwise (`add`, `mul`, `scale`, `gelu`), and 2D transpose as WGSL compute kernels in `packages/gpu`, including registry IDs and CPU-reference validation for small inputs.

### Modified Capabilities

- _(none — `gpu-bootstrap` behavior stays as specified; this change adds kernels and dispatch helpers without altering bootstrap requirements.)_

## Impact

- **Primary**: `packages/gpu` — new WGSL under `src/kernels/` (or project layout from `docs/project.md`), registration in bootstrap/kernel wiring, Vitest specs with CPU references.
- **Dependencies**: No new npm dependencies expected beyond existing `webgpu` usage; stay within `docs/project.md` allowlist.
- **Downstream**: `packages/tensor` can bind to these kernel IDs in a later change; no tensor package edits required here beyond compatibility awareness in task notes if any shared types emerge (prefer keeping helpers gpu-local).
