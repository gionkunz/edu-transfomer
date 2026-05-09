## Why

The monorepo foundation compiles, but `packages/gpu` is only a stub. Nothing can allocate device buffers, compile WGSL, or dispatch compute—the prerequisite for every layer above `gpu`. This change lands the **first vertical slice** of WebGPU plumbing so later work can attach kernels and tensor dispatch without reworking device lifecycle.

## What Changes

- Implement **`packages/gpu`** as named in `docs/project.md`: adapter/device initialization, default queue, `GPUBuffer` helpers (create, upload, download, sizing), optional **buffer pool** or reuse hooks if they reduce churn for upcoming kernels.
- Add **compute pipeline + bind group** construction helpers shared by future shaders.
- Introduce a **kernel registry**: deterministic string IDs, compiled pipeline lookup, and a single place to register WGSL entry points.
- Ship **one trivial WGSL compute kernel** (e.g. fill a buffer or elementwise add on two buffers) executed end-to-end from TypeScript to prove the path.
- Add the allowed **`webgpu` (npm)** dependency on `@llm-transformer-diy/gpu` and wire TypeScript types per project guidance.
- Enforce **fail loudly** when no suitable adapter/device is available on code paths that request a GPU context (align with `docs/project.md` GPU-only stance); tests that run in environments **without** WebGPU MUST **skip** cleanly so root `pnpm test` / Nx smoke still satisfies `monorepo-skeleton`.
- **Explicitly omit**: real matmul/elementwise/attention kernels, tensor types and op dispatch, autograd, training, tokenizer, CLI behavior—those follow the sequencing in `docs/sequential-change-proposals.md`.

## Capabilities

### New Capabilities

- `gpu-bootstrap`: WebGPU device/queue lifecycle, buffer and pipeline/bind-group utilities, WGSL load/compile integration, kernel registry with stable naming, and a minimal compute shader proving dispatch—without owning full kernel inventory or tensor semantics.

### Modified Capabilities

- _(none — no requirement-level changes to existing main specs; gpu tests defer or skip when WebGPU is unavailable so the monorepo smoke path stays valid.)_

## Impact

- **Primary**: `packages/gpu` (new source modules under `src/` per `docs/project.md`, first WGSL file under `src/kernels/` or equivalent), `package.json` dependency on `webgpu`.
- **Tests**: Vitest coverage in `packages/gpu` for registry/pipeline wiring; integration test runs the trivial kernel when an adapter is present.
- **Downstream**: `packages/tensor` and later layers can depend on a stable GPU bootstrap API; no changes required to `tensor` in this change beyond possibly documented import paths.
- **Docs**: `docs/project.md` remains authoritative; no new markdown deliverables beyond OpenSpec artifacts unless synced later.
