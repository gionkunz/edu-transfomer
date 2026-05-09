## 1. WGSL sources and loading

- [x] 1.1 Add `.wgsl` files under `packages/gpu/src/kernels/` for `matmul`, elementwise (`add`, `mul`, `scale`, `gelu`), and `transpose`, each with a documented `@compute` entry point and binding layout matching design (storage/read-only-storage/uniform as needed).
- [x] 1.2 Add thin TS loaders (mirror `load-trivial-fill.ts`) that return WGSL source strings for each shader module.

## 2. Registration and dispatch

- [x] 2.1 Extend bootstrap registration (new module or `bootstrap-kernels.ts` sibling) to compile pipelines via `createSimpleComputePipeline` and register IDs `core.matmul`, `core.add`, `core.mul`, `core.scale`, `core.gelu`, `core.transpose` on `KernelRegistry` (duplicate-ID behavior unchanged).
- [x] 2.2 Implement typed dispatch helpers for each kernel (dimensions, element counts, scalar factor) that create bind groups, encode compute passes, and submit on `GpuContext.queue`; document row-major layout and uniform packing in code comments.

## 3. CPU references and tests

- [x] 3.1 Implement small pure-TS CPU references for matmul, elementwise ops, transpose, and GELU using the **same** scalar formula as WGSL (share constants/helpers where practical).
- [x] 3.2 Add Vitest specs that compare GPU readbacks to CPU references for small shapes when WebGPU initializes; use `describe.skipIf` / `it.skipIf` consistent with existing gpu integration tests when no adapter.
- [x] 3.3 Cover registry lookup for each `core.*` ID after registration (can live beside existing `kernel-registry.spec.ts` patterns).

## 4. Verification

- [x] 4.1 Run `nx build @llm-transformer-diy/gpu`, root `pnpm test`, and `pnpm lint`; fix strict TS / ESLint issues from new modules.
