## 1. Package and layout

- [x] 1.1 Add allowed `webgpu` dependency to `packages/gpu/package.json` (pinned range consistent with `docs/project.md`); ensure no forbidden packages are pulled transitively beyond policy.
- [x] 1.2 Create source modules under `packages/gpu/src/` per `docs/project.md`: `device.ts`, `buffer.ts`, `pipeline.ts`, `kernel-registry.ts`, and `kernels/` for WGSL (start with one file, e.g. `trivial-fill.wgsl` or `trivial-add.wgsl`).
- [x] 1.3 Replace the stub-only `index.ts` with real exports (typed public surface: context factory, registry, buffer/pipeline helpers as designed); remove or repurpose `GPU_LAYER_MARKER` only if still useful for tests.

## 2. Device and buffers

- [x] 2.1 Implement adapter + `GPUDevice` + default queue acquisition; throw descriptive errors when `requestAdapter` / `requestDevice` fails (no silent `null` device on bootstrap path).
- [x] 2.2 Implement buffer helpers for sized `GPUBuffer` creation and host↔device copy suitable for f32 compute (upload + map/readback path used by tests).

## 3. Pipelines, registry, and trivial kernel

- [x] 3.1 Implement shared helpers to compile WGSL into a `GPUShaderModule` and build a `GPUComputePipeline` plus bind-group layout from a small, documented contract (workgroup size, binding indices).
- [x] 3.2 Implement `KernelRegistry` with stable string IDs, `register`, `get`, and duplicate-ID behavior per spec (default: throw).
- [x] 3.3 Author one trivial `@compute` shader (fill or elementwise add), register it under a stable ID, and add TypeScript dispatch that encodes workgroup count for a buffer-length parameter.

## 4. Tests and verification

- [x] 4.1 Update or replace `index.spec.ts`: keep a fast unit test for registry/pure logic if any; add WebGPU integration test that compares GPU output to CPU reference **only when** initialization succeeds.
- [x] 4.2 Gate WebGPU-dependent tests so that environments without an adapter **skip** cleanly (`vitest` `it.skipIf` / conditional `describe`) and the package suite still exits zero.
- [x] 4.3 Run `nx build @llm-transformer-diy/gpu`, root `pnpm test`, and `pnpm lint` from repo root; fix any strict-TS or ESLint issues introduced by WebGPU types.
