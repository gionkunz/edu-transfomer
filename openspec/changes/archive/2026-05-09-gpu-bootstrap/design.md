## Context

`docs/project.md` and `docs/sequential-change-proposals.md` position **`packages/gpu`** as the bottom layer: WebGPU via the allowed `webgpu` npm package, hand-written WGSL, and **GPU-first / fail loudly** when no device is available. The archived **project-foundation** change left `@llm-transformer-diy/gpu` as an empty stub (`GPU_LAYER_MARKER` only). Downstream packages (`tensor`, then autograd, model, …) need a **stable, small public surface** for device lifetime, buffers, pipeline/bind-group construction, and a **registry** so tensor dispatch can look up kernels by name without each op recompiling shaders.

Constraints from root specs: **`monorepo-skeleton`** still requires default root **lint → test → build** to complete without assuming physical GPU hardware; gpu integration tests must **skip** (or use CI-only adapters) so that smoke path stays green.

## Goals / Non-Goals

**Goals:**

- Provide **one obvious entry** to obtain an adapter-backed `GPUDevice` + default `GPUQueue` (or a small context object wrapping both) suitable for Node WebGPU.
- Implement **buffer utilities**: sized allocation, optional **staging** for host↔device copies, and readback where tests need numerical verification.
- Implement **pipeline helpers**: create compute pipeline from WGSL source/module, build bind groups from layouts with explicit visibility rules.
- Implement a **kernel registry**: string keys → compiled `GPUComputePipeline` (and metadata such as workgroup layout conventions documented in code), with deterministic naming conventions for future ops.
- Land **one minimal WGSL kernel** (recommended: **fill** or **add** two buffers) and a Vitest that runs it **when** WebGPU is available.
- Document how **“no adapter”** is surfaced (thrown error with actionable message) for production/bootstrap APIs, distinct from test skip behavior.

**Non-Goals:**

- Matmul, softmax, reductions, or any **training/tensor** logic (later changes).
- **Buffer pool** policy tuning beyond a minimal hook or simple pool—no production allocator yet unless trivial.
- **Shader bundling** strategy for the whole kernel tree (only one shader + loader pattern now).
- Changing **Nx graph** or other packages beyond `packages/gpu` except adding the `webgpu` dependency and exports.

## Decisions

| Decision                                                                                             | Rationale                                                                                                                                                                                                            |
| ---------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Public API as small factory + registry** (e.g. `createGpuContext()` / `getKernel()`)               | Keeps `tensor` from importing WebGPU globals everywhere; one place to enforce adapter limits and feature flags.                                                                                                      |
| **WGSL as template strings or `.wgsl` files** loaded via `readFileSync` / `import.meta.url` in tests | Matches educational repo clarity; avoid bundler magic until needed. Prefer `fs` + `fileURLToPath` for Node or inline strings for the first kernel to minimize path issues—pick one and use it consistently in tasks. |
| **`webgpu` dependency only on `packages/gpu`**                                                       | Respects layering; `tensor` will depend on `gpu` later, not on `webgpu` directly.                                                                                                                                    |
| **Kernel IDs are stable snake-case or dotted `namespace.op` strings**                                | Avoids collisions when the registry holds dozens of pipelines; document convention in registry module.                                                                                                               |
| **Fail loud on `requestAdapter` / `requestDevice` failure** for bootstrap `create*` APIs             | Aligns with project policy; callers that need soft handling can catch.                                                                                                                                               |
| **Vitest: `describe.skip` or runtime skip when `navigator.gpu` / `create()` unavailable**            | Preserves monorepo smoke without weakening product semantics for real call sites.                                                                                                                                    |

## Risks / Trade-offs

- **[Risk]** CI runners often lack WebGPU → integration test never runs in default pipelines. **Mitigation:** Document optional SwiftShader / Deno / GPU-enabled job later; keep correctness check local/optional.
- **[Risk]** Dawn/WebGPU API drift between Node versions. **Mitigation:** Pin `webgpu` version range per `docs/project.md`; note supported Node in package README comment or design follow-up.
- **[Risk]** Registry grows into a god-object. **Mitigation:** Single module now; split by domain in later kernel batches per sequencing doc.

## Migration Plan

- **Deploy:** Implement on main; replace stub `src/index.ts` with real exports; add `webgpu` to `packages/gpu/package.json`.
- **Rollback:** Revert `packages/gpu` commits; no database or external state.

## Open Questions

- Whether the first WGSL file lives as **`src/kernels/trivial-fill.wgsl`** vs inline string—tasks should pick one for consistency with future `kernels/` tree in `docs/project.md`.
- Exact shape of **`GpuContext`** (class vs plain object): prefer minimal read-only fields (`device`, `queue`, `registry`) to avoid over-abstraction.
