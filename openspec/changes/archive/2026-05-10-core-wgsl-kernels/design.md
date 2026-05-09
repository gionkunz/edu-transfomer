## Context

`packages/gpu` already initializes devices, builds simple compute pipelines (`createSimpleComputePipeline`), and registers the trivial `bootstrap.fill-pi` kernel (`docs/project.md` layering; `gpu-bootstrap` spec). The sequencing doc places **core numeric kernels** next so later tensor dispatch has real compute behind it without pulling autograd or tensor shape algebra into this change.

## Goals / Non-Goals

**Goals:**

- Ship WGSL **matmul**, **elementwise** (`add`, `mul`, `scale`, `gelu`), and **2D transpose** registered with stable string IDs on the existing `KernelRegistry`.
- Provide thin TypeScript helpers to compile/register (alongside or extending `bootstrap-kernels.ts` pattern) and dispatch with explicit dimensions/strides documented at call sites.
- Validate each kernel against a **small CPU reference** in Vitest when WebGPU is available; retain **skip-if-no-GPU** behavior consistent with current gpu tests.

**Non-Goals:**

- Tiled/shared-memory matmul, fused kernels, or microbenchmark tuning (separate hardening change).
- `packages/tensor`, autograd, softmax/reductions, optimizers, tokenizer, CLI.
- Non-`f32` dtypes and batched ND generality beyond what matmul/transpose need for the chosen interfaces (keep APIs explicitly 2D where stated).

## Decisions

1. **Kernel ID namespace** — Use a shared prefix such as `core.*` (e.g. `core.matmul`, `core.add`, `core.mul`, `core.scale`, `core.gelu`, `core.transpose`) so IDs stay deterministic and grep-friendly next to `bootstrap.*`.

2. **Matmul algorithm** — Implement a **correctness-first** kernel (e.g. one invocation covers the output matrix with a straightforward accumulation loop in WGSL). Document maximum practical sizes for CI only informally; avoid premature optimization.

3. **Elementwise shape contract** — Treat elementwise kernels as **equal-length 1D buffers** (`n` floats): `add`/`mul` read two inputs and write one output; `scale` reads one input, applies scalar from a **uniform** (or tiny uniform buffer), writes output; `gelu` reads one buffer, writes one buffer. Callers pass `elementCount`; tensor strides arrive later.

4. **Transpose contract** — Row-major `m × n` matrix as length `m*n` buffer; output length `m*n` representing `n × m`. Metadata (`m`, `n`) via uniforms matching the matmul pattern.

5. **Bindings and pipelines** — Reuse `createSimpleComputePipeline` with explicit binding lists per shader (storage vs read-only-storage vs uniform per WGSL access). Keep one bind group (group 0) unless a shader forces otherwise.

6. **GELU definition** — Use a single documented scalar formula shared by WGSL and CPU reference (e.g. standard `f32` GELU with `tanh`) so tests lock numerical behavior.

**Alternatives considered:** merging kernels into one mega-shader (rejected: harder to test and register independently); generic N-D strided dispatch now (rejected: belongs in tensor layer).

## Risks / Trade-offs

- **Naive matmul performance** → Acceptable for this milestone; sequencing explicitly splits optimization later.
- **WGSL compile / driver variance** → Mitigate with tolerant float comparisons (small ` rtol/atol`) on references and small matrices only.
- **Binding layout churn when tensor lands** → Mitigate by keeping helpers narrow and documenting that tensor dispatch may wrap or replace raw dispatch entry points without renaming registry IDs.

## Migration Plan

Not applicable: additive kernels and tests only; no rollout or data migration.

## Open Questions

- Whether `scale` should accept the scalar only via uniform struct vs dedicated small buffer (prefer uniform struct aligned to WGSL layout rules for simplicity).
