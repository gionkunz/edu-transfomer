## Context

Documentation (`docs/project.md`, `docs/guidelines/typescript.md`, `AGENTS.md`) fixes intent: Layered TS monorepo, WebGPU-backed DIY stack, forbidden high-level ML dependencies. The codebase currently has **no Node workspace** (`package.json`, Nx, pnpm) checked in — only Markdown and Cursor/OpenSpec scaffolding.

## Goals / Non-Goals

**Goals:**

- Produce a **runnable toolchain** — install, lint, test, compile — bounded to empty or trivial exports.
- Materialize **`apps/cli`** plus every **`packages/*`** entry defined in docs so downstream changes edit real folders.
- Encode **Nx + pnpm + strict TS + Vitest + ESLint/Prettier** per allowed dependency lists.
- Keep **stub surface area tiny**: one barrel file per package (no placeholder modules that mimic future internals).

**Non-Goals:**

- WGSL shaders, kernels, tokenizer training, wolf-tui dashboards, checkpoints, inference, autograd implementations.
- Installing **`webgpu`** at foundation time unless the team explicitly wants gpu package stubs to import types; default is **defer** so GPU work stays in a GPU-focused proposal.
- Production CI matrices, versioning, publishes, or changelog automation beyond what `@nx/js` minimal init adds.

## Decisions

| Decision | Rationale |
|----------|------------|
| **pnpm workspaces + Nx `@nx/js`** | Matches documented stack (`docs/project.md`); Nx selects projects without handwritten script orchestration.|
| **One shared TS base (`tsconfig.base.json`) plus per-package `tsconfig.json` references** | Prevents divergence; stubs compile identically.|
| **Vitest invoked from workspace root targeting packages** | Keeps boilerplate minimal; feature proposals can deepen per-package test layout later.|
| **No `main`/`bin` semantics until CLI proposal** aside from a **`node` entry** that prints a harmless placeholder message or `--help`** | Gives a compilable CLI target without implying command designs are final.|
| **Package dependency graph wired bottom-up**: `gpu` has no workspace deps — `tensor` depends on `gpu`, `autograd` on `tensor`, `model` on `autograd`, `training` and `inference` on `model`, `tokenizer` only where design places it vs model (tokenizer is typically parallel to training; docs show tokenizer branching from tokenizer package used by CLI/training — for skeleton, **`tokenizer` has no upward deps`; **`training`** may depend on `tokenizer`; **`apps/cli`** may depend on tokenizer + inference + training stubs only via edges that satisfy acyclic layering — avoid cycles). | Align layering with ASCII diagram — CLI top, tokenizer plugs into cli/training. Simplest skeleton: **`cli` depends on nothing or only `tokenizer` stub first** — but then training cannot import tokenizer from cli. Standard: **`cli` depends on multiple packages`; `training` deps: `model`, `tokenizer`; `inference`: `model`; `cli`: `tokenizer`, `training`, `inference` — check acyclicity. Order: gpu → tensor → autograd → model ← training,inference → tokenizer isolated? tokenizer used by training and cli → `tokenizer` no deps from stack; **`training`** → `tokenizer`, `model`; **`cli`** → `tokenizer`, `training`, `inference`; **`io`** attach to serialization — doc says checkpoint in training - could make `io` used by training or minimal leaf. For minimal edges: **`io` no workspace deps**, `training` might depend `io`. I'll note in tasks to match diagram: gpu→tensor→autograd→model; training deps model+tokenizer+io(?); inference deps model.|
| **Defer `wolf-tui` and `commander` until CLI feature proposal** | Reduces churn; stubs don’t render TUI.|
| **`packages/io`** included as sibling stub | Mentioned serially beside training in docs — keeps checkpoint work scoped later.|

## Risks / Trade-offs

- **[Risk]** Nx scaffolding noise (many config files). **Mitigation:** Use `preset` sparingly — only files required by `@nx/js` reference doc.
- **[Risk]** Over-specifying CLI UX before UX proposal. **Mitigation:** Minimal `main` exporting a function or exiting 0 — no commander.
- **[Risk]** Strict TS rejects empty-export packages. **Mitigation:** Explicit `export {}` where needed.

## Migration Plan

- **Deploy:** Implement change on `main`; no existing consumers beyond docs.
- **Rollback:** Delete generated config and stubs; revert is single commit.

## Open Questions

- Whether **`vitest`** should use project references or wildcard patterns first — pick simplest working root config unless Nx dictates otherwise upon init.
- Optional **`engines.node`** pinning to documented LTS (Node 24) vs looser ranges — defer to `.nvmrc` / `engines` consistency with team env.
