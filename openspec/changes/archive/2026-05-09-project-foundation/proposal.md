## Why

The repository documents architecture and dependency policy (`docs/project.md`) but lacks an executable workspace, so nothing can compile, test, or accept feature-sized changes cleanly. Establishing a **minimal monorepo skeleton** anchors later OpenSpec proposals to a real layered layout without front-loading transformer or GPU logic.

## What Changes

- Add root **pnpm** workspace configuration listing `apps/*` and `packages/*`.
- Initialize **Nx** with `@nx/js` so builds and targets have a shared orchestration surface.
- Create **canonical package placeholders** (`gpu`, `tensor`, `autograd`, `model`, `training`, `inference`, `tokenizer`, plus `apps/cli` and `packages/io` if already named in docs) as **minimal compilable stubs** (barrel/export only — no kernels, ops, or model code).
- Add a **baseline TypeScript** setup: strict mode, ESM modules, aligned with project guidance.
- Add **minimal test and lint scaffolding** at the workspace root (Vitest + ESLint/Prettier as allowed infra) wired so `pnpm test` / `pnpm lint` can run trivially against stubs.
- **Explicitly omit** WGSL shaders, tensor math, autograd, training loops, tokenizer training, wolf-tui views, checkpoint formats — those remain separate changes.

## Capabilities

### New Capabilities

- `monorepo-skeleton`: Workspace layout (pnpm + Nx), layered package placeholders, TS baseline, and root-quality gates only — no transformer feature behavior.

### Modified Capabilities

- *(none — no existing specs under `openspec/specs/`.)*

## Impact

- **New**: Root `package.json`, `pnpm-workspace.yaml`, Nx config files, shared `tsconfig` bases, Vitest and ESLint/Prettier config, stub `apps/cli` and `packages/*`.
- **Dependencies**: Allowed infra only (`webgpu` may appear only if needed for gpu package stubs — prefer **deferring** `webgpu` to a dedicated GPU/OpenSpec change to keep this foundation dependency-light; design will decide).
- **Docs**: Existing `docs/project.md` remains source of truth; foundation implementation must align with layering and forbidden-dependency lists.
- **Downstream proposals**: Feature work attaches to concrete packages rather than rewriting repo shape.
