## 1. Root workspace

- [x] 1.1 Add `pnpm-workspace.yaml` (or equivalent) declaring `apps/*` and `packages/*` per `monorepo-skeleton`.
- [x] 1.2 Add repository root `package.json` with `private: true`, `packageManager`, and placeholder scripts wired to lint, test, and build orchestration targets that will succeed on stubs.

## 2. Nx and project graph

- [x] 2.1 Initialize `@nx/js` workspace metadata (`nx.json`, `package.json` Nx deps) following current Nx package-based conventions.
- [x] 2.2 Declare an Nx project for `apps/cli` and every `packages/*` folder so graph/list-targets discovers each unit.

## 3. TypeScript baseline

- [x] 3.1 Add shared `tsconfig.base.json` with strict compilation options and ESM interop aligned to project docs.
- [x] 3.2 Add leaf `tsconfig.json` entries for each workspace project extending the base configuration.

## 4. Canonical package stubs

- [x] 4.1 Create `packages/gpu` with minimal `src/index.ts` barrel (no GPU runtime deps; no `webgpu` package unless a follow-up explicitly requires it).
- [x] 4.2 Create `packages/tensor` with a workspace dependency on the gpu package — use one consistent `@scope/name` naming scheme declared at root setup (match future publish story or stay unscoped internal names).
- [x] 4.3 Create `packages/autograd`, `packages/model`, `packages/tokenizer`, `packages/io`, `packages/training`, and `packages/inference` with stub barrels wired to upstream workspace deps consistent with layering (no inverted edges).
- [x] 4.4 Confirm no forbidden ML dependencies appear in manifests (cross-check lists in `docs/project.md`).

## 5. CLI application stub

- [x] 5.1 Scaffold `apps/cli` with typed entry exporting a noop or trivial main suitable for compilation (defer `wolf-tui` / `commander` to later proposals).
- [x] 5.2 Point Nx/Vitest/tsconfig at the CLI entry so `apps/cli` compiles referencing allowed stub packages without cycles.

## 6. Lint, format, tests

- [x] 6.1 Add ESLint + Prettier configuration at workspace root enforcing TypeScript conventions and documenting ignore patterns for build outputs if needed.
- [x] 6.2 Configure Vitest (root or per-project minimal) so at least one passing stub test executes under `pnpm test`.

## 7. Smoke verification

- [x] 7.1 Run documented root scripts `{lint,test,build}` sequentially with exit code zero in a clean install (no GPU, no corpus).
- [x] 7.2 Capture any developer notes only in-code comments sparingly — avoid new prose docs unless correcting factual drift versus `docs/project.md`.
