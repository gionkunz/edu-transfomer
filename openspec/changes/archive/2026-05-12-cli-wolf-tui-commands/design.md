## Context

Step **13** in `docs/sequential-change-proposals.md` owns **`apps/cli`**. Today `main.ts` only re-exports layer markers (`createCliPlaceholder`). **`docs/project.md`** specifies the **`llm`** binary, command surface (`tokenizer`, `train`, `generate`, `inspect`), and which flows use **wolf-tui** vs stdout. Lower packages already expose **`Trainer`**, **`generateAutoregressive`**, **`BPETokenizer`**, and **`packages/io`** checkpoint helpers—the CLI must compose them without new tensor or model logic.

## Goals / Non-Goals

**Goals:**

- Single **`llm`** entrypoint with subcommands aligned to the project doc table (flags may use long names first; keep parity with documented intent).
- **Thin wiring**: parse args, open files, create GPU device, call package APIs, render TUI or print results—no new kernels, optimizers, or sampling math in `apps/cli`.
- **Fail loudly** when GPU is required but unavailable (train, generate), consistent with repo policy.
- **Vitest** (or minimal integration tests) for pure parsing / inspect path where feasible; TUI components smoke-tested or kept so thin that manual verification is acceptable with documented checklist in tasks.

**Non-Goals:**

- New checkpoint formats, tokenizer algorithms, or training algorithms.
- Full wolf-tui feature parity with arbitrary layouts beyond what `docs/project.md` sketches (extra panels, themes, remote logging).
- Cross-platform packaging (npm global, brew, etc.) beyond what Nx + local `node` already allows.

## Decisions

1. **Argument parsing — `commander`** (allowed in `docs/project.md`, listed first). Rationale: stable subcommand nesting (`tokenizer` + `train|encode|decode`), good DX; **yargs** remains an acceptable alternative if the repo standardizes elsewhere—picker should match existing `package.json` once a dependency is added.

2. **Module layout** — mirror `docs/project.md` file hints: `tokenize.ts` (or `tokenizer.ts`), `train.ts`, `generate.ts`, `inspect.ts`, plus shared `device.ts` / `errors.ts` for GPU init and user-facing error messages.

3. **Train / generate UI** — **wolf-tui** views subscribe to callbacks or polling from **`Trainer`** and **`generateAutoregressive`** (e.g. step loss, tokens/sec). Rationale: keeps UI in the app layer; training/inference packages stay headless. If the current **`Trainer`** API lacks hooks, extend **`packages/training`** in a minimal way (e.g. optional `onStep` callback)—prefer a small API addition over duplicating the loop in CLI.

4. **Tokenizer subcommand naming** — align with **`docs/project.md`** table: **`llm tokenizer train|encode|decode`** (not `tokenize` as top-level) to match the canonical command list.

5. **Binary execution** — use **`package.json`** `bin` mapping to compiled **`main.js`** (or Nx-run wrapper documented in tasks) so `pnpm exec llm` / `npx` works after build.

## Risks / Trade-offs

- **[Risk] TUI complexity vs review size** → Mitigation: start with minimal panels (status line + loss text or single chart); avoid custom layout experiments in the first merge.
- **[Risk] `Trainer` / generate APIs lack streaming hooks** → Mitigation: add narrow optional callbacks in `training` / `inference` if needed; document in tasks before large CLI refactors.
- **[Risk] WebGPU absence in CI** → Mitigation: CLI unit tests focus on argv parsing and inspect; GPU paths guarded by existing skip patterns or smoke scripts documented for local runs.

## Migration Plan

1. Land new **`main`** and subcommand modules behind the same build target; remove **`createCliPlaceholder`** export or repurpose as a test-only stub.
2. Document “how to run `llm`” in tasks (build + `node` path); no production deployment or data migration.

## Open Questions

- Exact flag names for **`--config small`** (preset id vs path to JSON): resolve during implementation against existing `ModelConfig` / io loaders in the repo.
- Whether **`llm inspect`** reads only **`model.json`** or also peeks **`model.bin`** for byte size—spec requires useful summary either way.
