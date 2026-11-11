# Agent guidance — LLM Transformer DIY

Use this file as the anchor for autonomous work in this repository. Prefer **narrow, purposeful changes** aligned with the project goals below.

## What this project is

Educational transformer stack in **TypeScript**: custom tensors, WebGPU/WGSL kernels, autograd, model layers, training and inference — implemented from scratch rather than delegated to ML frameworks.

Canonical detail lives in **`docs/project.md`** (architecture, dependency policy, layout). Conceptual narration is in **`docs/transformer-explained.md`**. TypeScript guidance is in **`docs/guidelines/typescript.md`**.

## Non-negotiables

- **Own the stack**: tensor math, GPU dispatch, autograd, attention/FFN/norms, training/inference — no shortcuts via high-level ML libraries.
- **GPU path**: WebGPU via the allowed stack; treat “GPU required, fail loudly” as the default mindset from `docs/project.md`.
- **Dependency policy**: respect the allowed/forbidden lists in `docs/project.md`. If a new dependency is needed, it must match the “infrastructure or ergonomics only” spirit — not hidden compute or model logic.

## Architecture mental model

Layering is bottom-up; a package should only depend on layers **below** it:

`gpu` → `tensor` → `autograd` → `model` → `training` / `inference` → `apps/cli`

When adding code, place it in the lowest layer that can own the concern, and keep public surfaces small and explicit.

## Repository workflow (OpenSpec)

This repo uses an OpenSpec-style loop. The same commands and skills are installed for Cursor (**`.cursor/`**) and Claude Code (**`.claude/`**):

- **Commands** — `opsx-explore`, `opsx-propose`, `opsx-apply`, `opsx-sync`, `opsx-archive` for the change lifecycle (**`.cursor/commands/`**, **`.claude/commands/opsx/`**).
- **Skills** — `openspec-explore`, `openspec-propose`, `openspec-apply-change`, `openspec-sync-specs`, `openspec-archive-change` (**`.cursor/skills/`**, **`.claude/skills/`**); read the relevant `SKILL.md` before driving that phase.

For substantial features or behavior changes, follow that workflow instead of ad-hoc edits when the user expects spec-driven work.

## Cursor rules

Project-specific agent rules (if present) live in **`.cursor/rules/`** as `.mdc` files. Prefer them over duplicating policy here.

## Practical defaults for agents

- Match existing naming, imports, and test style in whichever package you touch.
- Prefer tests (e.g. Vitest) when behavior is easy to specify; each package/app owns a `vitest.config.ts`, and Nx runs them via `@nx/vitest:test` (`nx run <project>:test` or `nx run-many -t test --all`). Stay aligned with `docs/project.md`.
- **CLI smoke (local, WebGPU)**: `npm run build`. To exercise the **deployable** bundle under `dist/apps/cli` (after `@nx/node` prune): `npx nx run @llm-transformer-diy/cli:prune`, then `npm run cli:install-dist`, then `node dist/apps/cli/main.js --help`. For dev, `npx nx run @llm-transformer-diy/cli:serve` is typical.
- **CLI global install (`npm link`)**: The runnable package lives under **`dist/apps/cli`** (not `apps/cli`). Use **`npm run cli:link-global`** (runs prune, **`npm install --prefix dist/apps/cli`**, then **`npm link`** in that folder). Remove with **`npm run cli:unlink-global`**. Manual equivalent: `cd dist/apps/cli && npm install && npm link`. To consume from another local project: **`npm install /absolute/path/to/repo/dist/apps/cli`** (after installing deps in `dist/apps/cli`, same as prune flow).
- **Incremental compile**: `npx nx run <project>:build` emits that project’s artifacts under `dist/` per its `project.json` targets.
- Avoid new markdown/docs files unless the user asks for documentation deliverables.

When in doubt, re-read **`docs/project.md`** before refactoring or proposing dependencies.
