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

This repo uses an OpenSpec-style loop under **`.cursor/`**:

- **`.cursor/commands/`** — `opsx-explore`, `opsx-propose`, `opsx-apply`, `opsx-archive` for the change lifecycle.
- **`.cursor/skills/`** — `openspec-explore`, `openspec-propose`, `openspec-apply-change`, `openspec-archive-change` skills; read the relevant `SKILL.md` before driving that phase.

For substantial features or behavior changes, follow that workflow instead of ad-hoc edits when the user expects spec-driven work.

## Cursor rules

Project-specific agent rules (if present) live in **`.cursor/rules/`** as `.mdc` files. Prefer them over duplicating policy here.

## Practical defaults for agents

- Match existing naming, imports, and test style in whichever package you touch.
- Prefer tests (e.g. Vitest) when behavior is easy to specify; respect the testing approach described in `docs/project.md`.
- Avoid new markdown/docs files unless the user asks for documentation deliverables.

When in doubt, re-read **`docs/project.md`** before refactoring or proposing dependencies.
