## Why

Sequential plan step **13** (`docs/sequential-change-proposals.md`) calls for a real **`llm`** entry point in **`apps/cli`**: subcommands that exercise tokenizer, training, inference, and I/O through thin wiring only. The stack already exposes programmatic APIs; without a CLI, there is no user-facing way to train, generate, or inspect checkpoints as described in **`docs/project.md`**.

## What Changes

- Replace the **`createCliPlaceholder`** stub with a **Commander** (or equivalent allowed parser)–driven **`main`** that dispatches subcommands.
- Implement **`llm tokenizer`** (`train`, `encode`, `decode`) delegating to **`@llm-transformer-diy/tokenizer`** and filesystem paths from **`docs/project.md`**.
- Implement **`llm train`** with a **wolf-tui** live dashboard (loss/LR/ETA, etc.) delegating to **`@llm-transformer-diy/training`** and checkpoint output via **`@llm-transformer-diy/io`**.
- Implement **`llm generate`** with a **wolf-tui** streaming view delegating to **`@llm-transformer-diy/inference`** (model load + generate loop).
- Implement **`llm inspect`** printing config, parameter counts, and layer shapes from checkpoint sidecar + **`model.json`** (stdout, no TUI).
- Package **`bin`** / Nx build so **`llm`** is invocable as documented; **fail loudly** when WebGPU is unavailable for commands that require GPU.

## Capabilities

### New Capabilities

- **`cli-wolf-tui-commands`**: User-facing **`llm`** binary: tokenizer subcommands, training and generate wolf-tui experiences, inspect summary, and strict “no ML logic in CLI” wiring contract.

### Modified Capabilities

- _(none — existing `openspec/specs/*` requirements for lower layers are unchanged; this change only consumes them.)_

## Impact

- **Primary**: `apps/cli` (entrypoint, subcommand modules, wolf-tui views).
- **Dependencies**: `commander` or `yargs` (per **`docs/project.md`** allowlist), `wolf-tui`; existing workspace packages `tokenizer`, `training`, `inference`, `io`, `model` (load paths only), `gpu` (device init for train/generate).
- **Bin / Nx**: `project.json` or package.json **`bin`** for **`llm`** as needed for local and CI invocation.
