# cli-wolf-tui-commands

## Purpose

Define requirements for the **`llm`** CLI in **`apps/cli`**: **`tokenizer`** (BPE train/encode/decode), **`train`** and **`generate`** with **wolf-tui** dashboards, **`inspect`** for checkpoint summaries, wiring-only orchestration (no ML logic in the app), and loud failure when WebGPU is unavailable for GPU commands.

## Requirements

### Requirement: llm binary entrypoint

The workspace SHALL provide an executable **`llm`** command (via package **`bin`** and built output) whose entrypoint dispatches to documented subcommands and exits non-zero on user-facing errors.

#### Scenario: Unknown subcommand

- **WHEN** the user invokes **`llm`** with an unknown first argument
- **THEN** the process SHALL print a concise usage or error message and exit with a non-zero status

---

### Requirement: No ML logic in CLI

The **`apps/cli`** package SHALL NOT implement tensor operations, autograd, attention, training steps, or sampling algorithms; it SHALL only orchestrate allowed dependencies and call into **`tokenizer`**, **`training`**, **`inference`**, **`io`**, **`model`**, **`gpu`**, and **`tensor`** as needed for I/O and device setup.

#### Scenario: Generate delegates sampling

- **WHEN** the user runs **`llm generate`** successfully
- **THEN** token sampling SHALL be performed by **`@llm-transformer-diy/inference`** APIs (or their documented callees), not reimplemented in **`apps/cli`**

---

### Requirement: Tokenizer subcommands

The CLI SHALL implement **`llm tokenizer train`**, **`llm tokenizer encode`**, and **`llm tokenizer decode`** consistent with `docs/project.md`: train writes tokenizer artifacts; encode and decode use a loaded tokenizer path and SHALL print results to stdout without launching the full-screen training/generate TUI.

#### Scenario: Encode prints IDs

- **WHEN** the user runs **`llm tokenizer encode`** with a valid tokenizer path and input text
- **THEN** the process SHALL print token IDs to stdout and exit zero

---

### Requirement: Train command with wolf-tui

The CLI SHALL implement **`llm train`** that runs the training pipeline via **`@llm-transformer-diy/training`** and SHALL present a **wolf-tui** dashboard during training (e.g. step, loss, throughput, LR), as described in `docs/project.md`.

#### Scenario: Train requires GPU

- **WHEN** the user runs **`llm train`** and WebGPU cannot be initialized
- **THEN** the process SHALL fail with an explicit error (no silent CPU fallback unless project policy explicitly allows it for training)

---

### Requirement: Generate command with wolf-tui

The CLI SHALL implement **`llm generate`** that loads a checkpoint, runs **`@llm-transformer-diy/inference`** autoregressive generation, and SHALL stream output in a **wolf-tui** view (tokens and basic metadata such as throughput), consistent with `docs/project.md`.

#### Scenario: Generate requires GPU

- **WHEN** the user runs **`llm generate`** and WebGPU cannot be initialized
- **THEN** the process SHALL fail with an explicit error

---

### Requirement: Inspect command

The CLI SHALL implement **`llm inspect`** that accepts a model checkpoint directory or **`model.json`** path and prints model configuration summary, parameter count, and representative layer/tensor shapes using **`@llm-transformer-diy/io`** (and related loaders) without launching wolf-tui.

#### Scenario: Inspect stdout

- **WHEN** the user runs **`llm inspect`** against a valid checkpoint metadata path
- **THEN** the process SHALL print human-readable summary lines to stdout and exit zero
