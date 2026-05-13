# cli-wolf-tui-commands

## Purpose

Define requirements for the **`llm`** CLI in **`apps/cli`**: **`tokenizer`** (BPE train/encode/decode), **`train`** and **`generate`** / **`chat`** with **wolf-tui** dashboards, **`inspect`** for checkpoint summaries, wiring-only orchestration (no ML logic in the app), and loud failure when WebGPU is unavailable for GPU commands.

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

---

### Requirement: Chat command with wolf-tui

The CLI SHALL implement **`llm chat`** that loads a GPT checkpoint (**`--model`**), loads a tokenizer (**`--tokenizer`**), initializes WebGPU/tensor runtime (**same failure posture** as **`llm generate`**), and presents an interactive **wolf-tui**: a scrollable conversation transcript plus a user input affordance (**single-line input** is sufficient). Each submitted user message SHALL produce an assistant reply whose tokens are sampled using **`@llm-transformer-diy/inference`** (**`generateAutoregressive`** or its documented callees), not reimplemented in **`apps/cli`**. **`<Enter>`** (or documented equivalent) SHALL submit the current input line; **`Ctrl+C`** SHALL exit cleanly and release CLI-owned resources (**KV cache**, TUI **`unmount`**).

#### Scenario: Chat delegates sampling

- **WHEN** the user completes an assistant reply turn in **`llm chat`** successfully
- **THEN** each sampled token SHALL be obtained via **`generateAutoregressive`** (including **`sampleLogits`** within **`@llm-transformer-diy/inference`**), not by custom RNG or argmax logic in **`apps/cli`**

#### Scenario: Chat requires GPU

- **WHEN** the user runs **`llm chat`** and WebGPU/tensor runtime initialization fails (**same criteria** as **`llm generate`**)
- **THEN** the process SHALL fail with an explicit error (**no silent generation** path)

---

### Requirement: Chat conversation formatting

Across turns, the CLI SHALL build a deterministic plaintext transcript composed of alternating **`User:`** and **`Assistant:`** segments separated by **at least one** newline (**exact separator text MAY be documented in `apps/cli` but MUST be stable for a given binary build**). Before each assistant generation, this transcript SHALL extend with the latest **`User:`** line and conclude with **`Assistant:`** (**empty completion after the label**) so **`BPETokenizer.encode`** consumes the full conversational prefix. If encoding the prefix would exceed the loaded model **`maxSeqLen`**, the process SHALL report a clear error and SHALL NOT silently truncate unless a future spec explicitly replaces this rule with truncation behavior.

#### Scenario: Multi-turn concatenation

- **WHEN** the user submits a second message after an assistant reply
- **THEN** the next **`generateAutoregressive`** call SHALL use an encoded prompt that includes the prior **`User`** and **`Assistant`** text from the session plus the new **`User`** line, preserving order

---

### Requirement: Chat generate options

The **`llm chat`** command SHALL expose **Generate-compatible** tuning flags for each assistant completion: **`max-new-tokens`**, **`temperature`**, **`top-k`**, and optional **`repeat-penalty`** / **`repeat-window`** (**same meanings** as **`llm generate`** flag documentation). Missing values SHALL reuse **`llm generate`** defaults from **`apps/cli`**.

#### Scenario: Sampling flags propagate

- **WHEN** the user invokes **`llm chat`** with non-default **`temperature`**, **`top-k`**, and **`repeat-penalty`**
- **THEN** **`generateAutoregressive`** SHALL be invoked with **`sample`** and repetition options reflecting those CLI values

---

### Requirement: Train dashboard BPE encode progress and ETA

During **`llm train`**, when the pipeline is in **`encode_corpus`** and the **wolf-tui** is showing corpus **BPE** progress, the dashboard SHALL display, in addition to the existing expand/merge line (**chars / bytes / merge iteration / token count**):

1. **Elapsed wall time** since encoding of the joined corpus began (covering **expand** and **merge**).
2. In the **merge** sub-phase: a **pessimistic upper bound** on remaining merge steps derived from the **initial post-expand token count** **`N`** as **`max(0, max(1, N − 1) − mergeIteration)`** (documented in UI as a **worst-case** bound that may over-estimate).
3. In the **merge** sub-phase, when enough timing samples exist: **merge throughput** (merges per second, smoothed) and an estimated **remaining time** in seconds computed as **remaining bound ÷ merges per second**, labeled so the user understands the estimate depends on that pessimistic bound and **actual completion may be sooner**.
4. A **macro progress indication** for the merge sub-phase (e.g. a **ProgressBar** or equivalent) clipped to **`0–100%`** proportional to **`mergeIteration ÷ max(1, N − 1)`**, with the same **worst-case** caveat.

Host-side presentation SHALL remain wiring-only (**no tokenizer algorithm** in **`apps/cli`** beyond calling existing **`encodeAsync`** hooks).

#### Scenario: Expand phase shows elapsed

- **WHEN** **`encodeAsync`** reports **`expand`** progress for the corpus
- **THEN** the train **wolf-tui** SHALL display elapsed time for the encode phase and SHALL NOT imply a definitive merge ETA (merge has not begun or rate is unavailable)

#### Scenario: Merge phase shows bound-based ETA when rate known

- **WHEN** **`encodeAsync`** reports **`merge`** progress and the host has estimated a positive merges-per-second rate from prior merge samples
- **THEN** the train **wolf-tui** SHALL display a remainder ETA derived from the pessimistic merge bound divided by that rate, with wording that communicates **worst-case / may finish sooner**
