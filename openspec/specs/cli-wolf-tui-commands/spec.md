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

The CLI SHALL implement **`llm generate`** that loads a checkpoint, runs **`@llm-transformer-diy/inference`** autoregressive generation, and SHALL stream output in a **wolf-tui** view, consistent with `docs/project.md`. By default the **`generate`** view SHALL be the **explore** view (probability inspector, recent-token highlight, and pacing/pause hotkeys defined in the explore-mode requirements below). The CLI SHALL accept a **`--no-explore`** flag that selects a minimal streaming view (tokens and basic metadata such as throughput) without the explore affordances. Both views SHALL delegate token sampling to **`@llm-transformer-diy/inference`** (no sampling math in **`apps/cli`**).

#### Scenario: Generate requires GPU

- **WHEN** the user runs **`llm generate`** and WebGPU cannot be initialized
- **THEN** the process SHALL fail with an explicit error

#### Scenario: Default view is explore

- **WHEN** the user runs **`llm generate`** without **`--no-explore`**
- **THEN** the **wolf-tui** SHALL render the explore view defined by the explore-mode requirements (probability inspector + recent-token highlight + hotkeys)

#### Scenario: --no-explore restores minimal view

- **WHEN** the user runs **`llm generate --no-explore`**
- **THEN** the **wolf-tui** SHALL render only the streamed completion text plus throughput metadata, with no probability inspector, no recent-token highlight, and no **`s`** / **`p`** hotkey behavior

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

---

### Requirement: Generate explore probability inspector

When **`llm generate`** is running in explore mode, the **wolf-tui** SHALL display a side-panel inspector showing the **next-token probability distribution** for the most recent decode step. The inspector SHALL list vocabulary entries sorted by descending probability, rendering as many rows as fit in the current terminal height (after accounting for headers and status chrome). Each row SHALL show the token id, a decoded display piece (with a documented placeholder for ids that do not decode to a complete UTF-8 fragment), and the post-temperature / post-top-k / post-repetition-penalty probability used by the sampler for that step. The inspector SHALL be omitted (gracefully, without error) when the terminal is too narrow or too short to render it usefully (implementation MAY define a sensible minimum such as 60 columns × 8 rows).

The inspector contents SHALL come from a probability vector supplied by **`@llm-transformer-diy/inference`** for that step; **`apps/cli`** SHALL NOT compute softmax, top-k, top-p, or repetition penalty logic to populate the panel.

#### Scenario: Inspector ranks vocabulary by probability

- **WHEN** **`llm generate`** is running in explore mode and a new token has been sampled
- **THEN** the inspector SHALL display vocabulary rows ordered by descending probability for that step's distribution

#### Scenario: Inspector reflects sampler-shaped distribution

- **WHEN** **`llm generate`** runs in explore mode with non-default **`--top-k`** or **`--repeat-penalty`**
- **THEN** the probabilities shown in the inspector SHALL be the sampler-shaped distribution actually used to draw the token (i.e. they SHALL reflect temperature, top-k, and repetition penalty), and the values SHALL be obtained from **`@llm-transformer-diy/inference`** rather than recomputed in **`apps/cli`**

#### Scenario: Inspector fits terminal height

- **WHEN** the user resizes the terminal smaller during generation
- **THEN** the next render SHALL render fewer inspector rows so the panel still fits without overflowing the visible area

#### Scenario: Inspector omitted on tiny terminals

- **WHEN** the terminal is narrower or shorter than the documented minimum
- **THEN** the explore view SHALL omit the inspector panel and continue streaming text without throwing

---

### Requirement: Generate explore recent-token highlight

When **`llm generate`** is running in explore mode, the **wolf-tui** SHALL render the most recently sampled token's newly-rendered text in a visually distinct color from the rest of the streamed completion (e.g. a different foreground color and/or weight). When a subsequent token is sampled, the previously highlighted text SHALL revert to the default style and the new token's text SHALL receive the highlight. The highlight SHALL be applied via **`@wolf-tui/react`** **`Text`** style props (no new dependency).

#### Scenario: Highlight follows the most recent token

- **WHEN** the model samples a new token in explore mode
- **THEN** the new text emitted by that token SHALL be styled with the highlight color while earlier text SHALL be styled in the default color

---

### Requirement: Generate explore pacing and pause controls

When **`llm generate`** is running in explore mode, the **wolf-tui** SHALL accept the following keyboard inputs and update a visible status indicator describing the current pacing state:

1. **`s`** — toggles a **slow** pacing cap of **0.25 tokens / second** (≈ 4 seconds between sampled tokens). Pressing **`s`** again returns to the **uncapped** (max-speed) pacing. Default pacing on startup is uncapped. The slow cap SHALL be enforced by waiting between decode steps in the host (CLI), not by changing any sampling math in **`@llm-transformer-diy/inference`**.
2. **`p`** — pauses generation: no further forward passes SHALL be issued until **`p`** is pressed again. While paused, the inspector SHALL continue to display the most recent step's distribution (frozen).
3. Existing exit chords (**`q`** and **`Ctrl+C`**) SHALL continue to terminate the process and SHALL NOT deadlock if pressed while paused or while the slow-mode delay is in flight; the CLI SHALL flush any pending wait so teardown can run cleanly.

The status line SHALL display the current mode (e.g. **NORMAL**, **SLOW**, or **PAUSED**) and SHALL include hotkey hints for **`s`**, **`p`**, and exit.

The **`@llm-transformer-diy/inference`** package MAY expose an awaitable per-step host hook (e.g. an awaitable callback) so the CLI can implement pacing and pause without reimplementing the decode loop or sampling logic in **`apps/cli`**; the CLI SHALL use that hook (or an equivalent host-driven mechanism) and SHALL NOT introduce new sampling code.

#### Scenario: s toggles slow pacing

- **WHEN** the user presses **`s`** while explore-mode generation is producing tokens at max speed
- **THEN** subsequent tokens SHALL appear at approximately one token every **4 seconds** until **`s`** is pressed again, and the status line SHALL reflect the SLOW mode

#### Scenario: s toggles back to max speed

- **WHEN** the user presses **`s`** while explore-mode generation is in SLOW mode
- **THEN** the pacing cap SHALL be removed and subsequent tokens SHALL stream at the model's natural rate, with the status line reflecting the NORMAL mode

#### Scenario: p pauses and unpauses

- **WHEN** the user presses **`p`** during explore-mode generation
- **THEN** no further forward passes SHALL be issued, and the inspector and streamed text SHALL stop updating; **WHEN** the user presses **`p`** again, **THEN** generation SHALL resume from the next decode step

#### Scenario: Exit while paused does not deadlock

- **WHEN** the user presses **`q`** (or **`Ctrl+C`**) while explore-mode generation is paused or sleeping in SLOW mode
- **THEN** the process SHALL tear down the TUI and exit without hanging on the pending wait
