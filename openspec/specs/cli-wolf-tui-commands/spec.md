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

---

### Requirement: Generate explore attention spotlight panel

When **`llm generate`** is running in explore mode, the **wolf-tui** SHALL display an **Attention Spotlight** panel that visualizes the strongest attention links from the current decode query position to earlier context tokens for the **focused transformer block** (see the block-selector requirement below). The spotlight SHALL be active only in explore mode; **`--no-explore`** SHALL continue to use the pre-spotlight minimal streaming view with no attention readback or spotlight chrome.

The panel SHALL list up to **10** **global** connections — the ten largest post-softmax attention weights across all heads at the focused block for the current decode step — **grouped by attention head**. Within each head group, entries SHALL be sorted by descending strength. Each entry SHALL show:

1. The **source context token** (decoded display piece, with the same documented placeholder behavior as the probability inspector for incomplete UTF-8 fragments).
2. The **source position index** in the current prefix (0-based or 1-based, but consistent within a build).
3. The **relationship strength** (see the decode-only and semantics requirement below).

The panel has two forms, chosen by the layout-tier requirement below:

- **Full** — one row per attention head: a sparkline of that head's weights over the whole visible prefix (oldest position on the left, scaled to the strongest weight of any head at this step), the head's entropy in bits, then the head's entries from the global connections. A head with no entry among the global connections SHALL show its own strongest link in a muted style instead.
- **Compressed** — the global connections grouped by head on at most two lines.

The same up-to-10 connections SHALL also appear as **colored highlights** on the streamed completion context: each connection receives a fixed rank color from a **10-color palette** (rank 1 → color 1, …, rank 10 → color 10). The highlight color for a context token in the streamed text SHALL match the color used for that connection in the spotlight panel. Highlights SHALL be rendered with **`@wolf-tui/react`** primitives (no new dependency). The panel caption SHALL name the block the displayed weights were captured from; if the user has since focused a different block, the view SHALL say so until fresh weights arrive.

Spotlight data SHALL come from attention weights supplied by **`@llm-transformer-diy/inference`** for that decode step; **`apps/cli`** SHALL NOT compute softmax, top-k selection across heads, entropy, or other attention math to populate the panel (selection and per-head summaries are exported by **`@llm-transformer-diy/inference`**).

The existing **probability inspector** behavior (ranking, sampler-shaped probabilities, omission on tiny terminals) SHALL remain unchanged when the terminal has sufficient **`rows`** and **`cols`**.

#### Scenario: Spotlight lists up to ten connections grouped by head

- **WHEN** **`llm generate`** is running in explore mode and a decode step completes with captured attention for the focused block
- **THEN** the Attention Spotlight panel SHALL list at most ten connections, grouped by head, with each entry showing source token, position, and relationship strength ordered by descending strength within each head group

#### Scenario: Context highlights match spotlight rank colors

- **WHEN** the spotlight panel displays a connection with a given rank color
- **THEN** the corresponding source context token in the streamed completion SHALL be rendered with the same rank color until the next decode step refreshes the spotlight

#### Scenario: Spotlight omitted in --no-explore mode

- **WHEN** the user runs **`llm generate --no-explore`**
- **THEN** the process SHALL NOT render the Attention Spotlight panel, block selector, or context attention highlights, and SHALL NOT request attention readback beyond the existing logits path

#### Scenario: Inspector unchanged when layout allows

- **WHEN** explore mode renders on a terminal with sufficient **`rows`** and **`cols`** for both the probability inspector and spotlight chrome
- **THEN** the probability inspector SHALL continue to rank vocabulary by descending probability and reflect sampler-shaped distributions as defined in the explore probability-inspector requirement

---

### Requirement: Generate explore attention layout tiers

When **`llm generate`** is running in explore mode, the **wolf-tui** layout SHALL reserve vertical chrome for the block selector row and the Attention Spotlight panel without breaking core streamed output. When the terminal is too small to render all explore affordances, the implementation SHALL degrade in this order:

1. **Omit the probability inspector** first (existing minimum, e.g. **`cols < 60`** or **`rows < 8`**).
2. **Compress or omit the bottom panel** (Attention Spotlight or architecture view). The full form needs one row per head (or one to two rows per block for the architecture view), so its threshold depends on the loaded model; the implementation SHALL pick the largest form that still leaves a usable body for the streamed text, and hide the panel below **40** columns.
3. **Continue streaming completion text** without error even when the spotlight is compressed or omitted.

The **`llm generate --help`** text SHALL document the **minimum recommended terminal size** for the full explore experience (block selector + spotlight + probability inspector), including the fact that smaller terminals may omit or compress panels gracefully.

#### Scenario: Spotlight compresses on short terminals

- **WHEN** the user resizes the terminal below the documented spotlight threshold during explore-mode generation
- **THEN** the next render SHALL compress or omit the Attention Spotlight panel while continuing to stream tokens without throwing

#### Scenario: Help documents minimum terminal size

- **WHEN** the user runs **`llm generate --help`**
- **THEN** the help text SHALL include minimum recommended terminal dimensions for the full explore layout (spotlight + inspector)

---

### Requirement: Generate explore block selector

When **`llm generate`** is running in explore mode, the **wolf-tui** SHALL display a **block selector** row showing the currently **focused transformer block** in the form **`Block N / L`** (where **`N`** is the 1-based layer index and **`L`** is **`nLayers`**). On startup, the focused block SHALL default to the **last layer** (**`N = L`**, i.e. layer index **`nLayers − 1`**). The user SHALL change the focused block with keyboard hotkeys:

1. **`↓`** (down arrow) — move focus to the **next** layer (increment layer index, clamped at **`L − 1`**).
2. **`↑`** (up arrow) — move focus to the **previous** layer (decrement layer index, clamped at **`0`**).

The arrows follow the architecture view, which draws block 1 at the top. **`+`** / **`=`** and **`-`** MAY remain as aliases but SHALL NOT be the documented keys, since they are not reachable on every keyboard layout.

The focused block selection SHALL **persist across subsequent decode steps** until the user changes it again; it SHALL NOT reset to the default on each sampled token. Changing the focused block SHALL refresh the spotlight and context highlights on the next render using attention for the newly selected layer (when available for the current step).

The block selector and its hotkeys SHALL be available **only in explore mode**. **`--no-explore`** SHALL NOT show the block selector or bind the arrow keys for layer cycling.

#### Scenario: Default focused block is last layer

- **WHEN** the user starts **`llm generate`** in explore mode on a model with **`L`** layers
- **THEN** the block selector SHALL display **`Block L / L`** before any layer hotkey is pressed

#### Scenario: Arrow keys change focused block

- **WHEN** the user presses **`↓`** or **`↑`** while explore-mode generation is active
- **THEN** the focused block index SHALL increment or decrement within **`0 … L − 1`**, the selector label SHALL update, and subsequent decode steps SHALL drive the spotlight from the new focused layer until changed again

#### Scenario: Focus persists across token steps

- **WHEN** the user selects a focused block and the model samples one or more additional tokens without further layer hotkeys
- **THEN** the focused block SHALL remain unchanged across those decode steps

#### Scenario: Block selector absent in --no-explore mode

- **WHEN** the user runs **`llm generate --no-explore`**
- **THEN** the block selector row SHALL NOT be rendered and the arrow keys SHALL NOT change layer focus

---

### Requirement: Generate explore attention decode-only and semantics

Attention spotlight data SHALL reflect **decode steps only**, not the initial prefill forward that processes the prompt. The **first sampled token** after prefill SHALL NOT display attention connections or context attention highlights, because no prior decode attention row exists for that step in the v1 contract; the panel MAY show a one-line note in their place so the layout does not jump. From the **second sampled token onward**, each decode step MAY supply attention for the focused block and the spotlight SHALL update accordingly.

**Relationship strength** displayed in the spotlight SHALL mean the **post-softmax attention weight** from the current query row to each key position for the focused head — a value in **`[0, 1]`** where each head's query row sums to **1** over visible prefix keys. The UI SHALL NOT label or present pre-softmax Q·K dot products as relationship strength. Values SHALL be obtained from **`@llm-transformer-diy/inference`** (and its documented model callees), not recomputed in **`apps/cli`**.

#### Scenario: No spotlight on first sampled token

- **WHEN** **`llm generate`** is running in explore mode and the model samples the **first** token after prefill
- **THEN** no attention connections and no context attention highlights SHALL appear for that step

#### Scenario: Spotlight appears from second sampled token

- **WHEN** explore-mode generation has sampled at least **two** tokens and the current step captured attention for the focused block
- **THEN** the Attention Spotlight panel and matching context highlights SHALL render for that decode step

#### Scenario: Relationship strength is post-softmax probability

- **WHEN** the spotlight displays a relationship strength for a connection
- **THEN** the value SHALL be a post-softmax attention weight in **`[0, 1]`** sourced from inference, not a raw Q·K score

---

### Requirement: Generate explore architecture view

When **`llm generate`** is running in explore mode, pressing **`f`** SHALL switch the bottom panel between the Attention Spotlight and an **architecture view**, and back. The architecture view SHALL draw the model in the order data flows through it, top to bottom, with a continuous line for the **residual stream**: the query token entering it (token vector + position vector), one section per transformer block, and the output stage (final norm, vocabulary scores, softmax, sampler settings, sampled token and its sampler probability).

Each block section SHALL show, for the current decode step:

1. **Attention activity** — one cell per attention head whose height reflects how sharply the head focuses (its largest post-softmax weight), and the size of the attention update relative to the residual stream entering the block.
2. **Feed-forward activity** — the size of the feed-forward update relative to the same stream.
3. A **logit lens** — the token the model would most likely predict if the residual stream after that block were sent through the final norm and LM head, that token's probability, and the probability at that depth of the token that was finally sampled (bar and number).

When vertical space allows, each block SHALL also name its strongest attention link (head, source token, position, weight); on shorter terminals blocks SHALL collapse to one row each, and below that to the compressed panel form. The focused block SHALL be marked.

All values SHALL be computed in **`@llm-transformer-diy/inference`** from tensors captured by **`@llm-transformer-diy/model`** in a single decode forward; **`apps/cli`** SHALL NOT compute them. Lens probabilities SHALL be the **raw** softmax (no temperature or top-k). Capture SHALL be requested only while the architecture view is the visible panel. Like attention, the trace describes decode steps only: the first sampled token has none. When the view is opened or the focused block changes while generation is paused or in slow mode, the view SHALL fetch the data for the current step without advancing generation.

#### Scenario: f toggles the architecture view

- **WHEN** the user presses **`f`** during explore-mode generation
- **THEN** the bottom panel SHALL switch between the Attention Spotlight and the architecture view, and context attention highlights SHALL be shown only with the Attention Spotlight

#### Scenario: Last block matches the model output

- **WHEN** the architecture view shows a decode step
- **THEN** the last block's lens SHALL reflect the model's final raw distribution, and earlier blocks SHALL reflect the same residual stream at earlier depths

#### Scenario: Architecture view absent in --no-explore mode

- **WHEN** the user runs **`llm generate --no-explore`**
- **THEN** the architecture view SHALL NOT be rendered, **`f`** SHALL NOT be bound, and no trace SHALL be read back

---

### Requirement: Generate stops at an end-of-text marker

**`llm generate`** SHALL end generation when the **decoded generated text** contains a stop marker, in both the explore view and **`--no-explore`**. The marker SHALL default to **`<|endoftext|>`**, SHALL be configurable with **`--stop <text>`**, and SHALL be disabled with **`--no-stop`** (generation then runs to **`--max-new-tokens`**). Matching SHALL be done on text, not token ids, because the tokenizer has no dedicated end-of-text token and may split the marker differently depending on context. Only generated tokens SHALL be considered; a marker inside the prompt SHALL NOT stop generation.

The marker itself SHALL NOT be shown in the streamed text. The explore view SHALL state that generation finished at the marker; its panels MAY continue to describe the last decode step (the one that completed the marker). **`@llm-transformer-diy/inference`** SHALL offer hosts a way to end generation from the per-step hook; the text matching itself lives in **`apps/cli`**.

#### Scenario: Story ends at the marker

- **WHEN** the model writes **`<|endoftext|>`** during **`llm generate`** with default options
- **THEN** generation SHALL stop without running another decode forward, and the marker SHALL NOT appear in the displayed text

#### Scenario: --no-stop continues

- **WHEN** the user passes **`--no-stop`**
- **THEN** generation SHALL continue past any marker until **`--max-new-tokens`**, and the marker SHALL be displayed as ordinary text
