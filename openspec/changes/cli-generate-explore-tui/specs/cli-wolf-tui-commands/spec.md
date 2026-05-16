## MODIFIED Requirements

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

## ADDED Requirements

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
