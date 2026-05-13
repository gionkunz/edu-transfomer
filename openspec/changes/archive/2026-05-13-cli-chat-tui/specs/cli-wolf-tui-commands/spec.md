# cli-wolf-tui-commands

## ADDED Requirements

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
