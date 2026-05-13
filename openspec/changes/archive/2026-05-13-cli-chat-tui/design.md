## Context

- **`apps/cli`** already exposes **`llm generate`**: **`runGenerateCommand`** (**`generate.tsx`**) loads a GPT checkpoint (**`loadGptModelFromCheckpoint`**), runs **`generateAutoregressive`** with a **wolf-tui** **`GenView`** and streams decoded text incrementally (**`onSampledToken`** + **`BPETokenizer.decode`** on the growing id list), respecting the “no sampling in **`apps/cli`**” rule.
- **`generateAutoregressive`** today insists on an **empty KV cache** (**`seqLen === 0`**) each call (full **prefill** over **`promptIds`**). Efficient **decode-only continuation** across turns is **not** required here; conversational context is modeled by feeding a growing **plaintext transcript** converted to IDs each assistant turn (**full prefill** every turn). Educational trade-off: quadratic cost in transcript length versus minimal **`inference`** churn.

## Goals / Non-Goals

**Goals:**

- **`llm chat`** fullscreen **wolf-tui**: scrollable transcript + single-line input, **Enter** to send, assistant reply streams inline (same **`onSampledToken`** UX pattern as **`generate`**).
- **Stable transcript format**: fixed role delimiters (**e.g.** newline-separated **`User:`** / **`Assistant:`** blocks) appended deterministically each turn before encode, so checkpoints don’t rely on brittle freeform prompts.
- **CLI flags** aligned with **`generate`** for **`--model`**, **`--tokenizer`**, **`--max-new-tokens`**, **`--temperature`**, **`--top-k`**, **`--repeat-penalty`**, **`--repeat-window`** (same validation rules as **`cli-program`** for **`generate`**).
- Loud failure when WebGPU/tensor runtime init fails (**same posture** as **`train`**/**`generate`**).
- Reuse **`primeTuiTerminalCanvas`**, **`createTensorRuntime`**, **`resolvePackedCheckpointPaths`**, **`loadGptModelFromCheckpoint`**, **`createGptKvCache`**, **`destroyGptKvCache`** patterns from **`generate`**.

**Non-Goals:**

- Persisting chats, slash commands, configurable system prompts, model switching mid-session.
- Updating **`generateAutoregressive`** to reuse a **non-empty** KV cache across user turns (**future optimization**).
- Non-TUI (**stdio**) chat mode.
- **`docs/project.md`** dependency policy changes.

## Decisions

1. **Conversation state as text transcript** → Each turn concatenates `\nUser: …\nAssistant: ` onto the rolling history; **encode** builds **`promptIds`**; call **`generateAutoregressive`** with a **fresh** **`createGptKvCache`** instance; **`destroyGptKvCache`** in **`finally`**. Matches current cache contract and avoids new **`packages/inference`** surface for this proposal.

2. **Streaming decode** → Mirror **`generate.tsx`**: never decode single streaming ids alone; accumulate generated ids appending after the prompt id run, **`decode(prompt + generated, { utf8: 'replace' })`**, and display only the suffix after the transcript prefix length (handles BPE UTF-8 boundaries consistent with **`generate`**).

3. **TUI structure** → **Ink**/**wolf-tui** layout: **header** (title + hint), **scroll region** (transcript), **input row** (current draft). **Enter** submits; **Ctrl+C** exits (reuse **`exitOnCtrlC: true`** on **`render`** like **`generate`**). Optional **‘q’** quit for parity with **`GenView`** is acceptable if it doesn’t steal normal typing (only if `q` not needed in input—prefer **Ctrl+C**-only if single-line input makes **q** ambiguous).

4. **First turn** → Initial encode prefix is `User: <text>\nAssistant: ` (no prior history), so the model continues after **`Assistant:`** like a dialog schema.

## Risks / Trade-offs

- **[Performance]** Full-sequence **prefill** every turn (**O(tokens²)**-ish work over time**) → Mitigation**: document limitation; tune **`max-seq-len`** / short sessions for DIY use; KV reuse remains a separate change if needed later.
- **[Context length]** Transcript encoding can exceed **`config.maxSeqLen`** → Mitigation**: fail turn with explicit error suggesting shorter history or **`max-new-tokens`** reduction; optionally truncate transcript (implementer’s choice documented in **`tasks`** if spec requires specific behavior)—spec **SHALL** mandate **explicit\*\* failure over silent corruption.
- **[UX]** Narrow terminals garble layout → Mitigation**: simple flex layout and soft-wrap via **`Text`\*\*.

## Migration Plan

- **Additive**: New subcommand only; **`generate`** / **`inspect`** unaffected. Ship behind normal CLI build (**Nx** **`apps/cli`**). No data migration.

## Open Questions

- Transcript truncation policy when over **max-seq-len**: drop oldest messages vs hard error (**default in implementation**: hard error keeps behavior explainable).
