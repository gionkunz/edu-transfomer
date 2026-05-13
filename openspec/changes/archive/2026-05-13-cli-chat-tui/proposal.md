## Why

**`llm generate`** fits one-shot prompts, but exploratory use is nicer with a loop: send a message, read the reply, refine, repeat—without restarting the process or stitching prompts by hand. A dedicated **`chat`** command with a compact **wolf-tui** layout lowers friction for interactive inference without adding ML logic outside the existing **`inference`** stack.

## What Changes

- Add **`llm chat`**: launches a fullscreen **wolf-tui** chat UI (prompt input + scrollable transcript, streamed assistant tokens).
- Reuse **`generate`** knobs where they apply (`--model`, `--tokenizer`, temperature, top-k, repeat penalty/window, `max-new-tokens` per assistant turn—or clearly documented defaults).
- Wire-only in **`apps/cli`**: loads checkpoint + tokenizer like **`generate`**; delegates token sampling and forward passes to **`@llm-transformer-diy/inference`** (and existing model/KV primitives). Multi-turn conversation is built by **expanding an accumulated transcript into token prompts** each turn (**fresh KV cache** per assistant generation), so no new sampling rules in the app layer.
- **Ctrl+C** (or documented exit chord) cleanly tears down the TUI and GPU resources, consistent with other wolf-tui commands.

## Capabilities

### New Capabilities

- _(none)_

### Modified Capabilities

- **`cli-wolf-tui-commands`**: Add a **`chat`** subcommand alongside **`generate`** / **`train`**: wolf-tui chat session, **`inference`**-backed generation per turn, and the same GPU / “no ML logic in **`apps/cli`**” contract.

## Impact

- **Primary**: `apps/cli` (`cli-program.ts`, new `cmd/chat.tsx` or similar, shared loading helpers with **`generate`** where practical).
- **Dependencies**: No new packages expected (existing **Commander**, **wolf-tui**, **training** load path, **inference** `generateAutoregressive`, **tokenizer** BPE).
- **Lower layers**: Unchanged unless implementation discovers a reusable **`inference`** helper for chunked streaming (optional follow-up—not required if **`generate`**-style **`onSampledToken`** suffices).
