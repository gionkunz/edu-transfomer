# Proposal

## Why

The toolkit already captures what a transformer does while it predicts a token (attention weights, per-block updates, the logit lens), but only shows it in `llm generate`'s explore mode, at generation speed, to someone who already knows what the numbers mean. There is no place in the tool where a newcomer is walked through the mechanism itself. For an educational stack that is the missing piece: a lesson that explains each part of the model using the user's own checkpoint and prompt, so every number on screen is real.

## What Changes

- New command **`llm learn`**: an interactive, step-by-step lesson that explains how the loaded model predicts the next token after a prompt. Each step is one screen with a schematic drawn from real captured values, a short plain-language explanation, and an optional animation.
- A **lesson framework**: chapters made of ordered steps, keyboard navigation (step forward/back, replay or pause the animation, jump to a chapter, quit), a progress indicator, and a frame timer for animations with a no-animation mode.
- Three chapters in this change:
  - **One attention head** — the current token's query compared against the keys of all tokens so far, the resulting scores, softmax into weights, and the weighted mix of values. Block and head are selectable.
  - **The stack of blocks** — the residual stream passing through every block, with each block's attention and feed-forward update and the model's best guess after each block.
  - **Picking the next token** — raw vocabulary scores, the effect of temperature and top-k (both adjustable live), and the random draw (repeatable).
- After the last step the user can **append the sampled token and run the lesson again** for the next position.
- New capture in the decode step for the values the attention chapter needs and that are not captured today: the token and position vectors, each head's query, its scores before softmax, and its mixed value vector; plus reading a head's cached keys and values.
- A pure sampling breakdown in the inference package (raw → temperature → top-k → final distribution) so the lesson shows exactly what the sampler does.
- README section for `llm learn`.

Not in this change (the framework is built so they can be added as further chapters): tokens/BPE, embeddings, many heads side by side, feed-forward internals, a dedicated residual-stream chapter, and the KV cache.

## Capabilities

### New Capabilities

- `cli-learn-command`: the `llm learn` guided lesson — invocation and options, lesson structure and navigation, animation behaviour, the content each chapter must show, where its values come from, and behaviour on small or non-interactive terminals.

### Modified Capabilities

None. `cli-wolf-tui-commands` requirements for `generate`, `chat` and `train` are unchanged; its "No ML logic in CLI" requirement applies to the new command as written.

## Impact

- **`packages/model`** (`gpt-kv.ts`): one new optional capture mode on `forwardGptDecodeStep` and a helper to read a head's cached keys/values. Existing options and outputs unchanged; no cost when the mode is off.
- **`packages/inference`**: a lesson-trace module (runs prefill + one instrumented decode step and shapes the result) and a sampling-breakdown function. New exports only.
- **`apps/cli`**: new `learn` subcommand in `cli-program.ts`, new `cmd/learn*.ts(x)` files (framework, chapter renderers, lesson text). Reuses `FlowPanel` and the bar/sparkline helpers from `generate-panels.tsx`.
- **Docs**: README.
- **Dependencies**: none added.
- **Compatibility**: works with any existing checkpoint, including ones trained before this change; no checkpoint or tokenizer format change.
