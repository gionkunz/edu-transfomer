## Why

`llm generate` is the workhorse inference command but its TUI shows only the streamed text and a tok/s counter. Because this project is **educational**, the most interesting moment of inference — the **distribution over the next token** — is invisible. Surfacing the live ranked vocabulary, marking the **just-sampled** token, and letting the user **slow down** or **pause** generation turns each step into a teachable view of how the model "thinks", without changing any of the underlying ML logic.

## What Changes

- Add a side-by-side **probability inspector panel** to **`llm generate`**: right-hand box listing the full vocabulary sorted by probability for the **current next-token** distribution; row count auto-fits to terminal height. Each row shows token id, decoded piece (with whitespace/control rendering hints), and probability (e.g. **`0.1234`** + a small bar).
- The **most recently sampled token** in the streamed text is rendered in a distinct color (e.g. cyan/yellow) for one or more steps so the user can correlate "what was picked" with "what was likely".
- Keyboard controls during generation:
  - **`s`** toggles a **slow** rate cap of **0.25 tokens/sec** (≈ 4 s between tokens). Pressing **`s`** again returns to **uncapped** speed. Default is uncapped.
  - **`p`** **pauses** generation (no new forward passes) and freezes the inspector panel on the current distribution. **`p`** again **resumes**.
  - Existing exits (**`q`**, **`Ctrl+C`**) keep working.
- Activation: educational explore mode is **default-on** for **`llm generate`** (project is educational). A new flag **`--no-explore`** disables the inspector and reverts to the prior minimal `GenView` so scripted/headless runs are unaffected.
- Inference plumbing: extend the **`generateAutoregressive`** callback surface so the host can receive the **post-sampling probability vector** (or normalized logits) for the just-finished decode step, plus a way to **wait/await** between steps without reimplementing sampling in the CLI. No new sampling math in `apps/cli`.
- **No** changes to model code, tokenizer, training, or sampling algorithms; **no** new dependencies; existing **wolf-tui** primitives only.

## Capabilities

### New Capabilities

- _(none)_

### Modified Capabilities

- **`cli-wolf-tui-commands`**: Extend the **`llm generate`** requirement to specify the explore mode (probability inspector, recent-token highlight, **`s`** / **`p`** hotkeys, **`--no-explore`** opt-out), keeping the wiring-only / no-ML-logic contract.

## Impact

- **Primary**: `apps/cli` — `cli-program.ts` (new flag), `cmd/generate.tsx` (new view + key handlers + rate limiter + pause gate).
- **Secondary**: `packages/inference` — small additive surface on **`generateAutoregressive`** so the CLI can observe per-step probabilities and inject an awaitable delay without changing sampling. Existing call sites (chat, generate w/o explore, tests) keep working unchanged.
- **Dependencies**: None added. Uses existing **`@wolf-tui/react`**, **Commander**, and **`@llm-transformer-diy/inference`**.
- **Lower layers** (`tensor`, `gpu`, `autograd`, `model`, `training`): unchanged.
- **Docs / specs**: only the modified capability above; no `docs/project.md` policy changes.
