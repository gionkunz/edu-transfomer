# Design

## Context

See `proposal.md` for motivation and `specs/cli-learn-command/spec.md` for required behaviour. What shapes the approach:

- `forwardGptDecodeStep` (`packages/model/src/gpt-kv.ts`) already has opt-in capture: `attentionCapture`, `layerLens`, `flowTrace`. With `flowTrace` it returns post-softmax attention of every head, the logit lens, and per block the stream entering it plus the attention and feed-forward updates. Not captured today: the token/position vectors, each head's query, pre-softmax scores, and the mixed value vector. `kNew`/`vNew` are destroyed after being copied into the KV cache, so keys and values live only in the cache (`[maxSeqLen, dK]` per head).
- A decode step at position `p` attends over cache rows `0..p`. Decoding the **last prompt token** after prefilling the tokens before it therefore yields exactly the forward that predicts the next token, with all capture available. The prefill path has no capture and does not need any. A one-token prompt needs no prefill at all (`forwardGptPrefill` rejects empty input; decode at `seqLen 0` is valid).
- Attention has no bias and no causal mask on the decode path; scores are `(q · k) / √dK`. The model uses RMSNorm and learned position embeddings — lesson text must describe this model, not a textbook one.
- `apps/cli` may not compute ML values (`cli-wolf-tui-commands`: "No ML logic in CLI"). `buildFlowTrace` and `summarizeHeads` in `packages/inference` are the precedent for shaping captured tensors.
- `generate.tsx` (874 lines) mixes TUI state, timers and orchestration in one component; its pure parts (`generate-layout-tiers.ts`, `generate-pacing.ts`) are separately tested. `FlowPanel` in `generate-panels.tsx` already draws the architecture diagram from a `FlowTrace`.
- TUI code can only be driven interactively on software WebGPU in the dev environment, so correctness has to rest on pure, unit-tested pieces.

## Goals / Non-Goals

**Goals:**

- One model forward per lesson run; everything else is navigation over captured data.
- Chapters are data plus pure render functions, so adding the remaining chapters later is additive.
- Lesson state transitions and every schematic are testable without a terminal or a GPU.

**Non-Goals:**

- Explaining training, backpropagation or the tokenizer algorithm.
- Capture on the prefill path, or explaining positions other than the last one.
- Repetition penalty and stop markers in the lesson's sampler.
- Changing `llm generate`'s explore mode or sharing its state machine.
- Localisation of lesson text.

## Decisions

### 1. One instrumented decode step, captured in full

`llm learn` prefills `prompt[0..n-2]` (skipped when `n = 1`), then calls `forwardGptDecodeStep` for `prompt[n-1]` with a new option `lessonTrace: true`. It implies `flowTrace` and additionally returns:

| Field          | Shape               | Content                                       |
| -------------- | ------------------- | --------------------------------------------- |
| `inputVectors` | `[2, dModel]`       | token embedding row, position embedding row   |
| `allQueries`   | `[L·H, dK]`         | each head's query                             |
| `allScores`    | `[L·H, prefixRows]` | scaled scores before softmax                  |
| `allMixed`     | `[L·H, dK]`         | weights · values, before head-expand and `wo` |

Stacking and readback follow the existing `allAttention` pattern (one `createZeros` stack per field, `copyContiguousElems`, one `toArray`, destroy). The variables are already in the graph; capture adds copies only.

_Alternative — capture one head per forward and re-run on selection change:_ less data per readback, but re-running means rewinding `cache.seqLen` and re-decoding (the trick `recaptureCurrentStep` uses in `generate.tsx`, with its race guard). Capturing all heads is a few hundred KB even for the 56M-parameter model and makes head/block selection instant and stateless.

_Alternative — a separate "explain" forward outside `gpt-kv.ts`:_ would duplicate the block code and could drift from the real path. Rejected: the lesson must show the forward that inference actually runs.

### 2. Keys and values are read from the KV cache, lazily per head

New `readKvCacheHead(rt, cache, layerIndex, headIndex)` in `packages/model` returns `{ keys, values }` as `[seqLen, dK]` arrays via `view2dRowPrefix` + `toArray`. The lesson fetches them when a head is first shown and memoises per `(layer, head)`.

Reading all heads eagerly would be `2·L·H` readbacks up front (288 for a 12×12 model) for data mostly never looked at. Lazy reads cost two readbacks on the first visit to a head; the screen shows the previous head's strips dimmed until they arrive. The cache stays alive for the whole lesson run and is destroyed on quit or before a restart with `n`.

### 3. Inference owns the lesson trace and the sampling breakdown

New `packages/inference/src/lesson-trace.ts`:

- `runLessonForward(rt, model, promptIds)` → `{ trace: LessonTrace, cache }`. Runs prefill + instrumented decode, reads the logits, disposes both forward graphs (`disposeGptKvForwardGraph`), and returns plain host data: `flow: FlowTrace` (via existing `buildFlowTrace`), `rawLogits`, `inputVectors`, and `head(l, h)` → `{ query, scores, weights, mixed, peakIndex, entropy }`.
- `readLessonHeadKv(rt, cache, l, h)` wrapping decision 2.
- `mostFocusedHead(trace, layerIndex)` for the starting selection (lowest entropy).

New `explainSampling(logits, sampleOptions, topN)` in `sampling.ts` → for the `topN` most likely tokens plus an "others" remainder: raw probability, probability after temperature, kept-by-top-k flag, final probability; plus the full final distribution. It is built from the same code path as `sampleLogitsWithProbs` (factor the shared shaping into one internal function) so the spec's "shown distribution is the sampler's distribution" holds by construction, and a test compares the two. Drawing again calls the existing `sampleLogitsWithProbs` with the CLI's RNG.

_Alternative — compute softmax/temperature in the CLI because it is "just display":_ violates the CLI rule and risks the lesson disagreeing with the sampler.

### 4. Lesson framework: pure reducer + pure frame renderers + thin wolf-tui shell

- **`learn-lesson.ts`** — types and the chapter registry. A `Chapter` is `{ id, title, steps: Step[] }`; a `Step` is `{ id, frames: number, dial: 'block' | 'temperature' | 'none', render(ctx, frame): StepView }`. `StepView` is plain data: schematic as rows of styled spans, and explanation lines. Explanation text is produced by the step from real values (token pieces, head numbers, probabilities).
- **`learn-state.ts`** — `lessonReducer(state, action)`. State: `chapterIndex, stepIndex, frame, playing, layerIndex, headIndex, temperature, topK, drawnTokenId`. Actions: `next`, `prev`, `jumpChapter`, `tick`, `togglePlay`, `dialUp`, `dialDown`, `nextHead`, `prevHead`, `cycleTopK`, `redraw`, `restartWith`. Entering a step or changing a selection sets `frame = 0, playing = !noAnimation`; with `--no-animation`, `frame` is always the last frame and `tick`/`togglePlay` are no-ops.
- **`learn-view.tsx`** — the only React/wolf-tui file: holds reducer state, maps `useInput` keys to actions, runs one `setInterval` (≈120 ms) dispatching `tick` while `playing`, checks terminal size, and paints `StepView` plus header/footer.
- **`cmd/learn.tsx`** — `runLearnCommand`: TTY check before any GPU work, runtime + checkpoint + tokenizer loading as in `generate.tsx`, `runLessonForward`, render, cleanup.

Rationale: keeps everything a test needs (`lessonReducer`, each `step.render`) free of React, timers and GPU. `generate.tsx` shows what happens otherwise.

_Alternative — extend explore mode with a "lesson" toggle:_ explore is driven by a running generation loop and its pacing controller; a lesson is a static dataset with a cursor. Sharing the component would complicate both.

### 5. Reuse the architecture diagram, do not fork it

The stack chapter renders the existing `FlowPanel` (`boxed` variant, or `compact` when rows are short) from `trace.flow`. `FlowPanel` gains one optional prop, `reveal?: { steps: number; lens: number; output: boolean }`: `steps` counts block half-steps along the stream (block 1 attention, block 1 feed-forward, block 2 attention, …), `lens` the number of blocks whose guess is shown, `output` whether the output row is drawn. Hidden parts keep their width so nothing shifts. The default (undefined) is today's rendering, so `generate` is unaffected. The chapter's animation is a sweep of `reveal`. This chapter's `StepView` therefore carries `FlowPanel` props instead of span rows; `learn-view.tsx` handles both kinds.

When the terminal has too few rows for all blocks even at one row per block, the chapter passes a window of blocks around the focused one (the flow trace sliced; block numbers stay absolute) and says so in the explanation.

At 80 columns the one-row-per-block form of the panel did not fit its numbers (it truncated the lens probability in `generate` too). Below 88 columns it now shortens labels (`b1` instead of `block 1`, a 5-cell bar, `norm` / `T=` in the output row). This also changes `generate`'s architecture view on narrow terminals, for the better; wider terminals render as before.

### 6. Drawing vectors and bars

- A vector is a fixed-width strip (default 32 cells; longer vectors are averaged into buckets in inference, like `bucketAttentionRow`). Cell shade encodes magnitude with the block ramp already used for sparklines; colour encodes sign. Scaling is per chapter screen (shared max across query and visible keys) so strips are comparable.
- Scores can be negative: bars grow left or right from a centre axis. Weights and probabilities reuse `probabilityBar`.
- Token rows: the most recent `rows` tokens, newest at the bottom next to the query; hidden tokens summarised as count + total weight (computed in inference).
- Bar transitions on temperature change interpolate displayed lengths over a few frames in the reducer (`fromProbs`, `toProbs`, `frame`); the numbers shown switch immediately so they are never fabricated in-between values.

### 7. Keys

`←/→` step, digits chapter, `space` play/pause/replay, `↑/↓` the step's dial (block or temperature), `h/H` head, `k` top-k, `r` redraw, `n` continue, `q` quit. `↑/↓` for block matches `generate`. No `+/-` (not reachable on all keyboards). Temperature steps by 0.1 within `[0, 2]`; top-k cycles `off → 5 → 10 → 40 → <cli value>` (deduplicated).

### 8. Chapter numbering

Chapters are numbered by their position in the registry (1–3 in this change). Later chapters are inserted in teaching order and numbers shift; nothing persists a chapter number.

## Risks / Trade-offs

- [Lesson text describes a mechanism the code does not have] → Text is written against `gpt-kv.ts` (no bias, RMSNorm, learned positions, per-head projections); a reviewer task checks each step's text against the code path it names.
- [Animation timer causes flicker or full-screen repaints over SSH] → Frames change only the schematic area; `--no-animation` is the fallback and is what tests use.
- [`FlowPanel` change regresses `generate`] → New prop is optional with unchanged default. There is no component-render harness in this repo, so the equivalence is tested on the pure `revealedBlockSteps` helper and checked by eye in a pseudo-terminal run of `generate`'s view.
- [Refactoring `sampleLogitsWithProbs` to share shaping changes sampling results] → Existing `sampling.spec.ts` stays as is; add an equality test between `explainSampling().final` and `sampleLogitsWithProbs().probabilities`.
- [bf16 checkpoints] → Stacks are created with `floatDtypeOf(model.tokenEmbed.tensor)` as existing capture does; one capture test runs with bf16 if the runtime supports it.
- [Interactive behaviour verified only on software WebGPU] → Same limitation as explore mode; pure tests carry correctness, plus a scripted pseudo-terminal walk-through on a tiny checkpoint.
- [80×24 is tight for query + keys + explanation + chrome] → Layout budget: 2 header, 1 footer, 4 explanation, 1 separator leaves 16 schematic rows; each step's renderer takes `rows`/`cols` and a test renders every step at 80×24 asserting row count and max width.

## Open Questions

- Default prompt wording (must tokenize to a handful of tokens with common tokenizers); any short English phrase works and can be changed without touching specs.
- Exact colours for positive/negative vector components; to be chosen against the existing rank palette for contrast on dark and light terminals.
