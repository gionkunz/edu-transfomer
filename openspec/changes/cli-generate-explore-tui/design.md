## Context

- **`apps/cli/src/cmd/generate.tsx`** today renders a minimal **`GenView`** (title, throughput line, streamed text). It calls **`generateAutoregressive`** from **`@llm-transformer-diy/inference`**, decoding the full **`[...promptIds, ...generatedIds]`** on each **`onSampledToken`** to handle BPE / UTF-8 boundaries.
- **`packages/inference/src/sampling.ts`** computes — internally — a **post-temperature, post-top-k softmax** over the vocabulary inside **`sampleLogits`**, then performs a categorical draw. That probability vector is **discarded**; only the chosen token id is returned.
- **`generateAutoregressive`** exposes a **synchronous, void-returning** **`onSampledToken(tid, idx)`** callback. There is no host-driven way to **delay** or **pause** the decode loop (the loop calls **`forwardGptDecodeStep`** immediately after each draw).
- **`@wolf-tui/react`** provides **`Box`**, **`Text`**, **`useInput`**, **`useStdout`** (terminal rows) — already used by **`generate.tsx`** and **`chat.tsx`**.
- **`docs/project.md`** mandates "no ML logic in **`apps/cli`**" and additive, low-impact dependency additions.

## Goals / Non-Goals

**Goals:**

- Surface the **next-token probability distribution** to the CLI host on every decode step **without** reimplementing softmax/top-k in **`apps/cli`** and **without** breaking existing **`onSampledToken`** consumers (chat, tests).
- Render a side-by-side **explore TUI**: streamed text on the left with the **last-sampled token highlighted**, vocabulary inspector on the right (sorted by probability, fitted to terminal height).
- Keyboard control during generation: **`s`** toggles **0.25 tok/s** slow cap; **`p`** toggles **pause**; existing **`q`** / **Ctrl+C`** quit semantics preserved. Indicator in status line at all times.
- Default-on for **`llm generate`**; **`--no-explore`** restores the prior minimal view for scripted runs.

**Non-Goals:**

- Updating **`llm chat`** — chat keeps its current TUI; explore mode is **`generate`**-only this iteration.
- New sampling algorithms, beam search, top-p, or alternative samplers.
- Persisting probability traces, exporting CSVs, or sparkline / histogram visualizations beyond a per-row probability + small bar.
- Touching tensor / GPU / autograd / model packages.
- New dependencies (no `chalk`, no animation libs — colors via **`@wolf-tui/react`** **`Text`** style props).

## Decisions

1. **Expose probabilities via a new awaitable `onStep` callback in `GenerateOptions`**

   Add (additive, optional) to **`GenerateOptions`**:

   ```ts
   readonly onStep?: (step: {
     readonly tokenId: number;
     readonly newTokenIndex: number;
     /** Post-temperature, post-top-k softmax — same vector used for the categorical draw. Length === vocabSize. */
     readonly probabilities: Float32Array;
   }) => void | Promise<void>;
   ```

   - **Awaitable**: **`generateAutoregressive`** **`await`s** the returned promise (if any) **before** the next **`forwardGptDecodeStep`**. This is the only host hook for pacing/pausing — keeps **`apps/cli`** in charge of timing, **inference** in charge of math.
   - **Backward compatible**: **`onSampledToken`** still fires alongside **`onStep`**. Existing chat/generate-no-explore code paths keep working.
   - **Allocation**: the vector handed to **`onStep`** is a freshly allocated **`Float32Array`** **owned by the host** for that step (we'll **`slice()`** it inside the loop). The host must **not** mutate it after the await resolves — but since we allocate per step, no aliasing risk vs. internal sampler state.

   _Alternatives considered:_
   - **(a)** Mutate **`onSampledToken`** signature → rejected: breaking API for chat/tests.
   - **(b)** Expose only raw logits and re-softmax in CLI → rejected: violates "no sampling math in `apps/cli`".
   - **(c)** Return probs from **`sampleLogits`** as a tuple → rejected: ripples through tests; callback is the natural pacing hook anyway.

2. **Refactor `sampleLogits` minimally to expose the probability vector**

   Introduce **`sampleLogitsWithProbs(rng, logits, opts) → { tokenId, probabilities }`** in **`packages/inference/src/sampling.ts`** sharing the existing temperature + top-k + softmax + categorical-draw logic. **`sampleLogits`** keeps its current signature and delegates to the new helper (drops the probs). Greedy (`temperature ≤ 1e-8`) returns a one-hot **`probabilities`** vector to keep the contract uniform — host doesn't need to special-case greedy.

   _Alternative considered:_ keep two parallel implementations — rejected: drift risk.

3. **Generate explore TUI layout**

   Top-level **`Box`** with **`flexDirection: 'row'`**:
   - **Left column** (`flexGrow: 1`): header line, status line (mode + tok/s + hotkeys), then a single **`Text`** node rendering the streamed completion. The last sampled token is colored (e.g. **`color: 'cyan'`** with **`fontWeight: 'bold'`**) by emitting **two** **`Text`** spans inside the body **`Box`**: prefix piece and last piece, where "last piece" = `fullDecoded.slice(prevFullDecodedLen)`. Recompute `prevFullDecodedLen` on every step _before_ pushing the new id.
   - **Right column** (fixed `width`, e.g. **`32`**): **`Box`** titled "next-token p" containing one **`Text`** row per vocabulary entry sorted by descending probability. Number of rows is **`max(0, terminalRows − chromeRows)`** where `chromeRows` accounts for the panel border + title + bottom margin. We compute terminal rows via **`useStdout()`** (already used in **`chat.tsx`**) on each render. If the terminal is narrower than e.g. 60 columns or shorter than 8 rows, we **omit** the right panel to avoid garbled output.
   - Each inspector row format: `  42  ▏word  0.1234`
     - left-padded id (4 cols), a thin bar (Unicode block characters scaled to row width budget, e.g. **`▏▎▍▌▋▊▉█`**) — 6–8 cols, decoded piece (escape control chars + show `␣` / `↵` for whitespace), trailing fixed-width probability text. Decoded piece is the per-token decode (best-effort `decode([id])` with `utf8: 'replace'`); if it produces a replacement char only (lone surrogate / continuation byte), show `‹id›` instead. This rendering is **display-only**; encoding/decoding stays in **`@llm-transformer-diy/tokenizer`**.

4. **Sorting strategy for the inspector**

   Per spec answer: "all vocabulary sorted by probability". With typical educational vocab sizes (≈ 1k–32k), full sort is **O(V log V)** per step — negligible compared to a forward pass. Implementation: build an `Int32Array` of indices `[0..V−1]`, sort by `(j, i) => probs[j] − probs[i]`, slice the visible window. We do **not** introduce a heap-based partial sort yet; if profiling shows this matters later, it's a localized optimization.

5. **Pacing & pause via the awaitable `onStep`**

   The **`generate`** TUI maintains two pieces of state outside React (kept in plain refs / closures, with **`paint()`** to update React):
   - **`speedMode: 'max' | 'slow'`** — `slow` ⇒ wait until `4000 ms` have passed since the last step's start; `max` ⇒ no wait.
   - **`pauseGate: { paused: boolean; resolve?: () => void }`** — when `p` is pressed, set `paused = true`. When pressed again, resolve the pending wait promise.

   The `onStep` handler implementation:

   ```ts
   await sleepWithCancel(slowDelayMs); // 0 in max mode, ~4000 ms in slow mode
   while (pauseGate.paused) {
     await new Promise<void>((r) => {
       pauseGate.resolve = r;
     });
   }
   ```

   Both waits **MUST** also resolve early on **`q`** / **Ctrl+C** so we don't deadlock during teardown — implementation cancels the timer + flushes the gate before `inst.unmount()`.

   _Alternative considered:_ pushing pause/throttle into **`generateAutoregressive`** (e.g. add `delayBetweenSteps`). Rejected: timing/UI policy is host territory, and a single awaitable hook is more general.

6. **Hotkey handling**

   `useInput` in the explore view handles:
   - **`s`** → toggle `speedMode` (no effect when paused — state still flips so resume runs in the new mode).
   - **`p`** → toggle `pauseGate.paused`; when transitioning paused → unpaused, call `pauseGate.resolve?.()`.
   - **`q`** or **`Ctrl+C`** → request cancel: flush timer + gate, then `process.exit(0)` after `inst.unmount()` (mirrors current `generate.tsx`). We accept that mid-step exit may abandon the in-flight forward pass — same posture as today.

   Mode hint in the status line (e.g. `mode: NORMAL · s = slow · p = pause · q = quit`).

7. **Activation flag**

   `cli-program.ts` adds **`--no-explore`** to **`generate`** (Commander style: `.option('--no-explore', '...')` produces `opts.explore: false` when set, default `true`). Passes through as `runGenerateCommand({ explore: boolean, ... })`. When `explore === false`, render the **existing** **`GenView`** unchanged — exact byte-for-byte parity is preserved by keeping that component intact and branching at render time.

8. **No-ML-logic boundary**

   Inspector rendering reads **only** the `probabilities` Float32Array handed back via `onStep` plus a `tokenizer.decode([id])` call per visible row. No softmax, no top-k, no penalty math is reproduced in `apps/cli`. The repetition penalty (when set) is still applied in **inference** before sampling — the probabilities the host sees are post-penalty, so the panel reflects what actually drove the draw (educationally accurate).

## Risks / Trade-offs

- **[BC] Default-on changes user-visible output of `llm generate`** → Mitigation: prominent **`--no-explore`** flag documented in `--help` and in the change spec; default behavior diff called out in tasks. Headless / scripted callers SHALL pass `--no-explore` (deliberate choice — educational project values discoverability).
- **[Perf] Sorting full vocab every step on huge vocab** → Mitigation: V is bounded by current educational checkpoints (≤ tens of thousands); sort is single-digit ms. If this ever shows up in profiling, swap for partial top-N.
- **[Perf] Per-row `tokenizer.decode([id])` calls** → Mitigation: cache decoded strings in a `Map<number, string>` keyed by id; the visible window stays small (rows fit terminal). Cache survives across steps.
- **[UX] BPE byte fragments cannot decode cleanly alone** → Mitigation: fall back to `‹id›` placeholder when single-id decode yields only U+FFFD; this is a display affordance only and matches existing replacement behavior.
- **[UX] Last-token highlight using full-decode slicing may color slightly more than the new BPE piece** when the new token completes a previously-broken UTF-8 sequence → Mitigation: acceptable — colors the **newly-rendered glyphs**, which is what the user perceives as "what just appeared".
- **[Lifecycle] Pause/sleep can deadlock teardown** → Mitigation: `q` / `Ctrl+C` paths flush the timer and resolve the gate before unmount; covered by an explicit task and unit-testable at the `cmd/generate` level if we extract a small `pacingController` helper.
- **[API] Adding `onStep` to `GenerateOptions`** is a non-breaking surface widening — existing tests and callers (chat, headless tests) that omit `onStep` see zero behavior change; verified by keeping `onSampledToken` semantics intact.

## Migration Plan

- **Additive**: New flag, new optional callback, new view component. The existing **`GenView`** is preserved verbatim and selected when `--no-explore` is passed. **No data migration**.
- **Rollout**: Single PR aligned with this OpenSpec change. Tests:
  - **inference**: unit test covering `sampleLogitsWithProbs` (probabilities sum ≈ 1; greedy → one-hot; top-k → exact zeros outside top-k); existing `generateAutoregressive` tests run unchanged.
  - **apps/cli**: `cli-program.spec.ts` updated to assert `--no-explore` parses and reaches the runner; a small unit on the pacing controller (slow ≈ 4 s, pause/unpause flushes).
- **Rollback**: Revert PR or pass `--no-explore` to mute the new view.

## Open Questions

- Color choices for the highlighted token: cyan? yellow? Pick something that contrasts on default dark _and_ light terminal themes. Implementer's call; tasks will specify the default but allow easy change.
- Whether to display the probability of the **just-sampled** token even when it falls below the visible window (e.g. a footer line "sampled: id=42 p=0.0031 (rank 87)"). **Lean: yes**, surfaced in the status line — cheap and educational; will fold into tasks if not contentious.
- Whether the inspector should also show the **rank** of the sampled token. Same answer as above; `(rank N)` next to the sampled-token footer is sufficient and avoids cluttering each row.
