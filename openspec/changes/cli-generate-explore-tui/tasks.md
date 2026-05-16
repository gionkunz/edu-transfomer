## 1. Inference: expose per-step probabilities and awaitable host hook

- [ ] 1.1 Add **`sampleLogitsWithProbs(rng, logits, opts)`** in `packages/inference/src/sampling.ts` returning `{ tokenId: number; probabilities: Float32Array }` (length === `logits.length`). Greedy path SHALL return a one-hot probabilities vector for the argmax id. Top-k path SHALL return `0` for ids outside the top-k mask.
- [ ] 1.2 Refactor `sampleLogits` to delegate to `sampleLogitsWithProbs` (drops the probabilities) so existing callers and tests remain byte-compatible.
- [ ] 1.3 Export `sampleLogitsWithProbs` from `packages/inference/src/index.ts` (alongside `sampleLogits` / `greedyArgmax`).
- [ ] 1.4 Extend `GenerateOptions` in `packages/inference/src/generate.ts` with optional `onStep?: (step: { tokenId: number; newTokenIndex: number; probabilities: Float32Array }) => void | Promise<void>`. In `generateAutoregressive`, after sampling each token, allocate a fresh `probabilities` slice and call `onStep` (when provided), `await` its return value before the next `forwardGptDecodeStep`. Continue calling `onSampledToken` (sync) to preserve back-compat.
- [ ] 1.5 Document — in JSDoc on `onStep` — that the probabilities are post-temperature / post-top-k / post-repetition-penalty (i.e. the distribution actually used to draw), with length equal to `model.config.vocabSize`, and that `onStep` MAY be async (the loop will await before the next decode step).
- [ ] 1.6 Add a Vitest in `packages/inference/src/sampling.spec.ts` proving `sampleLogitsWithProbs`: probabilities sum ≈ 1 (within 1e-6), all probabilities `>= 0`, top-k mask zeros respected, greedy returns one-hot with sum exactly `1`.
- [ ] 1.7 Add a Vitest covering `generateAutoregressive` `onStep` semantics on a tiny CPU-friendly fixture (or alongside existing `kv-decode.spec.ts` if a runtime is available): callback is invoked once per sampled token with `newTokenIndex` matching `onSampledToken`, the loop awaits a returned promise (test by resolving after a microtask and asserting ordering against `onSampledToken` of the _next_ step).

## 2. CLI flag wiring (`apps/cli`)

- [ ] 2.1 In `apps/cli/src/cli-program.ts`, add `.option('--no-explore', 'disable the explore TUI (probability inspector, recent-token highlight, s/p hotkeys)')` to the `generate` subcommand. Commander parses `--no-explore` to `opts.explore === false`; default is `true`.
- [ ] 2.2 Pass `explore: opts.explore !== false` through to `runGenerateCommand({ ..., explore })`.
- [ ] 2.3 Update `apps/cli/src/cmd/generate.tsx` `runGenerateCommand` signature to accept `readonly explore: boolean` (no default — caller decides).

## 3. CLI explore TUI (`apps/cli/src/cmd/generate.tsx`)

- [ ] 3.1 Keep the existing `GenView` component verbatim and use it when `explore === false` (no behavioral change in `--no-explore` runs).
- [ ] 3.2 Add a new `ExploreView` component that renders a top-level `Box` with `flexDirection: 'row'`: left column for header / status / streamed text, right column for the inspector panel. Use `useStdout()` to read terminal `rows` / `columns` and recompute the visible inspector row count and panel visibility on each render.
- [ ] 3.3 Render the streamed completion in two `Text` spans inside a flex-wrapping `Box`: prefix piece (default color) and last piece (highlight color, e.g. `color: 'cyan', fontWeight: 'bold'`). Compute `prevFullDecodedLen` immediately _before_ appending the new sampled id and use it to slice the highlight piece out of `decode([...promptIds, ...generatedIds], { utf8: 'replace' })`.
- [ ] 3.4 Implement the inspector rows: take the latest `Float32Array` of probabilities, build an `Int32Array` of indices `[0..V-1]`, sort by descending probability, render the first `visibleRows` rows. Each row format: `id`, a small Unicode block bar scaled to row width, a per-id decoded string (cached in a `Map<number, string>`), and the probability formatted to 4 decimals. For ids whose lone-id decode yields only U+FFFD, show `‹id›` as the display piece.
- [ ] 3.5 Display a status line that ALWAYS shows: `mode: NORMAL|SLOW|PAUSED · tok/s X.X · new N · s = slow · p = pause · q = quit`. Include the sampled token's rank + probability footer (`sampled: id=N p=0.XXXX rank=K`).
- [ ] 3.6 Hide the inspector panel when `columns < 60` or `rows < 8`, and continue streaming the left column unchanged (no error).

## 4. CLI pacing controller and hotkeys

- [ ] 4.1 Implement a small in-file `PacingController` (or extract to `apps/cli/src/cmd/generate-pacing.ts` if helpful for testing) with state `{ speed: 'max' | 'slow'; paused: boolean }` and methods `setSpeed`, `togglePause`, `await waitBetweenSteps()`, `cancel()`. `waitBetweenSteps` SHALL: (a) sleep ~`4000 ms` from the start of the previous step in `slow` mode (or `0` in `max`), then (b) await an internal pause gate while `paused === true`. `cancel()` resolves the gate immediately and clears any pending timer.
- [ ] 4.2 Wire `useInput` in `ExploreView`: `s` calls `setSpeed(speed === 'slow' ? 'max' : 'slow')`; `p` calls `togglePause()`; `q` and `Ctrl+C` call `cancel()` then `process.exit(0)` after `inst.unmount()` in the `finally` block.
- [ ] 4.3 In the `runGenerateCommand` flow, pass `onStep: async ({ tokenId, newTokenIndex, probabilities }) => { /* update React state; */ await pacing.waitBetweenSteps(); }` to `generateAutoregressive`. Update the React state via the existing `paint()` pattern (or `inst.rerender`) before awaiting, so the inspector reflects the just-finished step while the user waits.
- [ ] 4.4 Ensure the `finally` of `runGenerateCommand` calls `pacing.cancel()` _before_ `inst.unmount()` to avoid teardown deadlocks.

## 5. Tests and verification

- [ ] 5.1 Update `apps/cli/src/cli-program.spec.ts` to assert that `--no-explore` is parsed and reaches `runGenerateCommand` with `explore === false`, and that omitting the flag produces `explore === true`.
- [ ] 5.2 If the pacing helper is extracted, add a Vitest covering: (a) `slow` waits ≥ ~4 s from previous-step start (use fake timers if available, else a relaxed lower bound), (b) `togglePause` blocks `waitBetweenSteps` until untoggled, (c) `cancel` resolves both the timer and the gate so awaits do not deadlock.
- [ ] 5.3 Run `npx nx run-many -t test --all` (or per-project: `nx run @llm-transformer-diy/inference:test`, `nx run @llm-transformer-diy/cli:test`) and ensure existing chat / generate-no-explore tests still pass.
- [ ] 5.4 Manual smoke (per `AGENTS.md`): `npm run build` followed by `node dist/apps/cli/main.js generate --help` lists the new `--no-explore` flag; running `generate` against an existing checkpoint shows the inspector by default and minimal view with `--no-explore`.

## 6. Documentation touch-ups

- [ ] 6.1 Update `apps/cli/src/cli-program.ts` `--help` strings (already done in 2.1) to make the explore default behavior discoverable.
- [ ] 6.2 (Optional, only if maintainers ask) Add a one-paragraph note to `docs/transformer-explained.md` referencing the new explore mode as a teaching aid; SKIP if no doc change requested in review.
