## 1. OpenSpec (this change)

- [x] 1.1 Add `proposal.md`, `design.md`, delta `specs/tokenizer-bpe-bootstrap/spec.md`, and `tasks.md` under `openspec/changes/optimize-bpe-encode-hot-loop/`.
- [x] 1.2 `openspec validate optimize-bpe-encode-hot-loop --strict` passes.

## 2. Tokenizer hot-loop kernel

- [x] 2.1 In `packages/tokenizer/src/bpe-tokenizer.ts`, replace `performSingleGreedyMergeUint32` with a private `performGreedyMergePassUint32(buf, len) → number | null` that does (a) a single scan to pick the lowest-rank pair (tie-break: smaller `L`, then smaller `R`) and (b) a single in-place left-to-right collapse pass (`r`/`w` cursors, `r += 2` on match else `r += 1`) returning the new logical length, or `null` when no learned pair exists in `buf[0..len)`.
- [x] 2.2 Confirm the kernel preserves the **deterministic encoding** spec (steps 1–2 unchanged; step 3 collapses all non-overlapping occurrences in one walk, mathematically equivalent to repeated leftmost-only merges).

## 3. Typed-buffer encode entry points

- [x] 3.1 Add `encodeUint32(text, hooks?): Uint32Array` to `BPETokenizer` that runs the new kernel and returns a freshly allocated `Uint32Array` sized to logical length (no `subarray` view sharing storage).
- [x] 3.2 Add `encodeUint32Async(text, hooks?): Promise<Uint32Array>` mirroring `encodeAsync`'s yield contract (wall-time-bounded primary trigger via `yieldEventLoopForHostUi`; coarse pass-counter as secondary).
- [x] 3.3 Reimplement `encode` and `encodeAsync` as **thin adapters** that call the typed entries and `Array.from(...)` the result, preserving the `Tokenizer` contract.
- [x] 3.4 Update `BpeEncodeHooks` types and JSDoc to document that `mergeIteration` now counts **outer best-rank passes** (not single-position merges) and that `max(1, initialTokenCount - 1)` remains a valid worst-case upper bound.

## 4. Tokenizer tests (Vitest)

- [x] 4.1 Add a **leftmost-baseline equivalence** test in `packages/tokenizer/src/__tests__/` that builds a small reference loop selecting the min-rank pair and merging only its leftmost occurrence per iteration, then asserts element-wise equality with `BPETokenizer.encode`/`encodeUint32` on at least: (a) `"héllo 世界"`, (b) a string with multiple non-overlapping occurrences of the same digram (e.g. `"abababab"`-style with a tokenizer trained on it), (c) a longer paragraph from existing fixtures.
- [x] 4.2 Add a `encodeUint32` ↔ `encode` parity test asserting `Array.from(encodeUint32(x))` equals `encode(x)` element-wise across the same probes plus the existing tie/round-trip probes.
- [x] 4.3 Add an `encodeUint32Async` ↔ `encodeAsync` parity test (await both, compare element-wise).
- [x] 4.4 Add a progress-monotonicity test: collect all `phase: 'merge'` events from `encodeUint32Async` on a synthetic input, assert `mergeIteration` is monotonically non-decreasing and the final value is `≤ max(1, initialTokenCount - 1)`.
- [x] 4.5 Confirm all existing `packages/tokenizer` tests still pass unchanged (round-trip, tie corpus, save/load, untrained errors).

## 5. CLI typed-buffer adoption

- [x] 5.1 In `apps/cli/src/cmd/train.tsx`, switch the `tokenizer.encodeAsync(corpus, …)` site (and the corresponding `encode` fallback if any) to call `encodeUint32Async` and consume the returned `Uint32Array` directly, removing the `Array.from` → `new Uint32Array(...)` round-trip.
- [x] 5.2 Verify the wolf-tui dashboard (`train-dashboard.tsx`) still renders elapsed, merges/s, pessimistic `ProgressBar`, and ETA correctly with outer-pass-counted `mergeIteration`. No code changes expected; visual smoke check via `npx nx run @llm-transformer-diy/cli:serve` is sufficient.

## 6. Verification

- [x] 6.1 `npx nx run @llm-transformer-diy/tokenizer:test` (or repo-standard `nx run-many -t test --all`) is green.
- [x] 6.2 `npx nx run @llm-transformer-diy/cli:test` is green.
- [x] 6.3 `npm run build` succeeds end-to-end.
- [x] 6.4 Spot-check wall time of `apps/cli` `train` corpus encode on the existing `corpus/real/` sample (or a representative fixture) and record the before/after timing in the change archive notes; expected ~2× wall-time reduction.
- [x] 6.5 `openspec validate optimize-bpe-encode-hot-loop --strict` passes.
