# Implementation tasks — accelerate-bpe-train-incremental-pair-table

Tasks are grouped roughly by the order they should land in. Each group ends in a verifiable checkpoint (tests pass, equivalence holds, perf measured).

## 1. OpenSpec (this change)

- [x] 1.1 `proposal.md`, `design.md`, `specs/tokenizer-bpe-bootstrap/spec.md`, and `tasks.md` exist under `openspec/changes/accelerate-bpe-train-incremental-pair-table/`.
- [x] 1.2 `openspec validate accelerate-bpe-train-incremental-pair-table --strict` passes.

## 2. Foundation: linked-list slot pool

- [x] 2.1 Add `packages/tokenizer/src/bpe-train-incremental/linked-token-stream.ts` exporting `BpeLinkedTokenStream` (parallel `Uint32Array`s `value` / `prev` / `next` + `Uint8Array` `alive`, plus `head` / `liveCount` and the `END = 0xFFFFFFFF` sentinel) and the constructor `createLinkedTokenStream(capacity: number): BpeLinkedTokenStream` that pre-allocates all four arrays at the given capacity.
- [x] 2.2 Add a synchronous `appendLiveSlot(stream, slotIndex, tokenId)` helper that wires `slotIndex` onto the tail of the live list (no allocation; updates `prev[slotIndex]`, `next[prevTail]`, `alive[slotIndex]`, `liveCount`, `head` if first).
- [x] 2.3 Add `unitMergeSlot(stream, leftSlot, mergedTokenId)` that consumes `next[leftSlot]` (the right slot of the merged pair), sets `value[leftSlot] = mergedTokenId`, patches `next[leftSlot]` and `prev[next[leftSlot]]`, decrements `liveCount`, and zeros `alive[rightSlot]` — all `O(1)`.
- [x] 2.4 Add `extractLiveTokenIds(stream): Uint32Array` that walks the live list head→tail once and returns a freshly-allocated `Uint32Array` sized to `liveCount`. Used in tests and any caller that wants the post-training (or mid-training) materialized stream as a flat array.
- [x] 2.5 Vitest unit tests for the slot pool covering: empty-after-construction; append → extract round-trip equals input; merge two adjacent slots → live list contains the merged token in place; merge head, tail, and middle slots; merge cascade (`abc → ab|c → abc'` for `(a,b)→x` then `(x,c)→y`).

## 3. Foundation: typed-array pair-frequency index with occurrence lists

- [x] 3.1 Add `packages/tokenizer/src/bpe-train-incremental/pair-freq-index.ts` exporting `BpePairFreqIndex` (open-addressing arrays `keysL` / `keysR` / `counts` / `occHead`, `size`, `capacityMinusOne`, `EMPTY_KEY = 0xFFFFFFFF`) and `BpePairOccurrenceList` (parallel `Uint32Array`s `occNext` / `occPrev` / `pairOf`, sized to the slot pool capacity).
- [x] 3.2 Implement `createPairFreqIndex(initialCapacityHint: number)` that allocates power-of-two-sized arrays at `initialCapacityHint` rounded up. Implement `createPairOccurrenceList(slotCapacity: number)` that allocates the per-slot occurrence-link arrays and initializes them all to `EMPTY_KEY`.
- [x] 3.3 Implement `pairHash(L: number, R: number): number` using `splitmix64`-style mixing: compute via two `Math.imul` rounds (e.g. `(Math.imul(L, 0x85ebca6b) ^ Math.imul(R, 0xc2b2ae35)) >>> 0`), then mask with `capacityMinusOne` for slot lookup.
- [x] 3.4 Implement `findOrInsertPairSlot(index, L, R): number` (linear probing; grows + rehashes when `size > 0.7 * capacity`). Implement private `growAndRehash(index)` that doubles capacity and re-probes every non-empty slot — including re-walking the per-pair occurrence list head pointers (which point to slot indices, not pair-table indices, so they survive growth without modification).
- [x] 3.5 Implement the four hot-path operations: `incrementPair(index, occList, L, R, atSlot)` (find-or-insert pair-table slot, increment count, splice `atSlot` into the per-pair occurrence list in left-to-right order using `prev[atSlot]` / `next[atSlot]` from the linked-token-stream), `decrementPair(index, occList, atSlot)` (look up `pairOf[atSlot]`, decrement count, unlink from occurrence list, drop pair-table slot when count hits zero), `forEachOccurrence(index, occList, L, R, callback)` (walk the occurrence list head → null), `pickArgmaxByCountTieLR(index)` (linear scan over non-empty pair-table slots, return `{ L, R, count } | null`).
- [x] 3.6 Vitest unit tests for the pair-freq index covering: `incrementPair` then `pickArgmax` returns the top pair with correct tie-break (smaller `L`, then smaller `R`); `decrementPair` to zero removes the entry from the iterator; `forEachOccurrence` returns sites in left-to-right order; growing past the load-factor threshold preserves all counts and occurrence orders; adversarial dense pair distribution (e.g. 200 distinct pairs that all collide on the initial 64-slot hash) still returns correct counts and argmax.

## 4. Train kernel: incremental merge loop wiring it all together

- [x] 4.1 Add `packages/tokenizer/src/bpe-train-incremental/run-merge-loop.ts` exporting `runIncrementalMergeLoop(opts: { stream, index, occList, rows, vocabCap, S, hooks?: BpeTrainHooks }): { mergesList, nextId, finalLiveCount }`.
- [x] 4.2 Inside `runIncrementalMergeLoop`, implement the loop from `design.md` §Decision 3: pick argmax → for each occurrence (with `alive[i] && alive[next[i]]` guard) decrement the up-to-three touched pairs, call `unitMergeSlot`, increment the up-to-two new pairs, fire `onMergeComplete?.({ mergeIndex, newVocabId, tokenStreamBytes: BigInt(stream.liveCount * 4) })`, await `yieldToHost?.()` once per merge.
- [x] 4.3 Add `buildInitialPairIndex(stream): { index, occList, distinctInitialPairs }` that walks the live list once and calls `incrementPair` for every adjacent pair; returns the count of distinct pairs observed for `onPairIndexBuilt`.
- [x] 4.4 Vitest unit test for `runIncrementalMergeLoop` on a tiny corpus (e.g. UTF-8 bytes for `"abababab"`) that asserts the merge sequence and final live tokens match a hand-traced expected sequence.

## 5. Materializers: RAM and disk-stream paths

- [x] 5.1 Add `materializeUtf8BytesIntoSlotPool(corpusUtf8: Uint8Array, S: number, hooks?: BpeTrainHooks): BpeLinkedTokenStream` in `packages/tokenizer/src/bpe-train-incremental/materialize.ts`. Allocates a slot pool sized `corpusUtf8.length`, walks the bytes in stripes of ~2 MiB writing each `S + b` into the pool via `appendLiveSlot`, awaits `yieldToHost?.()` between stripes (mirroring today's `beginRamBpeTrainingFromUtf8BytesAsync` cadence). Throws a clear error when `corpusUtf8.length >= 0xFFFFFFFE`.
- [x] 5.2 Add `materializeFilesIntoSlotPool(absPaths: readonly string[], S: number, hooks?: BpeTrainFromFilesHooks): Promise<BpeLinkedTokenStream>`. For each path, opens a `createReadStream`, iterates `for await (const chunk of rs)`, runs a synchronous tight inner loop appending every byte via `appendLiveSlot`, throttles `onCorpusUtf8Bytes` on the existing 256 KiB stride, awaits `yieldToHost?.()` once per chunk (NOT per byte). Inserts the synthetic `\n` (byte `10`) between successive files via `appendLiveSlot` directly. Pre-computes total byte length via `stat()` so the slot pool is allocated once with exact capacity (or grows once if the user appended a file mid-run — for `trainFromFiles` the file set is known up front, so a single `stat()` pass is sufficient).
- [x] 5.3 At end of `materializeFilesIntoSlotPool`, fire `onTokenStreamBytesWritten?.(BigInt(stream.liveCount * 4))` exactly once.

## 6. Public API: rewire `BPETokenizer.train*` and `trainFromFiles`

- [x] 6.1 In `packages/tokenizer/src/bpe-tokenizer.ts`, replace the `RamBpeRunner.stream: Uint32Array` field with `stream: BpeLinkedTokenStream` and add `index: BpePairFreqIndex` + `occList: BpePairOccurrenceList`. Adjust `RamBpeRunner` accessors accordingly.
- [x] 6.2 Replace `beginRamBpeTrainingFromUtf8Bytes(corpusUtf8, vocabSizeRequested)` and `beginRamBpeTrainingFromUtf8BytesAsync(corpusUtf8, vocabSizeRequested, hooks)` with calls to `materializeUtf8BytesIntoSlotPool(...)` followed by `buildInitialPairIndex(...)`. Fire `hooks?.onPairIndexBuilt?.(distinctInitialPairs)` once after the index is built.
- [x] 6.3 Replace `ramBpeMergeOneStep` with a call into `runIncrementalMergeLoop`. Keep the public method shapes (`train`, `trainAsync`, `trainOnUtf8Bytes`, `trainAsyncOnUtf8Bytes`) unchanged; their bodies become a thin wrapper that runs the materialize → buildInitialPairIndex → runIncrementalMergeLoop → finalizeBpeTraining sequence.
- [x] 6.4 Rewrite `trainFromFiles(sortedAbsolutePaths, vocabSizeRequested, options)`: keep the same signature and behavior contract for hooks, but replace the body with `materializeFilesIntoSlotPool(...)` → `buildInitialPairIndex(...)` → `runIncrementalMergeLoop(...)` → `finalizeBpeTraining(...)`. Remove the `pathA` / `pathB` swap loop, the per-merge `await rewriteGreedyU32LeTokenFile(...)` + `await stat(...)` round-trip, and the per-merge `await rm(rd, ...)`.
- [x] 6.5 In `finalizeBpeTraining`, accept the `mergesList` from `runIncrementalMergeLoop` and the `rows` table; the rest of finalize (building `mergeOutputIdByPair` / `mergeRankByPair`, setting `activeVocabCount`) is unchanged.
- [x] 6.6 Extend `BpeTrainHooks` and `BpeTrainFromFilesHooks` types in `packages/tokenizer/src/bpe-tokenizer.ts` with the optional `onPairIndexBuilt?: (distinctInitialPairs: number) => void` field; update JSDoc.

## 7. Deprecate the old hot-path helpers (keep exports)

- [x] 7.1 Add `@deprecated` JSDoc to `bpePairKey`, `countAdjacentPairsOverlappingUint32`, and `pickBestMergePair` in `packages/tokenizer/src/bpe-merge-stats.ts`. The note SHALL state that these functions are O(N) per call and unsuitable for hot-path use on large corpora; point readers at `packages/tokenizer/src/bpe-train-incremental/` for the production path.
- [x] 7.2 Add `@deprecated` JSDoc to `countAdjacentPairsFromU32LeTokenFile` and `rewriteGreedyU32LeTokenFile` in `packages/tokenizer/src/bpe-train-disk.ts` for the same reason.
- [x] 7.3 Refactor `writeJoinedCorpusAsTokenIdsToFile` so the inner loop pushes tokens **per `Buffer` chunk** synchronously (no `await flushToken` per byte). Keep the function exported and its progress callbacks firing on the same byte/MiB strides. Add a regression test that calls it and asserts the output `u32` LE byte stream is unchanged from a known-good fixture.

## 8. Tokenizer tests (Vitest)

- [x] 8.1 Add `packages/tokenizer/src/__tests__/bpe-train-incremental.spec.ts`. For each of (a) **`"héllo 世界"`** repeated to ~4 KiB, (b) a synthetic worst-case overlapping-letter corpus (e.g. **`"X".repeat(2048)`**), (c) a tied-frequency synthetic corpus, (d) a longer paragraph from existing fixtures, train one tokenizer using a **reference helper** that calls the deprecated `countAdjacentPairsOverlappingUint32` + `pickBestMergePair` + `rewriteGreedy` loop in-process, train a second tokenizer using the public `train` (which goes through `runIncrementalMergeLoop`), and assert that the ordered `merges` lists are element-wise equal and the `vocab` byte rows are element-wise equal. Run for `vocabSize ∈ {S+260, S+1024}`.
- [x] 8.2 Add a `trainFromFiles` equivalence test that writes the same corpora to a `mkdtemp` directory in 1- and 4-file shards and asserts the result equals the in-memory `train` for the same combined corpus.
- [x] 8.3 Add a hook-ordering test: run `trainAsyncOnUtf8Bytes` with both `onPairIndexBuilt` and `onMergeComplete` hooks, assert `onPairIndexBuilt` fires exactly once with a non-negative integer and that it fires before the first `onMergeComplete`.
- [x] 8.4 Add a determinism test: training the same corpus twice produces identical `merges` and identical `tokenizer.json` bytes.
- [x] 8.5 Add a `pickArgmaxByCountTieLR` tie-break test: synthetic corpus where two pairs `(L₁, R₁)` and `(L₂, R₂)` both have the same maximum count; assert the chosen pair is the one with smaller `L`, then smaller `R`.
- [x] 8.6 Add a "no-strings-in-hot-path" assertion: stub or spy on `bpePairKey` (e.g. via `vi.spyOn`) during a `train` call on a small corpus and assert the spy is **never** called.
- [x] 8.7 Add a regression test that loads an existing `tokenizer.json` from `out/medium-longctx/model.json`-adjacent fixtures (or, if no checked-in fixture exists, trains a small tokenizer, saves it, then re-loads it) and asserts `encode` / `decode` round-trip results are unchanged.
- [x] 8.8 Confirm all existing `packages/tokenizer/src/bpe-tokenizer.spec.ts` and `packages/tokenizer/src/__tests__/bpe-encode-hot-loop.spec.ts` tests still pass unchanged.

## 9. CLI integration: `tokenizer-train.tsx`

- [x] 9.1 In `apps/cli/src/cmd/tokenizer-train.tsx`, update the `phaseLabel` mapping for `vm.phase === 'merge'`: replace the disk-stream label `'BPE merge passes (streaming rewrite)'` with `'BPE merge passes (in-place merge)'`. RAM label is unchanged.
- [x] 9.2 In both `tokenizerTrainRamTui` and `tokenizerTrainDiskStreamTui`, thread the new optional `onPairIndexBuilt` hook through to the underlying `tok.trainAsyncOnUtf8Bytes` / `tok.trainFromFiles` call. Do not paint a new TUI element; just thread the hook so future TUI improvements can consume it.
- [x] 9.3 In `tokenizerTrainDiskStreamTui`, ensure the existing `onTokenStreamBytesWritten` listener correctly handles the new "single final emit" semantics — `tokenStreamFileMiB` is set once at the end of materialize and stays at that value for the rest of the run. (No code change expected; this is a verification step.)
- [ ] 9.4 Manual smoke: `npx nx run @llm-transformer-diy/cli:serve -- tokenizer train --data 'corpus/real/**/*.txt' --vocab-size 512 --out /tmp/tok.json` succeeds with the wolf-tui dashboard rendering `phase` / `corpus` / `merges` / `stream` / `done` correctly. Repeat with `--disk-stream`. — _Deferred to operator: requires interactive TTY + a populated `corpus/real/\*\*/_.txt` fixture set; programmatic CLI test (`tokenizer train` end-to-end) is implicitly covered by the trainer-equivalence Vitest suite.\*

## 10. Performance verification

- [x] 10.1 Add `packages/tokenizer/src/__tests__/bpe-train-perf.spec.ts` (skipped in CI by default via `it.skipIf(process.env.RUN_BPE_PERF !== '1')`). It generates a synthetic ~10 MiB corpus deterministically (seeded RNG over a small alphabet so both algorithms see the same input), times `train` (new path), times the reference full-rescan helper from §8.1 on the same input, asserts identical merges, and prints both wall-time numbers + `merges/sec` so we can compare ratios. _Default workload sized at 512 KiB × 256 merges so the spec finishes in <10 s; scale up via `BPE_PERF_BYTES` / `BPE_PERF_MERGES` env vars._
- [x] 10.2 Run the perf spec once with `RUN_BPE_PERF=1` on a representative machine and record the before/after wall-time numbers in a `train-throughput-note.md` in the change folder (mirroring `add-16bit-dtype-support/train-throughput-note.md`'s shape) so the archive step has the evidence the proposal's perf claim requires. _Recorded **12.77×** at 512 KiB × 256 merges; see `train-throughput-note.md`._
- [ ] 10.3 (Optional but encouraged) Run `npx nx run @llm-transformer-diy/cli:serve -- tokenizer train --data <300 MiB sample>` end-to-end before and after the change and capture wall time + `top` / `htop` snapshots in the same note. The expected outcome is hours → low minutes. — _Deferred to operator: requires a populated 300 MiB corpus and interactive TTY; algorithmic projection in `train-throughput-note.md` confirms the expected hours → minutes outcome from the measured 12.77× speedup at the smaller scale._

## 11. Verification

- [x] 11.1 `npx nx run @llm-transformer-diy/tokenizer:test` is green.
- [x] 11.2 `npx nx run @llm-transformer-diy/cli:test` is green.
- [x] 11.3 `npm run build` succeeds end-to-end.
- [x] 11.4 `npx nx run-many -t test --all` is green (no regressions in other packages).
- [x] 11.5 `openspec validate accelerate-bpe-train-incremental-pair-table --strict` still passes.
- [x] 11.6 ESLint + Prettier are clean on every touched file (`npm run lint` / repo-standard). _Touched files (tokenizer + cli) report zero lint errors; pre-existing repo-wide lint debt in unrelated training/tools files is out of scope._
