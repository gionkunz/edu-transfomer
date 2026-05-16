## 1. Foundation: pre-tokenizer + tokenizer state

- [x] 1.1 Add a `PRETOKENIZER_REGEX_WORD_AND_PUNCT_V1` constant in `packages/tokenizer/src/bpe-tokenizer.ts` set to `/\s+|\d+|\p{L}+|[^\s\p{L}\p{N}]+/gu`, and a `PretokenizerPolicy` type narrowed to **`'word-and-punct-v1'`** (no `null` legacy value).
- [x] 1.2 Add `splitIntoRegions(text: string, policy: PretokenizerPolicy): readonly string[]` exported helper that returns the matches of the regex via `text.matchAll(...)` mapped to `m[0]`. The function SHALL be pure and SHALL allocate one fresh array.
- [x] 1.3 Add `private pretokenizerPolicy: PretokenizerPolicy` to `BPETokenizer`. Initialize to `'word-and-punct-v1'`. Trained tokenizers SHALL set it based on the train-call output; loaders SHALL set it from the JSON manifest (see 4.x).
- [x] 1.4 Unit-test `splitIntoRegions` for: empty input, pure whitespace, mixed letters/digits/punct/whitespace, multi-codepoint emoji, the round-trip property `regions.join('') === text` for a random fuzz of UTF-8 strings.

## 2. Training under a pre-tokenizer

- [x] 2.1 Remove the prior `pretokenizer?: PretokenizerPolicy` option from `BpeTrainOptions` and `BpeTrainFromFilesOptions`: the policy is fixed at `'word-and-punct-v1'`.
- [x] 2.2 In the corpus-materialization step, pre-tokenize the input into regions, emit per-region byte sequences into the working buffer in order, and record a parallel `Uint8Array` (`regionBoundary`) marking the LAST byte of each region. Use this array to skip pair observations at boundaries during the initial scan.
- [x] 2.3 Guard the per-iteration `(prev, L)`, `(R, next)`, `(prev, mergedId)`, `(mergedId, next)` neighbour updates on the `regionBoundary` bit. The merge loop's `regionBoundary` parameter is now **required** (passed as an all-zero bitmap when callers want to disable region gating, e.g. in low-level unit tests).
- [x] 2.4 Plumb the policy into `BPETokenizer.train{,Async,AsyncOnUtf8Bytes,FromFiles}` so they store the policy on `this.pretokenizerPolicy` and pass it through to the pair-index build.
- [x] 2.5 Train-incremental reference rescan helper updated to fold over regions, matching the new mandatory pre-tokenization (otherwise the reference would diverge).
- [x] 2.6 Pretokenize unit test in `packages/tokenizer/src/__tests__/bpe-pretokenize.spec.ts` asserts no learned merge crosses a region boundary.

## 3. Encode: packed-key tables + per-region fold

- [x] 3.1 Replace `private mergeRankByPair: Map<number, Map<number, number>>` and `private mergeOutputIdByPair: Map<number, Map<number, number>>` with single-level `private mergeRankByPackedPair: Map<number, number>` and `private mergeOutputIdByPackedPair: Map<number, number>`. Keys are `L * this.vocabSize + R`.
- [x] 3.2 Add a private `packPairKey` helper plus an assert that both args are `< this.vocabSize`. Replace every nested-Map lookup with the inlined packed-key `Map.get` on the hot path.
- [x] 3.3 After training (and after `load`), rebuild the packed-key tables from the `merges` and `vocab` data in a single pass. Drop the `nestedNumberSet` / `nestedNumberGet` helpers.
- [x] 3.4 Refactor `encodeUint32` to: (a) call `splitIntoRegions(text, this.pretokenizerPolicy)`, (b) for each region, run the existing byte-expand + outer-pass loop on a per-region working buffer, (c) concatenate the per-region `Uint32Array` outputs into a single final `Uint32Array` of the right total length.
- [x] 3.5 Refactor `encodeUint32Async` to the same per-region fold, preserving the existing wall-time yield cadence (`TIME_SLICE_ENCODE_ASYNC_MS`) and the coarse outer-pass-count cadence (`EMIT_PASS_EVERY_ASYNC`).
- [x] 3.6 Extend `BpeEncodeProgressEvent` with **required** `regionIndex` and `regionsTotal` fields on every `'merge'` emission.

## 4. JSON schema v2 + load policy

- [x] 4.1 `save()` always emits `version: 2` and a top-level `pretokenizer: 'word-and-punct-v1'` field.
- [x] 4.2 `load()` rejects any `version` other than `2` with a descriptive "retrain the tokenizer" error.
- [x] 4.3 `load()` requires the `pretokenizer` field and rejects unknown labels (including `null`) with a descriptive error naming the supported set.
- [x] 4.4 Schema-rejection tests cover: missing `pretokenizer` field, unknown label, `null` label, obsolete `version: 1`.
- [x] 4.5 The hand-crafted fixtures in `packages/tokenizer/src/bpe-tokenizer.spec.ts` are upgraded to `version: 2` with an explicit `pretokenizer` field so the unrelated validation paths still execute.

## 5. Encode equivalence + bounded-passes property tests

- [x] 5.1 Concat-of-per-region-encodes == whole-input encode. _(Covered by `bpe-pretokenize.spec.ts > BPE encode equivalence under word-and-punct-v1`.)_
- [x] 5.2 "Doubling the input does not increase max per-region pass count" property test. _(Same spec file.)_
- [x] 5.3 Hot-loop equivalence (`bpe-encode-hot-loop.spec.ts`) updated to consume the narrowed `PretokenizerPolicy` and continues to pass.

## 6. CLI train dashboard wiring

- [x] 6.1 `EncodeProgressSnapshot` in `apps/cli/src/cmd/train-dash-types.ts` drops the legacy `pessimistic*` / `mergesPerSec` / `mergeProgressPct` fields in favor of `regionsPerSec`, `etaSec`, required `regionIndex` / `regionsTotal`, and `meanPassesPerRegion`. `pretokenizerPolicy` is narrowed to `'word-and-punct-v1'`.
- [x] 6.2 `apps/cli/src/cmd/train.tsx` now tracks `completedRegions`, `cumulativePassesAcrossRegions`, `lastPassCountForRegion`, and emits a region-aware ETA via `perRegionPessimisticEta`.
- [x] 6.3 `apps/cli/src/cmd/encode-eta.ts` keeps only `perRegionPessimisticEta`; all whole-corpus pessimistic helpers (`pessimisticMergeUpperBound`, `pessimisticMergeRemaining`, `pessimisticEtaFromRate`, `mergeProgressPct`) are removed.
- [x] 6.4 `apps/cli/src/cmd/train-dashboard.tsx` shows the policy label, region progress (`region A / B · pass #N`), mean passes/region, and the new ETA copy. The legacy "no pre-tokenization" branch is removed.
- [x] 6.5 `apps/cli/src/cmd/encode-eta.spec.ts` covers boundary inputs for the per-region ETA function.

## 7. Docs

- [x] 7.1 `docs/project.md` BPE Tokenizer section describes the mandatory `word-and-punct-v1` policy, the `version: 2`-only schema, and the explicit retrain requirement for pre-change tokenizers.
- [x] 7.2 `README.md` BPE encode caveat states that older tokenizer JSON files no longer load and must be retrained.

## 8. Verification

- [x] 8.1 Run `npx nx run-many -t test --all`; all 9 projects pass.
- [x] 8.2 Run `npx nx run @llm-transformer-diy/tokenizer:test`; 101 tokenizer specs pass (1 perf spec opt-in skipped).
- [ ] 8.3 Smoke-encode a 1 MB Shakespeare corpus and a 10 MB Cosmopedia shard with a freshly trained `word-and-punct-v1` tokenizer; record wall-clock encode times in a short note for future archive.
- [x] 8.4 Run `openspec validate speed-up-bpe-encode` — change validates cleanly.
