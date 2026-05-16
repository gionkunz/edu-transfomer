## MODIFIED Requirements

### Requirement: Deterministic byte-level BPE training

**`train`** SHALL encode `text` with **`TextEncoder`** to bytes, map each byte **`b`** to id **`S + b`**, and treat the resulting id sequence as the initial corpus stream. The merge statistics it operates on SHALL be the count, for each ordered adjacent pair **`(token[i], token[i + 1])`** for all valid **`i`** in the **current** corpus stream, of the number of positions **`i`** at which that pair occurs (**overlapping positions each contribute one pair observation**).

Training iterations SHALL proceed until **either** the number of distinct ids reached equals **`vocabSize`**, **or** every adjacent pair currently present in the corpus stream has total count **≤ 1**.

Each iteration SHALL:

1. Choose pair **`(L, R)`** with maximum total count among pairs with count **≥ 2**.

2. Break frequency ties by smaller **`L`**, then smaller **`R`** (**numeric id order**).

3. Create the next id whose byte sequence is the concatenation of the byte sequences for **`L`** and **`R`**.

4. Update the corpus id stream by replacing **non-overlapping** occurrences of **`L` followed immediately by `R`** with the new id, in **left-to-right order** (**greedy left-to-right pass per iteration**). The set of merge sites within a single iteration SHALL be the same set a left-to-right scan would identify when, on each match, the read cursor advances past both consumed positions before considering further matches.

If **`vocabSize` exceeds** the vocabulary reachable when no further merges qualify, training SHALL stop early; **`vocabSize` property** SHALL equal the final number of ids actually defined (**contiguous `0 .. vocabSize - 1`**).

This requirement describes the **observable training trace** (the ordered sequence of merge pairs and the resulting vocabulary). It does **not** prescribe the data structure used to count adjacent pairs or to apply the per-iteration rewrite. Implementations MAY maintain pair-frequency counts and merge sites incrementally across iterations (see _Incremental pair-frequency maintenance during training_), provided the observable trace per iteration is identical to the trace a full-corpus rescan + left-to-right rewrite would produce on the same corpus state.

#### Scenario: Training is reproducible for a fixed corpus

- **WHEN** two fresh tokenizer instances with identical options call **`train`** on the same `text` and `vocabSize`
- **THEN** resulting **`vocabSize`**, ordered **`merges`**, and **`encode`** outputs for a fixed probe string SHALL be identical

#### Scenario: Training trace matches a full-rescan reference for representative inputs

- **WHEN** a fresh tokenizer trains on each of (a) a synthetic worst-case overlapping-letter corpus (e.g. **`"XXXXXXXXXX"`**), (b) a tied-frequency synthetic corpus, (c) a multibyte UTF-8 corpus (e.g. **`"héllo 世界"`** repeated), with **`vocabSize ∈ {S+260, S+1024}`**
- **THEN** the resulting ordered **`merges`** list and the **`vocab`** byte rows SHALL be element-wise equal to those produced by a reference implementation that, on every iteration, recomputes the full pair-frequency map by scanning the entire current corpus stream end-to-end and applies a single greedy left-to-right rewrite pass

---

## ADDED Requirements

### Requirement: Incremental pair-frequency maintenance during training

**`BPETokenizer`** training (`train`, `trainAsync`, `trainAsyncOnUtf8Bytes`, `trainFromFiles`) SHALL maintain pair-frequency counts incrementally across iterations rather than recomputing them by scanning the entire current corpus stream on every iteration. Concretely, the implementation SHALL satisfy the following invariants:

1. **Initial scan is `O(N)`.** After the corpus is materialized into the working representation, the implementation SHALL build the initial pair-frequency table in a single pass over the corpus stream (one observation per adjacent pair).

2. **Per-iteration update is `O(K_m + P_m)`.** On each iteration that merges a pair `P = (L, R)` whose current count is `K_m`, the implementation SHALL:
   - Enumerate the `K_m` merge sites in left-to-right corpus order without re-scanning the entire stream.
   - At each merge site, mutate at most the **four** adjacent pair counts that touch the merged slot pair (the chosen pair itself, plus the at-most-three neighboring pairs `(prev, L)`, `(R, next)`, `(prev, mergedId)`, `(mergedId, next)`).
   - Select the next-iteration argmax in time bounded by the number `P_m` of distinct pairs currently in the table.

3. **Total work bound.** End-to-end training cost SHALL be `O(N + Σ_m (K_m + P_m))`, where `N` is the materialized corpus length, `K_m` is the count of the pair chosen at iteration `m`, and `P_m` is the number of distinct pairs in the table at the start of iteration `m`. Implementations SHALL NOT perform a full corpus-stream scan on every iteration.

4. **Trace equivalence.** The ordered `merges` list, the `vocab` byte rows, and the final `vocabSize` produced by training SHALL be element-wise equal to those produced by the reference full-rescan-and-rewrite algorithm on the same corpus and `vocabSize`.

5. **No per-pair string allocation on the hot path.** The pair-frequency table SHALL be keyed numerically (e.g. by `(L, R)` packed into a `BigInt`, or by parallel `Uint32Array`s), not by a fresh string allocated per adjacency. Implementations MAY retain string-keyed helpers for testing or external tooling but SHALL NOT use them in the per-iteration hot path of `train` / `trainAsync` / `trainAsyncOnUtf8Bytes` / `trainFromFiles`.

#### Scenario: Per-iteration cost is bounded by the chosen pair's occurrence count, not the corpus size

- **WHEN** a fresh tokenizer trains on a synthetic corpus shaped so that early merges have `K_m ≪ N` (e.g. a long random byte stream with a few injected high-frequency bigrams), and the test harness records wall time per iteration
- **THEN** the per-iteration wall time for those early merges SHALL scale with `K_m` (the chosen pair's count), not with the full corpus length `N`, when `N` is doubled while keeping `K_m` constant; concretely, doubling `N` SHALL NOT roughly double per-iteration wall time

#### Scenario: Hot-path pair-frequency keys are not `bpePairKey`-style strings

- **WHEN** training runs on any corpus
- **THEN** the per-iteration hot path SHALL NOT call **`bpePairKey`** (or any equivalent function that allocates a fresh JS string per adjacent pair); the per-pair table SHALL be keyed by numeric `(L, R)` pairs

---

### Requirement: Disk-streamed training materializes the corpus once

**`BPETokenizer.trainFromFiles`** SHALL materialize the joined corpus into the working representation in a **single streaming pass** over the input files (one read of the corpus, one synthetic UTF-8 `\n` (byte `10`) inserted between successive files), and SHALL run all subsequent merge iterations against the in-memory working representation. It SHALL NOT rewrite the corpus to disk on each iteration.

The streaming materializer SHALL push tokens into the working representation in **bulk per stream chunk** (one synchronous loop per `Buffer` chunk emitted by the read stream, with at most one cooperative event-loop yield per chunk). It SHALL NOT perform an `await` per byte or per token.

The materializer's user-visible progress SHALL continue to be reported via the existing throttled cadence: **`onCorpusUtf8Bytes`** (~256 KiB stride during reads; final call is the exact total) and **`onTokenStreamBytesWritten`** (final call equal to the materialized working representation's logical `u32`-equivalent byte size, i.e. **`liveCount * 4`**). The temp directory created by `trainFromFiles` SHALL be removed on success and on failure (existing best-effort behavior preserved).

#### Scenario: Disk-streamed training does not rewrite the corpus per merge

- **WHEN** **`trainFromFiles`** runs on a multi-file corpus to a `vocabSize` that requires many merges
- **THEN** the temp directory SHALL contain at most the materialized working representation's auxiliary state (and SHALL be cleaned up on completion); no per-merge rewritten token-stream files SHALL be created and deleted iteration by iteration

#### Scenario: Disk-streamed materializer issues at most one event-loop yield per stream chunk

- **WHEN** **`trainFromFiles`** materializes a corpus from a `createReadStream`-backed source whose chunks are observed at the default ~64 KiB granularity, and the test harness counts `await` resumptions between chunk boundaries
- **THEN** the number of `await` resumptions during materialization SHALL be on the order of (chunk count) rather than (token count) — concretely, at most one resumption per delivered `Buffer` chunk

---

### Requirement: Optional `onPairIndexBuilt` training hook

**`BPETokenizer`** training hook types (**`BpeTrainHooks`** and **`BpeTrainFromFilesHooks`**) SHALL expose an optional **`onPairIndexBuilt(distinctInitialPairs: number): void`** callback. When provided, this callback SHALL fire **exactly once** per training run, immediately after the initial pair-frequency table has been built from the materialized corpus (i.e. after the `O(N)` initial scan and before the first merge iteration). The callback's argument SHALL be the count of distinct adjacent pairs observed in the initial scan.

This hook is purely informational; the absence of a callback SHALL NOT change training behavior or output. All existing training hooks (**`onCorpusUtf8Bytes`**, **`onMergeComplete`**, **`onTokenStreamBytesWritten`**, **`yieldToHost`**) SHALL retain their current contracts.

#### Scenario: `onPairIndexBuilt` fires exactly once before the first merge

- **WHEN** **`trainAsyncOnUtf8Bytes`** or **`trainFromFiles`** is called with hooks containing both **`onPairIndexBuilt`** and **`onMergeComplete`**, on a corpus that yields at least one merge
- **THEN** **`onPairIndexBuilt`** SHALL be invoked exactly once with a non-negative integer, and SHALL be invoked **before** the first **`onMergeComplete`** invocation

#### Scenario: Omitting `onPairIndexBuilt` does not affect training output

- **WHEN** two fresh tokenizers train on the same corpus and `vocabSize`, one with hooks containing **`onPairIndexBuilt`** and one without
- **THEN** the resulting **`vocabSize`**, ordered **`merges`**, and **`encode`** outputs for a fixed probe string SHALL be identical between the two
