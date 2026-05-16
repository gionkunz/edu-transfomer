## ADDED Requirements

### Requirement: Pre-tokenizer policy

**`BPETokenizer`** SHALL expose a **pre-tokenizer policy** that splits a UTF-8 input string into adjacent non-overlapping **regions** before any byte-level expansion or merge step. The policy SHALL be identified by a stable string label persisted with the tokenizer. The only label this revision supports is **`word-and-punct-v1`**; future revisions MAY add additional labels but SHALL NOT silently change the meaning of an existing label. The policy label SHALL be set at training time and SHALL NOT be reconfigurable on an already-trained tokenizer.

The **`word-and-punct-v1`** policy SHALL split using the regular expression

```
/\s+|\d+|\p{L}+|[^\s\p{L}\p{N}]+/gu
```

applied with `String.prototype.matchAll`, in that alternative order. The full input SHALL be covered by the concatenation of matches. Each match SHALL produce one **region**.

#### Scenario: word-and-punct-v1 produces the expected region partition

- **WHEN** the **`word-and-punct-v1`** pre-tokenizer is applied to the input **`"Hello, world! 42 → done"`**
- **THEN** the resulting regions, in order, SHALL be exactly **`["Hello", ",", " ", "world", "!", " ", "42", " ", "→", " ", "done"]`** (each region one of: letters, digits, whitespace, or non-letter-non-digit-non-whitespace characters; no two adjacent regions of the same kind)

---

### Requirement: Training observes per-region adjacency only

When **`train`**, **`trainAsync`**, **`trainAsyncOnUtf8Bytes`**, or **`trainFromFiles`** runs, the materialized working corpus SHALL be the concatenation of per-region UTF-8 byte expansions, and the **initial pair-frequency table** SHALL observe only adjacent pairs **`(token[i], token[i+1])`** for which **`i`** and **`i+1`** lie inside the **same region**. Pair observations that would span a region boundary SHALL be skipped.

The subsequent incremental pair-frequency maintenance (required by `Requirement: Incremental pair-frequency maintenance during training`) SHALL preserve this invariant: a merge that fires at corpus position **`i`** SHALL update the at-most-three neighboring-pair counts (`(prev, L)`, `(R, next)`, `(prev, mergedId)`, `(mergedId, next)`) only when each neighboring pair lies inside the same region as the merge site. Cross-boundary neighbors SHALL be left untouched (they were never counted, so they cannot decrement; the merge SHALL NOT introduce a cross-boundary pair).

A merge rule **`(L, R) → mergedId`** SHALL never fire at a cross-region position during subsequent encodes (this is a consequence of the encode requirement below).

#### Scenario: Training does not learn cross-region merges

- **WHEN** a tokenizer is trained on a corpus of the form **`"AB AB AB ..."`** (where `AB` is a letter region and ` ` is a whitespace region, repeated many times) until vocabulary is exhausted
- **THEN** the learned merges SHALL NOT contain any merge whose left-side byte sequence ends in a letter byte and whose right-side byte sequence begins with the space byte (i.e. no `"B "` merge), and the **`encode`** of **`"AB AB"`** SHALL contain at least one occurrence of the space byte id between the two `AB` regions

---

### Requirement: Encode folds over pre-tokenized regions

**`encodeUint32`** and **`encodeUint32Async`** SHALL split the input via the tokenizer's active pre-tokenizer policy, encode each region independently using the deterministic merge-table algorithm (`Requirement: Deterministic encoding with merge table`), and return the concatenation of the per-region id arrays in input order. No merge rule SHALL match an adjacent pair whose two ids come from two different regions.

Per-region encode SHALL invoke the same byte-level expansion (UTF-8 via `TextEncoder`, ids `S + b`) and the same outer best-rank pass loop currently used for the whole corpus. The pair-lookup hot path SHALL satisfy the packed-key invariant below.

#### Scenario: Encode round-trip preserves the input

- **WHEN** a trained tokenizer runs **`decode(encodeUint32(x))`** for **`x = "héllo 世界 1234!"`**
- **THEN** the result SHALL equal **`x`**

#### Scenario: Concatenation of per-region encodes equals whole-input encode

- **WHEN** a trained tokenizer encodes input **`x`** via **`encodeUint32(x)`** and separately encodes each region (returned by the tokenizer's own pre-tokenize step) via **`encodeUint32(region)`** and concatenates those typed arrays
- **THEN** the two **`Uint32Array`** results SHALL be element-wise equal

#### Scenario: Per-region encode is bounded by region size, not corpus size

- **WHEN** a trained tokenizer encodes the inputs **`x`** and **`x ++ x`** (the same input concatenated with itself), and the test harness records the maximum **`mergeIteration`** observed across all per-region encodes
- **THEN** the maximum per-region outer-pass count for **`x ++ x`** SHALL equal the maximum per-region outer-pass count for **`x`** (doubling the corpus does not increase the per-region pass count)

---

### Requirement: Packed-key pair lookup tables for encode hot path

The encode pair-lookup tables (`mergeRankByPair`, `mergeOutputIdByPair`) SHALL be implemented as single-level **`Map<number, number>`** keyed by the packed integer **`L * vocabSize + R`**, where `L` and `R` are the left and right token ids of a learned merge and `vocabSize` is the final trained vocabulary size. The encode hot path SHALL perform exactly one **`Map.get`** call per adjacent pair probe.

For all `vocabSize ≤ 94 906 265` (i.e. `vocabSize ≤ floor(sqrt(2^53))`) the packed-key Map SHALL be used. Implementations SHALL reject training and loading at a `vocabSize` beyond that boundary with a descriptive error.

#### Scenario: Encode hot path uses a single Map.get per probe

- **WHEN** a microbenchmark instruments **`encodeUint32`** on a representative input and counts **`Map.prototype.get`** invocations on the pair-rank table inside one outer pass
- **THEN** the recorded count SHALL be approximately equal to the number of adjacent pairs scanned in that pass (one **`get`** per probe), and SHALL NOT scale linearly with the depth of any nested-Map structure (no `m.get(L)?.get(R)` two-step pattern is present on the hot path)

---

### Requirement: JSON schema version 2 persists the pre-tokenizer policy

**`save`** SHALL write tokenizer JSON with **`version: 2`** and a required top-level field **`pretokenizer`** whose value is the active policy label string (currently the only valid value is **`"word-and-punct-v1"`**). The remaining schema fields (**`kind`**, **`specialTokens`**, **`byteOffset`**, **`merges`**, **`vocab`**) SHALL be preserved unchanged.

**`load`** SHALL accept exactly **`version: 2`** files whose **`pretokenizer`** value is a known label, and SHALL throw a descriptive error otherwise. In particular:

- any **`version`** value other than **`2`** (including the obsolete **`1`**) SHALL be rejected with a message that names the supported version and instructs the user to retrain;
- a missing **`pretokenizer`** field on a **`version: 2`** file SHALL be rejected;
- a **`pretokenizer`** value that is not in the implementation's known-label set (in particular **`null`** and any unrecognized string) SHALL be rejected with a message naming the offending value and the supported set.

A round-trip of **`load`** then **`save`** SHALL preserve the in-memory pre-tokenizer policy.

#### Scenario: Save writes version 2 with the active policy

- **WHEN** a tokenizer trained under **`pretokenizer: 'word-and-punct-v1'`** is saved
- **THEN** the JSON SHALL have **`version === 2`** and **`pretokenizer === 'word-and-punct-v1'`**

#### Scenario: Legacy v1 file is rejected with a retrain message

- **WHEN** a tokenizer JSON written by the pre-change implementation (no **`pretokenizer`** field, **`version: 1`**) is loaded
- **THEN** **`load`** SHALL throw an error whose message names the unsupported version and the required action (retrain)

#### Scenario: Unknown pre-tokenizer string is rejected

- **WHEN** a tokenizer JSON with **`version: 2`** and **`pretokenizer: "future-policy-v3"`** is loaded by an implementation that does not recognize that label
- **THEN** **`load`** SHALL throw a descriptive error naming the unknown policy string and the supported set

#### Scenario: Null pre-tokenizer is rejected

- **WHEN** a tokenizer JSON with **`version: 2`** and **`pretokenizer: null`** is loaded
- **THEN** **`load`** SHALL throw the same "unknown pretokenizer" error as for any other unsupported value

---

### Requirement: Encode progress reports per-region context

**`BpeEncodeProgressEvent`** SHALL carry two required fields on every **`phase: 'merge'`** emission:

- **`regionIndex: number`** — zero-based index of the region currently being encoded.
- **`regionsTotal: number`** — total number of regions in the current encode call, determined once after pre-tokenization.

The existing **`mergeIteration`** SHALL denote outer best-rank passes within the **current region** (not aggregated across regions). When a region completes, the encoder SHALL emit a **`'merge'`** event with the final per-region **`mergeIteration`** before moving to the next region. The **`tokenCount`** in any merge event SHALL be the running aggregate token count across all regions encoded so far in the same call (so callers can compute throughput against a stable corpus-wide denominator).

#### Scenario: Region indices increase monotonically

- **WHEN** **`encodeUint32Async`** runs on an input that produces at least three regions and emits progress events
- **THEN** the sequence of **`regionIndex`** values reported in **`'merge'`** events SHALL be monotonically non-decreasing and SHALL eventually reach **`regionsTotal - 1`**
