## 1. Scaffold and exports

- [x] 1.1 Add `bpe-tokenizer.ts` beside `CharTokenizer`; export `BPETokenizer`, `BPETokenizerOptions` from `packages/tokenizer/src/index.ts`
- [x] 1.2 Mirror `CharTokenizer` errors for untrained use (`encode`/`decode`)

## 2. Core data structures

- [x] 2.1 Implement id→bytes `vocab` map and contiguous id allocation (`specials`, then bytes `S..S+255`, then merges)
- [x] 2.2 Build merge registry: `[leftId, rightId]` list + **`mergeRank`** index + **`pair -> newId`** lookup for **`encode`**
- [x] 2.3 Encode validation helpers reused by **`specialTokens`** (match `CharTokenizer` rules)

## 3. Training loop

- [x] 3.1 Corpus stream: **`TextEncoder`** → initial ids; sliding adjacency counting with overlaps
- [x] 3.2 Merge selection: max frequency **≥ 2**, tie-break `(L, R)` ascending by id
- [x] 3.3 Corpus rewrite after each merge: greedy **non-overlapping** left-to-right substitution of **`L,R`**
- [x] 3.4 Respect **`vocabSize`** ceiling and **`vocabSize < S + 256`** floor errors; **`vocabSize` property** reflects final reachable ids (**contiguous**)

## 4. Encode and decode

- [x] 4.1 **`encode`**: iterative merge reduction using minimal **`mergeRank`**, ties by id tuple, replace **leftmost** match per step
- [x] 4.2 **`decode`**: concatenate byte sequences strictly; **`TextDecoder`** with fatal UTF-8; unknown ids throw

## 5. Persistence

- [x] 5.1 **`save`**: write JSON **`kind`** / **`version`**, **`specialTokens`**, **`byteOffset`**, **`merges`**, **`vocab`** as specified in **`spec.md`**
- [x] 5.2 **`load`**: validate shape, rebuild maps/ranks on fresh instance; unsupported **`kind`/`version`** throw

## 6. Acceptance tests (`packages/tokenizer`)

- [x] 6.1 **`bpe-tokenizer.spec.ts`**: **`vocabSize` floor** rejection, save/load parity, malformed JSON paths
- [x] 6.2 Encode/decode round-trip on **`"héllo 世界"`** (and at least one short ASCII corpus)
- [x] 6.3 Determinism golden: fixed corpus + **`vocabSize`** yields identical **`merges`/`encode`** across two instances
