## ADDED Requirements

### Requirement: Implements shared Tokenizer contract

An implementation **`BPETokenizer`** SHALL satisfy the **`Tokenizer`** interface from **`packages/tokenizer`**: **`train(text, vocabSize)`**, **`encode(text)`**, **`decode(ids)`**, **`save(path)`**, **`load(path)`**, and **`readonly vocabSize`** after successful training.

#### Scenario: Untrained misuse is rejected

- **WHEN** **`encode`** or **`decode`** is called before **`train`** completes successfully
- **THEN** the call SHALL throw an error that states the tokenizer is not trained

---

### Requirement: Special tokens occupy stable low ids

**`BPETokenizer`** MUST accept **`specialTokens`**: a non-empty list of distinct non-empty Unicode strings with the same validation rules as **`CharTokenizerOptions.specialTokens`**. Each special string SHALL be assigned id **`i`** for index **`i`**, before any byte-derived tokens are introduced.

#### Scenario: Invalid special token configuration is rejected

- **WHEN** **`specialTokens`** is empty, contains an empty string, or contains duplicates
- **THEN** construction or **`train`** SHALL throw a descriptive error before emitting a trained tokenizer

---

### Requirement: Byte singleton id space

Let **`S = specialTokens.length`**. For every raw byte value **`b` in `0..255`**, the tokenizer SHALL assign the singleton token id **`S + b`** whose decoded bytes are exactly **`[b]`**. The initial training vocabulary SHALL contain ids **`0 .. S + 255`** inclusive (contiguous).

#### Scenario: Requested vocabulary smaller than required floor

- **WHEN** **`train`** is called with **`vocabSize < S + 256`**
- **THEN** **`train`** SHALL throw because the tokenizer cannot represent all UTF-8 bytes distinctly

---

### Requirement: Deterministic byte-level BPE training

**`train`** SHALL encode `text` with **`TextEncoder`** to bytes, map each byte **`b`** to id **`S + b`**, and build merge statistics on the resulting id sequence by counting each adjacent pair **`(token[i], token[i + 1])`** for all valid **`i`** (**overlapping positions each contribute one pair observation**).

Training iterations SHALL proceed until **either** the number of distinct ids reached equals **`vocabSize`**, **or** every observed adjacent pair has total count **≤ 1** across the current corpus stream.

Each iteration SHALL:

1. Choose pair **`(L, R)`** with maximum total count among pairs with count **≥ 2**.

2. Break frequency ties by smaller **`L`**, then smaller **`R`** (**numeric id order**).

3. Create the next id whose byte sequence is the concatenation of the byte sequences for **`L`** and **`R`**.

4. Update the corpus id stream by scanning **left to right** and replacing **non-overlapping** occurrences of **`L` followed immediately by `R`** with the new id (**greedy left-to-right pass per iteration**).

If **`vocabSize` exceeds** the vocabulary reachable when no further merges qualify, training SHALL stop early; **`vocabSize` property** SHALL equal the final number of ids actually defined (**contiguous `0 .. vocabSize - 1`**).

#### Scenario: Training is reproducible for a fixed corpus

- **WHEN** two fresh tokenizer instances with identical options call **`train`** on the same `text` and `vocabSize`
- **THEN** resulting **`vocabSize`**, ordered **`merges`**, and **`encode`** outputs for a fixed probe string SHALL be identical

---

### Requirement: Deterministic encoding with merge table

**`encode`** SHALL map input text to the same initial byte-id sequence as training (UTF-8 via **`TextEncoder`**, ids **`S + b`**), then repeatedly apply learned merges until no merge rule matches an adjacent pair:

1. Among all adjacent pairs present in the sequence that correspond to a learned merge, select the pair whose saved **merge rank** (**0-based order merges were created**) is **minimal**.

2. If several candidate pairs share the minimal rank, select the pair with smaller left id, then smaller right id.

3. Replace the **leftmost** occurrence of the selected pair with its merged id.

4. Repeat until no merge applies.

#### Scenario: Encode after training round-trips through decode for representative text

- **WHEN** a trained tokenizer runs **`decode(encode(x))`** for **`x`** equal to a short Unicode string containing spaces, digits, and non-ASCII characters (e.g. **`"héllo 世界"`**)
- **THEN** the result SHALL equal **`x`**

---

### Requirement: Decode maps ids to strict UTF-8 text

**`decode`** SHALL map each id to its byte sequence via the learned **`vocab`**, concatenate those bytes in order, and decode with **strict UTF-8** (**ill-formed byte sequences SHALL throw**). Unknown ids SHALL throw.

#### Scenario: Unknown id is rejected

- **WHEN** **`decode`** receives an id **≥** trained **`vocabSize`** or an id never defined in **`vocab`**
- **THEN** it SHALL throw a descriptive error

---

### Requirement: Versioned JSON persistence

**`save`** SHALL write UTF-8 JSON containing at minimum:

- **`kind`**: literal **`"llm-transformer-diy/bpe-tokenizer"`**

- **`version`**: positive integer (**initially `1`**)

- **`specialTokens`**: array of strings echoing configuration order

- **`byteOffset`**: number **`S`**

- **`merges`**: array of length-2 numeric arrays **`[leftId, rightId]`** in merge creation order

- **`vocab`**: object whose keys are decimal stringified ids and values are JSON arrays of byte values **`0..255`**

**`load`** SHALL parse the file, verify **`kind`** and supported **`version`**, reconstruct internal tables, and set **`vocabSize`** to the number of defined ids. Unsupported **`kind`**, **`version`**, or malformed shapes SHALL throw.

#### Scenario: Save and load round-trip preserves behavior

- **WHEN** a trained tokenizer **`save`**s to a temp path and a new instance **`load`**s that path
- **THEN** **`vocabSize`**, **`encode`**, and **`decode`** SHALL match the pre-save instance on the same probes

---

### Requirement: Automated acceptance coverage

Implementation work SHALL ship **Vitest** coverage in **`packages/tokenizer`** that exercises at minimum: **`train`** floor error (`vocabSize` too small), encode/decode round-trip, deterministic tie corpus (golden expectations), malformed **`load`** inputs (`kind`, `version`, missing fields), **`save/load`** parity, and **untrained** errors.

#### Scenario: CI can run tokenizer package tests

- **WHEN** **`pnpm exec nx run tokenizer:test`** (**or repo-standard equivalent**) executes
- **THEN** new tests accompanying **`BPETokenizer`** SHALL pass
