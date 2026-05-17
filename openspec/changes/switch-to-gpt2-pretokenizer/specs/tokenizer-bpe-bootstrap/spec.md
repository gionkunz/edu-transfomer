## MODIFIED Requirements

### Requirement: Pre-tokenizer policy

**`BPETokenizer`** SHALL expose a **pre-tokenizer policy** that splits a UTF-8 input string into adjacent non-overlapping **regions** before any byte-level expansion or merge step. The policy SHALL be identified by a stable string label persisted with the tokenizer. The only label this revision supports is **`gpt2-v1`**; future revisions MAY add additional labels but SHALL NOT silently change the meaning of an existing label. The policy label SHALL be set at training time and SHALL NOT be reconfigurable on an already-trained tokenizer.

The **`gpt2-v1`** policy SHALL split using the regular expression

```
/'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu
```

applied with `String.prototype.matchAll`, in that alternative order. The full input SHALL be covered by the concatenation of matches. Each match SHALL produce one **region**. A region produced by branches 2–4 (` ?\p{L}+`, ` ?\p{N}+`, ` ?[^\s\p{L}\p{N}]+`) MAY begin with a single ASCII space byte (`0x20`); the space SHALL be part of that region rather than a region of its own. A region produced by branches 5 or 6 (`\s+(?!\S)`, `\s+`) SHALL contain only whitespace bytes.

The byte-level streaming pre-tokenizer used by `trainFromFiles` (which classifies bytes without first decoding the corpus into a JS string, because V8's ~512 MB string cap rules out that path for large corpora) SHALL produce a `regionBoundary` bitmap that is **byte-for-byte identical** to the bitmap produced by running the regex above over the same corpus interpreted as UTF-8. Implementations SHALL provide an automated test that asserts this equivalence on representative inputs.

#### Scenario: gpt2-v1 produces the expected region partition

- **WHEN** the **`gpt2-v1`** pre-tokenizer is applied to the input **`"Hello, world! It's 42 → done"`**
- **THEN** the resulting regions, in order, SHALL be exactly **`["Hello", ",", " world", "!", " It", "'s", " 42", " →", " done"]`** (each content region MAY carry a single leading space; the contraction `'s` is a region of its own; no two adjacent letter regions appear without an intervening boundary)

#### Scenario: gpt2-v1 isolates trailing whitespace at end of line

- **WHEN** the **`gpt2-v1`** pre-tokenizer is applied to the input **`"foo \n  bar"`**
- **THEN** the resulting regions, in order, SHALL be exactly **`["foo", " \n", "  bar"]`** — the run of whitespace before the newline (and the newline itself) is one region produced by the `\s+(?!\S)` branch, while the leading two spaces of the next line are absorbed into ` bar`'s region by the ` ?\p{L}+` branch (only one of them, with the second behaving as the absorbed leading space)

_(Note: the exact distribution of leading spaces between branches 2–4 and branches 5–6 follows from the regex's alternative order; only the boundary positions are normative.)_

#### Scenario: Streaming byte-level boundary detector matches the regex path

- **WHEN** a fuzz harness generates a corpus of mixed ASCII / multi-byte UTF-8 letters / digits / punctuation / mixed CRLF / indentation runs / English contractions, runs the regex-based `splitIntoRegions` to produce a reference `regionBoundary` bitmap, and runs the byte-level `fillBoundaryFromUtf8Bytes` over the same bytes
- **THEN** the two `Uint8Array` bitmaps SHALL be element-wise equal

---

### Requirement: JSON schema version 3 persists the pre-tokenizer policy

**`save`** SHALL write tokenizer JSON with **`version: 3`** and a required top-level field **`pretokenizer`** whose value is the active policy label string (currently the only valid value is **`"gpt2-v1"`**). The remaining schema fields (**`kind`**, **`specialTokens`**, **`byteOffset`**, **`merges`**, **`vocab`**) SHALL be preserved unchanged.

**`load`** SHALL accept exactly **`version: 3`** files whose **`pretokenizer`** value is a known label, and SHALL throw a descriptive error otherwise. In particular:

- any **`version`** value other than **`3`** (including the obsolete **`1`** and **`2`**) SHALL be rejected with a message that names the supported version and instructs the user to retrain;
- a missing **`pretokenizer`** field on a **`version: 3`** file SHALL be rejected;
- a **`pretokenizer`** value that is not in the implementation's known-label set (in particular **`null`**, the legacy **`"word-and-punct-v1"`** label, and any unrecognized string) SHALL be rejected with a message naming the offending value and the supported set.

A round-trip of **`load`** then **`save`** SHALL preserve the in-memory pre-tokenizer policy.

#### Scenario: Save writes version 3 with the active policy

- **WHEN** a tokenizer trained under **`pretokenizer: 'gpt2-v1'`** is saved
- **THEN** the JSON SHALL have **`version === 3`** and **`pretokenizer === 'gpt2-v1'`**

#### Scenario: Legacy v2 file is rejected with a retrain message

- **WHEN** a tokenizer JSON written by the previous implementation (**`version: 2`**, **`pretokenizer: 'word-and-punct-v1'`**) is loaded
- **THEN** **`load`** SHALL throw an error whose message names the unsupported version and the required action (retrain)

#### Scenario: Legacy word-and-punct-v1 label is rejected even at version 3

- **WHEN** a tokenizer JSON with **`version: 3`** and **`pretokenizer: "word-and-punct-v1"`** is loaded
- **THEN** **`load`** SHALL throw a descriptive error naming the unsupported policy label and the supported set

#### Scenario: Unknown pre-tokenizer string is rejected

- **WHEN** a tokenizer JSON with **`version: 3`** and **`pretokenizer: "future-policy-v9"`** is loaded by an implementation that does not recognize that label
- **THEN** **`load`** SHALL throw a descriptive error naming the unknown policy string and the supported set

#### Scenario: Null pre-tokenizer is rejected

- **WHEN** a tokenizer JSON with **`version: 3`** and **`pretokenizer: null`** is loaded
- **THEN** **`load`** SHALL throw the same "unknown pretokenizer" error as for any other unsupported value
