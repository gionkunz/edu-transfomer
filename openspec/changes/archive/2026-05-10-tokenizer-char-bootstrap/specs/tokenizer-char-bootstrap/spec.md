# tokenizer-char-bootstrap

## Purpose

Define requirements for the first real **`packages/tokenizer`** slice: a char-level tokenizer implementing the `Tokenizer` interface from `docs/project.md`, with deterministic vocabulary construction, JSON persistence, and Vitest coverage. BPE and byte-pair training are explicitly out of scope.

## Requirements

### Requirement: Tokenizer interface is implemented for char-level encoding

The package SHALL export a `Tokenizer` interface matching `docs/project.md` (`train`, `encode`, `decode`, `save`, `load`, `vocabSize`). The package SHALL export `CharTokenizer` as a concrete implementation.

#### Scenario: Methods exist with documented semantics

- **WHEN** a consumer inspects exports of `@llm-transformer-diy/tokenizer`
- **THEN** `Tokenizer` (type), `CharTokenizer` (class), and `TOKENIZER_LAYER_MARKER` are available

---

### Requirement: Vocabulary combines special tokens then sorted unique code points

After `train(text, vocabSize)` succeeds, ids `0 .. specialTokens.length - 1` SHALL map to the configured special token strings in order. Remaining ids SHALL map to each distinct Unicode code point observed in `text` via string iteration, sorted by ascending numeric code point.

#### Scenario: Stable sort across platforms

- **WHEN** `train` is called twice on the same `text` with the same `CharTokenizer` configuration and sufficient `vocabSize`
- **THEN** the resulting token-id assignment for each code point is identical

---

### Requirement: train enforces a minimum vocabSize

`train` SHALL throw if `vocabSize` is strictly less than `specialTokens.length` plus the number of unique code points in `text`. If `vocabSize` is greater than that count, training SHALL succeed; `vocabSize` getter SHALL still equal the actual token count (`specialTokens.length + unique chars`).

#### Scenario: train rejects undersized cap

- **WHEN** `train` is called with `vocabSize` smaller than the required minimum
- **THEN** the call throws an `Error` before mutating the trained vocabulary

---

### Requirement: encode and decode are inverse for in-vocab text

`encode` SHALL map each code point in the input string to its id. `decode` SHALL map each id to the corresponding token string and concatenate without separators. `decode(encode(s)) === s` SHALL hold for any `s` whose code points are all present in the post-train vocabulary.

#### Scenario: Round-trip on mixed ASCII and newline

- **WHEN** a tokenizer is trained on a corpus containing `a`, `b`, and newline, and `encode` / `decode` are called on `"a ab b a\n"`
- **THEN** `decode(encode(input))` equals `input`

---

### Requirement: save and load round-trip vocabulary

`save(path)` SHALL write JSON describing `kind`, `version`, `specialTokens`, and `chars`. `load(path)` SHALL restore an equivalent tokenizer state such that `encode` / `decode` / `vocabSize` match the pre-save behavior for all inputs.

#### Scenario: Persistence preserves vocabSize

- **WHEN** a trained tokenizer saves to a file and a fresh `CharTokenizer` loads that file
- **THEN** both tokenizers report the same `vocabSize` and identical `encode` results for the same string

---

### Requirement: No forbidden tokenizer dependencies

The tokenizer package SHALL NOT add runtime dependencies on `tiktoken`, `sentencepiece`, HuggingFace `transformers`, or other forbidden per `docs/project.md`.

#### Scenario: package manifest stays minimal

- **WHEN** `packages/tokenizer/package.json` is inspected
- **THEN** no forbidden tokenizer packages appear in `dependencies`
