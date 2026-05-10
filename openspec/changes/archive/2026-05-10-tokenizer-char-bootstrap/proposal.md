## Why

`packages/tokenizer` is a marker-only stub while `docs/project.md` and `docs/sequential-change-proposals.md` step 8 require a real **`Tokenizer` surface** for pipeline testing (encode/decode, vocab size, save/load) without forbidden dependencies (`tiktoken`, `sentencepiece`). This change lands a **strict char-level** implementation matching the documented interface and JSON persistence suitable for the tiny / char-vocab model path.

## What Changes

- Implement **`CharTokenizer`** in `packages/tokenizer`: satisfies `Tokenizer` from `docs/project.md` with `train`, `encode`, `decode`, `save`, `load`, and `vocabSize`.
- **Vocabulary**: fixed-order **special tokens** (defaults `<|pad|>`, `<|endoftext|>`) assigned ids `0 .. k-1`, then every unique **Unicode code point** observed in training text, sorted by ascending code point for a **stable** ordering across runs and machines.
- **`train(text, vocabSize)`**: builds the vocabulary; throws if `vocabSize` is **less than** the required size (`specialTokens.length` + unique code points in `text`). Values **greater than** required are allowed (embedding table may be larger than the tokenizer’s active vocabulary); the tokenizer’s `vocabSize` property reflects **actual** token count after training.
- **Persistence**: JSON on disk with a versioned `kind` discriminator and explicit `specialTokens` + `chars` arrays (no external tokenizer libs).
- **Vitest** coverage: encode/decode round-trip, sorted vocab stability, `train` capacity errors, save/load round-trip, untrained errors.
- **Explicitly out of scope**: BPE / merge training, tiktoken-compatible encodings, SentencePiece, longest-match encoding of multi-character special strings (encode is **per code point**; special token ids exist for fixed indices such as padding).

## Capabilities

### New Capabilities

- `tokenizer-char-bootstrap`: Requirements for a char-level `Tokenizer` implementation with stable vocab ordering, JSON save/load, and Vitest coverage.

### Modified Capabilities

- _(none)_

## Impact

- **Primary**: `packages/tokenizer` — replace stub with `CharTokenizer`, shared `Tokenizer` type, and tests.
- **Dependencies**: none added; uses Node built-in `node:fs/promises` for save/load (CLI / training targets run on Node per project stack).
- **Downstream**: `apps/cli` and `packages/training` keep importing `TOKENIZER_LAYER_MARKER`; no breaking change to that symbol.
- **Docs**: `docs/project.md` remains authoritative; OO `Tokenizer` interface preserved.
