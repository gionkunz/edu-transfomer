## 1. Types and public surface

- [x] 1.1 Add `packages/tokenizer/src/types.ts` exporting the `Tokenizer` interface aligned with `docs/project.md`.
- [x] 1.2 Add `packages/tokenizer/src/char-tokenizer.ts` implementing `CharTokenizer` with constructor options for `specialTokens`, validation (no duplicates / empty strings), `train` / `encode` / `decode` / `save` / `load`, and a `vocabSize` getter.

## 2. Persistence

- [x] 2.1 Define JSON v1 schema with `kind: "llm-transformer-diy/char-tokenizer"` and `version: 1`.
- [x] 2.2 Implement `save` / `load` using `node:fs/promises`; loader validates shape and throws on mismatch.

## 3. Tests

- [x] 3.1 Add `packages/tokenizer/src/char-tokenizer.spec.ts`: encode/decode round-trip, vocab ordering, `train` throws when `vocabSize` too small, untrained errors, save/load round-trip via temp file.

## 4. Package exports and verification

- [x] 4.1 Update `packages/tokenizer/src/index.ts` to export `Tokenizer`, `CharTokenizer`, and keep `TOKENIZER_LAYER_MARKER`.
- [x] 4.2 Run `pnpm exec nx run @llm-transformer-diy/tokenizer:test` and `pnpm exec nx run @llm-transformer-diy/tokenizer:build`.
