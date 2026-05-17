## Why

The current `word-and-punct-v1` pre-tokenizer treats whitespace as its own region class, so spaces can never be merged onto adjacent words. For English prose this roughly doubles the token count of any encoded text, halves the effective context window in characters, and forces the model to spend half its loss budget predicting the trivial "space after word" event. It also wastes vocabulary on whitespace-only merges (the trained `out/medium-longctx/tokenizer.json` burns 27 of its 8 000 ids on pure-whitespace tokens). The scheme used by GPT-2 / Llama / SentencePiece — letting a word optionally absorb one leading space — is the long-standing standard precisely because it avoids these costs.

We have no production tokenizer artifacts that need to keep working, so the cleanest move is to replace the policy outright (single supported policy, single byte-path region detector to maintain) and let `load()` reject older artifacts with a clear "retrain" message.

## What Changes

- **BREAKING**: Replace the `word-and-punct-v1` pre-tokenizer with a new `gpt2-v1` policy. The `word-and-punct-v1` regex, label, and all references in `BPETokenizer` are removed.
- **BREAKING**: Bump tokenizer JSON to `version: 3`. `load()` rejects `version <= 2` (including any `version: 2` file emitted under `word-and-punct-v1`) with a descriptive "retrain the tokenizer" error.
- The new `gpt2-v1` policy splits input into regions using a regex equivalent to GPT-2's: contractions (`'s` `'t` `'re` `'ve` `'m` `'ll` `'d`), an optional single leading space + run of letters, an optional single leading space + run of digits, an optional single leading space + run of non-letter/non-digit/non-whitespace symbols, trailing whitespace at end of line (`\s+(?!\S)`), and finally other whitespace runs.
- Both implementations of the region-boundary detector — the `String.matchAll` regex path and the streaming-bytes byte-level path in `fillBoundaryFromUtf8Bytes` — are rewritten to the new policy. A test asserts they produce byte-for-byte identical boundary bitmaps for a representative corpus including ASCII, multi-byte UTF-8, mixed CRLF, indentation runs, and emoji.
- `splitIntoRegions` and the exported `PretokenizerPolicy` type are narrowed to a single label (`'gpt2-v1'`).
- Existing `out/tok-real-8000.json`, `out/tok-cosmo-10000.json`, and `out/medium-longctx/tokenizer.json` cease to load. The user retrains tokenizer + model.

## Capabilities

### New Capabilities

_(none — this change modifies an existing capability)_

### Modified Capabilities

- `tokenizer-bpe-bootstrap`: pre-tokenizer policy is replaced (label, regex, byte-level boundary rule), JSON schema version bumped to `3`, and `load()` error contract updated to reject `version: 2` files and the `word-and-punct-v1` label.

## Impact

- **Code**: `packages/tokenizer/src/bpe-tokenizer.ts` (regex constant, `PretokenizerPolicy` type, `splitIntoRegions`, `fillBoundaryFromUtf8Bytes`, `JSON_VERSION`, `KNOWN_PRETOKENIZER_LABELS`, `parseBpeTokenizerJson`, `save`/`load`).
- **Tests**: `packages/tokenizer/src/__tests__/bpe-pretokenize.spec.ts` (region partitioning under the new regex), a new test asserting regex-path / byte-path equivalence on representative inputs, `bpe-tokenizer.spec.ts` (load-error messages, save-version literal).
- **CLI / artifacts**: any user-held tokenizer JSON written by the current code becomes invalid and must be retrained. No code changes in `apps/cli` are expected (the train command always writes via `save()` and reads via `load()` without inspecting the policy label).
- **Loss numbers**: not comparable across the boundary. Training loss after the switch will be **higher per token** but **lower per character**; track bits-per-character as the tokenizer-invariant quality signal.
- **Other dependencies**: none. No new runtime packages.
