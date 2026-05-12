## Why

`docs/project.md` and `docs/sequential-change-proposals.md` step 8 call for **byte-pair encoding (BPE)** after the shipped **char-level** tokenizer. BPE fixes the main limitation of char vocabularies—very long sequences and weak compression—while staying within the repo rule of **no** `tiktoken`, `sentencepiece`, or other tokenizer libraries.

## What Changes

- Implement **`BPETokenizer`** in `packages/tokenizer` satisfying the shared **`Tokenizer`** interface (`train`, `encode`, `decode`, `save`, `load`, `vocabSize`).
- **Training**: classic byte-level BPE—base vocabulary of **256 raw bytes**, configurable **special tokens** with stable low ids (mirror `CharTokenizer` conventions where practical), iterative merge of the most frequent adjacent pairs until `vocabSize` is reached or no merges remain.
- **Persistence**: **JSON** on disk compatible with `docs/project.md`—ordered **`merges`**, **`vocab`** mapping ids → byte sequences, **`specialTokens`** metadata; include a **`kind`** discriminator + **`version`** for migration (same spirit as `CharTokenizer` JSON).
- **`encode` / `decode`**: deterministic round-trip for valid UTF-8 text; document behavior for invalid surrogate pairs or other edge cases consistent with byte-level tokenization.
- **`save` / `load`**: round-trip equality for vocab, merges, and trained state; reject unknown `kind` / incompatible versions with clear errors.
- **Vitest**: training convergence smoke (vocab grows as expected), encode/decode round-trip on representative strings, merge ordering stability for fixed input, save/load round-trip, untrained / invalid-argument errors.
- **Explicitly out of scope**: pretrained OpenAI/GPT-2 **vocab parity**, regex pretokenization splits, **tiktoken**/`cl100k_base` compatibility, SentencePiece/Unigram, multithreaded training, or integrating BPE into CLI (separate change).

## Capabilities

### New Capabilities

- `tokenizer-bpe-bootstrap`: Requirements for a byte-level BPE tokenizer (train/encode/decode/save/load), JSON persistence format, and Vitest coverage—all without forbidden dependencies.

### Modified Capabilities

- _(none)_ — existing OpenSpec specs do not define tokenizer behavior; char tokenizer remains the reference implementation for special-token defaults.

## Impact

- **Code**: `packages/tokenizer` (`bpe-tokenizer.ts`, tests, `index` exports); may add small shared helpers if duplication with `char-tokenizer` is worth factoring.
- **API**: New exported type `BPETokenizer` (+ options); `Tokenizer` interface unchanged.
- **Dependencies**: none beyond **Node** `fs/promises` and existing test tooling (Vitest / Nx)—no new npm packages.
