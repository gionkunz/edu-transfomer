## Context

`docs/project.md` defines `Tokenizer` with `train`, `encode`, `decode`, `save`, `load`, and `vocabSize`. The char-level section describes vocabulary as unique training characters plus special tokens, with encode as per-character mapping and decode as id → string join. Sequential step 8 calls for char-level first; BPE is a later iteration.

## Goals / Non-Goals

**Goals:**

- A **`CharTokenizer` class** implementing **`Tokenizer`** with deterministic vocabulary construction and explicit error messages.
- **Stable ordering**: special tokens in constructor order receive ids `0..k-1`; remaining tokens are unique code points from `for..of` iteration over `text`, sorted by numeric code point (tie-break `localeCompare`).
- **`train` capacity check**: required size is `specialTokens.length + uniqueCodePoints(text)`; throw if `vocabSize` is smaller; allow larger `vocabSize` argument (tokenizer still reports actual `vocabSize`).
- **JSON v1** format: `{ "kind": "llm-transformer-diy/char-tokenizer", "version": 1, "specialTokens": [...], "chars": [...] }` where `chars` lists each training code point string once, sorted.
- **Round-trip**: for any trained tokenizer and string using only code points present in the trained vocab, `decode(encode(s)) === s`.
- Preserve **`TOKENIZER_LAYER_MARKER`** export for existing CLI / training stubs.

**Non-Goals:**

- BPE training, merge tables, byte-level base vocab.
- Longest-prefix **`encode`** that collapses a literal `<|pad|>` substring to id `0` (encode remains per code point; specials remain usable as dedicated ids for batching / APIs).

## Decisions

| Decision                       | Rationale                                                                                          |
| ------------------------------ | -------------------------------------------------------------------------------------------------- |
| Class implementing `Tokenizer` | Matches the documented OO interface (`train` / `encode` / … as methods).                           |
| Per-code-point `encode`        | Matches `docs/project.md` char description; avoids scope creep into greedy special-token scanning. |
| `node:fs/promises` for I/O     | No new deps; matches Node-first stack.                                                             |
| `kind` + `version` in JSON     | Forward-compatible loader rejects unknown kinds/versions loudly.                                   |

## Risks / Trade-offs

- **Special strings vs. per-char encode**: consumers must not assume `encode('<|pad|>')` yields `[0]`; document in JSDoc.
- **Empty training text**: vocabulary is specials-only; `train('', n)` requires `n >= specials.length`.

## Migration Plan

- None required beyond replacing stub implementations; no other package consumed tokenizer behavior yet.
