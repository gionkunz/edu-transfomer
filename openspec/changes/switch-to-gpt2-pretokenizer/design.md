## Context

`BPETokenizer` currently runs a single pre-tokenizer policy, `word-and-punct-v1`, defined as `/\s+|\d+|\p{L}+|[^\s\p{L}\p{N}]+/gu` in `packages/tokenizer/src/bpe-tokenizer.ts`. Whitespace forms its own region class, so a space byte and an adjacent letter byte sit in different regions and the merge loop's `regionBoundary` bitmap blocks them from ever merging. Trained on the project's `corpus/real/*.txt` to a vocabulary of 8 000 ids, the resulting tokenizer encodes English text at roughly 0.7 bytes per token (cf. ≈3.5 bytes/token for GPT-2 BPE on the same content) and burns 27 of its ids on whitespace-only merges (`"\n\n"`, `"    "`, `"        "`, `"\n\n\n"`, …). Empirically the trained model collapses to a unigram-stage distribution after one near-epoch and the loss curve is dominated by the trivially predictable space token between every word.

The codebase already persists a `pretokenizer` field in tokenizer JSON (introduced by `speed-up-bpe-encode`, schema `version: 2`) and validates it on `load`. That gives us a clean mechanism to invalidate older artifacts: bump the version, change the only allowed label, and let `parseBpeTokenizerJson` reject everything written by the current code with a "retrain" message.

There are no production tokenizer artifacts: `out/tok-real-8000.json`, `out/tok-cosmo-10000.json`, and the `out/medium-longctx/tokenizer.json` shipped alongside the latest training run are all reproducible in seconds. Nothing on disk needs to keep loading.

## Goals / Non-Goals

**Goals:**

- Replace the pre-tokenizer policy with a GPT-2-style scheme in which a content region MAY include exactly one leading space, so common English words tokenize as `" the"` / `" and"` / `" of"` rather than `the` / `and` / `of` flanked by separate space tokens.
- Keep the existing region-boundary contract (no merge crosses a region boundary) and the existing two-path implementation (regex path for in-memory training, byte-level path for streaming file training), so the rest of the BPE machinery — incremental pair index, packed-key encode tables, save/load, progress events — does not have to change.
- Fail loudly on any tokenizer artifact written under the old scheme. The user-visible signal is a `load()` error that names the unsupported version and instructs the user to retrain.
- Keep the public `Tokenizer` interface (`train` / `encode` / `decode` / `save` / `load` / `vocabSize`) byte-for-byte unchanged.

**Non-Goals:**

- Supporting both policies side by side. The previous draft of this work proposed a dual-policy implementation; we explicitly decided against it because (a) it doubles the surface area of the byte-level boundary detector — the load-bearing piece — and (b) we have no artifacts to migrate.
- Changing how the BPE merge selection, encode hot path, packed-key tables, or progress events work. None of those care about the regex; they only care that `regionBoundary[i]` says "this byte is the last byte of its region".
- Mirroring tiktoken's exact regex (digit-grouping, contraction set extensions, etc.). The GPT-2 regex is sufficient for an educational small-model project and is cheaper to validate.
- Migrating `corpus/real/*.txt`-trained checkpoints. Old `model.bin` files are tied to old token ids and will simply be retrained.

## Decisions

### 1. Adopt the GPT-2 pre-tokenizer regex (verbatim modulo Unicode classes)

The active regex SHALL be:

```
/'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu
```

The alternatives are tried in this order on each match step (JS regex alternation is ordered). Reading them top-down:

1. The English contraction set. These are matched first so `Howard's` regions as `["Howard", "'s"]` rather than `["Howard", "'", "s"]`. Without this branch the `'s` would be one region (good) but `'re` etc. would split letter-from-quote, which BPE would have to relearn from scratch.
2. ` ?\p{L}+` — letters, optionally preceded by a single space. This is the load-bearing branch for English: it produces `" the"`, `" and"`, `" of"` as single regions whose first byte is `0x20`.
3. ` ?\p{N}+` — digits, same shape.
4. ` ?[^\s\p{L}\p{N}]+` — punctuation / symbols, same shape. This handles `, world` style tokens.
5. `\s+(?!\S)` — runs of whitespace at end-of-line / end-of-input. Without this branch, indentation at the start of a new paragraph would always get glued to the next word, which is fine for prose but unhelpful when the model has to predict line boundaries.
6. `\s+` — fallback for any remaining whitespace runs (e.g. multiple consecutive blank lines that are simultaneously end-of-line).

We do not prepend `'(?:s|t|re|ve|m|ll|d)` as a single alternative because JS regex alternation already short-circuits in declaration order and a flat list reads more clearly.

**Alternatives considered:**

- _Llama-3 / cl100k regex (digit grouping, multi-codepoint emoji handling)._ Strictly nicer for code corpora but adds branches we don't yet exercise; revisit when we train on something other than English Project Gutenberg.
- _SentencePiece's `▁` substitution._ Replaces ASCII spaces with `U+2581` before BPE. Equivalent in effect to the leading-space scheme but introduces a mandatory text-rewrite step on encode/decode and changes the byte-level grounding of the BPE base alphabet. Worse fit for a byte-level BPE that prides itself on 1:1 byte ↔ id at the floor.

### 2. The byte-level boundary detector becomes a small state machine

Today's `fillBoundaryFromUtf8Bytes` is a 1-byte-of-history class-transition detector: emit a boundary whenever the current code point's class differs from the previous. The new rule is no longer expressible that way, because " the" is one region but `" "` followed by `"x"` (where `x` follows a non-letter) might be two. The detector becomes:

```
state ::= IDLE | LETTER_RUN | DIGIT_RUN | OTHER_RUN | TRAILING_WS
```

with one byte of leading-space lookahead. Concretely, when we see a space at IDLE we don't emit a boundary yet; we peek at the next code point and:

- If the next code point starts a letter / digit / non-letter-symbol run, the space is absorbed into that run (no boundary at the space).
- If the next code point is also whitespace, we are inside a `TRAILING_WS` region and emit a boundary at the byte before the next non-whitespace code point.
- If we are at end-of-input the trailing whitespace is its own region.

Contractions (`'s` / `'t` / `'re` / `'ve` / `'m` / `'ll` / `'d`) are handled by the apostrophe being classified as "other", with a special look-ahead: if the previous region was a letter run AND the apostrophe is followed by one of the contraction tails, we end the letter run before the apostrophe and start a new region whose contents are exactly the apostrophe + tail.

A reference implementation of this state machine is exactly the regex above run through `String.fromCodePoint`-decoded bytes; we keep that as the test oracle (see Risk #1).

**Alternative considered:** drive the boundary computation purely from the regex even on the streaming path, by buffering full files into JS strings before passing them to `String.matchAll`. Rejected because V8 string size is capped at ~512 MB and `trainFromFiles` is the path we'd actually want to use for a 5 GB corpus later; it must stay byte-streamable.

### 3. Bump JSON `version` to `3` and reject everything older

`save()` always writes `{ version: 3, pretokenizer: 'gpt2-v1', ... }`. `load()` accepts only `version === 3` AND `pretokenizer === 'gpt2-v1'`. Any older artifact (including current `version: 2` `word-and-punct-v1` files) raises a descriptive error that names the unsupported version, the unsupported policy label if applicable, and tells the user to retrain.

We do not attempt to migrate old artifacts. Re-running `train-tokenizer` on `corpus/real/*.txt` takes seconds.

**Alternative considered:** keep `version: 2` and only switch the label to `'gpt2-v1'`. Rejected because the version field is the cleanest single check; mixing version-stable-but-label-incompatible files makes the load error harder to write and easier to silently miss in code review.

### 4. The `PretokenizerPolicy` type narrows back to a single literal

After this change the type is `'gpt2-v1'` (one literal). All `string`-keyed lookup tables, label sets, and switch statements collapse. If we ever add another policy we'll widen the union and add a new label, the same way `speed-up-bpe-encode` originally introduced `word-and-punct-v1`. The current change deliberately does not pre-engineer a multi-policy abstraction.

### 5. Keep all existing tests for byte-level BPE, retrain expectations only where regions changed

Tests that operate at the byte-BPE-mechanics layer (incremental pair index, packed-key tables, save/load round-trip, encode/decode round-trip on Unicode probes, untrained-misuse rejection, vocab-floor errors) keep passing as-is — they don't depend on the region partition. Tests that are written against `word-and-punct-v1`'s exact region partition (e.g. the existing `bpe-pretokenize.spec.ts` "split `Hello, world! 42 → done`" case) get rewritten to the new partition. The `speed-up-bpe-encode` deltas still describe most of the right behavior; we replace the regex string and the partition example.

## Risks / Trade-offs

- **Risk: regex-path and byte-path diverge silently on edge cases (multibyte UTF-8 letters, lone surrogates from ill-formed UTF-8, EOF-trailing whitespace, contraction lookahead).** The byte-level state machine is hand-written and the JS regex relies on V8's Unicode property tables. → Mitigation: ship a property-style equivalence test (`bpe-pretokenize.spec.ts`) that fuzzes `~10⁴` random Unicode strings (including emoji ZWJ sequences, mixed CRLF, indentation runs, and contraction-shaped inputs), runs both paths, and asserts byte-for-byte identical `regionBoundary` output. This test is the single most important piece of coverage in the change.

- **Risk: loss numbers across the boundary look like a regression.** Per-token loss WILL go up because tokens carry more information. → Mitigation: document this in the proposal (already done) and add a note to the train CLI's dashboard or to `docs/transformer-explained.md` explaining bits-per-character as the comparable metric. Don't gate CI on absolute loss values.

- **Risk: a user has a tokenizer file we forgot about.** → Mitigation: `load()`'s error message is explicit ("retrain the tokenizer"), and the only known artifacts (`out/*.json`, `out/medium-longctx/tokenizer.json`) are already accounted for in the proposal.

- **Trade-off: parameter sharing for `the` / `Ġthe`.** The old policy had one id for `the` regardless of position; the new one will typically have two (`the` at line start, ` the` mid-sentence). For an 8 000-vocab tokenizer this costs a handful of merges. The compression and effective-context wins outweigh it for English by an order of magnitude.

- **Trade-off: contraction list is English-specific.** Branch 1 of the regex hard-codes `'s|'t|'re|'ve|'m|'ll|'d`. On non-English corpora these branches still match but are harmless (they just fire less often). We accept this rather than hand-rolling a multilingual variant; revisit if/when we start training on non-English data.
