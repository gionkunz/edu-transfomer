## Context

`apps/cli/src/cmd/train.tsx` reads the entire training corpus into memory (`textParts.join('\n')`) and feeds it to `BPETokenizer.encodeUint32Async` as one JS string. The encoder runs a series of **outer best-rank passes**: each pass scans every adjacent pair in the token buffer to find the lowest-rank applicable pair, then walks the buffer again collapsing every non-overlapping occurrence.

On Cosmopedia-scale corpora (`316 M` chars, vocabulary `10 000`) we observe:

- ~316 M `Map<number, Map<number, number>>` double-lookups per outer pass.
- Outer-pass count is bounded above by the vocab size but in practice runs into the **hundreds** because each pass only retires one merge rule across the whole corpus.
- Measured throughput at `0.2 passes/sec` → tens of minutes per **pass**, hours-to-days for a full encode.

The hot path is therefore both **algorithmically wasteful** (whole-corpus scans for inherently local merges) and **constant-factor expensive** (two `Map.get`s per probe).

The `tokenizer-bpe-bootstrap` spec already mandates an incremental pair-frequency table during training (`Requirement: Incremental pair-frequency maintenance during training`) and permits "merge-all non-overlapping occurrences per pass" semantics for encode. It is **silent** on whether train/encode work on the corpus as one stream or on independent regions, and silent on the constant-factor key structure of the encode tables.

Mainstream BPE implementations (tiktoken, HF tokenizers, sentencepiece's BPE mode) all pre-tokenize on a regex before any merging, both during training and at inference. The result is that:

- Each region is short — typically ≤ 32 ASCII chars for English text.
- Each region reaches encode quiescence in O(log region_length) passes.
- Region encodes are independent and compose by concatenation.

This change brings the project's tokenizer in line with that approach, with one specific regex policy and a one-time JSON schema bump.

## Goals / Non-Goals

**Goals:**

- Drop the wall-clock encode cost on a 316 MB corpus from "hours" to **under one minute** on a modern laptop CPU.
- Make encode and train cost **linear in corpus size and independent of vocab size for typical inputs**, by guaranteeing per-region work is O(region_length).
- Keep the educational property of the tokenizer: the algorithm should still be readable, with no new dependencies or workers.
- Lock retrained tokenizers to a single, documented pre-tokenizer policy (`word-and-punct-v1`), persisted in the tokenizer JSON, with a clean reject path for unknown policies and for the obsolete schema.

**Non-Goals:**

- No backward compatibility with the prior `version: 1` JSON / `pretokenizer: null` whole-corpus path. **This change is intentionally breaking**: any tokenizer trained before it must be retrained.
- No priority-queue / linked-list merge engine (O(N log V) per region). The post-region inner loop is fast enough once regions are short; a heap-based merger is a future change.
- No `worker_threads`-based parallelism. Pre-tokenization already buys two orders of magnitude; threads are a separate, riskier change.
- No new public CLI flags. The pre-tokenizer policy is fixed at training time and persisted in the tokenizer JSON.
- No change to model checkpoints, training loops, GPU kernels, or any non-tokenizer surface.

## Decisions

### Decision 1 — Pre-tokenizer policy: a single named regex, `word-and-punct-v1`

The regex is **`/\s+|\d+|\p{L}+|[^\s\p{L}\p{N}]+/gu`** applied with `String.prototype.matchAll`. It splits the input into adjacent regions of one of four kinds, in priority order:

1. one or more whitespace chars (`\s+`),
2. one or more digits (`\d+`),
3. one or more Unicode letters (`\p{L}+`),
4. one or more characters that are none of the above (`[^\s\p{L}\p{N}]+` — punctuation, symbols, etc.).

**Why this regex?**

- It matches the `gpt2`-style "byte-level BPE with light pre-tokenization" intent without copying GPT-2's regex (which is verbose and English-biased) wholesale.
- Whitespace and digits get their own regions so things like `1234` stay grouped.
- It is a single, ~50-character regex that fits in the spec and never changes once a tokenizer ships under that label.

**Alternatives considered:**

- **HF `gpt2` regex (`r"'s|'t|'re|...|\s+(?!\S)|\s+"`).** English-specific, leaks contractions into the policy, ~150 chars. Rejected for being noisy in a teaching repo.
- **Whitespace-only split.** Loses too much information — `"hello,world"` becomes one region, which still suffers the same O(N · P) collapse on pathological inputs.
- **Sentence-level split.** Region too long; doesn't help the actual hot loop.

The policy string is **the only label the JSON persists** — implementations consume the label, not the regex literal, so we can fix bugs in the regex by bumping to `word-and-punct-v2` later without breaking already-trained tokenizers (those would simply fail to load until support is re-added — same shape as the current break).

### Decision 2 — Region boundaries are hard

Adjacent pairs that span two regions SHALL NOT be observed during training and SHALL NOT match merge rules during encode. The id sequence is reset per region: each region is treated as if it appeared alone.

This is the **only** way to make per-region encode independent and parallelizable in principle.

**Consequence:** GPT-2-style "leading space included in the word token" cannot be learned by merging `space-byte` with `letter-bytes` across regions, because they live in different regions. Trained tokenizers will simply have separate whitespace and word vocabularies — slightly less efficient bits/char than GPT-2 but simpler and a wash for educational throughput.

### Decision 3 — Persist the pre-tokenizer label in tokenizer JSON; the only supported schema is `version: 2`

The JSON gains the `pretokenizer` field:

```json
{
  "kind": "llm-transformer-diy/bpe-tokenizer",
  "version": 2,
  "pretokenizer": "word-and-punct-v1",
  "specialTokens": [...],
  "byteOffset": 2,
  "merges": [...],
  "vocab": {...}
}
```

Reader contract:

- `version !== 2`: refuse to load with a clear "retrain the tokenizer" message. The previous `version: 1` schema is rejected.
- `version === 2` and `pretokenizer` missing: refuse to load.
- `version === 2` and `pretokenizer` is any string outside the known set (currently only `'word-and-punct-v1'`, in particular **not** `null`): refuse to load.
- `version === 2` and `pretokenizer` is a known label: load on the corresponding region-folding path.

`BPETokenizer.save` always writes `version: 2` with the active in-memory policy.

**Alternatives considered:**

- **Implicit policy** (always pre-tokenize, no field). Rejected: silently breaks every existing tokenizer JSON, and removes the future-proofing for a `word-and-punct-v2`.
- **Per-tokenizer custom regex literal in JSON.** Rejected: invites a security/audit issue (loading a tokenizer would compile a regex from disk), and it would make spec scenarios non-determinable.
- **Keep `null` as a valid loaded policy.** Rejected by the breaking-change scope: the whole-corpus encode path is gone and we want load-time rejection rather than silent slow encode for old files.

### Decision 4 — Encode tables: single `Map<number, number>` keyed by `L * vocabSize + R`

`mergeRankByPair: Map<number, Map<number, number>>` → `mergeRankByPair: Map<number, number>` keyed by `L * V + R`. Same for `mergeOutputIdByPair`. Both `L` and `R` are `< V` (always — they're trained token ids), so the packed key is always `< V²` which is `≤ 1e8` for `V ≤ 10000` — well below `Number.MAX_SAFE_INTEGER` (`2^53 - 1`).

The packed-key invariant holds up to `V ≤ ~94 906 265` (`sqrt(2^53)`). Any `V` beyond that boundary is rejected at training time with a clear error.

Build the table once after `train` or `load` and never touch it during encode. Memory: 2 × `V` entries × ~32 bytes/Map entry = ~640 KB for `V = 10 000`. Negligible.

**Why not a flat `Int32Array(V * V)` lookup table?** For `V = 10 000` that's `4 × 10^8 = 400 MB`, prohibitive for the educational stack. For `V = 50 000` (a realistic upper bound elsewhere) it's `10 GB`. The packed-key Map is the sweet spot.

### Decision 5 — Training fold-over-regions reuses the existing incremental pair index

The `Requirement: Incremental pair-frequency maintenance during training` invariants in `tokenizer-bpe-bootstrap` already guarantee `O(N + Σ_m (K_m + P_m))` training cost. We pre-split the corpus into regions in a single pass during initial materialization, then run the same incremental algorithm **once on the concatenation of all region streams** with a `regionBoundary: Uint8Array` companion. The single change is that the initial pair-index scan SHALL skip the boundary pair `(region_k[last], region_{k+1}[0])` for every region boundary — i.e. the corpus is logically a sequence of independent token streams that all share the same incremental table, but no pair observation crosses a boundary.

The merge step is unchanged: when a merge fires inside a region, the at-most-four neighboring-pair updates already obey the boundary because the neighbors lie inside the same region. Boundary-spanning pair observations are simply never created in the first place.

**Alternative considered:** train each region separately, then union the per-region merge lists. Rejected: would change the trained-vocabulary order (merge ranks depend on global frequency, not per-region frequency), and the order is what the encode rank-table depends on.

### Decision 6 — TUI per-region progress

The dashboard's "BPE encode" subphase shows `region A / B · pass #N · M tok`. The pessimistic ETA SHALL be `regions_remaining × mean_passes_per_region / pass_rate`, computed from the running average passes/region and a wall-clock pass-rate EMA.

`BpeEncodeProgressEvent` carries required `regionIndex` and `regionsTotal` fields for every merge-phase emission so the CLI can render the new view without conditionals.

## Risks / Trade-offs

- **Risk**: Existing `out/<run>/tokenizer.json` users who simply update the repo and re-run `llm generate` get a hard `load` error. → **Mitigation**: This is the intended behavior — the user retrains. The error message names the schema version and tells the user to retrain.
- **Risk**: Bits/char efficiency drops slightly compared to GPT-2-style pretokenizers that bind leading whitespace into word tokens. → **Mitigation**: This is a teaching repo, not a production model. The simplicity of `word-and-punct-v1` is the point.
- **Risk**: Unicode regex `\p{L}` / `\p{N}` performance on JS engines varies. → **Mitigation**: We invoke `matchAll` once per encode-call on the full input string, not per byte. Node 22 V8 handles `\p{...}` at ~1 GB/s; this is not the bottleneck.
- **Trade-off**: We do not pursue worker-thread parallelism in this change. Single-thread pre-tokenize + packed-key already wins the user-visible orders of magnitude.

## Migration Plan

1. Ship the v2 schema reader and writer. `BPETokenizer.load` rejects any file that is not `version: 2` with a known `pretokenizer` label.
2. New tokenizers trained with `llm tokenizer train` write `version: 2` with `pretokenizer: 'word-and-punct-v1'`.
3. Document in `README.md` and `docs/project.md` that older tokenizer JSON files must be retrained.
4. No automated migration: retraining is the migration path.

## Open Questions

- _(none significant for v1)_. Future work: consider exposing a `tokenizer.encodeRegions(text): { region: string; ids: Uint32Array }[]` helper for downstream visualization tooling. Out of scope here.
