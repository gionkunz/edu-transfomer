## Why

**Encoding a real-world corpus is currently hours-to-days of wall time** because **`BPETokenizer.encodeUint32Async`** treats the input as one monolithic token stream and probes a two-level **`Map<number, Map<number, number>>`** for every adjacent pair, every outer pass. On a 316 MB Cosmopedia shard with a `10 000`-merge tokenizer the CLI **`train`** dashboard observed `0.2 passes/sec`, projecting **multi-hour** encode time before any GPU work begins. The current implementation is correct and predictably faster than the pre-`fc4823f` single-merge variant, but it is still **O(N · P)** with `N = 316 M` and **P** = hundreds of outer best-rank passes — too slow once the corpus exceeds a few MB.

Real-world BPE encoders (tiktoken, HF tokenizers) sidestep the wall by **pre-tokenizing** input on whitespace and word boundaries before learning or applying merges, so each merge pass walks short independent regions instead of the entire corpus. A pre-tokenized region of ≤ 32 ASCII characters reaches quiescence in **≤ 5 outer passes** regardless of corpus length, dropping the per-corpus encode cost from `O(N · P_corpus)` to `O(N · P_region)`. Combined with a single-`Map` packed-pair-key, this collapses the dashboard's "**~16 511 days**" worst-case ETA to **seconds** on the same input, with zero new dependencies.

## What Changes

- **Add a region pre-tokenizer** — a small, documented regex (`\s+|\d+|\p{L}+|[^\s\p{L}\p{N}]+`) that splits a UTF-8 input into adjacent non-overlapping **regions**. The regex is part of the spec and persisted in the tokenizer JSON so future tokenizers stay locked to the rule they were trained under. **`BPETokenizer.encodeUint32` / `encodeUint32Async`** SHALL encode each region in isolation, then concatenate the resulting id arrays in input order. Merges SHALL NOT cross region boundaries.
- **Train under the same pre-tokenizer** — **`train` / `trainAsync` / `trainAsyncOnUtf8Bytes` / `trainFromFiles`** SHALL pre-split the working corpus on the same regex before initializing pair counts. Adjacent pairs that span a region boundary SHALL NOT be observed during training and SHALL NOT learn merge rules. The incremental pair-frequency index from `tokenizer-bpe-bootstrap` is preserved per-region.
- **Pack the encode-hot-path pair lookup** — replace the two `Map<number, Map<number, number>>` tables (`mergeRankByPair`, `mergeOutputIdByPair`) with single-level `Map<number, number>` tables keyed by `L * vocabSize + R`. Constant-factor 2-3× win on the `O(N)` per-pass scan even after pre-tokenization.
- **Persist the pre-tokenizer marker in the tokenizer JSON** — schema version `2` is the only supported version; the `pretokenizer` field is required and must be a known label (currently only `'word-and-punct-v1'`). **`load`** SHALL refuse any other `version` value, any unknown `pretokenizer` label, and any file missing the `pretokenizer` field.
- **Update the train TUI ETA copy** — pass count is measured **per region**. The dashboard SHALL show **total regions encoded** and **mean passes/region** alongside the existing aggregate counters and SHALL compute ETA from `regions_remaining × mean_passes_per_region / pass_rate`.
- **Document the migration** — `README.md` and `docs/project.md` SHALL note that tokenizers trained before this change must be retrained; the legacy whole-corpus encode path is removed.

**BREAKING**: This is a hard break in two ways:

1. Encode outputs differ from any pre-change tokenizer (region-aware merges only).
2. Pre-change tokenizer JSON files (schema `version: 1`, no `pretokenizer` field) no longer load. **Retrain.**

## Capabilities

### New Capabilities

- _(none)_

### Modified Capabilities

- **`tokenizer-bpe-bootstrap`**: Generalize `train` and `encode` to operate on **pre-tokenized regions** under a named regex policy, with merges constrained to within-region adjacent pairs. Add the **packed pair-key** invariant for the encode hot path. Adopt JSON schema version `2` with a required `pretokenizer` field; the prior schema version `1` is rejected on load.
- **`cli-wolf-tui-commands`**: Replace the whole-corpus encode ETA on the **`train`** dashboard with **per-region** progress (regions encoded, mean passes per region) plus a per-region ETA derived from `regions_remaining × mean_passes_per_region × wall_time_per_pass`.

## Impact

- **`packages/tokenizer`**: `bpe-tokenizer.ts` adds `splitIntoRegions(text, policy)`, refactors `encodeUint32{,Async}` to fold regions, refactors `train{,Async,AsyncOnUtf8Bytes,FromFiles}` to fold regions before the incremental pair index, replaces the nested-Map encode tables with packed-key `Map<number, number>`, bumps the tokenizer JSON schema to `2`, and removes the prior `null` / `version: 1` paths. New tests for region splitting, region-boundary merge exclusion, packed-key encode equivalence, and schema `v2`/`v1` rejection.
- **`apps/cli`**: `cmd/train.tsx`, `cmd/train-dash-types.ts`, and `cmd/encode-eta.ts` switch the encode dashboard / ETA to a per-region model. CLI surface unchanged; no new flags.
- **`docs/`**: `project.md` and `README.md` note the pre-tokenizer policy and the schema-`v2`-only load. No new public dependencies.
- **No GPU / autograd / training / inference changes.** Model checkpoints are untouched; only the tokenizer JSON format changes.
- **Migration cost**: any existing `tok-*.json` file must be retrained. Model checkpoints stay valid but pair with newly-trained tokenizers.
