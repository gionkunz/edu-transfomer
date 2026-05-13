## Why

The training corpus **BPE encode** phase (**`BPETokenizer.encodeAsync`** called from **`apps/cli`** `train`) is the user-visible long pole on first-time training runs (low CPU, zero GPU, "ages" wall time on a multi-MB corpus). The `docs/research/bpe-encode-throughput.md` survey identifies a small set of **internal hot-loop wins** that ship roughly **2× wall-time reduction with zero change to encoded token output** and no new abstractions, dependencies, or threads — the highest ROI per unit of complexity available before we move on to pre-tokenization, algorithmic restructure (priority-queue + linked list), or worker-pool parallelism.

## What Changes

- **Merge-all-non-overlapping-occurrences per pass** in **`BPETokenizer`** internals: replace **`performSingleGreedyMergeUint32`** (which currently selects the lowest-rank pair, then merges only its **leftmost** occurrence and returns) with a single in-place left-to-right pass that selects the lowest-rank pair **and** collapses every non-overlapping occurrence of that pair in one walk. The encoded token sequence remains **bit-identical** to today's behavior; only the **number of outer iterations** drops (often 10×–100× on natural text).
- **Relax the `setTimeout(0)` yield** in **`encodeAsync`**: bound yields by **time slice OR a coarse outer-iteration counter** (e.g. every N "best-rank passes"), not every progress emit, and prefer **`setImmediate`** on Node. **`encodeAsync`** still completes within a wall-time slice budget for TUI repaint and **Ctrl+C**, but the inner loop pays far fewer event-loop trips.
- **Avoid the boxed-Number round-trip** at the encode boundary: have **`encode`** / **`encodeAsync`** return (or expose) a **`Uint32Array`** of token ids alongside (or in place of) the current **`number[]`** so callers like **`apps/cli/src/cmd/train.tsx`** that immediately re-pack into a `Uint32Array` can consume the typed buffer directly. Public `number[]`-returning surface stays available (compatibility) — the typed surface is additive.

Encoded outputs are **identical to current behavior** for all inputs (`encode(x)` and `encodeAsync(x)` produce the same token id sequence as today). This is an **internal performance** change, not a tokenizer/model migration.

## Capabilities

### New Capabilities

- _(none)_

### Modified Capabilities

- **`tokenizer-bpe-bootstrap`**: Generalize the encode requirement to permit "merge all non-overlapping left-to-right occurrences of the chosen lowest-rank pair in one pass" (mathematically equivalent to repeated leftmost-only merges for greedy-by-rank BPE) and add a typed-buffer return surface plus a **`encodeAsync`** yield-cadence requirement that decouples yields from the per-pair inner walk.

## Impact

- **`packages/tokenizer`**: `bpe-tokenizer.ts` (rewrite of `performSingleGreedyMergeUint32`, simplified inner loop in `encode` / `encodeAsync`, optional `encodeUint32` / `encodeUint32Async` typed-buffer entry points or a typed return); existing Vitest tests must continue to pass unchanged.
- **`apps/cli`**: `src/cmd/train.tsx` (consume the typed buffer where it currently calls `Array.from(...)` then re-packs into `Uint32Array`); `train-dash-types.ts` and `train-dashboard.tsx` are unchanged in semantics — `mergeIteration` now counts "best-rank passes" rather than "single-position merges", which is **what the existing pessimistic ETA math already assumes** (one slot removed per iteration is a worst-case bound; merging multiple slots per pass simply finishes faster than the bound, matching the existing **"may finish sooner"** TUI copy).
- **No new dependencies**, no `worker_threads`, no GPU work, no tokenizer JSON format change.
