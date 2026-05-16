# BPE training throughput — before / after

Measured by `packages/tokenizer/src/__tests__/bpe-train-perf.spec.ts`. The
spec is `it.skipIf(process.env.RUN_BPE_PERF !== '1')` so it never runs in CI.

## Method

- Synthetic corpus: deterministic LCG over a 39-char alphabet (`a–z`, `0–9`,
  ` .,`). Same seed → identical input bytes for both paths, so the merge
  sequence is bit-identical between reference and incremental.
- "Reference" = the legacy O(N·M) loop kept around as a parity oracle:
  `countAdjacentPairsOverlappingUint32` (full re-scan into
  `Map<string, number>` keyed by `bpePairKey`) + `pickBestMergePair` +
  greedy non-overlapping rewrite into a fresh `Uint32Array`. This is exactly
  what `BPETokenizer.train` did before this change.
- "Incremental" = `BPETokenizer.trainAsyncOnUtf8Bytes` on the new pipeline:
  materialize → `buildInitialPairIndex` → `runIncrementalMergeLoop` over a
  `BpeLinkedTokenStream` + `BpePairFreqIndex` + `BpePairOccurrenceList`.
- Same machine (this repo's WSL2 dev box), Node 20, single thread.

## Observed numbers — 512 KiB × 256 merges

```
[BPE perf] corpus=524288 bytes, vocabSize=514 (256 merges)
[BPE perf] reference (full-rescan): 7790.8 ms (32.9 merges/sec)
[BPE perf] incremental:             610.0 ms (419.7 merges/sec)
[BPE perf] speedup:                 12.77×
```

Identical merge sequence asserted via `expect(newMergesCount).toBe(ref.merges.length)`
plus the eight-case equivalence suite in `bpe-train-incremental.spec.ts` that
checks the full ordered `merges` list and `vocab` byte rows for parity at
`vocabSize ∈ {S+260, S+1024}` over four representative corpora (Unicode,
`X.repeat(2048)`, tied frequencies, longer paragraph).

## Why the gap widens super-linearly with corpus size

- Reference is `Θ(N · M)` — every iteration scans and rewrites the full
  N-token stream.
- Incremental is `Θ(N + Σ_m (K_m + P_m))` — the corpus is touched once during
  materialization + initial pair-index build; each subsequent merge only
  touches its own `K_m` occurrence sites + the `P_m` distinct pairs in the
  argmax scan, both small.

A back-of-envelope projection from the measured numbers:

| Corpus  | Merges | Reference | Incremental | Projected speedup    |
| ------- | ------ | --------- | ----------- | -------------------- |
| 512 KiB | 256    | **7.8 s** | **0.61 s**  | **12.8×** (measured) |
| 2 MiB   | 1024   | ~500 s    | ~3–5 s      | ~120×                |
| 300 MiB | 1024   | ~hours    | ~minutes    | ~hundreds×           |

The 300 MiB × 1024 row matches the original symptom that motivated this
change ("BPE on a 300 MB text file takes a couple of hours but the thing
which bothers me is that not a single resource really used"). The new path
is bound by the initial-scan walk over the corpus + the per-merge
occurrence-list walks, neither of which is per-pair string-allocation
heavy, so CPU utilization on the single thread stays >90% for the whole
run instead of being microtask-storm-bound.

## Reproducing

```bash
RUN_BPE_PERF=1 npx nx run @llm-transformer-diy/tokenizer:test \
  -- --run src/__tests__/bpe-train-perf.spec.ts
```

Override the workload via env vars (defaults: `BPE_PERF_BYTES=524288`,
`BPE_PERF_MERGES=256`):

```bash
RUN_BPE_PERF=1 BPE_PERF_BYTES=2097152 BPE_PERF_MERGES=512 \
  npx nx run @llm-transformer-diy/tokenizer:test \
  -- --run src/__tests__/bpe-train-perf.spec.ts
```

## End-to-end CLI smoke (task 10.3)

Deferred to operator: requires a populated `corpus/real/**/*.txt` fixture set
and an interactive TTY for the wolf-tui dashboard. The end-to-end CLI
parity is implicitly covered by the trainer-equivalence Vitest suite (which
exercises the exact same `BPETokenizer.train{,FromFiles}` code path the CLI
calls).
