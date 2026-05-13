## Context

- **`BPETokenizer.encodeAsync`** in **`packages/tokenizer/src/bpe-tokenizer.ts`** drives the **`encode_corpus`** phase of **`apps/cli`** `train`. The hot loop calls **`performSingleGreedyMergeUint32`** repeatedly. Each call: (a) **`O(N)`** scan to find the **lowest-rank applicable pair**, (b) **`O(N)`** scan to find the **first occurrence** of that pair, (c) **`copyWithin`** to delete one slot. End-to-end this is **`O(N · M)`** with one merged slot per outer iteration.
- The corresponding spec **`tokenizer-bpe-bootstrap`** § _Deterministic encoding with merge table_ currently prescribes "Replace the **leftmost** occurrence of the selected pair with its merged id; repeat until no merge applies." For the lowest-rank greedy selection used by BPE, replacing **all non-overlapping left-to-right occurrences** of the chosen pair in one pass produces the **same final token sequence** (proof sketch in Decisions below). This is the algorithm minBPE / Karpathy / SentencePiece-CPU use as the canonical inner loop.
- **`encodeAsync`** wraps the same loop and inserts an `await yieldEventLoopForHostUi()` either every **`EMIT_MERGE_EVERY_ASYNC = 512`** merges or every ~**`TIME_SLICE_ENCODE_ASYNC_MS`** of wall time, whichever comes first. With single-position merges, "merge iteration" is fine-grained (millions for a 3 MB corpus); with merge-all, an "iteration" represents many slot removals so the same time-slice budget produces fewer trips.
- **`encode` / `encodeAsync`** today end with **`Array.from(buf.subarray(0, len))`** to convert the typed working buffer back into a **`number[]`** for the **`Tokenizer`** contract. **`apps/cli/src/cmd/train.tsx`** then immediately repacks this back into a **`Uint32Array`**. Two unnecessary allocations and millions of boxed-Number objects on the GC.
- The existing TUI in **`train-dashboard.tsx`** and ETA helpers in `train-dash-types.ts` use the **pessimistic upper bound** `initialTokenCount - 1` for the `mergeIteration` denominator. With merge-all-per-pass, the same bound is still valid (each pass still removes ≥1 slot in the worst case); the run simply finishes well below it, which the existing **"worst-case bar — may finish sooner"** copy already communicates.

## Goals / Non-Goals

**Goals:**

- Reduce **first-time** corpus encode wall time by roughly **2×** on representative corpora (~1–10 MB) with **no change** to the encoded token id sequence for any input.
- Keep all changes **inside `packages/tokenizer`** plus a small typed-buffer adoption in `apps/cli/src/cmd/train.tsx`.
- Preserve **`Tokenizer`** interface compatibility (`encode(text): number[]`, `encodeAsync(text): Promise<number[]>` keep working).
- Keep **`encodeAsync`**'s wall-time-bounded yield contract so the **wolf-tui** continues to repaint and **Ctrl+C** stays responsive.

**Non-Goals:**

- **Pre-tokenization** (regex / whitespace+punctuation). Defer to a follow-up change — that is a tokenizer **format** change.
- **Priority-queue + linked-list** algorithmic restructure (Stage C in the research doc). Defer.
- **`worker_threads`** parallel encode (Stage D). Defer.
- **WGSL GPU encode** (Stage E). Defer.
- Any change to **tokenizer JSON `save` / `load`** format, version, or vocab layout. The `kind` and `version` stay at `"llm-transformer-diy/bpe-tokenizer"` / `1`.
- Any change to **train**-time merge construction (`train(text, vocabSize)` is untouched).

## Decisions

### 1. Merge-all-non-overlapping-occurrences-per-pass replaces single-leftmost merge.

The new inner kernel `performGreedyMergePassUint32(buf, len) → number | null` does a **single left-to-right walk** with two cursors (read `r`, write `w`):

1. First sub-walk: scan once to find the **chosen pair** `(chosenL, chosenR)` = pair present in `buf[0..len)` with **minimum merge rank**, breaking ties by smaller `L` then smaller `R` (unchanged from today).
2. If no learned pair exists in `buf[0..len)`, return `null`.
3. Second sub-walk: walk `r = 0..len`. If `r + 1 < len` and `buf[r] == chosenL && buf[r+1] == chosenR`, write `mergedId` to `buf[w]`, advance `r += 2`, `w += 1`. Otherwise, write `buf[r]` to `buf[w]`, advance `r += 1`, `w += 1`. **Non-overlapping by construction** because we step `r += 2` on a match.
4. Return the new logical length `w`.

**Why this is encoding-equivalent to the current "leftmost only" loop** (proof sketch):

- Greedy BPE picks the pair with **minimum rank** at each outer step. Let that pair be `P = (L, R)` and let its non-overlapping occurrences in the buffer be at positions `i₁ < i₂ < … < iₖ`.
- These occurrences are **mutually independent**: by non-overlap, no two share a slot, and merging `P` at position `iⱼ` only changes the buffer values at indices `iⱼ` and `iⱼ + 1` (they collapse to one slot of `mergedId`). It does not introduce or remove any _other_ `P` occurrence (the new neighbors of `mergedId` are not equal to `(L, R)` because they're `(prev, mergedId)` and `(mergedId, next)`).
- Therefore, after the current outer step in the leftmost loop merges the leftmost `P`, the next outer step finds `P` still present at all other occurrences with **the same minimum rank**, so it is chosen again, and so on, until all `k` occurrences of `P` are consumed. Only **after** the `k`-th merge can a different (necessarily ≥ rank) pair become a candidate.
- The merged buffer state after `k` outer iterations of "leftmost only" is **identical** to the buffer state after one outer iteration of "merge all non-overlapping `P`": same `mergedId` written at the same `k` positions, same surviving slots in between.
- Equivalence then follows by induction on outer iterations.

This is also the standard formulation in **minBPE** (Karpathy) — the spec change is a generalization of step 3, not a behavior change.

**Alternatives considered:**

- _Keep leftmost-only and just micro-optimize the inner walk_ — Rejected: leaves the dominant constant factor on the table (1.3–3× per the research doc).
- _Adopt priority-queue + linked-list now (Stage C)_ — Rejected for this change: M-effort, wider correctness surface, and orthogonal to the constant-factor wins. Listed as the natural follow-up.

### 2. Yield cadence in `encodeAsync` is **outer-pass**, not per-merge-iteration.

After (1), each "outer pass" removes potentially many slots, so the existing per-iteration progress + yield logic naturally fires far less often. We keep the **wall-time-bounded** yield (`now - lastYieldWallMs >= TIME_SLICE_ENCODE_ASYNC_MS`) as the **primary** trigger and gate the **counter-based** yield behind a coarse "every N passes" constant (e.g. `EMIT_PASS_EVERY_ASYNC = 8`) instead of `EMIT_MERGE_EVERY_ASYNC = 512`. We continue to **emit** progress on the same wall-time/cadence rules so the TUI keeps painting.

We do **not** introduce `setImmediate` vs `setTimeout(0)` differentiation in this change — that is environment-specific and the research doc lists it as a sub-1.3× win. **`yieldEventLoopForHostUi()`** stays as the abstraction; we may re-tune internals later in a separate change if profiling demands it.

**Alternatives considered:**

- _Drop yields entirely and run synchronously when called from CLI_ — Rejected: breaks Ctrl+C and TUI repaint guarantees during the multi-second encode window; users explicitly need progress.
- _`process.nextTick` for finer-grained yield_ — Rejected: doesn't drain libuv I/O, breaks the "TUI repaints" contract.

### 3. Add a typed-buffer encode surface; keep `number[]` returns for compatibility.

Introduce **additive** entry points:

- **`encodeUint32(text, hooks?): Uint32Array`** — synchronous, returns the encoded ids as a freshly allocated **`Uint32Array`** sized to logical length.
- **`encodeUint32Async(text, hooks?): Promise<Uint32Array>`** — async equivalent.

**`encode`** and **`encodeAsync`** are reimplemented as **thin adapters**: they call the `Uint32` form and return `Array.from(result)` to satisfy the existing `Tokenizer` contract. This way, callers that don't care about typed buffers see no change; **`apps/cli/src/cmd/train.tsx`** drops one allocation by calling the typed entry directly.

We deliberately do **not** widen the **`Tokenizer`** interface in this change. Doing so would force every implementation (incl. `CharTokenizer`) to add a typed-buffer method. The typed surface is **`BPETokenizer`**-specific and discoverable; widening the interface is a separate, larger decision.

**Alternatives considered:**

- _Change `Tokenizer.encode` return type to `Uint32Array`_ — Rejected: breaking change to the shared contract and to every implementation; out of scope.
- _Always return `Uint32Array` from `BPETokenizer.encode` and break local callers_ — Rejected: violates the `Tokenizer` interface and forces ripple edits unrelated to perf.

### 4. `mergeIteration` semantics evolve to "best-rank pass".

**`BpeEncodeHooks.onProgress`** events of `phase: 'merge'` will now report `mergeIteration` as the count of **outer passes** (each pass = one rank-selection + one full collapse walk), not single-position merges. The dashboard's pessimistic upper bound `max(1, initialTokenCount - 1)` is still a **valid worst-case** upper bound on `mergeIteration` (each pass removes ≥1 slot in the worst case where exactly one occurrence of the chosen pair existed). Real runs finish further below the bound, which the existing **"worst-case bar — may finish sooner"** TUI copy already covers — no UX change is required.

We document this semantic change in **`BpeEncodeHooks`** types so future readers don't conflate "iteration" with "single merge".

## Risks / Trade-offs

- **[Subtle encoding regression on edge cases]** → Mitigation: extend the existing tokenizer Vitest suite with a **golden equivalence test** that runs both the current "leftmost" loop and the new "merge-all" loop on a corpus sample drawn from `corpus/real/` (or a deterministic synthetic) and asserts **bit-identical** id sequences for `encode` and `encodeAsync`. Also keep all existing round-trip and tie-break tests.
- **[Progress UX regress because passes are coarser]** → Mitigation: keep wall-time yield cadence so the TUI still ticks at human-perceptible intervals; only the counter-based cadence is relaxed. Re-validate by eye on a `train --dry-run`-style pass.
- **[`mergeIteration` counter meaning shifts under existing dashboards/logs]** → Mitigation: documented in types and in the proposal/change-archive log; ETA math is unchanged because the **upper bound** is still valid. No external metrics depend on the old fine-grained counter.
- **[Typed-buffer adapter still pays `Array.from`]** → Accepted: callers that opt into `encodeUint32` skip it; `Tokenizer.encode` callers (e.g. tests, CharTokenizer parity) keep paying it. This is correct — the change is additive, not a global migration.
- **[Window-of-time when `encode` and `encodeAsync` paths diverge while we land the change]** → Mitigation: implement the merge-all pass once as a private helper; both `encodeUint32` and `encodeUint32Async` call it; `encode` and `encodeAsync` forward to those. Single source of truth in code.

## Migration Plan

This change is internal and **does not** alter:

- the tokenizer **JSON format** (`kind`, `version`, `merges`, `vocab` shapes);
- the **`Tokenizer`** interface;
- **`encode(x)` / `encodeAsync(x)` outputs** for any input `x` and any trained tokenizer.

Existing on-disk **`tokenizer.json`** files load and produce identical token sequences after this change. Existing on-disk **token caches** (`apps/cli/src/cmd/corpus-token-cache.ts`) remain valid because the cache key includes the tokenizer hash and the encoded ids are bit-identical.

**Rollback** is a straightforward revert of the tokenizer-internal change plus reverting `train.tsx` back to the `Array.from` adapter path.

## Open Questions

- None blocking. Future stages B–E from `docs/research/bpe-encode-throughput.md` (pre-tokenization, per-piece cache, priority-queue+linked-list, worker pool, WGSL) remain as separate proposals.
