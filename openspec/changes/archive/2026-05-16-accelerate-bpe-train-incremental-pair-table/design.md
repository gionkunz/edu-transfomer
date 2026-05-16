# Design: Accelerate BPE training with an incremental pair table

## Context

- **Hot loop today (RAM path).** `BPETokenizer.train` / `trainAsync` / `trainAsyncOnUtf8Bytes` materialize the corpus as a single `Uint32Array` (one slot per UTF-8 byte, value `S + b`) and then loop:

  ```typescript
  while (nextId < vocabCap) {
    const freq = countAdjacentPairsOverlappingUint32(stream); // O(N)
    const pair = pickBestMergePair(freq); // O(distinctPairs)
    if (pair === null) break;
    rows[nextId] = concatByteSeq(rows[L], rows[R]);
    stream = rewriteGreedy(stream, L, R, nextId); // O(N) + alloc Uint32Array(outLen)
    nextId += 1;
  }
  ```

  Per merge: **two full-stream passes**. Per-pair work in the count pass: allocate a fresh `${L}\0${R}` string via `bpePairKey` and probe a `Map<string, number>`. End-to-end this is `O(N · M)`.

- **Hot loop today (disk-stream path).** `trainFromFiles` materializes the joined corpus into a `uint32` LE token file and then performs the same loop, but reads the freq map from disk and writes the rewritten stream to a second on-disk file each iteration. `pipelineReadStreamToTokens` and `rewriteGreedyU32LeTokenFile` `await` once **per token** (one for `nextU32`, one for `batch.push`), creating a microtask storm — order `10¹²` microtasks for the user's reported 300 MB / 10 000-merge run, on top of the same `O(N · M)` algorithmic cost.

- **Bound from the literature.** SentencePiece (PR #1208), Karpathy's `minbpe`, and the "Becquin" Rust walkthrough all converge on the same data-structure shape: **doubly-linked list of token slots** + **pair-frequency table indexed by `(L, R)` with per-pair occurrence sets**. Per merge: enumerate the `K` occurrences of the chosen pair via the index, splice each one out of the linked list in `O(1)`, and update the pair counts and occurrence sets at the four affected adjacencies (`(prev, L)`, `(L, R)`, `(R, next)` removed; `(prev, mergedId)`, `(mergedId, next)` added). This brings the per-merge cost from `O(N)` to `O(K_m)` where `K_m` is the occurrence count for the pair chosen at step `m`, and the total work across all merges to `O(N + ΣK_m · log N)` (the `log N` tracks the priority queue / argmax cost, not the slot edits).

- **Equivalence boundary.** The current implementation is defined by two observable invariants per iteration:
  1. The **chosen pair** is the one with maximum count among pairs with count `≥ 2`, ties broken by smaller `L`, then smaller `R`.
  2. After choosing, the corpus stream is rewritten by a **left-to-right greedy non-overlapping** replacement (read cursor advances by 2 on a match, by 1 otherwise).

  The new implementation must preserve **both** invariants per iteration. Greedy left-to-right non-overlapping replacement on `(L, R)` is the same set of merge sites a linked-list walker enumerates from left to right when it consumes its own previously-merged slot before considering the next pair. Section _Decisions_ §4 gives the proof sketch.

- **TUI / UX.** The wolf-tui `TokTrainView` (`apps/cli/src/cmd/tokenizer-train.tsx`) shows phase, corpus size, merges done / cap, and stream size. Today `streamMiB` shrinks each merge (a real-time signal that the algorithm is making progress). After this change, the linked-list slot footprint is **fixed at materialization time** — there is no per-merge "stream size" to display. We replace the per-merge stream-size signal with the existing `mergesDone / mergesCapApprox` progress, which is the same information at a coarser cadence.

- **What `--disk-stream` means after this change.** Today `--disk-stream` selects two distinct behaviors at once: (a) stream the corpus from disk during initial materialization (avoids the giant `Buffer` concat in `concatCorpusAsUtf8Bytes`); (b) keep the working token stream **on disk** between merges. After this change, (b) is gone — there is no per-merge file rewrite. (a) remains, and is the only useful semantic of the flag for users on RAM-tight machines who otherwise hit the `Buffer.allocUnsafe(totalN)` ceiling. The flag stays, with its CLI surface unchanged.

## Goals / Non-Goals

**Goals:**

- Bring `BPETokenizer.trainFromFiles` and `trainAsyncOnUtf8Bytes` from `O(N · M)` to `O(N + M · log N + ΣK_m)` on the same single thread (no `worker_threads`, no GPU). Target for the user's reported 300 MB / ~10 000-merge case: **hours → low minutes**.
- Eliminate per-pair string allocations on the hot path by replacing `Map<string, number>` keyed by `bpePairKey` with a typed-array open-addressing hashmap keyed by `(L, R)` numerically.
- Eliminate the per-token `await` storm in the disk-stream materializer (`pipelineReadStreamToTokens`, `rewriteGreedyU32LeTokenFile`) by switching to bulk-per-`Buffer`-chunk loops with at most one `await` per chunk for cooperative yielding.
- Keep the `Tokenizer` interface, the `BPETokenizer` public surface, the on-disk `tokenizer.json` format, and the CLI commands & flags **unchanged**.
- Produce **bit-identical** ordered `merges` and `vocab` byte rows for any input corpus and any `vocabSize`, so existing tokenizer JSON files, tests, and downstream caches continue to work without migration.

**Non-Goals:**

- **Pre-tokenization** (regex / whitespace+punctuation). That is a tokenizer **format** change and would change encoded outputs. Deferred.
- **`worker_threads` parallel pair counting / encode.** With incremental maintenance the per-merge step is bound by `K_m` not `N`, so worker speedups become a follow-up optimization. Deferred.
- **Priority-queue-driven merge selection** beyond what is needed for `O(log N)` argmax. We adopt the priority queue as the simplest correct argmax structure (lazy invalidation), but do not introduce any other algorithmic complexity (e.g. arena-allocated rank trees, Fibonacci heaps, etc.).
- **GPU / WGSL BPE training.** The merge loop is inherently sequential at the per-merge granularity; GPU wins only with byte-level pretok (BlockBPE), which trades quality. Deferred.
- **Encode path changes.** The archived `optimize-bpe-encode-hot-loop` change already covers the encode-side work. This change is strictly about training.
- **Changes to `tokenizer.json` format, `kind`, `version`, or vocab layout.**
- **Changes to CLI command shape / exit codes / flags.**

## Decisions

### 1. Replace the working `Uint32Array` stream with a doubly-linked list of token slots stored as parallel `Uint32Array`s.

The training data structure becomes a fixed-capacity slot pool of length `N` (the initial byte count, including the synthetic inter-file `\n` separators):

```typescript
interface BpeLinkedTokenStream {
  readonly value: Uint32Array; // value[i] = token id at slot i (only meaningful when alive[i] === 1)
  readonly prev: Uint32Array; // prev[i] = previous live slot index, or 0xFFFFFFFF if none (head)
  readonly next: Uint32Array; // next[i] = next live slot index, or 0xFFFFFFFF if none (tail)
  readonly alive: Uint8Array; // alive[i] = 1 while slot i carries a live token; 0 after a merge consumed it
  head: number; // index of leftmost live slot, or 0xFFFFFFFF when empty
  liveCount: number; // number of live slots; the final encoded token-stream length when training stops
}
```

- **Why fixed-capacity, not growable.** A merge of `(L, R) → newId` replaces `value[i_L] = newId`, advances `next[i_L]` to skip past `i_R`, sets `alive[i_R] = 0`, and patches `prev[next[i_R]]` to point back to `i_L`. No new slot is ever allocated; the slot count strictly **decreases**. The pool sized to the **initial** `N` bounds peak memory at `4·(value+prev+next) + 1·alive = 13N` bytes, plus the pair index (Decision 2). For the user's 300 MB corpus, that is `~3.9 GiB` — well below the 32 GiB-class machines the workload targets, and lower than the existing `--disk-stream` path's working set.
- **Why `0xFFFFFFFF` as the end sentinel.** `Uint32Array` cannot store `null`. Using a `u32` sentinel keeps the representation TypedArray-friendly and lets us use `===` comparisons in the hot loop instead of `Number.isFinite` / boxed numbers. We hard-cap `N < 0xFFFFFFFE` at materialization (`N` of `0xFFFFFFFE` byte tokens is `~4 GiB` of UTF-8, which exceeds the project's reasonable single-corpus ceiling and warrants a clear error).
- **Why parallel `Uint32Array`s instead of an `Int32Array`-of-records or an array of objects.** Object-of-records loses cache locality for 1.2 GiB-class arrays (V8 hidden classes + boxed slots blow up RAM 5–10×). Parallel typed arrays mirror what SentencePiece does in C++ and what `tiktoken` does in Rust; in V8 they are guaranteed flat, dense, and unboxed.

**Alternatives considered:**

- _Keep `Uint32Array` and shift in place per merge, like the encode kernel `performGreedyMergePassUint32` does._ Rejected because that is still `O(N)` per merge; encode can do it because it does many merges per pass over the same buffer (`merge-all-occurrences`), but train picks one pair at a time and would still be `O(N · M)`.
- _`Int32Array` with `-1` sentinel._ Rejected: forces sign-extension thinking for slot indices and complicates `0..N-1` → `value[]` indexing. `Uint32Array` with `0xFFFFFFFF` sentinel is clearer and matches the `value[]` domain (token ids are non-negative).
- _Linked list of small objects (`{ value, prev, next }`)._ Rejected on memory and GC grounds; for 300M slots this would dominate RAM and trash the major GC.

### 2. Replace `Map<string, number>` with a `BpePairFreqIndex` backed by typed-array open-addressing.

```typescript
interface BpePairFreqIndex {
  // Open-addressing hashmap with linear probing.
  // Capacity is a power of two; load factor cap 0.7 triggers grow-and-rehash.
  // Key: (L, R) packed.
  // Slot i is empty when keysL[i] === EMPTY_KEY (we use 0xFFFFFFFF as the empty marker;
  // legitimate pair (0xFFFFFFFF, anything) cannot occur because token ids never reach 2^32 - 1).
  readonly keysL: Uint32Array;
  readonly keysR: Uint32Array;
  readonly counts: Uint32Array;
  readonly occHead: Uint32Array; // occHead[slot] = first slot index in the linked-list of slot positions where pair (L, R) starts
  size: number; // number of non-empty slots (for load-factor accounting)
  capacityMinusOne: number; // capacity - 1, used as a fast modulo
}

interface BpePairOccurrenceList {
  // Per-stream-slot occurrence next/prev pointers, sized to N (the slot pool size).
  // For a linked-stream slot i, occNext[i] and occPrev[i] point to the next/prev
  // stream slot j where pair (value[i], value[next[i]]) starts. 0xFFFFFFFF on the ends.
  readonly occNext: Uint32Array;
  readonly occPrev: Uint32Array;
  // pairOf[i] points back into the pair-index slot describing (value[i], value[next[i]]),
  // or 0xFFFFFFFF if slot i is the rightmost live slot (no successor).
  // Lets us update the per-pair occurrence list in O(1) when slot i is removed/changed.
  readonly pairOf: Uint32Array;
}
```

The pair-index entry-point operations the merge loop relies on are:

| Op                            | Cost   | Where called                                         |
| ----------------------------- | ------ | ---------------------------------------------------- |
| `incrementPair(L, R, atSlot)` | `O(1)` | initial scan + every neighbor change after a merge   |
| `decrementPair(atSlot)`       | `O(1)` | every neighbor change after a merge                  |
| `forEachOccurrence(L, R, cb)` | `O(K)` | once per merge to walk all sites for the chosen pair |
| `pickArgmaxByCountTieLR()`    | `O(P)` | once per merge — `P` = current distinct-pair count   |

The `pickArgmaxByCountTieLR` cost is the only place this change does not strictly improve. Today's `pickBestMergePair` is also `O(distinctPairs)`; we keep the simple linear-scan argmax for clarity and correctness, which makes the per-merge cost `O(K_m + P_m)` and the total `O(N + Σ(K_m + P_m))`. `P_m` is bounded by `min(N, vocabSize²)` in theory but in practice on natural language is `O(distinct bigrams)` — typically `~50 000` for a few-MB corpus and slowly **decreasing** as merges consume distinct pairs. For the user's 300 MB corpus this term is negligible compared to `O(N)` materialization.

If profiling later shows `pickArgmaxByCountTieLR` is the bottleneck (unlikely for the target workload), a follow-up change can swap in a `BinaryHeap`-with-lazy-invalidation; the public BPE API is unaffected.

**Alternatives considered:**

- _`BigInt` keys in a `Map<bigint, ...>`._ Rejected: `BigInt` allocations on the hot path are roughly as expensive as the strings they would replace; defeats the purpose. We use `BigInt` only in the public `pickBestMergePair`-equivalent surface for clarity.
- _`Map<number, Map<number, number>>`._ Better than `Map<string, number>` (no string allocs) but still object/`Map` GC pressure and ~5× slower than typed-array open addressing in V8 microbenchmarks. The encode path uses nested-`Map` as `mergeRankByPair` and that is a known-acceptable cost there because lookups are amortized over many merges per outer pass; the train path probes per pair on every adjacency change so the constant factor matters.
- _Cuckoo hashing or Robin Hood probing._ Rejected for this change; linear probing at load-factor 0.7 is well-known to be V8-friendly and meets the perf target. Open follow-up if profiling shows clustering.

### 3. Train loop becomes a streaming initial scan + incremental merge loop.

```pseudo
materialize():
  allocate slot pool of capacity N (Decision 1)
  for each input chunk (RAM: in stripes of ~2 MiB; disk: per stream Buffer):
    for each byte b in chunk (synchronous tight loop):
      append slot { value: S + b, prev: tail, next: 0xFFFFFFFF, alive: 1 }
      update tail
    if RAM stripe boundary OR disk chunk boundary:
      await yieldToHost?.()              # at most one await per ~MiB, NOT per byte

initial_scan():
  walk live list head → tail; for each slot i with successor j = next[i]:
    incrementPair(value[i], value[j], at: i)
  # cost: O(N), one Map-lookup-equivalent per adjacency

merge_loop():
  while nextId < vocabCap:
    pair = pickArgmaxByCountTieLR()
    if pair is None or pair.count < 2: break
    rows[nextId] = concat(rows[pair.L], rows[pair.R])
    mergesList.push([pair.L, pair.R])

    # Walk all occurrences of (pair.L, pair.R) in *current* left-to-right order.
    # The per-pair occurrence linked list (occNext) is maintained so that
    # iteration order matches the live-list left-to-right order — see Decision 4.
    siteIter = forEachOccurrence(pair.L, pair.R)
    for site i in siteIter:
      # site i: slot i holds pair.L; slot j = next[i] holds pair.R; merge them into i.
      if not alive[i] or not alive[next[i]]: continue   # invalidated by an earlier site this iteration (Decision 4)
      j = next[i]
      prevSlot = prev[i]
      nextSlot = next[j]

      # Drop the three pairs that touched slots {i, j} from the index BEFORE rewriting.
      if prevSlot !== END: decrementPair(at: prevSlot)        # was (value[prevSlot], pair.L)
      decrementPair(at: i)                                     # was (pair.L, pair.R), i.e. the chosen pair
      if nextSlot !== END: decrementPair(at: j)                # was (pair.R, value[nextSlot])

      # Rewrite slot i to the merged token; unlink slot j.
      value[i] = nextId
      next[i] = nextSlot
      if nextSlot !== END: prev[nextSlot] = i
      alive[j] = 0
      liveCount -= 1

      # Add the (up to) two new pairs that touch the new slot i.
      if prevSlot !== END: incrementPair(value[prevSlot], nextId, at: prevSlot)
      if nextSlot !== END: incrementPair(nextId, value[nextSlot], at: i)
    onMergeComplete?({ mergeIndex, newVocabId: nextId, tokenStreamBytes: BigInt(liveCount * 4) })
    await yieldToHost?.()
    nextId += 1

finalize(): same as today
```

Per-merge cost: enumerate `K_m` sites (`O(K_m)`), do `O(1)` work per site, then `pickArgmaxByCountTieLR` is `O(P_m)`. Total: `O(N) [materialize] + O(N) [initial_scan] + Σ(K_m + P_m) [merge loop]`. For natural language with `vocabSize=10 000`, `Σ K_m ≪ N` (each merge consumes at most all occurrences of one pair, and the corpus shrinks accordingly) and `Σ P_m` is dominated by the initial pair count; both terms are firmly sub-`O(N · M)`.

`tokenStreamBytes` reported on `onMergeComplete` becomes `BigInt(liveCount * 4)` — the equivalent on-disk-`u32` size if we were to materialize the current live list. The TUI displays the same number it does today.

### 4. Equivalence with today's "left-to-right greedy non-overlapping rewrite" is preserved.

For a fixed chosen pair `P = (L, R)`, the current `rewriteGreedy` (RAM) and `rewriteGreedyU32LeTokenFile` (disk) walk the stream left to right and replace the **leftmost** match, then advance the read cursor by 2 (so a freshly-emitted `mergedId` cannot become the left half of the next match for the **same** pair). The new linked-list walker enumerates occurrences of `P` in left-to-right order via the per-pair occurrence list (Decision 2).

Two correctness obligations follow:

1. **Iteration order matches left-to-right.** We maintain `occNext` / `occPrev` so that for each pair key, occurrences are linked in stream-position order. Whenever we `incrementPair(L, R, at: i)`, we splice slot `i` into the per-pair list **between the latest occurrence to its left and the earliest occurrence to its right**. This is `O(1)` because we already have direct pointers to slot `i`'s neighbors via `prev[i]` and `next[i]`, and from those neighbors' `pairOf[]` we can reach the adjacent occurrence list nodes for the same pair if any.
2. **Adjacent same-pair occurrences are skipped correctly.** Today's `r += 2` rule means a sequence `… L R L R …` produces **one** merge per pass — the leftmost — and the next would-be match (the freshly-merged `mergedId` paired with the next `R`) is not considered for `(L, R)` again because the value changed. The linked-list walker reproduces this exactly: when site `i_k` is merged and consumes `j = next[i_k]`, the site `i_{k+1}` (the next occurrence of `(L, R)` in the per-pair list) was originally bound to `next[i_{k+1}]` — but if `i_{k+1} === j`, that site is consumed by this same merge, and the `if not alive[i] or not alive[next[i]]` guard at the top of the loop body skips it. Concretely, for `… L R L R …` we have stream slots `[a, b, c, d]` with `value=[L,R,L,R]`; the per-pair list for `(L, R)` contains `[a, c]`; merging at `a` sets `alive[b]=0` and patches `next[a] = c`; when the iterator advances to `c`, the guard sees `alive[c]=1` and `alive[next[c]]=alive[d]=1` and proceeds — same as today's leftmost-then-next-leftmost pass.

Edge case for **overlapping** same-letter pairs like `(X, X)`: today's code uses `i + 1 < stream.length && stream[i] === L && stream[i + 1] === R` and advances `i += 2` on a match, so `… X X X …` merges position 0+1, then considers position 2 alone, then stops. The per-pair occurrence list for `(X, X)` on stream `[X, X, X]` (slots `[a, b, c]`) contains `[a, b]` because both adjacencies `(a,b)` and `(b,c)` are `(X, X)`. Merging at `a` consumes `b`; the iterator advances to `b` but the guard `alive[b] === 0` fires, so we skip — leaving stream `[merged, c]`, exactly today's behavior. We add a unit test that pins this scenario explicitly (§Tests).

The `pickBestMergePair` tie-break (smaller `L`, then smaller `R`) is preserved verbatim in `pickArgmaxByCountTieLR` — the count comparison is identical, and on tie we already have `(L, R)` in the index slot.

**Why this gives bit-identical merge tables:** on iteration `m` both algorithms (a) compute the same `(L, R) → count` map (an invariant of the corpus state, not of the data structure) and apply the same argmax + tie-break rule; (b) apply the same set of merge sites in the same left-to-right order. By induction the corpus state after iteration `m` is identical, so the next iteration sees the same map, etc. Therefore the ordered `mergesList`, the `rows[]` byte-row table, and the final `vocabSize` are bit-identical.

**Alternatives considered:**

- _Skip the `occNext` per-pair occurrence list and instead scan the live linked list for sites of the chosen pair on each merge._ Rejected: that re-introduces `O(N)` per merge and defeats the entire change.
- _Use a per-pair `Set<number>` of slot indices._ Functionally correct but JS `Set` is heap-heavy and lookup-unfriendly compared to a typed-array linked list. Same per-pair-occurrence-list trick is what SentencePiece uses.

### 5. Disk-stream materialization streams once into the slot pool with bulk per-`Buffer`-chunk loops.

`trainFromFiles` becomes:

```pseudo
trainFromFiles(paths, vocabCap, options):
  N = sum(stat(p).size for p in paths) + (paths.length - 1)  // include synthetic '\n' separators
  if N > 0xFFFFFFFE: throw "corpus too large for the linked-list slot pool"
  allocateSlotPool(N)
  for each path p (in order):
    if i > 0: appendByte(0x0A)
    for each Buffer chunk emitted by createReadStream(p):
      for j in 0 .. chunk.length - 1: appendByte(chunk[j])    // synchronous tight loop
      onCorpusUtf8Bytes?.(cumulative_bytes)                    // throttled by the existing 256 KiB stride
      await yieldToHost?.()                                    // at most one await per Buffer chunk
  onTokenStreamBytesWritten?.(BigInt(N * 4))                  // single value (no per-merge updates)
  initial_scan(); merge_loop(); finalize()
```

- **Why no temp file.** The `pathA` / `pathB` token files in today's `trainFromFiles` exist only because the merge loop rewrites the stream to disk every iteration. With incremental maintenance the rewrite happens in-place in the slot pool, so the temp files are pure overhead.
- **Why one `await` per `Buffer` chunk and not per byte.** Node's `createReadStream` defaults to 64 KiB chunks; one yield per ~64 KiB still gives the TUI ~5 ms ticks at SSD-bound throughput, which is invisible to humans, while removing ~99.99 % of the microtasks on the path.
- **`writeJoinedCorpusAsTokenIdsToFile` and `rewriteGreedyU32LeTokenFile` stop being part of the training hot path.** We keep `writeJoinedCorpusAsTokenIdsToFile` exported and refactor it internally to use the same bulk-per-chunk loop (no per-byte `await`), because tests and inspection tooling still call it. `rewriteGreedyU32LeTokenFile` and `countAdjacentPairsFromU32LeTokenFile` are kept exported with `@deprecated` JSDoc directing readers to the new in-RAM path; they remain available for the equivalence test (§Tests) and for any external consumers that might exist in downstream tooling.

**Alternatives considered:**

- _Memory-map the corpus and skip the slot copy entirely._ Rejected: ties us to platform-specific `mmap`, fights with how the `webgpu` npm dependency is bundled, and the slot-pool `value[]` array is needed anyway because we mutate values during merges. The cost we save (one streaming copy of `N` bytes) is negligible relative to the merge loop.
- _Keep the on-disk u32 file as a side effect, even though we don't read from it during merges._ Rejected: it would silently double the disk footprint with no payoff.

### 6. `--disk-stream` CLI flag stays; its meaning narrows.

`apps/cli/src/cmd/tokenizer-train.tsx`'s `--disk-stream` flag continues to select `trainFromFiles` (the streaming materializer) instead of `trainAsyncOnUtf8Bytes` + `concatCorpusAsUtf8Bytes` (the in-memory `Buffer.allocUnsafe(totalN)` materializer). The user-visible difference is:

| Path                         | Initial materialization                               | Working data structure during merges     |
| ---------------------------- | ----------------------------------------------------- | ---------------------------------------- |
| Default (`--no-disk-stream`) | `concatCorpusAsUtf8Bytes` builds a `~N`-byte `Buffer` | Linked-list slot pool (`~13N` bytes RAM) |
| `--disk-stream`              | Streamed bytes are appended directly to the slot pool | Linked-list slot pool (`~13N` bytes RAM) |

Peak RAM in the disk-stream path drops from `~5N` (today's `Buffer` + `Uint32Array` + transient rewrite alloc) to `~13N` (slot pool only, no concat), and from `~9N` (today's RAM path peak) to `~14N` (slot pool + the full UTF-8 `Buffer` we still build because `concatCorpusAsUtf8Bytes` is called). For very-large RAM-constrained corpora, `--disk-stream` is now the strictly-better option. The `ramTrainFailureHint` message still nudges users to `--disk-stream` on heap allocation failures.

**Alternatives considered:**

- _Delete `--disk-stream` and always stream from disk._ Rejected: the `concatCorpusAsUtf8Bytes` path tests are extensive and the synchronous-from-RAM path is genuinely simpler when the corpus fits comfortably; keeping both is cheap because they share `runIncrementalMergeLoop` after materialization.

### 7. Observable hooks: `onPairIndexBuilt(distinctInitialPairs)` is added; `tokenStreamBytes` reporting is preserved.

```typescript
type BpeTrainFromFilesHooks = {
  readonly onCorpusUtf8Bytes?: (cumulativeUtf8Bytes: bigint) => void;
  readonly onPairIndexBuilt?: (distinctInitialPairs: number) => void; // NEW: fires once after initial_scan
  readonly onMergeComplete?: (evt: BpeTrainMergeCompleteEvent) => void;
  readonly onTokenStreamBytesWritten?: (cumulativeTokenFileBytes: bigint) => void; // fires once at end of materialize
  readonly yieldToHost?: () => void | Promise<void>;
};
```

- `onPairIndexBuilt` is **optional** and the TUI does not need to render it — it exists so future enhancements (e.g. better progress denominator) can use a real number instead of guessing. The proposal commits to threading it through but not necessarily painting it.
- `onTokenStreamBytesWritten` semantics narrow to **one final emit** at the end of materialization. The TUI's `tokenStreamFileMiB` field shows it during the `materialize` phase; afterwards the field is no longer updated (it was already shown only conditionally on `vm.mode === 'disk' && vm.phase === 'materialize'`, so this matches today's UX exactly).
- `BpeTrainMergeCompleteEvent.tokenStreamBytes` becomes `BigInt(liveCount * 4)` (the slot pool's logical live size in `u32`-equivalent bytes), matching today's "size of the rewritten on-disk file" semantics. Existing `bigToApproxMiB(ev.tokenStreamBytes)` consumers in the TUI keep working unchanged.

### 8. We keep `bpePairKey`, `countAdjacentPairsOverlappingUint32`, `countAdjacentPairsFromU32LeTokenFile`, and `rewriteGreedyU32LeTokenFile` as `@deprecated` exports.

These are tiny modules with tiny APIs; deleting them risks breaking external tooling we may not be aware of (the repository has a CLI binary and downstream consumers per the `cli:link-global` workflow in `AGENTS.md`). We mark them `@deprecated` with a JSDoc pointer to the new internal helpers, and keep them in the equivalence-test harness as the **golden-oracle reference** for the new path. They get a single-line warning in their JSDoc that they are O(N) per call and not suitable for hot-path use on large corpora.

Future cleanup (separate change) can delete them once we are confident no consumer relies on them.

## Risks / Trade-offs

- **[Subtle merge-trace regression on edge cases]** → **Mitigation:** the equivalence test in `packages/tokenizer/src/__tests__/bpe-train-incremental.spec.ts` runs the existing in-process algorithm against the new one on (a) a worst-case overlapping-letters corpus (`"XXXXXX…"`), (b) a tied-frequency synthetic corpus, (c) the existing `corpus/real/` Hamlet sample, (d) a UTF-8 multibyte corpus. It asserts identical ordered `merges` lists and identical `vocab` byte rows for `vocabSize ∈ {S+260, S+1024}`. We also keep all existing `bpe-tokenizer.spec.ts` tests passing without modification.
- **[Linked-list slot pool peak RAM is ~3× today's Uint32Array stream]** → **Mitigation:** documented in the proposal; for the workloads this change targets (`docs/research/bpe-encode-throughput.md` calls out 300 MB / hours), `~4.8 GiB` of slot-pool RAM is well within budget. For RAM-tight machines the existing `--disk-stream` behavior already implies a willingness to spend RAM on the working set; we hard-cap `N < 0xFFFFFFFE` (~4 G byte-tokens) at materialization and emit a clear error message above that. If we later need to support larger corpora, a chunked-slot-pool variant is a contained follow-up — the merge loop does not depend on the slot pool being a single allocation.
- **[`pickArgmaxByCountTieLR` linear scan over distinct pairs is the new dominant per-merge cost when `K_m` is small]** → **Mitigation:** for the target workload (300 MB, 10 000 merges, natural-language) `P_m` is `~50 000` and dropping; the linear scan is `~50 µs` per merge in V8, i.e. `~0.5 s` total — invisible relative to materialization. If profiling later shows otherwise, swapping in a max-heap with lazy invalidation is local to `BpePairFreqIndex` and keeps the rest of the loop unchanged. We do not pre-emptively add the heap because lazy invalidation introduces correctness-surface area we do not need yet.
- **[Open-addressing hashmap clustering on adversarial pair distributions]** → **Mitigation:** the chosen hash is `splitmix64`-style (mix `(BigInt(L) * 0x9E3779B97F4A7C15n + BigInt(R)) >> 0n` then take the low 32 bits, fall back to `(L * 2654435761) ^ (R * 1597334677)` for the V8-fast `i32` path). Load factor capped at 0.7 with grow-and-rehash. Synthetic tests include a corpus designed to populate the hashmap with adversarial bigrams (long runs of the same byte); we assert the merge trace is correct and the wall-time stays sub-linear in `N`.
- **[Disk-stream path's `--disk-stream` flag semantics narrow without warning]** → **Mitigation:** the flag still does what users expect (avoid the giant in-memory `Buffer` concat); the previously implicit "per-merge file rewrite" was an implementation detail no user could observe except via wall time and a `ls -lah` of `tmpdir`. The TUI's "phase: merge" label no longer says "streaming rewrite" (it now says "in-place merge") — see `apps/cli/src/cmd/tokenizer-train.tsx` `phaseLabel` mapping; this is the only user-visible string change.
- **[`writeJoinedCorpusAsTokenIdsToFile` semantics change (one `await` per chunk instead of one per byte) could affect external consumers]** → **Mitigation:** the function's contract — "joined UTF-8 files written as a contiguous u32 LE token stream" — is unchanged. The internal yield cadence becoming coarser is strictly an improvement; any external consumer that depended on per-byte yields was buggy. We keep the existing `onCorpusUtf8Bytes` and `onTokenStreamBytesWritten` callbacks firing on the same byte/MiB strides.

## Migration Plan

This change is internal to `packages/tokenizer` plus a TUI label change in `apps/cli/src/cmd/tokenizer-train.tsx`. It does **not** alter:

- The `Tokenizer` interface.
- The `BPETokenizer` public class surface.
- The on-disk `tokenizer.json` format (`kind`, `version`, `merges`, `vocab` shapes).
- Any CLI command names, flags, exit codes, or environment variables.
- The output of `train` / `trainAsync` / `trainAsyncOnUtf8Bytes` / `trainFromFiles` / `encode` / `encodeAsync` / `encodeUint32` / `encodeUint32Async` / `decode` / `save` / `load` for any input.

Existing on-disk `tokenizer.json` files load unchanged. Existing on-disk **token caches** (`apps/cli/src/cmd/corpus-token-cache.ts`) remain valid because the cache key includes the tokenizer hash and the encoded ids are bit-identical (encode path is untouched).

**Rollback** is a revert of `packages/tokenizer/src/bpe-tokenizer.ts` + the new `bpe-train-incremental.ts` (and the TUI label tweak). The deprecated `bpe-merge-stats.ts` / `bpe-train-disk.ts` exports remain available throughout, so any downstream tooling that imported them keeps working.

## Open Questions

- **Should the TUI render `onPairIndexBuilt`?** Today's TUI shows merges done / cap, which is the most informative number. Adding "distinct pairs in initial scan" might confuse more than it helps. We thread the hook through but **defer** painting it; if a follow-up TUI improvement wants it, the data is there.
- **Should we remove `bpe-merge-stats.ts` exports in this change or in a follow-up?** Deferred — keeping them is cheap and the equivalence test depends on them as the golden oracle. Removing them is a separate decision driven by downstream consumer audit.
- **Should we add `worker_threads`-based parallel pair counting for the initial scan in this change?** No (per the explicit user decision when scoping this proposal). The initial scan is `O(N)` and likely the next bottleneck once the algorithmic restructure lands; addressing it is a clean follow-up that does not invalidate any of the data structures introduced here.
