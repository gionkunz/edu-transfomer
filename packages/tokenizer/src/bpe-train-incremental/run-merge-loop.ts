/**
 * Incremental BPE training merge loop. Operates on a {@link BpeLinkedTokenStream}
 * and the {@link BpePairFreqIndex} / {@link BpePairOccurrenceList} that were
 * populated for it, performing one pair merge per iteration in `O(K_m + P_m)`
 * where `K_m` is the chosen pair's occurrence count and `P_m` is the current
 * distinct-pair count in the table.
 *
 * Trace equivalence with the legacy `countAdjacentPairsOverlappingUint32` +
 * `pickBestMergePair` + `rewriteGreedy` loop is established by the spec
 * `tokenizer-bpe-bootstrap` § "Incremental pair-frequency maintenance during
 * training" (the merge sites enumerated by `forEachOccurrence` are exactly the
 * left-to-right non-overlapping match positions today's `rewriteGreedy` would
 * walk through, modulo the per-iteration `alive[i] && alive[next[i]]` guard
 * that handles `(X, X)`-style overlapping-letter corpora).
 */

import { END, type BpeLinkedTokenStream, unitMergeSlot } from './linked-token-stream';
import {
  type BpePairFreqIndex,
  type BpePairOccurrenceList,
  createPairFreqIndex,
  createPairOccurrenceList,
  decrementPair,
  forEachOccurrence,
  incrementPair,
  pickArgmaxByCountTieLR,
} from './pair-freq-index';

/**
 * Hooks consumed by {@link runIncrementalMergeLoop}. Mirrors the subset of
 * {@link import('../bpe-tokenizer').BpeTrainHooks} that fires during the merge
 * loop itself; the rest (corpus / materialize / pair-index hooks) is the caller's
 * responsibility.
 */
export interface IncrementalMergeLoopHooks {
  /** Fires after each successful merge with the merge index, the new vocab id,
   * and the size in bytes of the (logical) `u32`-equivalent token stream. */
  readonly onMergeComplete?: (evt: {
    readonly mergeIndex: number;
    readonly newVocabId: number;
    readonly tokenStreamBytes: bigint;
  }) => void;
  /** Cooperative event-loop yield invoked once per merge. May be omitted for the
   * synchronous training path. */
  readonly yieldToHost?: () => void | Promise<void>;
}

export interface RunIncrementalMergeLoopOpts {
  readonly stream: BpeLinkedTokenStream;
  readonly index: BpePairFreqIndex;
  readonly occList: BpePairOccurrenceList;
  /** Mutable vocab byte-row table; new merge ids are appended at `nextIdInitial`,
   * `nextIdInitial + 1`, ... */
  readonly rows: Uint8Array[];
  /** Inclusive upper bound on the next vocab id (training stops when `nextId === vocabCap`). */
  readonly vocabCap: number;
  /** First merge-derived id (caller passes `S + 256` after specials + byte singletons). */
  readonly nextIdInitial: number;
  readonly hooks?: IncrementalMergeLoopHooks;
}

export interface RunIncrementalMergeLoopResult {
  readonly mergesList: [number, number][];
  /** First id NOT used (i.e. one past the last merge id). Equals `rows.length`. */
  readonly nextId: number;
  readonly finalLiveCount: number;
}

/**
 * Concatenate two `Uint8Array`s. Mirrors the helper in `bpe-tokenizer.ts`.
 */
function concatByteSeq(a: Uint8Array, b: Uint8Array): Uint8Array {
  const out = new Uint8Array(a.length + b.length);
  out.set(a, 0);
  out.set(b, a.length);
  return out;
}

/**
 * Synchronous variant of {@link runIncrementalMergeLoop} for hosts that can't
 * tolerate microtask hops (the public synchronous `BPETokenizer.train` path).
 * Skips `yieldToHost` even if supplied; the `onMergeComplete` hook still fires.
 */
export function runIncrementalMergeLoopSync(
  opts: Omit<RunIncrementalMergeLoopOpts, 'hooks'> & {
    readonly hooks?: Pick<IncrementalMergeLoopHooks, 'onMergeComplete'>;
  },
): RunIncrementalMergeLoopResult {
  const { stream, index, occList, rows, vocabCap, nextIdInitial, hooks } = opts;
  const mergesList: [number, number][] = [];
  let nextId = nextIdInitial;
  while (nextId < vocabCap) {
    const top = pickArgmaxByCountTieLR(index);
    if (top === null) break;
    const advanced = applyOneMerge(stream, index, occList, rows, top.L, top.R, nextId);
    if (!advanced) break;
    mergesList.push([top.L, top.R]);
    hooks?.onMergeComplete?.({
      mergeIndex: mergesList.length - 1,
      newVocabId: nextId,
      tokenStreamBytes: BigInt(stream.liveCount * 4),
    });
    nextId += 1;
  }
  return { mergesList, nextId, finalLiveCount: stream.liveCount };
}

/**
 * Run the incremental BPE merge loop until either `nextId === vocabCap` or no
 * pair has count `≥ 2`. Returns the ordered merge list, the final `nextId`, and
 * the surviving `liveCount` of the slot pool.
 */
export async function runIncrementalMergeLoop(
  opts: RunIncrementalMergeLoopOpts,
): Promise<RunIncrementalMergeLoopResult> {
  const { stream, index, occList, rows, vocabCap, nextIdInitial, hooks } = opts;
  const mergesList: [number, number][] = [];
  let nextId = nextIdInitial;

  while (nextId < vocabCap) {
    const top = pickArgmaxByCountTieLR(index);
    if (top === null) break;
    const advanced = applyOneMerge(stream, index, occList, rows, top.L, top.R, nextId);
    if (!advanced) break;
    mergesList.push([top.L, top.R]);
    const mergeIndex = mergesList.length - 1;
    hooks?.onMergeComplete?.({
      mergeIndex,
      newVocabId: nextId,
      tokenStreamBytes: BigInt(stream.liveCount * 4),
    });
    if (hooks?.yieldToHost !== undefined) {
      await hooks.yieldToHost();
    }
    nextId += 1;
  }

  return {
    mergesList,
    nextId,
    finalLiveCount: stream.liveCount,
  };
}

/**
 * Internal: perform one full merge of pair `(L, R)` into the new id `mergedId`.
 * Walks the snapshot of `(L, R)`'s occurrence list, applying one in-place
 * `unitMergeSlot` per still-valid site, and updates the up-to-three touched
 * pairs (`(prev, L)`, `(L, R)`, `(R, next)`) → up-to-two new pairs (`(prev, merged)`,
 * `(merged, next)`) at each site. Returns `false` when the vocab table is missing
 * a row for `L` or `R` (defensive; callers check `top !== null` first so this
 * should never fire), `true` otherwise.
 */
function applyOneMerge(
  stream: BpeLinkedTokenStream,
  index: BpePairFreqIndex,
  occList: BpePairOccurrenceList,
  rows: Uint8Array[],
  L: number,
  R: number,
  mergedId: number,
): boolean {
  const lb = rows[L];
  const rb = rows[R];
  if (lb === undefined || rb === undefined) {
    throw new Error(
      `runIncrementalMergeLoop: vocab hole for ids ${String(L)}, ${String(R)} (rows.length=${String(rows.length)})`,
    );
  }
  rows[mergedId] = concatByteSeq(lb, rb);

  // Snapshot the occurrence list for (L, R) into a typed array first, because
  // we'll mutate the very list as we walk it (decrementPair unlinks each site).
  // Snapshot cost is `O(K_m)` with no allocation churn beyond the typed-array.
  const sites = collectOccurrencesIntoArray(index, occList, L, R);
  for (let k = 0; k < sites.length; k++) {
    const i = sites[k]!;
    // Site invalidated by an earlier merge in this same iteration (e.g. an
    // overlapping `(X, X)` adjacency that was consumed when its left neighbor
    // was merged). Skip — matches today's `r += 2` non-overlap rule.
    if (stream.alive[i] === 0) continue;
    const j = stream.next[i]!;
    if (j === END || stream.alive[j] === 0) continue;
    if (stream.value[i] !== L || stream.value[j] !== R) continue;

    const prevSlot = stream.prev[i]!;
    const nextSlot = stream.next[j]!;

    // Drop the up-to-three pairs that touched slots {i, j} from the index.
    // (prev, L) is recorded at slot `prevSlot` (its pairOf entry). (L, R) is
    // recorded at slot `i`. (R, next) is recorded at slot `j`.
    if (prevSlot !== END) {
      decrementPair(index, occList, prevSlot);
    }
    decrementPair(index, occList, i);
    if (nextSlot !== END) {
      decrementPair(index, occList, j);
    }

    // Rewrite slot i to the merged token; unlink slot j from the live list.
    unitMergeSlot(stream, i, mergedId);

    // Add the up-to-two new pairs that touch the new slot i.
    if (prevSlot !== END) {
      incrementPair(index, occList, stream, stream.value[prevSlot]!, mergedId, prevSlot);
    }
    if (nextSlot !== END) {
      incrementPair(index, occList, stream, mergedId, stream.value[nextSlot]!, i);
    }
  }
  return true;
}

/** Snapshot a pair's current occurrences into a fresh `Uint32Array` so we can
 * walk the sites without being affected by simultaneous mutations to the
 * occurrence list. Length == count of `(L, R)` at the moment of the snapshot. */
function collectOccurrencesIntoArray(
  index: BpePairFreqIndex,
  occList: BpePairOccurrenceList,
  L: number,
  R: number,
): Uint32Array {
  let count = 0;
  forEachOccurrence(index, occList, L, R, () => {
    count += 1;
  });
  const out = new Uint32Array(count);
  let i = 0;
  forEachOccurrence(index, occList, L, R, (slot) => {
    out[i] = slot;
    i += 1;
  });
  return out;
}

/**
 * Walk the live token stream once and populate the pair-frequency index with the
 * count and occurrence list of every adjacent pair. Returns the populated index
 * and occurrence list together with the count of distinct pairs observed (used
 * for the `onPairIndexBuilt` hook).
 *
 * Allocates the index with a capacity hint of `min(stream.liveCount, 1 << 14)`
 * to keep early grow-and-rehash cycles cheap on large corpora.
 */
export function buildInitialPairIndex(stream: BpeLinkedTokenStream): {
  readonly index: BpePairFreqIndex;
  readonly occList: BpePairOccurrenceList;
  readonly distinctInitialPairs: number;
} {
  const initialCapacityHint = Math.max(16, Math.min(stream.liveCount, 1 << 14));
  const index = createPairFreqIndex(initialCapacityHint);
  const occList = createPairOccurrenceList(stream.capacity);

  let cur = stream.head;
  while (cur !== END) {
    const nxt = stream.next[cur]!;
    if (nxt === END) break;
    incrementPair(index, occList, stream, stream.value[cur]!, stream.value[nxt]!, cur);
    cur = nxt;
  }
  return { index, occList, distinctInitialPairs: index.size };
}
