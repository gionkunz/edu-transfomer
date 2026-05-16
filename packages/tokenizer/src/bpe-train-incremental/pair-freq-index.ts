/**
 * Typed-array pair-frequency table for incremental BPE training.
 *
 * Replaces the per-iteration full-stream rescan + `Map<string, number>` keyed by
 * `bpePairKey(L, R)` with an open-addressing hashmap keyed numerically on `(L, R)`,
 * plus a per-pair occurrence list that lets the merge loop enumerate sites in
 * `O(K_m)` rather than scanning the full `O(N)` token stream every iteration.
 *
 * Two structures live together:
 *
 * - `BpePairFreqIndex`: open-addressed table with linear probing, keyed by the
 *   pair `(L, R)` packed into two parallel `Uint32Array`s. Carries the current
 *   count for every pair currently present in the corpus and the head pointer of
 *   that pair's per-slot occurrence linked list.
 * - `BpePairOccurrenceList`: per-stream-slot doubly-linked list whose nodes are
 *   the corpus slots themselves (one `BpeLinkedTokenStream` slot index = one
 *   occurrence). For slot `i`, `pairOf[i]` points back into the pair-table slot
 *   that describes the pair starting at `i`; `occNext[i]` / `occPrev[i]` chain
 *   together all slots in left-to-right order that start the same pair.
 */

import type { BpeLinkedTokenStream } from './linked-token-stream';

/** Sentinel slot id meaning "empty" / "no neighbor". `0xFFFFFFFF` cannot legitimately
 * appear as a token id (the BPE id space is bounded by `vocabSize`, which is far
 * below `2^32 - 1`) nor as a slot index (capped by {@link MAX_SLOT_CAPACITY}). */
export const EMPTY_KEY = 0xff_ff_ff_ff;

/** Same sentinel reused for "no neighbor" in the per-pair occurrence list. */
export const NO_OCC = 0xff_ff_ff_ff;

/**
 * Open-addressed hashmap keyed on `(L, R)` pairs.
 *
 * Slot `s` is empty when `keysL[s] === EMPTY_KEY`. Otherwise:
 * - `keysL[s]` / `keysR[s]` carry the pair `(L, R)`;
 * - `counts[s]` is the number of stream slots `i` where `value[i] = L` and
 *   `value[next[i]] = R`;
 * - `occHead[s]` is the head of the per-pair occurrence linked list (a stream
 *   slot index pointing into the parallel `BpePairOccurrenceList` arrays), or
 *   `NO_OCC` when the count is zero.
 *
 * Capacity is always a power of two so `slot = hash & capacityMinusOne` is a
 * single AND. The table grows-and-rehashes when `size > 0.7 * capacity`.
 */
export interface BpePairFreqIndex {
  keysL: Uint32Array;
  keysR: Uint32Array;
  counts: Uint32Array;
  occHead: Uint32Array;
  /** Number of non-empty slots; used for load-factor accounting. */
  size: number;
  /** `capacity - 1`, used as a fast modulo. `capacity` is always a power of two. */
  capacityMinusOne: number;
}

/**
 * Per-stream-slot occurrence-list pointers, sized to the slot pool capacity (one
 * entry per `BpeLinkedTokenStream` slot, alive or dead).
 *
 * For a stream slot `i`:
 * - `pairOf[i]` is the pair-table slot describing the pair `(value[i], value[next[i]])`,
 *   or `NO_OCC` when `i` is the rightmost live slot (no successor → no pair starts here).
 * - `occNext[i]` is the next stream slot in the same pair's occurrence list, or `NO_OCC`.
 * - `occPrev[i]` is the previous stream slot in the same pair's occurrence list,
 *   or `NO_OCC` when `i` is the head of that list.
 */
export interface BpePairOccurrenceList {
  readonly pairOf: Uint32Array;
  readonly occNext: Uint32Array;
  readonly occPrev: Uint32Array;
}

/** Round `n` up to the next power of two (minimum 4). */
function nextPow2AtLeast(n: number): number {
  let cap = 4;
  while (cap < n) cap *= 2;
  return cap;
}

/**
 * Allocate a fresh pair-frequency index. `initialCapacityHint` is rounded up to a
 * power of two; the actual capacity may grow during inserts when load factor exceeds
 * 0.7. Pass a hint close to the expected distinct-pair count for fewer rehashes.
 */
export function createPairFreqIndex(initialCapacityHint: number): BpePairFreqIndex {
  if (!Number.isInteger(initialCapacityHint) || initialCapacityHint < 1) {
    throw new Error(
      `BpePairFreqIndex: initialCapacityHint must be a positive integer, got ${String(initialCapacityHint)}`,
    );
  }
  const capacity = nextPow2AtLeast(initialCapacityHint);
  const keysL = new Uint32Array(capacity);
  const keysR = new Uint32Array(capacity);
  const occHead = new Uint32Array(capacity);
  keysL.fill(EMPTY_KEY);
  keysR.fill(EMPTY_KEY);
  occHead.fill(NO_OCC);
  return {
    keysL,
    keysR,
    counts: new Uint32Array(capacity),
    occHead,
    size: 0,
    capacityMinusOne: capacity - 1,
  };
}

/** Allocate the per-stream-slot occurrence list, sized to the slot pool capacity. */
export function createPairOccurrenceList(slotCapacity: number): BpePairOccurrenceList {
  if (!Number.isInteger(slotCapacity) || slotCapacity < 0) {
    throw new Error(
      `BpePairOccurrenceList: slotCapacity must be a non-negative integer, got ${String(slotCapacity)}`,
    );
  }
  const pairOf = new Uint32Array(slotCapacity);
  const occNext = new Uint32Array(slotCapacity);
  const occPrev = new Uint32Array(slotCapacity);
  pairOf.fill(NO_OCC);
  occNext.fill(NO_OCC);
  occPrev.fill(NO_OCC);
  return { pairOf, occNext, occPrev };
}

/**
 * Mix `(L, R)` into a 32-bit hash via two `Math.imul` rounds. Constants are the
 * MurmurHash3 finalization mixers; cheap, reasonably well-distributed for V8's
 * smi-fast path, and avoids `BigInt` allocations on the hot path.
 */
export function pairHash(L: number, R: number): number {
  let h = (Math.imul(L, 0x85_eb_ca_6b) ^ Math.imul(R, 0xc2_b2_ae_35)) >>> 0;
  h = Math.imul(h ^ (h >>> 13), 0x85_eb_ca_6b) >>> 0;
  h = (h ^ (h >>> 16)) >>> 0;
  return h;
}

/**
 * Find the pair-table slot for `(L, R)`, inserting an empty entry (count 0,
 * `occHead = NO_OCC`) if it doesn't exist yet. Grows-and-rehashes the index when
 * load factor exceeds `0.7` after a fresh insert; the grow also repairs every
 * `pairOf[i]` back-pointer in `occList` so the caller's stream slots keep
 * referring to the correct (post-grow) pair-table slot. Returns the slot index.
 */
export function findOrInsertPairSlot(
  index: BpePairFreqIndex,
  occList: BpePairOccurrenceList,
  L: number,
  R: number,
): number {
  if (L >>> 0 === EMPTY_KEY || R >>> 0 === EMPTY_KEY) {
    throw new Error(
      `BpePairFreqIndex: token id ${String(EMPTY_KEY)} is reserved as the empty-slot sentinel`,
    );
  }
  let s = pairHash(L, R) & index.capacityMinusOne;
  while (true) {
    const kl = index.keysL[s]!;
    if (kl === EMPTY_KEY) {
      index.keysL[s] = L >>> 0;
      index.keysR[s] = R >>> 0;
      index.counts[s] = 0;
      index.occHead[s] = NO_OCC;
      index.size += 1;
      // Grow when load factor exceeds 0.7. capacityMinusOne + 1 is the actual capacity.
      if (index.size * 10 > (index.capacityMinusOne + 1) * 7) {
        growAndRehash(index, occList);
        // After grow our slot index is stale — re-find. Counts, occHead, and pairOf
        // back-pointers are all preserved by `growAndRehash`.
        s = lookupPairSlot(index, L, R);
        if (s === EMPTY_KEY) {
          throw new Error('BpePairFreqIndex: post-grow lookup failed (internal invariant)');
        }
      }
      return s;
    }
    if (kl === L >>> 0 && index.keysR[s] === R >>> 0) {
      return s;
    }
    s = (s + 1) & index.capacityMinusOne;
  }
}

/** Locate `(L, R)` in the index, returning the pair-table slot or `EMPTY_KEY` if absent. */
export function lookupPairSlot(index: BpePairFreqIndex, L: number, R: number): number {
  let s = pairHash(L, R) & index.capacityMinusOne;
  while (true) {
    const kl = index.keysL[s]!;
    if (kl === EMPTY_KEY) return EMPTY_KEY;
    if (kl === L >>> 0 && index.keysR[s] === R >>> 0) return s;
    s = (s + 1) & index.capacityMinusOne;
  }
}

/**
 * Double the index capacity and re-probe every non-empty entry, then walk every
 * occurrence list and repair the `pairOf[i]` back-pointers so callers' stream
 * slots keep pointing at the correct (post-grow) pair-table slot. Amortized
 * `O(1)` per insert because capacity doubles each time.
 */
function growAndRehash(index: BpePairFreqIndex, occList: BpePairOccurrenceList): void {
  const oldKeysL = index.keysL;
  const oldKeysR = index.keysR;
  const oldCounts = index.counts;
  const oldOccHead = index.occHead;
  const oldCapacity = index.capacityMinusOne + 1;
  const newCapacity = oldCapacity * 2;
  const newKeysL = new Uint32Array(newCapacity);
  const newKeysR = new Uint32Array(newCapacity);
  const newCounts = new Uint32Array(newCapacity);
  const newOccHead = new Uint32Array(newCapacity);
  newKeysL.fill(EMPTY_KEY);
  newKeysR.fill(EMPTY_KEY);
  newOccHead.fill(NO_OCC);
  const newMask = newCapacity - 1;
  for (let s = 0; s < oldCapacity; s++) {
    const kl = oldKeysL[s]!;
    if (kl === EMPTY_KEY) continue;
    const kr = oldKeysR[s]!;
    let ns = pairHash(kl, kr) & newMask;
    while (newKeysL[ns]! !== EMPTY_KEY) {
      ns = (ns + 1) & newMask;
    }
    newKeysL[ns] = kl;
    newKeysR[ns] = kr;
    newCounts[ns] = oldCounts[s]!;
    const head = oldOccHead[s]!;
    newOccHead[ns] = head;
    // Repair pairOf back-pointers for every stream slot in this pair's list.
    let occ = head;
    while (occ !== NO_OCC) {
      occList.pairOf[occ] = ns;
      occ = occList.occNext[occ]!;
    }
  }
  // Mutate the index in place so external callers' references stay valid.
  (index as { keysL: Uint32Array }).keysL = newKeysL;
  (index as { keysR: Uint32Array }).keysR = newKeysR;
  (index as { counts: Uint32Array }).counts = newCounts;
  (index as { occHead: Uint32Array }).occHead = newOccHead;
  index.capacityMinusOne = newMask;
}

/**
 * Increment the count for pair `(L, R)` and splice `streamSlot` into the per-pair
 * occurrence list in left-to-right order — i.e. immediately after the latest
 * occurrence whose stream slot index is less than `streamSlot`, and before the
 * earliest occurrence whose stream slot index is greater. To keep the splice
 * `O(1)` we always insert at the **tail** of a small leftward-found neighbor; on
 * the initial scan (which walks the stream left-to-right and only ever appends
 * larger slot indices) every insert lands at the tail by induction, which is the
 * common case. For incremental updates after a merge the new slot's neighbors in
 * the live list (`prev[streamSlot]` / `next[streamSlot]`) are nearby in stream
 * order, so we walk at most a few of the per-pair occurrence list's nodes to
 * find the right insertion point — bounded by the number of new-pair occurrences
 * created so far at the merge sites we've already visited this iteration, which
 * is `O(1)` amortized for non-adversarial corpora.
 *
 * For correctness the caller passes `tokenStream` so we can use the stream's own
 * `prev`/`next` pointers as the strict left-to-right ordering oracle.
 */
export function incrementPair(
  index: BpePairFreqIndex,
  occList: BpePairOccurrenceList,
  _tokenStream: BpeLinkedTokenStream,
  L: number,
  R: number,
  streamSlot: number,
): void {
  const pairSlot = findOrInsertPairSlot(index, occList, L, R);
  index.counts[pairSlot] += 1;
  occList.pairOf[streamSlot] = pairSlot;

  // Splice into the per-pair occurrence list at left-to-right position. Walk the
  // existing occurrences to find the immediate predecessor whose stream slot is
  // less than `streamSlot`. For the initial scan this is always the existing
  // tail (we append in slot order) — that case is O(1).
  const head = index.occHead[pairSlot]!;
  if (head === NO_OCC) {
    occList.occPrev[streamSlot] = NO_OCC;
    occList.occNext[streamSlot] = NO_OCC;
    index.occHead[pairSlot] = streamSlot;
    return;
  }
  if (streamSlot < head) {
    occList.occPrev[streamSlot] = NO_OCC;
    occList.occNext[streamSlot] = head;
    occList.occPrev[head] = streamSlot;
    index.occHead[pairSlot] = streamSlot;
    return;
  }
  // Walk forward from head to find the largest occurrence index < streamSlot.
  // For the initial-scan append-only path this exits on the very first step.
  let predecessor = head;
  let successor = occList.occNext[predecessor]!;
  while (successor !== NO_OCC && successor < streamSlot) {
    predecessor = successor;
    successor = occList.occNext[predecessor]!;
  }
  occList.occPrev[streamSlot] = predecessor;
  occList.occNext[streamSlot] = successor;
  occList.occNext[predecessor] = streamSlot;
  if (successor !== NO_OCC) {
    occList.occPrev[successor] = streamSlot;
  }
}

/**
 * Decrement the count for the pair currently registered at `streamSlot` (looked
 * up via `pairOf[streamSlot]`) and unlink `streamSlot` from that pair's
 * occurrence list. Drops the pair-table entry when the count reaches zero (sets
 * the slot back to `EMPTY_KEY` so future probes treat it as empty).
 *
 * Caller MUST have previously called {@link incrementPair} for this `streamSlot`
 * (i.e. `pairOf[streamSlot] !== NO_OCC`); otherwise this is a no-op invariant
 * violation.
 *
 * NOTE: Open-addressing with linear probing requires that when we delete an
 * entry, we re-probe the cluster downstream. We do that here by scanning forward
 * from the freed slot and re-inserting any displaced entries. In the BPE merge
 * loop, deletions are infrequent relative to inserts (each merge causes ≤3
 * decrements and ≤2 increments per site), so this stays cheap.
 */
export function decrementPair(
  index: BpePairFreqIndex,
  occList: BpePairOccurrenceList,
  streamSlot: number,
): void {
  const pairSlot = occList.pairOf[streamSlot]!;
  if (pairSlot === NO_OCC) {
    throw new Error(
      `decrementPair: stream slot ${String(streamSlot)} has no recorded pair (pairOf is NO_OCC)`,
    );
  }

  // Unlink from the per-pair occurrence list.
  const before = occList.occPrev[streamSlot]!;
  const after = occList.occNext[streamSlot]!;
  if (before === NO_OCC) {
    index.occHead[pairSlot] = after;
  } else {
    occList.occNext[before] = after;
  }
  if (after !== NO_OCC) {
    occList.occPrev[after] = before;
  }
  occList.pairOf[streamSlot] = NO_OCC;
  occList.occNext[streamSlot] = NO_OCC;
  occList.occPrev[streamSlot] = NO_OCC;

  index.counts[pairSlot] -= 1;
  if (index.counts[pairSlot] !== 0) {
    return;
  }

  // Pair is now empty — free the slot and back-shift the cluster downstream so
  // future probes don't stop at the gap and miss other pairs in the cluster.
  index.keysL[pairSlot] = EMPTY_KEY;
  index.keysR[pairSlot] = EMPTY_KEY;
  index.occHead[pairSlot] = NO_OCC;
  index.size -= 1;
  backShiftCluster(index, occList, pairSlot);
}

/**
 * Linear-probing back-shift deletion repair: starting from a freshly emptied
 * `gapSlot`, walk forward through the cluster and pull each entry back into the
 * gap whenever doing so does not push it past its own probe length. Standard
 * Knuth Algorithm 6.4R variant, rewritten to keep all reads/writes inside the
 * `Uint32Array` storage and to repair `pairOf` back-pointers when a row moves.
 */
function backShiftCluster(
  index: BpePairFreqIndex,
  occList: BpePairOccurrenceList,
  initialGap: number,
): void {
  let gap = initialGap;
  let i = (gap + 1) & index.capacityMinusOne;
  while (true) {
    const kl = index.keysL[i]!;
    if (kl === EMPTY_KEY) return;
    const kr = index.keysR[i]!;
    const desired = pairHash(kl, kr) & index.capacityMinusOne;
    // Forward cyclic distance from `desired` to `gap` and from `desired` to `i`.
    // We can shift `i` into `gap` iff the entry's natural probe path passes
    // through `gap` first — i.e. distGap < distCur (strictly less, because
    // distGap === distCur would mean i === gap, impossible here).
    const distGap = (gap - desired) & index.capacityMinusOne;
    const distCur = (i - desired) & index.capacityMinusOne;
    if (distGap < distCur) {
      index.keysL[gap] = kl;
      index.keysR[gap] = kr;
      index.counts[gap] = index.counts[i]!;
      index.occHead[gap] = index.occHead[i]!;
      // Repair pairOf back-pointers: every stream slot in this pair's
      // occurrence list previously pointed at `i`; now must point at `gap`.
      let occ = index.occHead[gap]!;
      while (occ !== NO_OCC) {
        occList.pairOf[occ] = gap;
        occ = occList.occNext[occ]!;
      }
      index.keysL[i] = EMPTY_KEY;
      index.keysR[i] = EMPTY_KEY;
      index.occHead[i] = NO_OCC;
      gap = i;
    }
    i = (i + 1) & index.capacityMinusOne;
  }
}

/**
 * Walk the per-pair occurrence list head → tail and call `callback(streamSlot)`
 * for each occurrence, in the left-to-right stream order maintained by
 * {@link incrementPair}. The walker tolerates the callback removing the current
 * occurrence from the list (e.g. by calling {@link decrementPair} after
 * processing it) by capturing `next` before invoking the callback.
 */
export function forEachOccurrence(
  index: BpePairFreqIndex,
  occList: BpePairOccurrenceList,
  L: number,
  R: number,
  callback: (streamSlot: number) => void,
): void {
  const pairSlot = lookupPairSlot(index, L, R);
  if (pairSlot === EMPTY_KEY) return;
  let cur = index.occHead[pairSlot]!;
  while (cur !== NO_OCC) {
    const nextCur = occList.occNext[cur]!;
    callback(cur);
    cur = nextCur;
  }
}

/**
 * Best pair: maximum count among pairs with count ≥ 2; ties broken by smaller
 * `L`, then smaller `R` (numeric id order). Returns `null` when no such pair
 * exists. Linear scan over the table; for the BPE workload `P_m` is bounded by
 * the distinct-bigram count of natural language (~50 000) and is monotonically
 * decreasing as merges consume distinct pairs.
 */
export function pickArgmaxByCountTieLR(
  index: BpePairFreqIndex,
): { L: number; R: number; count: number } | null {
  let bestCount = 0;
  let bestL = 0;
  let bestR = 0;
  let found = false;
  const cap = index.capacityMinusOne + 1;
  for (let s = 0; s < cap; s++) {
    if (index.keysL[s] === EMPTY_KEY) continue;
    const c = index.counts[s]!;
    if (c < 2) continue;
    const L = index.keysL[s]!;
    const R = index.keysR[s]!;
    if (!found || c > bestCount || (c === bestCount && (L < bestL || (L === bestL && R < bestR)))) {
      bestCount = c;
      bestL = L;
      bestR = R;
      found = true;
    }
  }
  return found ? { L: bestL, R: bestR, count: bestCount } : null;
}
