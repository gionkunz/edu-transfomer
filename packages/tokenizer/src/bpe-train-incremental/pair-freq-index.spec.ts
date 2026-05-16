import { describe, expect, it } from 'vitest';

import {
  appendLiveSlot,
  createLinkedTokenStream,
  type BpeLinkedTokenStream,
} from './linked-token-stream';
import {
  EMPTY_KEY,
  NO_OCC,
  createPairFreqIndex,
  createPairOccurrenceList,
  decrementPair,
  findOrInsertPairSlot,
  forEachOccurrence,
  incrementPair,
  lookupPairSlot,
  pickArgmaxByCountTieLR,
} from './pair-freq-index';

function makeStream(values: readonly number[]): BpeLinkedTokenStream {
  const stream = createLinkedTokenStream(values.length);
  for (let i = 0; i < values.length; i++) appendLiveSlot(stream, i, values[i]);
  return stream;
}

function collectOccurrences(
  index: ReturnType<typeof createPairFreqIndex>,
  occList: ReturnType<typeof createPairOccurrenceList>,
  L: number,
  R: number,
): number[] {
  const out: number[] = [];
  forEachOccurrence(index, occList, L, R, (slot) => {
    out.push(slot);
  });
  return out;
}

describe('BpePairFreqIndex', () => {
  it('returns null argmax on an empty index', () => {
    const index = createPairFreqIndex(8);
    expect(pickArgmaxByCountTieLR(index)).toBeNull();
  });

  it('records counts and picks argmax with smaller-L, smaller-R tie-break', () => {
    const stream = makeStream([10, 20, 10, 20, 30, 40]);
    const index = createPairFreqIndex(8);
    const occ = createPairOccurrenceList(stream.capacity);
    incrementPair(index, occ, stream, 10, 20, 0);
    incrementPair(index, occ, stream, 20, 10, 1);
    incrementPair(index, occ, stream, 10, 20, 2);
    incrementPair(index, occ, stream, 20, 30, 3);
    incrementPair(index, occ, stream, 30, 40, 4);
    const top = pickArgmaxByCountTieLR(index);
    expect(top).toEqual({ L: 10, R: 20, count: 2 });
  });

  it('argmax breaks frequency ties by smaller L, then smaller R', () => {
    const stream = makeStream([5, 9, 7, 9, 5, 9, 7, 9, 5, 9]);
    const index = createPairFreqIndex(8);
    const occ = createPairOccurrenceList(stream.capacity);
    incrementPair(index, occ, stream, 7, 9, 0);
    incrementPair(index, occ, stream, 7, 9, 1);
    incrementPair(index, occ, stream, 5, 9, 2);
    incrementPair(index, occ, stream, 5, 9, 3);
    expect(pickArgmaxByCountTieLR(index)).toEqual({ L: 5, R: 9, count: 2 });
  });

  it('forEachOccurrence walks sites in left-to-right insert order', () => {
    const stream = makeStream([1, 2, 1, 2, 1, 2]);
    const index = createPairFreqIndex(8);
    const occ = createPairOccurrenceList(stream.capacity);
    incrementPair(index, occ, stream, 1, 2, 0);
    incrementPair(index, occ, stream, 1, 2, 2);
    incrementPair(index, occ, stream, 1, 2, 4);
    expect(collectOccurrences(index, occ, 1, 2)).toEqual([0, 2, 4]);
  });

  it('forEachOccurrence preserves order when inserts arrive out-of-order', () => {
    const stream = makeStream([1, 2, 1, 2, 1, 2]);
    const index = createPairFreqIndex(8);
    const occ = createPairOccurrenceList(stream.capacity);
    incrementPair(index, occ, stream, 1, 2, 4);
    incrementPair(index, occ, stream, 1, 2, 0);
    incrementPair(index, occ, stream, 1, 2, 2);
    expect(collectOccurrences(index, occ, 1, 2)).toEqual([0, 2, 4]);
  });

  it('decrementPair to zero removes the pair from argmax', () => {
    const stream = makeStream([1, 2, 1, 2]);
    const index = createPairFreqIndex(8);
    const occ = createPairOccurrenceList(stream.capacity);
    incrementPair(index, occ, stream, 1, 2, 0);
    incrementPair(index, occ, stream, 1, 2, 2);
    expect(pickArgmaxByCountTieLR(index)).toEqual({ L: 1, R: 2, count: 2 });
    decrementPair(index, occ, 0);
    decrementPair(index, occ, 2);
    expect(pickArgmaxByCountTieLR(index)).toBeNull();
    expect(lookupPairSlot(index, 1, 2)).toBe(EMPTY_KEY);
  });

  it('grows past the load-factor threshold preserving counts and orders', () => {
    const N = 200;
    const stream = createLinkedTokenStream(N * 2);
    for (let i = 0; i < N * 2; i++) appendLiveSlot(stream, i, i + 100);
    const index = createPairFreqIndex(4);
    const occ = createPairOccurrenceList(stream.capacity);
    // Insert N distinct pairs, each twice, at known stream slots.
    for (let i = 0; i < N; i++) {
      const L = 1000 + i;
      const R = 2000 + i;
      incrementPair(index, occ, stream, L, R, 2 * i);
      incrementPair(index, occ, stream, L, R, 2 * i + 1);
    }
    // Argmax must respect tie-break (smallest L wins).
    expect(pickArgmaxByCountTieLR(index)).toEqual({ L: 1000, R: 2000, count: 2 });
    // Each pair's occurrence list must still hold both slots in order.
    for (let i = 0; i < N; i++) {
      expect(collectOccurrences(index, occ, 1000 + i, 2000 + i)).toEqual([2 * i, 2 * i + 1]);
    }
  });

  it('back-shift cluster repair preserves lookups for colliding pairs', () => {
    // Force a hash collision by picking pairs whose hashes mod 4 happen to coincide
    // — but since pairHash is opaque, we just stuff many distinct pairs into a tiny
    // index (capacity 4) and trigger repeated grow-and-rehash, then delete one and
    // re-look-up the others.
    const stream = createLinkedTokenStream(2000);
    for (let i = 0; i < 2000; i++) appendLiveSlot(stream, i, i + 1);
    const index = createPairFreqIndex(4);
    const occ = createPairOccurrenceList(stream.capacity);
    const pairs: [number, number][] = [];
    for (let i = 0; i < 1000; i++) {
      pairs.push([10000 + i, 20000 + i]);
      incrementPair(index, occ, stream, 10000 + i, 20000 + i, i);
    }
    // Delete every other pair and confirm survivors are still findable.
    for (let i = 0; i < 1000; i += 2) {
      decrementPair(index, occ, i);
    }
    for (let i = 0; i < 1000; i++) {
      const slot = lookupPairSlot(index, 10000 + i, 20000 + i);
      if (i % 2 === 0) {
        expect(slot).toBe(EMPTY_KEY);
      } else {
        expect(slot).not.toBe(EMPTY_KEY);
        expect(index.counts[slot]).toBe(1);
      }
    }
  });

  it('findOrInsertPairSlot is idempotent for the same key', () => {
    const index = createPairFreqIndex(8);
    const occ = createPairOccurrenceList(8);
    const a = findOrInsertPairSlot(index, occ, 7, 11);
    const b = findOrInsertPairSlot(index, occ, 7, 11);
    expect(b).toBe(a);
    expect(index.size).toBe(1);
  });

  it('rejects token id 0xFFFFFFFF as a sentinel collision guard', () => {
    const index = createPairFreqIndex(4);
    const occ = createPairOccurrenceList(4);
    expect(() => findOrInsertPairSlot(index, occ, EMPTY_KEY, 1)).toThrow(/reserved/);
    expect(() => findOrInsertPairSlot(index, occ, 1, EMPTY_KEY)).toThrow(/reserved/);
  });

  it('pairOf back-pointer survives the cluster repair after a grow', () => {
    const stream = createLinkedTokenStream(64);
    for (let i = 0; i < 64; i++) appendLiveSlot(stream, i, 100 + i);
    const index = createPairFreqIndex(4);
    const occ = createPairOccurrenceList(stream.capacity);
    // Force several growths.
    for (let i = 0; i < 32; i++) {
      incrementPair(index, occ, stream, 1000, 2000 + i, i);
    }
    // Each stream slot's pairOf should still resolve to the slot whose (L, R)
    // matches via lookupPairSlot.
    for (let i = 0; i < 32; i++) {
      const slot = lookupPairSlot(index, 1000, 2000 + i);
      expect(slot).not.toBe(EMPTY_KEY);
      expect(occ.pairOf[i]).toBe(slot);
    }
  });

  it('decrementPair throws when called on a slot with no recorded pair', () => {
    const index = createPairFreqIndex(8);
    const occ = createPairOccurrenceList(8);
    expect(() => {
      decrementPair(index, occ, 0);
    }).toThrow(/no recorded pair/);
  });

  it('NO_OCC sentinel matches EMPTY_KEY (single 0xFFFFFFFF marker)', () => {
    expect(NO_OCC).toBe(EMPTY_KEY);
  });
});
