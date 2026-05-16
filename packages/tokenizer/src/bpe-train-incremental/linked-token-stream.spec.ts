import { describe, expect, it } from 'vitest';

import {
  END,
  appendLiveSlot,
  createLinkedTokenStream,
  extractLiveTokenIds,
  unitMergeSlot,
} from './linked-token-stream';

function appendAll(values: readonly number[]): ReturnType<typeof createLinkedTokenStream> {
  const stream = createLinkedTokenStream(values.length);
  for (let i = 0; i < values.length; i++) {
    appendLiveSlot(stream, i, values[i]!);
  }
  return stream;
}

describe('BpeLinkedTokenStream', () => {
  it('is empty after construction', () => {
    const stream = createLinkedTokenStream(0);
    expect(stream.liveCount).toBe(0);
    expect(stream.head).toBe(END);
    expect(stream.tail).toBe(END);
    expect(extractLiveTokenIds(stream)).toEqual(new Uint32Array(0));
  });

  it('allocates pre-zeroed pointers and supports a non-zero capacity with no appends', () => {
    const stream = createLinkedTokenStream(8);
    expect(stream.capacity).toBe(8);
    expect(stream.liveCount).toBe(0);
    expect(stream.head).toBe(END);
    for (let i = 0; i < stream.capacity; i++) {
      expect(stream.alive[i]).toBe(0);
      expect(stream.prev[i]).toBe(END);
      expect(stream.next[i]).toBe(END);
    }
  });

  it('append → extract round-trips the input sequence', () => {
    const values = [10, 20, 30, 40, 50];
    const stream = appendAll(values);
    expect(stream.liveCount).toBe(values.length);
    expect(stream.head).toBe(0);
    expect(stream.tail).toBe(values.length - 1);
    expect(Array.from(extractLiveTokenIds(stream))).toEqual(values);
  });

  it('rejects double-append into the same slot', () => {
    const stream = createLinkedTokenStream(2);
    appendLiveSlot(stream, 0, 1);
    expect(() => appendLiveSlot(stream, 0, 2)).toThrow(/already alive/);
  });

  it('merging two adjacent middle slots replaces them with the merged token in place', () => {
    const stream = appendAll([10, 20, 30, 40]);
    // Merge (20, 30) → 99 at slot 1.
    const consumed = unitMergeSlot(stream, 1, 99);
    expect(consumed).toBe(2);
    expect(stream.liveCount).toBe(3);
    expect(Array.from(extractLiveTokenIds(stream))).toEqual([10, 99, 40]);
    // The newly merged slot's neighbors now point past the consumed slot.
    expect(stream.next[1]).toBe(3);
    expect(stream.prev[3]).toBe(1);
    expect(stream.alive[2]).toBe(0);
    expect(stream.tail).toBe(3);
  });

  it('merging the head pair updates head to remain the same slot index', () => {
    const stream = appendAll([10, 20, 30]);
    unitMergeSlot(stream, 0, 99);
    expect(stream.head).toBe(0);
    expect(Array.from(extractLiveTokenIds(stream))).toEqual([99, 30]);
  });

  it('merging the tail pair updates tail to the surviving slot', () => {
    const stream = appendAll([10, 20, 30]);
    unitMergeSlot(stream, 1, 99);
    expect(stream.tail).toBe(1);
    expect(stream.next[1]).toBe(END);
    expect(Array.from(extractLiveTokenIds(stream))).toEqual([10, 99]);
  });

  it('cascaded merges (a,b)→x then (x,c)→y yield the expected stream', () => {
    const stream = appendAll([1, 2, 3]);
    unitMergeSlot(stream, 0, 7); // (1,2) → 7
    expect(Array.from(extractLiveTokenIds(stream))).toEqual([7, 3]);
    unitMergeSlot(stream, 0, 8); // (7,3) → 8
    expect(Array.from(extractLiveTokenIds(stream))).toEqual([8]);
    expect(stream.head).toBe(0);
    expect(stream.tail).toBe(0);
    expect(stream.liveCount).toBe(1);
  });

  it('rejects merging a dead slot or a tail slot', () => {
    const stream = appendAll([1, 2, 3]);
    unitMergeSlot(stream, 1, 99); // consumes slot 2
    expect(() => unitMergeSlot(stream, 2, 100)).toThrow(/not alive/);
    expect(() => unitMergeSlot(stream, 1, 100)).toThrow(/no live successor/);
  });

  it('rejects out-of-range capacity', () => {
    expect(() => createLinkedTokenStream(-1)).toThrow(/non-negative/);
    expect(() => createLinkedTokenStream(1.5)).toThrow(/non-negative integer/);
  });
});
