import { describe, expect, it } from 'vitest';

import {
  appendLiveSlot,
  createLinkedTokenStream,
  extractLiveTokenIds,
} from './linked-token-stream';
import { buildInitialPairIndex, runIncrementalMergeLoop } from './run-merge-loop';

const SPECIALS_COUNT = 2;
const FLOOR = SPECIALS_COUNT + 256;

function streamFromBytes(bytes: ArrayLike<number>): ReturnType<typeof createLinkedTokenStream> {
  const stream = createLinkedTokenStream(bytes.length);
  for (let i = 0; i < bytes.length; i++) {
    appendLiveSlot(stream, i, SPECIALS_COUNT + (bytes[i] as number));
  }
  return stream;
}

function makeRows(): Uint8Array[] {
  const rows: Uint8Array[] = [];
  for (let i = 0; i < SPECIALS_COUNT; i++) {
    rows[i] = new Uint8Array([0]);
  }
  for (let b = 0; b < 256; b++) {
    rows[SPECIALS_COUNT + b] = new Uint8Array([b]);
  }
  return rows;
}

describe('runIncrementalMergeLoop', () => {
  it('produces the expected merge sequence on "abababab"', async () => {
    // "abababab" → bytes [97, 98] repeating 4× → 8 tokens after byte mapping.
    const corpus = new TextEncoder().encode('abababab');
    const stream = streamFromBytes(corpus);
    const { index, occList } = buildInitialPairIndex(stream);
    const rows = makeRows();
    const result = await runIncrementalMergeLoop({
      stream,
      index,
      occList,
      rows,
      vocabCap: FLOOR + 4,
      nextIdInitial: FLOOR,
    });
    // First merge: (a, b) — count 4 in "abababab" — wins over (b, a) (count 3).
    // Bytes a=97, b=98 → ids S+97=99, S+98=100.
    expect(result.mergesList[0]).toEqual([SPECIALS_COUNT + 97, SPECIALS_COUNT + 98]);
    // After merging (a,b)→258 four times, stream becomes [258, 258, 258, 258].
    // Second merge: (258, 258) → 259, count 3 (overlapping). Greedy non-overlapping
    // picks positions 0 and 2 → stream collapses to [259, 259].
    expect(result.mergesList[1]).toEqual([FLOOR, FLOOR]);
    // Third would-be merge: (259, 259) has count 1 → loop stops.
    expect(result.mergesList.length).toBe(2);
    expect(result.nextId).toBe(FLOOR + 2);
    expect(Array.from(extractLiveTokenIds(stream))).toEqual([FLOOR + 1, FLOOR + 1]);
  });

  it('handles overlapping (X, X) corpus correctly (greedy non-overlapping merge)', async () => {
    // "XXXX" → bytes [88, 88, 88, 88]. Initial pair (X, X) appears 3 times overlapping;
    // greedy non-overlapping merge consumes positions 0 and 2, leaving stream [merged, merged].
    const corpus = new TextEncoder().encode('XXXX');
    const stream = streamFromBytes(corpus);
    const { index, occList } = buildInitialPairIndex(stream);
    const rows = makeRows();
    const result = await runIncrementalMergeLoop({
      stream,
      index,
      occList,
      rows,
      vocabCap: FLOOR + 4,
      nextIdInitial: FLOOR,
    });
    expect(result.mergesList[0]).toEqual([SPECIALS_COUNT + 88, SPECIALS_COUNT + 88]);
    // After first merge, stream is [258, 258]. Pair (258, 258) has count 1, loop stops.
    expect(result.mergesList.length).toBe(1);
    expect(Array.from(extractLiveTokenIds(stream))).toEqual([FLOOR, FLOOR]);
  });

  it('respects tie-break (smaller L, then smaller R)', async () => {
    // Construct a corpus where two pairs both have max count.
    // bytes: a b c b a b → pairs (a,b) ×2, (b,c) ×1, (c,b) ×1, (b,a) ×1
    // No tie. Use a tie-inducing pattern instead: "abab cdcd"
    // bytes: a b a b _ c d c d → pairs (a,b)=2, (b,a)=1, (b,_)=1, (_,c)=1, (c,d)=2, (d,c)=1
    // Tie between (a,b) and (c,d) at count 2. Smaller L wins → (a, b) which is (97, 98) < (99, 100).
    const corpus = new TextEncoder().encode('abab cdcd');
    const stream = streamFromBytes(corpus);
    const { index, occList } = buildInitialPairIndex(stream);
    const rows = makeRows();
    const result = await runIncrementalMergeLoop({
      stream,
      index,
      occList,
      rows,
      vocabCap: FLOOR + 1,
      nextIdInitial: FLOOR,
    });
    expect(result.mergesList[0]).toEqual([SPECIALS_COUNT + 97, SPECIALS_COUNT + 98]);
  });

  it('fires onMergeComplete with monotonic mergeIndex and shrinking liveCount', async () => {
    // Use a corpus with repeating pairs so we get multiple merges.
    const corpus = new TextEncoder().encode('abababcccdd extra abababab');
    const stream = streamFromBytes(corpus);
    const { index, occList } = buildInitialPairIndex(stream);
    const rows = makeRows();
    type Evt = { mergeIndex: number; newVocabId: number; tokenStreamBytes: bigint };
    const events: Evt[] = [];
    await runIncrementalMergeLoop({
      stream,
      index,
      occList,
      rows,
      vocabCap: FLOOR + 8,
      nextIdInitial: FLOOR,
      hooks: {
        onMergeComplete: (e) => {
          events.push({
            mergeIndex: e.mergeIndex,
            newVocabId: e.newVocabId,
            tokenStreamBytes: e.tokenStreamBytes,
          });
        },
      },
    });
    expect(events.length).toBeGreaterThan(0);
    for (let i = 0; i < events.length; i++) {
      expect(events[i]?.mergeIndex).toBe(i);
    }
    for (let i = 1; i < events.length; i++) {
      expect(events[i]!.tokenStreamBytes <= events[i - 1]!.tokenStreamBytes).toBe(true);
    }
  });

  it('awaits yieldToHost once per merge', async () => {
    const corpus = new TextEncoder().encode('abababab');
    const stream = streamFromBytes(corpus);
    const { index, occList } = buildInitialPairIndex(stream);
    const rows = makeRows();
    let yieldCount = 0;
    const result = await runIncrementalMergeLoop({
      stream,
      index,
      occList,
      rows,
      vocabCap: FLOOR + 4,
      nextIdInitial: FLOOR,
      hooks: {
        yieldToHost: () => {
          yieldCount += 1;
          return Promise.resolve();
        },
      },
    });
    expect(yieldCount).toBe(result.mergesList.length);
  });
});

describe('buildInitialPairIndex', () => {
  it('reports the correct distinct-pair count', () => {
    const corpus = new TextEncoder().encode('aabb');
    const stream = streamFromBytes(corpus);
    const { distinctInitialPairs } = buildInitialPairIndex(stream);
    // pairs: (a,a), (a,b), (b,b) — 3 distinct.
    expect(distinctInitialPairs).toBe(3);
  });

  it('returns an empty index for a corpus with fewer than 2 tokens', () => {
    const stream = streamFromBytes(new TextEncoder().encode(''));
    const { index, distinctInitialPairs } = buildInitialPairIndex(stream);
    expect(distinctInitialPairs).toBe(0);
    expect(index.size).toBe(0);

    const single = streamFromBytes(new TextEncoder().encode('a'));
    const r2 = buildInitialPairIndex(single);
    expect(r2.distinctInitialPairs).toBe(0);
  });
});
