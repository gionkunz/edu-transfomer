import { describe, expect, it } from 'vitest';

import {
  bestConnectionByKeyIndex,
  bucketAttentionRow,
  bestRankByKeyIndex,
  groupConnectionsByHead,
  selectTop10GlobalConnections,
  summarizeHeads,
  unstackAttentionWeights,
  type GlobalAttentionConnection,
} from './attention-connections';

describe('selectTop10GlobalConnections', () => {
  it('returns all connections when fewer than 10 candidates exist', () => {
    const weights = [new Float32Array([0.5, 0.3, 0.2])];
    const out = selectTop10GlobalConnections(weights);
    expect(out).toHaveLength(3);
    expect(out.map((c) => c.globalRank)).toEqual([1, 2, 3]);
    expect(out[0]?.keyIndex).toBe(0);
    expect(out[1]?.keyIndex).toBe(1);
    expect(out[2]?.keyIndex).toBe(2);
  });

  it('selects the global top 10 across multiple heads', () => {
    const head0 = new Float32Array([0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08]);
    const head1 = new Float32Array([0.09, 0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.16]);
    const out = selectTop10GlobalConnections([head0, head1]);
    expect(out).toHaveLength(10);
    expect(out[0]).toMatchObject({ headIndex: 1, keyIndex: 7, globalRank: 1 });
    expect(out[9]).toMatchObject({ headIndex: 0, keyIndex: 6, globalRank: 10 });
    for (let i = 1; i < out.length; i++) {
      const prev = out[i - 1]!;
      const cur = out[i]!;
      expect(prev.strength).toBeGreaterThanOrEqual(cur.strength);
    }
  });

  it('breaks strength ties by head index then key index', () => {
    const head0 = new Float32Array([0.9, 0.5]);
    const head1 = new Float32Array([0.9, 0.4]);
    const out = selectTop10GlobalConnections([head0, head1]);
    expect(out[0]).toMatchObject({ headIndex: 0, keyIndex: 0, globalRank: 1 });
    expect(out[1]).toMatchObject({ headIndex: 1, keyIndex: 0, globalRank: 2 });
    expect(out[2]).toMatchObject({ headIndex: 0, keyIndex: 1, globalRank: 3 });
  });

  it('handles empty head list', () => {
    expect(selectTop10GlobalConnections([])).toEqual([]);
  });
});

describe('unstackAttentionWeights', () => {
  it('splits row-major stacked weights into per-head rows', () => {
    const stacked = new Float32Array([1, 2, 3, 4]);
    const rows = unstackAttentionWeights(stacked, 2, 2);
    expect(rows).toHaveLength(2);
    expect(Array.from(rows[0]!)).toEqual([1, 2]);
    expect(Array.from(rows[1]!)).toEqual([3, 4]);
  });
});

describe('bestRankByKeyIndex', () => {
  it('keeps the best rank when the same key appears from multiple heads', () => {
    const connections = selectTop10GlobalConnections([
      new Float32Array([0.2, 0.8]),
      new Float32Array([0.9, 0.1]),
    ]);
    const ranks = bestRankByKeyIndex(connections);
    expect(ranks.get(0)).toBe(1);
    expect(ranks.get(1)).toBe(2);
  });
});

describe('bestConnectionByKeyIndex', () => {
  it('keeps the strongest connection when the same key appears from multiple heads', () => {
    const connections: GlobalAttentionConnection[] = [
      { headIndex: 3, keyIndex: 5, strength: 0.26, globalRank: 4 },
      { headIndex: 7, keyIndex: 5, strength: 0.89, globalRank: 2 },
    ];
    const byKey = bestConnectionByKeyIndex(connections);
    expect(byKey.get(5)).toMatchObject({ headIndex: 7, globalRank: 2 });
  });
});

describe('groupConnectionsByHead', () => {
  it('groups and sorts within each head by strength', () => {
    const connections = selectTop10GlobalConnections([
      new Float32Array([0.3, 0.7]),
      new Float32Array([0.5, 0.1]),
    ]);
    const grouped = groupConnectionsByHead(connections);
    expect(grouped.get(0)?.map((c) => c.keyIndex)).toEqual([1, 0]);
    expect(grouped.get(1)?.map((c) => c.keyIndex)).toEqual([0, 1]);
  });
});

describe('summarizeHeads', () => {
  it('reports the peak position and zero entropy for a one-hot row', () => {
    const [h] = summarizeHeads([new Float32Array([0, 0, 1, 0])]);
    expect(h).toMatchObject({ headIndex: 0, peakKeyIndex: 2, peakStrength: 1 });
    expect(h?.entropyBits).toBeCloseTo(0, 6);
  });

  it('reports log2(n) bits for a uniform row', () => {
    const [h] = summarizeHeads([new Float32Array([0.25, 0.25, 0.25, 0.25])]);
    expect(h?.entropyBits).toBeCloseTo(2, 6);
  });
});

describe('bucketAttentionRow', () => {
  it('keeps short rows one column per position', () => {
    expect([...bucketAttentionRow(new Float32Array([0.5, 0.5]), 8)]).toEqual([0.5, 0.5]);
  });

  it('keeps the peak of each bucket when downsampling', () => {
    const row = new Float32Array(100);
    row[37] = 0.75;
    const out = bucketAttentionRow(row, 10);
    expect(out).toHaveLength(10);
    expect(out[3]).toBeCloseTo(0.75, 6);
    expect([...out].filter((v) => v > 0)).toHaveLength(1);
  });
});
