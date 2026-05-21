import { describe, expect, it } from 'vitest';

import {
  correlateBadRowsWithTokens,
  scanFloat32FiniteStats,
  scanLogitsRowsByPosition,
} from './train-failure-diagnostics';

describe('scanFloat32FiniteStats', () => {
  it('reports finite min/max and maxAbs', () => {
    const data = new Float32Array([-3, 0, 2.5, -2.5]);
    expect(scanFloat32FiniteStats(data)).toEqual({
      nonFinite: 0,
      maxAbs: 3,
      min: -3,
      max: 2.5,
    });
  });

  it('counts NaN and Inf separately from finite stats', () => {
    const data = new Float32Array([1, Number.NaN, Number.POSITIVE_INFINITY, -4]);
    const s = scanFloat32FiniteStats(data);
    expect(s.nonFinite).toBe(2);
    expect(s.maxAbs).toBe(4);
    expect(s.min).toBe(-4);
    expect(s.max).toBe(1);
  });

  it('returns NaN stats when every element is non-finite', () => {
    const data = new Float32Array([Number.NaN, Number.POSITIVE_INFINITY]);
    const s = scanFloat32FiniteStats(data);
    expect(s.nonFinite).toBe(2);
    expect(Number.isNaN(s.maxAbs)).toBe(true);
    expect(Number.isNaN(s.min)).toBe(true);
    expect(Number.isNaN(s.max)).toBe(true);
  });
});

describe('scanLogitsRowsByPosition', () => {
  const seqLen = 4;
  const vocabSize = 3;

  it('finds a single row with all NaN logits', () => {
    const data = new Float32Array(seqLen * vocabSize);
    data.fill(1);
    const row = 2;
    for (let j = 0; j < vocabSize; j++) {
      data[row * vocabSize + j] = Number.NaN;
    }
    const bad = scanLogitsRowsByPosition(data, seqLen, vocabSize);
    expect(bad).toHaveLength(1);
    expect(bad[0]?.row).toBe(2);
    expect(bad[0]?.nonFiniteInRow).toBe(3);
    expect(bad[0]?.finiteInRow).toBe(0);
  });

  it('reports partial non-finite within a row', () => {
    const data = new Float32Array([1, 2, 3, 4, Number.NaN, 6, 7, 8, 9, 10, 11, 12]);
    const bad = scanLogitsRowsByPosition(data, seqLen, vocabSize);
    expect(bad).toHaveLength(1);
    expect(bad[0]?.row).toBe(1);
    expect(bad[0]?.nonFiniteInRow).toBe(1);
    expect(bad[0]?.finiteInRow).toBe(2);
    expect(bad[0]?.maxAbsFinite).toBe(6);
  });
});

describe('correlateBadRowsWithTokens', () => {
  it('maps row index to corpus indices', () => {
    const bad = [{ row: 1, nonFiniteInRow: 3, finiteInRow: 0, maxAbsFinite: 0 }];
    const input = new Uint32Array([10, 20, 30, 40]);
    const target = new Uint32Array([11, 21, 31, 41]);
    const pos = correlateBadRowsWithTokens(bad, input, target, 1000);
    expect(pos).toHaveLength(1);
    expect(pos[0]).toMatchObject({
      row: 1,
      inputTokenId: 20,
      targetTokenId: 21,
      corpusInputIndex: 1001,
      corpusTargetIndex: 1002,
    });
  });
});
