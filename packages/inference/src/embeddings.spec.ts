import { describe, expect, it } from 'vitest';

import {
  assertEmbeddingMatchesTokenizer,
  classifyCosine,
  cosineSimilarity,
  cosineStatsAgainstMatrix,
  describeSimilarityBand,
  EmbeddingDimensionError,
  EmbeddingIdOutOfRangeError,
  EmbeddingTokenizerMismatchError,
  EmbeddingZeroNormError,
  lookupEmbedding,
  makeEmbeddingMatrix,
  topKByCosine,
} from './embeddings';

function rowMajor(rows: number[][]): Float32Array {
  const r = rows.length;
  const c = rows[0]?.length ?? 0;
  const out = new Float32Array(r * c);
  for (let i = 0; i < r; i++) {
    const row = rows[i] as number[];
    for (let j = 0; j < c; j++) out[i * c + j] = row[j] as number;
  }
  return out;
}

describe('embeddings: lookupEmbedding', () => {
  const m = makeEmbeddingMatrix(
    rowMajor([
      [1, 0, 0],
      [0, 2, 0],
      [0, 0, 3],
    ]),
    3,
    3,
  );

  it('returns the row for a single id (default mean pooling, n=1)', () => {
    const v = lookupEmbedding(m, [1]) as Float32Array;
    expect(Array.from(v)).toEqual([0, 2, 0]);
  });

  it('mean-pools multi-token input', () => {
    const v = lookupEmbedding(m, [0, 1]) as Float32Array;
    expect(Array.from(v)).toEqual([0.5, 1, 0]);
  });

  it("pool: 'none' returns per-token rows", () => {
    const rows = lookupEmbedding(m, [0, 2], { pool: 'none' }) as Float32Array[];
    expect(rows).toHaveLength(2);
    expect(Array.from(rows[0]!)).toEqual([1, 0, 0]);
    expect(Array.from(rows[1]!)).toEqual([0, 0, 3]);
  });

  it('throws on empty ids', () => {
    expect(() => lookupEmbedding(m, [])).toThrow(/non-empty/);
  });

  it('throws on out-of-range id', () => {
    expect(() => lookupEmbedding(m, [3])).toThrow(EmbeddingIdOutOfRangeError);
    expect(() => lookupEmbedding(m, [-1])).toThrow(EmbeddingIdOutOfRangeError);
  });
});

describe('embeddings: cosineSimilarity', () => {
  it('identical vectors → 1', () => {
    const v = new Float32Array([0.3, -1.5, 2.1]);
    expect(cosineSimilarity(v, v)).toBeCloseTo(1, 5);
  });

  it('orthogonal vectors → 0', () => {
    const a = new Float32Array([1, 0, 0]);
    const b = new Float32Array([0, 1, 0]);
    expect(cosineSimilarity(a, b)).toBeCloseTo(0, 6);
  });

  it('opposite vectors → -1', () => {
    const a = new Float32Array([1, 2, 3]);
    const b = new Float32Array([-1, -2, -3]);
    expect(cosineSimilarity(a, b)).toBeCloseTo(-1, 5);
  });

  it('throws on dimension mismatch', () => {
    expect(() => cosineSimilarity(new Float32Array([1, 2]), new Float32Array([1, 2, 3]))).toThrow(
      EmbeddingDimensionError,
    );
  });

  it('throws on zero norm', () => {
    const z = new Float32Array([0, 0, 0]);
    const v = new Float32Array([1, 2, 3]);
    expect(() => cosineSimilarity(z, v)).toThrow(EmbeddingZeroNormError);
    expect(() => cosineSimilarity(v, z)).toThrow(EmbeddingZeroNormError);
  });
});

describe('embeddings: topKByCosine', () => {
  const m = makeEmbeddingMatrix(
    rowMajor([
      [1, 0, 0],
      [0.99, 0.01, 0],
      [0, 1, 0],
      [0, 0, 1],
      [-1, 0, 0],
    ]),
    5,
    3,
  );

  it('orders rows by descending cosine to query', () => {
    const top = topKByCosine(new Float32Array([1, 0, 0]), m, 3);
    expect(top.map((e) => e.index)).toEqual([0, 1, 2]);
    expect(top[0]!.cosine).toBeCloseTo(1, 5);
    expect(top[2]!.cosine).toBeCloseTo(0, 6);
  });

  it('respects exclude set', () => {
    const top = topKByCosine(new Float32Array([1, 0, 0]), m, 5, {
      exclude: new Set([0, 1]),
    });
    expect(top[0]!.index).toBe(2);
    for (const e of top) {
      expect(e.index).not.toBe(0);
      expect(e.index).not.toBe(1);
    }
  });

  it('breaks ties by ascending index', () => {
    const tieMatrix = makeEmbeddingMatrix(
      rowMajor([
        [1, 0],
        [1, 0],
        [1, 0],
        [0, 1],
      ]),
      4,
      2,
    );
    const top = topKByCosine(new Float32Array([1, 0]), tieMatrix, 3);
    expect(top.map((e) => e.index)).toEqual([0, 1, 2]);
  });

  it('throws on dimension mismatch', () => {
    expect(() => topKByCosine(new Float32Array([1, 0]), m, 3)).toThrow(EmbeddingDimensionError);
  });

  it('throws on zero query norm', () => {
    expect(() => topKByCosine(new Float32Array([0, 0, 0]), m, 3)).toThrow(EmbeddingZeroNormError);
  });

  it('throws on non-positive k', () => {
    expect(() => topKByCosine(new Float32Array([1, 0, 0]), m, 0)).toThrow(/positive/);
  });
});

describe('embeddings: cosineStatsAgainstMatrix + classifyCosine', () => {
  it('returns mean+std+count over the (non-excluded) rows', () => {
    const m = makeEmbeddingMatrix(
      rowMajor([
        [1, 0],
        [-1, 0],
        [0, 1],
        [0, -1],
      ]),
      4,
      2,
    );
    const stats = cosineStatsAgainstMatrix(new Float32Array([1, 0]), m);
    expect(stats.count).toBe(4);
    expect(stats.mean).toBeCloseTo(0, 6);
    expect(stats.std).toBeGreaterThan(0);
  });

  it('skips excluded rows in the stats', () => {
    const m = makeEmbeddingMatrix(
      rowMajor([
        [1, 0],
        [1, 0],
        [0, 1],
      ]),
      3,
      2,
    );
    const stats = cosineStatsAgainstMatrix(new Float32Array([1, 0]), m, {
      exclude: new Set([0]),
    });
    expect(stats.count).toBe(2);
    expect(stats.mean).toBeCloseTo(0.5, 6);
  });

  it('classifies self-cosine as identical, mean as unrelated, anti-correlated as opposite', () => {
    const stats = { mean: 0, std: 0.1, count: 1000 } as const;
    expect(classifyCosine(1, stats).band).toBe('identical');
    expect(classifyCosine(0.4, stats).band).toBe('very-similar');
    expect(classifyCosine(0.25, stats).band).toBe('similar');
    expect(classifyCosine(0.12, stats).band).toBe('weakly-similar');
    expect(classifyCosine(0.05, stats).band).toBe('unrelated');
    expect(classifyCosine(-0.05, stats).band).toBe('unrelated');
    expect(classifyCosine(-0.2, stats).band).toBe('weakly-opposite');
    expect(classifyCosine(-0.5, stats).band).toBe('opposite');
  });

  it('describeSimilarityBand returns human strings', () => {
    expect(describeSimilarityBand('identical')).toBe('identical');
    expect(describeSimilarityBand('very-similar')).toBe('very similar');
    expect(describeSimilarityBand('weakly-opposite')).toBe('weakly opposite');
  });

  it('classifyCosine handles a degenerate reference (std = 0)', () => {
    // Cosine equals the constant mean: structurally indistinguishable from the rest.
    const same = classifyCosine(0.7, { mean: 0.7, std: 0, count: 10 });
    expect(same.z).toBe(0);
    expect(same.band).toBe('unrelated');
    // Cosine above the constant mean: outlier above → identical.
    const above = classifyCosine(0.95, { mean: 0.7, std: 0, count: 10 });
    expect(above.band).toBe('identical');
    expect(above.z).toBe(Number.POSITIVE_INFINITY);
    // Cosine below the constant mean: outlier below → opposite.
    const below = classifyCosine(0.1, { mean: 0.7, std: 0, count: 10 });
    expect(below.band).toBe('opposite');
    expect(below.z).toBe(Number.NEGATIVE_INFINITY);
  });
});

describe('embeddings: assertEmbeddingMatchesTokenizer', () => {
  it('passes when vocab matches', () => {
    const m = makeEmbeddingMatrix(new Float32Array(6), 3, 2);
    expect(() => assertEmbeddingMatchesTokenizer(m, { vocabSize: 3 })).not.toThrow();
  });
  it('throws with both sizes in the message', () => {
    const m = makeEmbeddingMatrix(new Float32Array(6), 3, 2);
    try {
      assertEmbeddingMatchesTokenizer(m, { vocabSize: 5 });
      throw new Error('expected throw');
    } catch (e) {
      expect(e).toBeInstanceOf(EmbeddingTokenizerMismatchError);
      const msg = (e as Error).message;
      expect(msg).toContain('3');
      expect(msg).toContain('5');
    }
  });
});
