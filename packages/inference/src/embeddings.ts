/**
 * Read-only utilities for exploring a trained model's token-embedding geometry:
 * row lookup, mean pooling, cosine similarity, and top-K nearest-neighbour search.
 *
 * Stays CPU-only on **`Float32Array`**; no GPU / autograd dependency. Intended to back
 * **`llm embeddings`** CLI subcommands (encode / decode / similarity / nearest / expression).
 */

/** Shape `[vocab, dModel]`, row-major. */
export type EmbeddingMatrix = {
  readonly data: Float32Array;
  readonly vocab: number;
  readonly dModel: number;
};

export class EmbeddingTokenizerMismatchError extends Error {
  readonly embeddingVocab: number;
  readonly tokenizerVocab: number;
  constructor(embeddingVocab: number, tokenizerVocab: number) {
    super(
      `embedding/tokenizer vocab mismatch: embedding=${String(embeddingVocab)} tokenizer=${String(tokenizerVocab)}`,
    );
    this.name = 'EmbeddingTokenizerMismatchError';
    this.embeddingVocab = embeddingVocab;
    this.tokenizerVocab = tokenizerVocab;
  }
}

export class EmbeddingDimensionError extends Error {
  readonly expected: number;
  readonly got: number;
  constructor(expected: number, got: number, what = 'vector') {
    super(`${what} dimension mismatch: expected ${String(expected)}, got ${String(got)}`);
    this.name = 'EmbeddingDimensionError';
    this.expected = expected;
    this.got = got;
  }
}

export class EmbeddingIdOutOfRangeError extends Error {
  readonly id: number;
  readonly vocab: number;
  constructor(id: number, vocab: number) {
    super(`token id ${String(id)} is out of range [0, ${String(vocab)})`);
    this.name = 'EmbeddingIdOutOfRangeError';
    this.id = id;
    this.vocab = vocab;
  }
}

export class EmbeddingZeroNormError extends Error {
  constructor(which: 'a' | 'b' | 'query') {
    super(`cosine similarity undefined: ${which} has zero norm`);
    this.name = 'EmbeddingZeroNormError';
  }
}

/** Builds a thin {@link EmbeddingMatrix} view over a flat Float32Array. */
export function makeEmbeddingMatrix(
  data: Float32Array,
  vocab: number,
  dModel: number,
): EmbeddingMatrix {
  if (!Number.isInteger(vocab) || vocab <= 0) {
    throw new Error(`makeEmbeddingMatrix: vocab must be positive integer (got ${String(vocab)})`);
  }
  if (!Number.isInteger(dModel) || dModel <= 0) {
    throw new Error(`makeEmbeddingMatrix: dModel must be positive integer (got ${String(dModel)})`);
  }
  if (data.length !== vocab * dModel) {
    throw new Error(
      `makeEmbeddingMatrix: data length ${String(data.length)} ≠ vocab*dModel (${String(vocab * dModel)})`,
    );
  }
  return { data, vocab, dModel };
}

/**
 * Throws {@link EmbeddingTokenizerMismatchError} when the embedding matrix's vocab
 * dimension does not match the tokenizer's vocab. Callers SHOULD invoke this before
 * any lookup or top-K query (per `inference-embeddings-api` spec).
 */
export function assertEmbeddingMatchesTokenizer(
  matrix: EmbeddingMatrix,
  tokenizer: { readonly vocabSize: number },
): void {
  if (matrix.vocab !== tokenizer.vocabSize) {
    throw new EmbeddingTokenizerMismatchError(matrix.vocab, tokenizer.vocabSize);
  }
}

export type LookupPool = 'mean' | 'none';

/**
 * Returns the embedding row(s) for the given token ids.
 *
 * - `pool: 'mean'` (default): one **`Float32Array`** of length `dModel` (mean of rows).
 * - `pool: 'none'`: per-token rows as **`Float32Array[]`**, one per id.
 *
 * Throws on empty input or out-of-range id.
 */
export function lookupEmbedding(
  matrix: EmbeddingMatrix,
  ids: ReadonlyArray<number> | Uint32Array | Int32Array,
  opts?: { readonly pool?: LookupPool },
): Float32Array | Float32Array[] {
  const pool = opts?.pool ?? 'mean';
  const n = ids.length;
  if (n === 0) {
    throw new Error('lookupEmbedding: ids must be non-empty');
  }
  const { data, vocab, dModel } = matrix;
  for (let i = 0; i < n; i++) {
    const id = ids[i] as number;
    if (!Number.isInteger(id) || id < 0 || id >= vocab) {
      throw new EmbeddingIdOutOfRangeError(id, vocab);
    }
  }

  if (pool === 'none') {
    const rows: Float32Array[] = [];
    for (let i = 0; i < n; i++) {
      const id = ids[i] as number;
      const start = id * dModel;
      const row = new Float32Array(dModel);
      row.set(data.subarray(start, start + dModel));
      rows.push(row);
    }
    return rows;
  }

  const out = new Float32Array(dModel);
  for (let i = 0; i < n; i++) {
    const id = ids[i] as number;
    const start = id * dModel;
    for (let j = 0; j < dModel; j++) {
      out[j] += data[start + j] as number;
    }
  }
  if (n > 1) {
    const inv = 1 / n;
    for (let j = 0; j < dModel; j++) {
      out[j] *= inv;
    }
  }
  return out;
}

function l2Norm(v: Float32Array): number {
  let s = 0;
  for (let i = 0; i < v.length; i++) {
    const x = v[i] as number;
    s += x * x;
  }
  return Math.sqrt(s);
}

/** Cosine similarity in **`[-1, 1]`**. Throws {@link EmbeddingZeroNormError} on zero norm. */
export function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) {
    throw new EmbeddingDimensionError(a.length, b.length, 'cosineSimilarity argument');
  }
  const na = l2Norm(a);
  if (na === 0) throw new EmbeddingZeroNormError('a');
  const nb = l2Norm(b);
  if (nb === 0) throw new EmbeddingZeroNormError('b');
  let dot = 0;
  for (let i = 0; i < a.length; i++) {
    dot += (a[i] as number) * (b[i] as number);
  }
  const c = dot / (na * nb);
  if (c > 1) return 1;
  if (c < -1) return -1;
  return c;
}

export type TopKEntry = {
  readonly index: number;
  readonly cosine: number;
};

export type TopKOptions = {
  /** Row indices to drop from the candidate set before ranking. */
  readonly exclude?: ReadonlySet<number>;
};

/**
 * Returns the top-K rows of `matrix` by cosine similarity to `query`,
 * sorted by descending cosine. Ties break by ascending row index for
 * determinism. Computes the query norm once and caches matrix row norms
 * per call (cheap; this is not a hot path).
 */
export function topKByCosine(
  query: Float32Array,
  matrix: EmbeddingMatrix,
  k: number,
  opts?: TopKOptions,
): TopKEntry[] {
  if (query.length !== matrix.dModel) {
    throw new EmbeddingDimensionError(matrix.dModel, query.length, 'query');
  }
  if (!Number.isInteger(k) || k <= 0) {
    throw new Error(`topKByCosine: k must be a positive integer (got ${String(k)})`);
  }
  const qNorm = l2Norm(query);
  if (qNorm === 0) throw new EmbeddingZeroNormError('query');

  const { data, vocab, dModel } = matrix;
  const exclude = opts?.exclude;

  const scores: TopKEntry[] = [];
  for (let row = 0; row < vocab; row++) {
    if (exclude !== undefined && exclude.has(row)) continue;
    const start = row * dModel;
    let dot = 0;
    let rNorm2 = 0;
    for (let j = 0; j < dModel; j++) {
      const x = data[start + j] as number;
      dot += x * (query[j] as number);
      rNorm2 += x * x;
    }
    if (rNorm2 === 0) continue;
    const cos = dot / (Math.sqrt(rNorm2) * qNorm);
    const clamped = cos > 1 ? 1 : cos < -1 ? -1 : cos;
    scores.push({ index: row, cosine: clamped });
  }

  scores.sort((x, y) => {
    if (y.cosine !== x.cosine) return y.cosine - x.cosine;
    return x.index - y.index;
  });
  return scores.slice(0, Math.min(k, scores.length));
}

/**
 * Per-query summary of the cosine distribution from `query` to **every other** row of
 * `matrix`. Used to calibrate qualitative similarity bands (see {@link classifyCosine}):
 * raw cosine values are hard to read because the geometry of any given embedding cone
 * is narrow and corpus-dependent — z-scores against the empirical distribution stay
 * meaningful across checkpoints.
 *
 * Skips zero-norm rows. Pass `exclude` to drop e.g. the query's own row(s) so
 * trivial self-similarity doesn't skew the mean.
 */
export type CosineStats = {
  readonly mean: number;
  readonly std: number;
  /** Number of rows that contributed (excluded / zero-norm rows are not counted). */
  readonly count: number;
};

export function cosineStatsAgainstMatrix(
  query: Float32Array,
  matrix: EmbeddingMatrix,
  opts?: { readonly exclude?: ReadonlySet<number> },
): CosineStats {
  if (query.length !== matrix.dModel) {
    throw new EmbeddingDimensionError(matrix.dModel, query.length, 'query');
  }
  const qNorm = l2Norm(query);
  if (qNorm === 0) throw new EmbeddingZeroNormError('query');

  const { data, vocab, dModel } = matrix;
  const exclude = opts?.exclude;

  let sum = 0;
  let sumSq = 0;
  let count = 0;
  for (let row = 0; row < vocab; row++) {
    if (exclude !== undefined && exclude.has(row)) continue;
    const start = row * dModel;
    let dot = 0;
    let rNorm2 = 0;
    for (let j = 0; j < dModel; j++) {
      const x = data[start + j] as number;
      dot += x * (query[j] as number);
      rNorm2 += x * x;
    }
    if (rNorm2 === 0) continue;
    const cos = dot / (Math.sqrt(rNorm2) * qNorm);
    sum += cos;
    sumSq += cos * cos;
    count += 1;
  }
  if (count === 0) return { mean: 0, std: 0, count: 0 };
  const mean = sum / count;
  const variance = Math.max(0, sumSq / count - mean * mean);
  return { mean, std: Math.sqrt(variance), count };
}

/**
 * Qualitative similarity bands, derived from the **z-score** of `cosine` against the
 * cosine distribution from the query to the rest of the vocab. Thresholds are chosen
 * to track standard-normal tail probabilities so labels stay calibrated across models:
 *
 * - `identical`        z ≥ 5      (self-cosine sits here)
 * - `very-similar`     z ≥ 3      (~top 0.13% of pairs)
 * - `similar`          z ≥ 2      (~top 2.3%)
 * - `weakly-similar`   z ≥ 1      (~top 16%)
 * - `unrelated`        |z| < 1    (~middle 68%)
 * - `weakly-opposite`  z ≤ −1
 * - `opposite`         z ≤ −3
 */
export type SimilarityBand =
  | 'identical'
  | 'very-similar'
  | 'similar'
  | 'weakly-similar'
  | 'unrelated'
  | 'weakly-opposite'
  | 'opposite';

export function classifyCosine(
  cosine: number,
  stats: CosineStats,
): { readonly z: number; readonly band: SimilarityBand } {
  if (stats.std === 0) {
    // Degenerate reference: every reference cosine is the same (often the case
    // for very small or constructed matrices). Any deviation from that constant
    // is structurally an outlier; report z=±Infinity and pick the extreme band.
    const eps = 1e-9;
    if (Math.abs(cosine - stats.mean) <= eps) {
      return { z: 0, band: 'unrelated' };
    }
    if (cosine > stats.mean) {
      return { z: Number.POSITIVE_INFINITY, band: 'identical' };
    }
    return { z: Number.NEGATIVE_INFINITY, band: 'opposite' };
  }
  const z = (cosine - stats.mean) / stats.std;
  let band: SimilarityBand;
  if (z >= 5) band = 'identical';
  else if (z >= 3) band = 'very-similar';
  else if (z >= 2) band = 'similar';
  else if (z >= 1) band = 'weakly-similar';
  else if (z > -1) band = 'unrelated';
  else if (z > -3) band = 'weakly-opposite';
  else band = 'opposite';
  return { z, band };
}

/** Human-readable form of {@link SimilarityBand} (lowercase, with spaces). */
export function describeSimilarityBand(band: SimilarityBand): string {
  switch (band) {
    case 'identical':
      return 'identical';
    case 'very-similar':
      return 'very similar';
    case 'similar':
      return 'similar';
    case 'weakly-similar':
      return 'weakly similar';
    case 'unrelated':
      return 'unrelated';
    case 'weakly-opposite':
      return 'weakly opposite';
    case 'opposite':
      return 'opposite';
  }
}
