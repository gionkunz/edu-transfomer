/**
 * Summaries of one decode step's attention for hosts (TUIs): the strongest
 * links across heads, and per-head views. Selection lives here, next to the
 * weights' producer, so a UI never re-implements attention math.
 */

/** One post-softmax link in the global top-10 across all heads at a decode step. */
export type GlobalAttentionConnection = {
  readonly headIndex: number;
  readonly keyIndex: number;
  readonly strength: number;
  /** 1-based rank among the selected top connections (1 = strongest). */
  readonly globalRank: number;
};

type Candidate = {
  readonly headIndex: number;
  readonly keyIndex: number;
  readonly strength: number;
};

function compareCandidates(a: Candidate, b: Candidate): number {
  if (b.strength !== a.strength) return b.strength - a.strength;
  if (a.headIndex !== b.headIndex) return a.headIndex - b.headIndex;
  return a.keyIndex - b.keyIndex;
}

/**
 * Pick up to 10 strongest post-softmax weights across all per-head rows.
 *
 * @param weightsByHead - one softmax row per head, each length **`prefixRows`**
 */
export function selectTop10GlobalConnections(
  weightsByHead: ReadonlyArray<Float32Array>,
): GlobalAttentionConnection[] {
  const candidates: Candidate[] = [];
  for (let h = 0; h < weightsByHead.length; h++) {
    const row = weightsByHead[h];
    if (row === undefined) continue;
    for (let k = 0; k < row.length; k++) {
      candidates.push({ headIndex: h, keyIndex: k, strength: row[k] ?? 0 });
    }
  }
  candidates.sort(compareCandidates);
  return candidates.slice(0, 10).map((c, i) => ({
    headIndex: c.headIndex,
    keyIndex: c.keyIndex,
    strength: c.strength,
    globalRank: i + 1,
  }));
}

/** Unstack row-major `[nHeads * prefixRows]` GPU readback into per-head rows. */
export function unstackAttentionWeights(
  weights: Float32Array,
  nHeads: number,
  prefixRows: number,
): Float32Array[] {
  const rows: Float32Array[] = [];
  for (let h = 0; h < nHeads; h++) {
    const start = h * prefixRows;
    rows.push(weights.subarray(start, start + prefixRows));
  }
  return rows;
}

/** Map key position → best (lowest) global rank when the same key appears via multiple heads. */
export function bestRankByKeyIndex(
  connections: ReadonlyArray<GlobalAttentionConnection>,
): ReadonlyMap<number, number> {
  const map = new Map<number, number>();
  for (const c of connections) {
    const prev = map.get(c.keyIndex);
    if (prev === undefined || c.globalRank < prev) {
      map.set(c.keyIndex, c.globalRank);
    }
  }
  return map;
}

/** Map key position → strongest global connection when the same key appears via multiple heads. */
export function bestConnectionByKeyIndex(
  connections: ReadonlyArray<GlobalAttentionConnection>,
): ReadonlyMap<number, GlobalAttentionConnection> {
  const map = new Map<number, GlobalAttentionConnection>();
  for (const c of connections) {
    const prev = map.get(c.keyIndex);
    if (prev === undefined || c.globalRank < prev.globalRank) {
      map.set(c.keyIndex, c);
    }
  }
  return map;
}

/** Group global top connections by head, sorted by strength within each head. */
export function groupConnectionsByHead(
  connections: ReadonlyArray<GlobalAttentionConnection>,
): ReadonlyMap<number, GlobalAttentionConnection[]> {
  const byHead = new Map<number, GlobalAttentionConnection[]>();
  for (const c of connections) {
    const list = byHead.get(c.headIndex) ?? [];
    list.push(c);
    byHead.set(c.headIndex, list);
  }
  for (const [head, list] of byHead) {
    list.sort((a, b) => compareCandidates(a, b));
    byHead.set(head, list);
  }
  return byHead;
}

/** Per-head summary of one attention row: where the head looks hardest and how spread out it is. */
export type HeadAttentionSummary = {
  readonly headIndex: number;
  /** Key position with the largest weight. */
  readonly peakKeyIndex: number;
  readonly peakStrength: number;
  /**
   * Shannon entropy of the row in bits. **0** means the head looks at exactly
   * one position; **`log2(prefixRows)`** means it averages the whole prefix.
   */
  readonly entropyBits: number;
};

/** One {@link HeadAttentionSummary} per head row (rows as from {@link unstackAttentionWeights}). */
export function summarizeHeads(weightsByHead: ReadonlyArray<Float32Array>): HeadAttentionSummary[] {
  return weightsByHead.map((row, headIndex) => {
    let peakKeyIndex = 0;
    let peakStrength = 0;
    let entropyBits = 0;
    for (let k = 0; k < row.length; k++) {
      const w = row[k] ?? 0;
      if (w > peakStrength) {
        peakStrength = w;
        peakKeyIndex = k;
      }
      if (w > 0) {
        entropyBits -= w * Math.log2(w);
      }
    }
    return { headIndex, peakKeyIndex, peakStrength, entropyBits };
  });
}

/**
 * Downsample one attention row to **`width`** columns for a sparkline: each
 * column is the **maximum** weight of the key positions it covers, so a sharp
 * peak stays visible however long the prefix is. Rows shorter than **`width`**
 * are returned one column per position (no stretching).
 */
export function bucketAttentionRow(row: Float32Array, width: number): Float32Array {
  if (!Number.isInteger(width) || width <= 0) {
    throw new Error(`bucketAttentionRow: width must be a positive integer (got ${String(width)})`);
  }
  if (row.length <= width) {
    return Float32Array.from(row);
  }
  const out = new Float32Array(width);
  for (let k = 0; k < row.length; k++) {
    const col = Math.min(width - 1, Math.floor((k * width) / row.length));
    const w = row[k] ?? 0;
    if (w > (out[col] ?? 0)) {
      out[col] = w;
    }
  }
  return out;
}
