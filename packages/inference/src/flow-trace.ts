import { summarizeHeads, type HeadAttentionSummary } from './attention-connections';

/**
 * Per-block trace of one decode forward, for views that show the model at
 * work: what each block's attention heads did, how much the block changed the
 * residual stream, and what the model would predict if it stopped there.
 */

/** What the model would predict if it stopped after one block (logit lens). */
export type LayerLensEntry = {
  readonly layerIndex: number;
  /** Most likely next token at this depth, from the **raw** softmax (no temperature / top-k). */
  readonly topTokenId: number;
  readonly topProbability: number;
  /** Raw probability, at this depth, of the token that was finally sampled. */
  readonly sampledProbability: number;
};

/** Logit lens for one decode forward: one entry per block, first block first. */
export type LayerLensPayload = {
  readonly layers: readonly LayerLensEntry[];
};

/**
 * Turn stacked **`[nLayers, vocabSize]`** lens logits into per-layer entries.
 * Softmax runs on the host: it is `nLayers` passes over one vocabulary row.
 */
export function summarizeLayerLens(
  layerLogits: Float32Array,
  nLayers: number,
  sampledTokenId: number,
): LayerLensPayload {
  if (nLayers <= 0 || layerLogits.length % nLayers !== 0) {
    throw new Error(
      `summarizeLayerLens: logits length ${String(layerLogits.length)} not divisible by nLayers ${String(nLayers)}`,
    );
  }
  const V = layerLogits.length / nLayers;
  const layers: LayerLensEntry[] = [];
  for (let l = 0; l < nLayers; l++) {
    const row = layerLogits.subarray(l * V, (l + 1) * V);
    let max = -Infinity;
    let topTokenId = 0;
    for (let i = 0; i < V; i++) {
      const v = row[i] ?? -Infinity;
      if (v > max) {
        max = v;
        topTokenId = i;
      }
    }
    let sum = 0;
    for (let i = 0; i < V; i++) {
      sum += Math.exp((row[i] ?? -Infinity) - max);
    }
    const sampledLogit = row[sampledTokenId];
    layers.push({
      layerIndex: l,
      topTokenId,
      topProbability: 1 / sum,
      sampledProbability: sampledLogit === undefined ? 0 : Math.exp(sampledLogit - max) / sum,
    });
  }
  return { layers };
}

/** Everything one block did during a decode forward. */
export type FlowLayer = LayerLensEntry & {
  /** One summary per attention head of this block (peak link, entropy). */
  readonly heads: readonly HeadAttentionSummary[];
  /**
   * Size of the attention update relative to the residual stream it is added
   * to: **`‖attention output‖ / ‖stream entering the block‖`**. 0.1 means the
   * step nudged the stream by a tenth of its length.
   */
  readonly attentionUpdate: number;
  /** Same ratio for the feed-forward update. */
  readonly feedForwardUpdate: number;
};

export type FlowTrace = {
  readonly layers: readonly FlowLayer[];
  /** Number of context positions the attention rows cover. */
  readonly prefixRows: number;
};

function l2Norm(v: Float32Array): number {
  let sum = 0;
  for (let i = 0; i < v.length; i++) {
    const x = v[i] ?? 0;
    sum += x * x;
  }
  return Math.sqrt(sum);
}

/**
 * Combine the raw captures of a **`flowTrace`** decode forward (see
 * **`GptDecodeStepResult`**) into one entry per block.
 */
export function buildFlowTrace(input: {
  readonly layerLogits: Float32Array;
  readonly allAttention: Float32Array;
  readonly blockVectors: Float32Array;
  readonly nLayers: number;
  readonly nHeads: number;
  readonly sampledTokenId: number;
}): FlowTrace {
  const { layerLogits, allAttention, blockVectors, nLayers, nHeads, sampledTokenId } = input;
  const rowsTotal = nLayers * nHeads;
  if (rowsTotal <= 0 || allAttention.length % rowsTotal !== 0) {
    throw new Error('buildFlowTrace: attention length does not match nLayers × nHeads');
  }
  if (blockVectors.length % (3 * nLayers) !== 0) {
    throw new Error('buildFlowTrace: block vectors length does not match 3 × nLayers');
  }
  const prefixRows = allAttention.length / rowsTotal;
  const dModel = blockVectors.length / (3 * nLayers);
  const lens = summarizeLayerLens(layerLogits, nLayers, sampledTokenId);

  const layers: FlowLayer[] = [];
  for (let l = 0; l < nLayers; l++) {
    const headRows: Float32Array[] = [];
    for (let h = 0; h < nHeads; h++) {
      const start = (l * nHeads + h) * prefixRows;
      headRows.push(allAttention.subarray(start, start + prefixRows));
    }
    const vec = (k: number): Float32Array =>
      blockVectors.subarray((3 * l + k) * dModel, (3 * l + k + 1) * dModel);
    const streamNorm = l2Norm(vec(0));
    const scale = streamNorm > 0 ? streamNorm : 1;
    const entry = lens.layers[l];
    if (entry === undefined) {
      throw new Error('buildFlowTrace: missing lens entry');
    }
    layers.push({
      ...entry,
      heads: summarizeHeads(headRows),
      attentionUpdate: l2Norm(vec(1)) / scale,
      feedForwardUpdate: l2Norm(vec(2)) / scale,
    });
  }
  return { layers, prefixRows };
}
