import {
  createGptKvCache,
  destroyGptKvCache,
  disposeGptKvForwardGraph,
  forwardGptDecodeStep,
  forwardGptPrefill,
  readKvCacheHead,
  type GptKVCache,
  type GptModel,
} from '@llm-transformer-diy/model';
import { createIndices, toArray, type TensorRuntime } from '@llm-transformer-diy/tensor';

import { buildFlowTrace, type FlowTrace } from './flow-trace';

/**
 * One fully instrumented forward of the model, for step-by-step explanations:
 * the forward that turns a prompt into the scores for its next token, with the
 * intermediate values of every attention head and every block kept.
 */

/** What one attention head did for the last prompt token. */
export type LessonHead = {
  readonly layerIndex: number;
  readonly headIndex: number;
  /** The token's query, length **`d_k`**: what this head is looking for. */
  readonly query: Float32Array;
  /** One score per prompt position: **`(query · key) / √d_k`**. */
  readonly scores: Float32Array;
  /** Softmax of **`scores`**; sums to 1. */
  readonly weights: Float32Array;
  /** The head's output, length **`d_k`**: the values averaged with **`weights`**. */
  readonly mixed: Float32Array;
  /** Position with the largest weight. */
  readonly peakIndex: number;
  /** Shannon entropy of **`weights`** in bits; 0 = looks at exactly one position. */
  readonly entropyBits: number;
};

export type LessonTrace = {
  readonly promptLength: number;
  readonly nLayers: number;
  readonly nHeads: number;
  readonly dModel: number;
  readonly dK: number;
  /** The model's scores for the next token, one per vocabulary entry (before any sampling). */
  readonly rawLogits: Float32Array;
  /** Embedding row of the last prompt token. */
  readonly tokenVector: Float32Array;
  /** Embedding row of its position. */
  readonly positionVector: Float32Array;
  readonly head: (layerIndex: number, headIndex: number) => LessonHead;
  /**
   * Per-block summary (see **`FlowTrace`**). The lens of each block reports the
   * probability of **`sampledTokenId`** at that depth, so it is built per token.
   */
  readonly flowFor: (sampledTokenId: number) => FlowTrace;
};

function rowOf(stack: Float32Array, row: number, width: number): Float32Array {
  return stack.subarray(row * width, (row + 1) * width);
}

/**
 * Run the model on **`promptIds`** and keep everything about the forward of the
 * **last** prompt token — the one whose output is the next-token prediction.
 *
 * All earlier tokens go through the plain prefill path (they only need to leave
 * their keys and values in the cache); the last one is decoded with capture on.
 * The returned **`cache`** holds the keys/values of the whole prompt and must be
 * released with **`destroyGptKvCache`** by the caller.
 */
export async function runLessonForward(
  rt: TensorRuntime,
  model: GptModel,
  promptIds: Uint32Array,
): Promise<{ readonly trace: LessonTrace; readonly cache: GptKVCache }> {
  const n = promptIds.length;
  if (n === 0) {
    throw new Error('runLessonForward: promptIds must be non-empty');
  }
  const { maxSeqLen, nLayers, nHeads, dModel } = model.config;
  if (n > maxSeqLen) {
    throw new Error(
      `runLessonForward: prompt has ${String(n)} tokens, model context is ${String(maxSeqLen)}`,
    );
  }
  const lastId = promptIds[n - 1];
  if (lastId === undefined) {
    throw new Error('runLessonForward: missing last prompt token');
  }
  const cache = createGptKvCache(rt, model);
  try {
    if (n > 1) {
      const prefix = createIndices(rt, promptIds.slice(0, n - 1), [n - 1]);
      const prefillLogits = forwardGptPrefill(rt, model, prefix, cache);
      // Only the cache rows are needed; this also releases the prefix tensor.
      disposeGptKvForwardGraph(prefillLogits, model, cache);
    }
    const step = await forwardGptDecodeStep(rt, model, lastId, cache, { lessonTrace: true });
    let rawLogits: Float32Array;
    try {
      rawLogits = await toArray(rt, step.logits.tensor);
    } finally {
      disposeGptKvForwardGraph(step.logits, model, cache);
    }
    const {
      layerLogits,
      allAttention,
      blockVectors,
      inputVectors,
      allQueries,
      allScores,
      allMixed,
    } = step;
    if (
      layerLogits === undefined ||
      allAttention === undefined ||
      blockVectors === undefined ||
      inputVectors === undefined ||
      allQueries === undefined ||
      allScores === undefined ||
      allMixed === undefined
    ) {
      throw new Error('runLessonForward: decode step returned an incomplete lesson trace');
    }
    const dK = dModel / nHeads;
    const flowBase = buildFlowTrace({
      layerLogits,
      allAttention,
      blockVectors,
      nLayers,
      nHeads,
      sampledTokenId: 0,
    });

    const head = (layerIndex: number, headIndex: number): LessonHead => {
      if (
        !Number.isInteger(layerIndex) ||
        layerIndex < 0 ||
        layerIndex >= nLayers ||
        !Number.isInteger(headIndex) ||
        headIndex < 0 ||
        headIndex >= nHeads
      ) {
        throw new Error(
          `LessonTrace.head: no head ${String(headIndex)} in block ${String(layerIndex)}`,
        );
      }
      const row = layerIndex * nHeads + headIndex;
      const summary = flowBase.layers[layerIndex]?.heads[headIndex];
      return {
        layerIndex,
        headIndex,
        query: rowOf(allQueries, row, dK),
        scores: rowOf(allScores, row, n),
        weights: rowOf(allAttention, row, n),
        mixed: rowOf(allMixed, row, dK),
        peakIndex: summary?.peakKeyIndex ?? 0,
        entropyBits: summary?.entropyBits ?? 0,
      };
    };

    const trace: LessonTrace = {
      promptLength: n,
      nLayers,
      nHeads,
      dModel,
      dK,
      rawLogits,
      tokenVector: rowOf(inputVectors, 0, dModel),
      positionVector: rowOf(inputVectors, 1, dModel),
      head,
      flowFor: (sampledTokenId) =>
        buildFlowTrace({
          layerLogits,
          allAttention,
          blockVectors,
          nLayers,
          nHeads,
          sampledTokenId,
        }),
    };
    return { trace, cache };
  } catch (err) {
    destroyGptKvCache(cache);
    throw err;
  }
}

/** Keys and values of one head, one row per prompt position. */
export type LessonHeadKv = {
  readonly keys: readonly Float32Array[];
  readonly values: readonly Float32Array[];
};

/** Read one head's keys and values from the cache left by **`runLessonForward`**. */
export async function readLessonHeadKv(
  rt: TensorRuntime,
  cache: GptKVCache,
  layerIndex: number,
  headIndex: number,
): Promise<LessonHeadKv> {
  const { keys, values, dK } = await readKvCacheHead(rt, cache, layerIndex, headIndex);
  const rows = keys.length / dK;
  const split = (stack: Float32Array): Float32Array[] =>
    Array.from({ length: rows }, (_, i) => rowOf(stack, i, dK));
  return { keys: split(keys), values: split(values) };
}

/** The head of a block whose attention is most concentrated (lowest entropy). */
export function mostFocusedHead(trace: LessonTrace, layerIndex: number): number {
  let best = 0;
  let bestEntropy = Infinity;
  for (let h = 0; h < trace.nHeads; h++) {
    const e = trace.head(layerIndex, h).entropyBits;
    if (e < bestEntropy) {
      bestEntropy = e;
      best = h;
    }
  }
  return best;
}

/**
 * Shrink a vector to at most **`width`** entries for drawing: each output is
 * the **mean** of the components it covers, so sign and rough size survive.
 * Vectors that already fit are copied unchanged.
 */
export function bucketVector(v: Float32Array, width: number): Float32Array {
  if (!Number.isInteger(width) || width <= 0) {
    throw new Error(`bucketVector: width must be a positive integer (got ${String(width)})`);
  }
  if (v.length <= width) {
    return Float32Array.from(v);
  }
  const sums = new Float32Array(width);
  const counts = new Uint32Array(width);
  for (let i = 0; i < v.length; i++) {
    const col = Math.min(width - 1, Math.floor((i * width) / v.length));
    sums[col] = (sums[col] ?? 0) + (v[i] ?? 0);
    counts[col] = (counts[col] ?? 0) + 1;
  }
  for (let c = 0; c < width; c++) {
    const count = counts[c] ?? 0;
    if (count > 0) {
      sums[c] = (sums[c] ?? 0) / count;
    }
  }
  return sums;
}

/** Largest absolute component over several vectors (a shared drawing scale). */
export function maxAbsComponent(vectors: readonly Float32Array[]): number {
  let max = 0;
  for (const v of vectors) {
    for (let i = 0; i < v.length; i++) {
      const a = Math.abs(v[i] ?? 0);
      if (a > max) {
        max = a;
      }
    }
  }
  return max;
}

/** Which prompt positions a list with limited rows shows, and what it leaves out. */
export type TokenWindow = {
  /** Positions shown, oldest first: always the most recent ones. */
  readonly indices: readonly number[];
  /** Number of earlier positions not shown. */
  readonly hiddenCount: number;
  /** Total attention weight held by the hidden positions. */
  readonly hiddenWeight: number;
};

/** The most recent **`rows`** positions of an attention row, plus a summary of the rest. */
export function visibleTokenWindow(weights: Float32Array, rows: number): TokenWindow {
  if (!Number.isInteger(rows) || rows <= 0) {
    throw new Error(`visibleTokenWindow: rows must be a positive integer (got ${String(rows)})`);
  }
  const first = Math.max(0, weights.length - rows);
  let hiddenWeight = 0;
  for (let i = 0; i < first; i++) {
    hiddenWeight += weights[i] ?? 0;
  }
  const indices: number[] = [];
  for (let i = first; i < weights.length; i++) {
    indices.push(i);
  }
  return { indices, hiddenCount: first, hiddenWeight };
}
