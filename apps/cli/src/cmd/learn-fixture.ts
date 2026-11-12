import {
  buildFlowTrace,
  explainSampling,
  type LessonHead,
  type LessonHeadKv,
  type LessonTrace,
} from '@llm-transformer-diy/inference';

import { LESSON_CHAPTERS } from './learn-chapters';
import { SAMPLING_TOP_N } from './learn-chapter-sampling';
import { buildLessonMeta, type LessonContext, type LessonMeta } from './learn-lesson';
import { initialLessonState, type LessonState } from './learn-state';

/**
 * A made-up but self-consistent lesson trace for tests of the lesson screens:
 * no GPU, no checkpoint. Weights are the softmax of the scores, the last lens
 * row is the output, as with a real forward.
 */

export const FIXTURE = { nLayers: 2, nHeads: 2, dModel: 64, dK: 32, vocabSize: 40 } as const;

/** Deterministic pseudo-random numbers in [-1, 1). */
function noise(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (Math.imul(s, 1664525) + 1013904223) >>> 0;
    return s / 0x80000000 - 1;
  };
}

function vector(n: number, next: () => number, scale = 1): Float32Array {
  return Float32Array.from({ length: n }, () => next() * scale);
}

function softmax(v: Float32Array): Float32Array {
  let max = -Infinity;
  for (const x of v) max = Math.max(max, x);
  const e = Float32Array.from(v, (x) => Math.exp(x - max));
  let sum = 0;
  for (const x of e) sum += x;
  return e.map((x) => x / sum);
}

export function fixtureTrace(promptLength: number): { trace: LessonTrace; kv: LessonHeadKv } {
  const { nLayers, nHeads, dModel, dK, vocabSize } = FIXTURE;
  const next = noise(7);
  const heads: LessonHead[] = [];
  const allAttention = new Float32Array(nLayers * nHeads * promptLength);
  for (let l = 0; l < nLayers; l++) {
    for (let h = 0; h < nHeads; h++) {
      const scores = vector(promptLength, next, 3);
      const weights = softmax(scores);
      allAttention.set(weights, (l * nHeads + h) * promptLength);
      let peakIndex = 0;
      let entropyBits = 0;
      weights.forEach((w, i) => {
        if (w > (weights[peakIndex] ?? 0)) peakIndex = i;
        if (w > 0) entropyBits -= w * Math.log2(w);
      });
      heads.push({
        layerIndex: l,
        headIndex: h,
        query: vector(dK, next),
        scores,
        weights,
        mixed: vector(dK, next),
        peakIndex,
        entropyBits,
      });
    }
  }
  const rawLogits = vector(vocabSize, next, 4);
  const layerLogits = new Float32Array(nLayers * vocabSize);
  for (let l = 0; l < nLayers - 1; l++) layerLogits.set(vector(vocabSize, next, 2), l * vocabSize);
  layerLogits.set(rawLogits, (nLayers - 1) * vocabSize);
  const blockVectors = vector(3 * nLayers * dModel, next);

  const trace: LessonTrace = {
    promptLength,
    nLayers,
    nHeads,
    dModel,
    dK,
    rawLogits,
    tokenVector: vector(dModel, next),
    positionVector: vector(dModel, next),
    head: (l, h) => {
      const found = heads[l * nHeads + h];
      if (found === undefined) throw new Error('fixture: no such head');
      return found;
    },
    flowFor: (sampledTokenId) =>
      buildFlowTrace({ layerLogits, allAttention, blockVectors, nLayers, nHeads, sampledTokenId }),
  };
  const kv: LessonHeadKv = {
    keys: Array.from({ length: promptLength }, () => vector(dK, next)),
    values: Array.from({ length: promptLength }, () => vector(dK, next)),
  };
  return { trace, kv };
}

export function fixtureMeta(promptLength: number, noAnimation = false): LessonMeta {
  return buildLessonMeta(
    LESSON_CHAPTERS,
    { nLayers: FIXTURE.nLayers, promptLength },
    { nHeads: FIXTURE.nHeads, cliTopK: 40, noAnimation },
  );
}

/** A lesson context on the fixture trace, as the view builds it for an 80×24 terminal. */
export function fixtureContext(
  promptLength: number,
  overrides: Partial<LessonState> = {},
  size: { rows: number; cols: number } = { rows: 15, cols: 80 },
): { ctx: LessonContext; meta: LessonMeta } {
  const { trace, kv } = fixtureTrace(promptLength);
  const meta = fixtureMeta(promptLength, true);
  const top = explainSampling(trace.rawLogits, { temperature: 0.8, topK: 40 }, 1).rows[0];
  const state: LessonState = {
    ...initialLessonState(meta, {
      layerIndex: FIXTURE.nLayers - 1,
      headIndex: 0,
      temperature: 0.8,
      topK: 40,
      drawnTokenId: top?.tokenId ?? 0,
    }),
    ...overrides,
  };
  return {
    meta,
    ctx: {
      trace,
      kv,
      tokenLabels: Array.from({ length: promptLength }, (_, i) => `tok${String(i)}`),
      labelForId: (id) => `v${String(id)}`,
      vocabSize: FIXTURE.vocabSize,
      state,
      explain: (temperature, topK) =>
        explainSampling(trace.rawLogits, { temperature, topK }, SAMPLING_TOP_N),
      rows: size.rows,
      cols: size.cols,
    },
  };
}
