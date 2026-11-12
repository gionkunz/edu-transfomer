import {
  GpuContext,
  KernelRegistry,
  probeWebGpu,
  registerCoreKernels,
} from '@llm-transformer-diy/gpu';
import { createIndices, toArray, type TensorRuntime } from '@llm-transformer-diy/tensor';
import { beforeAll, describe, expect, it } from 'vitest';

import type { GptConfig } from './config';
import { createGptModel, type GptModel } from './gpt';
import {
  createGptKvCache,
  destroyGptKvCache,
  disposeGptKvForwardGraph,
  forwardGptDecodeStep,
  forwardGptPrefill,
  readKvCacheHead,
  type GptDecodeStepResult,
  type GptKVCache,
} from './gpt-kv';

const webgpuAvailable = await probeWebGpu();

const cfg: GptConfig = {
  vocabSize: 13,
  maxSeqLen: 16,
  dModel: 16,
  nHeads: 2,
  dFf: 32,
  nLayers: 2,
};
const dK = cfg.dModel / cfg.nHeads;
const PREFIX = [3, 7, 2];
const LAST = 5;
const ROWS = PREFIX.length + 1;

function softmax(row: Float32Array): number[] {
  let max = -Infinity;
  for (const v of row) max = Math.max(max, v);
  const e = Array.from(row, (v) => Math.exp(v - max));
  const sum = e.reduce((a, b) => a + b, 0);
  return e.map((v) => v / sum);
}

describe.skipIf(!webgpuAvailable)('decode lesson trace', () => {
  let rt: TensorRuntime;
  let model: GptModel;

  beforeAll(async () => {
    const ctx = await GpuContext.create();
    const registry = new KernelRegistry();
    registerCoreKernels(ctx.device, registry);
    rt = { ctx, registry };
    model = createGptModel(rt, cfg);
  });

  /** Prefill `PREFIX`, decode `LAST`; returns the result with logits already read. */
  async function run(
    lessonTrace: boolean,
  ): Promise<{ result: GptDecodeStepResult; logits: Float32Array; cache: GptKVCache }> {
    const cache = createGptKvCache(rt, model);
    const tok = createIndices(rt, Uint32Array.from(PREFIX), [PREFIX.length]);
    const pre = forwardGptPrefill(rt, model, tok, cache);
    disposeGptKvForwardGraph(pre, model, cache);
    const result = await forwardGptDecodeStep(rt, model, LAST, cache, { lessonTrace });
    const logits = await toArray(rt, result.logits.tensor);
    disposeGptKvForwardGraph(result.logits, model, cache);
    return { result, logits, cache };
  }

  it('returns the four lesson captures with the documented shapes, plus the flow trace', async () => {
    const { result, cache } = await run(true);
    const heads = cfg.nLayers * cfg.nHeads;
    expect(result.inputVectors?.length).toBe(2 * cfg.dModel);
    expect(result.allQueries?.length).toBe(heads * dK);
    expect(result.allScores?.length).toBe(heads * ROWS);
    expect(result.allMixed?.length).toBe(heads * dK);
    expect(result.allAttention?.length).toBe(heads * ROWS);
    expect(result.layerLogits?.length).toBe(cfg.nLayers * cfg.vocabSize);
    expect(result.blockVectors?.length).toBe(3 * cfg.nLayers * cfg.dModel);
    destroyGptKvCache(cache);
  });

  it('omits the lesson captures when not requested', async () => {
    const { result, cache } = await run(false);
    expect(result.inputVectors).toBeUndefined();
    expect(result.allQueries).toBeUndefined();
    expect(result.allScores).toBeUndefined();
    expect(result.allMixed).toBeUndefined();
    expect(result.allAttention).toBeUndefined();
    destroyGptKvCache(cache);
  });

  it('does not change the logits', async () => {
    const plain = await run(false);
    const traced = await run(true);
    expect(traced.logits.length).toBe(plain.logits.length);
    for (let i = 0; i < plain.logits.length; i++) {
      expect(traced.logits[i]).toBeCloseTo(plain.logits[i]!, 5);
    }
    destroyGptKvCache(plain.cache);
    destroyGptKvCache(traced.cache);
  });

  it('readKvCacheHead returns one row per cached position and rejects bad indices', async () => {
    const { cache } = await run(true);
    const kv = await readKvCacheHead(rt, cache, 1, 0);
    expect(cache.seqLen).toBe(ROWS);
    expect(kv.dK).toBe(dK);
    expect(kv.keys.length).toBe(ROWS * dK);
    expect(kv.values.length).toBe(ROWS * dK);
    await expect(readKvCacheHead(rt, cache, 0, cfg.nHeads)).rejects.toThrow(/headIndex/);
    await expect(readKvCacheHead(rt, cache, cfg.nLayers, 0)).rejects.toThrow(/layerIndex/);
    destroyGptKvCache(cache);
  });

  it('scores are q·k/√d_k, weights are softmax(scores), mixed is weights·values', async () => {
    const { result, cache } = await run(true);
    const queries = result.allQueries!;
    const scores = result.allScores!;
    const weights = result.allAttention!;
    const mixed = result.allMixed!;
    for (let l = 0; l < cfg.nLayers; l++) {
      for (let h = 0; h < cfg.nHeads; h++) {
        const row = l * cfg.nHeads + h;
        const { keys, values } = await readKvCacheHead(rt, cache, l, h);
        const q = queries.subarray(row * dK, (row + 1) * dK);
        const sc = scores.subarray(row * ROWS, (row + 1) * ROWS);
        const w = weights.subarray(row * ROWS, (row + 1) * ROWS);
        for (let t = 0; t < ROWS; t++) {
          let dot = 0;
          for (let d = 0; d < dK; d++) dot += q[d]! * keys[t * dK + d]!;
          expect(sc[t]).toBeCloseTo(dot / Math.sqrt(dK), 4);
        }
        const expectedW = softmax(sc);
        for (let t = 0; t < ROWS; t++) {
          expect(w[t]).toBeCloseTo(expectedW[t]!, 4);
        }
        for (let d = 0; d < dK; d++) {
          let sum = 0;
          for (let t = 0; t < ROWS; t++) sum += w[t]! * values[t * dK + d]!;
          expect(mixed[row * dK + d]).toBeCloseTo(sum, 4);
        }
      }
    }
    destroyGptKvCache(cache);
  });

  it('input vectors add up to the stream entering the first block', async () => {
    const { result, cache } = await run(true);
    const input = result.inputVectors!;
    const stream = result.blockVectors!.subarray(0, cfg.dModel);
    for (let d = 0; d < cfg.dModel; d++) {
      expect(stream[d]).toBeCloseTo(input[d]! + input[cfg.dModel + d]!, 5);
    }
    destroyGptKvCache(cache);
  });

  it('leaves no buffers checked out once the graph and cache are released', async () => {
    const liveBuffers = (): number => {
      const st = rt.ctx.bufferPool.stats();
      return st.acquireHits + st.acquireMisses - st.releases - st.overflowDestroys;
    };
    const before = liveBuffers();
    const traced = await run(true);
    await readKvCacheHead(rt, traced.cache, 0, 0);
    destroyGptKvCache(traced.cache);
    expect(liveBuffers() - before).toBe(0);
  });
});
