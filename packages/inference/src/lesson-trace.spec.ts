import {
  GpuContext,
  KernelRegistry,
  probeWebGpu,
  registerCoreKernels,
} from '@llm-transformer-diy/gpu';
import {
  createGptKvCache,
  createGptModel,
  destroyGptKvCache,
  type GptConfig,
  type GptModel,
} from '@llm-transformer-diy/model';
import type { TensorRuntime } from '@llm-transformer-diy/tensor';
import { beforeAll, describe, expect, it } from 'vitest';

import { generateAutoregressive } from './generate';
import {
  bucketVector,
  maxAbsComponent,
  mostFocusedHead,
  readLessonHeadKv,
  runLessonForward,
  visibleTokenWindow,
} from './lesson-trace';
import { explainSampling } from './sampling';

const webgpuAvailable = await probeWebGpu();

describe('lesson display helpers', () => {
  it('bucketVector keeps short vectors and averages long ones', () => {
    expect(Array.from(bucketVector(new Float32Array([1, -2, 3]), 8))).toEqual([1, -2, 3]);
    expect(Array.from(bucketVector(new Float32Array([1, 3, -2, -4]), 2))).toEqual([2, -3]);
    expect(() => bucketVector(new Float32Array(3), 0)).toThrow();
  });

  it('maxAbsComponent spans all vectors', () => {
    expect(maxAbsComponent([new Float32Array([0.5, -3]), new Float32Array([2])])).toBe(3);
    expect(maxAbsComponent([])).toBe(0);
  });

  it('visibleTokenWindow shows the most recent positions and totals the rest', () => {
    const w = new Float32Array([0.1, 0.2, 0.3, 0.4]);
    const all = visibleTokenWindow(w, 9);
    expect(all.indices).toEqual([0, 1, 2, 3]);
    expect(all.hiddenCount).toBe(0);
    expect(all.hiddenWeight).toBe(0);

    const cut = visibleTokenWindow(w, 2);
    expect(cut.indices).toEqual([2, 3]);
    expect(cut.hiddenCount).toBe(2);
    const shown = cut.indices.reduce((a, i) => a + w[i]!, 0);
    expect(shown + cut.hiddenWeight).toBeCloseTo(1, 6);
  });
});

describe.skipIf(!webgpuAvailable)('runLessonForward', () => {
  let rt: TensorRuntime;
  let model: GptModel;
  const cfg: GptConfig = {
    vocabSize: 13,
    maxSeqLen: 16,
    dModel: 16,
    nHeads: 2,
    dFf: 32,
    nLayers: 2,
  };
  const dK = cfg.dModel / cfg.nHeads;

  beforeAll(async () => {
    const ctx = await GpuContext.create();
    const registry = new KernelRegistry();
    registerCoreKernels(ctx.device, registry);
    rt = { ctx, registry };
    model = createGptModel(rt, cfg);
  });

  for (const prompt of [[4], [3, 7, 2, 9]]) {
    it(`captures the forward of the last of ${String(prompt.length)} prompt token(s)`, async () => {
      const { trace, cache } = await runLessonForward(rt, model, Uint32Array.from(prompt));
      try {
        expect(cache.seqLen).toBe(prompt.length);
        expect(trace.promptLength).toBe(prompt.length);
        expect(trace.rawLogits.length).toBe(cfg.vocabSize);
        expect(trace.tokenVector.length).toBe(cfg.dModel);
        expect(trace.positionVector.length).toBe(cfg.dModel);
        for (let l = 0; l < cfg.nLayers; l++) {
          for (let h = 0; h < cfg.nHeads; h++) {
            const head = trace.head(l, h);
            expect(head.query.length).toBe(dK);
            expect(head.mixed.length).toBe(dK);
            expect(head.scores.length).toBe(prompt.length);
            let sum = 0;
            let peak = 0;
            for (let i = 0; i < head.weights.length; i++) {
              sum += head.weights[i]!;
              if (head.weights[i]! > head.weights[peak]!) peak = i;
            }
            expect(sum).toBeCloseTo(1, 4);
            expect(head.peakIndex).toBe(peak);
          }
        }
        expect(() => trace.head(cfg.nLayers, 0)).toThrow();

        const kv = await readLessonHeadKv(rt, cache, 1, 1);
        expect(kv.keys.length).toBe(prompt.length);
        expect(kv.values.length).toBe(prompt.length);
        expect(kv.keys[0]!.length).toBe(dK);

        const focused = mostFocusedHead(trace, 1);
        for (let h = 0; h < cfg.nHeads; h++) {
          expect(trace.head(1, focused).entropyBits).toBeLessThanOrEqual(
            trace.head(1, h).entropyBits,
          );
        }
      } finally {
        destroyGptKvCache(cache);
      }
    });
  }

  it('the last block of the flow trace is the raw output distribution', async () => {
    const { trace, cache } = await runLessonForward(rt, model, new Uint32Array([3, 7, 2, 9]));
    destroyGptKvCache(cache);
    const raw = explainSampling(trace.rawLogits, { temperature: 1, topK: 0 }, 3);
    const top = raw.rows[0]!;
    const flow = trace.flowFor(raw.rows[1]!.tokenId);
    expect(flow.layers.length).toBe(cfg.nLayers);
    expect(flow.prefixRows).toBe(4);
    const last = flow.layers[cfg.nLayers - 1]!;
    expect(last.topTokenId).toBe(top.tokenId);
    expect(last.topProbability).toBeCloseTo(top.rawProbability, 4);
    expect(last.sampledProbability).toBeCloseTo(raw.rows[1]!.rawProbability, 4);
  });

  it('predicts from the same distribution generation samples its first token from', async () => {
    const prompt = new Uint32Array([3, 7, 2, 9]);
    const { trace, cache } = await runLessonForward(rt, model, prompt);
    destroyGptKvCache(cache);
    const sample = { temperature: 0.8, topK: 5 };
    const expected = explainSampling(trace.rawLogits, sample, 3).final;

    let seen: Float32Array | undefined;
    const genCache = createGptKvCache(rt, model);
    try {
      await generateAutoregressive(rt, model, prompt, genCache, {
        maxNewTokens: 1,
        sample,
        onStep: ({ probabilities }) => {
          seen = probabilities;
        },
      });
    } finally {
      destroyGptKvCache(genCache);
    }
    expect(seen).toBeDefined();
    for (let i = 0; i < expected.length; i++) {
      expect(expected[i]).toBeCloseTo(seen![i]!, 4);
    }
  });

  it('rejects an empty or over-long prompt and leaves no buffers checked out', async () => {
    const liveBuffers = (): number => {
      const st = rt.ctx.bufferPool.stats();
      return st.acquireHits + st.acquireMisses - st.releases - st.overflowDestroys;
    };
    const before = liveBuffers();
    await expect(runLessonForward(rt, model, new Uint32Array(0))).rejects.toThrow(/non-empty/);
    await expect(
      runLessonForward(rt, model, new Uint32Array(cfg.maxSeqLen + 1).fill(1)),
    ).rejects.toThrow(/context/);
    const { cache } = await runLessonForward(rt, model, new Uint32Array([1, 2, 3]));
    await readLessonHeadKv(rt, cache, 0, 0);
    destroyGptKvCache(cache);
    expect(liveBuffers() - before).toBe(0);
  });
});
