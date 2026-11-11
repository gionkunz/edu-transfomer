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

import { summarizeLayerLens, type FlowTrace } from './flow-trace';
import { generateAutoregressive, type DecodeAttentionPayload } from './generate';

const webgpuAvailable = await probeWebGpu();

describe.skipIf(!webgpuAvailable)('generateAutoregressive: onStep host hook', () => {
  let rt: TensorRuntime;
  let model: GptModel;
  const cfg: GptConfig = {
    vocabSize: 11,
    maxSeqLen: 16,
    dModel: 32,
    nHeads: 4,
    dFf: 64,
    nLayers: 1,
  };

  beforeAll(async () => {
    const ctx = await GpuContext.create();
    const registry = new KernelRegistry();
    registerCoreKernels(ctx.device, registry);
    rt = { ctx, registry };
    model = createGptModel(rt, cfg);
  });

  it('fires onStep after onSampledToken with matching index and a normalized probability vector', async () => {
    const promptIds = new Uint32Array([3, 7, 2]);
    const events: Array<
      | { kind: 'sampled'; id: number; idx: number }
      | { kind: 'step'; id: number; idx: number; probs: Float32Array }
    > = [];
    const cache = createGptKvCache(rt, model);
    try {
      await generateAutoregressive(rt, model, promptIds, cache, {
        maxNewTokens: 3,
        sample: { temperature: 0, topK: 0 },
        onSampledToken: (id, idx) => {
          events.push({ kind: 'sampled', id, idx });
        },
        onStep: ({ tokenId, newTokenIndex, probabilities }) => {
          events.push({ kind: 'step', id: tokenId, idx: newTokenIndex, probs: probabilities });
        },
      });
    } finally {
      destroyGptKvCache(cache);
    }

    expect(events.length).toBe(6);
    for (let n = 0; n < 3; n++) {
      const a = events[2 * n];
      const b = events[2 * n + 1];
      expect(a.kind).toBe('sampled');
      expect(b.kind).toBe('step');
      expect(a.idx).toBe(n);
      expect(b.idx).toBe(n);
      expect(a.id).toBe(b.id);
      if (b.kind === 'step') {
        expect(b.probs.length).toBe(cfg.vocabSize);
        let s = 0;
        for (let i = 0; i < b.probs.length; i++) s += b.probs[i];
        expect(s).toBeCloseTo(1, 6);
        expect(b.probs[b.id]).toBe(1);
      }
    }
  });

  it('omits attention on the first sampled token and includes it from the second onward', async () => {
    const promptIds = new Uint32Array([3, 7, 2]);
    const cache = createGptKvCache(rt, model);
    const attentions: Array<DecodeAttentionPayload | undefined> = [];
    try {
      await generateAutoregressive(rt, model, promptIds, cache, {
        maxNewTokens: 3,
        sample: { temperature: 0, topK: 0 },
        attentionLayerIndex: 0,
        onStep: ({ attention }) => {
          attentions.push(attention);
        },
      });
    } finally {
      destroyGptKvCache(cache);
    }

    expect(attentions.length).toBe(3);
    expect(attentions[0]).toBeUndefined();
    expect(attentions[1]).toBeDefined();
    expect(attentions[2]).toBeDefined();

    const attn = attentions[1];
    if (!attn) {
      throw new Error('expected attention on second sampled token');
    }
    expect(attn.layerIndex).toBe(0);
    expect(attn.nHeads).toBe(cfg.nHeads);
    expect(attn.prefixRows).toBe(promptIds.length + 1);
    expect(attn.weights.length).toBe(cfg.nHeads * attn.prefixRows);

    for (let h = 0; h < cfg.nHeads; h++) {
      const row = attn.weights.subarray(h * attn.prefixRows, (h + 1) * attn.prefixRows);
      let sum = 0;
      for (let i = 0; i < row.length; i++) sum += row[i];
      expect(sum).toBeCloseTo(1, 5);
    }
  });

  it('awaits an onStep promise before the next decode step', async () => {
    const promptIds = new Uint32Array([3, 7]);
    const log: string[] = [];
    let release: (() => void) | null = null;
    const cache = createGptKvCache(rt, model);
    try {
      const gen = generateAutoregressive(rt, model, promptIds, cache, {
        maxNewTokens: 2,
        sample: { temperature: 0, topK: 0 },
        onSampledToken: (_id, idx) => {
          log.push(`sampled:${String(idx)}`);
        },
        onStep: ({ newTokenIndex }) => {
          log.push(`stepStart:${String(newTokenIndex)}`);
          if (newTokenIndex === 0) {
            return new Promise<void>((resolve) => {
              release = (): void => {
                log.push(`stepResume:${String(newTokenIndex)}`);
                resolve();
              };
            });
          }
          log.push(`stepResume:${String(newTokenIndex)}`);
          return undefined;
        },
      });

      const tStart = Date.now();
      while (!log.includes('stepStart:0')) {
        if (Date.now() - tStart > 2000) {
          throw new Error('timed out waiting for stepStart:0');
        }
        await new Promise<void>((r) => setTimeout(r, 5));
      }
      await new Promise<void>((r) => setTimeout(r, 50));
      expect(log).toEqual(['sampled:0', 'stepStart:0']);
      release?.();
      await gen;
    } finally {
      destroyGptKvCache(cache);
    }

    expect(log).toEqual([
      'sampled:0',
      'stepStart:0',
      'stepResume:0',
      'sampled:1',
      'stepStart:1',
      'stepResume:1',
    ]);
  });

  it('flow trace: last block matches the sampling distribution; heads and updates are sane', async () => {
    const deep = createGptModel(rt, { ...cfg, nLayers: 3 });
    const cache = createGptKvCache(rt, deep);
    const seen: Array<{ flow: FlowTrace | undefined; probs: Float32Array; id: number }> = [];
    try {
      await generateAutoregressive(rt, deep, new Uint32Array([3, 7, 2]), cache, {
        maxNewTokens: 3,
        // temperature 1, no top-k: the sampler's distribution is the raw softmax.
        sample: { temperature: 1, topK: 0 },
        rng: () => 0.5,
        captureFlow: true,
        onStep: ({ flow, probabilities, tokenId }) => {
          seen.push({ flow, probs: probabilities, id: tokenId });
        },
      });
    } finally {
      destroyGptKvCache(cache);
    }
    expect(seen).toHaveLength(3);
    expect(seen[0]?.flow).toBeUndefined();
    for (const [n, step] of seen.slice(1).entries()) {
      expect(step.flow?.layers).toHaveLength(3);
      // prompt (3) + tokens already decoded before this forward
      expect(step.flow?.prefixRows).toBe(3 + n + 1);
      const last = step.flow?.layers[2];
      expect(last?.sampledProbability).toBeCloseTo(step.probs[step.id] ?? -1, 4);
      for (const layer of step.flow?.layers ?? []) {
        expect(layer.topProbability).toBeGreaterThan(0);
        expect(layer.sampledProbability).toBeLessThanOrEqual(layer.topProbability + 1e-6);
        expect(layer.heads).toHaveLength(cfg.nHeads);
        for (const head of layer.heads) {
          expect(head.peakStrength).toBeGreaterThan(0);
          expect(head.peakStrength).toBeLessThanOrEqual(1 + 1e-5);
        }
        expect(layer.attentionUpdate).toBeGreaterThan(0);
        expect(layer.feedForwardUpdate).toBeGreaterThan(0);
        expect(Number.isFinite(layer.attentionUpdate + layer.feedForwardUpdate)).toBe(true);
      }
    }
  });

  it("stops after the current token when onStep returns 'stop'", async () => {
    const cache = createGptKvCache(rt, model);
    let steps = 0;
    try {
      const out = await generateAutoregressive(rt, model, new Uint32Array([3, 7, 2]), cache, {
        maxNewTokens: 8,
        sample: { temperature: 0, topK: 0 },
        onStep: () => {
          steps += 1;
          return steps === 2 ? 'stop' : undefined;
        },
      });
      expect(steps).toBe(2);
      expect(out).toHaveLength(3 + 2);
      // Tokens before the stopping one were forwarded; the stopping one was not.
      expect(cache.seqLen).toBe(3 + 1);
    } finally {
      destroyGptKvCache(cache);
    }
  });

  it('releases each forward graph: no buffers stay checked out after a generation', async () => {
    // Buffers currently handed out by the pool and not yet returned.
    const liveBuffers = (): number => {
      const st = rt.ctx.bufferPool.stats();
      return st.acquireHits + st.acquireMisses - st.releases - st.overflowDestroys;
    };
    const run = async (maxNewTokens: number): Promise<number> => {
      const before = liveBuffers();
      const cache = createGptKvCache(rt, model);
      try {
        await generateAutoregressive(rt, model, new Uint32Array([3, 7, 2]), cache, {
          maxNewTokens,
          sample: { temperature: 0, topK: 0 },
          attentionLayerIndex: 0,
          captureFlow: true,
        });
      } finally {
        destroyGptKvCache(cache);
      }
      return liveBuffers() - before;
    };
    expect(await run(4)).toBe(0);
    expect(await run(12)).toBe(0);
  });
});

describe('summarizeLayerLens', () => {
  it('reports the top token and the sampled token probability per layer', () => {
    const logits = new Float32Array([0, Math.log(3), 0, /* layer 2 */ Math.log(2), 0, 0]);
    const { layers } = summarizeLayerLens(logits, 2, 0);
    expect(layers[0]).toMatchObject({ layerIndex: 0, topTokenId: 1 });
    expect(layers[0]?.topProbability).toBeCloseTo(0.6, 6);
    expect(layers[0]?.sampledProbability).toBeCloseTo(0.2, 6);
    expect(layers[1]).toMatchObject({ layerIndex: 1, topTokenId: 0 });
    expect(layers[1]?.sampledProbability).toBeCloseTo(0.5, 6);
  });
});
