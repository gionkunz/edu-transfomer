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
});
