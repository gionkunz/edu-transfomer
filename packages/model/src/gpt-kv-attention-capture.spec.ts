import {
  GpuContext,
  KernelRegistry,
  probeWebGpu,
  registerCoreKernels,
} from '@llm-transformer-diy/gpu';
import { createIndices, destroy, type TensorRuntime } from '@llm-transformer-diy/tensor';
import { beforeAll, describe, expect, it } from 'vitest';

import type { GptConfig } from './config';
import { createGptModel, type GptModel } from './gpt';
import {
  createGptKvCache,
  destroyGptKvCache,
  forwardGptDecodeStep,
  forwardGptPrefill,
} from './gpt-kv';

const webgpuAvailable = await probeWebGpu();

const cfg: GptConfig = {
  vocabSize: 11,
  maxSeqLen: 16,
  dModel: 32,
  nHeads: 4,
  dFf: 64,
  nLayers: 2,
};

describe.skipIf(!webgpuAvailable)('decode attention capture', () => {
  let rt: TensorRuntime;
  let model: GptModel;

  beforeAll(async () => {
    const ctx = await GpuContext.create();
    const registry = new KernelRegistry();
    registerCoreKernels(ctx.device, registry);
    rt = { ctx, registry };
    model = createGptModel(rt, cfg);
  });

  it('returns stacked [nHeads, prefixRows] weights with per-head row sums ≈ 1', async () => {
    const cache = createGptKvCache(rt, model);
    const tokPre = createIndices(rt, new Uint32Array([3, 7]), [2]);
    forwardGptPrefill(rt, model, tokPre, cache);
    destroy(tokPre);

    const prefixRowsBeforeDecode = cache.seqLen + 1;
    expect(prefixRowsBeforeDecode).toBe(3);

    const { logits, attention } = await forwardGptDecodeStep(rt, model, 2, cache, {
      attentionCapture: { layerIndex: 1 },
    });

    expect(logits.tensor.shape).toEqual([1, cfg.vocabSize]);
    expect(attention).toBeDefined();
    expect(attention!.length).toBe(cfg.nHeads * prefixRowsBeforeDecode);

    for (let h = 0; h < cfg.nHeads; h++) {
      const rowStart = h * prefixRowsBeforeDecode;
      const row = attention!.subarray(rowStart, rowStart + prefixRowsBeforeDecode);
      let sum = 0;
      for (let i = 0; i < row.length; i++) {
        expect(row[i]).toBeGreaterThanOrEqual(0);
        expect(row[i]).toBeLessThanOrEqual(1);
        sum += row[i]!;
      }
      expect(sum).toBeCloseTo(1, 3);
    }

    expect(cache.seqLen).toBe(3);
    destroyGptKvCache(cache);
  });

  it('omits attention when capture is not requested (decode-only path)', async () => {
    const cache = createGptKvCache(rt, model);
    const tokPre = createIndices(rt, new Uint32Array([1, 2]), [2]);
    forwardGptPrefill(rt, model, tokPre, cache);
    destroy(tokPre);

    const { attention } = await forwardGptDecodeStep(rt, model, 3, cache);
    expect(attention).toBeUndefined();

    destroyGptKvCache(cache);
  });

  it('does not capture during prefill', () => {
    const cache = createGptKvCache(rt, model);
    const tokPre = createIndices(rt, new Uint32Array([4, 5, 6]), [3]);
    const logits = forwardGptPrefill(rt, model, tokPre, cache);
    expect(logits.tensor.shape).toEqual([3, cfg.vocabSize]);
    expect(cache.seqLen).toBe(3);
    destroy(tokPre);
    destroyGptKvCache(cache);
  });
});
