import { backward } from '@llm-transformer-diy/autograd';
import { GpuContext, KernelRegistry, probeWebGpu, registerCoreKernels } from '@llm-transformer-diy/gpu';
import { createIndices, destroy, toArray, type TensorRuntime } from '@llm-transformer-diy/tensor';
import { beforeAll, describe, expect, it } from 'vitest';

import type { GptConfig } from './config';
import { collectGptParameters, createGptModel, forwardGpt } from './gpt';

const webgpuAvailable = await probeWebGpu();

const tinyCfg: GptConfig = {
  vocabSize: 8,
  maxSeqLen: 8,
  dModel: 32,
  nHeads: 4,
  dFf: 64,
  nLayers: 1,
};

describe.skipIf(!webgpuAvailable)('GPT forward', () => {
  let rt: TensorRuntime;

  beforeAll(async () => {
    const ctx = await GpuContext.create();
    const registry = new KernelRegistry();
    registerCoreKernels(ctx.device, registry);
    rt = { ctx, registry };
  });

  it('produces logits [seq, vocab] and runs backward', async () => {
    const model = createGptModel(rt, tinyCfg);
    const T = 5;
    const tok = createIndices(rt, Uint32Array.from([0, 1, 2, 3, 4]), [T]);

    const logits = forwardGpt(rt, model, tok);
    expect(logits.tensor.shape).toEqual([T, tinyCfg.vocabSize]);

    backward(rt, logits);

    const params = collectGptParameters(model);
    for (const p of params) {
      expect(p.grad).not.toBeNull();
    }

    const outHost = await toArray(rt, logits.tensor);
    expect(outHost.length).toBe(T * tinyCfg.vocabSize);

    if (logits.grad) {
      destroy(logits.grad);
    }
    destroy(tok);
    destroy(logits.tensor);
    for (const p of params) {
      destroy(p.tensor);
      if (p.grad) {
        destroy(p.grad);
      }
    }
    for (const hv of model.headExpand) {
      destroy(hv.tensor);
    }
  });
});
