/* eslint-disable @typescript-eslint/no-non-null-assertion --
 * Single-parameter Adam smoke test uses a defined gradient buffer.
 */
import { createParameter, createParameterRegistry, registerParameter } from '@llm-transformer-diy/autograd';
import { GpuContext, KernelRegistry, probeWebGpu, registerCoreKernels } from '@llm-transformer-diy/gpu';
import { createFromArray, destroy, toArray, type TensorRuntime } from '@llm-transformer-diy/tensor';
import { beforeAll, describe, expect, it } from 'vitest';

import { AdamW } from './adamw';

const webgpuAvailable = await probeWebGpu();

describe.skipIf(!webgpuAvailable)('AdamW', () => {
  let rt: TensorRuntime;

  beforeAll(async () => {
    const ctx = await GpuContext.create();
    const registry = new KernelRegistry();
    registerCoreKernels(ctx.device, registry);
    rt = { ctx, registry };
  });

  it('applies an update and clears grad', async () => {
    const reg = createParameterRegistry();
    const w0 = new Float32Array([1, 0, -1]);
    const w = createParameter(createFromArray(rt, new Float32Array(w0), [3]));
    registerParameter(reg, w);
    w.grad = createFromArray(rt, new Float32Array([1, 1, 1]), [3]);

    const opt = new AdamW(rt, reg, {
      lr: 0.05,
      beta1: 0.9,
      beta2: 0.99,
      eps: 1e-8,
      weightDecay: 0,
    });
    opt.step();

    expect(w.grad).toBeNull();
    const out = await toArray(rt, w.tensor);
    expect(out[0]!).not.toBeCloseTo(w0[0]!, 6);
    destroy(w.tensor);
    opt.dispose();
  });
});
