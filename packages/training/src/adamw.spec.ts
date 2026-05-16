import {
  createParameter,
  createParameterRegistry,
  createParameterWithMaster,
  registerParameter,
} from '@llm-transformer-diy/autograd';
import {
  GpuContext,
  KernelRegistry,
  probeWebGpu,
  registerCoreKernels,
  supportsDtype,
} from '@llm-transformer-diy/gpu';
import {
  cast,
  createFromArray,
  destroy,
  toArray,
  type TensorRuntime,
} from '@llm-transformer-diy/tensor';
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
    expect(out[0]).not.toBeCloseTo(w0[0], 6);
    destroy(w.tensor);
    opt.dispose();
  });

  it('bf16 compute + f32 master: moments stay f32; compute weights match cast-down master after step', async () => {
    if (!supportsDtype(rt.ctx, 'bf16')) {
      return;
    }
    const reg = createParameterRegistry();
    const w0 = new Float32Array([1, 0, -1]);
    const w = createParameterWithMaster(rt, w0, [3], 'bf16');
    registerParameter(reg, w);
    w.grad = createFromArray(rt, new Float32Array([1, 1, 1]), [3], 'bf16');

    const opt = new AdamW(rt, reg, {
      lr: 0.05,
      beta1: 0.9,
      beta2: 0.99,
      eps: 1e-8,
      weightDecay: 0,
    });
    opt.step();
    expect(w.grad).toBeNull();
    const moments = opt as unknown as { m: Map<typeof w, { dtype: string }> };
    expect(moments.m.get(w)?.dtype).toBe('f32');

    const master = w.masterTensor;
    if (master === null) throw new Error('expected masterTensor for bf16 parameter');
    const castRef = cast(rt, master, 'bf16');
    try {
      const got = await toArray(rt, w.tensor);
      const exp = await toArray(rt, castRef);
      for (let i = 0; i < 3; i++) {
        expect(got[i]).toBeCloseTo(exp[i] ?? 0, 3);
      }
    } finally {
      destroy(castRef);
    }

    destroy(w.tensor);
    destroy(master);
    opt.dispose();
  });
});
