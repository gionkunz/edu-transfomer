import {
  GpuContext,
  KernelRegistry,
  probeWebGpu,
  registerCoreKernels,
} from '@llm-transformer-diy/gpu';
import { createFromArray, destroy, toArray, type TensorRuntime } from '@llm-transformer-diy/tensor';
import { beforeAll, describe, expect, it } from 'vitest';

import { clipGradNorm, createParameterRegistry, registerParameter } from './parameter-registry';
import { createParameter } from './variable';

const webgpuAvailable = await probeWebGpu();

describe.skipIf(!webgpuAvailable)('clipGradNorm', () => {
  let rt: TensorRuntime;

  beforeAll(async () => {
    const ctx = await GpuContext.create();
    const registry = new KernelRegistry();
    registerCoreKernels(ctx.device, registry);
    rt = { ctx, registry };
  });

  it('scales gradients when total norm exceeds maxNorm', async () => {
    const reg = createParameterRegistry();
    const p = createParameter(createFromArray(rt, new Float32Array([0, 0]), [2]));
    registerParameter(reg, p);
    p.grad = createFromArray(rt, new Float32Array([3, 4]), [2]);

    await clipGradNorm(rt, reg, 1);

    expect(p.grad).not.toBeNull();
    const g = await toArray(rt, p.grad);
    expect(g[0]).toBeCloseTo(3 / 5);
    expect(g[1]).toBeCloseTo(4 / 5);

    destroy(p.grad);
    destroy(p.tensor);
  });

  it('is a no-op when norm is below maxNorm', async () => {
    const reg = createParameterRegistry();
    const p = createParameter(createFromArray(rt, new Float32Array([0]), [1]));
    registerParameter(reg, p);
    p.grad = createFromArray(rt, new Float32Array([0.3]), [1]);

    await clipGradNorm(rt, reg, 1);

    expect(p.grad).not.toBeNull();
    const g = await toArray(rt, p.grad);
    expect(g[0]).toBeCloseTo(0.3);

    destroy(p.grad);
    destroy(p.tensor);
  });
});
