/// <reference types="@webgpu/types" />

import {
  GpuContext,
  KernelRegistry,
  probeWebGpu,
  registerCoreKernels,
} from '@llm-transformer-diy/gpu';
import { createFromArray, createIndices, destroy, toArray } from '@llm-transformer-diy/tensor';
import type { Tensor, TensorRuntime } from '@llm-transformer-diy/tensor';
import { afterEach, beforeAll, describe, expect, it } from 'vitest';

import { backward } from './backward';
import {
  clipGradNorm,
  createParameterRegistry,
  disposeAutogradGraphExceptParameters,
  registerParameter,
} from './parameter-registry';
import { crossEntropyMeanFromLogits, reshape, sumFlat } from './ops';
import { createLeafVariable, createParameter } from './variable';

const webgpuAvailable = await probeWebGpu();

describe.skipIf(!webgpuAvailable)('disposeAutogradGraphExceptParameters (GPU)', () => {
  let rt: TensorRuntime;
  const tracked: Tensor[] = [];

  function trackTensor(t: Tensor): Tensor {
    tracked.push(t);
    return t;
  }

  beforeAll(async () => {
    const ctx = await GpuContext.create();
    const registry = new KernelRegistry();
    registerCoreKernels(ctx.device, registry);
    rt = { ctx, registry };
  });

  afterEach(() => {
    while (tracked.length > 0) {
      const t = tracked.pop();
      if (t) {
        try {
          destroy(t);
        } catch {
          //
        }
      }
    }
  });

  it('keeps parameter .tensor / .grad readable after dispose + clip', async () => {
    const rows = 2;
    const cols = 4;
    const logitsData = new Float32Array([0.1, -0.4, 0.7, 0.05, -0.2, 1.3, -0.1, 0.4]);
    const lt = trackTensor(createFromArray(rt, logitsData, [rows, cols]));
    const reg = createParameterRegistry();
    const L = createParameter(lt);
    registerParameter(reg, L);

    const tt = Uint32Array.from([1, 3]);
    const targTen = trackTensor(createIndices(rt, tt, [rows]));
    const targLeaf = createLeafVariable(targTen, false);

    const loss = crossEntropyMeanFromLogits(rt, L, targLeaf);
    backward(rt, loss);
    disposeAutogradGraphExceptParameters(loss, reg);
    await clipGradNorm(rt, reg, 1e6);

    expect(L.grad).not.toBeNull();
    await expect(toArray(rt, L.grad as Tensor)).resolves.toBeDefined();
    await expect(toArray(rt, L.tensor)).resolves.toEqual(logitsData);

    if (L.grad !== null) destroy(L.grad);
    L.grad = null;
  });

  it('does not destroy a parameter buffer shared via reshape views', async () => {
    const data = new Float32Array([1, 2, 3, 4, 5, 6, 7, 8]);
    const wTen = trackTensor(createFromArray(rt, data, [2, 2, 2]));
    const reg = createParameterRegistry();
    const w = createParameter(wTen);
    registerParameter(reg, w);

    const r1 = reshape(w, [4, 2]);
    const flat = reshape(r1, [8]);
    const pooled = sumFlat(rt, flat);
    backward(rt, pooled);
    disposeAutogradGraphExceptParameters(pooled, reg);

    await expect(toArray(rt, w.tensor)).resolves.toEqual(data);
    expect(w.grad).not.toBeNull();
    await expect(toArray(rt, w.grad as Tensor)).resolves.toBeDefined();

    if (w.grad !== null) destroy(w.grad);
    w.grad = null;
  });
});
