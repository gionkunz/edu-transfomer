/* eslint-disable @typescript-eslint/no-non-null-assertion --
 * Grad tensors are guarded by assertions before dereferencing.
 */
import {
  cpuCrossEntropyLogitsGradFromProbs,
  cpuSoftmax,
  GpuContext,
  KernelRegistry,
  probeWebGpu,
  registerCoreKernels,
} from '@llm-transformer-diy/gpu';
import {
  createFromArray,
  createIndices,
  destroy,
  toArray,
  scatterEmbeddingAccum,
} from '@llm-transformer-diy/tensor';
import type { Tensor, TensorRuntime } from '@llm-transformer-diy/tensor';
import { afterEach, beforeAll, describe, expect, it } from 'vitest';

import { backward } from './backward';
import { crossEntropyMeanFromLogits, embedding as embeddingVar } from './ops';
import { createLeafVariable, createParameter } from './variable';

const webgpuAvailable = await probeWebGpu();

function expectCloseVec(a: Float32Array, e: Float32Array, places = 5): void {
  expect(a.length).toBe(e.length);
  for (let i = 0; i < a.length; i++) expect(a[i]).toBeCloseTo(e[i]!, places);
}

describe.skipIf(!webgpuAvailable)('autograd embedding + CE mean (GPU)', () => {
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

  it('embedding backward aggregates into weight grads like scatterEmbeddingAccum', async () => {
    const vocab = 4;
    const embed = 2;
    const tokens = 3;
    const wFlat = Float32Array.from([1, 2, 3, 4, 10, 20, -1, -2]);
    const wTen = trackTensor(createFromArray(rt, wFlat, [vocab, embed]));
    const w = createParameter(wTen);

    const idx = Uint32Array.from([2, 0, 2]);
    const ixTen = trackTensor(createIndices(rt, idx, [tokens]));
    const ixLeaf = createLeafVariable(ixTen, false);

    const y = embeddingVar(rt, w, ixLeaf);

    backward(rt, y);

    trackTensor(y.tensor);

    expect(w.grad).not.toBeNull();
    const gW = await toArray(rt, w.grad as Tensor);
    const onesTok = trackTensor(
      createFromArray(rt, new Float32Array(tokens * embed).fill(1), [tokens, embed]),
    );
    const ref = trackTensor(scatterEmbeddingAccum(rt, ixTen, onesTok, vocab));
    expectCloseVec(gW, await toArray(rt, ref));

    if (y.grad !== null) destroy(y.grad);
    if (w.grad !== null) destroy(w.grad);
    w.grad = null;
  });

  it('crossEntropyMeanFromLogits backward matches softmax CE analytic grad', async () => {
    const rows = 2;
    const cols = 4;
    const logitsData = new Float32Array([0.1, -0.4, 0.7, 0.05, -0.2, 1.3, -0.1, 0.4]);
    const lt = trackTensor(createFromArray(rt, logitsData, [rows, cols]));
    const L = createParameter(lt);

    const tt = Uint32Array.from([1, 3]);
    const targTen = trackTensor(createIndices(rt, tt, [rows]));
    const targLeaf = createLeafVariable(targTen, false);

    const loss = crossEntropyMeanFromLogits(rt, L, targLeaf);
    trackTensor(loss.tensor);

    backward(rt, loss);

    expect(L.grad).not.toBeNull();
    const probs = cpuSoftmax(logitsData, rows, cols);
    const analytic = cpuCrossEntropyLogitsGradFromProbs(
      probs,
      tt,
      new Float32Array(rows).fill(1 / rows),
      rows,
      cols,
    );
    expectCloseVec(await toArray(rt, L.grad as Tensor), analytic);

    if (loss.grad !== null) destroy(loss.grad);
    if (L.grad !== null) destroy(L.grad);
    L.grad = null;
  });
});
