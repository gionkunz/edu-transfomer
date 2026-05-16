import {
  collectGptParameters,
  createGptModel,
  forwardGpt,
  gptParameterTensorNames,
  type GptConfig,
  type GptModel,
} from '@llm-transformer-diy/model';
import {
  GpuContext,
  KernelRegistry,
  probeWebGpu,
  registerCoreKernels,
} from '@llm-transformer-diy/gpu';
import { createIndices, destroy, toArray, type TensorRuntime } from '@llm-transformer-diy/tensor';
import { readFile, mkdtemp, rm, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { beforeAll, describe, expect, it } from 'vitest';

import {
  loadGptCheckpointTensors,
  loadGptModelFromCheckpoint,
  saveGptCheckpoint,
} from './checkpoint';

const webgpuAvailable = await probeWebGpu();

const tinyCfg: GptConfig = {
  vocabSize: 8,
  maxSeqLen: 8,
  dModel: 32,
  nHeads: 4,
  dFf: 64,
  nLayers: 1,
};

function maxAbsDiff(a: Float32Array, b: Float32Array): number {
  expect(a.length).toBe(b.length);
  let m = 0;
  for (let i = 0; i < a.length; i++) {
    m = Math.max(m, Math.abs(a[i] - b[i]));
  }
  return m;
}

function disposeGptModel(model: GptModel): void {
  for (const p of collectGptParameters(model)) {
    destroy(p.tensor);
    if (p.masterTensor !== null) {
      destroy(p.masterTensor);
    }
    if (p.grad) {
      destroy(p.grad);
    }
  }
  for (const v of model.headExpand) {
    destroy(v.tensor);
    if (v.grad) {
      destroy(v.grad);
    }
  }
}

describe.skipIf(!webgpuAvailable)('Checkpoint load with bf16 compute dtype', () => {
  let rt: TensorRuntime;

  beforeAll(async () => {
    const ctx = await GpuContext.create();
    const registry = new KernelRegistry();
    registerCoreKernels(ctx.device, registry);
    rt = { ctx, registry };
  });

  it('f32 checkpoint → bf16 model forward matches f32 within bf16 tolerance', async () => {
    const modelF32 = createGptModel(rt, tinyCfg);
    const dir = await mkdtemp(path.join(os.tmpdir(), 'llm-diy-dtype-'));
    const binPath = path.join(dir, 'w.bin');
    const jsonPath = path.join(dir, 'w.json');
    try {
      await saveGptCheckpoint({
        rt,
        model: modelF32,
        binPath,
        jsonPath,
        config: tinyCfg,
      });

      const { model: modelBf16 } = await loadGptModelFromCheckpoint(rt, binPath, jsonPath, {
        computeDtype: 'bf16',
      });

      const T = 5;
      const tok = createIndices(rt, Uint32Array.from([0, 1, 2, 3, 4]), [T]);

      const logitsF = forwardGpt(rt, modelF32, tok);
      const logitsB = forwardGpt(rt, modelBf16, tok);

      const hF = await toArray(rt, logitsF.tensor);
      const hB = await toArray(rt, logitsB.tensor);

      expect(maxAbsDiff(hF, hB)).toBeLessThan(0.25);

      destroy(tok);
      destroy(logitsF.tensor);
      destroy(logitsB.tensor);
      disposeGptModel(modelF32);
      disposeGptModel(modelBf16);
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });

  it('bf16 checkpoint binaries round-trip bit-identically through save → load → save', async () => {
    const modelA = createGptModel(rt, tinyCfg, { dtype: 'bf16' });
    const dir1 = await mkdtemp(path.join(os.tmpdir(), 'llm-diy-bf16a-'));
    const dir2 = await mkdtemp(path.join(os.tmpdir(), 'llm-diy-bf16b-'));
    const bin1 = path.join(dir1, 'w.bin');
    const json1 = path.join(dir1, 'w.json');
    const bin2 = path.join(dir2, 'w.bin');
    const json2 = path.join(dir2, 'w.json');
    try {
      await saveGptCheckpoint({
        rt,
        model: modelA,
        binPath: bin1,
        jsonPath: json1,
        config: tinyCfg,
      });
      const rawBin1 = await readFile(bin1);

      const { model: modelB } = await loadGptModelFromCheckpoint(rt, bin1, json1, {
        computeDtype: 'bf16',
      });
      await saveGptCheckpoint({
        rt,
        model: modelB,
        binPath: bin2,
        jsonPath: json2,
        config: tinyCfg,
      });
      const rawBin2 = await readFile(bin2);

      expect(Buffer.compare(rawBin1, rawBin2)).toBe(0);

      const meta = JSON.parse(await readFile(json1, 'utf8')) as {
        tensors: Record<string, { dtype?: string }>;
      };
      for (const e of Object.values(meta.tensors)) {
        expect(e.dtype).toBe('bf16');
      }

      disposeGptModel(modelA);
      disposeGptModel(modelB);
    } finally {
      await rm(dir1, { recursive: true, force: true });
      await rm(dir2, { recursive: true, force: true });
    }
  });

  it('bf16 checkpoint → f32 model with allowDtypeUpcast matches unpacked host tensors', async () => {
    const modelBf16 = createGptModel(rt, tinyCfg, { dtype: 'bf16' });
    const dir = await mkdtemp(path.join(os.tmpdir(), 'llm-diy-upcast-'));
    const binPath = path.join(dir, 'w.bin');
    const jsonPath = path.join(dir, 'w.json');
    try {
      await saveGptCheckpoint({
        rt,
        model: modelBf16,
        binPath,
        jsonPath,
        config: tinyCfg,
      });
      const { tensors: fromDisk } = await loadGptCheckpointTensors(binPath, jsonPath);
      const { model: modelF32 } = await loadGptModelFromCheckpoint(rt, binPath, jsonPath, {
        computeDtype: 'f32',
        allowDtypeUpcast: true,
      });

      const names = gptParameterTensorNames(modelF32);
      const params = collectGptParameters(modelF32);
      for (let i = 0; i < names.length; i++) {
        const name = names[i];
        const p = params[i];
        const h = await toArray(rt, p.tensor);
        const ref = fromDisk[name];
        expect(ref).toBeDefined();
        expect(maxAbsDiff(h, ref)).toBeLessThan(1e-6);
      }

      disposeGptModel(modelBf16);
      disposeGptModel(modelF32);
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });

  it('bf16 checkpoint → f32 model without allowDtypeUpcast throws', async () => {
    const modelBf16 = createGptModel(rt, tinyCfg, { dtype: 'bf16' });
    const dir = await mkdtemp(path.join(os.tmpdir(), 'llm-diy-reject-'));
    const binPath = path.join(dir, 'w.bin');
    const jsonPath = path.join(dir, 'w.json');
    try {
      await saveGptCheckpoint({
        rt,
        model: modelBf16,
        binPath,
        jsonPath,
        config: tinyCfg,
      });
      await expect(
        loadGptModelFromCheckpoint(rt, binPath, jsonPath, { computeDtype: 'f32' }),
      ).rejects.toThrow(/allowDtypeUpcast/);

      disposeGptModel(modelBf16);
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });

  it('legacy sidecar tensors without dtype key still load (treated as f32 storage)', async () => {
    const modelF32 = createGptModel(rt, tinyCfg);
    const dir = await mkdtemp(path.join(os.tmpdir(), 'llm-diy-legacy-'));
    const binPath = path.join(dir, 'w.bin');
    const jsonPath = path.join(dir, 'w.json');
    try {
      await saveGptCheckpoint({
        rt,
        model: modelF32,
        binPath,
        jsonPath,
        config: tinyCfg,
      });
      const raw = JSON.parse(await readFile(jsonPath, 'utf8')) as {
        tensors: Record<string, { offset: number; shape: number[]; dtype?: string }>;
      };
      for (const e of Object.values(raw.tensors)) {
        delete e.dtype;
      }
      await writeFile(jsonPath, `${JSON.stringify(raw, null, 2)}\n`, 'utf8');

      const { model: modelBf16 } = await loadGptModelFromCheckpoint(rt, binPath, jsonPath, {
        computeDtype: 'bf16',
      });
      expect(modelBf16).toBeDefined();

      disposeGptModel(modelF32);
      disposeGptModel(modelBf16);
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });
});
