import { mkdir, rm, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';
import { describe, expect, it } from 'vitest';

import { formatInspectReport, loadSidecarFromJsonFile } from './cmd/inspect';

describe('inspect helpers', () => {
  it('formatInspectReport summarizes tensors', () => {
    const text = formatInspectReport({
      config: {
        vocabSize: 64,
        maxSeqLen: 32,
        dModel: 32,
        nHeads: 4,
        dFf: 64,
        nLayers: 1,
      },
      tensors: {
        a: { offset: 0, shape: [2, 3], dtype: 'f32' },
        b: { offset: 24, shape: [4], dtype: 'f32' },
      },
    });
    expect(text).toContain('d_model=32');
    expect(text).toContain('parameters (elements): 10');
    expect(text).toContain('a: [2, 3]');
  });

  it('loadSidecarFromJsonFile parses fixture', async () => {
    const dir = await mkdir(path.join(os.tmpdir(), `llm-cli-inspect-${String(process.pid)}`), {
      recursive: true,
    });
    const jsonPath = path.join(dir, 'model.json');
    const payload = {
      config: {
        vocabSize: 12,
        maxSeqLen: 8,
        dModel: 16,
        nHeads: 2,
        dFf: 32,
        nLayers: 1,
      },
      tensors: {
        w: { offset: 0, shape: [16, 16], dtype: 'f32' },
      },
    };
    await writeFile(jsonPath, `${JSON.stringify(payload)}\n`, 'utf8');
    const sidecar = await loadSidecarFromJsonFile(jsonPath);
    expect(Object.keys(sidecar.tensors)).toEqual(['w']);
    const out = formatInspectReport(sidecar);
    expect(out).toContain('parameters (elements): 256');
    await rm(dir, { recursive: true, force: true });
  });
});
