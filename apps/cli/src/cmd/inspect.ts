import type { PackedCheckpointSidecar } from '@llm-transformer-diy/io';
import {
  readJsonFile,
  tensorElementCount,
  validatePackedCheckpointSidecar,
} from '@llm-transformer-diy/io';
import type { GptConfig } from '@llm-transformer-diy/model';

import { resolvePackedCheckpointPaths } from '../paths';

function asGptConfig(c: unknown): GptConfig | null {
  if (c === null || typeof c !== 'object') return null;
  const o = c as Record<string, unknown>;
  const v = Number(o.vocabSize);
  const m = Number(o.maxSeqLen);
  if (!Number.isFinite(v) || !Number.isFinite(m)) return null;
  return c as GptConfig;
}

/** Pure summary for tests and stdout. */
export function formatInspectReport(sidecar: PackedCheckpointSidecar): string {
  const lines: string[] = [];
  const cfgRaw = sidecar.config;
  const cfg = asGptConfig(cfgRaw);
  if (cfg) {
    lines.push(
      `config: d_model=${String(cfg.dModel)} n_layers=${String(cfg.nLayers)} n_heads=${String(cfg.nHeads)} d_ff=${String(cfg.dFf)} max_seq_len=${String(cfg.maxSeqLen)} vocab=${String(cfg.vocabSize)}`,
    );
  } else {
    lines.push('config: <non-GptConfig JSON>');
  }

  let params = 0;
  const tensorNames = Object.keys(sidecar.tensors).sort((a, b) => a.localeCompare(b));
  for (const n of tensorNames) {
    const e = sidecar.tensors[n];
    if (!e) continue;
    params += tensorElementCount(e.shape);
  }
  lines.push(`parameters (elements): ${String(params)}`);

  const preview = tensorNames.slice(0, 12);
  for (const n of preview) {
    const e = sidecar.tensors[n];
    if (!e) continue;
    lines.push(`  ${n}: [${e.shape.join(', ')}] ${e.dtype}`);
  }
  if (tensorNames.length > preview.length) {
    lines.push(`  … ${String(tensorNames.length - preview.length)} more tensors`);
  }

  const meta = sidecar.metadata;
  if (sidecar.metadata !== undefined) {
    lines.push(`metadata: ${JSON.stringify(meta)}`);
  }

  return `${lines.join('\n')}\n`;
}

export async function runInspect(modelArg: string): Promise<void> {
  const { jsonPath } = resolvePackedCheckpointPaths(modelArg);
  const raw = await readJsonFile(jsonPath);
  const sidecar = validatePackedCheckpointSidecar(raw);
  console.log(formatInspectReport(sidecar));
}

/** Reads **`model.json`** only (for tests / fast path). */
export async function loadSidecarFromJsonFile(jsonPath: string): Promise<PackedCheckpointSidecar> {
  const raw = await readJsonFile(jsonPath);
  return validatePackedCheckpointSidecar(raw);
}
