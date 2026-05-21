import fs from 'node:fs';
import path from 'node:path';

/**
 * Resolves the checkpoint directory for **`modelOrJson`** (directory containing
 * **`model.json`**, or the parent of a **`*.json`** sidecar).
 */
export function resolveCheckpointDir(modelOrJson: string): string {
  const abs = path.resolve(modelOrJson);
  if (!fs.existsSync(abs)) {
    throw new Error(`path not found: ${modelOrJson}`);
  }
  const stat = fs.statSync(abs);
  if (stat.isDirectory()) {
    return abs;
  }
  if (abs.endsWith('.json')) {
    return path.dirname(abs);
  }
  throw new Error(`Expected checkpoint directory or a *.json sidecar path, got: ${modelOrJson}`);
}

/**
 * Resolves a BPE tokenizer JSON path. When **`tokenizerArg`** is omitted, looks for
 * **`tokenizer.json`** in the checkpoint directory implied by **`modelOrJson`**.
 */
export function resolveTokenizerPath(modelOrJson: string, tokenizerArg?: string): string {
  const explicit = tokenizerArg !== undefined ? String(tokenizerArg).trim() : '';
  if (explicit !== '') {
    return path.resolve(explicit);
  }
  const checkpointDir = resolveCheckpointDir(modelOrJson);
  const tokenizerPath = path.join(checkpointDir, 'tokenizer.json');
  if (!fs.existsSync(tokenizerPath)) {
    throw new Error(
      `no --tokenizer given and tokenizer.json not found in checkpoint dir: ${checkpointDir}`,
    );
  }
  return tokenizerPath;
}

/**
 * Resolves **`model.bin`** + **`model.json`** for a checkpoint (directory containing
 * **`model.json`**, or a path to **`*.json`** sidecar).
 */
export function resolvePackedCheckpointPaths(modelOrJson: string): {
  binPath: string;
  jsonPath: string;
} {
  const abs = path.resolve(modelOrJson);
  if (!fs.existsSync(abs)) {
    throw new Error(`path not found: ${modelOrJson}`);
  }
  const stat = fs.statSync(abs);
  if (stat.isDirectory()) {
    return {
      jsonPath: path.join(abs, 'model.json'),
      binPath: path.join(abs, 'model.bin'),
    };
  }
  if (abs.endsWith('.json')) {
    const dir = path.dirname(abs);
    const base = path.basename(abs, '.json');
    return {
      jsonPath: abs,
      binPath: path.join(dir, `${base}.bin`),
    };
  }
  throw new Error(`Expected checkpoint directory or a *.json sidecar path, got: ${modelOrJson}`);
}
