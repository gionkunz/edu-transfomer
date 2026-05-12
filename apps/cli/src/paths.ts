import fs from 'node:fs';
import path from 'node:path';

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
