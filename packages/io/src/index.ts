/**
 * Checkpoint I/O (`model.bin` + JSON sidecars), generic JSON loaders, and corpus globs —
 * aligned with Component 9 in `docs/project.md`.
 */

export const IO_LAYER_MARKER = Symbol.for('@llm-transformer-diy/io');

export type {
  CheckpointDtype,
  PackedCheckpointSidecar,
  TensorManifestEntry,
  TensorPayload,
} from './checkpoint-format';

export {
  packPackedCheckpoint,
  readPackedCheckpointFiles,
  tensorElementCount,
  tensorPackedByteLength,
  unpackPackedCheckpoint,
  validatePackedCheckpointSidecar,
  writePackedCheckpointFiles,
} from './checkpoint-format';

export { readJsonFile } from './config-loader';

export { discoverTrainingFiles } from './glob-loader';
