import { IO_LAYER_MARKER } from '@llm-transformer-diy/io';
import { MODEL_LAYER_MARKER } from '@llm-transformer-diy/model';
import { TOKENIZER_LAYER_MARKER } from '@llm-transformer-diy/tokenizer';

export {
  ADAMW_DEFAULT_BETA1,
  ADAMW_DEFAULT_BETA2,
  ADAMW_DEFAULT_EPS,
  ADAMW_DEFAULT_WEIGHT_DECAY,
  AdamW,
} from './adamw';
export type { AdamWHyperparams, AdamWOptions } from './adamw';
export type { SaveGptCheckpointOptions, LoadGptModelFromCheckpointOptions } from './checkpoint';
export {
  hydrateGptModelWeights,
  loadGptCheckpointTensors,
  loadGptModelFromCheckpoint,
  saveGptCheckpoint,
} from './checkpoint';
export type {
  ResumeTrainingCheckpoint,
  SaveTrainingCheckpointOptions,
  TrainStateSidecar,
  TrainingCheckpointPaths,
} from './train-checkpoint';
export {
  TRAIN_DEFAULT_CHECKPOINT_EVERY,
  TRAIN_STATE_KIND,
  TRAIN_STATE_VERSION,
  hydrateAdamWFromCheckpoint,
  loadTrainingCheckpointForResume,
  resolveCheckpointEvery,
  saveTrainingCheckpoint,
  shouldSavePeriodicCheckpoint,
  trainingCheckpointPaths,
} from './train-checkpoint';
export { createLmBatch, assertCorpusTokenIdsInVocab } from './lm-batch';
export {
  assertModelParametersFiniteForResume,
  auditModelParametersFinite,
  buildNonFiniteLossDiagnostics,
  correlateBadRowsWithTokens,
  scanFloat32FiniteStats,
  scanLogitsRowsByPosition,
  scanTokenEmbeddingRows,
  type EmbeddingRowScan,
  type Float32FiniteScan,
  type LogitsRowScan,
  type ParameterFiniteAudit,
  type ParameterTensorAudit,
  type TokenPositionDiagnostic,
  type TrainStepBatchContext,
} from './train-failure-diagnostics';
export { getCosineLr, type CosineLrSchedule } from './scheduler';
export {
  registerModelParameters,
  Trainer,
  type TrainerOptions,
  type TrainerStepInfo,
} from './trainer';
export { runReplayStepProbe } from './train-replay';

/** Training loop scaffolding */
export const TRAINING_LAYER_MARKER = Symbol.for('@llm-transformer-diy/training');

export function wiringProof(): readonly unknown[] {
  return [MODEL_LAYER_MARKER, TOKENIZER_LAYER_MARKER, IO_LAYER_MARKER];
}
