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
export type { SaveGptCheckpointOptions } from './checkpoint';
export {
  hydrateGptModelWeights,
  loadGptCheckpointTensors,
  loadGptModelFromCheckpoint,
  saveGptCheckpoint,
} from './checkpoint';
export { createLmBatch } from './lm-batch';
export { getCosineLr, type CosineLrSchedule } from './scheduler';
export {
  registerModelParameters,
  Trainer,
  type TrainerOptions,
  type TrainerStepInfo,
} from './trainer';

/** Training loop scaffolding */
export const TRAINING_LAYER_MARKER = Symbol.for('@llm-transformer-diy/training');

export function wiringProof(): readonly unknown[] {
  return [MODEL_LAYER_MARKER, TOKENIZER_LAYER_MARKER, IO_LAYER_MARKER];
}
