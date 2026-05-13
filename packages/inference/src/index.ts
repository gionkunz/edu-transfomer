import { MODEL_LAYER_MARKER } from '@llm-transformer-diy/model';

/** Sampling + KV-cache decode (`packages/inference`). */
export const INFERENCE_LAYER_MARKER = Symbol.for('@llm-transformer-diy/inference');

export function __inferenceTouchesModel(): typeof MODEL_LAYER_MARKER {
  return MODEL_LAYER_MARKER;
}

export { generateAutoregressive, type GenerateOptions } from './generate';
export {
  applyRepetitionPenalty,
  DEFAULT_SAMPLE_OPTIONS,
  greedyArgmax,
  sampleLogits,
  type Random01,
  type SampleOptions,
} from './sampling';
