import { MODEL_LAYER_MARKER } from '@llm-transformer-diy/model';

/** Sampling + KV-cache scaffolding lives here eventually */
export const INFERENCE_LAYER_MARKER = Symbol.for('@llm-transformer-diy/inference');

export function __inferenceTouchesModel(): typeof MODEL_LAYER_MARKER {
  return MODEL_LAYER_MARKER;
}
