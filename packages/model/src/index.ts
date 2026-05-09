import { AUTOGRAD_LAYER_MARKER } from '@llm-transformer-diy/autograd';

/** Transformer architecture scaffolding */
export const MODEL_LAYER_MARKER = Symbol.for('@llm-transformer-diy/model');

export function __modelDependsOnAutograd(): typeof AUTOGRAD_LAYER_MARKER {
  return AUTOGRAD_LAYER_MARKER;
}
