import { AUTOGRAD_LAYER_MARKER } from '@llm-transformer-diy/autograd';

/** Transformer graph + GPT config (`packages/model`). */
export const MODEL_LAYER_MARKER = Symbol.for('@llm-transformer-diy/model');

export function __modelDependsOnAutograd(): typeof AUTOGRAD_LAYER_MARKER {
  return AUTOGRAD_LAYER_MARKER;
}

export { assertValidGptConfig, type GptConfig } from './config';
export {
  buildCausalBiasTensor,
  collectGptParameters,
  createGptModel,
  forwardGpt,
  type GptModel,
  type TransformerBlockParams,
} from './gpt';
