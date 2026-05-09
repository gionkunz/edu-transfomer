import { TENSOR_LAYER_MARKER } from '@llm-transformer-diy/tensor';

/** Computation graph scaffolding — differentiation lives here in later proposals. */
export const AUTOGRAD_LAYER_MARKER = Symbol.for('@llm-transformer-diy/autograd');

export function __autogradDependsOnTensor(): typeof TENSOR_LAYER_MARKER {
  return TENSOR_LAYER_MARKER;
}
