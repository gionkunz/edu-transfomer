import { GPU_LAYER_MARKER } from '@llm-transformer-diy/gpu';

/** Tensor primitive — scaffold only */
export const TENSOR_LAYER_MARKER = Symbol.for('@llm-transformer-diy/tensor');

export function __tensorDependsOnGpu(): typeof GPU_LAYER_MARKER {
  return GPU_LAYER_MARKER;
}
