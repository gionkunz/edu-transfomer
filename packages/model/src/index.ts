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
  gptParameterTensorNames,
  listGptParameterInfos,
  type CreateGptModelOptions,
  type GptModel,
  type GptParameterInfo,
  type TransformerBlockParams,
} from './gpt';
export {
  formatForwardProbeReport,
  layerIndexFromProbeStage,
  probeGptForward,
  readTokenIdsSliceFromFile,
  trainingBatchStart,
  type ForwardProbeReport,
  type ForwardProbeStage,
  type HiddenRowScan,
} from './gpt-forward-probe';
export {
  createGptKvCache,
  destroyGptKvCache,
  disposeGptKvForwardGraph,
  forwardGptDecodeStep,
  forwardGptPrefill,
  type DecodeAttentionCaptureOptions,
  type ForwardGptDecodeStepOptions,
  type GptDecodeStepResult,
  type GptKVCache,
  type GptKVHeadCache,
  type GptKVLayerCache,
} from './gpt-kv';
