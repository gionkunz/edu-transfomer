import { MODEL_LAYER_MARKER } from '@llm-transformer-diy/model';

/** Sampling + KV-cache decode (`packages/inference`). */
export const INFERENCE_LAYER_MARKER = Symbol.for('@llm-transformer-diy/inference');

export function __inferenceTouchesModel(): typeof MODEL_LAYER_MARKER {
  return MODEL_LAYER_MARKER;
}

export {
  assertEmbeddingMatchesTokenizer,
  classifyCosine,
  cosineSimilarity,
  cosineStatsAgainstMatrix,
  describeSimilarityBand,
  EmbeddingDimensionError,
  EmbeddingIdOutOfRangeError,
  EmbeddingTokenizerMismatchError,
  EmbeddingZeroNormError,
  lookupEmbedding,
  makeEmbeddingMatrix,
  topKByCosine,
  type CosineStats,
  type EmbeddingMatrix,
  type LookupPool,
  type SimilarityBand,
  type TopKEntry,
  type TopKOptions,
} from './embeddings';
export {
  collectAtomNodes,
  collectAtoms,
  EmbeddingExpressionEvalError,
  EmbeddingExpressionParseError,
  evaluateEmbeddingExpression,
  parseEmbeddingExpression,
  tokenizeEmbeddingExpression,
  type ExprAdd,
  type ExprAtom,
  type ExprNeg,
  type ExprNode,
  type ExprSub,
  type ResolveAtom,
} from './embedding-expression';
export { generateAutoregressive, type GenerateOptions } from './generate';
export {
  applyRepetitionPenalty,
  DEFAULT_SAMPLE_OPTIONS,
  greedyArgmax,
  sampleLogits,
  sampleLogitsWithProbs,
  type Random01,
  type SampleOptions,
  type SampleWithProbsResult,
} from './sampling';
