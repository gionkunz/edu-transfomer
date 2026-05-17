export {
  BPETokenizer,
  PRETOKENIZER_REGEX_GPT2_V1,
  splitIntoRegions,
  type BpeTrainFromFilesHooks,
  type BpeTrainFromFilesOptions,
  type BpeTrainHooks,
  type BpeTrainMergeCompleteEvent,
  type BpeTrainOptions,
  type BPETokenizerOptions,
  type BpeDecodeOptions,
  type BpeEncodeHooks,
  type BpeEncodeProgressEvent,
  type PretokenizerPolicy,
} from './bpe-tokenizer';
export { CharTokenizer, type CharTokenizerOptions } from './char-tokenizer';
export type { Tokenizer } from './types';

/** Dependency / layer marker for CLIs and downstream stubs. */
export const TOKENIZER_LAYER_MARKER = Symbol.for('@llm-transformer-diy/tokenizer');
