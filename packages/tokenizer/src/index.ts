export { CharTokenizer, type CharTokenizerOptions } from './char-tokenizer';
export type { Tokenizer } from './types';

/** Dependency / layer marker for CLIs and downstream stubs. */
export const TOKENIZER_LAYER_MARKER = Symbol.for('@llm-transformer-diy/tokenizer');
