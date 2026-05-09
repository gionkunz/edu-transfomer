import { INFERENCE_LAYER_MARKER } from '@llm-transformer-diy/inference';
import { TOKENIZER_LAYER_MARKER } from '@llm-transformer-diy/tokenizer';
import { TRAINING_LAYER_MARKER } from '@llm-transformer-diy/training';

/**
 * Executable placeholder only — Commander / wolf-tui arrive via future OpenSpec
 * proposals aligned with docs/project.md.
 */
export function createCliPlaceholder(): {
  tokenizer: typeof TOKENIZER_LAYER_MARKER;
  training: typeof TRAINING_LAYER_MARKER;
  inference: typeof INFERENCE_LAYER_MARKER;
} {
  return {
    tokenizer: TOKENIZER_LAYER_MARKER,
    training: TRAINING_LAYER_MARKER,
    inference: INFERENCE_LAYER_MARKER,
  };
}

/** Reserved for wired entrypoint (`node dist/main.js`) — no side-effects at import time. */
export function main(): void {
  void createCliPlaceholder();
}
