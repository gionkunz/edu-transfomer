/** Minimal GPT hyperparameters for `packages/model` (decoder-only, Pre-LN). */
export type GptConfig = {
  readonly vocabSize: number;
  readonly maxSeqLen: number;
  readonly dModel: number;
  readonly nHeads: number;
  readonly dFf: number;
  readonly nLayers: number;
};

/** Validates invariants required by the reference implementation. */
export function assertValidGptConfig(c: GptConfig): void {
  if (!Number.isInteger(c.vocabSize) || c.vocabSize <= 0) {
    throw new Error(`assertValidGptConfig: vocabSize must be a positive integer`);
  }
  if (!Number.isInteger(c.maxSeqLen) || c.maxSeqLen <= 0) {
    throw new Error(`assertValidGptConfig: maxSeqLen must be a positive integer`);
  }
  if (!Number.isInteger(c.dModel) || c.dModel <= 0) {
    throw new Error(`assertValidGptConfig: dModel must be a positive integer`);
  }
  if (!Number.isInteger(c.nHeads) || c.nHeads <= 0) {
    throw new Error(`assertValidGptConfig: nHeads must be a positive integer`);
  }
  if (c.dModel % c.nHeads !== 0) {
    throw new Error(
      `assertValidGptConfig: dModel (${String(c.dModel)}) must be divisible by nHeads (${String(c.nHeads)})`,
    );
  }
  if (!Number.isInteger(c.dFf) || c.dFf <= 0) {
    throw new Error(`assertValidGptConfig: dFf must be a positive integer`);
  }
  if (!Number.isInteger(c.nLayers) || c.nLayers <= 0) {
    throw new Error(`assertValidGptConfig: nLayers must be a positive integer`);
  }
}
