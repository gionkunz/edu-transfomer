import { createIndices, type Tensor, type TensorRuntime } from '@llm-transformer-diy/tensor';

/**
 * Next-token LM slice: **input** **`token[start .. start+seqLen-1]`**, **targets**
 * **`token[start+1 .. start+seqLen]`** (same length **`seqLen`**).
 *
 * Caller must ensure **`start + seqLen < token.length`**.
 */
export function createLmBatch(
  rt: TensorRuntime,
  token: Uint32Array,
  start: number,
  seqLen: number,
): { readonly input: Tensor; readonly targets: Tensor } {
  if (!Number.isInteger(seqLen) || seqLen <= 0) {
    throw new Error(`createLmBatch: seqLen must be positive (got ${String(seqLen)})`);
  }
  if (!Number.isInteger(start) || start < 0) {
    throw new Error(`createLmBatch: start must be a non-negative integer`);
  }
  if (start + seqLen >= token.length) {
    throw new Error(
      `createLmBatch: need start + seqLen < token.length (${String(start)} + ${String(seqLen)} >= ${String(token.length)})`,
    );
  }
  const inp = token.subarray(start, start + seqLen);
  const tgt = token.subarray(start + 1, start + seqLen + 1);
  return {
    input: createIndices(rt, inp, [seqLen]),
    targets: createIndices(rt, tgt, [seqLen]),
  };
}
