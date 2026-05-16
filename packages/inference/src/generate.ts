import type { Variable } from '@llm-transformer-diy/autograd';
import {
  forwardGptDecodeStep,
  forwardGptPrefill,
  type GptKVCache,
  type GptModel,
} from '@llm-transformer-diy/model';
import { createIndices, destroy, toArray, type TensorRuntime } from '@llm-transformer-diy/tensor';

import {
  DEFAULT_SAMPLE_OPTIONS,
  sampleLogitsWithProbs,
  type Random01,
  type SampleOptions,
} from './sampling';

/** Per-step hook payload for **`GenerateOptions.onStep`**. */
export type GenerateStep = {
  readonly tokenId: number;
  readonly newTokenIndex: number;
  /**
   * Distribution actually used for the categorical draw at this step
   * (post-temperature, post-top-k, post-repetition-penalty).
   *
   * Length equals **`model.config.vocabSize`**. Each call receives a freshly-allocated
   * **`Float32Array`** owned by the host (safe to retain or mutate after the await resolves).
   */
  readonly probabilities: Float32Array;
};

export type GenerateOptions = {
  readonly maxNewTokens: number;
  readonly eosTokenId?: number;
  readonly sample?: SampleOptions;
  readonly rng?: Random01;
  /** Called for each **new** sampled token (not prompt positions). */
  readonly onSampledToken?: (tokenId: number, newTokenIndex: number) => void;
  /**
   * Per-step host hook. Fires **after** **`onSampledToken`** with the post-shaping
   * **`probabilities`** that drove the draw. The decode loop **`await`s** the returned promise
   * (if any) before issuing the next **`forwardGptDecodeStep`**, so hosts can implement
   * pacing / pause without reimplementing sampling math here.
   */
  readonly onStep?: (step: GenerateStep) => void | Promise<void>;
};

async function lastRowLogits(
  rt: TensorRuntime,
  logits: Variable,
  vocabSize: number,
): Promise<Float32Array> {
  const t = logits.tensor;
  const full = await toArray(rt, t);
  if (t.shape.length !== 2) {
    throw new Error('lastRowLogits: expected rank-2 logits tensor');
  }
  const tDim = t.shape[0];
  const vDim = t.shape[1];
  if (tDim === undefined || vDim === undefined) {
    throw new Error('lastRowLogits: logits tensor missing shape dimensions');
  }
  if (vDim !== vocabSize) {
    throw new Error(
      `lastRowLogits: vocab mismatch (tensor ${String(vDim)} vs config ${String(vocabSize)})`,
    );
  }
  if (tDim === 1) {
    return full;
  }
  const start = (tDim - 1) * vocabSize;
  return full.subarray(start, start + vocabSize);
}

/**
 * Appends up to **`maxNewTokens`** tokens after **`promptIds`** using KV-cache prefill + decode steps.
 *
 * Requires a non-empty **`promptIds`** array. **`cache`** must be fresh (**`seqLen === 0`**) or match this call only.
 */
export async function generateAutoregressive(
  rt: TensorRuntime,
  model: GptModel,
  promptIds: Uint32Array,
  cache: GptKVCache,
  opts: GenerateOptions,
): Promise<Uint32Array> {
  if (promptIds.length === 0) {
    throw new Error('generateAutoregressive: promptIds must be non-empty');
  }
  if (!Number.isInteger(opts.maxNewTokens) || opts.maxNewTokens < 0) {
    throw new Error('generateAutoregressive: maxNewTokens must be a non-negative integer');
  }
  if (cache.seqLen !== 0) {
    throw new Error('generateAutoregressive: KV cache must be empty (seqLen === 0)');
  }

  const sample = opts.sample ?? DEFAULT_SAMPLE_OPTIONS;
  const rng = opts.rng ?? Math.random;
  const V = model.config.vocabSize;
  const out: number[] = [...promptIds];

  const promptTensor = createIndices(rt, promptIds, [promptIds.length]);
  let logitsVar = forwardGptPrefill(rt, model, promptTensor, cache);
  destroy(promptTensor);

  for (let n = 0; n < opts.maxNewTokens; n++) {
    const row = await lastRowLogits(rt, logitsVar, V);
    const { tokenId: tid, probabilities } = sampleLogitsWithProbs(rng, row, sample);
    out.push(tid);
    const genIndex = out.length - promptIds.length - 1;
    if (opts.onSampledToken !== undefined && genIndex >= 0) {
      opts.onSampledToken(tid, genIndex);
    }
    if (opts.onStep !== undefined && genIndex >= 0) {
      const r = opts.onStep({ tokenId: tid, newTokenIndex: genIndex, probabilities });
      if (r !== undefined) {
        await r;
      }
    }

    if (opts.eosTokenId !== undefined && tid === opts.eosTokenId) {
      break;
    }
    if (cache.seqLen >= model.config.maxSeqLen) {
      break;
    }
    logitsVar = forwardGptDecodeStep(rt, model, tid, cache);
  }

  return Uint32Array.from(out);
}
