import type { Variable } from '@llm-transformer-diy/autograd';
import {
  disposeGptKvForwardGraph,
  forwardGptDecodeStep,
  forwardGptPrefill,
  type GptKVCache,
  type GptModel,
} from '@llm-transformer-diy/model';
import { createIndices, toArray, type TensorRuntime } from '@llm-transformer-diy/tensor';

import { buildFlowTrace, type FlowTrace } from './flow-trace';
import {
  DEFAULT_SAMPLE_OPTIONS,
  sampleLogitsWithProbs,
  type Random01,
  type SampleOptions,
} from './sampling';

/** Post-softmax per-head attention captured during a decode forward. */
export type DecodeAttentionPayload = {
  readonly layerIndex: number;
  readonly nHeads: number;
  readonly prefixRows: number;
  /**
   * Row-major stacked **`[nHeads, prefixRows]`** weights for the decode query row.
   * Each head row sums to **≈ 1** (post-softmax).
   */
  readonly weights: Float32Array;
};

/**
 * What **`onStep`** may return: nothing to continue, or **`'stop'`** to end
 * generation after the token just sampled (no further decode forward runs).
 * Lets a host stop on conditions inference cannot see, e.g. a marker in the
 * decoded text.
 */
export type GenerateStepSignal = void | 'stop';

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
  /**
   * Post-softmax attention for the focused layer when **`attentionLayerIndex`** is set.
   * Omitted on the first sampled token (prefill logits only; decode-only spotlight).
   */
  readonly attention?: DecodeAttentionPayload;
  /**
   * Per-block trace of the forward that produced this token's distribution,
   * when **`captureFlow`** is on. Omitted on the first sampled token (prefill).
   */
  readonly flow?: FlowTrace;
};

export type GenerateOptions = {
  readonly maxNewTokens: number;
  readonly eosTokenId?: number;
  readonly sample?: SampleOptions;
  readonly rng?: Random01;
  /**
   * When set, each decode forward captures post-softmax attention at this block index.
   * A function is re-evaluated each decode step so hosts can change the focused layer live.
   * Omit to skip attention readback (**`--no-explore`** / headless callers).
   */
  readonly attentionLayerIndex?: number | (() => number);
  /**
   * When true, each decode forward also reads back a per-block trace (see
   * **`GenerateStep.flow`**): logit lens, attention of every block, and how much
   * each block changes the residual stream. A function is re-evaluated each
   * step so hosts can switch it on and off live.
   */
  readonly captureFlow?: boolean | (() => boolean);
  /** Called for each **new** sampled token (not prompt positions). */
  readonly onSampledToken?: (tokenId: number, newTokenIndex: number) => void;
  /**
   * Per-step host hook. Fires **after** **`onSampledToken`** with the post-shaping
   * **`probabilities`** that drove the draw. The decode loop **`await`s** the returned promise
   * (if any) before issuing the next **`forwardGptDecodeStep`**, so hosts can implement
   * pacing / pause without reimplementing sampling math here. Returning **`'stop'`**
   * ends generation after this token.
   */
  readonly onStep?: (step: GenerateStep) => GenerateStepSignal | Promise<GenerateStepSignal>;
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

function attentionPayloadFromWeights(
  layerIndex: number,
  nHeads: number,
  weights: Float32Array,
): DecodeAttentionPayload {
  if (weights.length % nHeads !== 0) {
    throw new Error(
      `attentionPayloadFromWeights: weights length ${String(weights.length)} not divisible by nHeads ${String(nHeads)}`,
    );
  }
  return {
    layerIndex,
    nHeads,
    prefixRows: weights.length / nHeads,
    weights,
  };
}

function resolveAttentionLayerIndex(
  layer: number | (() => number) | undefined,
): number | undefined {
  if (layer === undefined) return undefined;
  return typeof layer === 'function' ? layer() : layer;
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
  // `liveLogits` is the one forward graph whose logits have not been read yet.
  // Each graph is released right after its readback (this includes the prompt
  // tensor, a leaf of the prefill graph), so GPU memory stays flat over a long
  // generation instead of growing with every token.
  let liveLogits: Variable | null = forwardGptPrefill(rt, model, promptTensor, cache);

  let pendingAttention: DecodeAttentionPayload | undefined;
  let pendingFlow: Omit<Parameters<typeof buildFlowTrace>[0], 'sampledTokenId'> | undefined;

  try {
    for (let n = 0; n < opts.maxNewTokens; n++) {
      const row = await lastRowLogits(rt, liveLogits, V);
      disposeGptKvForwardGraph(liveLogits, model, cache);
      liveLogits = null;

      const { tokenId: tid, probabilities } = sampleLogitsWithProbs(rng, row, sample);
      out.push(tid);
      const genIndex = out.length - promptIds.length - 1;
      if (opts.onSampledToken !== undefined && genIndex >= 0) {
        opts.onSampledToken(tid, genIndex);
      }
      if (opts.onStep !== undefined && genIndex >= 0) {
        const step: GenerateStep = {
          tokenId: tid,
          newTokenIndex: genIndex,
          probabilities,
          ...(genIndex >= 1 && pendingAttention !== undefined
            ? { attention: pendingAttention }
            : {}),
          ...(genIndex >= 1 && pendingFlow !== undefined
            ? { flow: buildFlowTrace({ ...pendingFlow, sampledTokenId: tid }) }
            : {}),
        };
        const signal = await opts.onStep(step);
        if (signal === 'stop') {
          break;
        }
      }

      if (opts.eosTokenId !== undefined && tid === opts.eosTokenId) {
        break;
      }
      if (cache.seqLen >= model.config.maxSeqLen) {
        break;
      }

      const captureLayer = resolveAttentionLayerIndex(opts.attentionLayerIndex);
      const captureFlow =
        typeof opts.captureFlow === 'function' ? opts.captureFlow() : opts.captureFlow === true;
      const decodeResult = await forwardGptDecodeStep(rt, model, tid, cache, {
        ...(captureLayer !== undefined ? { attentionCapture: { layerIndex: captureLayer } } : {}),
        ...(captureFlow ? { flowTrace: true } : {}),
      });
      liveLogits = decodeResult.logits;

      pendingAttention =
        captureLayer !== undefined && decodeResult.attention !== undefined
          ? attentionPayloadFromWeights(captureLayer, model.config.nHeads, decodeResult.attention)
          : undefined;
      pendingFlow =
        decodeResult.layerLogits !== undefined &&
        decodeResult.allAttention !== undefined &&
        decodeResult.blockVectors !== undefined
          ? {
              layerLogits: decodeResult.layerLogits,
              allAttention: decodeResult.allAttention,
              blockVectors: decodeResult.blockVectors,
              nLayers: model.config.nLayers,
              nHeads: model.config.nHeads,
            }
          : undefined;
    }
  } finally {
    if (liveLogits !== null) {
      disposeGptKvForwardGraph(liveLogits, model, cache);
    }
  }

  return Uint32Array.from(out);
}
