import {
  add,
  createLeafVariable,
  createParameterRegistry,
  disposeAutogradGraphExceptParameters,
  embedding,
  gelu,
  matmul,
  registerParameter,
  rmsnorm,
  scale,
  softmax,
  transpose,
  type ParameterRegistry,
  type Variable,
} from '@llm-transformer-diy/autograd';
import {
  copyContiguousElems,
  createIndices,
  createZeros,
  destroy,
  floatDtypeOf,
  toArray,
  view2dRowPrefix,
  type Tensor,
  type TensorRuntime,
} from '@llm-transformer-diy/tensor';

import {
  buildCausalBiasTensor,
  collectGptParameters,
  type GptModel,
  type TransformerBlockParams,
} from './gpt';

export type GptKVHeadCache = {
  readonly k: Tensor;
  readonly v: Tensor;
};

export type GptKVLayerCache = {
  readonly heads: readonly GptKVHeadCache[];
};

/**
 * Per-layer K/V tensors shaped **`[maxSeqLen, d_k]`** per head; **`seqLen`** is the
 * number of valid prefix rows (**`[0 .. seqLen-1]`**) shared across layers after each step.
 */
export type GptKVCache = {
  readonly layers: readonly GptKVLayerCache[];
  seqLen: number;
};

/** Request post-softmax attention weights for one transformer block during decode. */
export type DecodeAttentionCaptureOptions = {
  readonly layerIndex: number;
};

export type ForwardGptDecodeStepOptions = {
  readonly attentionCapture?: DecodeAttentionCaptureOptions;
  /**
   * Also read back a **logit lens**: after every block, the residual stream is
   * pushed through the final norm and the LM head as if the model stopped
   * there. Shows the prediction forming layer by layer. One extra
   * `[1, dModel] × [dModel, vocab]` matmul per block and one readback.
   */
  readonly layerLens?: boolean;
  /**
   * Capture everything a "what is the model doing" view needs, in one forward:
   * the logit lens (as **`layerLens`**), post-softmax attention of **every**
   * block, and for each block the residual stream entering it plus the two
   * updates (attention, feed-forward) it adds.
   */
  readonly flowTrace?: boolean;
  /**
   * Everything **`flowTrace`** captures, plus what a step-by-step explanation
   * of attention needs: the two input vectors, and for every head its query,
   * its scores before softmax and its weighted mix of values. Keys and values
   * are not returned here; they live in the KV cache (see **`readKvCacheHead`**).
   */
  readonly lessonTrace?: boolean;
};

/** Decode step output; **`attention`** is row-major stacked **`[nHeads, prefixRows]`**. */
export type GptDecodeStepResult = {
  readonly logits: Variable;
  readonly attention?: Float32Array;
  /**
   * With **`layerLens`**: row-major **`[nLayers, vocabSize]`** raw logits, row
   * **`l`** = prediction from the residual stream after block **`l`**. The last
   * row equals **`logits`**.
   */
  readonly layerLogits?: Float32Array;
  /** With **`flowTrace`**: row-major **`[nLayers · nHeads, prefixRows]`** attention, block by block. */
  readonly allAttention?: Float32Array;
  /**
   * With **`flowTrace`**: row-major **`[3 · nLayers, dModel]`**. For block **`l`**,
   * row **`3l`** is the residual stream entering the block, **`3l + 1`** the
   * attention update added to it, **`3l + 2`** the feed-forward update.
   */
  readonly blockVectors?: Float32Array;
  /** With **`lessonTrace`**: row-major **`[2, dModel]`** — token embedding row, then position embedding row. */
  readonly inputVectors?: Float32Array;
  /** With **`lessonTrace`**: row-major **`[nLayers · nHeads, d_k]`** — each head's query for this token. */
  readonly allQueries?: Float32Array;
  /**
   * With **`lessonTrace`**: row-major **`[nLayers · nHeads, prefixRows]`** — each
   * head's scores **`(q · k) / √d_k`** before softmax. Softmax of a row is the
   * matching row of **`allAttention`**.
   */
  readonly allScores?: Float32Array;
  /**
   * With **`lessonTrace`**: row-major **`[nLayers · nHeads, d_k]`** — each head's
   * output before it is placed back into the residual stream: its values
   * averaged with its attention weights.
   */
  readonly allMixed?: Float32Array;
};

/** Per-head intermediates of one block, collected for **`lessonTrace`**. */
type HeadDetailCapture = {
  readonly queries: Variable[];
  readonly scores: Variable[];
  readonly mixed: Variable[];
};

export function createGptKvCache(rt: TensorRuntime, model: GptModel): GptKVCache {
  const { maxSeqLen, dModel, nHeads } = model.config;
  const dK = dModel / nHeads;
  const dt = floatDtypeOf(model.tokenEmbed.tensor);
  const layers: GptKVLayerCache[] = model.blocks.map((blk) => ({
    heads: blk.wq.map(() => ({
      k: createZeros(rt, [maxSeqLen, dK], dt),
      v: createZeros(rt, [maxSeqLen, dK], dt),
    })),
  }));
  return { layers, seqLen: 0 };
}

export function destroyGptKvCache(cache: GptKVCache): void {
  for (const layer of cache.layers) {
    for (const h of layer.heads) {
      destroy(h.k);
      destroy(h.v);
    }
  }
}

/** Residual stream → final norm → LM head, as if the model ended at this point. */
function lensLogits(
  rt: TensorRuntime,
  model: GptModel,
  h: Variable,
): { readonly normed: Variable; readonly logits: Variable } {
  const normed = rmsnorm(rt, h, model.finalNormGamma);
  return { normed, logits: matmul(rt, normed, model.lmHead) };
}

const disposalRegistries = new WeakMap<GptModel, ParameterRegistry>();

/**
 * Release every intermediate buffer of one prefill / decode forward once its
 * logits have been read back.
 *
 * Inference never calls `backward`, but each forward still builds an autograd
 * graph of small tensors (per layer, per head). Without this, a generation of
 * `n` tokens keeps `n` graphs alive. Model parameters, the constant head-expand
 * matrices and the KV cache (whose row-prefix views share its buffers) are kept;
 * everything else — including the token-id tensor passed to the forward — is
 * returned to the buffer pool.
 */
export function disposeGptKvForwardGraph(
  output: Variable,
  model: GptModel,
  cache: GptKVCache,
): void {
  let registry = disposalRegistries.get(model);
  if (registry === undefined) {
    registry = createParameterRegistry();
    for (const p of collectGptParameters(model)) {
      registerParameter(registry, p);
    }
    disposalRegistries.set(model, registry);
  }
  const keep: Tensor[] = model.headExpand.map((he) => he.tensor);
  for (const layer of cache.layers) {
    for (const head of layer.heads) {
      keep.push(head.k, head.v);
    }
  }
  disposeAutogradGraphExceptParameters(output, registry, keep);
}

function mhaForwardPrefill(
  rt: TensorRuntime,
  blk: TransformerBlockParams,
  x: Variable,
  maskVar: Variable,
  invSqrtDk: number,
  headExpand: readonly Variable[],
  layerIndex: number,
  cache: GptKVCache,
  seqLen: number,
): Variable {
  const H = blk.wq.length;
  const dK = blk.wq[0]?.tensor.shape[1];
  if (dK === undefined) {
    throw new Error('mhaForwardPrefill: missing head weights');
  }
  const elemCount = seqLen * dK;
  let acc: Variable | null = null;
  for (let h = 0; h < H; h++) {
    const wqh = blk.wq[h];
    const wkh = blk.wk[h];
    const wvh = blk.wv[h];
    const he = headExpand[h];
    if (!wqh || !wkh || !wvh || !he) {
      throw new Error('mhaForwardPrefill: missing head tensors');
    }
    const q = matmul(rt, x, wqh);
    const k = matmul(rt, x, wkh);
    const vMat = matmul(rt, x, wvh);
    const slot = cache.layers[layerIndex]?.heads[h];
    if (!slot) {
      throw new Error('mhaForwardPrefill: cache slot missing');
    }
    copyContiguousElems(rt, k.tensor, 0, slot.k, 0, elemCount);
    copyContiguousElems(rt, vMat.tensor, 0, slot.v, 0, elemCount);
    const kt = transpose(rt, k);
    let scores = matmul(rt, q, kt);
    scores = scale(rt, scores, invSqrtDk);
    scores = add(rt, scores, maskVar);
    const prob = softmax(rt, scores);
    const mixed = matmul(rt, prob, vMat);
    const contrib = matmul(rt, mixed, he);
    acc = acc === null ? contrib : add(rt, acc, contrib);
  }
  if (acc === null) {
    throw new Error('mhaForwardPrefill: nHeads must be >= 1');
  }
  return matmul(rt, acc, blk.wo);
}

function mhaForwardDecode(
  rt: TensorRuntime,
  blk: TransformerBlockParams,
  x: Variable,
  invSqrtDk: number,
  headExpand: readonly Variable[],
  layerIndex: number,
  cache: GptKVCache,
  rowWrite: number,
  captureProbs?: Variable[],
  captureDetail?: HeadDetailCapture,
): Variable {
  const H = blk.wq.length;
  const dK = blk.wq[0]?.tensor.shape[1];
  if (dK === undefined) {
    throw new Error('mhaForwardDecode: missing head weights');
  }
  const prefixRows = rowWrite + 1;
  let acc: Variable | null = null;
  for (let h = 0; h < H; h++) {
    const wqh = blk.wq[h];
    const wkh = blk.wk[h];
    const wvh = blk.wv[h];
    const he = headExpand[h];
    if (!wqh || !wkh || !wvh || !he) {
      throw new Error('mhaForwardDecode: missing head tensors');
    }
    const q = matmul(rt, x, wqh);
    const kNew = matmul(rt, x, wkh);
    const vNew = matmul(rt, x, wvh);
    const slot = cache.layers[layerIndex]?.heads[h];
    if (!slot) {
      throw new Error('mhaForwardDecode: cache slot missing');
    }
    copyContiguousElems(rt, kNew.tensor, 0, slot.k, rowWrite * dK, dK);
    copyContiguousElems(rt, vNew.tensor, 0, slot.v, rowWrite * dK, dK);
    // From here on attention reads K/V through the cache, so the fresh rows are
    // not part of the graph that leads to the logits and nothing else would
    // ever release them.
    destroy(kNew.tensor);
    destroy(vNew.tensor);
    const kPrefix = view2dRowPrefix(slot.k, prefixRows);
    const kt = transpose(rt, createLeafVariable(kPrefix, false));
    let scores = matmul(rt, q, kt);
    scores = scale(rt, scores, invSqrtDk);
    const prob = softmax(rt, scores);
    if (captureProbs !== undefined) {
      captureProbs.push(prob);
    }
    const vPrefix = view2dRowPrefix(slot.v, prefixRows);
    const mixed = matmul(rt, prob, createLeafVariable(vPrefix, false));
    if (captureDetail !== undefined) {
      captureDetail.queries.push(q);
      captureDetail.scores.push(scores);
      captureDetail.mixed.push(mixed);
    }
    const contrib = matmul(rt, mixed, he);
    acc = acc === null ? contrib : add(rt, acc, contrib);
  }
  if (acc === null) {
    throw new Error('mhaForwardDecode: nHeads must be >= 1');
  }
  return matmul(rt, acc, blk.wo);
}

function ffnForward(rt: TensorRuntime, blk: TransformerBlockParams, x: Variable): Variable {
  let h = matmul(rt, x, blk.w1);
  h = gelu(rt, h);
  return matmul(rt, h, blk.w2);
}

function blockForwardPrefill(
  rt: TensorRuntime,
  blk: TransformerBlockParams,
  x: Variable,
  maskVar: Variable,
  invSqrtDk: number,
  headExpand: readonly Variable[],
  layerIndex: number,
  cache: GptKVCache,
  seqLen: number,
): Variable {
  let y = rmsnorm(rt, x, blk.attnNormGamma);
  const attn = mhaForwardPrefill(
    rt,
    blk,
    y,
    maskVar,
    invSqrtDk,
    headExpand,
    layerIndex,
    cache,
    seqLen,
  );
  x = add(rt, x, attn);
  y = rmsnorm(rt, x, blk.ffnNormGamma);
  const ff = ffnForward(rt, blk, y);
  return add(rt, x, ff);
}

function blockForwardDecode(
  rt: TensorRuntime,
  blk: TransformerBlockParams,
  x: Variable,
  invSqrtDk: number,
  headExpand: readonly Variable[],
  layerIndex: number,
  cache: GptKVCache,
  rowWrite: number,
  captureProbs?: Variable[],
  captureParts?: { input: Variable; attn: Variable; ff: Variable }[],
  captureDetail?: HeadDetailCapture,
): Variable {
  const input = x;
  let y = rmsnorm(rt, x, blk.attnNormGamma);
  const attn = mhaForwardDecode(
    rt,
    blk,
    y,
    invSqrtDk,
    headExpand,
    layerIndex,
    cache,
    rowWrite,
    captureProbs,
    captureDetail,
  );
  x = add(rt, x, attn);
  y = rmsnorm(rt, x, blk.ffnNormGamma);
  const ff = ffnForward(rt, blk, y);
  captureParts?.push({ input, attn, ff });
  return add(rt, x, ff);
}

/**
 * Prompt forward: fills **`cache`** with per-head K/V for sequence length **`T`**, sets **`cache.seqLen = T`**.
 *
 * **`tokenIds`**: rank-1 **`u32`**, length **`T`**; positions **`0 … T-1`**.
 */
export function forwardGptPrefill(
  rt: TensorRuntime,
  model: GptModel,
  tokenIds: Tensor,
  cache: GptKVCache,
): Variable {
  if (tokenIds.dtype !== 'u32' || tokenIds.shape.length !== 1) {
    throw new Error('forwardGptPrefill: tokenIds must be a rank-1 u32 tensor');
  }
  const T = tokenIds.shape[0];
  if (T === undefined || T === 0) {
    throw new Error('forwardGptPrefill: tokenIds must be non-empty');
  }
  if (T > model.config.maxSeqLen) {
    throw new Error(
      `forwardGptPrefill: sequence length ${String(T)} exceeds maxSeqLen ${String(model.config.maxSeqLen)}`,
    );
  }
  if (cache.seqLen !== 0) {
    throw new Error('forwardGptPrefill: expected empty KV cache (seqLen === 0)');
  }
  if (cache.layers.length !== model.blocks.length) {
    throw new Error('forwardGptPrefill: cache does not match model layer count');
  }

  const tokIdxVar = createLeafVariable(tokenIds, false);
  const posHost = new Uint32Array(T);
  for (let i = 0; i < T; i++) {
    posHost[i] = i;
  }
  const posTensor = createIndices(rt, posHost, [T]);
  const posVar = createLeafVariable(posTensor, false);

  let h: Variable = add(
    rt,
    embedding(rt, model.tokenEmbed, tokIdxVar),
    embedding(rt, model.posEmbed, posVar),
  );

  const maskTensor = buildCausalBiasTensor(rt, T, -1e4, floatDtypeOf(model.tokenEmbed.tensor));
  const maskVar = createLeafVariable(maskTensor, false);
  const dK = model.config.dModel / model.config.nHeads;
  const invSqrtDk = 1 / Math.sqrt(dK);

  for (let li = 0; li < model.blocks.length; li++) {
    const blk = model.blocks[li];
    if (!blk) {
      throw new Error('forwardGptPrefill: missing block');
    }
    h = blockForwardPrefill(rt, blk, h, maskVar, invSqrtDk, model.headExpand, li, cache, T);
  }

  cache.seqLen = T;
  h = rmsnorm(rt, h, model.finalNormGamma);
  return matmul(rt, h, model.lmHead);
}

/**
 * Single-token decode: reads/writes cache rows at **`cache.seqLen`**, then increments **`cache.seqLen`**.
 *
 * **Position** is **`cache.seqLen`** (0-based index of this token). Requires **`cache.seqLen < maxSeqLen`**.
 */
export async function forwardGptDecodeStep(
  rt: TensorRuntime,
  model: GptModel,
  tokenId: number,
  cache: GptKVCache,
  options?: ForwardGptDecodeStepOptions,
): Promise<GptDecodeStepResult> {
  if (!Number.isInteger(tokenId) || tokenId < 0 || tokenId >= model.config.vocabSize) {
    throw new Error(`forwardGptDecodeStep: tokenId out of range (got ${String(tokenId)})`);
  }
  const pos = cache.seqLen;
  if (pos >= model.config.maxSeqLen) {
    throw new Error('forwardGptDecodeStep: sequence would exceed maxSeqLen');
  }
  const rowWrite = pos;

  const tokTensor = createIndices(rt, new Uint32Array([tokenId]), [1]);
  const posTensor = createIndices(rt, new Uint32Array([pos]), [1]);
  const tokVar = createLeafVariable(tokTensor, false);
  const posVar = createLeafVariable(posTensor, false);

  const tokenRow = embedding(rt, model.tokenEmbed, tokVar);
  const positionRow = embedding(rt, model.posEmbed, posVar);
  let h: Variable = add(rt, tokenRow, positionRow);

  const dK = model.config.dModel / model.config.nHeads;
  const invSqrtDk = 1 / Math.sqrt(dK);
  const captureLayer = options?.attentionCapture?.layerIndex;
  if (captureLayer !== undefined) {
    if (
      !Number.isInteger(captureLayer) ||
      captureLayer < 0 ||
      captureLayer >= model.blocks.length
    ) {
      throw new Error(
        `forwardGptDecodeStep: attentionCapture.layerIndex out of range (got ${String(captureLayer)}, nLayers=${String(model.blocks.length)})`,
      );
    }
  }
  const lesson = options?.lessonTrace === true;
  const traceAll = options?.flowTrace === true || lesson;
  const headDetails: HeadDetailCapture[] | undefined = lesson ? [] : undefined;
  const probsByLayer: (Variable[] | undefined)[] = [];
  const blockParts: { input: Variable; attn: Variable; ff: Variable }[] | undefined = traceAll
    ? []
    : undefined;
  // Lens branches hang off the residual stream but are not ancestors of the
  // final logits, so the graph disposal never reaches them; they are released
  // here once copied out.
  const lensRows: { readonly normed: Variable; readonly logits: Variable }[] | undefined =
    options?.layerLens === true || traceAll ? [] : undefined;

  for (let li = 0; li < model.blocks.length; li++) {
    const blk = model.blocks[li];
    if (!blk) {
      throw new Error('forwardGptDecodeStep: missing block');
    }
    const probsForLayer: Variable[] | undefined = traceAll || captureLayer === li ? [] : undefined;
    probsByLayer.push(probsForLayer);
    if (lensRows !== undefined && li > 0) {
      lensRows.push(lensLogits(rt, model, h));
    }
    const detail: HeadDetailCapture | undefined =
      headDetails !== undefined ? { queries: [], scores: [], mixed: [] } : undefined;
    if (detail !== undefined) {
      headDetails?.push(detail);
    }
    h = blockForwardDecode(
      rt,
      blk,
      h,
      invSqrtDk,
      model.headExpand,
      li,
      cache,
      rowWrite,
      probsForLayer,
      blockParts,
      detail,
    );
  }
  const captureProbs = captureLayer !== undefined ? probsByLayer[captureLayer] : undefined;

  cache.seqLen = pos + 1;
  h = rmsnorm(rt, h, model.finalNormGamma);
  const logits = matmul(rt, h, model.lmHead);

  let attention: Float32Array | undefined;
  if (captureProbs !== undefined) {
    const prefixRows = rowWrite + 1;
    const { nHeads } = model.config;
    if (captureProbs.length !== nHeads) {
      throw new Error(
        `forwardGptDecodeStep: expected ${String(nHeads)} captured head probs, got ${String(captureProbs.length)}`,
      );
    }
    const dt = floatDtypeOf(model.tokenEmbed.tensor);
    const stack = createZeros(rt, [nHeads, prefixRows], dt);
    for (let hi = 0; hi < nHeads; hi++) {
      const prob = captureProbs[hi];
      if (!prob) {
        throw new Error('forwardGptDecodeStep: missing captured head prob');
      }
      copyContiguousElems(rt, prob.tensor, 0, stack, hi * prefixRows, prefixRows);
    }
    attention = await toArray(rt, stack);
    destroy(stack);
  }

  let layerLogits: Float32Array | undefined;
  if (lensRows !== undefined) {
    // lensRows holds blocks 0 … L-2 (captured before the next block ran); the
    // last block's lens is the real output, so reuse `logits` for that row.
    const V = model.config.vocabSize;
    const L = model.blocks.length;
    const stack = createZeros(rt, [L, V], floatDtypeOf(model.tokenEmbed.tensor));
    for (let li = 0; li < L - 1; li++) {
      const row = lensRows[li];
      if (!row) {
        throw new Error('forwardGptDecodeStep: missing lens row');
      }
      copyContiguousElems(rt, row.logits.tensor, 0, stack, li * V, V);
    }
    copyContiguousElems(rt, logits.tensor, 0, stack, (L - 1) * V, V);
    layerLogits = await toArray(rt, stack);
    destroy(stack);
    for (const row of lensRows) {
      destroy(row.normed.tensor);
      destroy(row.logits.tensor);
    }
  }

  let allAttention: Float32Array | undefined;
  let blockVectors: Float32Array | undefined;
  if (blockParts !== undefined) {
    const prefixRows = rowWrite + 1;
    const { nHeads, dModel } = model.config;
    const L = model.blocks.length;
    const dt = floatDtypeOf(model.tokenEmbed.tensor);
    const attnStack = createZeros(rt, [L * nHeads, prefixRows], dt);
    const vecStack = createZeros(rt, [3 * L, dModel], dt);
    for (let li = 0; li < L; li++) {
      const probs = probsByLayer[li];
      const parts = blockParts[li];
      if (!probs || probs.length !== nHeads || !parts) {
        throw new Error('forwardGptDecodeStep: incomplete flow trace capture');
      }
      for (let hi = 0; hi < nHeads; hi++) {
        const prob = probs[hi];
        if (!prob) {
          throw new Error('forwardGptDecodeStep: missing captured head prob');
        }
        const row = (li * nHeads + hi) * prefixRows;
        copyContiguousElems(rt, prob.tensor, 0, attnStack, row, prefixRows);
      }
      copyContiguousElems(rt, parts.input.tensor, 0, vecStack, 3 * li * dModel, dModel);
      copyContiguousElems(rt, parts.attn.tensor, 0, vecStack, (3 * li + 1) * dModel, dModel);
      copyContiguousElems(rt, parts.ff.tensor, 0, vecStack, (3 * li + 2) * dModel, dModel);
    }
    allAttention = await toArray(rt, attnStack);
    blockVectors = await toArray(rt, vecStack);
    destroy(attnStack);
    destroy(vecStack);
  }

  let inputVectors: Float32Array | undefined;
  let allQueries: Float32Array | undefined;
  let allScores: Float32Array | undefined;
  let allMixed: Float32Array | undefined;
  if (headDetails !== undefined) {
    const prefixRows = rowWrite + 1;
    const { nHeads, dModel } = model.config;
    const L = model.blocks.length;
    const dt = floatDtypeOf(model.tokenEmbed.tensor);
    const inputStack = createZeros(rt, [2, dModel], dt);
    copyContiguousElems(rt, tokenRow.tensor, 0, inputStack, 0, dModel);
    copyContiguousElems(rt, positionRow.tensor, 0, inputStack, dModel, dModel);
    const queryStack = createZeros(rt, [L * nHeads, dK], dt);
    const scoreStack = createZeros(rt, [L * nHeads, prefixRows], dt);
    const mixedStack = createZeros(rt, [L * nHeads, dK], dt);
    for (let li = 0; li < L; li++) {
      const detail = headDetails[li];
      if (
        !detail ||
        detail.queries.length !== nHeads ||
        detail.scores.length !== nHeads ||
        detail.mixed.length !== nHeads
      ) {
        throw new Error('forwardGptDecodeStep: incomplete lesson trace capture');
      }
      for (let hi = 0; hi < nHeads; hi++) {
        const q = detail.queries[hi];
        const sc = detail.scores[hi];
        const mx = detail.mixed[hi];
        if (!q || !sc || !mx) {
          throw new Error('forwardGptDecodeStep: missing captured head detail');
        }
        const row = li * nHeads + hi;
        copyContiguousElems(rt, q.tensor, 0, queryStack, row * dK, dK);
        copyContiguousElems(rt, sc.tensor, 0, scoreStack, row * prefixRows, prefixRows);
        copyContiguousElems(rt, mx.tensor, 0, mixedStack, row * dK, dK);
      }
    }
    inputVectors = await toArray(rt, inputStack);
    allQueries = await toArray(rt, queryStack);
    allScores = await toArray(rt, scoreStack);
    allMixed = await toArray(rt, mixedStack);
    destroy(inputStack);
    destroy(queryStack);
    destroy(scoreStack);
    destroy(mixedStack);
  }

  return {
    logits,
    attention,
    layerLogits,
    allAttention,
    blockVectors,
    inputVectors,
    allQueries,
    allScores,
    allMixed,
  };
}

/**
 * Read one head's cached keys and values: row-major **`[cache.seqLen, d_k]`**
 * each, row **`i`** belonging to the token at position **`i`**. These are the
 * rows attention compares the query against (keys) and mixes (values).
 */
export async function readKvCacheHead(
  rt: TensorRuntime,
  cache: GptKVCache,
  layerIndex: number,
  headIndex: number,
): Promise<{ readonly keys: Float32Array; readonly values: Float32Array; readonly dK: number }> {
  const layer = cache.layers[layerIndex];
  if (!Number.isInteger(layerIndex) || layer === undefined) {
    throw new Error(
      `readKvCacheHead: layerIndex out of range (got ${String(layerIndex)}, nLayers=${String(cache.layers.length)})`,
    );
  }
  const slot = layer.heads[headIndex];
  if (!Number.isInteger(headIndex) || slot === undefined) {
    throw new Error(
      `readKvCacheHead: headIndex out of range (got ${String(headIndex)}, nHeads=${String(layer.heads.length)})`,
    );
  }
  const dK = slot.k.shape[1];
  if (dK === undefined) {
    throw new Error('readKvCacheHead: cache tensor missing head dimension');
  }
  const elems = cache.seqLen * dK;
  const keys = (await toArray(rt, slot.k)).slice(0, elems);
  const values = (await toArray(rt, slot.v)).slice(0, elems);
  return { keys, values, dK };
}
