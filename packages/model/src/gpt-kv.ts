import {
  add,
  createLeafVariable,
  embedding,
  gelu,
  matmul,
  rmsnorm,
  scale,
  softmax,
  transpose,
  type Variable,
} from '@llm-transformer-diy/autograd';
import {
  copyContiguousElems,
  createIndices,
  createZeros,
  destroy,
  floatDtypeOf,
  view2dRowPrefix,
  type Tensor,
  type TensorRuntime,
} from '@llm-transformer-diy/tensor';

import { buildCausalBiasTensor, type GptModel, type TransformerBlockParams } from './gpt';

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
    const kPrefix = view2dRowPrefix(slot.k, prefixRows);
    const kt = transpose(rt, createLeafVariable(kPrefix, false));
    let scores = matmul(rt, q, kt);
    scores = scale(rt, scores, invSqrtDk);
    const prob = softmax(rt, scores);
    const vPrefix = view2dRowPrefix(slot.v, prefixRows);
    const mixed = matmul(rt, prob, createLeafVariable(vPrefix, false));
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
): Variable {
  let y = rmsnorm(rt, x, blk.attnNormGamma);
  const attn = mhaForwardDecode(rt, blk, y, invSqrtDk, headExpand, layerIndex, cache, rowWrite);
  x = add(rt, x, attn);
  y = rmsnorm(rt, x, blk.ffnNormGamma);
  const ff = ffnForward(rt, blk, y);
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
export function forwardGptDecodeStep(
  rt: TensorRuntime,
  model: GptModel,
  tokenId: number,
  cache: GptKVCache,
): Variable {
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

  let h: Variable = add(
    rt,
    embedding(rt, model.tokenEmbed, tokVar),
    embedding(rt, model.posEmbed, posVar),
  );

  const dK = model.config.dModel / model.config.nHeads;
  const invSqrtDk = 1 / Math.sqrt(dK);

  for (let li = 0; li < model.blocks.length; li++) {
    const blk = model.blocks[li];
    if (!blk) {
      throw new Error('forwardGptDecodeStep: missing block');
    }
    h = blockForwardDecode(rt, blk, h, invSqrtDk, model.headExpand, li, cache, rowWrite);
  }

  cache.seqLen = pos + 1;
  h = rmsnorm(rt, h, model.finalNormGamma);
  return matmul(rt, h, model.lmHead);
}
