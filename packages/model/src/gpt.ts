import {
  add,
  createLeafVariable,
  createParameter,
  createParameterWithMaster,
  embedding,
  gelu,
  matmul,
  rmsnorm,
  scale,
  softmax,
  transpose,
  type Parameter,
  type Variable,
} from '@llm-transformer-diy/autograd';
import {
  createFromArray,
  createIndices,
  floatDtypeOf,
  type FloatDType,
  type Tensor,
  type TensorRuntime,
} from '@llm-transformer-diy/tensor';

import { assertValidGptConfig, type GptConfig } from './config';

/** One transformer block (Pre-LN): MHA + FFN, separate RMSNorm γ vectors. */
export type TransformerBlockParams = {
  readonly attnNormGamma: Parameter;
  readonly ffnNormGamma: Parameter;
  readonly wq: readonly Parameter[];
  readonly wk: readonly Parameter[];
  readonly wv: readonly Parameter[];
  readonly wo: Parameter;
  readonly w1: Parameter;
  readonly w2: Parameter;
};

export type CreateGptModelOptions = {
  /**
   * Storage dtype for every trainable **`Parameter`** and the constant **`headExpand`**
   * gather matrices. Defaults to **`f32`** (legacy behavior).
   */
  readonly dtype?: FloatDType;
  /**
   * PRNG seed for weight initialization (Box–Muller Gaussian, see {@link GPT_INIT_STD}).
   * Defaults to a fixed constant so training runs are deterministic; override
   * to draw a different initial weight sample for the same architecture.
   */
  readonly initSeed?: number;
};

/** Default {@link CreateGptModelOptions.initSeed} — fixed for repro across runs. */
export const DEFAULT_GPT_INIT_SEED = 0xc0ffee;

/** Trainable weight with stable checkpoint **name** and its compute **dtype**. */
export type GptParameterInfo = {
  readonly name: string;
  readonly parameter: Parameter;
  readonly dtype: FloatDType;
};

export type GptModel = {
  readonly config: GptConfig;
  readonly tokenEmbed: Parameter;
  readonly posEmbed: Parameter;
  readonly blocks: readonly TransformerBlockParams[];
  readonly finalNormGamma: Parameter;
  readonly lmHead: Parameter;
  /** Constant `{@link Variable}`s: per-head `[d_k, d_model]` gather matrices (see design notes). */
  readonly headExpand: readonly Variable[];
};

function paramFromInit(
  rt: TensorRuntime,
  data: Float32Array,
  shape: readonly number[],
  dtype: FloatDType,
): Parameter {
  if (dtype === 'f32') {
    return createParameter(createFromArray(rt, data, shape, dtype));
  }
  return createParameterWithMaster(rt, data, shape, dtype);
}

/**
 * Splitmix64-style 32-bit PRNG. Cheap, fast, statistically adequate for weight
 * initialization (this is *not* a cryptographic RNG). Deterministic given the
 * seed so training runs are reproducible.
 */
function createPrng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x9e3779b9) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 16), 0x85ebca6b) >>> 0;
    t = Math.imul(t ^ (t >>> 13), 0xc2b2ae35) >>> 0;
    t = (t ^ (t >>> 16)) >>> 0;
    return t / 0x1_0000_0000;
  };
}

/**
 * Standard deviation shared by every weight matrix at init (the GPT-2 recipe).
 *
 * With Pre-LN blocks the input of every matmul has unit RMS, so a projection
 * `x @ W` with `W ~ N(0, σ²)` produces activations of std `σ·√fanIn`. At
 * `σ = 0.02` and `dModel` in the hundreds that is below 1, which keeps the
 * attention scores `q·k / √dK` near zero: softmax starts close to **uniform**
 * and every position receives gradient from step 0.
 */
export const GPT_INIT_STD = 0.02;

/**
 * Extra shrink for the two matrices that write into the residual stream
 * (`wo`, `w2`): `1 / √(2 · nLayers)`. Each block adds two such outputs, so
 * without the shrink the residual variance grows linearly with depth.
 */
export function residualOutInitStd(nLayers: number): number {
  return GPT_INIT_STD / Math.sqrt(2 * nLayers);
}

/**
 * Gaussian `N(0, std²)` draw (Box–Muller) from a shared {@link createPrng}
 * stream. The shared stream is what breaks symmetry across heads and across
 * **Q/K/V** within a head — every call advances the PRNG, so two `wq` heads
 * receive *independent* values even though they have identical shapes.
 *
 * History, kept because both mistakes are instructive:
 *
 * 1. The first initializer used a deterministic `sin(i · 0.13)` pattern for
 *    every matrix. All heads started identical and **Q = K = V** within a head;
 *    training plateaued near the unigram loss for hundreds of steps.
 * 2. The second used `std = 1 / √cols`, meant as fan-in scaling. Weights here
 *    are laid out `[in, out]` and applied as `x @ W`, so `cols` is the
 *    **fan-out**. For `wq`/`wk` (`[dModel, dK]`) that gave `std = 1/√dK`
 *    (0.125 at `dK = 64`): attention scores started with a standard deviation
 *    around 8, so softmax was nearly one-hot on a random position, and each
 *    attention block wrote an output ~50× larger than the token embedding into
 *    the residual stream. Training only survived with a tiny learning rate.
 */
function gaussianInit(prng: () => number, rows: number, cols: number, std: number): Float32Array {
  const a = new Float32Array(rows * cols);
  for (let i = 0; i < a.length; i++) {
    // Box–Muller. Clamp u1 away from 0 so log() stays finite.
    const u1 = Math.max(1e-7, prng());
    const u2 = prng();
    const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    a[i] = z * std;
  }
  return a;
}

function headExpandMatrix(h: number, _nHeads: number, dK: number, dModel: number): Float32Array {
  const data = new Float32Array(dK * dModel);
  const col0 = h * dK;
  for (let i = 0; i < dK; i++) {
    data[i * dModel + col0 + i] = 1;
  }
  return data;
}

/** Strict upper triangle = **`maskValue`**, else **0** (for causal attention bias). */
export function buildCausalBiasTensor(
  rt: TensorRuntime,
  seqLen: number,
  maskValue = -1e4,
  dtype: FloatDType = 'f32',
): Tensor {
  const data = new Float32Array(seqLen * seqLen);
  for (let i = 0; i < seqLen; i++) {
    for (let j = 0; j < seqLen; j++) {
      data[i * seqLen + j] = j > i ? maskValue : 0;
    }
  }
  return createFromArray(rt, data, [seqLen, seqLen], dtype);
}

export function createGptModel(
  rt: TensorRuntime,
  config: GptConfig,
  options?: CreateGptModelOptions,
): GptModel {
  assertValidGptConfig(config);
  const dtype = options?.dtype ?? 'f32';
  const initSeed = options?.initSeed ?? DEFAULT_GPT_INIT_SEED;
  const prng = createPrng(initSeed);
  const { vocabSize, maxSeqLen, dModel, nHeads, dFf, nLayers } = config;
  const dK = dModel / nHeads;
  const residualStd = residualOutInitStd(nLayers);

  const tokenEmbed = paramFromInit(
    rt,
    gaussianInit(prng, vocabSize, dModel, GPT_INIT_STD),
    [vocabSize, dModel],
    dtype,
  );
  const posEmbed = paramFromInit(
    rt,
    gaussianInit(prng, maxSeqLen, dModel, GPT_INIT_STD),
    [maxSeqLen, dModel],
    dtype,
  );

  const headExpand: Variable[] = [];
  for (let h = 0; h < nHeads; h++) {
    const t = createFromArray(rt, headExpandMatrix(h, nHeads, dK, dModel), [dK, dModel], dtype);
    headExpand.push(createLeafVariable(t, false));
  }

  const blocks: TransformerBlockParams[] = [];
  for (let l = 0; l < nLayers; l++) {
    const attnNormGamma = paramFromInit(rt, new Float32Array(dModel).fill(1), [dModel], dtype);
    const ffnNormGamma = paramFromInit(rt, new Float32Array(dModel).fill(1), [dModel], dtype);
    const wq: Parameter[] = [];
    const wk: Parameter[] = [];
    const wv: Parameter[] = [];
    for (let h = 0; h < nHeads; h++) {
      wq.push(paramFromInit(rt, gaussianInit(prng, dModel, dK, GPT_INIT_STD), [dModel, dK], dtype));
      wk.push(paramFromInit(rt, gaussianInit(prng, dModel, dK, GPT_INIT_STD), [dModel, dK], dtype));
      wv.push(paramFromInit(rt, gaussianInit(prng, dModel, dK, GPT_INIT_STD), [dModel, dK], dtype));
    }
    const wo = paramFromInit(
      rt,
      gaussianInit(prng, dModel, dModel, residualStd),
      [dModel, dModel],
      dtype,
    );
    const w1 = paramFromInit(
      rt,
      gaussianInit(prng, dModel, dFf, GPT_INIT_STD),
      [dModel, dFf],
      dtype,
    );
    const w2 = paramFromInit(
      rt,
      gaussianInit(prng, dFf, dModel, residualStd),
      [dFf, dModel],
      dtype,
    );
    blocks.push({
      attnNormGamma,
      ffnNormGamma,
      wq,
      wk,
      wv,
      wo,
      w1,
      w2,
    });
  }

  const finalNormGamma = paramFromInit(rt, new Float32Array(dModel).fill(1), [dModel], dtype);
  const lmHeadData = gaussianInit(prng, dModel, vocabSize, GPT_INIT_STD);
  const lmHead = paramFromInit(rt, lmHeadData, [dModel, vocabSize], dtype);

  return {
    config,
    tokenEmbed,
    posEmbed,
    blocks,
    finalNormGamma,
    lmHead,
    headExpand,
  };
}

function mhaForward(
  rt: TensorRuntime,
  blk: TransformerBlockParams,
  x: Variable,
  maskVar: Variable,
  invSqrtDk: number,
  headExpand: readonly Variable[],
): Variable {
  const H = blk.wq.length;
  let acc: Variable | null = null;
  for (let h = 0; h < H; h++) {
    const wqh = blk.wq[h];
    const wkh = blk.wk[h];
    const wvh = blk.wv[h];
    const he = headExpand[h];
    if (!wqh || !wkh || !wvh || !he) {
      throw new Error('mhaForward: missing head tensors');
    }
    const q = matmul(rt, x, wqh);
    const k = matmul(rt, x, wkh);
    const vMat = matmul(rt, x, wvh);
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
    throw new Error('mhaForward: nHeads must be >= 1');
  }
  return matmul(rt, acc, blk.wo);
}

function ffnForward(rt: TensorRuntime, blk: TransformerBlockParams, x: Variable): Variable {
  let h = matmul(rt, x, blk.w1);
  h = gelu(rt, h);
  return matmul(rt, h, blk.w2);
}

function blockForward(
  rt: TensorRuntime,
  blk: TransformerBlockParams,
  x: Variable,
  maskVar: Variable,
  invSqrtDk: number,
  headExpand: readonly Variable[],
): Variable {
  let y = rmsnorm(rt, x, blk.attnNormGamma);
  const attn = mhaForward(rt, blk, y, maskVar, invSqrtDk, headExpand);
  x = add(rt, x, attn);
  y = rmsnorm(rt, x, blk.ffnNormGamma);
  const ff = ffnForward(rt, blk, y);
  return add(rt, x, ff);
}

/**
 * Decoder-only GPT forward: **`tokenIds`** shape **`[seqLen]`** (`u32`).
 * Position indices are **`0 … seqLen-1`** (must be `< config.maxSeqLen`).
 */
export function forwardGpt(rt: TensorRuntime, model: GptModel, tokenIds: Tensor): Variable {
  if (tokenIds.dtype !== 'u32' || tokenIds.shape.length !== 1) {
    throw new Error('forwardGpt: tokenIds must be a rank-1 u32 tensor');
  }
  const T = tokenIds.shape[0];
  if (T === undefined) {
    throw new Error('forwardGpt: tokenIds must be non-empty');
  }
  if (T > model.config.maxSeqLen) {
    throw new Error(
      `forwardGpt: sequence length ${String(T)} exceeds maxSeqLen ${String(model.config.maxSeqLen)}`,
    );
  }

  const tokIdxVar = createLeafVariable(tokenIds, false);
  const posHost = new Uint32Array(T);
  for (let i = 0; i < T; i++) {
    posHost[i] = i;
  }
  const posTensor = createIndices(rt, posHost, [T]);
  const posVar = createLeafVariable(posTensor, false);

  let h = add(
    rt,
    embedding(rt, model.tokenEmbed, tokIdxVar),
    embedding(rt, model.posEmbed, posVar),
  );

  const maskTensor = buildCausalBiasTensor(rt, T, -1e4, floatDtypeOf(model.tokenEmbed.tensor));
  const maskVar = createLeafVariable(maskTensor, false);
  const dK = model.config.dModel / model.config.nHeads;
  const invSqrtDk = 1 / Math.sqrt(dK);

  for (const blk of model.blocks) {
    h = blockForward(rt, blk, h, maskVar, invSqrtDk, model.headExpand);
  }

  h = rmsnorm(rt, h, model.finalNormGamma);
  return matmul(rt, h, model.lmHead);
}

/** All trainable parameters for **`registerParameter`** / optimizers, with dtypes on **`tensor`**. */
export function collectGptParameters(model: GptModel): Parameter[] {
  const out: Parameter[] = [model.tokenEmbed, model.posEmbed, model.finalNormGamma, model.lmHead];
  for (const b of model.blocks) {
    out.push(b.attnNormGamma, b.ffnNormGamma, b.wo, b.w1, b.w2);
    out.push(...b.wq, ...b.wk, ...b.wv);
  }
  return out;
}

/**
 * Lexically stable tensor keys matching **{@link collectGptParameters}** order for checkpoints.
 */
export function gptParameterTensorNames(model: GptModel): string[] {
  const names: string[] = ['token_embed', 'pos_embed', 'final_norm_gamma', 'lm_head'];
  let L = 0;
  for (const b of model.blocks) {
    const p = `L${String(L)}`;
    names.push(`${p}_attn_norm_gamma`, `${p}_ffn_norm_gamma`, `${p}_wo`, `${p}_w1`, `${p}_w2`);
    for (let h = 0; h < b.wq.length; h++) {
      names.push(`${p}_wq_${String(h)}`);
    }
    for (let h = 0; h < b.wk.length; h++) {
      names.push(`${p}_wk_${String(h)}`);
    }
    for (let h = 0; h < b.wv.length; h++) {
      names.push(`${p}_wv_${String(h)}`);
    }
    L += 1;
  }
  return names;
}

/**
 * Ordered **(name, parameter, dtype)** triples for checkpoints / telemetry (**`headExpand`** excluded).
 * **`dtype`** matches **`parameter.tensor.dtype`** (always a float dtype).
 */
export function listGptParameterInfos(model: GptModel): readonly GptParameterInfo[] {
  const names = gptParameterTensorNames(model);
  const params = collectGptParameters(model);
  if (names.length !== params.length) {
    throw new Error(
      `listGptParameterInfos: name/param mismatch (${String(names.length)} vs ${String(params.length)})`,
    );
  }
  const out: GptParameterInfo[] = [];
  for (let i = 0; i < names.length; i++) {
    const name = names[i];
    const parameter = params[i];
    if (name === undefined || parameter === undefined) {
      throw new Error('listGptParameterInfos: internal index error');
    }
    out.push({ name, parameter, dtype: floatDtypeOf(parameter.tensor) });
  }
  return out;
}
