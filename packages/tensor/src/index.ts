/**
 * `@llm-transformer-diy/tensor` — functional tensor layer.
 *
 * Public surface is **data + free functions**. There are no classes, no methods
 * on `Tensor`, and no module-level mutable state (no global default runtime).
 * Every function that allocates or dispatches takes a {@link TensorRuntime} as
 * its first argument; {@link reshape} is a pure metadata transform and takes none.
 */

/**
 * Layer marker retained for the `@llm-transformer-diy/autograd` stub's dependency-edge
 * import. Will be removed once autograd uses real tensor types.
 */
export const TENSOR_LAYER_MARKER: unique symbol = Symbol.for('@llm-transformer-diy/tensor');

export type { Device, DType, Tensor, TensorRuntime } from './types';
export {
  packBf16FromFloat32,
  packFp16FromFloat32,
  unpackBf16ToFloat32,
  unpackFp16ToFloat32,
} from '@llm-transformer-diy/gpu';
export {
  assertDtype,
  assertFloatDtype,
  assertRank,
  assertSameDtype,
  assertSameShape,
  bytesPerElement,
  computeContiguousStrides,
  floatDtypeOf,
  isContiguous,
  productOfShape,
  sameShape,
  type FloatDType,
} from './shape';
export {
  cast,
  copyContiguousElems,
  copyContiguousTensorBytes,
  createIndices,
  createFromArray,
  createZeros,
  destroy,
  gpuStorageByteLength,
  makeIndicesTensor,
  readIndices,
  toArray,
} from './tensor';
export {
  add,
  adamWStepInPlace,
  broadcastFillFromScalar,
  copyF32Elems,
  crossEntropyLogitsGrad,
  crossEntropyRowsFromLogits,
  embedding,
  gelu,
  geluBackward,
  matmul,
  mul,
  reshape,
  rmsnorm,
  rmsnormGradGamma,
  rmsnormGradInput,
  scale,
  scatterEmbeddingAccum,
  softmax,
  softmaxBackward,
  sumFlat,
  transpose,
  view2dRowPrefix,
} from './ops';
export type { AdamWStepHyperparams } from './ops';
