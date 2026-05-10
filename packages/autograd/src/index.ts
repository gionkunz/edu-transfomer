/**
 * `@llm-transformer-diy/autograd` — define-by-run computation graph.
 *
 * Public surface is **data + free functions** built on top of the
 * `@llm-transformer-diy/tensor` functional surface. There are no classes, no
 * methods on `Variable`, and no module-level mutable state besides a
 * process-monotonic id counter scoped to `./variable`. The single intentional
 * mutability is `Variable.grad`, which is written only by `accumulateGrad`
 * (during backward) and `zeroGrad` (between training steps).
 *
 * Layering: `autograd → tensor → gpu`. This package does not import from
 * `@llm-transformer-diy/gpu` directly.
 */

import { TENSOR_LAYER_MARKER } from '@llm-transformer-diy/tensor';

/**
 * Layer marker retained for the `@llm-transformer-diy/model` stub's
 * dependency-edge import. Will be removed once a downstream consumer is
 * migrated off it.
 */
export const AUTOGRAD_LAYER_MARKER: unique symbol = Symbol.for('@llm-transformer-diy/autograd');

/**
 * Layer-edge probe retained for the `@llm-transformer-diy/model` stub. The
 * function exists only so the model package's stub can verify that the
 * autograd → tensor edge resolves; will be removed alongside
 * {@link AUTOGRAD_LAYER_MARKER} once consumers switch to the real surface.
 */
export function __autogradDependsOnTensor(): typeof TENSOR_LAYER_MARKER {
  return TENSOR_LAYER_MARKER;
}

export type { Operation, Parameter, ParameterRegistry, Variable } from './types';
export { accumulateGrad, createLeafVariable, createParameter } from './variable';
export {
  clipGradNorm,
  createParameterRegistry,
  parameters,
  registerParameter,
  totalParams,
  zeroGrad,
} from './parameter-registry';
export {
  add,
  crossEntropyMeanFromLogits,
  crossEntropyRowsFromLogits,
  embedding,
  gelu,
  matmul,
  mul,
  reshape,
  scale,
  softmax,
  sumFlat,
  transpose,
} from './ops';
export { backward, topologicalOrder } from './backward';
