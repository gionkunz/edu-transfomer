import './webgpu-globals';

/** Symbol retained so higher-layer stubs can assert a dependency edge on `gpu` without importing WebGPU types. */
export const GPU_LAYER_MARKER = Symbol.for('@llm-transformer-diy/gpu');

export {
  createFloat32StorageBuffer,
  float32ByteLength,
  readFloat32Buffer,
  writeFloat32Buffer,
} from './buffer';
export {
  BOOTSTRAP_KERNEL_FILL_PI,
  dispatchFillPi,
  registerBootstrapKernels,
  smokeFillPiBuffer,
} from './bootstrap-kernels';
export {
  CORE_KERNEL_ADD,
  CORE_KERNEL_GELU,
  CORE_KERNEL_IDS,
  CORE_KERNEL_MATMUL,
  CORE_KERNEL_MUL,
  CORE_KERNEL_REDUCE_MAX,
  CORE_KERNEL_REDUCE_SUM,
  CORE_KERNEL_RMSNORM,
  CORE_KERNEL_SCALE,
  CORE_KERNEL_SOFTMAX,
  CORE_KERNEL_TRANSPOSE,
  dispatchAdd,
  dispatchGelu,
  dispatchMatMul,
  dispatchMul,
  dispatchReduceMax,
  dispatchReduceSum,
  dispatchRmsNorm,
  dispatchScale,
  dispatchSoftmax,
  dispatchTranspose,
  registerCoreKernels,
} from './core-kernels';
export {
  cpuAdd,
  cpuGelu,
  cpuMatMul,
  cpuMul,
  cpuReduceMax,
  cpuReduceSum,
  cpuRmsNorm,
  cpuScale,
  cpuSoftmax,
  cpuTranspose,
  geluScalar,
  GELU_CUBE_COEF,
  GELU_TANH_COEF,
} from './cpu-reference';
export { GpuContext, probeWebGpu, type GpuBootstrapOptions } from './device';
export { KernelRegistry } from './kernel-registry';
export {
  createShaderModule,
  createSimpleComputePipeline,
  type SimpleComputePipelineArgs,
} from './pipeline';
