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
export { GpuContext, probeWebGpu, type GpuBootstrapOptions } from './device';
export { KernelRegistry } from './kernel-registry';
export {
  createShaderModule,
  createSimpleComputePipeline,
  type SimpleComputePipelineArgs,
} from './pipeline';
