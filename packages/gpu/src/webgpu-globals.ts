import { globals } from 'webgpu';

/**
 * WebGPU exposes `GPUBufferUsage`, `GPUShaderStage`, etc. on `globalThis` in browsers.
 * node-webgpu provides them via `globals`; install once for the process.
 */
Object.assign(globalThis, globals as object);
