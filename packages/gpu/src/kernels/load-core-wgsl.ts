import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

function loadWgsl(relativePath: string): string {
  const url = new URL(relativePath, import.meta.url);
  return readFileSync(fileURLToPath(url), 'utf8');
}

export function loadMatMulWgsl(): string {
  return loadWgsl('./matmul.wgsl');
}

export function loadTransposeWgsl(): string {
  return loadWgsl('./transpose.wgsl');
}

export function loadElementwiseAddWgsl(): string {
  return loadWgsl('./elementwise-add.wgsl');
}

export function loadElementwiseMulWgsl(): string {
  return loadWgsl('./elementwise-mul.wgsl');
}

export function loadElementwiseScaleWgsl(): string {
  return loadWgsl('./elementwise-scale.wgsl');
}

export function loadElementwiseGeluWgsl(): string {
  return loadWgsl('./elementwise-gelu.wgsl');
}
