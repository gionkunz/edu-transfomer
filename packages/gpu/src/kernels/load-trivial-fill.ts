import { readFileSync } from 'node:fs';

import { resolveWgslPath, rewriteMainForLargeDispatch } from './load-core-wgsl';

export function loadTrivialFillWgsl(): string {
  return rewriteMainForLargeDispatch(readFileSync(resolveWgslPath('trivial-fill.wgsl'), 'utf8'));
}
