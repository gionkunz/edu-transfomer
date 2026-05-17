import { readFileSync } from 'node:fs';
import path from 'node:path';

import { rewriteMainForLargeDispatch } from './load-core-wgsl';

export function loadTrivialFillWgsl(): string {
  return rewriteMainForLargeDispatch(
    readFileSync(path.join(__dirname, 'trivial-fill.wgsl'), 'utf8'),
  );
}
