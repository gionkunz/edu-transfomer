import { readFileSync } from 'node:fs';
import path from 'node:path';

export function loadTrivialFillWgsl(): string {
  return readFileSync(path.join(__dirname, 'trivial-fill.wgsl'), 'utf8');
}
