import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';

export function loadTrivialFillWgsl(): string {
  const url = new URL('./trivial-fill.wgsl', import.meta.url);
  return readFileSync(fileURLToPath(url), 'utf8');
}
