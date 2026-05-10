import path from 'node:path';

import { glob as globPatterns } from 'glob';

/** Discover corpus/dataset paths from one or more glob patterns (deterministic sorted order, deduped). */
export async function discoverTrainingFiles(
  patterns: string | readonly string[],
  cwd: string = process.cwd(),
): Promise<string[]> {
  const list = typeof patterns === 'string' ? [patterns] : [...patterns];
  if (list.length === 0) {
    throw new Error('discoverTrainingFiles: at least one glob pattern is required');
  }

  const matches = (
    await Promise.all(
      list.map((pattern) =>
        globPatterns(pattern, {
          nodir: true,
          nocase: false,
          follow: false,
          dot: false,
          absolute: false,
          cwd,
        }),
      ),
    )
  ).flat();

  const unique = [...new Set(matches)].sort((a, b) => a.localeCompare(b));
  return unique.map((rel) => path.normalize(rel));
}
