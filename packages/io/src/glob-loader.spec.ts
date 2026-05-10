import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';

import { describe, expect, it } from 'vitest';

import { discoverTrainingFiles } from './glob-loader';

describe('discoverTrainingFiles', () => {
  it('matches nested text files across multiple patterns and dedupes', async () => {
    const dir = await mkdtemp(path.join(tmpdir(), 'io-glob-'));

    await mkdir(path.join(dir, 'a', 'nested'), { recursive: true });
    await writeFile(path.join(dir, 'a', 'nested', 'hello.txt'), 'hello', 'utf8');
    await writeFile(path.join(dir, 'readme.md'), '-', 'utf8');

    try {
      const hits = await discoverTrainingFiles(
        ['**/*.txt', '**/*.txt', '**/missing/**/*.txt'],
        dir,
      );
      expect(hits).toEqual([path.normalize('a/nested/hello.txt')]);

      const all = await discoverTrainingFiles(['**/*'], dir);
      expect(all).toContain(path.normalize('a/nested/hello.txt'));
      expect(all).toContain(path.normalize('readme.md'));
      expect(all).toStrictEqual([...all].sort((x, y) => x.localeCompare(y)));
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });

  it('throws when patterns list empty', async () => {
    await expect(discoverTrainingFiles([])).rejects.toThrow(/at least one glob pattern/);
  });
});
