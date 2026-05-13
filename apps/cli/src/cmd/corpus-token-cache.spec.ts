import { describe, expect, it } from 'vitest';

import {
  type CorpusTokenCacheFingerprint,
  corpusTokenFingerprintsEqual,
} from './corpus-token-cache.js';

describe('corpusTokenFingerprintsEqual', () => {
  function fp(partial: Partial<CorpusTokenCacheFingerprint>): CorpusTokenCacheFingerprint {
    return {
      formatVersion: 1,
      joinBetweenFiles: '\n',
      tokenizerResolved: '/abs/tok.json',
      tokenizerMtimeMs: 1,
      corpusFiles: [],
      ...partial,
    };
  }

  it('matches identical fingerprints', () => {
    const a = fp({
      corpusFiles: [{ path: '/abs/a.txt', size: 3, mtimeMs: 2 }],
    });
    expect(corpusTokenFingerprintsEqual(a, { ...a, corpusFiles: [...a.corpusFiles] })).toBe(true);
  });

  it('rejects mtime drift', () => {
    const a = fp({});
    expect(
      corpusTokenFingerprintsEqual(a, {
        ...a,
        tokenizerMtimeMs: 99,
      }),
    ).toBe(false);
  });

  it('rejects corpus file length mismatch', () => {
    expect(
      corpusTokenFingerprintsEqual(
        fp({ corpusFiles: [{ path: '/a', size: 1, mtimeMs: 1 }] }),
        fp({ corpusFiles: [] }),
      ),
    ).toBe(false);
  });
});
