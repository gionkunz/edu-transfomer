import { describe, expect, it } from 'vitest';

import { assertCorpusTokenIdsInVocab } from './lm-batch';

describe('assertCorpusTokenIdsInVocab', () => {
  it('accepts empty corpus', () => {
    expect(() => {
      assertCorpusTokenIdsInVocab(new Uint32Array(0), 100);
    }).not.toThrow();
  });

  it('accepts ids in range', () => {
    assertCorpusTokenIdsInVocab(Uint32Array.of(0, 7999, 4000), 8000);
  });

  it('rejects id >= vocabSize', () => {
    expect(() => {
      assertCorpusTokenIdsInVocab(Uint32Array.of(0, 8000, 1), 8000);
    }).toThrow(/corpus token id 8000/);
  });
});

describe('corpus range vs training', () => {
  it('max id used in LM batches stays below vocab when corpus is valid', () => {
    const tok = Uint32Array.from({ length: 20 }, (_, i) => i);
    assertCorpusTokenIdsInVocab(tok, 512);
    expect(tok[tok.length - 1]).toBeLessThan(512);
  });
});
