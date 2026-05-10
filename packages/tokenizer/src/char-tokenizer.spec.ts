import { mkdir, mkdtemp, rm } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';

import { describe, expect, it } from 'vitest';

import { CharTokenizer } from './char-tokenizer';

describe('CharTokenizer', () => {
  it('throws when encode is used before train', () => {
    const tok = new CharTokenizer();
    expect(() => tok.encode('a')).toThrow(/train\(\)|load()/);
  });

  it('throws when train vocabSize is below required minimum', () => {
    const tok = new CharTokenizer();
    expect(() => tok.train('aab', 3)).toThrow(/less than required vocabulary size/);
  });

  it('train accepts vocabSize larger than required', () => {
    const tok = new CharTokenizer();
    tok.train('aab', 100);
    expect(tok.vocabSize).toBe(4);
  });

  it('assigns specials ids 0.. before sorted code points', () => {
    const tok = new CharTokenizer({ specialTokens: ['<|a|>', '<|b|>'] });
    tok.train('cba', 20);
    expect(tok.decode([0])).toBe('<|a|>');
    expect(tok.decode([1])).toBe('<|b|>');
    expect(tok.encode('a')).toEqual([2]);
    expect(tok.encode('b')).toEqual([3]);
    expect(tok.encode('c')).toEqual([4]);
    const ids = [4, 3, 2];
    expect(tok.decode(ids)).toBe('cba');
  });

  it('round-trips mixed text when all code points were in training', () => {
    const tok = new CharTokenizer();
    const corpus = 'a ab b a\n';
    tok.train(corpus, 256);
    expect(tok.decode(tok.encode(corpus))).toBe(corpus);
  });

  it('stable ordering: same corpus yields same ids across instances', () => {
    const a = new CharTokenizer();
    const b = new CharTokenizer();
    const corpus = 'βαa';
    a.train(corpus, 64);
    b.train(corpus, 64);
    expect(a.encode(corpus)).toEqual(b.encode(corpus));
  });

  it('excludes code points that match an entire special token string', () => {
    const tok = new CharTokenizer({ specialTokens: ['z'] });
    tok.train('zabc', 64);
    expect(tok.vocabSize).toBe(4);
    expect(tok.encode('z')).toEqual([0]);
    expect(tok.encode('abc')).toEqual([1, 2, 3]);
  });

  it('save and load round-trip', async () => {
    const dir = await mkdtemp(path.join(os.tmpdir(), 'tokenizer-test-'));
    try {
      await mkdir(dir, { recursive: true });
      const file = path.join(dir, 'tok.json');
      const first = new CharTokenizer();
      first.train('hello\n', 128);
      await first.save(file);

      const second = new CharTokenizer();
      await second.load(file);
      expect(second.vocabSize).toBe(first.vocabSize);
      expect(second.encode('hello\n')).toEqual(first.encode('hello\n'));
      expect(second.decode([1, 2, 3, 4, 5])).toBe(first.decode([1, 2, 3, 4, 5]));
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });

  it('load restores specials from file', async () => {
    const dir = await mkdtemp(path.join(os.tmpdir(), 'tokenizer-test-'));
    try {
      const file = path.join(dir, 'tok.json');
      const first = new CharTokenizer({ specialTokens: ['<|x|>'] });
      first.train('ab', 32);
      await first.save(file);

      const second = new CharTokenizer({ specialTokens: ['<|wrong|>'] });
      await second.load(file);
      expect(second.decode([0])).toBe('<|x|>');
    } finally {
      await rm(dir, { recursive: true, force: true });
    }
  });
});
