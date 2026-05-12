import { readFile, writeFile } from 'node:fs/promises';

import type { Tokenizer } from './types';

export interface CharTokenizerOptions {
  /**
   * Assigned ids `0 .. length - 1` in order. Must be non-empty strings with no duplicates.
   * Defaults: `<|pad|>`, `<|endoftext|>` (typical placeholder indices for training batches).
   */
  readonly specialTokens?: readonly string[];
}

const JSON_KIND = 'llm-transformer-diy/char-tokenizer' as const;
const JSON_VERSION = 1 as const;

interface CharTokenizerJsonV1 {
  readonly kind: typeof JSON_KIND;
  readonly version: typeof JSON_VERSION;
  readonly specialTokens: readonly string[];
  readonly chars: readonly string[];
}

function assertValidSpecials(specials: readonly string[]): void {
  if (specials.length === 0) {
    throw new Error('CharTokenizer: specialTokens must be non-empty');
  }
  const seen = new Set<string>();
  for (const s of specials) {
    if (s.length === 0) {
      throw new Error('CharTokenizer: specialTokens must not contain empty strings');
    }
    if (seen.has(s)) {
      throw new Error(`CharTokenizer: duplicate special token ${JSON.stringify(s)}`);
    }
    seen.add(s);
  }
}

/** Lexicographic sort by numeric Unicode code point, then `localeCompare` tie-break. */
function compareCharTokens(a: string, b: string): number {
  const ac = a.codePointAt(0) ?? 0;
  const bc = b.codePointAt(0) ?? 0;
  if (ac !== bc) {
    return ac - bc;
  }
  return a.localeCompare(b);
}

function assertSortedUniqueChars(chars: readonly string[], label: string): void {
  const seen = new Set<string>();
  for (let i = 0; i < chars.length; i++) {
    const c = chars[i];
    if (c === undefined) {
      throw new Error(`${label}: chars[${String(i)}] is undefined`);
    }
    if (seen.has(c)) {
      throw new Error(`${label}: duplicate entry ${JSON.stringify(c)} in chars`);
    }
    seen.add(c);
    if (i > 0) {
      const prev = chars[i - 1];
      if (prev !== undefined && compareCharTokens(prev, c) > 0) {
        throw new Error(`${label}: chars are not sorted by code point`);
      }
    }
  }
}

function parseCharTokenizerJson(raw: unknown): CharTokenizerJsonV1 {
  if (raw === null || typeof raw !== 'object' || Array.isArray(raw)) {
    throw new Error('CharTokenizer.load: root must be a JSON object');
  }
  const rec = raw as Record<string, unknown>;
  if (rec.kind !== JSON_KIND) {
    throw new Error(
      `CharTokenizer.load: expected kind ${JSON.stringify(JSON_KIND)}, got ${JSON.stringify(rec.kind)}`,
    );
  }
  if (rec.version !== JSON_VERSION) {
    throw new Error(
      `CharTokenizer.load: unsupported version ${String(rec.version)} (expected ${String(JSON_VERSION)})`,
    );
  }
  if (!Array.isArray(rec.specialTokens) || !rec.specialTokens.every((t) => typeof t === 'string')) {
    throw new Error('CharTokenizer.load: specialTokens must be an array of strings');
  }
  if (!Array.isArray(rec.chars) || !rec.chars.every((t) => typeof t === 'string')) {
    throw new Error('CharTokenizer.load: chars must be an array of strings');
  }
  assertValidSpecials(rec.specialTokens);
  assertSortedUniqueChars(rec.chars, 'CharTokenizer.load');
  const specialSet = new Set(rec.specialTokens);
  for (const ch of rec.chars) {
    if (specialSet.has(ch)) {
      throw new Error(
        `CharTokenizer.load: char ${JSON.stringify(ch)} duplicates a special token string`,
      );
    }
  }
  const expectedLength = rec.specialTokens.length + rec.chars.length;
  if (expectedLength !== new Set([...rec.specialTokens, ...rec.chars]).size) {
    throw new Error('CharTokenizer.load: vocabulary entries must be unique');
  }
  return {
    kind: JSON_KIND,
    version: JSON_VERSION,
    specialTokens: rec.specialTokens,
    chars: rec.chars,
  };
}

/**
 * Character-level tokenizer: each distinct code point from training text becomes a token,
 * after reserved special tokens. `encode` maps **per code point** (not greedy special
 * substrings like `<|pad|>`), so `encode('<|pad|>')` is not necessarily `[0]` unless the
 * corpus and specials align for each character.
 */
export class CharTokenizer implements Tokenizer {
  private specialTokens: string[];
  private idToToken: string[] = [];
  private charToId: Map<string, number> = new Map();

  constructor(options?: CharTokenizerOptions) {
    const defaults = ['<|pad|>', '<|endoftext|>'] as const;
    const specials = options?.specialTokens ?? defaults;
    assertValidSpecials(specials);
    this.specialTokens = [...specials];
  }

  get vocabSize(): number {
    return this.idToToken.length;
  }

  train(text: string, vocabSize: number): void {
    const specialSet = new Set(this.specialTokens);
    const seen = new Set<string>();
    for (const ch of text) {
      if (!specialSet.has(ch)) {
        seen.add(ch);
      }
    }
    const sortedChars = Array.from(seen).sort(compareCharTokens);
    const required = this.specialTokens.length + sortedChars.length;
    if (!Number.isFinite(vocabSize) || vocabSize < required) {
      throw new Error(
        `CharTokenizer.train: vocabSize ${String(vocabSize)} is less than required vocabulary size ${String(required)} (specialTokens=${String(this.specialTokens.length)}, uniqueChars=${String(sortedChars.length)})`,
      );
    }
    this.idToToken = [...this.specialTokens, ...sortedChars];
    this.rebuildCharToId();
  }

  encode(text: string): number[] {
    this.assertTrained();
    const out: number[] = [];
    for (const ch of text) {
      const id = this.charToId.get(ch);
      if (id === undefined) {
        throw new Error(
          `CharTokenizer.encode: unknown code point not in vocabulary (train on text that includes this character): ${JSON.stringify(ch)}`,
        );
      }
      out.push(id);
    }
    return out;
  }

  decode(ids: number[]): string {
    this.assertTrained();
    let s = '';
    for (let i = 0; i < ids.length; i++) {
      const id = ids[i];
      if (id === undefined) {
        throw new Error('CharTokenizer.decode: ids entry is undefined');
      }
      const t = this.idToToken[id];
      if (t === undefined) {
        throw new Error(
          `CharTokenizer.decode: unknown id ${String(id)} (vocabSize=${String(this.vocabSize)})`,
        );
      }
      s += t;
    }
    return s;
  }

  async save(path: string): Promise<void> {
    this.assertTrained();
    const payload: CharTokenizerJsonV1 = {
      kind: JSON_KIND,
      version: JSON_VERSION,
      specialTokens: this.specialTokens,
      chars: this.idToToken.slice(this.specialTokens.length),
    };
    await writeFile(path, `${JSON.stringify(payload)}\n`, 'utf8');
  }

  async load(path: string): Promise<void> {
    const rawText = await readFile(path, 'utf8');
    const parsed: unknown = JSON.parse(rawText) as unknown;
    const data = parseCharTokenizerJson(parsed);
    assertValidSpecials(data.specialTokens);
    this.specialTokens = [...data.specialTokens];
    this.idToToken = [...data.specialTokens, ...data.chars];
    this.rebuildCharToId();
  }

  private assertTrained(): void {
    if (this.idToToken.length === 0) {
      throw new Error('CharTokenizer: call train() or load() before encode/decode/save');
    }
  }

  private rebuildCharToId(): void {
    const m = new Map<string, number>();
    for (let i = 0; i < this.idToToken.length; i++) {
      const t = this.idToToken[i];
      if (t === undefined) {
        throw new Error('CharTokenizer: internal vocabulary hole');
      }
      m.set(t, i);
    }
    this.charToId = m;
  }
}
