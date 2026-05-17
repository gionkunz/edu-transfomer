import { mkdtemp, rm, writeFile } from 'node:fs/promises';
import os from 'node:os';
import path from 'node:path';

import { writePackedCheckpointFiles } from '@llm-transformer-diy/io';
import { BPETokenizer } from '@llm-transformer-diy/tokenizer';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { createCliProgram } from '../cli-program';
import {
  runEmbeddingsDecode,
  runEmbeddingsEncode,
  runEmbeddingsExpression,
  runEmbeddingsSimilarity,
} from './embeddings';

const D_MODEL = 4;

type Fixture = {
  readonly dir: string;
  readonly modelPath: string;
  readonly tokenizerPath: string;
  readonly tokenizer: BPETokenizer;
  readonly idA: number;
  readonly idB: number;
};

async function makeFixture(): Promise<Fixture> {
  const dir = await mkdtemp(path.join(os.tmpdir(), 'llm-embeddings-spec-'));

  const tokenizer = new BPETokenizer();
  tokenizer.train('a b c d e f g h i a b a b a b a b a b a b a b ', 300);
  const tokenizerPath = path.join(dir, 'tokenizer.json');
  await tokenizer.save(tokenizerPath);
  const vocabSize = tokenizer.vocabSize;

  const idsA = tokenizer.encode('a');
  const idsB = tokenizer.encode('b');
  if (idsA.length !== 1 || idsB.length !== 1) {
    throw new Error(`fixture: expected single-token encodes for 'a' and 'b'`);
  }
  const idA = idsA[0] as number;
  const idB = idsB[0] as number;

  // Background rows live entirely in the z/w plane so the x/y plane is "owned" by the
  // crafted `a` and `b` rows below. This makes top-K against `a + b` deterministic.
  const data = new Float32Array(vocabSize * D_MODEL);
  for (let i = 0; i < vocabSize; i++) {
    data[i * D_MODEL + 0] = 0;
    data[i * D_MODEL + 1] = 0;
    data[i * D_MODEL + 2] = ((i % 7) - 3) * 0.13 + 0.001;
    data[i * D_MODEL + 3] = ((i * 5) % 11) * 0.07 + 0.001;
  }
  data[idA * D_MODEL + 0] = 1;
  data[idA * D_MODEL + 1] = 0;
  data[idA * D_MODEL + 2] = 0;
  data[idA * D_MODEL + 3] = 0;
  data[idB * D_MODEL + 0] = 0;
  data[idB * D_MODEL + 1] = 1;
  data[idB * D_MODEL + 2] = 0;
  data[idB * D_MODEL + 3] = 0;

  const config = {
    dModel: D_MODEL,
    nLayers: 1,
    nHeads: 1,
    dFf: 8,
    maxSeqLen: 8,
    vocabSize,
  };
  const modelPath = path.join(dir, 'model.json');
  const binPath = path.join(dir, 'model.bin');
  await writePackedCheckpointFiles({
    binPath,
    jsonPath: modelPath,
    config,
    tensors: {
      token_embed: { shape: [vocabSize, D_MODEL], data, dtype: 'f32' },
    },
  });

  return { dir, modelPath, tokenizerPath, tokenizer, idA, idB };
}

describe('llm embeddings commands', () => {
  let fix: Fixture;
  let logs: string[];
  let logSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(async () => {
    fix = await makeFixture();
    logs = [];
    logSpy = vi.spyOn(console, 'log').mockImplementation((...args: unknown[]) => {
      logs.push(args.map((a) => (typeof a === 'string' ? a : JSON.stringify(a))).join(' '));
    });
  });

  afterEach(async () => {
    logSpy.mockRestore();
    await rm(fix.dir, { recursive: true, force: true });
  });

  it('encode prints input-expansion trace + vector by default', async () => {
    await runEmbeddingsEncode({
      modelArg: fix.modelPath,
      tokenizerPath: fix.tokenizerPath,
      text: 'a',
      perToken: false,
      json: false,
      autoSpace: false,
    });
    const text = logs.join('\n');
    expect(text).toMatch(/1 token/);
    expect(text).toMatch(/\[1, 0, 0, 0\]/);
  });

  it('encode --json emits parseable object with vector', async () => {
    await runEmbeddingsEncode({
      modelArg: fix.modelPath,
      tokenizerPath: fix.tokenizerPath,
      text: 'a',
      perToken: false,
      json: true,
      autoSpace: false,
    });
    expect(logs).toHaveLength(1);
    const parsed = JSON.parse(logs[0] as string) as {
      input: string;
      encoded: string;
      tokens: { id: number; surface: string }[];
      vector: number[];
    };
    expect(parsed.input).toBe('a');
    expect(parsed.encoded).toBe('a');
    expect(parsed.tokens).toHaveLength(1);
    expect(parsed.vector).toEqual([1, 0, 0, 0]);
  });

  it('encode with autoSpace=true prefixes a leading space and reports `encoded`', async () => {
    await runEmbeddingsEncode({
      modelArg: fix.modelPath,
      tokenizerPath: fix.tokenizerPath,
      text: 'a',
      perToken: false,
      json: true,
      autoSpace: true,
    });
    const parsed = JSON.parse(logs[0] as string) as {
      input: string;
      encoded: string;
    };
    expect(parsed.input).toBe('a');
    expect(parsed.encoded).toBe(' a');
  });

  it('decode returns top-K nearest tokens with band + z labels', async () => {
    await runEmbeddingsDecode({
      modelArg: fix.modelPath,
      tokenizerPath: fix.tokenizerPath,
      vectorSpec: 'a',
      topK: 5,
      json: true,
      autoSpace: false,
    });
    const parsed = JSON.parse(logs[0] as string) as {
      referenceStats: { mean: number; std: number; count: number };
      results: { id: number; cosine: number; z: number; band: string }[];
    };
    expect(parsed.results.length).toBeGreaterThan(0);
    expect(parsed.results.length).toBeLessThanOrEqual(5);
    expect(parsed.results.map((r) => r.id)).not.toContain(fix.idA);
    for (const r of parsed.results) {
      expect(typeof r.z).toBe('number');
      expect(typeof r.band).toBe('string');
    }
    expect(parsed.referenceStats.count).toBeGreaterThan(0);
  });

  it('decode @file with wrong dimension errors loudly', async () => {
    const vecPath = path.join(fix.dir, 'badvec.json');
    await writeFile(vecPath, JSON.stringify({ vector: [1, 2, 3] }), 'utf8');
    await expect(
      runEmbeddingsDecode({
        modelArg: fix.modelPath,
        tokenizerPath: fix.tokenizerPath,
        vectorSpec: `@${vecPath}`,
        topK: 5,
        json: false,
        autoSpace: false,
      }),
    ).rejects.toThrow(/dimension mismatch/);
  });

  it('similarity computes cosine in [-1,1] and labels with a calibrated band', async () => {
    await runEmbeddingsSimilarity({
      modelArg: fix.modelPath,
      tokenizerPath: fix.tokenizerPath,
      a: 'a',
      b: 'a',
      json: true,
      autoSpace: false,
    });
    const parsed = JSON.parse(logs[0] as string) as {
      cosine: number;
      z: number | string;
      band: string;
      referenceStats: { mean: number; std: number; count: number };
    };
    expect(parsed.cosine).toBeCloseTo(1, 5);
    expect(parsed.band).toBe('identical');
    // In the fixture, reference rows are orthogonal to `a`'s axis so std == 0
    // and z is reported as '+inf'. With a real-world distribution it would be
    // a large finite number; both are accepted here.
    expect(parsed.z === '+inf' || (typeof parsed.z === 'number' && parsed.z > 5)).toBe(true);
    expect(parsed.referenceStats.count).toBeGreaterThan(0);

    logs.length = 0;
    await runEmbeddingsSimilarity({
      modelArg: fix.modelPath,
      tokenizerPath: fix.tokenizerPath,
      a: 'a',
      b: 'b',
      json: true,
      autoSpace: false,
    });
    const parsed2 = JSON.parse(logs[0] as string) as { cosine: number; band: string };
    expect(parsed2.cosine).toBeCloseTo(0, 6);
    // Both `a` and `b` are crafted to be far from the background z/w plane,
    // so they're orthogonal to the reference distribution and should land in
    // the unrelated / weakly-similar zone (definitely not "identical").
    expect(parsed2.band).not.toBe('identical');
  });

  it('expression "a + b" excludes inputs by default and produces top-K', async () => {
    await runEmbeddingsExpression({
      modelArg: fix.modelPath,
      tokenizerPath: fix.tokenizerPath,
      expression: 'a + b',
      topK: 3,
      excludeInputs: true,
      json: true,
      autoSpace: false,
    });
    const parsed = JSON.parse(logs[0] as string) as {
      expression: string;
      excludeInputs: boolean;
      results: { id: number; cosine: number }[];
    };
    expect(parsed.expression).toBe('a + b');
    expect(parsed.excludeInputs).toBe(true);
    expect(parsed.results.length).toBeGreaterThan(0);
    expect(parsed.results.map((r) => r.id)).not.toContain(fix.idA);
    expect(parsed.results.map((r) => r.id)).not.toContain(fix.idB);
  });

  it('expression with --exclude-inputs=false includes input tokens', async () => {
    await runEmbeddingsExpression({
      modelArg: fix.modelPath,
      tokenizerPath: fix.tokenizerPath,
      expression: 'a + b',
      topK: 5,
      excludeInputs: false,
      json: true,
      autoSpace: false,
    });
    const parsed = JSON.parse(logs[0] as string) as {
      results: { id: number; cosine: number }[];
    };
    const ids = parsed.results.map((r) => r.id);
    expect(ids).toContain(fix.idA);
    expect(ids).toContain(fix.idB);
  });

  it('expression with autoSpace=true prefixes bare atoms but leaves quoted atoms literal', async () => {
    await runEmbeddingsExpression({
      modelArg: fix.modelPath,
      tokenizerPath: fix.tokenizerPath,
      expression: 'a + "a"',
      topK: 5,
      excludeInputs: false,
      json: true,
      autoSpace: true,
    });
    const parsed = JSON.parse(logs[0] as string) as {
      autoSpace: boolean;
      inputs: { atom: string; quoted: boolean; encoded: string }[];
    };
    expect(parsed.autoSpace).toBe(true);
    const bare = parsed.inputs.find((i) => !i.quoted);
    const quoted = parsed.inputs.find((i) => i.quoted);
    expect(bare?.encoded).toBe(' a');
    expect(quoted?.encoded).toBe('a');
  });

  it('expression with malformed input errors with parser position', async () => {
    await expect(
      runEmbeddingsExpression({
        modelArg: fix.modelPath,
        tokenizerPath: fix.tokenizerPath,
        expression: 'a +',
        topK: 5,
        excludeInputs: true,
        json: false,
        autoSpace: false,
      }),
    ).rejects.toThrow(/at position/);
  });
});

describe('createCliProgram: embeddings registration', () => {
  it('exposes the embeddings subcommand group with all subcommands', () => {
    const p = createCliProgram();
    const emb = p.commands.find((c) => c.name() === 'embeddings');
    expect(emb).toBeDefined();
    const names = emb?.commands.map((c) => c.name()) ?? [];
    expect(names).toEqual(
      expect.arrayContaining(['encode', 'decode', 'similarity', 'expression', 'nearest']),
    );
  });

  it('unknown embeddings subcommand exits non-zero', async () => {
    const program = createCliProgram();
    program.exitOverride();
    for (const sub of program.commands) sub.exitOverride();
    const emb = program.commands.find((c) => c.name() === 'embeddings');
    if (emb) for (const sub of emb.commands) sub.exitOverride();

    await expect(
      program.parseAsync(['node', 'llm', 'embeddings', 'frobnicate']),
    ).rejects.toBeDefined();
  });
});
