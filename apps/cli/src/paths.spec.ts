import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import { afterEach, describe, expect, it } from 'vitest';

import { resolveCheckpointDir, resolveTokenizerPath } from './paths';

describe('resolveCheckpointDir', () => {
  let tmpDir: string;

  afterEach(() => {
    if (tmpDir) {
      fs.rmSync(tmpDir, { recursive: true, force: true });
    }
  });

  it('returns the directory when modelOrJson is a checkpoint dir', () => {
    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'llm-paths-'));
    expect(resolveCheckpointDir(tmpDir)).toBe(tmpDir);
  });

  it('returns the parent when modelOrJson is a model.json sidecar', () => {
    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'llm-paths-'));
    const jsonPath = path.join(tmpDir, 'model.json');
    fs.writeFileSync(jsonPath, '{}');
    expect(resolveCheckpointDir(jsonPath)).toBe(tmpDir);
  });
});

describe('resolveTokenizerPath', () => {
  let tmpDir: string;

  afterEach(() => {
    if (tmpDir) {
      fs.rmSync(tmpDir, { recursive: true, force: true });
    }
  });

  it('uses an explicit --tokenizer path when provided', () => {
    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'llm-paths-'));
    const explicit = path.join(tmpDir, 'custom-tok.json');
    fs.writeFileSync(explicit, '{}');
    expect(resolveTokenizerPath(tmpDir, explicit)).toBe(explicit);
  });

  it('defaults to tokenizer.json in the checkpoint dir', () => {
    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'llm-paths-'));
    const bundled = path.join(tmpDir, 'tokenizer.json');
    fs.writeFileSync(bundled, '{}');
    expect(resolveTokenizerPath(tmpDir)).toBe(bundled);
  });

  it('defaults via model.json sidecar path', () => {
    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'llm-paths-'));
    const bundled = path.join(tmpDir, 'tokenizer.json');
    fs.writeFileSync(bundled, '{}');
    fs.writeFileSync(path.join(tmpDir, 'model.json'), '{}');
    expect(resolveTokenizerPath(path.join(tmpDir, 'model.json'))).toBe(bundled);
  });

  it('throws when no explicit path and tokenizer.json is missing', () => {
    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'llm-paths-'));
    expect(() => resolveTokenizerPath(tmpDir)).toThrow(/tokenizer\.json not found/);
  });
});
