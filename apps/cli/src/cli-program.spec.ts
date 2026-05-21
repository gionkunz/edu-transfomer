import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { createCliProgram } from './cli-program';
import * as generateMod from './cmd/generate';

describe('createCliProgram', () => {
  it('registers tokenizer, train, generate, chat, inspect', () => {
    const p = createCliProgram();
    const names = p.commands.map((c) => c.name());
    expect(names).toContain('tokenizer');
    expect(names).toContain('train');
    expect(names).toContain('generate');
    expect(names).toContain('chat');
    expect(names).toContain('inspect');
  });

  it('generate command exposes the --no-explore flag', () => {
    const p = createCliProgram();
    const generate = p.commands.find((c) => c.name() === 'generate');
    expect(generate).toBeDefined();
    const flags = generate?.options.map((o) => o.flags);
    expect(flags).toContain('--no-explore');
  });
});

describe('createCliProgram: --no-explore parsing', () => {
  let runSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    runSpy = vi.spyOn(generateMod, 'runGenerateCommand').mockImplementation(async () => {});
  });

  afterEach(() => {
    runSpy.mockRestore();
  });

  it('defaults explore to true when --no-explore is omitted', async () => {
    const program = createCliProgram();
    await program.parseAsync([
      'node',
      'llm',
      'generate',
      '--model',
      'm',
      '--tokenizer',
      't',
      '--prompt',
      'hello',
    ]);
    expect(runSpy).toHaveBeenCalledTimes(1);
    const args = runSpy.mock.calls[0]?.[0] as { explore?: unknown } | undefined;
    expect(args?.explore).toBe(true);
  });

  it('sets explore to false when --no-explore is passed', async () => {
    const program = createCliProgram();
    await program.parseAsync([
      'node',
      'llm',
      'generate',
      '--model',
      'm',
      '--tokenizer',
      't',
      '--prompt',
      'hello',
      '--no-explore',
    ]);
    expect(runSpy).toHaveBeenCalledTimes(1);
    const args = runSpy.mock.calls[0]?.[0] as { explore?: unknown } | undefined;
    expect(args?.explore).toBe(false);
  });
});

describe('createCliProgram: default tokenizer from checkpoint dir', () => {
  let runSpy: ReturnType<typeof vi.spyOn>;
  let tmpDir: string;

  beforeEach(() => {
    runSpy = vi.spyOn(generateMod, 'runGenerateCommand').mockImplementation(async () => {});
    tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'llm-cli-tok-'));
    fs.writeFileSync(path.join(tmpDir, 'tokenizer.json'), '{}');
  });

  afterEach(() => {
    runSpy.mockRestore();
    fs.rmSync(tmpDir, { recursive: true, force: true });
  });

  it('uses tokenizer.json from the model dir when --tokenizer is omitted', async () => {
    const program = createCliProgram();
    await program.parseAsync(['node', 'llm', 'generate', '--model', tmpDir, '--prompt', 'hello']);
    expect(runSpy).toHaveBeenCalledTimes(1);
    const args = runSpy.mock.calls[0]?.[0] as { tokenizerPath?: string } | undefined;
    expect(args?.tokenizerPath).toBe(path.join(tmpDir, 'tokenizer.json'));
  });

  it('prefers an explicit --tokenizer path', async () => {
    const explicit = path.join(tmpDir, 'custom.json');
    fs.writeFileSync(explicit, '{}');
    const program = createCliProgram();
    await program.parseAsync([
      'node',
      'llm',
      'generate',
      '--model',
      tmpDir,
      '--tokenizer',
      explicit,
      '--prompt',
      'hello',
    ]);
    expect(runSpy).toHaveBeenCalledTimes(1);
    const args = runSpy.mock.calls[0]?.[0] as { tokenizerPath?: string } | undefined;
    expect(args?.tokenizerPath).toBe(explicit);
  });
});
