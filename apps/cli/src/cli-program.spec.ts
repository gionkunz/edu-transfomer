import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';

import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';

import { createCliProgram } from './cli-program';
import * as generateMod from './cmd/generate';
import * as learnMod from './cmd/learn';

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

describe('createCliProgram: learn', () => {
  let runSpy: ReturnType<typeof vi.spyOn>;

  beforeEach(() => {
    runSpy = vi.spyOn(learnMod, 'runLearnCommand').mockImplementation(async () => {});
  });

  afterEach(() => {
    runSpy.mockRestore();
  });

  it('registers learn with its options', () => {
    const learn = createCliProgram().commands.find((c) => c.name() === 'learn');
    expect(learn).toBeDefined();
    const flags = learn?.options.map((o) => o.flags) ?? [];
    for (const flag of [
      '--model <path>',
      '--tokenizer <path>',
      '--temperature <x>',
      '--top-k <k>',
      '--dtype <d>',
      '--no-animation',
    ]) {
      expect(flags).toContain(flag);
    }
  });

  it("uses the default prompt, generate's sampler defaults and animation on", async () => {
    await createCliProgram().parseAsync([
      'node',
      'llm',
      'learn',
      '--model',
      'm',
      '--tokenizer',
      't',
    ]);
    expect(runSpy).toHaveBeenCalledTimes(1);
    expect(runSpy.mock.calls[0]?.[0]).toMatchObject({
      modelArg: 'm',
      tokenizerArg: 't',
      prompt: learnMod.DEFAULT_LEARN_PROMPT,
      temperature: 0.8,
      topK: 40,
      animation: true,
      computeDtype: 'f32',
    });
  });

  it('joins a positional prompt and passes sampler options and --no-animation', async () => {
    await createCliProgram().parseAsync([
      'node',
      'llm',
      'learn',
      '--model',
      'm',
      '--tokenizer',
      't',
      '--temperature',
      '1.2',
      '--top-k',
      '5',
      '--no-animation',
      'The',
      'little',
      'dragon',
    ]);
    expect(runSpy.mock.calls[0]?.[0]).toMatchObject({
      prompt: 'The little dragon',
      temperature: 1.2,
      topK: 5,
      animation: false,
    });
  });

  it('rejects invalid sampler options', async () => {
    const program = createCliProgram();
    program.exitOverride();
    await expect(
      program.parseAsync([
        'node',
        'llm',
        'learn',
        '--model',
        'm',
        '--tokenizer',
        't',
        '--top-k',
        '-1',
      ]),
    ).rejects.toThrow(/--top-k/);
    expect(runSpy).not.toHaveBeenCalled();
  });
});
