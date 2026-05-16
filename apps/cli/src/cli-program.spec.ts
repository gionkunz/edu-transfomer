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
