import { describe, expect, it } from 'vitest';

import { createCliProgram } from './cli-program.js';

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
});
