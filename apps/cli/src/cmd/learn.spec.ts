import { describe, expect, it } from 'vitest';

import { assertInteractiveTerminal, assertLessonPromptFits, runLearnCommand } from './learn';

describe('llm learn preconditions', () => {
  it('requires stdin and stdout to be terminals', () => {
    expect(() => {
      assertInteractiveTerminal({ stdinIsTTY: true, stdoutIsTTY: true });
    }).not.toThrow();
    expect(() => {
      assertInteractiveTerminal({ stdinIsTTY: false, stdoutIsTTY: true });
    }).toThrow(/interactive terminal/);
    expect(() => {
      assertInteractiveTerminal({ stdinIsTTY: true, stdoutIsTTY: false });
    }).toThrow(/interactive terminal/);
  });

  it('fails before loading anything when not run in a terminal', async () => {
    // Vitest runs without a TTY; a missing checkpoint would give a different error.
    await expect(
      runLearnCommand({
        modelArg: '/nonexistent/checkpoint',
        prompt: 'x',
        temperature: 0.8,
        topK: 40,
        animation: true,
      }),
    ).rejects.toThrow(/interactive terminal/);
  });

  it('accepts prompts that leave room for the predicted token and rejects longer ones', () => {
    expect(() => {
      assertLessonPromptFits(1, 16);
    }).not.toThrow();
    expect(() => {
      assertLessonPromptFits(15, 16);
    }).not.toThrow();
    expect(() => {
      assertLessonPromptFits(16, 16);
    }).toThrow(/16 tokens long.*up to 15 tokens/);
    expect(() => {
      assertLessonPromptFits(0, 16);
    }).toThrow(/empty/);
  });
});
