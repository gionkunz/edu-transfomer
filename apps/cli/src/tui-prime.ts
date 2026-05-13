/** Skip automatic clear when debugging raw terminal output (`LLM_SKIP_TUI_PRIME=1`). */
export function shouldPrimeTuiCanvas(): boolean {
  const v = process.env.LLM_SKIP_TUI_PRIME;
  return v === undefined || v === '' || v === '0' || String(v).toLowerCase() === 'false';
}

/**
 * Flush scrollback / main screen before the fullscreen TUI mounts. Mirrors a manual terminal
 * “clear”: avoids duplicated or folded rows in Windows Terminal, WSL, and some POSIX shells when
 * prior output remains in the buffer and redraws accumulate until a resize/reflow.
 */
export function primeTuiTerminalCanvas(): void {
  if (!shouldPrimeTuiCanvas() || process.env.TERM === 'dumb') {
    return;
  }
  try {
    if (process.stdout.isTTY) {
      process.stdout.write('\u001b[3J\u001b[2J\u001b[H');
    }
    if (process.stderr.isTTY) {
      process.stderr.write('\u001b[3J\u001b[2J\u001b[H');
    }
  } catch {
    /* ignore terminals that deny raw writes */
  }
}
