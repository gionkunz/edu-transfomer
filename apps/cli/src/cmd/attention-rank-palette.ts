/** Fixed 10-color rank palette (high-contrast, distinct hues) for attention highlights; rank 1 first. */
export const ATTENTION_RANK_COLORS = [
  '#e6194b',
  '#3cb44b',
  '#ffe119',
  '#4363d8',
  '#f58231',
  '#911eb4',
  '#46f0f0',
  '#f032e6',
  '#bcf60c',
  '#fabebe',
] as const;

export type AttentionRankColor = (typeof ATTENTION_RANK_COLORS)[number];

const HIGHLIGHT_FOREGROUND = '#000000';
const ANSI_RESET = '\u001b[0m';

function hexToRgb(hex: string): [number, number, number] {
  const normalized = hex.replace('#', '');
  return [
    Number.parseInt(normalized.slice(0, 2), 16),
    Number.parseInt(normalized.slice(2, 4), 16),
    Number.parseInt(normalized.slice(4, 6), 16),
  ];
}

/** True-color ANSI: black foreground on a hex background (no extra deps). */
function ansiHighlight(text: string, backgroundHex: string, foregroundHex: string): string {
  const [br, bg, bb] = hexToRgb(backgroundHex);
  const [fr, fg, fb] = hexToRgb(foregroundHex);
  return `\u001b[38;2;${String(fr)};${String(fg)};${String(fb)}m\u001b[48;2;${String(br)};${String(bg)};${String(bb)}m${text}${ANSI_RESET}`;
}

/** 1-based global rank → wolf-tui/chalk color string. */
export function colorForGlobalRank(rank: number): AttentionRankColor | undefined {
  if (!Number.isInteger(rank) || rank < 1 || rank > ATTENTION_RANK_COLORS.length) {
    return undefined;
  }
  return ATTENTION_RANK_COLORS[rank - 1];
}

/** Terminal transform: rank-colored background with black foreground text. */
export function rankHighlightTransform(rank: number): ((text: string) => string) | undefined {
  const backgroundColor = colorForGlobalRank(rank);
  if (backgroundColor === undefined) return undefined;
  return (text: string) => ansiHighlight(text, backgroundColor, HIGHLIGHT_FOREGROUND);
}
