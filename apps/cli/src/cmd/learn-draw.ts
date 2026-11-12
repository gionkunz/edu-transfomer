import { bucketVector } from '@llm-transformer-diy/inference';

import { probabilityBar } from './generate-panels';
import type { Dial, Line, Span } from './learn-lesson';

/**
 * Drawing helpers of **`llm learn`**. They turn numbers that were computed in
 * the inference package into characters; no model maths happens here.
 */

/** Columns every lesson line must fit in (an 80-column terminal minus a margin). */
export const LESSON_TEXT_WIDTH = 78;
/** Explanation lines under the schematic. */
export const EXPLANATION_ROWS = 4;
/** Smallest terminal the lesson is laid out for. */
export const LESSON_MIN_COLUMNS = 80;
export const LESSON_MIN_ROWS = 24;
/** Header (2) + gap (1) + explanation (4) + footer (1) + one spare row so the screen never scrolls. */
export const LESSON_CHROME_ROWS = 9;

const SHADES = ['·', '░', '▒', '▓', '█'] as const;

export function span(text: string, color?: string, bold?: boolean): Span {
  return {
    text,
    ...(color !== undefined ? { color } : {}),
    ...(bold === true ? { bold: true } : {}),
  };
}

export function lineWidth(line: Line): number {
  return line.reduce((n, s) => n + s.text.length, 0);
}

export function lineText(line: Line): string {
  return line.map((s) => s.text).join('');
}

/**
 * A vector as a strip of cells: brightness is the size of each component
 * relative to **`scale`**, colour its sign (cyan positive, magenta negative).
 * Vectors longer than **`width`** are averaged into **`width`** cells. With
 * **`reveal`** < 1 only that leading fraction is drawn (for animations).
 */
export function vectorStrip(v: Float32Array, width: number, scale: number, reveal = 1): Span[] {
  const cells = bucketVector(v, width);
  const shown = Math.max(0, Math.min(cells.length, Math.round(cells.length * reveal)));
  const s = scale > 0 ? scale : 1;
  const spans: Span[] = [];
  let run = '';
  let runColor: string | undefined;
  const flush = (): void => {
    if (run.length > 0) {
      spans.push(span(run, runColor));
      run = '';
    }
  };
  for (let i = 0; i < cells.length; i++) {
    let ch = ' ';
    let color: string | undefined;
    if (i < shown) {
      const x = cells[i] ?? 0;
      const rel = Math.min(1, Math.abs(x) / s);
      const level = rel < 0.08 ? 0 : Math.min(4, Math.ceil(rel * 4));
      ch = SHADES[level] ?? '·';
      color = level === 0 ? 'gray' : x >= 0 ? 'cyan' : 'magenta';
    }
    if (color !== runColor) {
      flush();
      runColor = color;
    }
    run += ch;
  }
  flush();
  return spans;
}

/** Number of cells **`vectorStrip`** draws for a vector of length **`n`**. */
export function stripWidth(n: number, maxWidth: number): number {
  return Math.min(n, maxWidth);
}

/**
 * A bar for a value that can be negative: it grows right of the centre line
 * for positive values and left of it for negative ones. **`width`** cells in
 * total (an odd number keeps both sides equal).
 */
export function signedBar(value: number, scale: number, width: number): Span[] {
  const half = Math.floor((width - 1) / 2);
  const s = scale > 0 ? scale : 1;
  const rel = Math.max(-1, Math.min(1, Number.isFinite(value) ? value / s : 0));
  const cells = Math.round(Math.abs(rel) * half);
  if (rel < 0) {
    return [
      span(' '.repeat(half - cells)),
      span('█'.repeat(cells), 'magenta'),
      span('│', 'gray'),
      span(' '.repeat(width - half - 1)),
    ];
  }
  return [
    span(' '.repeat(half)),
    span('│', 'gray'),
    span('█'.repeat(cells), 'cyan'),
    span(' '.repeat(width - half - 1 - cells)),
  ];
}

/** `+1.23` / `-0.40`, always the same width. */
export function fmtSigned(x: number): string {
  if (!Number.isFinite(x)) return '    ?';
  const s = `${x >= 0 ? '+' : '-'}${Math.abs(x).toFixed(2)}`;
  return s.padStart(6, ' ');
}

/** A probability or weight in four characters. */
export function fmtShare(p: number): string {
  if (!Number.isFinite(p)) return '   ?';
  if (p >= 0.995) return '1.00';
  if (p > 0 && p < 0.005) return '<.01';
  return p.toFixed(2);
}

export function fmtPercent(p: number): string {
  const pct = p * 100;
  if (pct >= 99.5) return '100%';
  if (pct > 0 && pct < 1) return '<1%';
  return `${pct.toFixed(0)}%`;
}

/** Probability bar that never rounds a non-zero value down to nothing. */
export function shareBar(p: number, width: number): string {
  const bar = probabilityBar(p, width);
  return p > 0 && bar.trim().length === 0 ? '▏'.padEnd(width, ' ') : bar;
}

/** Make a token piece safe and legible on one line: visible whitespace, bounded length. */
export function tokenLabel(piece: string, maxLength = 10): string {
  let out = '';
  for (const ch of piece) {
    const code = ch.codePointAt(0) ?? 0;
    if (ch === ' ') out += '␣';
    else if (ch === '\n') out += '↵';
    else if (ch === '\t') out += '→';
    else if (code <= 0x1f || code === 0x7f || (code >= 0x80 && code <= 0x9f)) out += '�';
    else out += ch;
  }
  if (out.length === 0) out = '∅';
  return out.length <= maxLength ? out : `${out.slice(0, Math.max(0, maxLength - 1))}…`;
}

/** Break text into lines of at most **`width`** characters at spaces. */
export function wrapText(text: string, width: number): string[] {
  const lines: string[] = [];
  let current = '';
  for (const word of text.split(/\s+/).filter((w) => w.length > 0)) {
    if (current.length === 0) {
      current = word;
    } else if (current.length + 1 + word.length <= width) {
      current += ` ${word}`;
    } else {
      lines.push(current);
      current = word;
    }
  }
  if (current.length > 0) lines.push(current);
  return lines;
}

function truncate(text: string, width: number): string {
  return text.length <= width ? text : `${text.slice(0, Math.max(0, width - 1))}…`;
}

export type HeaderInfo = {
  readonly chapterNumber: number;
  readonly chapterCount: number;
  readonly chapterTitle: string;
  readonly stepNumber: number;
  readonly stepCount: number;
  readonly stepTitle: string;
  /** Steps per chapter, for the progress dots. */
  readonly stepsPerChapter: readonly number[];
  /** 0-based index of the current step among all steps. */
  readonly overallIndex: number;
  /** e.g. `block 8/8 · head 3/8`; empty when the step has no selection. */
  readonly selection: string;
  /** Labels of the prompt tokens, oldest first. */
  readonly tokenLabels: readonly string[];
};

/** The two header rows: where we are, and the prompt whose next token is being predicted. */
export function headerLines(info: HeaderInfo, cols: number): [Line, Line] {
  const width = Math.min(cols, LESSON_MIN_COLUMNS) - 2;
  const where = `${String(info.chapterNumber)}/${String(info.chapterCount)} ${info.chapterTitle}`;
  const separator = ' · ';
  // The selection (block / head) is always shown in full; the step title gives way.
  const selectionWidth = info.selection.length > 0 ? separator.length + info.selection.length : 0;
  const step = truncate(
    `step ${String(info.stepNumber)}/${String(info.stepCount)}: ${info.stepTitle}`,
    Math.max(8, width - where.length - separator.length - selectionWidth),
  );
  const first: Span[] = [span(where, 'yellow', true), span(separator, 'gray'), span(step)];
  if (info.selection.length > 0) {
    first.push(span(separator, 'gray'), span(info.selection, 'cyan'));
  }

  let dots = '';
  let seen = 0;
  info.stepsPerChapter.forEach((count, c) => {
    if (c > 0) dots += ' ';
    for (let i = 0; i < count; i++) {
      dots += seen < info.overallIndex ? '●' : seen === info.overallIndex ? '◉' : '○';
      seen += 1;
    }
  });
  // Prompt tail: as many of the most recent tokens as fit, last one emphasised.
  const last = info.tokenLabels[info.tokenLabels.length - 1] ?? '';
  const tailLabel = `«${last}»`;
  const suffix = ' ▸ ?';
  const prefixLabel = '   prompt ';
  let room = width - dots.length - prefixLabel.length - tailLabel.length - suffix.length;
  const earlier: string[] = [];
  for (let i = info.tokenLabels.length - 2; i >= 0; i--) {
    const piece = `«${info.tokenLabels[i] ?? ''}»`;
    if (piece.length + 1 > room - 2) {
      earlier.unshift('…');
      break;
    }
    earlier.unshift(piece);
    room -= piece.length + 1;
  }
  const second: Span[] = [
    span(dots, 'yellow'),
    span(prefixLabel, 'gray'),
    span(earlier.length > 0 ? `${earlier.join(' ')} ` : '', 'gray'),
    span(tailLabel, undefined, true),
    span(suffix, 'cyan'),
  ];
  return [first, second];
}

export type FooterInfo = {
  readonly dial: Dial;
  readonly chapterCount: number;
  readonly selectsHead: boolean;
  readonly samples: boolean;
  readonly animated: boolean;
  /** `true` / `false` on the last step (continue possible or context full); `null` elsewhere. */
  readonly canContinue: boolean | null;
};

/**
 * The key legend. Items are dropped from the least important end until the
 * line fits, so the keys that act on the current step are always listed.
 */
export function footerLine(info: FooterInfo, cols: number): string {
  const width = Math.min(cols, LESSON_MIN_COLUMNS) - 2;
  // [text, priority]: lower priority is dropped first.
  const items: [string, number][] = [['←→ step', 9]];
  if (info.dial === 'block') items.push(['↑↓ block', 8]);
  if (info.dial === 'temperature') items.push(['↑↓ temperature', 8]);
  if (info.selectsHead) items.push(['h/H head', 8]);
  if (info.samples) items.push(['k top-k', 7], ['r redraw', 7]);
  if (info.canContinue === true) items.push(['n next token', 8]);
  if (info.canContinue === false) items.push(['n: context full', 8]);
  if (info.animated) items.push(['space replay', 4]);
  items.push([`1-${String(info.chapterCount)} chapter`, 3], ['q quit', 9]);

  const join = (list: [string, number][]): string => list.map(([t]) => t).join(' · ');
  let kept = items;
  while (join(kept).length > width && kept.length > 1) {
    const lowest = Math.min(...kept.map(([, p]) => p));
    const at = kept.findIndex(([, p]) => p === lowest);
    kept = kept.filter((_, i) => i !== at);
  }
  return truncate(join(kept), width);
}
