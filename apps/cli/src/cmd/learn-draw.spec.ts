import { describe, expect, it } from 'vitest';

import {
  fmtShare,
  fmtSigned,
  footerLine,
  headerLines,
  lineText,
  lineWidth,
  shareBar,
  signedBar,
  tokenLabel,
  vectorStrip,
  wrapText,
} from './learn-draw';

describe('vectorStrip', () => {
  it('draws one cell per component, brighter for larger, coloured by sign', () => {
    const spans = vectorStrip(new Float32Array([1, -1, 0, 0.5]), 8, 1);
    expect(lineText(spans)).toBe('██·▒');
    expect(spans.map((s) => s.color)).toEqual(['cyan', 'magenta', 'gray', 'cyan']);
  });

  it('averages long vectors down to the width and keeps the width when partly revealed', () => {
    expect(lineWidth(vectorStrip(new Float32Array(32).fill(1), 24, 1))).toBe(24);
    const half = vectorStrip(new Float32Array([1, 1, 1, 1]), 4, 1, 0.5);
    expect(lineText(half)).toBe('██  ');
  });
});

describe('signedBar', () => {
  it('grows right for positive and left for negative values at a fixed width', () => {
    expect(lineText(signedBar(1, 1, 13))).toBe('      │██████');
    expect(lineText(signedBar(-0.5, 1, 13))).toBe('   ███│      ');
    expect(lineText(signedBar(0, 1, 13))).toBe('      │      ');
    expect(lineWidth(signedBar(5, 1, 13))).toBe(13);
  });
});

describe('number and label formatting', () => {
  it('keeps fixed widths', () => {
    expect(fmtSigned(1.234)).toBe(' +1.23');
    expect(fmtSigned(-12.5)).toBe('-12.50');
    expect(fmtShare(1)).toBe('1.00');
    expect(fmtShare(0.001)).toBe('<.01');
    expect(fmtShare(0.25)).toBe('0.25');
  });

  it('never draws a non-zero share as an empty bar', () => {
    expect(shareBar(0.0001, 8).trim()).toBe('▏');
    expect(shareBar(0, 8).trim()).toBe('');
    expect(shareBar(0.3, 8).length).toBe(8);
  });

  it('makes whitespace visible and bounds the length', () => {
    expect(tokenLabel(' time')).toBe('␣time');
    expect(tokenLabel('\n')).toBe('↵');
    expect(tokenLabel('\x1b[2J')).toBe('�[2J');
    expect(tokenLabel('')).toBe('∅');
    expect(tokenLabel('abcdefghijklmnop')).toBe('abcdefghi…');
  });

  it('wraps at spaces', () => {
    const lines = wrapText('one two three four five', 9);
    expect(lines).toEqual(['one two', 'three', 'four five']);
  });
});

describe('header and footer at 80 columns', () => {
  const info = {
    chapterNumber: 1,
    chapterCount: 3,
    chapterTitle: 'One attention head',
    stepNumber: 2,
    stepCount: 5,
    stepTitle: 'compare with keys',
    stepsPerChapter: [5, 3, 4],
    overallIndex: 1,
    selection: 'block 12/12 head 12/12',
    tokenLabels: Array.from({ length: 40 }, (_, i) => `token${String(i)}`),
  };
  const longTitle = {
    ...info,
    chapterTitle: 'Picking the next token',
    stepTitle: 'a rather long step title that cannot fit',
  };

  it('header names chapter, step and the last prompt token, and fits', () => {
    const [a, b] = headerLines(info, 80);
    expect(lineWidth(a)).toBeLessThanOrEqual(78);
    expect(lineWidth(b)).toBeLessThanOrEqual(78);
    expect(lineText(a)).toContain('1/3 One attention head');
    expect(lineText(a)).toContain('step 2/5');
    expect(lineText(a)).toContain('block 12/12 head 12/12');
    expect(lineText(b)).toContain('●◉○○○ ○○○ ○○○○');
    expect(lineText(b)).toContain('«token39» ▸ ?');
    expect(lineText(b)).toContain('…');
  });

  it('header keeps the selection and shortens the step title when space is tight', () => {
    const [a] = headerLines(longTitle, 80);
    expect(lineWidth(a)).toBeLessThanOrEqual(78);
    expect(lineText(a)).toContain('block 12/12 head 12/12');
    expect(lineText(a)).toContain('…');
  });

  it('footer says what up/down control and always fits', () => {
    const base = { chapterCount: 3, selectsHead: false, samples: false, animated: true };
    const block = footerLine({ ...base, dial: 'block', selectsHead: true, canContinue: null }, 80);
    expect(block).toContain('↑↓ block');
    expect(block).toContain('h/H head');
    const draw = footerLine({ ...base, dial: 'temperature', samples: true, canContinue: true }, 80);
    expect(draw).toContain('↑↓ temperature');
    expect(draw).toContain('n next token');
    expect(draw).toContain('q quit');
    const full = footerLine(
      { ...base, dial: 'temperature', samples: true, canContinue: false },
      80,
    );
    expect(full).toContain('context full');
    for (const line of [block, draw, full]) {
      expect(line.length).toBeLessThanOrEqual(78);
    }
  });
});
