import { describe, expect, it } from 'vitest';

import { formatDurationSec } from './format-duration';

describe('formatDurationSec', () => {
  it('returns em-dash for non-finite', () => {
    expect(formatDurationSec(Number.NaN)).toBe('—');
    expect(formatDurationSec(Number.POSITIVE_INFINITY)).toBe('—');
    expect(formatDurationSec(null)).toBe('—');
    expect(formatDurationSec(undefined)).toBe('—');
  });

  it('formats zero and sub-minute', () => {
    expect(formatDurationSec(0)).toBe('0s');
    expect(formatDurationSec(45)).toBe('45s');
    expect(formatDurationSec(45.2)).toBe('45.2s');
    expect(formatDurationSec(59.94)).toBe('59.9s');
  });

  it('formats minutes and hours', () => {
    expect(formatDurationSec(60)).toBe('1m');
    expect(formatDurationSec(90)).toBe('1m 30s');
    expect(formatDurationSec(3600)).toBe('1h');
    expect(formatDurationSec(3723)).toBe('1h 2m 3s');
  });

  it('formats days and skips zero units', () => {
    expect(formatDurationSec(86_400)).toBe('1d');
    expect(formatDurationSec(86_400 + 7200 + 30)).toBe('1d 2h 30s');
    expect(formatDurationSec(90_061)).toBe('1d 1h 1m 1s');
  });

  it('clamps negatives', () => {
    expect(formatDurationSec(-10)).toBe('0s');
  });
});
