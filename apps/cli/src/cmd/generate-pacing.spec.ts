import { describe, expect, it } from 'vitest';

import { PacingController } from './generate-pacing';

describe('PacingController', () => {
  it('returns immediately in max mode', async () => {
    const p = new PacingController();
    const t0 = Date.now();
    await p.waitBetweenSteps();
    const dt = Date.now() - t0;
    expect(dt).toBeLessThan(50);
    expect(p.statusLabel).toBe('NORMAL');
  });

  it('sleeps ~slowDelayMs in slow mode', async () => {
    const p = new PacingController({ slowDelayMs: 80 });
    p.setSpeed('slow');
    expect(p.statusLabel).toBe('SLOW');
    const t0 = Date.now();
    await p.waitBetweenSteps();
    const dt = Date.now() - t0;
    expect(dt).toBeGreaterThanOrEqual(60);
    expect(dt).toBeLessThan(500);
  });

  it('cancel() flushes the slow timer so waitBetweenSteps returns', async () => {
    const p = new PacingController({ slowDelayMs: 60_000 });
    p.setSpeed('slow');
    const wait = p.waitBetweenSteps();
    setTimeout(() => {
      p.cancel();
    }, 10);
    const t0 = Date.now();
    await wait;
    expect(Date.now() - t0).toBeLessThan(500);
  });

  it('togglePause blocks waitBetweenSteps until untoggled', async () => {
    const p = new PacingController({ slowDelayMs: 0 });
    p.togglePause();
    expect(p.paused).toBe(true);
    expect(p.statusLabel).toBe('PAUSED');
    const wait = p.waitBetweenSteps();

    let resolved = false;
    void wait.then(() => {
      resolved = true;
    });
    await new Promise<void>((r) => setTimeout(r, 30));
    expect(resolved).toBe(false);

    p.togglePause();
    expect(p.paused).toBe(false);
    await wait;
    expect(resolved).toBe(true);
  });

  it('cancel() releases a pending pause gate', async () => {
    const p = new PacingController({ slowDelayMs: 0 });
    p.togglePause();
    const wait = p.waitBetweenSteps();
    setTimeout(() => {
      p.cancel();
    }, 10);
    const t0 = Date.now();
    await wait;
    expect(Date.now() - t0).toBeLessThan(500);
  });

  it('setSpeed(max) while sleeping in slow mode releases the timer', async () => {
    const p = new PacingController({ slowDelayMs: 60_000 });
    p.setSpeed('slow');
    const wait = p.waitBetweenSteps();
    setTimeout(() => {
      p.setSpeed('max');
    }, 10);
    const t0 = Date.now();
    await wait;
    expect(Date.now() - t0).toBeLessThan(500);
    expect(p.speed).toBe('max');
  });

  it('toggleSpeed flips between max and slow', () => {
    const p = new PacingController();
    expect(p.speed).toBe('max');
    p.toggleSpeed();
    expect(p.speed).toBe('slow');
    p.toggleSpeed();
    expect(p.speed).toBe('max');
  });

  it('PAUSED status wins over SLOW', () => {
    const p = new PacingController();
    p.setSpeed('slow');
    p.togglePause();
    expect(p.statusLabel).toBe('PAUSED');
    p.togglePause();
    expect(p.statusLabel).toBe('SLOW');
  });
});
