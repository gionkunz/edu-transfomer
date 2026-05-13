## 1. OpenSpec (this change)

- [x] 1.1 Add `proposal.md`, `design.md`, delta `specs/cli-wolf-tui-commands/spec.md`, and `tasks.md` under `openspec/changes/train-tui-bpe-encode-eta/`.

## 2. Types and helpers

- [x] 2.1 Extend `EncodeProgressSnapshot` in `train-dash-types.ts` with fields for elapsed sec, pessimistic merge upper bound, remaining merges, smoothed merges/s, optional ETA sec, and document semantics in types.
- [x] 2.2 Add `encode-eta.ts` (or similar) with pure helpers: pessimistic upper bound, remaining merges, ETA from rate; add a small Vitest spec.

## 3. Train pipeline + TUI

- [x] 3.1 In `train.tsx`, track encode wall-clock start, initial token count from `expand`, EWMA merges/s between `onProgress` merge samples, and populate extended `encodeProgress`.
- [x] 3.2 In `train-dashboard.tsx`, render elapsed, merge rate, pessimistic `ProgressBar`, and formatted ETA with **worst-case / may finish sooner** copy.

## 4. Verification

- [x] 4.1 `pnpm exec nx run @llm-transformer-diy/cli:compile` and `pnpm exec nx run @llm-transformer-diy/cli:test`.
