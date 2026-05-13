## 1. CLI wiring

- [x] 1.1 Register `llm chat` in `apps/cli/src/cli-program.ts` with `--model`, `--tokenizer`, `max-new-tokens`, `temperature`, `top-k`, `repeat-penalty`, and `repeat-window` mirroring `llm generate` (same parsing/validation messages for repetition options).
- [x] 1.2 Implement `runChatCommand` in a new `apps/cli/src/cmd/chat.tsx` (wolf-tui) and hook it from `cli-program.ts`.

## 2. TUI chat session

- [x] 2.1 Implement wolf-tui layout: transcript region + single-line user input; Enter submits non-empty drafts; block new submissions while assistant generation runs.
- [x] 2.2 Maintain a deterministic transcript with alternating `User:` / `Assistant:` segments (see change spec); before each assistant turn, append the latest user line ending with `Assistant:` and encode the full prefix; use a fresh `createGptKvCache` per `generateAutoregressive` call and always `destroyGptKvCache` on completion or error.
- [x] 2.3 Stream assistant text like `generate.tsx`: use `onSampledToken` and decode `[...promptIds, ...generatedIds]` with UTF-8 replace, showing only text after the prefix render length.
- [x] 2.4 If encoded prompt length plus `max-new-tokens` would exceed `model.config.maxSeqLen`, fail with a concise error (no silent truncation).
- [x] 2.5 Match `generate` startup ergonomics (`primeTuiTerminalCanvas`, software-adapter warning on stderr).

## 3. Verification

- [x] 3.1 Update `apps/cli/src/cli-program.spec.ts` expectations to include `chat`.
- [x] 3.2 Smoke: `pnpm exec nx run @llm-transformer-diy/cli:build`, apply `tools/patch-dist-esm-imports.mjs` if required, then confirm `node apps/cli/dist/main.js chat --help` lists flags.
