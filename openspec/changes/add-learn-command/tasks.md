# Tasks

## 1. Model: lesson capture in the decode step

- [x] 1.1 Add `lessonTrace?: boolean` to `ForwardGptDecodeStepOptions` and `inputVectors`, `allQueries`, `allScores`, `allMixed` to `GptDecodeStepResult` in `packages/model/src/gpt-kv.ts` (implies `flowTrace`); collect the per-head `q`, scaled `scores` and `mixed` variables in `mhaForwardDecode` and the two embedding rows in `forwardGptDecodeStep`, stack and read them back like `allAttention`. Verify: `nx run model:test` passes with a new test asserting the four shapes for a 2-layer, 2-head model.
- [x] 1.2 Add `readKvCacheHead(rt, cache, layerIndex, headIndex)` returning `{ keys, values }` as `[seqLen, dK]` arrays, with range errors for bad indices; export it from the package index. Verify: test that after prefill + decode the returned row count equals `cache.seqLen` and an out-of-range head throws.
- [x] 1.3 Add consistency tests: (a) logits with and without `lessonTrace` are equal for the same cache state; (b) `allScores` equals `query · key / √dK` using `readKvCacheHead`; (c) softmax of `allScores` equals `allAttention`; (d) `allMixed` equals weights · values; (e) buffers checked out after dispose are unchanged versus a run without capture. Verify: `nx run model:test` passes.

## 2. Inference: lesson trace and sampling breakdown

- [x] 2.1 Factor the temperature/top-k shaping in `packages/inference/src/sampling.ts` into one internal function used by `sampleLogitsWithProbs`, and add `explainSampling(logits, opts, topN)` returning per-token raw / tempered / kept / final values, an "others" remainder and the full final distribution. Verify: existing `sampling.spec.ts` passes unchanged, and new tests cover greedy (temperature 0), top-k off, and equality of `explainSampling().final` with `sampleLogitsWithProbs().probabilities`.
- [x] 2.2 Add `packages/inference/src/lesson-trace.ts` with `runLessonForward(rt, model, promptIds)` (prefill of all but the last token, skipped for one token; instrumented decode; logits readback; graph disposal) returning `{ trace, cache }`, where `trace` exposes `flow`, `rawLogits`, `inputVectors` and `head(l, h)` → `{ query, scores, weights, mixed, peakIndex, entropy }`. Verify: tests for a 1-token and a multi-token prompt assert shapes, that `trace.flow` equals `buildFlowTrace` output for the same forward, and that `rawLogits` equals the logits `generateAutoregressive` samples its first token from.
- [x] 2.3 Add `readLessonHeadKv`, `mostFocusedHead(trace, layerIndex)`, a vector-bucketing helper for display strips, and a visible-window summary (`visibleTokenWindow(weights, rows)` → shown indices, hidden count, hidden weight total); export everything from `packages/inference/src/index.ts`. Verify: unit tests for each, including hidden weight + shown weights summing to 1.

## 3. CLI: lesson framework

- [x] 3.1 Add `apps/cli/src/cmd/learn-lesson.ts` (types `Chapter`, `Step`, `StepView`, `LessonContext`; chapter registry; helpers to count steps and compute overall progress). Verify: unit test builds a two-chapter fixture registry and checks step counting and progress.
- [x] 3.2 Add `apps/cli/src/cmd/learn-state.ts` with `lessonReducer` and all actions from design decision 4, including `--no-animation` behaviour, head/block wrap-around, temperature bounds, top-k cycling and `restartWith`. Verify: `learn-state.spec.ts` covers every scenario of the spec requirements "Lesson structure and navigation", "Step animations" and "Selecting block and head" that does not need a screen.
- [x] 3.3 Add drawing helpers in `apps/cli/src/cmd/learn-draw.ts`: vector strip (magnitude shade + sign colour), centred signed bar, token row label, header (chapter, step, progress) and footer (active keys, current `↑/↓` meaning). Verify: unit tests on returned strings/spans, including that header and footer fit 80 columns.
- [x] 3.4 Add `apps/cli/src/cmd/learn-view.tsx`: wolf-tui component holding reducer state, key mapping per design decision 7, one interval timer dispatching `tick` while playing, the below-80×24 message, and painting of `StepView` (span rows or `FlowPanel` props). Verify: `nx run @llm-transformer-diy/cli:build` succeeds and lint passes.

## 4. CLI: chapters

- [x] 4.1 Implement the "One attention head" chapter (`learn-chapter-attention.ts`): five steps (query, keys, scores, weights, mix) with frame counts, explanation text built from real token pieces and values, lazy key/value loading state, and the long-prompt window. Verify: tests render every step's final frame at 80×24 from a fixture trace and assert row count ≤ 16, width ≤ 80, weights summing to 1 in the text, and the hidden-token summary line for a long prompt.
- [x] 4.2 Add optional `reveal` prop to `FlowPanel` in `generate-panels.tsx` with unchanged default rendering. Verify: existing CLI tests pass untouched and a new test shows that a full `reveal` draws the same block steps as no prop.
- [x] 4.3 Implement the "Stack of blocks" chapter (`learn-chapter-stack.ts`): steps for stream entry, one block's two updates, and best guess per block, animated by sweeping `reveal`; focused block follows `layerIndex`. Verify: tests assert the reveal sequence per frame and that the last block's guess equals the top token of `explainSampling` raw probabilities for the fixture.
- [x] 4.4 Implement the "Picking the next token" chapter (`learn-chapter-sampling.ts`): steps raw, temperature, top-k, draw; bar interpolation on temperature change; `r` redraw; removed candidates marked. Verify: tests assert that lowering temperature does not lower the top token's shown probability, that top-k marks exactly `k` kept rows, and that the drawn token is one of the kept tokens.
- [x] 4.5 Check every step's explanation text against the code path it describes in `gpt-kv.ts` and `sampling.ts` (no bias, RMSNorm, learned positions, `1/√dK` scaling, temperature before top-k or the actual order used). Verify: a test asserts each step has 1–4 explanation lines of ≤ 78 characters, and the review is noted in the PR/commit message.

## 5. CLI: command wiring

- [x] 5.1 Add `apps/cli/src/cmd/learn.tsx` (`runLearnCommand`): TTY check before GPU init, runtime/checkpoint/tokenizer loading as in `generate.tsx`, prompt tokenization and length check, `runLessonForward`, initial head via `mostFocusedHead`, render, `n` restart (destroy cache, re-run with the extended prompt), cleanup on quit. Verify: unit test of the non-TTY and prompt-too-long error paths with injected stdio/tokenizer.
- [x] 5.2 Register `learn [prompt...]` in `apps/cli/src/cli-program.ts` with `--model`, `--tokenizer`, `--temperature`, `--top-k`, `--dtype`, `--no-animation` and a default prompt. Verify: `cli-program.spec.ts` gains cases for option parsing and defaults, and `node dist/apps/cli/main.js learn --help` lists them after a build.
- [x] 5.3 Add an "Example: learn how the model predicts" section to `README.md` and `learn` to the CLI overview table, with the command and the key list. Verify: the documented command runs as written against a local checkpoint.

## 6. Integration check

- [ ] 6.1 Run `npx nx run-many -t test --all`, lint and `npm run build`. Verify: all projects pass.
- [x] 6.2 Drive the built CLI in a pseudo-terminal on a small checkpoint (software WebGPU is fine): step through all chapters with `→`, change block/head, change temperature and top-k, redraw, continue with `n`, resize below 80×24 and back, quit. Verify: captured frames show each spec scenario's outcome, no stray output above the lesson, and the terminal is restored on quit.
- [x] 6.3 Run `llm learn` headless-safe checks against a real GPU checkpoint via the bundled build: load, `runLessonForward`, print the trace shapes from a small script. Verify: shapes match the model config and no GPU validation errors are logged.
- [x] 6.4 Run `openspec validate add-learn-command --strict`. Verify: it reports the change as valid.
