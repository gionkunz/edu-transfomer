## 1. Foundation: replace the pre-tokenizer policy in `bpe-tokenizer.ts`

- [x] 1.1 Rename `PRETOKENIZER_REGEX_WORD_AND_PUNCT_V1` to `PRETOKENIZER_REGEX_GPT2_V1` and set its value to `/'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu`. Remove the old regex constant entirely.
- [x] 1.2 Narrow `PretokenizerPolicy` to the single literal type `'gpt2-v1'`. Update the `KNOWN_PRETOKENIZER_LABELS` set to `new Set(['gpt2-v1'])`.
- [x] 1.3 Update the default value of `BPETokenizer.pretokenizerPolicy` (and every internal `policy: PretokenizerPolicy = '...'` literal in `runRamTrainingSync`, `runRamTrainingAsync`, `trainFromFiles`, `buildRegionBoundaryFromSlotStream`) from `'word-and-punct-v1'` to `'gpt2-v1'`.
- [x] 1.4 Rewrite `splitIntoRegions` to use the new regex. The function signature, allocation contract (one fresh array per call), and `regions.join('') === text` round-trip property remain unchanged.

## 2. Byte-level streaming boundary detector

- [x] 2.1 Replace `fillBoundaryFromUtf8Bytes` with a state-machine implementation matching the new regex. State: `{ IDLE, LETTER_RUN, DIGIT_RUN, OTHER_RUN, TRAILING_WS }`. Look-ahead: at most the next code point's class. Contraction look-ahead: when an apostrophe (`U+0027`) follows a letter run, peek the next 1–2 code points to decide whether to start an `OTHER_RUN` of just the apostrophe, or a region of `'s | 't | 're | 've | 'm | 'll | 'd`.
- [x] 2.2 Drop the now-unused `WORD_PUNCT_CP_CLASS_CACHE` and `classifyWordAndPunctCodePoint` helpers if their behaviour is fully subsumed by the state machine, OR retain them under new names if the state machine reuses the class lookup. Either way leave only one classifier in the module.
- [x] 2.3 Update `buildRegionBoundaryBitmap`'s in-string fast path to use the new regex via `splitIntoRegions`-equivalent `matchAll` loop.

## 3. JSON schema bump (v2 → v3)

- [x] 3.1 Change `JSON_VERSION` from `2` to `3`. Update the type `BPETokenizerJsonParsed['version']` and the comparison in `parseBpeTokenizerJson` accordingly.
- [x] 3.2 In `parseBpeTokenizerJson`, the unsupported-version error message SHALL name `3` as the supported version and instruct the caller to retrain. Mention that `version: 1` and `version: 2` are both rejected.
- [x] 3.3 In `parseBpeTokenizerJson`, the unknown-pretokenizer error message SHALL list `'gpt2-v1'` as the only supported label. The legacy `'word-and-punct-v1'` string falls through to this error path; verify the message names it explicitly.
- [x] 3.4 `BPETokenizer.save()` writes `version: 3` and `pretokenizer: 'gpt2-v1'`.

## 4. Tests in `packages/tokenizer/src`

- [x] 4.1 Update `__tests__/bpe-pretokenize.spec.ts`: delete the `word-and-punct-v1` partition golden, add the `gpt2-v1` partition goldens for `"Hello, world! It's 42 → done"` and `"foo \n  bar"` from the spec, and add the cross-region invariant test under the new regex (no learned merge ends in a letter byte and starts with a space byte except via the absorbed-leading-space branch — which is, by construction, the merge `(0x20, letter)` inside a single region; that's the new shape).
- [x] 4.2 Add a new test (same file or a sibling) that fuzzes ~5 000 random Unicode strings (mixed ASCII / multi-byte letters / digits / punctuation / contraction shapes / multiline indentation / mixed CRLF / trailing whitespace), runs both `splitIntoRegions` and `fillBoundaryFromUtf8Bytes` on each, and asserts the resulting `regionBoundary` `Uint8Array`s are element-wise equal. This is the load-bearing equivalence test from the design.
- [x] 4.3 Update `bpe-tokenizer.spec.ts`: bump every JSON literal that asserts `version: 2` to `version: 3`, every `pretokenizer: 'word-and-punct-v1'` to `pretokenizer: 'gpt2-v1'`, and rewrite the legacy-rejection test cases so they assert that `version: 2` files are rejected with the new error, and that `version: 3` + `pretokenizer: 'word-and-punct-v1'` is also rejected.
- [x] 4.4 Update any incremental-train / encode tests that asserted token-id goldens against the old policy. Either retrain the goldens against `gpt2-v1` or relax those assertions to round-trip / structural properties (`decode(encode(x)) === x`) where the exact ids do not carry the test's intent.
- [x] 4.5 Run `nx run @llm-transformer-diy/tokenizer:test` and confirm green.

## 5. Sweep call-sites and references

- [x] 5.1 `rg "word-and-punct"` across the repo and remove every remaining reference outside of archived OpenSpec changes (which are immutable). Expected hits: `bpe-tokenizer.ts`, `bpe-tokenizer.spec.ts`, `bpe-pretokenize.spec.ts`, `bpe-train-perf.spec.ts`, `bpe-train-incremental.spec.ts`, `bpe-encode-hot-loop.spec.ts`, `apps/cli/src/cmd/train-dashboard.tsx`, `apps/cli/src/cmd/encode-eta.ts`, `apps/cli/src/cmd/train-dash-types.ts`, `docs/project.md`, `README.md`.
- [x] 5.2 In each non-archived hit, replace the label / regex reference with `gpt2-v1` (or remove if the reference was purely descriptive prose about the removed policy).
- [x] 5.3 Confirm the CLI's `train` and `train-tokenizer` commands compile and the dashboard renders without referring to the removed policy label.

## 6. Validation against real corpora

- [x] 6.1 From a clean working tree, run `nx run @llm-transformer-diy/tokenizer:test` (passes after section 4) and `nx run-many -t test --all` (no other package depends on the policy label, but verify nothing surprises us).
- [x] 6.2 Retrain the project's reference tokenizer on `corpus/real/*.txt` to vocab 8 000 with the new policy via the CLI's `train-tokenizer` command, write to `out/tok-real-8000.json`, and inspect the result: confirm the vocab now contains tokens like ` the`, ` and`, ` of`, ` was`, that the count of pure-whitespace tokens is in the low single digits, and that `decode(encode(sample))` round-trips on a paragraph from the corpus.
- [x] 6.3 (Optional, for our own visibility) Repeat the `medium-longctx` 1 000-step training run with the freshly retrained tokenizer and record the new bits-per-character so we have a clean before/after comparison number.

## 7. OpenSpec validation

- [x] 7.1 `openspec validate switch-to-gpt2-pretokenizer --strict` passes with no warnings or errors.
- [x] 7.2 Once `speed-up-bpe-encode` is archived (or as part of archiving this change), confirm `openspec/specs/tokenizer-bpe-bootstrap/spec.md` ends up describing the `gpt2-v1` policy with `version: 3` JSON.
