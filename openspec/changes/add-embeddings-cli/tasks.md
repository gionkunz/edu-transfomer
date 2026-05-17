## 1. Inference: embeddings core API

- [x] 1.1 Add `packages/inference/src/embeddings.ts` with skeleton exports: `lookupEmbedding`, `cosineSimilarity`, `topKByCosine`, `assertEmbeddingMatchesTokenizer`
- [x] 1.2 Implement `assertEmbeddingMatchesTokenizer(embeddingMatrix, tokenizer)` with a typed error class (e.g. `EmbeddingTokenizerMismatchError`)
- [x] 1.3 Implement `lookupEmbedding(matrix, ids, { pool: 'mean' | 'none' })` with id-range validation
- [x] 1.4 Implement `cosineSimilarity(a, b)` (CPU, Float32Array) with zero-norm guard
- [x] 1.5 Implement `topKByCosine(query, matrix, k, { exclude? })` with deterministic tie-breaking (ascending row index) and pre-computed row norms
- [x] 1.6 Re-export the new API from `packages/inference/src/index.ts`
- [x] 1.7 Vitest: `embeddings.spec.ts` covering identity cosine ≈ 1, orthogonal ≈ 0, exclude set, tie ordering, dimension mismatch, out-of-range id

## 2. Inference: embedding expression language

- [x] 2.1 Add `packages/inference/src/embedding-expression.ts` with a hand-written lexer for tokens: identifier, quotedString, `+`, `-`, `(`, `)`
- [x] 2.2 Implement the whitespace-aware `-` rule (operator only when whitespace-separated or adjacent to `(`/`)`/start; otherwise part of an identifier)
- [x] 2.3 Implement recursive-descent parser producing an AST (`Add | Sub | Neg | Atom`)
- [x] 2.4 Implement `evaluateEmbeddingExpression(ast, ctx)` where `ctx` resolves an atom string → `Float32Array` via tokenizer + mean-pooled `lookupEmbedding`
- [x] 2.5 Typed parser/evaluator errors with character offsets
- [x] 2.6 Re-export from `packages/inference/src/index.ts`
- [x] 2.7 Vitest: `embedding-expression.spec.ts` covering all spec scenarios (basic +/-, unary, parens, quoted multi-word, hyphen identifier, malformed inputs, unknown atom)

## 3. CLI: embeddings command wiring

- [x] 3.1 Create `apps/cli/src/cmd/embeddings.ts` exporting a Commander subcommand group
- [x] 3.2 Subcommand: `encode <text...>` with `--per-token`, `--json`, `--checkpoint <path>`, prints input-expansion trace
- [x] 3.3 Subcommand: `decode <vector-spec>` (supports `@<file>` JSON and bare word), `--top-k`, `--json`
- [x] 3.4 Subcommand: `similarity <a> <b>` with `--json`
- [x] 3.5 Subcommand: `expression <expr>` with `--top-k` (default 10), `--exclude-inputs` (default true), `--json`
- [x] 3.6 Subcommand: `nearest <word>` (alias-style; calls `decode` internally)
- [x] 3.7 Register the subcommand group in `apps/cli/src/cli-program.ts`
- [x] 3.8 Shared loader helper: load checkpoint config + tokenizer + token-embedding tensor via `@llm-transformer-diy/io`; call `assertEmbeddingMatchesTokenizer`
- [x] 3.9 Output renderer: aligned table (rank, token, id, cosine) + JSON variant
- [x] 3.10 Ensure no vector arithmetic / cosine / top-K math lives in `apps/cli` (delegate everything to `@llm-transformer-diy/inference`)

## 4. CLI: tests

- [x] 4.1 `apps/cli/src/cmd/embeddings.spec.ts`: snapshot/integration tests for `encode`, `decode`, `similarity`, `expression`, `nearest` against a tiny fixture checkpoint (or a stubbed loader)
- [x] 4.2 Test: unknown subcommand exits non-zero with helpful message
- [x] 4.3 Test: malformed expression exits non-zero and prints a parser error with position
- [x] 4.4 Test: vector-of-wrong-dimension input errors loudly
- [x] 4.5 Test: `--exclude-inputs=false` includes input tokens in results when applicable

## 5. Validation and polish

- [x] 5.1 `npm run build` clean
- [x] 5.2 `nx run-many -t test --all` green for `inference` and `apps/cli`
- [x] 5.3 `openspec validate add-embeddings-cli --strict` passes
- [ ] 5.4 Manual smoke: run `llm embeddings expression "king - man + woman" --top-k 10` against a small trained checkpoint and confirm output format
- [x] 5.5 Update CLI `--help` text to document the new subcommand group (no separate docs file)
