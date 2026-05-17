## Why

A trained transformer learns rich token-embedding geometry, but today there is no way to inspect that geometry from the **`llm`** CLI. Adding an `embeddings` command — including a small expression language for vector arithmetic like **`woman - man + king`** — turns the trained checkpoint into an exploratory, educational tool that lets a user _see_ what the model has learned about word relationships, in line with the project's goal of deeply understanding how LLMs work.

## What Changes

- New top-level CLI subcommand: **`llm embeddings`** with the following subcommands:
  - **`encode`** — encode text/word(s) and emit the embedding vector(s) (token-embedding lookups; one row per token from the checkpoint's token-embedding matrix).
  - **`decode`** — given a vector (file or inline) return the top-K nearest tokens by cosine similarity (inverse of `encode`).
  - **`similarity`** — compute cosine similarity between two inputs (each may be a word/phrase or `@file` vector); print scalar similarity.
  - **`expression`** (default subcommand if a positional expression is given) — evaluate a small arithmetic expression over embeddings (e.g. **`"woman - man + king"`**) and return the top-K nearest tokens by cosine similarity to the resulting vector.
- Small expression language (parser + evaluator) supporting:
  - Word/phrase atoms (resolved via the tokenizer to a single token id; multi-token inputs SHALL be averaged or rejected per a documented rule).
  - Operators **`+`** and **`-`** with left-to-right associativity.
  - Parentheses for grouping.
  - Optional scalar multiplication (e.g. **`0.5 * king`**) — nice-to-have, gated.
- Output options: **`--top-k`** (default 10), **`--exclude-inputs`** (default true; suppress the literal input tokens from the result list when it would otherwise dominate, à la the classic _king − man + woman ≈ queen_ demo), **`--json`** for machine-readable results, plain table by default.
- Reads the **token-embedding matrix** (and tokenizer) from a checkpoint loaded via `@llm-transformer-diy/io`. No training, no autograd, no GPU forward pass beyond a small matmul/cosine kernel that MAY use the existing GPU stack or a simple CPU reduction (decided in design).
- New embeddings logic lives in **`@llm-transformer-diy/inference`** (new module, e.g. `embeddings.ts`); **`apps/cli`** stays orchestration-only per existing CLI policy.

## Capabilities

### New Capabilities

- `cli-embeddings-explorer`: defines the **`llm embeddings`** subcommand surface, its subcommands (`encode`, `decode`, `similarity`, `expression`), the embedding-expression language (atoms, `+`, `-`, parentheses), output formats, and the orchestration-only contract with `@llm-transformer-diy/inference`.
- `inference-embeddings-api`: defines the inference-package API for token-embedding lookup, cosine similarity, top-K nearest-neighbour search over the embedding matrix, and expression evaluation, plus how multi-token inputs are reduced to a single vector.

### Modified Capabilities

- `cli-wolf-tui-commands`: extended with a new requirement section that the `llm` binary SHALL expose an **`embeddings`** subcommand that prints to stdout (no full-screen wolf-tui), and that it MUST delegate all embedding math to `@llm-transformer-diy/inference` (preserving "no ML logic in CLI").

## Impact

- **`apps/cli`**: new subcommand wiring under `apps/cli/src/cmd/embeddings.ts`, registered in `cli-program.ts`; new CLI smoke/integration tests.
- **`packages/inference`**: new `embeddings.ts` module exporting `lookupEmbedding`, `cosineSimilarity`, `topKByCosine`, `evaluateEmbeddingExpression` (names indicative). New unit tests.
- **`packages/io`** (read-only impact): used to load checkpoint config, tokenizer, and the token-embedding tensor. No format changes expected.
- **Dependencies**: none new — expression parsing is a tiny hand-written recursive-descent parser (consistent with "own the stack").
- **Docs**: brief CLI usage notes; canonical detail tracked in this change's specs.
- **Risk**: low — purely additive; the only subtle correctness concern is multi-token input reduction and tokenizer/embedding-matrix consistency between the loaded tokenizer and checkpoint.
