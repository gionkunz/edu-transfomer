## Context

The repo trains GPT-style transformers from scratch (own tensors, autograd, WGSL kernels). A trained checkpoint contains a token-embedding matrix `tokenEmbed` of shape `[vocab, dModel]` (see `packages/model/src/gpt.ts`). The current CLI exposes `tokenizer`, `train`, `generate`, `chat`, and `inspect` (see `openspec/specs/cli-wolf-tui-commands/spec.md`), but offers no way to interrogate the embedding geometry — which is one of the most pedagogically rich artifacts of training.

This change adds an **`llm embeddings`** subcommand whose centerpiece is an expression evaluator: the user enters something like **`woman - man + king`** and gets back a top-K list of tokens nearest the resulting vector by cosine similarity, classic word2vec-style. This is purely an exploratory/inference-time tool — no training, no autograd, no logits — but it must use the existing tokenizer + checkpoint loader and respect the project's "own the stack" rule: no embedding/cosine library, hand-written parser.

## Goals / Non-Goals

**Goals:**

- A single `llm embeddings` command that lets a user encode text → vector, decode vector → tokens, compute pairwise cosine similarity, and **evaluate arithmetic expressions** over embeddings.
- A small expression language: atoms (words/quoted phrases), `+`, `-`, parentheses, and (optional) scalar multiplication; left-to-right associativity.
- Stay orchestration-only in `apps/cli` per existing policy; all math lives in `@llm-transformer-diy/inference`.
- Top-K result quality should reproduce textbook results when run against a sufficiently trained checkpoint (`king - man + woman ≈ queen`); when run against an undertrained one, results should still be deterministic and explainable.
- Deterministic output, machine-readable mode (`--json`).

**Non-Goals:**

- Sentence/sequence embeddings (mean of contextual states); only **token-level** embeddings from the checkpoint's input embedding matrix.
- Embedding fine-tuning or any gradient flow.
- Cross-checkpoint comparison or analogy benchmarks.
- A full DSL (no functions, no variables, no comparison operators); just additive arithmetic and parentheses.
- Visualization (PCA / UMAP / plots).

## Decisions

### Decision 1: Use the input token-embedding matrix `tokenEmbed`, not output projection or contextual states.

**Rationale:** The user's request — "embedding of woman" — implies a fixed per-token vector, which is exactly `tokenEmbed[id]`. Output `lmHead` is shape-compatible with weight tying but conceptually different; we keep this version dependent only on the input embedding to avoid silently changing semantics if a checkpoint is untied. **Alternative considered:** average of last-layer hidden states for a single-token forward pass — rejected because (a) it requires a GPU forward pass, (b) it's contextual, not a stable per-token "word vector", and (c) it's harder to reason about pedagogically.

### Decision 2: Multi-token inputs are reduced by **mean pooling** of their token embeddings.

When the user types `"new york"` and the tokenizer splits it into multiple BPE pieces, we average the embedding rows to produce a single vector. **Rationale:** simple, well-known baseline; preserves direction; documented behavior. **Alternative considered:** require single-token inputs and error out otherwise — too restrictive given BPE will fragment many common words. We will document this clearly and emit an info line like `"king" → 1 token, "queen" → 2 tokens (mean-pooled)`.

### Decision 3: Hand-written recursive-descent parser for the expression language.

Grammar (EBNF-ish):

```
expr   := term (('+' | '-') term)*
term   := factor                         // (scalar '*' factor)? if scalar mul is enabled
factor := atom | '(' expr ')' | '-' factor
atom   := word | quotedString
```

Tokens: identifiers (`[A-Za-z_][A-Za-z0-9_-]*`), quoted strings (`"…"`), numbers (only if scalar mul enabled), `+`, `-`, `(`, `)`, `*`. Whitespace separates tokens; identifiers are _not_ split on `-` (it must be surrounded by whitespace to act as an operator: `man-woman` is one identifier, `man - woman` is subtraction). **Rationale:** zero deps, ~80 lines, easy to test, fits "own the stack". **Alternative considered:** tiny PEG library — disallowed by dependency policy and overkill.

### Decision 4: Computation runs on **CPU** initially, GPU optional later.

The matrix is `[vocab, dModel]`, typically `vocab ∈ [1k..50k]` and `dModel ∈ [128..1024]` for this project's configs (`config/*.json`). Computing cosine similarity against the full vocab is a single dense matmul of `[1, d] · [d, V]` plus norms — ~50M FLOPs at the high end, sub-100 ms in JS. **Rationale:** keeps the path simple, avoids a WGSL kernel for a non-hot path, and produces predictable results without GPU init overhead for what is fundamentally an inspection tool. **Alternative considered:** dispatch via existing GPU matmul — rejected for v1; tracked as future work. We MAY pre-compute and cache row norms once per loaded checkpoint.

### Decision 5: Subcommand layout.

```
llm embeddings encode      <text...>            [--json] [--checkpoint <path>]
llm embeddings decode      <vector-spec>        [--top-k N] [--json]
llm embeddings similarity  <a> <b>              [--json]
llm embeddings expression  "<expr>"             [--top-k N] [--exclude-inputs] [--json]
llm embeddings nearest     <word>               [--top-k N] [--json]   # convenience for `decode <word>`
```

`<vector-spec>` is either a word/phrase (encoded on the fly) or `@path/to/file.json` containing `{ "vector": [..] }`. **Rationale:** mirrors existing tokenizer subcommand style; keeps `expression` explicit and discoverable while still supporting the simple `encode`/`decode`/`similarity` flows the user mentioned. **Alternative considered:** a single subcommand that auto-detects intent from input — rejected as too magic; users learn the surface faster with explicit verbs.

### Decision 6: `--exclude-inputs` defaults to **true** for `expression` and `nearest`.

Without this, `king - man + woman` returns `king` and `woman` at the top because they are literally the strongest contributors to the vector. **Rationale:** matches what the user actually wants to see in an educational demo; opt out with `--exclude-inputs=false`. We deduplicate by token-id, not surface form.

### Decision 7: Output format.

Default: aligned table `rank | token | id | cos`. JSON: `{ inputs: [...], expression?: string, results: [{ token, id, cosine }] }`. **Rationale:** human-friendly default, scriptable opt-in, no new dependencies.

## Risks / Trade-offs

- **Tokenizer/checkpoint mismatch** → loud error if `tokenizer.vocab.size ≠ tokenEmbed.shape[0]`; refuse to run rather than silently produce nonsense.
- **Multi-token mean-pooling can hide quality issues** → mitigated by always printing a one-line "input expansion" trace (token ids, surface forms, pooling note) before results, plus `--json` exposure of the same.
- **Untrained / undertrained checkpoint produces near-random nearest neighbours** → expected; no mitigation needed beyond clearly documenting that this is an inspection tool, not a benchmark. Output is still deterministic.
- **Expression-language ambiguity around `-`** (operator vs identifier character) → mitigated by the lexer rule that `-` is only an operator when surrounded by whitespace or adjacent to `(` / `)` / start; otherwise part of an identifier. We will unit-test this exhaustively.
- **CPU-only performance for very large vocab** → acceptable in the current configs; if it ever becomes a bottleneck we can dispatch through the existing GPU matmul, but that's tracked as a follow-up change, not part of this one.
- **Scalar multiplication is a nice-to-have** → behind a documented gate in the parser; we ship it only if it falls out cleanly, otherwise reject `*` with a clear error and revisit later.
