## ADDED Requirements

### Requirement: llm embeddings command surface

The **`llm`** CLI SHALL expose a top-level **`embeddings`** subcommand with these subcommands: **`encode`**, **`decode`**, **`similarity`**, **`expression`**, and **`nearest`**. The command SHALL print results to stdout (no full-screen wolf-tui), exit zero on success, and exit non-zero with an explicit error message on user-facing failures (missing checkpoint, malformed expression, vocab/embedding mismatch).

#### Scenario: Help lists subcommands

- **WHEN** the user runs **`llm embeddings --help`**
- **THEN** the help output SHALL list **`encode`**, **`decode`**, **`similarity`**, **`expression`**, and **`nearest`** with one-line descriptions

#### Scenario: Unknown subcommand fails loudly

- **WHEN** the user runs **`llm embeddings frobnicate`**
- **THEN** the process SHALL exit non-zero and print a message that names the unknown subcommand and lists valid ones

---

### Requirement: encode subcommand

**`llm embeddings encode <text...>`** SHALL load the checkpoint's tokenizer and token-embedding matrix, tokenize the input, look up each token's embedding row, and emit either a single mean-pooled vector (default) or one vector per token (with **`--per-token`**). It SHALL also emit a one-line input-expansion trace listing token ids and their surface forms.

#### Scenario: Single token encode

- **WHEN** the user runs **`llm embeddings encode king`** against a checkpoint whose tokenizer maps `king` to a single token
- **THEN** the process SHALL print one vector of length **`dModel`** to stdout and exit zero

#### Scenario: Multi-token mean pooling

- **WHEN** the user runs **`llm embeddings encode "new york"`** and the input expands to multiple tokens
- **THEN** the process SHALL print exactly one vector (the mean of the per-token embedding rows) and SHALL print an input-expansion trace naming the constituent tokens and noting that mean pooling was applied

#### Scenario: JSON output

- **WHEN** the user runs **`llm embeddings encode king --json`**
- **THEN** stdout SHALL contain a single JSON object with at minimum the keys `input`, `tokens` (array of `{id, surface}`), and `vector` (array of numbers)

---

### Requirement: decode and nearest subcommands

**`llm embeddings decode <vector-spec>`** SHALL accept either a word/phrase (which it encodes per the encode rules) or a vector loaded from a file via **`@<path>`**, then return the top-K tokens from the vocabulary ranked by cosine similarity to that vector. **`llm embeddings nearest <word>`** is a convenience alias for **`decode`** with a word input. Default **`--top-k`** is **10**.

#### Scenario: Top-K listing

- **WHEN** the user runs **`llm embeddings nearest king --top-k 5`**
- **THEN** stdout SHALL contain a 5-row ranked table with columns for rank, token surface, token id, and cosine similarity, ordered by descending cosine

#### Scenario: Vector-from-file input

- **WHEN** the user runs **`llm embeddings decode @./vec.json`** where `vec.json` contains `{ "vector": [...] }` of length **`dModel`**
- **THEN** the process SHALL rank the full vocabulary against that vector and exit zero

#### Scenario: Wrong-dimension vector errors

- **WHEN** the loaded vector has length **`!= dModel`**
- **THEN** the process SHALL exit non-zero with a message naming both dimensions

---

### Requirement: similarity subcommand

**`llm embeddings similarity <a> <b>`** SHALL compute the cosine similarity between the (mean-pooled) embeddings of two inputs and print a single scalar in **`[-1, 1]`** to stdout. Each input MAY be a word/phrase or a `@file` vector spec.

#### Scenario: Word-to-word similarity

- **WHEN** the user runs **`llm embeddings similarity king queen`**
- **THEN** stdout SHALL contain a single number in **`[-1, 1]`** and the process SHALL exit zero

#### Scenario: JSON shape

- **WHEN** **`--json`** is supplied
- **THEN** stdout SHALL contain `{ "a": ..., "b": ..., "cosine": <number> }`

---

### Requirement: expression subcommand and embedding expression language

**`llm embeddings expression "<expr>"`** SHALL parse and evaluate an arithmetic expression over token embeddings and return the top-K vocabulary tokens nearest the resulting vector by cosine similarity. The expression language SHALL support:

- Atoms: bare identifiers (e.g. **`king`**) and double-quoted strings (e.g. **`"new york"`**), each resolved to a vector via the same encoding rules as **`encode`** (mean pooling for multi-token inputs).
- Binary operators **`+`** and **`-`** (left-to-right associative).
- Unary **`-`** (negation).
- Parentheses for grouping.
- The lexer SHALL treat **`-`** as an operator only when it is whitespace-separated or adjacent to **`(`** / **`)`** / start-of-input; otherwise **`-`** is part of an identifier (so **`co-founder`** is a single atom).

By default the result list SHALL exclude any token whose id appears in the inputs of the expression (`--exclude-inputs`, default true; opt out with `--exclude-inputs=false`). Default **`--top-k`** is **10**.

#### Scenario: Classic analogy

- **WHEN** the user runs **`llm embeddings expression "king - man + woman"`** against a sufficiently trained checkpoint
- **THEN** stdout SHALL contain a top-10 ranked table where `queen` appears (typically near the top) and where `king`, `man`, `woman` are excluded by default

#### Scenario: Parentheses and unary minus

- **WHEN** the user runs **`llm embeddings expression "-(man) + woman + king"`**
- **THEN** the parser SHALL accept the expression, the evaluator SHALL produce a vector equivalent to `woman + king - man`, and the process SHALL exit zero

#### Scenario: Quoted multi-word atom

- **WHEN** the user runs **`llm embeddings expression '"new york" - usa + france'`**
- **THEN** the lexer SHALL parse `"new york"` as a single atom, mean-pool its tokens, and produce a top-K result list

#### Scenario: Operator vs identifier hyphen

- **WHEN** the user runs **`llm embeddings expression "co-founder + ceo"`**
- **THEN** the lexer SHALL treat **`co-founder`** as a single identifier (not as `co - founder`) and the evaluator SHALL produce one result vector

#### Scenario: Malformed expression

- **WHEN** the user runs **`llm embeddings expression "king +"`**
- **THEN** the process SHALL exit non-zero with a parser error message that points at the offending position

#### Scenario: Disable input exclusion

- **WHEN** the user runs **`llm embeddings expression "king - man + woman" --exclude-inputs=false`**
- **THEN** the result list MAY contain `king`, `man`, or `woman` if their cosine similarity ranks within the top-K

---

### Requirement: No ML logic in CLI for embeddings

The **`apps/cli`** package SHALL NOT implement vector arithmetic, cosine similarity, top-K selection, or expression evaluation directly; it SHALL delegate all such work to **`@llm-transformer-diy/inference`** (consistent with the existing "no ML logic in CLI" policy).

#### Scenario: Delegation

- **WHEN** any **`llm embeddings`** subcommand performs vector math or top-K ranking
- **THEN** that math SHALL be performed by **`@llm-transformer-diy/inference`** APIs, not reimplemented in **`apps/cli`**

---

### Requirement: Output format

By default the CLI SHALL render top-K results as an aligned table with columns `rank`, `token`, `id`, `cosine` and SHALL emit a structured JSON object (containing inputs, optional expression string, and `results` array of `{ token, id, cosine }`) when **`--json`** is supplied. Floating-point values SHALL be printed with at least 4 significant digits.

#### Scenario: Default table

- **WHEN** the user runs any subcommand that produces a top-K list without **`--json`**
- **THEN** stdout SHALL contain a header row and K data rows with the columns above

#### Scenario: JSON

- **WHEN** **`--json`** is supplied
- **THEN** stdout SHALL contain a single parseable JSON object with the documented keys
