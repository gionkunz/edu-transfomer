## ADDED Requirements

### Requirement: Token embedding lookup API

**`@llm-transformer-diy/inference`** SHALL export a function (indicative name: **`lookupEmbedding`**) that, given a loaded checkpoint's token-embedding matrix and a sequence of token ids, returns either the per-token embedding rows or their mean as a single vector. The function SHALL validate that every input id is in **`[0, vocab)`** and SHALL throw a typed error otherwise.

#### Scenario: Single id

- **WHEN** `lookupEmbedding` is called with one token id
- **THEN** it SHALL return a Float32Array (or equivalent) of length **`dModel`** equal to the corresponding row of the token-embedding matrix

#### Scenario: Mean pooling

- **WHEN** `lookupEmbedding` is called with multiple ids and mean-pooling enabled
- **THEN** it SHALL return one vector of length **`dModel`** equal to the per-element mean of the selected rows

#### Scenario: Out-of-range id

- **WHEN** any input id is outside **`[0, vocab)`**
- **THEN** the function SHALL throw a typed error naming the offending id

---

### Requirement: Cosine similarity and top-K nearest neighbours

The package SHALL export **`cosineSimilarity(a, b)`** returning a scalar in **`[-1, 1]`** and **`topKByCosine(query, matrix, k, options?)`** returning the **K** rows of `matrix` with the highest cosine similarity to `query`, ordered by descending cosine. Results SHALL be deterministic and stable; ties SHALL break by ascending row index. **`options.exclude`** SHALL accept a set of row indices to exclude from the candidate set before ranking.

#### Scenario: Identical vectors

- **WHEN** `cosineSimilarity(v, v)` is called for any non-zero vector `v`
- **THEN** the returned value SHALL be within `1e-5` of `1.0`

#### Scenario: Zero vector

- **WHEN** either argument to `cosineSimilarity` has zero norm
- **THEN** the function SHALL throw a typed error rather than returning NaN

#### Scenario: Top-K excludes inputs

- **WHEN** `topKByCosine` is called with `options.exclude = new Set([3, 7])`
- **THEN** the returned list SHALL NOT contain entries with row index `3` or `7`

#### Scenario: Tie ordering

- **WHEN** two rows have identical cosine similarity to the query
- **THEN** the row with the smaller index SHALL appear first in the returned list

---

### Requirement: Embedding expression language parser and evaluator

The package SHALL export a parser and evaluator for a small expression language over embeddings (indicative names: **`parseEmbeddingExpression`**, **`evaluateEmbeddingExpression`**). The grammar SHALL be:

```
expr   := term (('+' | '-') term)*
term   := factor
factor := atom | '(' expr ')' | '-' factor
atom   := identifier | quotedString
```

The lexer SHALL recognise **`-`** as an operator only when whitespace-separated or adjacent to **`(`** / **`)`** / start-of-input; otherwise **`-`** is part of an identifier. Identifiers SHALL match **`[A-Za-z_][A-Za-z0-9_-]*`**. Quoted strings SHALL use double quotes and SHALL NOT support backslash escapes in v1 (a literal **`"`** inside cannot be expressed; documented as a v1 limitation). The evaluator SHALL resolve each atom to an embedding vector via the same lookup-and-pool rules as the lookup API and SHALL combine vectors via element-wise addition / subtraction / unary negation.

#### Scenario: Basic addition and subtraction

- **WHEN** `evaluateEmbeddingExpression("king - man + woman", ctx)` is called against a context that resolves each identifier to a vector
- **THEN** the returned vector SHALL equal `vec(king) - vec(man) + vec(woman)` element-wise

#### Scenario: Parentheses and unary minus

- **WHEN** `evaluateEmbeddingExpression("-(man) + woman + king", ctx)` is called
- **THEN** the returned vector SHALL equal `-vec(man) + vec(woman) + vec(king)`

#### Scenario: Quoted multi-word atom

- **WHEN** the input is `'"new york" - usa'`
- **THEN** the lexer SHALL emit a single quoted-string atom for `new york`, the evaluator SHALL mean-pool its tokens, and the result SHALL equal `meanPool(tokens("new york")) - vec(usa)`

#### Scenario: Hyphen inside identifier

- **WHEN** the input is `"co-founder + ceo"`
- **THEN** the lexer SHALL treat `co-founder` as a single identifier and the result SHALL equal `vec(co-founder) + vec(ceo)`

#### Scenario: Parser error reporting

- **WHEN** the input is malformed (e.g. trailing operator, mismatched parentheses, empty input)
- **THEN** `parseEmbeddingExpression` SHALL throw a typed error containing a 0-indexed character position pointing at or near the offending token

#### Scenario: Unknown atom

- **WHEN** an atom resolves to zero token ids (empty tokenization)
- **THEN** the evaluator SHALL throw a typed error naming the offending atom

---

### Requirement: Vocab/embedding consistency check

The package SHALL provide a guard (indicative name: **`assertEmbeddingMatchesTokenizer`**) that throws a typed error when the token-embedding matrix's first dimension does not equal the tokenizer's vocabulary size. Callers (CLI subcommands) SHALL invoke this guard before any embedding lookup or top-K query.

#### Scenario: Mismatch

- **WHEN** the embedding matrix has shape **`[V1, d]`** and the tokenizer's vocab size is **`V2 != V1`**
- **THEN** the guard SHALL throw a typed error whose message contains both **`V1`** and **`V2`**

#### Scenario: Match

- **WHEN** **`V1 == V2`**
- **THEN** the guard SHALL return without throwing
