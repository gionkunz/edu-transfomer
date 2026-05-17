## ADDED Requirements

### Requirement: Embeddings command

The CLI SHALL implement **`llm embeddings`** as a top-level subcommand group with subcommands **`encode`**, **`decode`**, **`similarity`**, **`expression`**, and **`nearest`**, as defined in the **`cli-embeddings-explorer`** capability. The command SHALL print to stdout (no full-screen wolf-tui), exit zero on success, and exit non-zero with an explicit error message on failure. Token-embedding lookup, cosine similarity, top-K ranking, and expression evaluation SHALL be performed by **`@llm-transformer-diy/inference`** (not reimplemented in **`apps/cli`**), preserving the existing "No ML logic in CLI" requirement.

#### Scenario: Encode prints a vector

- **WHEN** the user runs **`llm embeddings encode king`** against a valid checkpoint path
- **THEN** the process SHALL print the embedding vector (or its mean-pooled form) to stdout and exit zero

#### Scenario: Expression returns top-K

- **WHEN** the user runs **`llm embeddings expression "king - man + woman"`**
- **THEN** the process SHALL print a top-K ranked table of nearest tokens by cosine similarity to the resulting vector and exit zero

#### Scenario: Embeddings delegates math

- **WHEN** any **`llm embeddings`** subcommand performs vector arithmetic, cosine similarity, or top-K selection
- **THEN** that math SHALL be implemented in **`@llm-transformer-diy/inference`** and only invoked from **`apps/cli`**
