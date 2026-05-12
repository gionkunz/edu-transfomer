## Context

The repo ships **`CharTokenizer`** with versioned JSON persistence and the shared **`Tokenizer`** contract in `packages/tokenizer`. Sequential planning (`docs/sequential-change-proposals.md` step 8) calls for **BPE** next to improve sequence length and lexical efficiency versus char vocabularies—still **pure TypeScript**, no forbidden tokenizer crates.

Training and inference pipelines already assume callers can **`train(..., vocabSize)`**, **`encode`**, **`decode`**, **`save`**, **`load`**, and observe **`vocabSize`**.

## Goals / Non-Goals

**Goals:**

- Byte-level **BPE tokenizer** compliant with **`Tokenizer`** and `docs/project.md` JSON shape (`merges`, `vocab` byte sequences keyed by numeric id, **`specialTokens`**, plus discriminator fields).
- **Deterministic** training and encode given fixed input text, `vocabSize`, specials list, randomness-free algorithm.
- **Stable id assignment**: specials occupy low ids in config order (**same spirit** as **`CharTokenizer` defaults**); unused slot budget when `actual vocab < vocabSize` behaves consistently with **`train`** contract (expose **effective** vocab size vs requested cap—see Risks).

**Non-Goals:**

- OpenAI GPT-2 / tiktoken-compatible pretokenizer and merge tables.
- Subword preprocessing for punctuation or languages beyond UTF-8 **encode as bytes → BPE merges**.
- Training-time automatic discovery/injection of special multi-byte strings beyond what `train` corpus supplies (unless explicitly spelled in spec/tests as simple behavior).
- Performance work (indexed counts, SIMD, workers).

## Decisions

1. **`train` corpus representation.**  
   **Choice:** Normalize input `text` to **UTF-8 bytes** (`TextEncoder`). Build pair statistics on sequences of byte tokens only; **do not** attempt to align UTF-16 `string` indexing with merges in v1 beyond correct UTF-8 round-trip guarantees for `encode`/`decode` of scalar strings Python/JS would emit.  
   **Rationale:** Matches standard byte-BPE narratives in `docs/project.md`/`docs/transformer-explained.md` and keeps merges well-defined across runtimes.

2. **Token id allocation.**  
   **Choice:**
   - Ids **`0 .. S - 1`**: **`specialTokens[i]`**, where `S = specialTokens.length` (must satisfy same validation rules as **`CharTokenizerOptions.specialTokens`**).
   - Ids **`S .. S + 255`**: singleton byte tokens **`b ∈ [0..255]`** mapped by **`id = S + b`**.
   - Ids **`S + 256` onward**: newly merged tokens in **merge iteration order**.

   Persisted **`vocab`** MUST map ids to contiguous byte sequences (**`Uint8Array` / numeric arrays in JSON**) that exactly equal the concatenation semantics used when performing merges.  
   **Alternatives rejected:** reserving bytes `0..255` without offset collides once specials occupy low ids; letting bytes share ids with specials breaks stable padding contracts.

3. **Merge ranking, training passes, and encoding.**  
   **Choice:** Record each merge as `[leftId, rightId]` plus implicit **`mergeRank`** equal to merge creation index (`0` first).

   During **training** each iteration recomputes pair counts on the full corpus tokenizer stream (**overlapping** adjacency counting), selects the maximizing pair (**ties:** smaller left id, then smaller right id), allocates the next vocab id representing **byte-sequence concatenation** of the two subtokens, then **rewrites the corpus stream** in a **single left-to-right pass** replacing **non-overlapping** occurrences of **`L R`** with the new id (greedy “first available match wins”).

   During **`encode`**, expand text to id bytes, then repeatedly pick among **present adjacent pairs** that have a merge the pair with **minimal `mergeRank`** (**ties:** smaller left, then right id), replace the **leftmost** such occurrence, and repeat until no merge applies.

   **Alternatives rejected:** “replace all overlapping pairs in parallel” (ambiguous) and undefined tie behavior (breaks golden tests).

4. **`train(text, vocabSize)` stopping rule.**  
   **Choice:**
   - If requested `vocabSize` is **below** minimal required **`S + 256`** (specials plus all byte singletons): **throw** with a concise error (tokenizer cannot physically represent UTF-8 bytes otherwise).
   - Stop merging when **`activeVocab.size === vocabSize`** **or** no pair occurs with frequency **≥ 2** (**cannot** shrink sequence further)—whichever occurs first.

   **`vocabSize` property** MUST equal final active id count (**contiguous `[0 .. active - 1]`** invariant).

   **Alternative considered:** permitting sparse id holes when user requests oversized `vocabSize`; rejected because callers use `vocabSize` to size embeddings.

5. **Persistence.**  
   **Choice:** JSON with top-level discriminator:
   - `kind: "llm-transformer-diy/bpe-tokenizer"`
   - `version: number` (start `1`)

   Payload includes ordered **`merges`** array of `[leftId, rightId]` (numbers), **`vocab`**: `{ [id: string]: number[] /* 0–255 bytes */ }` keyed as numeric id strings (JSON-safe), mirrored **`specialTokens`**, persisted **`byteOffset`** constant `S`, plus optional informational fields (`vocabRequested`, `mergeCount`) for debugging; such extras are non-normative unless exercised by acceptance tests.

   **`load`** rejects unknown **`kind`/version**.

6. **`encode`/`decode` error surface.**  
   **Choice:**
   - `encode` refuses when tokenizer untrained (**same pattern** as **`CharTokenizer`** throwing on missing training).
   - `decode`: invalid ids throw; invalid UTF-8 byte sequences reconstructed from merges throw with actionable message (**UTF-8 decode strict**).

## Risks / Trade-offs

**[Risk] Incorrect or ambiguous tie-breaking ⇒ flaky tests.** → Mitigation: spec scenarios lock tie cases; utilities factor tuple comparison.

**[Risk] Oversized `vocabSize` versus achievable merges ⇒ confusion.** → Mitigation: document that training stops early; **`vocabSize` reflects reachable ids**, not padded tables (match char tokenizer precedent for vocab meaning).

**[Risk] Applying merges differs from GPT-2 pretokenizer tokenization ⇒ user expectations.** → Mitigation: proposal **`Explicitly out of scope`** + docs cross-link.

**[Risk] Giant corpora ⇒ slow pure-JS training.** → Mitigation acceptable for MVP; defer perf change.

## Migration Plan

Additive package surface: **existing `CharTokenizer` consumers unaffected**. Consumers opt into **`BPETokenizer`** by import. No rollout/rollback complexity beyond semver as usual inside monorepo.

## Open Questions

- Whether **`encode`** should split user-provided literals of special-token strings (`"<|pad|>"`) into specials automatically is **unset** unless the spec demands it—default design **does not**, to avoid brittle longest-match overlaps; resolve via smallest follow-up change if workflows need literal injection.
