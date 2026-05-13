## MODIFIED Requirements

### Requirement: Deterministic encoding with merge table

**`encode`** SHALL map input text to the same initial byte-id sequence as training (UTF-8 via **`TextEncoder`**, ids **`S + b`**), then repeatedly apply learned merges until no merge rule matches an adjacent pair:

1. Among all adjacent pairs present in the sequence that correspond to a learned merge, select the pair whose saved **merge rank** (**0-based order merges were created**) is **minimal**.

2. If several candidate pairs share the minimal rank, select the pair with smaller left id, then smaller right id.

3. Replace **all non-overlapping left-to-right occurrences** of the selected pair with its merged id in a **single in-place pass**: walk the sequence left to right, and at each position **`r`** either consume two slots (`buf[r]`, `buf[r+1]`) and emit one **`mergedId`** (when they equal the selected pair) or copy `buf[r]` through unchanged. **Non-overlap** is enforced by advancing the read cursor by 2 on a match and by 1 otherwise.

4. Repeat until no merge applies.

The resulting token id sequence SHALL be **identical to** the sequence produced by repeatedly selecting the minimum-rank pair (steps 1–2) and replacing its **leftmost** occurrence one at a time until no merge applies. Equivalence holds because non-overlapping occurrences of a fixed pair `(L, R)` are mutually independent — merging one does not introduce, remove, or alter any other occurrence of `(L, R)` — so collapsing all of them in one pass produces the same buffer state as collapsing them serially.

#### Scenario: Encode after training round-trips through decode for representative text

- **WHEN** a trained tokenizer runs **`decode(encode(x))`** for **`x`** equal to a short Unicode string containing spaces, digits, and non-ASCII characters (e.g. **`"héllo 世界"`**)
- **THEN** the result SHALL equal **`x`**

#### Scenario: Encode output is identical to leftmost-only baseline

- **WHEN** a trained tokenizer encodes a representative input containing **multiple non-overlapping occurrences of the same lowest-rank pair** (e.g. a string where the same byte digram appears several times separated by other bytes)
- **THEN** the resulting **`number[]`** SHALL be element-wise equal to the result a reference loop would produce when it selects the minimum-rank pair and replaces only its **leftmost** occurrence per outer iteration until no merge applies

## ADDED Requirements

### Requirement: Typed-buffer encode entry points

**`BPETokenizer`** SHALL expose typed-buffer encode entry points that avoid the boxed-Number round-trip used by the **`Tokenizer`**-contract `encode` / `encodeAsync`:

- **`encodeUint32(text, hooks?): Uint32Array`** — synchronous; returns a freshly allocated **`Uint32Array`** whose **`length`** equals the logical token count and whose elements are the encoded token ids in order.

- **`encodeUint32Async(text, hooks?): Promise<Uint32Array>`** — equivalent async entry point that participates in the same yield-cadence contract as **`encodeAsync`** (see _Async encode yield cadence preserves host UI responsiveness_).

The typed entry points SHALL produce token ids that are **element-wise equal** to the corresponding **`encode`** / **`encodeAsync`** output for the same input text. The existing **`Tokenizer`**-contract methods (**`encode(text): number[]`** and **`encodeAsync(text): Promise<number[]>`**) SHALL remain available and SHALL be implemented in terms of the typed entry points (e.g. by `Array.from`-ing the returned **`Uint32Array`**), so the **`Tokenizer`** interface and all current callers continue to work without modification.

#### Scenario: Typed and untyped entries agree

- **WHEN** a trained tokenizer encodes the same input text via **`encodeUint32(x)`** and via **`encode(x)`**
- **THEN** the **`Uint32Array`** result and the **`number[]`** result SHALL have the same length and SHALL contain element-wise equal numeric ids in the same order

#### Scenario: Typed result is independently owned

- **WHEN** **`encodeUint32(x)`** returns a **`Uint32Array`**
- **THEN** mutating that buffer SHALL NOT affect a subsequent encode call's output, i.e. each call SHALL allocate a fresh **`Uint32Array`** sized to the logical token length (not a `subarray` view into shared storage)

---

### Requirement: Async encode yield cadence preserves host UI responsiveness

**`encodeAsync`** and **`encodeUint32Async`** SHALL periodically yield to the host event loop so that wolf-tui dashboards can repaint and **Ctrl+C** stays responsive during long encodes. The yield cadence SHALL be governed primarily by a **wall-time slice budget**: after the running outer loop has consumed roughly **`TIME_SLICE_ENCODE_ASYNC_MS`** of wall time without yielding, the next yield SHALL fire. A **coarse outer-pass counter** MAY be used as a secondary trigger but SHALL NOT fire on every individual pair merge.

`onProgress` events of `phase: 'merge'` SHALL report **`mergeIteration`** as the count of **outer best-rank passes performed so far** (each pass selects the lowest-rank pair and collapses all of its non-overlapping occurrences in one walk). The pessimistic upper bound `max(1, initialTokenCount - 1)` SHALL remain a valid worst-case upper bound on **`mergeIteration`** because each pass removes at least one slot in the worst case.

#### Scenario: Long encode yields within a bounded wall-time slice

- **WHEN** **`encodeAsync(x)`** runs on an input large enough to require many outer passes
- **THEN** the host event loop SHALL receive a yield (via the existing **`yieldEventLoopForHostUi`** helper) at intervals not exceeding `TIME_SLICE_ENCODE_ASYNC_MS` of accumulated synchronous work, so a host UI tick scheduled in parallel SHALL be observed before the encode completes

#### Scenario: Progress reports outer-pass counts

- **WHEN** an **`encodeAsync`** call with an `onProgress` hook completes
- **THEN** every emitted **`'merge'`** event's **`mergeIteration`** SHALL be **monotonically non-decreasing** and SHALL count **outer best-rank passes**, not single-position merges; and the final **`mergeIteration`** SHALL be **≤ `max(1, initialTokenCount - 1)`**
