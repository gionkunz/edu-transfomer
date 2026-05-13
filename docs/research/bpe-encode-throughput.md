# Research: BPE encode throughput during training

> Status: research notes (no code changes proposed yet). Gathered from a code
> walk of `packages/tokenizer` + `apps/cli/src/cmd/train.tsx` and a literature
> survey of fast/parallel/GPU BPE implementations. The intent is to give a
> follow-up OpenSpec change a clear menu of options ranked by effort and ROI.

## 1. Symptom recap (from the user report)

- During `llm train` the corpus encode (`BPETokenizer.encodeAsync`) phase runs
  for "ages" with very low resource pressure:
  - CPU ≈ 10% on a 16-core box (i.e. ~1 core saturated, the rest idle).
  - GPU 0% — no WebGPU work is dispatched while encoding.
- Memory is not reported as a problem.

The combination of low CPU and zero GPU is the classic fingerprint of a
single-threaded JavaScript hot loop with no kernel offload — _not_ an I/O wait
and not a memory thrash. Nothing is "saturating" because nothing else is
participating: one V8 main thread is doing all the work.

## 2. What the encode actually does today

The training command joins every corpus file into one giant string and hands
the whole thing to `BPETokenizer.encodeAsync`:

```195:211:apps/cli/src/cmd/train.tsx
    const corpus = textParts.join('\n');

    console.error(
      `train: encoding corpus (${String(corpus.length)} chars); BPE merges stream progress in the dashboard…`,
    );

    let expandChars = corpus.length;
    let expandBytes = corpus.length;

    const encodeWallStartMs = Date.now();
    let initialTokForEncode = 0;
    let lastMergeSampleWallMs = encodeWallStartMs;
    let lastMergeIterForRate = 0;
    let mergesPerSecEma = null;
    let mergeRateSamples = 0;

    const tokenIdList = await tokenizer.encodeAsync(corpus, {
```

`encodeAsync` then loops over `performSingleGreedyMergeUint32`:

```333:389:packages/tokenizer/src/bpe-tokenizer.ts
  async encodeAsync(text: string, hooks?: BpeEncodeHooks): Promise<number[]> {
    this.assertTrained();
    const emit = hooks?.onProgress;
    const onYieldCb = hooks?.onYield;
    const sliceYield = async (): Promise<void> => {
      await yieldEventLoopForHostUi();
      onYieldCb?.();
    };
    const S = this.specialTokens.length;
    const bytes = this.textEncoder.encode(text);
    const n = bytes.length;
    const buf = new Uint32Array(n);
    for (let i = 0; i < n; i++) {
      buf[i] = S + bytes[i];
    }
    let len = n;
    // ...
    while (true) {
      const nextLen = this.performSingleGreedyMergeUint32(buf, len);
      if (nextLen === null) break;
      len = nextLen;
      mergeIteration += 1;
      // ... emit + setTimeout(0) yield
    }
```

And the per-iteration kernel:

```239:284:packages/tokenizer/src/bpe-tokenizer.ts
  private performSingleGreedyMergeUint32(buf: Uint32Array, len: number): number | null {
    if (len < 2) return null;

    let minRank = Infinity;
    let chosenL = Infinity;
    let chosenR = Infinity;

    for (let j = 0; j < len - 1; j++) {
      const L = buf[j];
      const R = buf[j + 1];
      const rk = nestedNumberGet(this.mergeRankByPair, L, R);
      if (rk === undefined) continue;

      if (rk < minRank || (rk === minRank && (L < chosenL || (L === chosenL && R < chosenR)))) {
        minRank = rk;
        chosenL = L;
        chosenR = R;
      }
    }
    // ... linear scan to find the FIRST occurrence, then copyWithin to delete one slot
  }
```

### What this means in big-O terms

Let `N` be the number of byte tokens in the joined corpus (≈ corpus byte
length) and let `M` be the number of merges performed end-to-end (each merge
removes exactly one slot, so `M ≤ N − 1`; for typical English text with a
~10k vocab `M ≈ 0.4N–0.6N`).

| Step                          | Cost per merge | Total cost                   |
| ----------------------------- | -------------- | ---------------------------- |
| Find best (lowest-rank) pair  | `O(N)` scan    | `O(N · M)`                   |
| Find first occurrence of pair | `O(N)` scan    | `O(N · M)`                   |
| `copyWithin` shift left by 1  | `O(N)` memmove | `O(N · M)`                   |
| **End-to-end**                |                | **`O(N · M)`** ≈ **`O(N²)`** |

For our `corpus/real/` (~3 MB ≈ 3.1M bytes), this is roughly **3M × 1–2M ≈
3×10¹²–6×10¹² primitive operations** — a lower bound of _minutes-to-hours_ of
single-threaded work even in optimized V8, before counting Map-lookup constants
and GC pressure.

### Ancillary issues found in the same pass

- **Single merge per pass is left on the table.** `performSingleGreedyMergeUint32`
  finds _the_ best-rank pair, then merges only the **first** matching position,
  even though that exact pair almost always recurs many times across the
  buffer. minBPE / Karpathy and SentencePiece-CPU collapse all occurrences of
  the chosen pair in the same pass — same total work, but ~0.5×–10× fewer
  outer iterations on natural text.
- **Per-iteration `setTimeout(0)` yield.** `encodeAsync` calls `sliceYield`
  every ~50 ms or on every progress emit (`EMIT_MERGE_EVERY_ASYNC = 512`).
  Each yield is a full event-loop trip (~hundreds of µs on Node), and on a
  hot inner loop those add up. Useful for TUI repaint, but worth measuring —
  the synchronous `encode()` path may actually be faster per-merge.
- **Final `Array.from(buf.subarray(0, len))`** at the end of `encode` /
  `encodeAsync` allocates a JS Array of millions of boxed numbers and then
  the caller (`train.tsx`) immediately re-packs them into a `Uint32Array`.
  Not the bottleneck, but wasted bandwidth and GC pressure during a phase
  where memory bandwidth is already the limiting factor.
- **Whole corpus is one chunk.** No pre-tokenization. The merge loop is
  forced to scan across `\n`-joined documents; a regex-based or even
  whitespace-based pre-split would shrink each "BPE problem" from `N`
  bytes to ~10–30 bytes (a typical pre-token), cutting both `N` and `M`
  to per-piece values where naive `O(N²)` is essentially free.
- **`mergeRankByPair` is `Map<number, Map<number, number>>`.** Generic JS
  `Map` lookups are object-shape-friendly but ~5–10× slower than typed-array
  access; this is a constant-factor item but visible in profiles for hot
  inner loops with a 256+ vocab.

## 3. Why CPU is "only" 10% and GPU is 0%

| Observation              | Mechanical reason                                                                                                                        |
| ------------------------ | ---------------------------------------------------------------------------------------------------------------------------------------- |
| ~1 of 16 cores busy      | Encode runs on the V8 main isolate. There is no `worker_threads` use, no `Atomics`, no shared buffer. Pure single-thread JS.             |
| GPU 0%                   | Nothing in `BPETokenizer` touches the `TensorRuntime`. The encode never enqueues a WebGPU command. The GPU only wakes up at `trainStep`. |
| Even the 1 core not 100% | Each `setTimeout(0)` yield parks the loop, lets libuv/event loop run, then resumes. Wall time includes idle gaps, lowering observed %.   |
| No I/O wait              | The corpus is read once (`Promise.all(files.map(readFile))`) and held in memory.                                                         |

So the system is correctly reporting the truth: **the hot loop is sequential
JS scalar work with periodic event-loop trips**. There is nothing for the
other 15 cores or the GPU to do unless we restructure the algorithm.

## 4. State of the art (literature survey)

What follows is condensed from the references at the bottom. Numbers are
order-of-magnitude.

### 4.1 Algorithmic wins (single-threaded)

- **Priority queue + linked list (`O(N log N)`).** SentencePiece's
  `bpe_model.cc`, `tiktoken`'s `_byte_pair_merge`, HuggingFace `tokenizers`,
  and `rust-tokenizers` all use this shape. A min-heap keyed by merge rank
  drives candidate selection (`O(log N)` instead of `O(N)`); a doubly-linked
  list of symbols supports `O(1)` merge and predecessor/successor lookup.
  Lazy invalidation (`size` stamps or generation counters on heap entries)
  lets stale entries be discarded in `O(log N)` without expensive heap
  searches. **Reported speedups vs naive: 10–100×** on long inputs; the
  Rust-NLP benchmark shows the priority-queue/linked-list variant beats
  naive-pre-split on inputs ≥10 lines and is the only safe choice past 100
  lines (naive: 18.6 s for 1000 lines vs 12.8 ms — ~1450×).
- **Heap-based BPE _training_** (Sennrich-style PR to `sentencepiece`)
  brings per-merge cost from `O(N)` to `O(log N)` — 10–24× on large training
  runs. We do not currently have this problem (training the tokenizer is
  cheap) but the same data structures help both train and encode.

### 4.2 Pre-tokenization (still single-thread, near-free)

- All production BPE encoders pre-split the input (regex or whitespace+
  punctuation) before running merges. tiktoken's regex chops into pieces of
  typical size 1–30 bytes; merges then run independently per piece.
- Why it matters: BPE merges are _local_ — by construction, no learned merge
  in `mergeRankByPair` ever crosses a regex boundary, so splitting cannot
  change the encoding (assuming the same regex was used at train time, or
  byte-level pre-tokenization is acceptable).
- Pre-tokenization also enables the next two wins (cache + parallelism),
  because each piece becomes an independent unit of work.

### 4.3 Per-piece encode cache

- Natural language corpora are heavily Zipfian; a small LRU keyed by the
  raw bytes of each pre-token typically catches >90% of pieces after a few
  thousand documents.
- Both `tiktoken` (DashMap on rank, FxHash) and HuggingFace use a cache.
  The wall-time reduction is large because the cached pieces skip the
  inner merge loop entirely.

### 4.4 Process-level parallelism (Node)

- `node:worker_threads` lets you run JS on multiple OS threads with a
  shared `SharedArrayBuffer`. Recommended pattern for CPU-bound JS:
  spin up one worker per physical core, ship work via a queue.
- Once pre-tokenization splits the corpus into independent pieces, this
  is embarrassingly parallel: each worker takes a contiguous slice of
  bytes (snapped to pre-token boundaries) and emits an `(offset, ids)`
  pair. The main thread concatenates token arrays in order.
- Expected speedup at `K` workers: between `0.6K` and `0.9K` (synchronization
  overhead is small because the pieces are tiny).
- Riptoken (Rust+rayon clone of tiktoken) reports up to 7× speedup vs
  single-threaded tiktoken on batch encode.

### 4.5 Regex/pretok engine

- tiktoken's #331 PR (replacing `fancy-regex` with the linear-time
  `regex` crate) gave a ~6× total runtime improvement because pre-tok was
  ~90% of runtime in their workload after the merge loop was already
  optimized. We don't have a regex pre-tok yet, but if we adopt one we
  should pick a linear-time DFA engine (e.g. `re2`-style) rather than a
  PCRE-with-backtracking implementation.

### 4.6 GPU implementations

- **BlockBPE (You, 2025)** is the current SOTA for GPU BPE. It throws away
  regex pretok and replaces it with byte-level pretok so that all merges
  can be run in parallel via warp-level prefix sums and a GPU-resident
  hashmap (cuCollections). Achieves `O(n·d)` (`d ≪ n`) and 2–2.5× higher
  throughput than tiktoken / HF tokenizers — but **only at high batch sizes
  with long sequences** (e.g. 1024 strings × 1024 bytes). Below those sizes
  the GPU is starved and CPU wins. Quality is _not_ identical: byte-level
  pretok hurts numerical-token quality measurably (GSM8K accuracy drop
  from 0.78 → 0.22 in their paper). Trades clarity for throughput in a
  way that is questionable for an educational repo.
- **NVIDIA RAPIDS / cuDF** ships a GPU subword tokenizer (BERT WordPiece,
  not BPE) reported at 270× over CPU. BPE is on their roadmap but
  incomplete.
- For our project the relevant question is whether a hand-rolled WGSL BPE
  kernel makes sense. It is _technically_ in scope (we own the GPU stack
  already), but: (a) the wins only appear on inputs and batch shapes far
  beyond what a learning project needs, (b) implementing a GPU hashmap +
  prefix-sum compaction is a major work item, and (c) the corpus encode
  is a one-shot per training run that is then cached on disk — so absolute
  wall time matters more than throughput per dollar of GPU.

## 5. Diagnosis: why is _our_ implementation slow?

Combining §2 and §4, the picture is unambiguous:

1. **Algorithmic complexity (~`O(N²)`)** is the dominant factor. A 3 MB
   corpus encoded with a naive single-merge-per-pass scanner is fundamentally
   minutes-to-hours of work.
2. **Single-threaded, no offload** is why the system shows 10% / 0%. Every
   other observation (low GPU, idle cores, periodic event-loop yields) is
   downstream of "all the work is in one V8 isolate".
3. **Constant factors** (no pre-tokenization, no cache, generic Map lookups,
   `Array.from` round-trips) make the `O(N²)` curve start higher and grow
   faster. They are 2–10× each, but they _multiply_ with the algorithmic
   problem.

The on-disk **token cache** (`apps/cli/src/cmd/corpus-token-cache.ts`) is
a real product-level mitigation — the second run is instant — but it does
not help the _first_ run, which is exactly where the user gets stuck.

## 6. Options ranked by ROI

The numbers below are wall-time multipliers I expect from each option,
assuming the others have not been applied yet. Multipliers compose
sub-linearly (you don't multiply 100× by 10× and get 1000×).

| #   | Option                                                                      | Effort | Expected wall-time win | Resource use after                 |
| --- | --------------------------------------------------------------------------- | ------ | ---------------------- | ---------------------------------- |
| 1   | Priority queue + linked-list encode                                         | M      | **30–300×**            | 1 core saturated, GPU idle         |
| 2   | Pre-tokenization (regex or whitespace+punct)                                | S      | **5–50×**              | 1 core saturated, GPU idle         |
| 3   | Per-pretoken LRU cache                                                      | XS     | 2–10× on real corpora  | 1 core saturated, GPU idle         |
| 4   | `worker_threads` parallel encode (post-pretok)                              | M      | 6–14× on 16 cores      | **All 16 cores at ~80%**, GPU idle |
| 5   | Merge-all-occurrences-per-pass (cheap fix in current loop)                  | XS     | 1.3–3×                 | 1 core saturated, GPU idle         |
| 6   | Drop `setTimeout(0)` yield in hot loop / move it behind a step counter      | XS     | 1.05–1.3×              | 1 core saturated                   |
| 7   | Skip `Array.from(buf.subarray)` round-trip                                  | XS     | 1.05–1.2×              | Less GC pressure                   |
| 8   | Replace nested `Map<L, Map<R, …>>` with typed-array open-addressing hashmap | M      | 1.5–3×                 | 1 core saturated                   |
| 9   | WGSL BlockBPE-style GPU encode                                              | XL     | 1.5–3× on top of (1–4) | **GPU active**, CPU idle           |

### Suggested staged plan

**Stage A — quick wins (a single afternoon).**
Items 5, 6, 7. Together: maybe 2× and zero behavioral risk. Ship as a
small refactor of `performSingleGreedyMergeUint32` and `encodeAsync`.
Use this stage to also wire **a real profiler** (`node --cpu-prof`) into
the CLI so subsequent stages have ground truth.

**Stage B — pre-tokenization + cache (1–3 days).**
Items 2 and 3. Add a pre-tokenizer (start with whitespace+punctuation;
upgrade to a tiktoken-style regex later) that returns `Uint8Array` slices.
Encode each slice independently and cache the result by `Bytes → number[]`.
This is also the prerequisite for stage C — and on its own it should
reduce the user-reported wall time by an order of magnitude.

**Stage C — algorithmic upgrade (1 week).**
Item 1. Re-implement the per-piece encode as priority-queue + linked
list with lazy invalidation. Reference implementations: Karpathy minBPE
(Python, very readable), tiktoken `src/lib.rs::_byte_pair_merge_large`,
and the rust-tokenizers walkthrough cited below. Keep `encode` /
`encodeAsync` API stable; this is purely an internal change. After this
stage we are _competitive single-threaded_ and the algorithm scales.

**Stage D — multi-core (3–5 days).**
Item 4. Pool of `worker_threads`, one per physical core. Each worker
loads the same tokenizer JSON once, then services encode requests over
a `SharedArrayBuffer`. Re-use the per-piece cache per worker (or share
it via a `SharedArrayBuffer`-backed open-addressing hashmap if we want
to be fancy). Now the dashboard should report ~80–90% per core.

**Stage E (optional, educational) — WGSL GPU encode.**
Item 9. Defer until A–D land. Worth doing as a _learning exercise_
because the project's stated goal is "own the stack", but unlikely to
beat A–D on actual training-corpus sizes. If we do it, mirror BlockBPE:
byte-level pretok, GPU-resident open-addressing hashmap, block-level
prefix scan for compaction. Document the quality trade-off (no regex
pretok ⇒ different encodings of digit runs etc.).

### Sequencing rationale

- A → B → C → D matches "fix constants → reduce work → reduce big-O →
  parallelize". Doing them in any other order wastes engineering time
  (e.g. parallelizing an `O(N²)` algorithm just makes 16 cores idle in
  parallel; SOTA single-thread is `O(N log N)`).
- C and D are independent of each other in principle, but landing C
  first means D gets a cleaner unit of work to fan out (per-piece
  encodes) and a smaller correctness surface (the per-piece kernel
  is the same as serial).

## 7. Quick wins inside today's `BPETokenizer`

If we want to ship something _this week_ without restructuring the
algorithm, the highest-leverage edits are:

```diff
// performSingleGreedyMergeUint32 — collapse all positions of best pair in one pass
- // find the FIRST occurrence, do one merge, return
+ // walk left-to-right; whenever buf[j..j+2] == (chosenL, chosenR)
+ // emit mergedId at write head, skip 2 reads; else copy buf[j] to write head
+ // single allocation-free pass; produces correct greedy-by-rank result
```

That alone changes the inner cost per "best-rank pass" from
`O(N) find + O(N) copy = 2N` to `O(N) collapse = 1N`, and turns ~M
outer iterations into roughly the number of _distinct_ lowest-rank
pairs encountered (often 10×–100× fewer). It is also the structure
that minBPE uses and is easy to validate against the existing tests.

Two more drop-in changes that need no new abstractions:

- Move `setTimeout(0)` yields out of the hot loop. Use `setImmediate` and
  only after `>= TIME_SLICE_ENCODE_ASYNC_MS` _or_ `mergeIteration % 8192 === 0`.
- Return `Uint32Array` (or accept an out-buffer) from `encode` / `encodeAsync`
  so callers don't pay the boxed-Number round-trip. Update `train.tsx` to
  consume that directly.

## 8. Open questions for the proposal

1. Are we okay introducing a regex-based pre-tokenizer? It changes encodings
   (and therefore _the model_) — so it must be set at train time and stored
   in the tokenizer JSON. We control both, so this is a one-time migration.
2. If we add `worker_threads`, do we ship a Node-specific module or do we
   keep the package runnable in browsers? `BPETokenizer` is currently pure
   ESM/CJS; workers would put us in Node-only territory unless we abstract.
3. How far do we want to push the GPU path? It is extremely on-brand for
   the project ("own the stack, GPU-first"), but objectively last-mile ROI
   on training-corpus encode. Maybe a follow-on, scoped as "BlockBPE-style
   educational kernel" rather than a perf necessity.

## 9. References

- Sennrich, Haddow, Birch (2016). _Neural Machine Translation of Rare Words
  with Subword Units._ — Original BPE for NLP.
  <https://aclanthology.org/P16-1162/>
- Karpathy. _minbpe._ — Minimal Python BPE; the cleanest reference for the
  "merge all occurrences in a single pass" inner loop.
  <https://github.com/karpathy/minbpe>
- OpenAI. _tiktoken — `src/lib.rs::_byte_pair_merge_large`._ — Heap-based
  encode in Rust; arena vocabulary; FxHash open-addressing maps.
  <https://github.com/openai/tiktoken/blob/main/src/lib.rs>
- tiktoken PR #331 _Uses Regex instead of fancy-regex — 6× speedup._ —
  Demonstrates that on a properly-optimized merge loop, regex pre-tok
  becomes the bottleneck.
  <https://github.com/openai/tiktoken/pull/331>
- tiktoken PR #442 _Optimize `_byte_pair_merge` to `O(M log N)` using
  heap-based candidate selection._
  <https://github.com/openai/tiktoken/pull/442>
- HuggingFace `tokenizers` — production Rust BPE with cache, regex pretok,
  parallel batch encode.
  <https://github.com/huggingface/tokenizers>
- Becquin, _Byte Pair Encoding and Data Structures._ — Walks through naive
  → pre-split → priority-queue+BST → priority-queue+linked-list with Rust
  benchmarks; shows ~1450× win for the linked-list variant on 1000 lines
  of Hamlet.
  <https://guillaume-be.github.io/2021-09-16/byte_pair_encoding>
- Google `sentencepiece` PR #1208 _Faster BPE learning method:
  `O(N) → O(log N)` per merge._ — Same heap-based idea applied to training.
  <https://github.com/google/sentencepiece/pull/1208>
- _A Formal Perspective on Byte-Pair Encoding_ (Findings of ACL 2023). —
  Submodular formulation; proves the `O(N log M)` bound and discusses
  approximation guarantees of greedy BPE.
  <https://aclanthology.org/2023.findings-acl.38v1.pdf>
- Amos You. _BlockBPE: Parallel BPE Tokenization_ (2025). — GPU BPE with
  block-level prefix scans and GPU-resident hashmaps; 2–2.5× over tiktoken
  at high batch / long sequence; quality trade-off documented.
  <https://arxiv.org/abs/2507.11941>
- NVIDIA. _Run State of the Art NLP Workloads at Scale with RAPIDS,
  HuggingFace, and Dask._ — RAPIDS GST tokenizer; 270× over CPU on the
  WordPiece subword tokenizer; BPE on roadmap.
  <https://developer.nvidia.com/blog/run-state-of-the-art-nlp-workloads-at-scale-with-rapids-huggingface-and-dask/>
- Node.js docs. _Worker threads._ — `worker_threads`,
  `SharedArrayBuffer`, `Atomics`; recommended pattern is a worker pool.
  <https://nodejs.org/api/worker_threads.html>
- daechoi/`riptoken` — Rust+rayon parallel re-implementation of tiktoken;
  up to 7× speedup in batch mode.
  <https://github.com/daechoi/riptoken>
