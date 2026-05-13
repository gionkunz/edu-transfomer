## Context

- **`llm train`** calls **`BPETokenizer.encodeAsync`** during **`encode_corpus`** with **`onProgress`** (`expand` then repeated **`merge`** emissions). See `apps/cli/src/cmd/train.tsx`.
- Each greedy merge step removes **exactly one** token from the working buffer, so the number of merges is **bounded above** by **`initialTokenCount - 1`** (worst case: every step merges until length 1—real runs often stop much earlier when no pair matches).
- ETA must be **heuristic**: show a **pessimistic** bound + **measured merge rate** so the label can say the run may complete sooner.

## Goals / Non-Goals

**Goals:**

- Show **elapsed** time for the whole encode phase (expand + merge).
- In **merge** phase: **merges/s** (smoothed EWMA), **pessimistic remaining merges** `max(0, upper - mergeIteration)`, **ETA sec** = remaining / rate when rate is reliable.
- Compact **ProgressBar** capped at 100% using **`mergeIteration / pessimisticUpperBound`** (clamp); user copy notes **“worst-case bar (may finish sooner)”** in the TUI line.

**Non-Goals:**

- Changing **`@llm-transformer-diy/tokenizer`** merge algorithm or progress cadence (unless a bug is found).
- Exact ETAs or machine-learning predictors.

## Decisions

1. **Worst-case merge bound** → `pessimisticMergeUpperBound = max(1, initialTokenCount - 1)`.
2. **Rate** → Exponential moving average of instantaneous **ΔmergeIteration / Δt** between progress samples (α ≈ **0.25** on new instantaneous rate); first reliable sample seeds the EMA.
3. **Expand phase** → No merge ETA; show elapsed + initial token count only.
4. **Display** → Format large ETAs as **`~NNNs`** or **`~NNm`** for readability; hide ETA if rate unknown or zero until enough samples.

## Risks / Trade-offs

- **[Misleading ETA]** Pessimistic denom can make the bar crawl while the run ends “early” → Mitigation: subtitle **“worst-case bound; may finish sooner”**.
- **[Jitter]** Noisy rate early → Mitigation: EWMA; only show ETA after **≥2** merge progress updates with positive **Δiter**.

## Open Questions

- None.
