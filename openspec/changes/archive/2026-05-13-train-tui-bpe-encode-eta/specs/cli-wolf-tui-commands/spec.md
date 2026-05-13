# cli-wolf-tui-commands

## ADDED Requirements

### Requirement: Train dashboard BPE encode progress and ETA

During **`llm train`**, when the pipeline is in **`encode_corpus`** and the **wolf-tui** is showing corpus **BPE** progress, the dashboard SHALL display, in addition to the existing expand/merge line (**chars / bytes / merge iteration / token count**):

1. **Elapsed wall time** since encoding of the joined corpus began (covering **expand** and **merge**).
2. In the **merge** sub-phase: a **pessimistic upper bound** on remaining merge steps derived from the **initial post-expand token count** **`N`** as **`max(0, max(1, N − 1) − mergeIteration)`** (documented in UI as a **worst-case** bound that may over-estimate).
3. In the **merge** sub-phase, when enough timing samples exist: **merge throughput** (merges per second, smoothed) and an estimated **remaining time** in seconds computed as **remaining bound ÷ merges per second**, labeled so the user understands the estimate depends on that pessimistic bound and **actual completion may be sooner**.
4. A **macro progress indication** for the merge sub-phase (e.g. a **ProgressBar** or equivalent) clipped to **`0–100%`** proportional to **`mergeIteration ÷ max(1, N − 1)`**, with the same **worst-case** caveat.

Host-side presentation SHALL remain wiring-only (**no tokenizer algorithm** in **`apps/cli`** beyond calling existing **`encodeAsync`** hooks).

#### Scenario: Expand phase shows elapsed

- **WHEN** **`encodeAsync`** reports **`expand`** progress for the corpus
- **THEN** the train **wolf-tui** SHALL display elapsed time for the encode phase and SHALL NOT imply a definitive merge ETA (merge has not begun or rate is unavailable)

#### Scenario: Merge phase shows bound-based ETA when rate known

- **WHEN** **`encodeAsync`** reports **`merge`** progress and the host has estimated a positive merges-per-second rate from prior merge samples
- **THEN** the train **wolf-tui** SHALL display a remainder ETA derived from the pessimistic merge bound divided by that rate, with wording that communicates **worst-case / may finish sooner**
