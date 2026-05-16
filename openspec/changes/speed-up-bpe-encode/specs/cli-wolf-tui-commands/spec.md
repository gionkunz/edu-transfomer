## MODIFIED Requirements

### Requirement: Train dashboard BPE encode progress and ETA

During **`llm train`**, when the pipeline is in **`encode_corpus`** and the **wolf-tui** is showing corpus **BPE** progress, the dashboard SHALL display, in addition to the existing expand/merge line (**chars / bytes / merge iteration / token count**):

1. **Elapsed wall time** since encoding of the joined corpus began (covering **expand** and **merge**).
2. The **pre-tokenizer policy label** in use for the active tokenizer (e.g. **`word-and-punct-v1`**). The label SHALL be shown once at the top of the encode subphase.
3. In the **merge** sub-phase:
   - **Regions encoded** out of **regions total** (counts from the **`regionIndex`** / **`regionsTotal`** fields of **`BpeEncodeProgressEvent`**).
   - **Mean outer passes per region** (running average over completed regions).
   - When at least one region has completed and a positive wall-time-per-pass estimate is available: an ETA computed as **`regions_remaining × mean_passes_per_region ÷ passes_per_second`** with wording that communicates the **per-region cost model** and that actual completion may be sooner.
4. A **macro progress indication** for the merge sub-phase (e.g. a **ProgressBar** or equivalent) clipped to **`0–100%`** showing **`regions_encoded ÷ regions_total`**.

Host-side presentation SHALL remain wiring-only (**no tokenizer algorithm** in **`apps/cli`** beyond calling existing **`encodeUint32Async`** hooks).

#### Scenario: Expand phase shows elapsed

- **WHEN** **`encodeUint32Async`** reports **`expand`** progress for the corpus
- **THEN** the train **wolf-tui** SHALL display elapsed time for the encode phase and SHALL NOT imply a definitive merge ETA (merge has not begun or rate is unavailable)

#### Scenario: Merge phase shows per-region ETA

- **WHEN** **`encodeUint32Async`** reports **`merge`** progress and at least one region has fully completed
- **THEN** the train **wolf-tui** SHALL display **regions encoded / total**, **mean passes per region**, and a remainder ETA derived from the per-region cost model, with wording that communicates the per-region estimate and that **actual completion may be sooner**

#### Scenario: Active pre-tokenizer policy is displayed

- **WHEN** the train **wolf-tui** enters the BPE encode sub-phase for a tokenizer with policy **`word-and-punct-v1`**
- **THEN** the dashboard SHALL display the literal label **`word-and-punct-v1`** once for the duration of the encode subphase
