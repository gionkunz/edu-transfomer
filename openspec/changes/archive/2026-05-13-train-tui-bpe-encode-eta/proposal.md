## Why

Corpus **BPE encoding** before training can run a long time on large texts, but the train **wolf-tui** only showed merge counts with no **elapsed time** or **ETA**, so users could not tell whether the process was healthy or how long to wait.

## What Changes

- Extend the **`encode_corpus`** dashboard block with **elapsed wall time**, optional **merge throughput** (merges/s), a **pessimistic bounded progress** indication for the merge loop, and an **ETA** derived from that bound and measured rate (clearly labeled so users know it may finish sooner).
- Pure helper(s) optional for **worst-case merge bound** and ETA math to keep **`train.tsx`** readable and testable.

## Capabilities

### New Capabilities

- _(none)_

### Modified Capabilities

- **`cli-wolf-tui-commands`**: Extend the **train** wolf-tui encode phase with progress/ETA UX for **`BPE encodeAsync`** corpus encoding (no tokenizer algorithm changes).

## Impact

- **`apps/cli`**: `train-dash-types.ts`, `train.tsx`, `train-dashboard.tsx`, optional small helper + spec file under **`src/cmd/`**.
