# optimize-bpe-encode-hot-loop

Stage A quick wins for BPE encode: merge-all-occurrences per pass, relax event-loop yield, return Uint32Array — ~2x wall time, zero encoding change.

## Performance spot-check (implementation)

- **Environment:** WSL2, Node (repo `dist` build), 2026-05-13.
- **Fixture:** `corpus/real/pg35.txt` (~204 KiB UTF-8).
- **Train:** `BPETokenizer.train(text, 320)` ≈ **604 ms** (one run).
- **Encode:** `encodeUint32(text)` steady-state ≈ **177 ms** per call (after warm JIT).

A precise before/after A/B for the prior single-slot merge loop was not captured in automation here; the OpenSpec design targets roughly **~2×** less wall time for corpus-scale encode versus the old inner loop + `Array.from` + `Uint32Array.from` repack in **`train`**.
