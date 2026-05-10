## LR schedule

- **`getCosineLr(step, schedule)`** is **0-based** on the optimizer step index passed into **`Trainer.trainStep`**.
- **Warmup**: **`lrMax * (step + 1) / warmupSteps`** for **`step < warmupSteps`** so the first step is not LR zero.
- **Cosine**: **`lrMin + 0.5 * (lrMax - lrMin) * (1 + cos(π * progress))`** with **progress** spanning post-warmup steps such that the **last** global step (**`totalSteps - 1`**) maps to **`progress === 1`** (use **`postLen - 1`** in the denominator when **`postLen > 1`**; **`postLen <= 1`** uses **`progress = 1`** after warmup).

## LM batching

- **`createLmBatch`** builds index tensors: input **`token[start:start+seqLen]`**, targets **`token[start+1:start+seqLen+1]`**; caller ensures **`start + seqLen < token.length`**.

## Trainer

- One step: **`zeroGrad`**, **`optimizer.setLr(getCosineLr(...))`**, **`forwardGpt`**, **`crossEntropyMeanFromLogits`**, loss readback, **`backward`**, **`clipGradNorm`** only when **`maxGradNorm`** is finite and **> 0**, **`AdamW.step`**.
- **`registerModelParameters`** appends **`collectGptParameters`** entries to a registry (**`registerParameter`**).

## Checkpoints

- **`saveGptCheckpoint`**: CPU readback per parameter via **`toArray`**, names from **`gptParameterTensorNames`**, **`writePackedCheckpointFiles`** (optional **`metadata.step`**).
- **`loadGptCheckpointTensors`**: thin **`readPackedCheckpointFiles`** for tests and future resume wiring.
