# training-loop-bootstrap

## Requirements

### R1 — Cosine LR schedule

The training package SHALL export **`getCosineLr`** and **`CosineLrSchedule`** implementing linear warmup then cosine decay from **`lrMax`** toward **`lrMin`**, with the final training step (**`totalSteps - 1`**) reaching **`lrMin`** when the post-warmup phase has more than one step.

### R2 — LM batch helper

The training package SHALL export **`createLmBatch(rt, token, start, seqLen)`** producing **`input`** and **`targets`** index tensors for next-token prediction with length **`seqLen`**, validating bounds on **`start`** and **`seqLen`**.

### R3 — Trainer step

The training package SHALL export **`Trainer`** performing **`zeroGrad`**, LR from **`getCosineLr`**, **`forwardGpt`**, mean cross-entropy loss, **`backward`**, **`clipGradNorm`** only when **`maxGradNorm`** is finite and strictly positive, then **`AdamW.step`**, and returning the scalar mean loss.

### R4 — Parameter registration helper

The training package SHALL export **`registerModelParameters(registry, params)`** registering each **`Parameter`** from **`collectGptParameters`** (or any caller-supplied list) on the given registry.

### R5 — GPT checkpoint I/O

The training package SHALL export **`saveGptCheckpoint`** writing **`f32`** payloads for every GPT parameter in **`collectGptParameters`** order with keys from **`gptParameterTensorNames`**, using **`writePackedCheckpointFiles`**, and **`loadGptCheckpointTensors`** reading via **`readPackedCheckpointFiles`**.

### R6 — Stable checkpoint keys

The model package SHALL export **`gptParameterTensorNames(model)`** returning a string array of the same length and order as **`collectGptParameters(model)`**.

### R7 — Tests

The training package SHALL include Vitest coverage for **`getCosineLr`** (pure) and, when WebGPU is available, **`Trainer`** overfitting a fixed LM batch (loss decreases) and checkpoint save/load producing all expected tensor keys.
