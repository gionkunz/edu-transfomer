# inference-kv-sampling

## Requirements

### R1 — Tensor copy and view

The tensor package SHALL export **`copyF32Elems`** to copy a contiguous **f32** element range between GPU buffers with bounds checks, and **`view2dRowPrefix`** to present the first **N** rows of a rank-2 contiguous **f32** tensor as a new logical shape sharing the same buffer.

### R2 — KV cache allocation

The model package SHALL export **`createGptKvCache`** allocating, per transformer layer and attention head, **K** and **V** tensors of shape **`[maxSeqLen, d_k]`**, and **`destroyGptKvCache`** releasing them.

### R3 — Prefill forward

The model package SHALL export **`forwardGptPrefill`** accepting **`GptKVCache`** with **`seqLen === 0`**, running a causal GPT forward over **`tokenIds`**, filling each layer’s head caches with the **K** and **V** projections for all prompt positions, then setting **`cache.seqLen`** to the prompt length. It SHALL return logits **`[T, vocabSize]`** consistent with **`forwardGpt`** on the same **`tokenIds`**.

### R4 — Decode step

The model package SHALL export **`forwardGptDecodeStep`** extending **`GptKVCache`** at the current **`seqLen`**, embedding **`tokenId`** at position **`seqLen`**, and returning logits **`[1, vocabSize]`** equivalent to **`forwardGpt`** on the full token history built from prior cache length plus this token (within numerical tolerance). It SHALL increment **`cache.seqLen`** by one after the step.

### R5 — Sampling

The inference package SHALL export **`greedyArgmax`**, **`sampleLogits`** supporting temperature and optional top-k (**`0`** = disabled), and document defaults aligned with project guidance.

### R6 — Autoregressive helper

The inference package SHALL export **`generateAutoregressive`** that performs prefill on a non-empty **`promptIds`**, then samples up to **`maxNewTokens`** using **`forwardGptDecodeStep`**, respecting optional **EOS** id and **`maxSeqLen`**.

### R7 — Tests

The repository SHALL include Vitests proving sampling behavior on CPU and, when WebGPU is available, that combined prefill + decode matches **`forwardGpt`** for small sequences.
