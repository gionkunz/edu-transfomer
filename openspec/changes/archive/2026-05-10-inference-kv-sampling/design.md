## Tensor primitives

- **`copyF32Elems`** issues **`copyBufferToBuffer`** for a contiguous **f32** range; used to append one row of **K** or **V** (**`d_k`** elements) into a preallocated **`[maxSeqLen, d_k]`** cache tensor.
- **`view2dRowPrefix`** reduces the logical **`M`** dimension for matmul so attention reads only the filled prefix; shares the parent buffer (same destroy contract as **`reshape`**).

## Model KV path

- **Prefill**: identical causal MHA to training forward for length **`T`**, but after each head’s **`matmul`** for **K** and **V**, results are copied into the layer/head cache for rows **`0 .. T-1`**; **`cache.seqLen = T`** afterward.
- **Decode**: single-row hidden state; for each head, project **Q/K/V**, write new **K/V** at row **`seqLen`**, then **Q @ K_prefixᵀ** (softmax, no extra mask — all cached positions are “past”), **@ V_prefix**; **`seqLen`** increments once after the full layer stack + LM head.

## Inference

- Sampling runs on **CPU** from a logits **`Float32Array`** (read back from WebGPU): greedy, temperature scaling, optional top-k mask then softmax + categorical draw.
- **`generateAutoregressive`** requires an empty cache, runs **prefill** on **`promptIds`**, then loops **decode** + sample until **`maxNewTokens`**, **EOS**, or **maxSeqLen**.
