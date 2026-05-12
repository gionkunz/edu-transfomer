## Why

Sequential plan step **12** (`docs/sequential-change-proposals.md`) requires **`packages/inference`** to own generation: sampling strategies and a KV-cache decode path so autoregressive steps are **O(n)** in context length instead of re-running a full quadratic attention over the prefix every token.

## What Changes

- **`packages/tensor`**: **`copyF32Elems`** (GPU buffer span copy) and **`view2dRowPrefix`** (metadata view for valid cache prefix rows); Vitest coverage.
- **`packages/model`**: **`gpt-kv.ts`** — **`GptKVCache`**, **`createGptKvCache`** / **`destroyGptKvCache`**, **`forwardGptPrefill`**, **`forwardGptDecodeStep`** (per-head K/V slabs **`[maxSeqLen, d_k]`**).
- **`packages/inference`**: **`greedyArgmax`**, **`sampleLogits`** (temperature + top-k), **`generateAutoregressive`**; tests (CPU sampling + GPU KV vs full forward).

## Explicitly Out of Scope

- Checkpoint load + full **`llm generate`** CLI wiring, wolf-tui streaming UI, MQA/GQA, speculative decoding.

## Capabilities

### New

- **`inference-kv-sampling`**: KV prefill/decode API on the model, sampling helpers, high-level generate loop.

### Modified

- **`tensor-bootstrap`**: small buffer/view primitives consumed by KV cache writes.

## Impact

- Primary: `packages/inference`, `packages/model`, `packages/tensor`.
- Downstream: CLI **`generate`** can call **`generateAutoregressive`** without adding ML dependencies.
