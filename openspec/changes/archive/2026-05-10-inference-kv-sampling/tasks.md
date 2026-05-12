# Tasks — inference-kv-sampling

- [x] Tensor `copyF32Elems` + `view2dRowPrefix` + unit tests
- [x] Model `GptKVCache`, create/destroy, `forwardGptPrefill`, `forwardGptDecodeStep` + exports
- [x] Inference sampling (greedy, temperature, top-k) + `generateAutoregressive`
- [x] Vitest: CPU sampling; GPU KV parity vs `forwardGpt`
- [x] Package deps (`inference` → autograd, tensor; dev gpu for tests)
- [x] OpenSpec archive
