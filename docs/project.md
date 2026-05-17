# LLM Transformer DIY — Project Documentation

## Goal

Build a transformer-based language model engine from scratch in **TypeScript** for the purpose of deeply understanding how LLMs work internally. The implementation prioritizes clarity, educational value, and genuine ownership of every layer — from GPU compute kernels to the autograd engine to the training loop.

**Constraints:**

- All tensor operations implemented natively — custom `Tensor` type backed by `Float32Array` (CPU) and `GPUBuffer` (GPU)
- All GPU compute written as hand-authored WGSL shaders dispatched via the WebGPU API
- Custom autograd engine (define-by-run computation graph with automatic backward pass)
- All model layers (attention, FFN, norms) written from scratch — no `nn.Module` equivalents from libraries
- All training infrastructure (AdamW, LR schedule, gradient clipping) implemented by hand
- Inference runtime with KV cache, multiple sampling strategies

**Allowed dependencies (and why):**

| Package               | Purpose                                  | Justification                                                               |
| --------------------- | ---------------------------------------- | --------------------------------------------------------------------------- |
| `webgpu` (npm)        | WebGPU device access in Node.js via Dawn | Hardware interface only — we write all shaders                              |
| `wolf-tui`            | Terminal UI rendering                    | UX layer — not ML logic                                                     |
| Nx + `@nx/js`         | Monorepo tooling                         | Build infrastructure only                                                   |
| `@nx/vitest`          | Nx Vitest test executor                  | Wires `project.json` `test` targets to Vitest only                          |
| `vite`                | Vitest config loading (Vite-backed)      | Required peer for `@nx/vitest` / Vitest 3 config — not application bundling |
| `vitest`              | Unit/integration tests                   | Test infrastructure only                                                    |
| `eslint` + `prettier` | Code quality                             | Style enforcement only                                                      |
| `commander` / `yargs` | CLI argument parsing                     | Ergonomics only                                                             |
| `glob`                | File discovery for training data         | Utility only                                                                |

**Explicitly forbidden:**

| Package                                          | Reason                     |
| ------------------------------------------------ | -------------------------- |
| `@tensorflow/tfjs`                               | Hides compute              |
| `onnxruntime`                                    | Hides inference            |
| `transformers` / HuggingFace                     | Hides everything           |
| `tiktoken` / `sentencepiece`                     | We write our own tokenizer |
| Any autograd library (`surfgrad`, `shade`, etc.) | We write our own           |

---

## Technology Stack

| Concern           | Choice                                                                                                                                                                     |
| ----------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Language          | TypeScript (strict mode; CommonJS emit for Node libs/app)                                                                                                                  |
| Runtime           | Node.js 24 LTS                                                                                                                                                             |
| GPU               | WebGPU via `webgpu` npm package (Dawn v0.4.0)                                                                                                                              |
| Shader language   | WGSL                                                                                                                                                                       |
| Monorepo          | Nx with `@nx/js`; **`packages/*`** use **`tsconfig` paths** for `@llm-transformer-diy/*`; **`apps/cli`** uses **`file:`** deps for npm install / prune (no npm workspaces) |
| TUI               | wolf-tui (framework-agnostic, Taffy layout engine)                                                                                                                         |
| Tests             | Vitest                                                                                                                                                                     |
| Linting           | ESLint + Prettier                                                                                                                                                          |
| Package manager   | npm                                                                                                                                                                        |
| GPU strategy      | GPU-only (hard requirement — fail loudly if no adapter)                                                                                                                    |
| Checkpoint format | Binary `.bin` (packed float tensors: **`f32`**, **`bf16`**) + `.json` sidecar (config + per-tensor dtype manifest)                                                         |

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                         CLI (apps/cli)                               │
│   wolf-tui based • subcommands: tokenize, train, generate, inspect  │
└────────────┬───────────────────┬────────────────────┬───────────────┘
             │                   │                    │
┌────────────▼──────┐ ┌─────────▼─────────┐ ┌───────▼───────────────┐
│   Tokenizer       │ │   Training        │ │   Inference           │
│   (packages/      │ │   (packages/      │ │   (packages/          │
│    tokenizer)     │ │    training)      │ │    inference)         │
└────────┬──────────┘ └────────┬──────────┘ └───────┬───────────────┘
         │                     │                    │
┌────────▼─────────────────────▼────────────────────▼───────────────┐
│                     Model (packages/model)                         │
│   GPT assembly • TransformerBlock • Attention • FFN • Norms       │
└────────────────────────────────┬──────────────────────────────────┘
                                 │
┌────────────────────────────────▼──────────────────────────────────┐
│                   Autograd (packages/autograd)                     │
│   Computation graph • backward pass • parameter registry          │
└────────────────────────────────┬──────────────────────────────────┘
                                 │
┌────────────────────────────────▼──────────────────────────────────┐
│                    Tensor (packages/tensor)                        │
│   Tensor type • shape algebra • GPU↔CPU transfer • op dispatch    │
└────────────────────────────────┬──────────────────────────────────┘
                                 │
┌────────────────────────────────▼──────────────────────────────────┐
│                      GPU (packages/gpu)                            │
│   WebGPU device init • buffer pool • kernel registry • WGSL code  │
└───────────────────────────────────────────────────────────────────┘
```

The stack is layered bottom-up: `gpu` → `tensor` → `autograd` → `model` → `training`/`inference` → `cli`. Each layer only depends on those below it.

---

## Monorepo Structure

```
llm-transformer-diy/
├── apps/
│   └── cli/                        TUI-based CLI binary (`llm`)
│       ├── src/
│       │   ├── main.ts             Entry point, command routing
│       │   ├── commands/
│       │   │   ├── tokenize.ts     `llm tokenize train|encode|decode`
│       │   │   ├── train.ts        `llm train --data ./corpus/**/*.txt`
│       │   │   ├── generate.ts     `llm generate --model ./out/model`
│       │   │   └── inspect.ts      `llm inspect --model ./out/model`
│       │   └── views/
│       │       ├── train-dashboard.ts   Live TUI: loss curve, throughput, ETA
│       │       └── generate-view.ts     Streaming token output
│       ├── package.json
│       └── tsconfig.json
│
├── packages/
│   ├── gpu/                        WebGPU abstraction + WGSL kernels
│   │   ├── src/
│   │   │   ├── device.ts           Adapter discovery, device init, queue
│   │   │   ├── buffer.ts           GPUBuffer creation, read/write, pool
│   │   │   ├── pipeline.ts         Compute pipeline + bind group factory
│   │   │   ├── kernel-registry.ts  Registry of compiled kernels
│   │   │   └── kernels/
│   │   │       ├── matmul.wgsl
│   │   │       ├── matmul-backward.wgsl
│   │   │       ├── elementwise.wgsl    (add, mul, scale, gelu, exp, log)
│   │   │       ├── softmax.wgsl        (numerically stable 2-pass)
│   │   │       ├── rmsnorm.wgsl
│   │   │       ├── embedding.wgsl      (gather + scatter for backward)
│   │   │       ├── cross-entropy.wgsl
│   │   │       ├── adamw.wgsl          (fused optimizer step)
│   │   │       └── reduce.wgsl         (sum, max, mean)
│   │   └── package.json
│   │
│   ├── tensor/                     Tensor primitive
│   │   ├── src/
│   │   │   ├── tensor.ts           Tensor class (shape, strides, data, GPUBuffer)
│   │   │   ├── shape.ts            Shape inference, broadcasting rules
│   │   │   ├── ops.ts              High-level op dispatch (matmul, add, etc.)
│   │   │   ├── random.ts           Xavier/Kaiming init, uniform, normal
│   │   │   └── types.ts            DType, Device, TensorLike
│   │   └── package.json
│   │
│   ├── autograd/                   Automatic differentiation
│   │   ├── src/
│   │   │   ├── graph.ts            Computation graph (DAG of operations)
│   │   │   ├── backward.ts         Topological sort + backward traversal
│   │   │   ├── parameter.ts        Parameter wrapper (tensor + grad + requires_grad)
│   │   │   ├── operations/
│   │   │   │   ├── matmul.ts       Forward + backward for matmul
│   │   │   │   ├── add.ts          Forward + backward for add
│   │   │   │   ├── gelu.ts         Forward + backward for GELU
│   │   │   │   ├── softmax.ts      Forward + backward for softmax
│   │   │   │   ├── rmsnorm.ts      Forward + backward for RMSNorm
│   │   │   │   ├── embedding.ts    Forward + backward for embedding lookup
│   │   │   │   ├── cross-entropy.ts Forward + backward for loss
│   │   │   │   └── reshape.ts      Forward + backward for view/reshape
│   │   │   └── index.ts
│   │   └── package.json
│   │
│   ├── tokenizer/                  Char-level + BPE tokenizer
│   │   ├── src/
│   │   │   ├── interface.ts        Tokenizer interface (train, encode, decode, save, load)
│   │   │   ├── char.ts             CharTokenizer implementation
│   │   │   ├── bpe.ts              BPETokenizer implementation
│   │   │   └── special-tokens.ts   <|endoftext|>, <|pad|>
│   │   └── package.json
│   │
│   ├── model/                      Transformer architecture
│   │   ├── src/
│   │   │   ├── config.ts           ModelConfig type (all hyperparameters)
│   │   │   ├── embedding.ts        Token + positional embeddings
│   │   │   ├── attention.ts        CausalMultiHeadSelfAttention
│   │   │   ├── feedforward.ts      FFN (GELU, expand/project)
│   │   │   ├── norm.ts             RMSNorm
│   │   │   ├── block.ts            TransformerBlock (attn + ffn + residual)
│   │   │   ├── gpt.ts             GPT class (full forward pass)
│   │   │   └── init.ts             Weight initialization (Xavier)
│   │   └── package.json
│   │
│   ├── training/                   Training infrastructure
│   │   ├── src/
│   │   │   ├── dataset.ts          Load .txt files, tokenize, chunk, batch
│   │   │   ├── adamw.ts            AdamW optimizer (from scratch, GPU-accelerated)
│   │   │   ├── scheduler.ts        Cosine LR with linear warmup
│   │   │   ├── gradient-clip.ts    Global norm clipping
│   │   │   ├── trainer.ts          Training loop orchestration
│   │   │   └── checkpoint.ts       Save/load model + optimizer state
│   │   └── package.json
│   │
│   ├── inference/                  Inference runtime
│   │   ├── src/
│   │   │   ├── runtime.ts          Load model, manage generation loop
│   │   │   ├── sampler.ts          Greedy, temperature, top-k sampling
│   │   │   ├── kv-cache.ts         KV cache management per layer
│   │   │   └── generate.ts         Autoregressive token generation
│   │   └── package.json
│   │
│   ├── io/                         Serialization + file I/O
│   │   ├── src/
│   │   │   ├── checkpoint-format.ts Binary format spec + reader/writer
│   │   │   ├── config-loader.ts     Load model/training configs from JSON
│   │   │   └── glob-loader.ts       Discover + read training text files
│   │   └── package.json
│   │
│   └── utils/                      Shared utilities
│       ├── src/
│       │   ├── logger.ts            Structured logging
│       │   ├── progress.ts          Progress reporting for TUI
│       │   └── timer.ts             Performance timing
│       └── package.json
│
├── docs/
│   └── project.md                  ← this file
├── nx.json
├── tsconfig.base.json
├── package-lock.json
├── package.json
├── .eslintrc.json
├── .prettierrc
├── vitest.config.ts                Root Vitest `test.projects` (optional all-in-one run)
├── vitest.config.base.ts           Shared defaults for each package/app `vitest.config.ts`
└── eslint.config.js
```

---

## Component 1 — GPU Layer (`packages/gpu`)

### WebGPU Primer

WebGPU is a modern GPU compute API (W3C standard). In Node.js, the `webgpu` npm package provides access via Dawn (Google's C++ WebGPU implementation). Key concepts:

| Concept              | What it is                                                 |
| -------------------- | ---------------------------------------------------------- |
| **Adapter**          | A handle to a physical GPU                                 |
| **Device**           | Logical connection to the GPU — all work goes through this |
| **Queue**            | Where you submit command buffers for execution             |
| **Buffer**           | A chunk of GPU memory (`GPUBuffer`)                        |
| **Bind Group**       | Maps buffers to shader variables (inputs/outputs)          |
| **Compute Pipeline** | A compiled WGSL shader ready to dispatch                   |
| **Dispatch**         | Launch N workgroups of M threads each                      |

### Initialization flow

```typescript
const adapter = await navigator.gpu.requestAdapter();
if (!adapter) throw new Error('No WebGPU adapter — GPU required');
const device = await adapter.requestDevice();
```

### WGSL Kernel Example: Matrix Multiplication

```wgsl
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read> b: array<f32>;
@group(0) @binding(2) var<storage, read_write> result: array<f32>;

struct Dimensions {
  M: u32, N: u32, K: u32,
}
@group(0) @binding(3) var<uniform> dims: Dimensions;

@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let row = gid.x;
  let col = gid.y;
  if (row >= dims.M || col >= dims.N) { return; }

  var sum: f32 = 0.0;
  for (var k: u32 = 0u; k < dims.K; k = k + 1u) {
    sum = sum + a[row * dims.K + k] * b[k * dims.N + col];
  }
  result[row * dims.N + col] = sum;
}
```

This is a naive matmul — we'll optimize with tiled shared-memory access in later phases.

### Required WGSL Kernels (complete inventory)

| Kernel          | Forward                                    | Backward                     | Notes                          |
| --------------- | ------------------------------------------ | ---------------------------- | ------------------------------ |
| `matmul`        | C = A @ B                                  | dA = dC @ B^T, dB = A^T @ dC | Core operation, tiled for perf |
| `add`           | C = A + B                                  | dA = dC, dB = dC (broadcast) | Element-wise with broadcasting |
| `mul`           | C = A \* B                                 | dA = dC _ B, dB = dC _ A     | Element-wise (Hadamard)        |
| `scale`         | Y = α·X                                    | dX = α·dY                    | Scalar multiply                |
| `gelu`          | y = x·Φ(x)                                 | dy·gelu'(x)                  | Approximation formula          |
| `softmax`       | Stable 2-pass: max → exp → sum → normalize | Jacobian-vector product      | Numerically stable             |
| `rmsnorm`       | y = x / rms(x) · γ                         | Chain rule through norm      | Fused kernel                   |
| `embedding`     | Gather rows by index                       | Scatter-add gradients        | Index-based lookup             |
| `cross_entropy` | -log(softmax(logits)[target])              | Combined softmax+CE grad     | Fused for stability            |
| `adamw`         | θ -= lr·m̂/(√v̂+ε) + λ·θ                     | N/A (optimizer step)         | Fused update                   |
| `reduce_sum`    | Sum along axis                             | Expand gradient              | For loss computation           |
| `reduce_max`    | Max along axis                             | Mask gradient                | For softmax stability          |
| `clip_norm`     | Scale if norm > threshold                  | N/A                          | Gradient clipping              |
| `causal_mask`   | Fill upper-tri with -inf                   | N/A                          | Attention masking              |
| `transpose`     | Swap axes                                  | Same transpose               | For matmul backward            |

---

## Component 2 — Tensor Primitive (`packages/tensor`)

### Design

The tensor layer is **functional**: plain immutable data records plus free functions. There are no classes, no methods on `Tensor`, and no module-level mutable state. Every function that allocates or dispatches takes a `TensorRuntime` as its first argument; pure metadata transforms (e.g. `reshape`) take none.

```typescript
interface Tensor {
  readonly shape: readonly number[];
  readonly strides: readonly number[]; // row-major, contiguous in the current scope
  readonly dtype: 'f32';
  readonly device: 'gpu';
  readonly buffer: GPUBuffer;
  readonly length: number; // total elements
}

interface TensorRuntime {
  readonly ctx: GpuContext;
  readonly registry: KernelRegistry;
}
```

All tensors live on GPU. We only read back to CPU when:

- Logging a scalar loss value
- Saving a checkpoint
- Printing generated tokens

### Key operations

Op functions are free, return new `Tensor` records, and never mutate inputs. They route to kernels in `packages/gpu`'s registry under stable IDs (`core.matmul`, `core.add`, …):

- `matmul(rt, a, b)` → dispatches `core.matmul` (rank-2 only in the current scope)
- `add(rt, a, b)` → dispatches `core.add` (equal shape; no broadcasting yet)
- `mul(rt, a, b)` → dispatches `core.mul` (equal shape; no broadcasting yet)
- `scale(rt, a, k)` → dispatches `core.scale`
- `gelu(rt, a)` → dispatches `core.gelu`
- `transpose(rt, a)` → dispatches `core.transpose` (rank-2 only in the current scope)
- `softmax(rt, a, dim)` → dispatches `softmax.wgsl` _(planned: kernel and op land in a later change)_
- `reshape(a, newShape)` → pure metadata transform (no `rt`, shares the same `GPUBuffer`; rejects non-contiguous inputs)

Composition is plain function nesting: `gelu(rt, add(rt, x, b))`.

### Initialization

```typescript
createZeros(rt, shape); // allocate GPU buffer and zero-fill
createFromArray(rt, data, shape); // allocate GPU buffer and upload Float32Array
toArray(rt, t); // async readback: GPUBuffer → Float32Array
destroy(t); // release the GPU buffer (no internal alive-flag)
```

`createOnes`, `createRandn` (Box-Muller on GPU or seed from CPU), and `createXavier` (Xavier/Glorot) are planned for a later **tensor-init** change.

---

## Component 3 — Autograd Engine (`packages/autograd`)

### Design: Define-by-Run (PyTorch-style)

Every operation builds a computation graph as it executes. Each node records:

- The operation type
- Input tensors (parents)
- A closure or method to compute the backward pass (local Jacobian × incoming gradient)

```typescript
class Variable {
  readonly tensor: Tensor;
  readonly requiresGrad: boolean;
  grad: Tensor | null;
  readonly creator: Operation | null; // null for leaf parameters
  readonly parents: Variable[];
}

interface Operation {
  forward(...inputs: Variable[]): Variable;
  backward(gradOutput: Tensor): Tensor[]; // one grad per input
}
```

### Backward pass

```
1. Call loss.backward()
2. Topologically sort the graph from loss → leaves
3. Walk in reverse order:
   - For each node, call node.creator.backward(node.grad)
   - Accumulate returned gradients onto each parent's .grad
4. After traversal, all parameters have populated .grad fields
```

### Parameter management

A `ParameterRegistry` tracks all trainable `Variable` instances. It provides:

- `parameters()` → iterator of all Variables with `requiresGrad = true`
- `zeroGrad()` → set all `.grad = null`
- `clipGradNorm(maxNorm)` → compute global L2 norm, scale if exceeds threshold
- `totalParams()` → count for display

---

## Component 4 — Tokenizer (`packages/tokenizer`)

### Interface

```typescript
interface Tokenizer {
  train(text: string, vocabSize: number): void;
  encode(text: string): number[];
  decode(ids: number[]): string;
  save(path: string): Promise<void>;
  load(path: string): Promise<void>;
  vocabSize: number;
}
```

### Char-Level Tokenizer

Maps each unique character to an integer ID. Dead simple — useful for initial development and testing before BPE is ready.

- Vocabulary = set of all unique characters in training text + special tokens
- `encode`: split string into characters, map each to its ID
- `decode`: map IDs back to characters, join

### BPE Tokenizer

Full byte-pair encoding as described in the research section:

1. Start with 256 base tokens (one per byte value)
2. Add special tokens (`<|endoftext|>`, `<|pad|>`)
3. Iteratively merge the most frequent adjacent pair (incremental pair-frequency index, no full-corpus rescan per merge)
4. Save ordered merge list + vocabulary

**Pre-tokenizer policy.** Both `train` and `encode` operate on **pre-tokenized regions** under a named regex policy. The only currently supported policy is `gpt2-v1` — GPT-2 / Llama-style splitting where a content region MAY absorb a single leading ASCII space (` the`, ` and`, ` 42`, ` →` are single regions), English contractions (`'s 't 'm 'd 're 've 'll`) form their own regions, and runs of trailing whitespace at end-of-line stay isolated via the `\s+(?!\S)` branch. The exact regex is `/'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+/gu`. Merges never cross region boundaries. This bounds per-region pass count at `O(log region_length)` so encode time scales with corpus size linearly, not with vocab size × corpus size, and roughly halves the token count for English prose vs. an isolated-whitespace scheme.

- **Hot-path lookup.** The encode pair-rank table is a single-level `Map<number, number>` keyed by `L * vocabSize + R` so probes are one hash lookup, not two. Safe for any `vocabSize ≤ 94 906 265`.
- **Future policies.** New policies can be added as new labels without breaking saved files; `load` rejects any unknown label.

**Saved format.** JSON file, **schema `version: 3`**:

```jsonc
{
  "kind": "llm-transformer-diy/bpe-tokenizer",
  "version": 3,
  "pretokenizer": "gpt2-v1",
  "specialTokens": ["<|pad|>", "<|endoftext|>"],
  "byteOffset": 2,
  "merges": [[L, R], ...],
  "vocab": { "id": [bytes...], ... }
}
```

`load` requires `version: 3` and a known `pretokenizer` label (currently only `'gpt2-v1'`). Older v1 / v2 files (including the obsolete `'word-and-punct-v1'` pre-tokenizer) are rejected with a clear "retrain" message.

---

## Component 5 — Transformer Model (`packages/model`)

### Architecture: Decoder-only (GPT-style)

```
Input token IDs: [batch, seq_len]
       │
  Token Embedding ─────────────────── [vocab_size, d_model]
       +
  Positional Embedding ────────────── [max_seq_len, d_model]
       │
       ▼
  ┌─────────────────────────────────────────────┐
  │         Transformer Block × n_layers         │
  │                                             │
  │   x' = x + Attention(RMSNorm(x))           │
  │   x'' = x' + FFN(RMSNorm(x'))              │
  └─────────────────────────────────────────────┘
       │
  Final RMSNorm
       │
  Linear Head ─────────────────────── [d_model, vocab_size]
       │
  Logits: [batch, seq_len, vocab_size]
```

### Default Model Configs

| Config | `d_model` | `n_layers` | `n_heads` | `d_ff` | `max_seq_len` | `vocab_size` | ~Params |
| ------ | --------- | ---------- | --------- | ------ | ------------- | ------------ | ------- |
| tiny   | 128       | 2          | 2         | 512    | 128           | 512 (char)   | ~0.5M   |
| small  | 256       | 4          | 4         | 1024   | 256           | 4096         | ~5M     |
| medium | 512       | 8          | 8         | 2048   | 512           | 8192         | ~30M    |
| large  | 768       | 12         | 12        | 3072   | 1024          | 16384        | ~85M    |

Start with `tiny` for development (trains in seconds), graduate to `small`/`medium` for real text generation quality.

### Sub-components

**RMSNorm:**

```
RMSNorm(x) = x / sqrt(mean(x²) + ε) · γ
```

- `γ`: learned scale, shape `[d_model]`
- `ε = 1e-6`
- Applied **before** each sub-layer (Pre-Norm convention)

**Causal Multi-Head Self-Attention:**

```
Q = x @ W_Q    K = x @ W_K    V = x @ W_V
  each: [batch, seq_len, d_model] → reshape to [batch, n_heads, seq_len, d_k]
  where d_k = d_model / n_heads

scores = (Q @ K^T) / sqrt(d_k)
scores += causal_mask   (upper-triangular filled with -inf)
weights = softmax(scores, dim=-1)
attn_out = weights @ V
  reshape back to [batch, seq_len, d_model]
output = attn_out @ W_O
```

**Feed-Forward Network (FFN):**

```
FFN(x) = GELU(x @ W1 + b1) @ W2 + b2
  W1: [d_model, d_ff]
  W2: [d_ff, d_model]
```

**Weight Tying:** Output head shares weights with token embedding (`W_head = W_embed^T`). Requires small embedding init std (~0.02).

---

## Component 6 — Training (`packages/training`)

### Data Pipeline

```
1. Discover files via glob: `llm train --data ./corpus/**/*.txt`
2. Read all files, concatenate with <|endoftext|> separator
3. Encode full corpus → single array of token IDs
4. Split 90/10 → train/val
5. Chunk into windows of (max_seq_len + 1) tokens
6. Each window → input = window[0..n-1], target = window[1..n]
7. Shuffle chunks, form batches of batch_size
8. Upload batches to GPU as needed
```

### AdamW Optimizer (GPU-accelerated)

Implemented as a WGSL kernel (`adamw.wgsl`) that fuses the entire update step per-parameter:

```
For each parameter θ with gradient g:
  m = β₁·m + (1-β₁)·g
  v = β₂·v + (1-β₂)·g²
  m̂ = m / (1 - β₁^t)
  v̂ = v / (1 - β₂^t)
  θ = θ - lr · (m̂ / (√v̂ + ε)) - lr·λ·θ
```

Defaults: `β₁=0.9`, `β₂=0.95`, `ε=1e-8`, `weight_decay=0.1`

Weight decay is NOT applied to: bias vectors, embedding tables, norm scale parameters.

### Cosine LR Schedule with Warmup

```
if step < warmup_steps:
    lr = lr_max × (step / warmup_steps)
else:
    progress = (step - warmup_steps) / (total_steps - warmup_steps)
    lr = lr_min + 0.5 × (lr_max - lr_min) × (1 + cos(π × progress))
```

### Gradient Clipping (Global Norm)

```
1. Compute sum of squared gradients across all parameters (GPU reduce)
2. global_norm = sqrt(total)
3. If global_norm > max_norm (default 1.0):
     scale = max_norm / global_norm
     multiply all gradients by scale (GPU kernel)
```

### Training Loop

```typescript
for (let step = 0; step < totalSteps; step++) {
  const lr = scheduler.getLR(step);
  optimizer.setLR(lr);
  registry.zeroGrad();

  const { input, target } = dataset.nextBatch();
  const logits = model.forward(input);
  const loss = crossEntropy(logits, target);

  loss.backward();
  registry.clipGradNorm(1.0);
  optimizer.step();

  if (step % evalInterval === 0) {
    const valLoss = evaluate(model, valData);
    tui.updateMetrics({ step, trainLoss: loss, valLoss, lr });
  }
  if (step % saveInterval === 0) {
    checkpoint.save(model, optimizer, step, outputDir);
  }
}
```

---

## Component 7 — Inference Runtime (`packages/inference`)

### Loading

Read checkpoint (`.bin` + `.json` sidecar), reconstruct model architecture from config, load weights into GPU buffers. No autograd needed — forward pass only.

### Autoregressive Generation

```typescript
let context = tokenizer.encode(prompt);

for (let i = 0; i < maxNewTokens; i++) {
  const input = context.slice(-maxSeqLen);
  const logits = model.forward(input); // [seq_len, vocab_size]
  const nextLogits = logits.slice(-1); // last position only
  const nextToken = sampler.sample(nextLogits);
  context.push(nextToken);
  onToken(tokenizer.decode([nextToken])); // stream to TUI
  if (nextToken === eosTokenId) break;
}
```

### Sampling Strategies

| Strategy    | Behavior                                                     |
| ----------- | ------------------------------------------------------------ |
| Greedy      | `argmax(logits)` — deterministic, repetitive                 |
| Temperature | `softmax(logits / τ)` then sample — τ<1 sharper, τ>1 flatter |
| Top-k       | Zero out all but top-k logits, then temperature sample       |

Default: `temperature=0.8`, `topK=40`.

### KV Cache

Without cache: O(n²) per generated token (full context reprocessed).
With cache: store K, V tensors from each layer, only compute new token's Q.

```typescript
interface KVCache {
  keys: Tensor[]; // one per layer, shape grows: [batch, n_heads, seq_so_far, d_k]
  values: Tensor[]; // same
}
```

On each step:

1. Compute Q, K, V for the new token only
2. Append new K, V to cache
3. Attend: new Q against full cached K → weights → apply to full cached V
4. Return single-token output

---

## Component 8 — CLI (`apps/cli`)

### Binary: `llm`

Single entry point with subcommands. Short commands use plain stdout; long-running commands (`train`, `generate`) launch a wolf-tui dashboard.

### Commands

| Command                                                                    | Description                                   | TUI?                   |
| -------------------------------------------------------------------------- | --------------------------------------------- | ---------------------- |
| `llm tokenizer train --data ./text.txt --vocab-size 4096 --out ./tok.json` | Train a tokenizer on text                     | No (progress bar only) |
| `llm tokenizer encode --tokenizer ./tok.json "Hello world"`                | Encode text to IDs                            | No                     |
| `llm tokenizer decode --tokenizer ./tok.json 42,17,8`                      | Decode IDs to text                            | No                     |
| `llm train --data ./corpus/**/*.txt --config small --out ./model/`         | Train a model                                 | Yes — live dashboard   |
| `llm generate --model ./model/ --prompt "Once upon"`                       | Generate text                                 | Yes — streaming output |
| `llm inspect --model ./model/`                                             | Print model config, param count, layer shapes | No                     |

### Train Dashboard (wolf-tui)

```
┌─ Training ──────────────────────────────────────────────────┐
│  Model: small (5.2M params)  Device: NVIDIA RTX 4090        │
│  Step: 1,247 / 10,000       Tokens/sec: 42,380              │
│                                                              │
│  Loss ─────────────────────────────────────────              │
│  8.0 ┤╲                                                     │
│  6.0 ┤ ╲                                                    │
│  4.0 ┤  ╲___                                                │
│  3.0 ┤      ╲___________                                    │
│  2.5 ┤                   ────────── train                    │
│      └──────────────────────────── val ─ ─ ─                │
│                                                              │
│  LR: 2.4e-4 (cosine)  Grad norm: 0.87  ETA: 14m 32s        │
└──────────────────────────────────────────────────────────────┘
```

### Generate View (wolf-tui)

Tokens stream character-by-character with a subtle cursor. Metadata (tokens/sec, total tokens) shown in a status bar.

---

## Component 9 — Checkpoint Format (`packages/io`)

### Structure on disk

```
model-output/
├── model.bin           Packed tensors (concatenated; dtype per manifest entry; see below)
├── model.json          Config + tensor manifest (name → offset + shape + dtype)
├── optimizer.bin       Adam state (m, v per parameter)
├── optimizer.json      Optimizer config + state manifest
└── tokenizer.json      Vocabulary + merge rules
```

### `model.json` schema

```json
{
  "config": {
    "d_model": 256,
    "n_layers": 4,
    "n_heads": 4,
    "d_ff": 1024,
    "max_seq_len": 256,
    "vocab_size": 4096
  },
  "tensors": {
    "token_embedding.weight": { "offset": 0, "shape": [4096, 256], "dtype": "bf16" },
    "pos_embedding.weight": { "offset": 4194304, "shape": [256, 256], "dtype": "f32" },
    "blocks.0.norm1.gamma": { "offset": 4456448, "shape": [256], "dtype": "f32" }
  },
  "metadata": {
    "step": 5000,
    "train_loss": 2.81,
    "created_at": "2026-05-09T19:00:00Z"
  }
}
```

### Binary layout

Tensors are concatenated in **lexical order** of their manifest keys. Each tensor occupies **`Π(shape) × bytesPerElement(dtype)`** bytes (`2` for **`bf16`**, `4` for **`f32`**). **Padding** with zero bytes follows each tensor so the next tensor starts on a **4-byte boundary** (so pure **`f32`** checkpoints remain byte-for‑byte compatible with the pre–16-bit layout, where every payload length is already a multiple of 4). **Legacy manifests** may omit **`dtype`** on each tensor entry; readers treat that as **`f32`**.

---

## Numerical Stability

### 16-bit dtypes (compute & checkpoints)

- **Master weights (training):** Optimizer state (**AdamW** moments) stays in **`f32`**. When parameters use **`bf16`** compute buffers, an optional **`f32` master** copy receives the update, then values are cast back into the compute tensor so small steps are not fully lost to 16-bit rounding.
- **Accumulators in kernels:** Reductions and matmul inner products accumulate in **`f32`** where the WGSL uses an explicit wider accumulator; results are rounded to the storage dtype at the end of the kernel.
- **Why `bf16` is the only 16-bit dtype:** **`bf16`** keeps the same exponent range as **`f32`**, so training is stable without loss scaling. **`fp16`** was dropped because we never implemented loss scaling, its memory and bandwidth wins are identical to **`bf16`**'s, and the extra mantissa precision is rarely worth the instability for this educational stack. The single packed storage layout (two **`bf16`** scalars per `u32` word) is described in `packages/gpu/src/kernels/shared-bf16.wgsl`.
- **Capability probe:** **`supportsDtype`** on **`GpuContext`** documents whether a dtype may be used; **`bf16`** is always available via packed storage (no extra WebGPU features required). CLI commands call **`assertRuntimeSupportsComputeDtype`** after creating the tensor runtime.

**Checkpoint load policy:** Manifest **`dtype`** must be **`f32`** or **`bf16`** (or omitted → **`f32`**). **`f32`** checkpoints into a **`bf16`** model narrow weights via **`createFromArray`**. A **`bf16` checkpoint → `f32` model** requires **`{ allowDtypeUpcast: true }`** in code or **`--allow-dtype-upcast`** on **`llm generate` / `chat`**.

**CLI dtype flags:**

| Command                            | Flag                       | Meaning                                                          |
| ---------------------------------- | -------------------------- | ---------------------------------------------------------------- |
| **`llm train`**                    | **`--dtype f32\|bf16`**    | Parameter / forward compute dtype (default **`f32`**).           |
| **`llm generate`**, **`llm chat`** | **`--dtype …`**            | Runtime model dtype; must fit the checkpoint / cast rules above. |
| **`llm generate`**, **`llm chat`** | **`--allow-dtype-upcast`** | Allow a **`bf16`** checkpoint to load into an **`f32`** model.   |

### Common pathology cheatsheet

| Issue                              | Symptom                     | Fix                                                                             |
| ---------------------------------- | --------------------------- | ------------------------------------------------------------------------------- |
| Softmax overflow                   | NaN logits                  | Subtract `max(scores)` before `exp()` — implemented in `softmax.wgsl` as 2-pass |
| Exploding gradients                | NaN loss after N steps      | Gradient clipping (global norm ≤ 1.0)                                           |
| Poor weight init                   | Loss stuck or diverges      | Xavier init for linear; embedding std ≈ 0.02                                    |
| Weight tying + large embedding std | Logit overflow              | Keep std at 0.02 when tying                                                     |
| Large LR at step 0                 | Immediate divergence        | Linear warmup for first 100 steps                                               |
| FP32 precision on GPU              | Rounding in long reductions | Use workgroup-level Kahan summation if needed                                   |
| Cross-entropy with raw softmax     | Catastrophic cancellation   | Fuse log-softmax + NLL into single kernel                                       |

---

## Implementation Phases

### Phase 1 — Foundation (GPU + Tensor)

- Set up Nx monorepo, all packages scaffolded
- Implement `packages/gpu`: device init, buffer management, pipeline factory
- Write and test `matmul.wgsl` kernel (naive version)
- Implement `packages/tensor`: Tensor class, `matmul`, `add`, `zeros`, `randn`
- Verify: multiply two matrices on GPU, read result back, compare to CPU reference

### Phase 2 — Autograd

- Implement computation graph + backward traversal
- Implement `Variable`, `Parameter`, `Operation` interfaces
- Add backward implementations for: matmul, add, mul, gelu, softmax, rmsnorm, embedding, cross_entropy, reshape
- Write corresponding backward WGSL kernels
- Verify: compute gradient of simple function, compare to finite-difference approximation

### Phase 3 — Tokenizer

- Implement `CharTokenizer` (trivial, for testing)
- Implement `BPETokenizer` (train, encode, decode, save, load)
- Verify: encode/decode round-trip on sample text
- Wire into CLI: `llm tokenizer train/encode/decode`

### Phase 4 — Model

- Implement `RMSNorm`, `CausalMultiHeadAttention`, `FFN`, `TransformerBlock`, `GPT`
- Each layer is an Operation in the autograd graph
- Verify: forward pass produces correct output shapes
- Verify: causal mask blocks future tokens (attention pattern test)
- Verify: full backward pass through the model (loss → all gradients populated)

### Phase 5 — Training

- Implement `AdamW` optimizer (fused GPU kernel)
- Implement cosine LR schedule with warmup
- Implement gradient clipping
- Build data pipeline (glob files, tokenize, chunk, batch)
- Write training loop
- Wire into CLI: `llm train`
- Build wolf-tui training dashboard
- Train on Shakespeare or similar — verify loss decreases

### Phase 6 — Inference

- Implement forward-pass-only model loading from checkpoint
- Implement autoregressive generation loop
- Implement greedy / temperature / top-k sampling
- Implement KV cache
- Wire into CLI: `llm generate`
- Build wolf-tui streaming token view
- Verify: generate coherent text from trained model

### Phase 7 — Polish + Experiments (optional)

- Optimize matmul kernel (tiled shared-memory, 16×16 workgroups)
- Add RoPE positional embeddings as alternative
- Add SwiGLU activation as alternative
- Profile GPU utilization, optimize bottlenecks
- Train larger model on TinyStories / OpenWebText subset
- Benchmark tokens/sec for training and inference

---

## Key Concepts Learned by the End

| #   | Concept                                                   | Where you'll encounter it        |
| --- | --------------------------------------------------------- | -------------------------------- |
| 1   | GPU compute programming (WGSL, workgroups, dispatch)      | `packages/gpu`                   |
| 2   | Tensor operations on GPU (matmul, reduce, element-wise)   | `packages/tensor`                |
| 3   | Automatic differentiation (computation graph, chain rule) | `packages/autograd`              |
| 4   | Why attention works (content-based routing)               | `packages/model/attention.ts`    |
| 5   | Positional embeddings (why order matters)                 | `packages/model/embedding.ts`    |
| 6   | Residual connections (gradient highways)                  | `packages/model/block.ts`        |
| 7   | Normalization (preventing activation explosion)           | `packages/model/norm.ts`         |
| 8   | BPE tokenization (subword decomposition)                  | `packages/tokenizer/bpe.ts`      |
| 9   | Next-token prediction training objective                  | `packages/training/trainer.ts`   |
| 10  | AdamW optimizer (adaptive LR + weight decay)              | `packages/training/adamw.ts`     |
| 11  | LR scheduling + gradient clipping (training stability)    | `packages/training/scheduler.ts` |
| 12  | Autoregressive generation + sampling strategies           | `packages/inference/generate.ts` |
| 13  | KV cache (O(n²) → O(n) inference)                         | `packages/inference/kv-cache.ts` |
| 14  | Numerical stability on GPU (softmax, init, clipping)      | Throughout                       |

---

## References

- Vaswani et al., _Attention Is All You Need_ (2017) — original transformer
- Radford et al., _Language Models are Unsupervised Multitask Learners_ (2019) — GPT-2
- Touvron et al., _LLaMA: Open and Efficient Foundation Language Models_ (2023) — RMSNorm, RoPE, SwiGLU
- Karpathy, _minbpe_ (2024) — minimal BPE tokenizer reference
- Timashov, _LLM From Scratch: Building TinyGPT_ (2025) — stability tricks
- W3C, _WebGPU Specification_ (2024) — GPU compute API
- W3C, _WGSL Specification_ (2024) — shader language
- dawn-gpu/node-webgpu — Node.js WebGPU bindings
- wolf-tui — framework-agnostic terminal UI
- _Optimizing a WebGPU Matmul Kernel for 1TFLOP+_ (2025) — kernel optimization reference
