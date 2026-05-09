# How a Transformer-Based LLM Works

A progressive explanation from the simplest mental model to the full architecture, using diagrams at each level.

## Brief History

The transformer architecture was introduced in **2017** by a team at Google in the paper *"Attention Is All You Need"* (Vaswani et al.). It replaced the dominant approach at the time (recurrent neural networks / LSTMs) which processed text sequentially — one word at a time — making them slow and bad at remembering long-range context.

The transformer's key innovation was **self-attention**: a mechanism that lets the model look at all positions in a sequence simultaneously, in parallel. This made it both faster to train (GPU-friendly parallelism) and better at capturing relationships between distant words.

The timeline from there:
- **2018** — GPT-1 (OpenAI) showed that a decoder-only transformer, pre-trained on next-token prediction, could learn general language understanding
- **2018** — BERT (Google) used the encoder side for bidirectional understanding
- **2019** — GPT-2 demonstrated that scaling up (1.5B parameters) produced surprisingly coherent text generation
- **2020** — GPT-3 (175B parameters) showed that scale alone unlocks few-shot learning
- **2023–2026** — LLaMA, Mistral, Claude, GPT-4, and others pushed the architecture further with better data, training techniques, and scale — but the core transformer mechanism remains essentially unchanged from 2017

Everything in this document describes that core mechanism.

---

## Level 1 — The Simplest View

An LLM is a function: you give it text, it predicts the next word.

```mermaid
flowchart LR
    A["The cat sat on the"] --> B["LLM"]
    B --> C["mat"]
```

That's it. The entire complexity of a transformer exists to make this single prediction as good as possible.

---

## Level 2 — It's Actually Tokens, Not Words

Text isn't fed in as words. It's broken into **tokens** — subword chunks that the model can represent with numbers.

```mermaid
flowchart LR
    subgraph Tokenizer
        direction LR
        T1["The = 42"] ~~~ T2["cat = 891"] ~~~ T3["sat = 1204"]
    end

    Raw["The cat sat on the"] --> Tokenizer
    Tokenizer --> IDs["42, 891, 1204, 87, 42"]
    IDs --> Model["Transformer"]
    Model --> Logits["Probability over all tokens"]
    Logits --> Next["Token 3847 = mat"]
```

The model never sees characters or words directly — only integer IDs from a fixed vocabulary.

---

## Level 3 — Generating Multiple Tokens

The model generates text one token at a time in a loop. Each new token gets appended to the input, and the model runs again.

```mermaid
flowchart TD
    Start["Prompt: Once upon a"] --> Step1

    subgraph Step1["Step 1"]
        I1["Once, upon, a"] --> M1["Transformer"] --> P1["predicts: time"]
    end

    Step1 --> Step2

    subgraph Step2["Step 2"]
        I2["Once, upon, a, time"] --> M2["Transformer"] --> P2["predicts: there"]
    end

    Step2 --> Step3

    subgraph Step3["Step 3"]
        I3["Once, upon, a, time, there"] --> M3["Transformer"] --> P3["predicts: was"]
    end

    Step3 --> Done["Once upon a time there was ..."]
```

This is called **autoregressive generation** — the output feeds back as input.

---

## Level 4 — Inside the Transformer (High Level)

So the transformer takes in token IDs and outputs a prediction. But what actually happens in between? Think of it as a three-stage pipeline:

**Stage 1 — Embedding:** The model can't do math on raw integers like `42`. So the first step converts each token ID into a **vector** — a list of numbers (e.g. 256 numbers like `[0.12, -0.84, 0.37, 1.02, ...]`).

Think of it like GPS coordinates, but instead of 2 dimensions (latitude, longitude) describing where you are on Earth, the model uses 256 dimensions to describe where a token is in "meaning space." Tokens with similar meanings end up near each other in this space — for example, "cat" and "kitten" would have similar vectors, while "cat" and "democracy" would be far apart. The model *learns* these vectors during training — it discovers on its own which tokens should be close together and which should be far apart.

We also add a second vector that encodes *where* in the sequence this token sits — because "cat" in position 2 should behave differently from "cat" in position 50 (it's surrounded by different context).

In papers and code, these have standard names:
- **E** (or **W_E**) — the token embedding matrix. `E[42]` gives you the vector for token ID 42.
- **P** (or **W_P**) — the position embedding matrix. `P[3]` gives you the vector for position 3.
- **x** — the combined result: `x = E[token_id] + P[position]`. This is what flows into the transformer blocks.

As **x** passes through each block, it gets refined. You'll sometimes see it called **h** (for "hidden state") with a subscript for the layer: h₀ is the input, h₁ is after block 1, h₂ after block 2, etc.

**Stage 2 — Transformer Blocks:** This is where the magic happens. The sequence of vectors passes through N identical processing blocks, one after another. Each block does two things: (a) lets every token look at the other tokens to gather context, and (b) lets every token independently "think" and transform its representation. After N blocks, the model has built up a rich understanding of the entire sequence.

**Stage 3 — Output Head:** The final block's output vectors get converted back into predictions over the vocabulary. For each position, the model produces a score for every possible next token (e.g. 4096 scores), then normalizes those scores into probabilities.

```mermaid
flowchart TD
    Tokens["Token IDs: 42, 891, 1204, 87, 42"]

    subgraph Embed["Stage 1: Embedding"]
        TE["Token Embedding\neach ID becomes a vector of numbers"]
        PE["Position Embedding\neach position gets a vector too"]
        Sum["Add them together:\nthe model now knows\nWHAT each token is and WHERE it sits"]
        TE --> Sum
        PE --> Sum
    end

    subgraph Blocks["Stage 2: Transformer Blocks x N"]
        direction LR
        B1["Block 1\ngather context\nthen think"] --> B2["Block 2\nrefine further"] --> B3["..."] --> BN["Block N\nfinal refinement"]
    end

    subgraph Head["Stage 3: Output Head"]
        Norm["Normalize values"]
        Proj["Project to vocab_size scores\none score per possible next token"]
        SM["Softmax: turn scores\ninto probabilities that sum to 1"]
        Norm --> Proj --> SM
    end

    Tokens --> Embed
    Embed --> Blocks
    Blocks --> Head
    Head --> Prediction["Probability distribution:\nwhich token comes next?"]
```

**Analogy:** Think of it like a reading comprehension pipeline. Embedding = "read each word." Transformer blocks = "re-read and think about what each word means in context, multiple times, building deeper understanding." Output head = "make your prediction based on everything you've understood."

The key insight is that *stacking more blocks* gives the model more "rounds of thinking." A 2-block model does basic pattern matching. A 12-block model can track complex grammar, references, and logic.

---

## Level 5 — What's Inside One Transformer Block?

Imagine a team meeting. First, everyone shares what they know with the group — each person listens to everyone else and picks up what's relevant to their own task (**attention**: tokens talk to each other). Then, everyone goes back to their desk and thinks privately about what they just heard, updating their own notes (**feed-forward network**: each token processes independently).

That's one transformer block: a "discuss" step followed by a "think" step. The model repeats this cycle N times — each round builds richer understanding because the discussion gets more informed with every iteration.

Two additional mechanisms keep this process stable:
- **Residual connections** — a safety net that says "even if this step is unhelpful, don't lose what you already had." Each step's output is *added* to its input rather than replacing it.
- **Normalization** — keeps the numbers from growing out of control as they pass through many layers.

```mermaid
flowchart TD
    Input["Input from previous block"]

    subgraph Attention_Path["Self-Attention Path"]
        N1["RMSNorm"]
        ATT["Multi-Head\nSelf-Attention"]
        N1 --> ATT
    end

    subgraph FFN_Path["Feed-Forward Path"]
        N2["RMSNorm"]
        FFN["Feed-Forward\nNetwork MLP"]
        N2 --> FFN
    end

    Input --> N1
    ATT --> R1["+ residual"]
    Input --> R1

    R1 --> N2
    FFN --> R2["+ residual"]
    R1 --> R2

    R2 --> Output["Output to next block"]
```

**Residual connections** (`+`) let information skip over sub-layers. Without them, gradients vanish in deep networks and the model can't learn.

**RMSNorm** keeps numbers in a stable range so training doesn't explode.

---

## Level 6 — Self-Attention: The Core Idea

Attention answers: "When predicting at this position, which other positions should I focus on?"

Each token produces three vectors:
- **Query (Q)**: "What am I looking for?"
- **Key (K)**: "What do I contain?"
- **Value (V)**: "What information should I pass along?"

```mermaid
flowchart TD
    subgraph Token_Vectors["Each token produces Q, K, V"]
        direction LR
        X["Token\nembedding x"]
        X --> Q["Q = x * W_Q"]
        X --> K["K = x * W_K"]
        X --> V["V = x * W_V"]
    end

    subgraph Scores["Compute attention scores"]
        direction TB
        QK["score = Q dot K transposed\nhow relevant is each position?"]
        Scale["divide by sqrt of d_k\nprevent large numbers"]
        Mask["add causal mask\nblock future tokens"]
        Soft["softmax\nnormalize to weights 0 to 1"]
        QK --> Scale --> Mask --> Soft
    end

    subgraph Out["Produce output"]
        WeightV["weights * V\nweighted sum of values"]
    end

    Q --> QK
    K --> QK
    Soft --> WeightV
    V --> WeightV
    WeightV --> Result["Attention output"]
```

---

## Level 7 — Causal Masking (Why It Can't Cheat)

In a decoder-only model, token 3 can only attend to tokens 1, 2, 3 — never to tokens 4, 5, etc. This is enforced by the **causal mask**: a triangular matrix that blocks future positions.

```
Causal Attention Mask (who can attend to whom):

              Key position →
              Tok1  Tok2  Tok3  Tok4  Tok5
Query    Tok1 [ ✓     ✗     ✗     ✗     ✗  ]
position Tok2 [ ✓     ✓     ✗     ✗     ✗  ]
  ↓      Tok3 [ ✓     ✓     ✓     ✗     ✗  ]
         Tok4 [ ✓     ✓     ✓     ✓     ✗  ]
         Tok5 [ ✓     ✓     ✓     ✓     ✓  ]

✓ = allowed (real attention score computed)
✗ = blocked (score set to -infinity → becomes 0 after softmax)
```

The upper-right triangle is masked out. This ensures the model learns to predict the future from the past, not by peeking at it.

---

## Level 8 — Multi-Head Attention

Instead of one attention computation, we run **multiple heads in parallel** — each with its own Q, K, V projections. This lets different heads specialize in different types of relationships.

```mermaid
flowchart TD
    X["Input: batch x seq_len x d_model"]

    subgraph Heads["Parallel Attention Heads"]
        direction LR
        H1["Head 1\nd_k dims"]
        H2["Head 2\nd_k dims"]
        H3["Head 3\nd_k dims"]
        HN["Head N\nd_k dims"]
    end

    X --> H1
    X --> H2
    X --> H3
    X --> HN

    H1 --> Cat["Concatenate all heads"]
    H2 --> Cat
    H3 --> Cat
    HN --> Cat

    Cat --> WO["* W_O project back to d_model"]
    WO --> Out["Output: batch x seq_len x d_model"]
```

Each head operates on a slice of the full dimension: `d_k = d_model / n_heads`. After attention, results are concatenated and projected back to the full dimension.

**Why multi-head?** One head might learn "attend to the previous adjective," another might learn "attend to the verb of this clause," another might track long-range dependencies. The model decides.

---

## Level 9 — The Feed-Forward Network

After attention mixes information *between* positions, the FFN transforms each position *independently*:

```mermaid
flowchart LR
    X["x of size d_model"] --> W1["* W1\nd_model to d_ff"]
    W1 --> Act["GELU\nnonlinear activation"]
    Act --> W2["* W2\nd_ff to d_model"]
    W2 --> Y["y of size d_model"]
```

- **Expansion**: `d_ff = 4 × d_model` (e.g., 256 → 1024)
- **GELU activation**: smooth nonlinearity that lets the model learn complex patterns
- **Compression**: projects back to original dimension

Think of attention as "gather relevant context" and FFN as "process that context into useful features."

---

## Level 10 — Embeddings: Turning IDs Into Vectors

The model can't do math on integers. Embeddings convert token IDs into dense vectors the model can compute with.

```mermaid
flowchart TD
    subgraph Token_Embed["Token Embedding Table"]
        direction LR
        Row0["ID 0 maps to vector"]
        Row1["ID 1 maps to vector"]
        RowN["ID N maps to vector"]
    end

    subgraph Pos_Embed["Position Embedding Table"]
        direction LR
        P0["Pos 0 maps to vector"]
        P1["Pos 1 maps to vector"]
        PN["Pos N maps to vector"]
    end

    TokenID["Token ID: 42"] --> Lookup["Lookup row 42"]
    Lookup --> TV["Token vector"]

    Position["Position: 3"] --> PLookup["Lookup row 3"]
    PLookup --> PV["Position vector"]

    TV --> Add["Add"]
    PV --> Add
    Add --> Final["Combined embedding\nknows WHAT token and WHERE it is"]
```

**Why positional embeddings?** Attention is permutation-invariant — without position info, "the cat sat" and "sat the cat" would look identical to the model. Position embeddings break this symmetry.

---

## Level 11 — Training: How It Learns

The model learns by predicting the next token at every position simultaneously, then adjusting weights to reduce prediction error.

```mermaid
flowchart TD
    subgraph Data["Training Data"]
        Input["Input:  The cat sat on"]
        Target["Target: cat sat on the"]
    end

    subgraph Forward["Forward Pass"]
        Tok["Tokenize"] --> Model["Transformer"]
        Model --> Logits["Logits for each position\nvocab_size probabilities"]
    end

    subgraph Loss_Calc["Loss Calculation"]
        CE["Cross-entropy loss:\nhow wrong were the predictions?"]
    end

    subgraph Backward["Backward Pass"]
        Grad["Compute gradients\nhow to adjust each weight"]
        Opt["AdamW optimizer\nupdate weights intelligently"]
        Grad --> Opt
    end

    Input --> Tok
    Logits --> CE
    Target --> CE
    CE --> Grad
    Opt --> |"weights updated"| Model
```

**Key insight**: the model predicts the next token at *every* position, not just the last one. For a sequence of 256 tokens, it makes 256 predictions per forward pass — this is efficient self-supervised learning.

---

## Level 12 — The Loss Function (Cross-Entropy)

The model outputs a probability distribution over the vocabulary. Cross-entropy measures how far this distribution is from the correct answer.

```mermaid
flowchart TD
    subgraph Model_Output["Model Output - logits"]
        L["2.1, -0.5, 4.8, 0.3, ..., 1.2\none score per vocab token"]
    end

    subgraph Softmax_Step["Softmax: convert to probabilities"]
        S["0.04, 0.003, 0.61, 0.007, ..., 0.02\nsums to 1.0"]
    end

    subgraph Loss_Step["Cross-Entropy Loss"]
        direction TB
        Correct["Correct token was ID 2"]
        Prob["Model gave it probability 0.61"]
        NLL["-log 0.61 = 0.49"]
        Correct --> Prob --> NLL
    end

    L --> S --> Loss_Step
    NLL --> Goal["Goal: minimize this number\ndrive probability towards 1.0"]
```

- Perfect prediction → loss = 0
- Random prediction (vocab 4096) → loss = ln(4096) ≈ 8.3
- A well-trained model → loss ≈ 2.5–3.5

---

## Level 13 — Backpropagation and the Computation Graph

Every operation in the forward pass is recorded in a graph. Backward traversal computes how much each weight contributed to the loss.

```mermaid
flowchart BT
    subgraph Forward_Direction["Forward pass: left to right"]
        direction LR
        E["Embedding\nlookup"] --> A["Attention\nQ K V"]
        A --> F["FFN\nGELU"]
        F --> N["Norm"] --> L["Loss"]
    end

    subgraph Backward_Direction["Backward pass: right to left"]
        direction RL
        dL["dLoss/dLoss = 1"] --> dN["dLoss/dNorm"]
        dN --> dF["dLoss/dFFN"]
        dF --> dA["dLoss/dAttention"]
        dA --> dE["dLoss/dEmbedding"]
    end

    L -.->|"start backward"| dL
```

Each operation knows its local derivative (e.g., "how does my output change if my input changes?"). The **chain rule** multiplies these local derivatives together to get the gradient of the loss with respect to any weight in the network.

---

## Level 14 — The AdamW Optimizer

Gradients tell us *which direction* to move weights. The optimizer decides *how far* to move them.

```mermaid
flowchart TD
    G["Gradient g\nnoisy estimate of best direction"]

    subgraph Adam["AdamW Update"]
        M["m = running average of g\nmomentum - smooth out noise"]
        V["v = running average of g squared\nvariance - how confident?"]
        Update["step = m / sqrt v\nlarge step when confident\nsmall when unsure"]
        WD["weight decay\ngently shrink weights toward zero"]
    end

    G --> M
    G --> V
    M --> Update
    V --> Update
    Update --> Apply["w = w - lr * step - lr * decay * w"]
    WD --> Apply
    Apply --> New["Updated weight w"]
```

**Why not just subtract the gradient?** Raw gradients are noisy (one batch ≠ the full dataset). Adam's running averages smooth this out. The per-parameter variance scaling means frequently-updated parameters get smaller steps (already refined) while rarely-updated ones get larger steps (need more movement).

---

## Level 15 — KV Cache (Fast Inference)

During generation, we recompute the full context every time we add a token. This is O(n²) work. The KV cache eliminates redundancy.

```mermaid
flowchart TD
    subgraph Without_Cache["Without KV Cache - slow"]
        direction TB
        S1["Step 1: process A"]
        S2["Step 2: process A,B - recomputes A"]
        S3["Step 3: process A,B,C - recomputes A,B"]
        S4["Step 4: process A,B,C,D - recomputes A,B,C"]
        S1 --> S2 --> S3 --> S4
    end

    subgraph With_Cache["With KV Cache - fast"]
        direction TB
        C1["Step 1: process A, cache K_A and V_A"]
        C2["Step 2: process B only, attend to cached KV"]
        C3["Step 3: process C only, attend to cached KV"]
        C4["Step 4: process D only, attend to cached KV"]
        C1 --> C2 --> C3 --> C4
    end
```

The insight: K and V vectors for past tokens never change (they only depend on that token's position and content). So we cache them and only compute Q, K, V for the new token.

---

## Level 16 — Sampling Strategies

The model outputs probabilities over the vocabulary. How we pick from that distribution controls the creativity of generation.

```mermaid
flowchart TD
    Logits["Logits: 2.1, 4.8, 1.5, 0.3, -0.2, ..."]

    subgraph Greedy["Greedy - temperature=0"]
        G_desc["Always pick highest probability"]
        G_out["deterministic, repetitive"]
    end

    subgraph Temp["Temperature Sampling"]
        T_scale["Divide logits by T before softmax"]
        T_low["T less than 1: sharper, more conservative"]
        T_high["T greater than 1: flatter, more random"]
    end

    subgraph TopK["Top-K Sampling"]
        K_desc["Keep only K highest probabilities"]
        K_zero["Set rest to 0"]
        K_sample["Sample from remaining K options"]
    end

    Logits --> Greedy
    Logits --> Temp
    Logits --> TopK
```

In practice, you combine temperature + top-k: scale the logits for creativity control, then restrict to top-k for safety (never pick extremely unlikely tokens).

---

## Level 17 — BPE Tokenization

Byte Pair Encoding builds a vocabulary by iteratively merging the most common adjacent pairs:

```mermaid
flowchart TD
    subgraph Training["BPE Training Process"]
        direction TB
        Start["Start: split into bytes\nh, e, l, l, o, w, o, r, l, d"]
        Count1["Count pairs: l+l appears most"]
        Merge1["Merge into new token: ll\nh, e, ll, o, w, o, r, l, d"]
        Count2["Count pairs: h+e appears most"]
        Merge2["Merge into new token: he\nhe, ll, o, w, o, r, l, d"]
        CountN["... repeat until vocab_size reached"]

        Start --> Count1 --> Merge1 --> Count2 --> Merge2 --> CountN
    end

    subgraph Encoding["Encoding at inference"]
        direction TB
        Text["hello world"]
        Bytes["Split to bytes"]
        Apply["Apply learned merges in order"]
        IDs["Token IDs: 412, 87, 293"]

        Text --> Bytes --> Apply --> IDs
    end

    subgraph Decoding["Decoding"]
        direction TB
        InputIDs["Token IDs: 412, 87, 293"]
        LookupBytes["Look up byte sequences"]
        UTF8["Concatenate and decode UTF-8"]
        OutputText["hello world"]

        InputIDs --> LookupBytes --> UTF8 --> OutputText
    end
```

**Why BPE?** Common words become single tokens (efficient), rare words decompose into pieces (handles any input), and vocabulary size is fixed and controllable.

---

## Level 18 — Putting It All Together

The complete forward pass of a decoder-only transformer LLM:

```mermaid
flowchart TD
    Raw["Raw text: The cat sat on"]

    subgraph Tokenize["Tokenization"]
        BPE["BPE encode"]
        IDs["Token IDs: 42, 891, 1204, 87"]
        BPE --> IDs
    end

    subgraph Embed["Embedding Layer"]
        TE["Token embed: lookup each ID"]
        PE["Position embed: lookup each position"]
        ADD["element-wise add"]
        TE --> ADD
        PE --> ADD
    end

    subgraph Block1["Transformer Block 1"]
        N1_1["RMSNorm"] --> MHA1["Multi-Head Attention\nwith causal mask"]
        MHA1 --> RES1["+ residual"]
        RES1 --> N1_2["RMSNorm"] --> FFN1["FFN with GELU"]
        FFN1 --> RES1_2["+ residual"]
    end

    subgraph BlockN["Transformer Block N"]
        N2_1["RMSNorm"] --> MHA2["Multi-Head Attention"]
        MHA2 --> RES2["+ residual"]
        RES2 --> N2_2["RMSNorm"] --> FFN2["FFN"]
        FFN2 --> RES2_2["+ residual"]
    end

    subgraph Output["Output"]
        FN["Final RMSNorm"]
        Linear["Linear: d_model to vocab_size"]
        Softmax["Softmax to probabilities"]
        Sample["Sample next token"]
    end

    Raw --> Tokenize
    IDs --> Embed
    ADD --> Block1
    RES1_2 --> |"... more blocks ..."| BlockN
    RES2_2 --> Output
    FN --> Linear --> Softmax --> Sample

    Sample --> Decode["Decode: token ID to text"]
    Decode --> Result["the - predicted next token"]
```

---

## Level 19 — The Training Pipeline End-to-End

```mermaid
flowchart TD
    subgraph Prep["1. Preparation - once"]
        Corpus["Text corpus\n.txt files"]
        TrainTok["Train BPE tokenizer\non corpus"]
        Encode["Encode full corpus\ninto token ID array"]
        Chunk["Slice into windows\nof max_seq_len"]
        Corpus --> TrainTok --> Encode --> Chunk
    end

    subgraph Loop["2. Training Loop - thousands of steps"]
        Batch["Get next batch"]
        FW["Forward pass\npredict next token"]
        Loss["Compute cross-entropy loss"]
        BW["Backward pass\ncompute all gradients"]
        Clip["Clip gradient norm"]
        Opt["AdamW optimizer step"]
        LR["Update learning rate\ncosine schedule"]
        Batch --> FW --> Loss --> BW --> Clip --> Opt --> LR
        LR --> |"next step"| Batch
    end

    subgraph Save["3. Checkpointing"]
        Weights["Save weights .bin"]
        Config["Save config .json"]
        Tok["Save tokenizer .json"]
    end

    Prep --> Loop
    Loop --> |"periodically"| Save

    subgraph Infer["4. Inference"]
        Load["Load checkpoint"]
        Gen["Autoregressive generation\nwith KV cache"]
        Stream["Stream tokens to user"]
        Load --> Gen --> Stream
    end

    Save --> Infer
```

---

## Level 20 — What Makes This Work (Key Insights)

```mermaid
mindmap
    root((Transformer LLM))
        Attention
            Content-based routing
            Every token can attend to every other
            Causal mask prevents cheating
            Multi-head means multiple perspectives
        Scale
            More parameters means more capacity
            More data means better generalization
            More compute means better optimization
            Scaling laws predict performance
        Stability
            Residual connections for gradient flow
            Normalization for bounded activations
            Warmup for safe early training
            Gradient clipping prevents explosions
        Efficiency
            KV cache for fast inference
            BPE for compact sequences
            Weight tying for fewer parameters
            GPU parallelism for fast matmul
        Training Signal
            Next-token prediction
            Self-supervised, no labels needed
            Every position is a training example
            Cross-entropy drives probability up
```

---

## Summary Table

| Layer | Purpose | Key Formula |
|-------|---------|-------------|
| BPE Tokenizer | Text → integers | Iterative pair merging |
| Token Embedding | Integer → vector | Lookup table: `embed[id]` |
| Position Embedding | Encode order | Lookup table: `pos[i]` |
| RMSNorm | Stabilize magnitudes | `x / sqrt(mean(x²) + ε) · γ` |
| Multi-Head Attention | Mix information across positions | `softmax(QKᵀ/√d + mask) · V` |
| Feed-Forward (FFN) | Transform each position independently | `GELU(xW₁)W₂` |
| Residual Connection | Preserve information + gradient flow | `x + sublayer(x)` |
| Output Head | Vectors → vocabulary scores | `x · W_vocab` |
| Cross-Entropy Loss | Measure prediction quality | `-log(p[correct])` |
| AdamW | Intelligently update weights | `θ -= lr · m̂/(√v̂+ε) + λθ` |
| KV Cache | Skip redundant computation | Cache K, V; only compute new token |

---

## What You're Building

In this project, every single row in the table above gets implemented from scratch in TypeScript — from the WGSL GPU kernels that compute the matrix multiplications, to the autograd engine that computes gradients, to the training loop that orchestrates learning, to the inference runtime that generates text token by token.

No magic. No black boxes. Just math, running on a GPU, that you wrote.
