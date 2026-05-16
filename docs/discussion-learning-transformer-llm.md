So the thing is I actually built together with the agent, um, locally kind of a... I try to implement a very simple GPT style, uh, transform based LLM engine. So a tokenizer and a model trainer and then an inference, um, kind of part. So the goal is to educate myself. I really wanna understand better how everything works. And now I have a couple of questions, so that's what it is what is all is all about.

May 15
Ah, that's awesome. Building it from scratch locally is honestly the best way to really grok how it all fits together. You'll hit corners of the architecture that pure reading never shows you.

Fire away with your questions — what's on your mind?

Okay. So in terms of the architecture of a transformer, I'm still not a hundred percent sure, especially what pieces are in the architecture. Like, what is the embedding? And then, you know, the interlayers or what they're called, the... that there is some layers there that can even add more special meanings to words. When is the model active? So let let maybe start with the training. So what exactly, um, what parts are in the architecture and in the model when we are training it? What layers? How many dimensions? This stuff.

May 15
Right, okay, so let me break down the core pieces during training.

You've got your input tokens going in, and the first thing is the embedding layer — that's converting each token ID into a vector. So token 42 becomes like a 768-dimensional vector of floating point numbers. That's your embedding dimension, and it's one of the key hyperparameters you pick at the start.

Then you stack a bunch of transformer blocks on top of each other. Each block has two main things inside: first is the attention layer, which looks at all the previous tokens and figures out which ones matter for predicting the next word. Then after that comes a feed-forward network — just a couple of dense layers with a nonlinearity in between that adds more representational power.

You typically have somewhere between 12 and 96 of these blocks depending on how big your model is. Each block operates on the same 768-dimensional space, so the dimensions stay consistent all the way through.

At the very end, after all the blocks, you have an output projection that squashes those 768 dimensions back down to your vocabulary size — so if you have 50,000 tokens in your vocab, you get 50,000 logits out, one per token, and softmax turns those into probabilities.

What size are you building — like, how many blocks and what embedding dimension are you thinking?

So, actually, at the moment, I'm training with a vocab size of eight thousand, a max sequence length of thousand twenty four, d model of two hundred fifty six, number of heads, eight, d f f thousand and twenty four, and layers, four.

May 15
Nice, that's a solid starter config — small enough to actually train and iterate on locally, but big enough to learn meaningful patterns. So you've got four transformer blocks, each one operating in 256 dimensions. With eight attention heads, that means each head is working with 32 dimensions per head, which is a clean split. And your feed-forward layer expands up to 1024 and then back down to 256, which is typical — that expansion gives the model more capacity to learn complex transformations.

One thing to sanity-check: with a vocab size of 8000, are you training on a specific language or dataset? And what's your actual training loss looking like — are you seeing it drop meaningfully as you train?

So I'm training with, um, a subset of of, uh, Cosmopedia from Hogging Face, only stories. And I have forty three, um, files, three hundred gigabyte... yeah. Sorry. Three three hundred megabyte each. And I'm just training on two of them right now. the the the tokenizer, um, I'm also feeding two of them in, so the same the same data, actually.

May 15
Okay, so you're using Cosmopedia stories to build your tokenizer and train on a subset of that same data. That's smart — keeps things aligned. So you're working with roughly 600 megabytes of training data right now, which is honestly a nice size for local iteration. With your model size, you should see meaningful learning happen pretty quickly. A couple of practical questions: are you using byte-pair encoding for the tokenizer, or something simpler? And when you trained your tokenizer on those two files, did you end up with the full 8000 vocab or did you set it as a target and it settled there naturally?

I'm using PPE. Yes. And I'm actually trying to land at eight thousand, so the algorithm is kind of trying to get there. And, um, I see that... initially, I start with, uh, loss of nine. And after fifty thousand iterations, I went down to six dot eight. Yeah.

May 15
That's solid progress — going from nine down to 6.8 over 50k iterations shows the model is definitely learning. With a vocab of 8000 and a four-layer transformer, that trajectory makes sense. A couple things to think about: are you using any learning rate scheduling, or just a fixed learning rate? And have you hit a plateau yet, or does it still feel like it's dropping? Because at some point you'll start seeing diminishing returns, especially on a smaller model like this where four layers can only memorize so much.

I think... I I see, uh, diminishing returns. Yeah. So in the last, I would say, ten thousand iterations or so, uh, there was not much progress at all. So what I'm looking for is... now I see all these parameters, and I have a rough idea of what's happening, but maybe you can tell me more, you know, with these parameters, like vocabulary size, max sequence lengths, demodel size of two fifty six, n hats of eight, n layers of four. You know, what are the implications of these, uh, settings? How does this work out if we think about the architecture as kind of a graphical model or a model?

May 15
Good question. So let me think through the implications of your specific setup. Your max sequence length of 1024 means the model can only look back at the previous 1024 tokens — that's your context window. With stories from Cosmopedia, that's probably a couple paragraphs, which is decent but not huge. The attention mechanism within each block has to compute pairwise relationships between all tokens in that window, so 1024 squared attention scores get computed. That's quadratic, which is why longer sequences get expensive fast.

Your d_model of 256 is the bottleneck for representational capacity. Every token flows through this 256-dimensional space across all four layers. Imagine that as a narrow pipeline — information has to squeeze through there. With eight heads, each head sees 32 dimensions, which is quite small. Smaller dimensions mean less expressive power per head, but also cheaper computation. Your d_ff of 1024 is four times d_model, which is standard and gives the feed-forward layers room to learn nonlinear transformations.

The big constraint is

Okay. Wait. Let's let's, uh, focus on the model. What does it mean?

May 15
Yeah, good call. Let me zoom in. So when I say "model," I'm talking about the actual weights — the numbers that get trained. In your transformer, you've got embedding weights, attention weights in each of your four blocks, feed-forward weights in each block, and the output projection weights. All of those together — that's your model. During training, those weights get updated over and over to minimize loss. During inference, those frozen weights process new tokens. Does that make sense, or

But what does what does d model two hundred fifty six means? Is it two hundred fifty six dimensions in the vector, or is that the tensor size? Or...

May 15
Good question. It's 256 dimensions in the vector. So every token gets embedded as a 256-dimensional vector — that's d_model. Think of it as a point in 256-dimensional space. Each dimension is a separate weight that the model learns to fill with meaning. So when you feed a token through the embedding layer, you get back a vector with 256 floating-point numbers. That same 256-dimensional representation flows through

Okay. Okay. And what makes sure that, like, each each, um, number in the vector... each position, sorry, in the vector is actually the same meaning, the same kind of kind. Right? So I'm looking at the dimension as a kind. Let's say one is, I don't know, I don't know, I don't know how to compare it. But if it's if it's x y zed in space, how how does how does the the model then know that the first position is always let's say, the aggressiveness of a word and the second position is whatever else, quality.

May 15
Ah, that's a really good question. Honest answer: it doesn't guarantee that, and we don't really control it. The model learns what each dimension means through training. So dimension zero might end up encoding something like "is this a noun," dimension one might encode "sentiment," dimension 47 might encode "frequency in the training data" — but it's emergent. There's no explicit rule saying "dimension zero is always nouns." It just falls out of the optimization process. We can kind of reverse-engineer it after training by looking at which tokens activate which dimensions, but we don't pre-assign meanings. The key insight is that the model figures out whatever internal representation is most useful for predicting the next token. If splitting out "aggressiveness" into a separate dimension helps it predict better, it'll learn to do that. If mixing concepts helps more, it'll do that instead. So the dimensions are learned features, not pre-defined ones. Does that make sense?

Yeah. Kind of makes sense. It's just somewhat magical. That's the the the the thing about it. It's really weird to think about it like that. How many dimensions do, like, the big models have?

May 15
Yeah, it really is magical when you think about it. Big models like GPT-4 or Claude have d_model values in the thousands. GPT-3 had 12,288 dimensions. So imagine 12,288-dimensional space instead of your 256. That's way more room for the model to carve out distinct features and meanings. More dimensions means more expressivity, but also way more parameters to train and more compute needed. Your 256 is tiny by comparison, which is why your model hits that ceiling pretty quick. But honestly, for learning and understanding how this all works, your setup is perfect — small enough to actually train locally and see what's happening, but big enough that it's the same architecture as the massive models.

So let's start with the transformer architecture in a trained model, um, or let's say it's a model that is still trained. And what is... I mean, now I'm really trying to get through this. Um, so what is the the first layer in the transformer architecture is the embedding layer. Right?

May 15
Right, exactly. The embedding layer is the first thing. Token IDs come in — those are just integers from zero to 7999 in your case — and the embedding layer converts each one into a 256-dimensional vector. So if you feed in token ID 42, it looks up row 42 in an embedding table and pulls out a 256-dimensional vector. That's it for the embedding layer — it's basically a lookup table with 8000 rows and 256 columns. Then those embedded vectors flow into your first transformer block.

Okay. So we start with the vocabulary. Uh, and the vocabulary is really, like, what words are known to the model. Is that true?

May 15
Exactly. Your vocabulary of 8000 is the set of tokens the model knows. Each token has a unique ID from zero to 7999. When you tokenize text, you break it into these 8000 known pieces. Anything outside that vocabulary can't be represented — well, actually in practice you'd have an unknown token or special tokens, but the point is: if a word wasn't in your training data when you built the tokenizer, it won't be in the vocab and the model can't handle it directly. So vocab size is a fundamental constraint on what the model can express.

So, um, if a model is only English speaking, wouldn't it need, like, a vocab size of all the English words that exist?

May 15
Good instinct, but not quite. You don't need every single English word — there are hundreds of thousands of English words, but most people only use a few thousand regularly. So a vocab of 8000 covers maybe 99 percent of common English text. The tokenizer also does subword tokenization with byte-pair encoding, which you're using. So even if a rare word isn't in the vocab as a whole token, it gets broken into smaller pieces — maybe prefixes and suffixes — that are in the vocab. So the model can still represent rare words by combining multiple tokens. That's the whole point of byte-pair encoding: you trade off vocab size for sequence length. A model with vocab 50,000 will handle rare words more cleanly in fewer tokens. Your vocab of 8000 means more tokens per sentence, but smaller memory footprint.

But, uh, multilingual model that supports, like, hundred languages or whatever, uh, needs, uh, way bigger vocab size. No?

May 15
Yeah, exactly. A multilingual model needs way bigger vocab because you're covering characters and subword pieces from a hundred different writing systems and languages. English alphabetic text compresses really efficiently with byte-pair encoding, but throw in Chinese characters, Arabic script, Cyrillic, and suddenly you need way more tokens to represent the same amount of meaning. So multilingual models typically have vocab sizes in the hundreds of thousands — GPT-3 had around 50,000, but models like BERT that support many languages go much higher. The tradeoff is real: bigger vocab means fewer tokens per sentence, but more memory and slower embedding lookups. Smaller vocab means more tokens and longer sequences, but cheaper overall.

Okay. So then let's start from the beginning. When I start to train a model freshly, then the model is using the tokenizer and the result of that. So the... let's say, in my case, the eight thousand tokens in the vocabulary. And then, um, it creates, um, one matrix for... or one vector for every for every single word with the two hundred fifty six dimensions that we've, uh, we've discussed. And how does it create that initially, or is the embedding itself. Um, Yeah. How does the first embedding come come to existence? I I don't understand how... is that part of the tokenizing? Is... what is what is the quality of, uh, embedding created by the tokenizer, or is the tokenizer even creating embeddings? No. Right? So how is the first embedding created without the model knowing anything?

May 15
Great question — this is a key distinction. The tokenizer doesn't create embeddings at all. The tokenizer just breaks text into token IDs. So "hello world" becomes maybe token 42 and token 89. That's it. Then separately, the embedding layer is part of the model, and it starts with random weights. So you've got an 8000 by 256 matrix — 8000 rows for each token, 256 columns for the embedding dimensions — and initially every number in that matrix is random garbage. When you feed token 42 in, you just look up row 42 and get 256 random numbers. Then during training, backprop updates those random numbers to become meaningful. So the embeddings aren't created or known by the tokenizer — they're learned from scratch by the model during training. The tokenizer is purely a text-to-numbers converter, nothing semantic about it.

Okay. Got it. So it's initializing, um, eight thousand times the two hundred fifty six random numbers in the vector at every single position is a random number. And then I have this initial embeddings in the embedding layer. What's next?

May 15
Right, so you've got your random embedding vectors. Now a token ID flows through: it gets looked up in that embedding table and you get a

Wait. Wait. Wait. What do you mean the token is flowing by? So now, um, it's... we start to feed it the training data.

May 15
Yes, exactly. During training, you feed in sequences from your training data. So you tokenize a sentence from Cosmopedia into token IDs, then you feed those IDs one by one or as a batch into the model. Each token ID gets embedded by looking it up in that random embedding table. So if your sequence is tokens 42, 89, 156, each one becomes a 256-dimensional vector. Then those vectors flow through your four transformer blocks, which manipulate and transform them based on their random weights. At the end, you compare the model's prediction to what the actual next token should be, calculate the loss, and backprop to update all those weights —

Okay. Wait. Wait. Wait. Wait. You're going too fast again. So we're still at, um, the embedding layer. So my first token comes in from the training data. And then, basically, um, it gets recognized. The embedding is getting selected for that token. And then, uh, the next layer is feed forward. Right?

May 15
No wait, not feed forward yet. After embedding, the token vector goes into your first transformer block. Inside that block, there are two things: first is the attention layer, which looks at all the previous tokens in the sequence and figures out which ones are relevant for this token. Then after attention comes the feed-forward layer. So the order is: embedding, then transformer block one (attention plus feed-forward), then transformer block two, and so on through all four blocks. Does that make sense?

Okay. So then let's talk about the attention layer.

May 15
Good. So the attention layer is where the magic happens. It takes your embedded token vector — that 256-dimensional vector — and asks: which previous tokens in the sequence should I pay attention to? It does this by computing relevance scores between the current token and every previous token. You've got three things in attention: Q, K, and V. Q is the query — it's a projection of your current token asking "what should I look at?" K is the key — projections of all previous tokens saying "here's what I am." V is the value — the actual information from those previous tokens. The model computes a similarity score between Q and each K, then uses those scores to weight which V's matter. Tokens that score high get attended to, low-scoring ones get ignored. Your eight attention heads do this in parallel, each looking at 32-dimensional chunks of your 256-dimensional space. So you get eight different views of what matters, then concatenate them back together. Does that make sense?

Can you give me a very simple, like, mechanical walk through? Let's say, I have, um, we train on on some data that is hello world. Let's let's say hello world. Let's go through these stages that we have so far with these two tokens. Hello world.

May 15
Perfect. So you've got two tokens: "hello" becomes token ID, let's say, 42. "world" becomes token ID 89. During training, you feed in token 42 first. The embedding layer looks up row 42 in the random embedding table and pulls out a 256-dimensional vector — let's call it a bunch of random numbers. That vector flows into transformer block one. Inside block one is the attention layer. The attention layer projects that 256-dimensional vector into three things: a query Q, a key K, and a value V. Each is also 256-dimensional. Then attention computes how much token 42 should attend to itself — it computes a score between Q and K, applies softmax to turn it into a probability, and uses that to weight the V. Since there's only one token so far, it just attends to itself. The output is a new 256-dimensional vector. Then that flows through the feed-forward layer in block one, which applies some nonlinear transformations and outputs another 256-dimensional vector. That vector flows through blocks two, three, and four the same way. At the very end, you project from 256 dimensions down to 8000 logits — one score for each possible next token. The model predicts token 89 should come next, but its prediction is probably garbage because everything is random. Then you calculate loss and backprop to update all the weights. Does that track?

Honestly, it was it was a bit too fast.

May 15
You're right, my bad. Let me slow way down. So you've got "hello world." Token 42 is "hello," token 89 is "world." We're training, so we feed in token 42 first. The embedding layer looks it up and outputs a 256-dimensional vector of random numbers. Let's call that vector E42. That's it for now — E42 just sits there. What happens next in the attention layer?

So the attention layer has three vectors.

May 15
Right. So the attention layer takes E42 and projects it into three separate vectors. Q, which is the query. K, which is the key. And V, which is the value. Each one is 256-dimensional. They're all derived from E42, but they're different projections — different linear transformations of the same input. Does that make sense so far?

But they are persistent. So, like, does every one of the eight thousand... or how should I put this? Yeah. Does does every, like, token that flows in during training has its own persistent, uh, three vectors in the attention layer? Or how is... how does that work?

May 15
Good question. No, they're not persistent per token. The Q, K, V projections are weights in the attention layer — they're shared across all tokens. So there's one set of Q weights, one set of K weights, one set of V weights. Every token that flows through gets projected by those same weights. Token 42 gets projected into its own Q, K, V. Then token 89 gets projected into its own Q, K, V using the same weights. The weights themselves persist and get updated during training, but each token gets its own Q, K, V vectors computed on the fly. Make sense?

Okay. So Then in the training, um, presumably, I train my model with thousand words, and each of the words is getting its own tell... token. So every of the thousand tokens I use to train gets its own q, k, v vector.

May 15
Exactly right. So if your sequence is a thousand tokens long, each of the thousand tokens gets its own Q vector, its own K vector, and its own V vector. All computed using the same shared weights in the attention layer. So you've got a thousand Q's, a thousand K's, and a thousand V's all flowing through at the same time. Then attention compares all the Q's against all the K's to figure out which tokens should attend to which.

And in training, um, so I have three hundred megabytes of of text. So while training, how do we do we put all that in? Probably not. Right? That would be too... way too much. Right? So is it, like, chunked? Is it split up? And then is it selected randomly from all this sample like a passage? Or how is that done?

May 15
Right, you can't fit 300 megabytes into memory all at once. So you split it into chunks — typically called batches. Your max sequence length is 1024, so you probably take passages of 1024 tokens at a time. Then you might group several of those together — say, 32 passages of 1024 tokens each — and that's one batch. You run that batch through the model, calculate loss, backprop, update weights. Then you grab the next batch. You keep cycling through your training data, grabbing random batches, until you've seen enough of it. That's one epoch. You typically do multiple epochs so the model sees different random orderings of the data. Does that make sense?

And q k v is actually dropped after after learning from one sequence?

May 15
Yes, exactly. Q, K, V are computed on the fly for each sequence. After you process one batch and backprop, those specific Q, K, V vectors for that batch are thrown away. The next batch comes in, and you compute brand new Q, K, V vectors using the same weights. The weights themselves persist and get updated — that's what learning is — but the actual Q, K, V vectors are temporary, computed fresh for each input.

Okay. So then let's go back to our hello world, uh, example. So now hello as a first token comes from the embeddings and then gets... this vector is actually getting into the attention layer. So how is q k v also random, or how do they get initialized? What's How how do they work?

May 15
Good question. So Q, K, V are computed by multiplying E42 by weight matrices. Those weight matrices are initialized randomly at the start, just like the embedding table. So when token 42 flows in as E42, the attention layer does three matrix multiplications: Q equals E42 times W_Q, K equals E42 times W_K, V equals E42 times W_V. W_Q, W_K, W_V are weight matrices that are part of the model and get trained. Since they start random, the Q, K, V vectors that come out are also effectively random initially. But as training progresses and those weight matrices get updated by backprop, the Q, K, V projections become more meaningful. Does that click?

Alright. And, uh, weight matrices, um, so we have three of them, or do we have, uh, three of them per all the trained tokens? Or how can I imagine that?

May 15
Three weight matrices total per attention head. You've got eight attention heads, so actually you've got three times eight — 24 weight matrices in the attention layer. Each head has its own W_Q, W_K, W_V. So head one projects E42 into Q1, K1, V1. Head two projects E42 into Q2, K2, V2. And so on through all eight heads. Then each head computes attention separately, and at the end you concatenate all eight outputs back together. Does that make sense?

Okay. That now brings in another concept of the hats. So you need to explain to me what the attention hats mean.

May 15
Right, so attention heads are just a way to split the work. Instead of doing all the attention computation in one 256-dimensional space, you split it into eight smaller spaces of 32 dimensions each. Each head independently computes its own Q, K, V projections and its own attention scores. So head one looks at a 32-dimensional slice and figures out which tokens matter. Head two looks at a different 32-dimensional slice and figures out its own version of which tokens matter. Then you concatenate all eight outputs back together into 256 dimensions. The idea is that different heads can learn to focus on different kinds of relationships — maybe head one learns to track pronouns, head two learns to track sentence structure, head three learns something else. It's like having eight parallel attention mechanisms each with their own view of the data. Does that make sense?

Well, what doesn't make sense so far is I I actually don't understand. So whenever we token of the training data comes in, and then there is the embedding vector for that token. And the embedding vector flows into the attention layer. The attention layer then actually computes the queue k v based on the matrix or the matrices. Um, how does it learn? Because it does not yet... I mean, from what we've discussed, it it it doesn't compare to other words or something like that.

May 15
Ah, that's the key question. You're right — so far we've only talked about one token. Let me add the next token. So "hello" comes in, gets embedded, flows through attention. But attention doesn't just look at one token — it looks at all tokens that came before it. In training, you feed in a whole sequence: tokens 42, 89 (hello, world). Token 42 embeds to E42. Token 89 embeds to E89. Now in the attention layer, token 89's Q vector compares itself against token 42's K vector. That comparison tells you "should token 89 pay attention to token 42?" The model learns by comparing its predictions against the ground truth. Token 42 predicts the next token should be 89. Does it guess 89? Probably not, because everything is random. So backprop updates the weights to make it better. Over thousands of iterations, the attention weights learn to connect related tokens. Does that clarify it?

I still have issues understanding how q, k, and v are getting logical parts in a system becoming logical parts in a system. For me now, they're just detached vectors that compute within detached matrices. Why is q a question and k a key and v a value?

May 15
That's a really good question, and honestly it's somewhat of a naming abstraction. Let me try to make it concrete. Think of it like a retrieval system. Q is your query — "I'm looking for something." K is a key in a filing cabinet — "here's a label." V is the actual value in the filing cabinet — "here's the information."

So when token 89 comes in, its Q asks "what previous tokens are relevant to me?" You compute a similarity score between token 89's Q and token 42's K. High similarity means "yes, token 42's information is relevant." Then you use that score to weight token 42's V — its actual information. If the score is high, you take a lot of token 42's V. If low, you take very little.

The magic is that during training, the model learns to set up Q, K, V so that relevant tokens get high similarity scores and irrelevant ones get low scores. Token 89 learns to craft a Q that matches the K of tokens it should attend to. Token 42 learns to craft a K that matches queries from tokens that depend on it. It's not magic — it's just that backprop adjusts the weight matrices so this retrieval pattern emerges naturally. Does that help?

Okay. So on my... in my simple model of, um, of GPT, um, and LLMs in general, um, transformer based LLMs, I always found it really logical that during training, we can actually compare a random next token prediction to whatever is the next token in the training data. Alright? So we can actually verify you know, that this is a valid next token because we have some occurrences of that in training data. So we can actually compare to real data. But how on earth can you actually verify if one token is related to another one, uh, that is before it... I mean, I don't understand how we can validate or how the system can learn because isn't that just random? I mean, Yeah.

May 15
That's the deepest question you've asked so far, honestly. And the answer is: you can't directly validate it during training. You're right — there's no ground truth label saying "token 89 should attend to token 42 with strength 0.7." The model has to figure it out indirectly. Here's how: the model learns attention patterns that help it predict the next token correctly. So if attending to token 42 when processing token 89 helps predict the correct next token, then backprop rewards that attention pattern by updating the weights to strengthen it. If ignoring token 42 helps prediction, it weakens that pattern. It's all emergent from the single objective: predict the next token. The attention weights aren't trained directly — they fall out of optimizing for next-token prediction. So in a weird way, the model figures out which tokens are related by discovering which attention patterns make better predictions. Does that click?

Yeah. That makes sense. Okay. So it's an indirect, um, it's an indirect learning, actually. And it just... it's it's based on the fact that next word prediction is actually depending on knowing the context to a to a very good extent by applying those queries and finding the right keys with those values. And if this is done correctly and the prediction was correct, then those queries get rewarded kind of.

May 15
Exactly. You nailed it. It's all indirect and emergent from the single loss signal — next token prediction. The model discovers that attending to relevant previous tokens helps predict better. So over thousands of iterations, the Q, K, V weights evolve to create attention patterns that actually make linguistic sense, even though nobody explicitly programmed "pronouns should attend to nouns" or anything like that. The model figures out the logic of language purely by optimizing for prediction accuracy. That's the beautiful and somewhat mysterious part of how transformers work.

Are you there?

May 15
Yeah, I'm here. Sorry, I think my response got cut off. I was saying: that's the beautiful and somewhat mysterious part of how transformers work — they learn meaningful linguistic structure purely by chasing the next-token prediction signal. No explicit instruction needed.

Okay. So why do we need q, k, and v? I mean, just thinking about it very simply, wouldn't a q k lookup already be sufficient?

May 15
Great question. So Q and K alone give you similarity scores — which tokens to attend to. But that's just weights, zero to one. You still need to decide what information to actually pass along from those attended tokens. That's where V comes in. V is the actual content. So Q-K tells you "pay attention to token 42." V tells you "here's what token 42 actually contains that you should use." You could theoretically use the token's embedding directly as V, but then you'd lose flexibility. With separate V, the model can learn to extract only the relevant information from each token for this particular attention head. One head might extract grammatical structure from V, another might extract semantic meaning. Same K-Q similarity scores, but

So v is not the embedding.

May 15
Right, V is not the embedding. V is a separate projection of the embedding, just like Q and K are. So you've got E (the embedding), and then you compute Q equals E times W_Q, K equals E times W_K, and V equals E times W_V. All three are different linear transformations of the same embedding. They start from the same source but get projected into different spaces by different weight matrices.

But then how is v related to the embedding of the token that was, um, was found that attention is is, uh, pointing to?

May 15
Good question. So let's say token 42 is being attended to by token 89. Token 42's embedding E42 gets projected into V42 using the weight matrix W_V. Token 89's Q89 computes a similarity score with token 42's K42. If that score is high, then token 89 takes a lot of token 42's V42 — the projected version of token 42's embedding. So V42 is derived from E42, but it's a transformed version. The transformation W_V is learned during training to extract whatever information is useful to pass along. Without V being separate from the embedding, you'd be forced to pass along the raw embedding unchanged, which is less flexible.

Oh, so WV is a transformation matrix that transforms the attended, um, vector into a shape that is relevant.

May 15
Exactly. W_V transforms the embedding of the attended token into a form that's useful for the current context. So token 42's embedding E42 gets multiplied by W_V to produce V42 — a version of token 42's information that's been shaped to be relevant. The model learns what shape that should be through training. Different heads have different W_V matrices, so each head extracts different aspects of the attended token's information.

Okay. Let's take an example of a already trained model, and then there is a prompt coming in that says, I have a dog. I'm going to... I'm going for a walk with him. So let's say him is related to dog. I guess there is a relation there, um, when a fully trained model sees this. So let's build let's build, like, a mental model. What could v the v matrix, the WV matrix, might contain, or how could it transform dog?

May 15
Good example. So when the model processes "him," the attention mechanism compares "him's" Q against all previous tokens' K's, including "dog's" K. If the model learned that pronouns should attend to nouns they refer to, then "him's" Q will have high similarity with "dog's" K. Then "dog's" V — which is "dog's" embedding transformed by W_V — gets weighted heavily into the output. What might W_V have learned? Maybe it extracts gender information from "dog" (masculine), maybe it extracts animacy (living thing), maybe semantic role (the thing being walked with). The model learned that when a pronoun attends to a noun, these properties are relevant to predict what comes next. So W_V might suppress irrelevant details about "dog" — like its color or size — and amplify the grammatical gender and semantic role. It's not explicit, but through training on thousands of pronoun-noun pairs, W_V learns to extract what matters.

Do people really understand all this? It sounds so crazy.

May 15
Haha, honestly? No, not really. Even the researchers who built transformers don't fully understand how all this works at a deep level. We can describe the mechanics — Q, K, V, attention scores, backprop — but why it works so well, why it discovers linguistic structure, that's still somewhat magical. You're hitting on something real: it's a black box. We know next-token prediction is the training signal, and somehow that optimizes weights to create meaningful attention patterns. But the actual learned representations inside — what each dimension means, how different heads specialize — that's hard to reverse-engineer. The fact that you're questioning this means you're thinking about it right. Most people just accept "attention works" without really grappling with the mystery.

And I understand attention heads are not only important for performance optimization during training because, I guess, they can run-in parallel. They're also important to specialize.

May 15
Exactly. Yes, there's definitely a performance angle — you can parallelize eight heads. But the specialization is the bigger deal conceptually. Different heads learn different types of relationships. One might learn syntax, another semantics, another tracking entities. If you had one giant attention head instead of eight smaller ones, the model would have to learn all those relationships in one place. With eight heads, each can specialize. And empirically, when you look at what trained models learn, you do see this — some heads clearly focus on pronouns, others on structural relationships. So it's not just a trick for speed, it's actually a better learning architecture.

Can there be a, like, a practical limit for Hats? So if we assume that we have, uh, a vector dimensions of two hundred fifty six and we have eight Hats, that means we have, what, thirty two, uh, vectors... vector dimensions in every hat. Is that correct?

May 15
Yes, exactly right. 256 divided by 8 equals 32 dimensions per head

So if we then, um, say, would have, um, hundred twenty eight heads, then we had only two dimensions per head. I guess this would be a problem.

May 15
Exactly right. With 256 dimensions and 128 heads, you'd get two dimensions per head, which is way too small. Each head needs enough dimensions to learn meaningful patterns. Two dimensions is basically just a line in space — you can't encode much complexity there. In practice, people keep d_k (dimensions per head) at least 64 or so. GPT-3 has d_model of 12,288 and 96 heads, so d_k is 128 per head. Your setup has 256 divided by 8, so 32 per head, which is on the smaller side but still workable. So there's definitely a practical lower limit. You need enough dimensions per head to actually learn something. That's one reason bigger models have both more heads and larger d_model — they grow together.

So it... was this already part of the paper from Google in twenty seventeen?

May 15
Good question. The original transformer paper in 2017 did use multi-head attention, yes. But I don't think they spent a ton of time justifying the specific design choices — like why eight heads, why that exact ratio of heads to dimensions. It was more empirical: they tried it, it worked well, so they used it. The paper shows the architecture worked better than RNNs, but a lot of the deeper understanding of why different heads specialize has come from later research analyzing trained models. So the mechanism was there from 2017, but the full appreciation of what it does has evolved since then.

And the attention architecture with QKV, is that something, I guess, that that was in the paper for sure. But how did they come up with this? I mean, it could also be a different way of thinking how to represent relations. Was there any rationale to this? Was there any previous research?

May 15
Yeah, that's a great historical question. The Q, K, V framework wasn't entirely new — attention mechanisms existed before transformers. But the specific formulation of scaled dot-product attention with separate Q, K, V projections was a key innovation in that 2017 paper. The intuition was: you want a fast, differentiable way to compute "which tokens matter?" Dot products are cheap and differentiable, so Q dot K gives you similarity. Scaling by square root of d_k prevents scores from blowing up. Then using those scores to weight V values is a natural retrieval idea. But honestly, I'm not 100 percent sure how much they borrowed from prior attention work versus invented fresh. That's more of a history-of-deep-learning question. Do you want me to search for the actual paper context?

Let's, um, go back to our architecture. I wanna fully understand that. So we are now through embedding layer to the attention layer for every new word or token in our sequence that we're train... we're training. We look back at all the tokens before and compute, uh, queue, or we gonna look up if we have matches. with with queue attention matches for all hats that we have. And then what happens then?

May 15
Right, so after attention computes those weighted values for each head, you've got eight output vectors — one from each head, each 32 dimensions. Then you concatenate all eight together back into 256 dimensions. After that, you multiply by a projection matrix W_O to project back to the full 256-dimensional space. So the output of the attention layer is a single 256-dimensional vector that incorporates information from all eight heads. That output vector then goes into the feed-forward network, which is the second part of the transformer block.

So is attention and self attention the same thing?

May 15
Good question. Self-attention means a token is attending to other tokens in the same sequence — including itself. Regular attention could mean attending to a different sequence entirely. In transformers, we use self-attention — each token looks back at the sequence it's part of. So yes, in the context of transformers, attention and self-attention are essentially the same thing. Every token attends to tokens in its own sequence.

But only to tokens before. Right? Never after.

May 15
Right, exactly. That's the causal mask we talked about earlier. During training and inference, each token can only attend to tokens that came before it in the sequence. Never to future tokens. That's what prevents the model from cheating — it can't look ahead to predict the next word. It has to learn from context that's already available.

Okay. So let's let's go back to the math. Um, so we... what's the math involved to discover similarity q and k?

May 15
So the similarity is computed as a dot product. You take Q — the query vector from the current token — and K transpose — the key vector from a previous token flipped around — and multiply them together. Q dot K gives you a single number, the similarity score. High number means similar, low means dissimilar. Then you divide by the square root of d_k — which is 32 in your case — to scale it down so the softmax doesn't saturate. Then you apply softmax to turn those scores into weights that sum to one. Does that make sense mathematically?

Yeah. The the dot product is actually a way how to find out if a vector is similarly aligned to another vector. Right?

May 15
Exactly. If two vectors point in the same direction, their dot product is large. If they point in opposite directions, it's negative. If they're perpendicular, it's zero. So dot product is a measure of alignment. In attention, Q and K are learned to align when the tokens are semantically or syntactically related. Token 89 learns to craft a Q that aligns well with token 42's K when they're related.

So q is just a vector, and k is also a vector. So us saying this is query is actually just that we can give a meaning to it. Right? It's, uh, we actually give mathematically a meaning to it because we use q to check from the current token to check if any case from the previous tokens are similar. So q is kind of the lookup. It's kind of the query vector, and we look for similar case... keys. And we named this key because we identify those tokens by this vector. Right?

May 15
Perfectly said. You nailed it. Q and K are just vectors mathematically, but we give them meaning by how we use them. Q is "this is what I'm looking for" because we use it to search. K is "this is what I am" because we compare against it. V is "here's my information" because we weight and sum it. The names are metaphors that help us think about what's happening, but mathematically they're all just vectors being multiplied and compared. The model learns their actual meaning through training.

But what I what I don't get is, like... now if if if key would... k, the key, um, would actually identify, like, the position in the sequence of words, then it would make sense for me, like, logically, okay. We're actually looking for that position. But how is k... I mean, k is not really encoding the position. It's not encoding the meaning of the word. It's I don't get that.

May 15
That's the thing — K actually encodes both implicitly. K is a projection of the embedding, and the embedding already contains information about the token's meaning. But K also gets influenced by position because the embedding

Okay. Can you tell me again how k is projected from the embedding? Is that through the transformation may... matrix q k?

May 15
Right, so K is projected from the embedding using W_K. The embedding E contains information about what the token is — its semantic meaning. Position information comes separately from the position embedding, which gets added to the token embedding before it even enters attention. So when you compute K equals

Okay. But the q... WK, sorry, WK, how is that transformation matrix initialized? How is it coming... I mean,

May 15
W_K is initialized randomly, just like all the weight matrices in the model. It starts as random numbers. During training, backprop updates those random numbers so that K vectors become more useful for matching with Q vectors. The model learns

Okay. Can you remind me again w k w q w v? How can I imagine them being persistent? I mean, these are weights. Right? So they need to be persistent. They get trained. Right? How many of them exist again? I'll... I forgot that.

May 15
Right, so W_Q, W_K, W_V are weight matrices — they're persistent parameters of the model that get trained. You've got eight attention heads, so you have eight sets of these. That's eight W_Q matrices, eight W_K matrices, and eight W_V matrices. Each head has its own set. They persist across the entire training — every token that flows through uses the same W_Q, W_K, W_V for each head. Backprop updates them incrementally with each batch. After training, they stay frozen during inference. Does that make sense?

Okay. So, uh, w q, w k, and w v, I have only, uh, two hundred fifty six, um, values in those three, um, matrices. And those three matrices are existing n times as many heads as I have, and that's all I have in the model attention layer in the whole model.

May 15
Not quite. Each W_Q, W_K, W_V matrix is 256 by 32 — it takes a 256-dimensional input (the embedding) and projects it to 32 dimensions (the per-head dimension). So each matrix has 256 times 32 numbers, which is 8,192 values. And you've got eight heads, so that's eight copies of each. So roughly 24 times 8,192 — about 196,000 weights just in the attention part of one block. Multiply that by four blocks, and you're looking at hundreds of thousands of weights in attention alone, before you even count the feed-forward layers.

Okay. I'm slowly but surely seeing the direction of this. Okay. So let me think about this. Okay. So we have we have... so the matrices... so these are the tensors. Right? We we we we call them also tensors. These matrices of the attention heads. So we can we can actually say each attention head has a fraction of the total number of dimensions.

May 15
Exactly. Each attention head operates on a slice of the 256-dimensional space — specifically 32 dimensions. So you can think of it as splitting the 256-dimensional representation into eight

So it has actually... when we have two hundred fifty six dimensions, and then we have, uh, eight heads, we have thirty two dimensions inside of each head. So for w, uh, sorry. For w q, w k, w v, which are themselves... so so these are the the matrices. And for each of these, we have these thirty two by by two hundred fifty six again. That's in those three matrices. So three times the three times the thirty two times two hundred fifty six times the number of heads.

May 15
Wait, let me clarify. Each W_Q, W_K, W_V is 256 by 32. So one W_Q matrix has 256 times 32 equals 8,192 values. You have eight heads, so eight W_Q matrices — that's eight times 8,192 equals 65,536 values just for all the W_Q's. Same for

So let's say let's say we don't have multihead attention. Let's say we have one single attention. So that would mean the... in that attention layer, in such an attention layer, we would have three matrices, w, q, w, k, w, v, and each of them is two hundred fifty six by two hundred fifty six. Right?

May 15
Exactly right. With single-head attention, each matrix would be 256 by 256. W_Q takes a 256-dimensional embedding and projects it to 256 dimensions. Same for W_K and W_V. So you'd have three matrices of 256 by 256, which is 65,536 values each. Three times that is about 196,000 values total for the attention weights in one block.

Okay. And one dimension of the matrix, column row, uh, let's say column, is actually the size of the vectors because we need to multiply by those vectors. And the other dimension of the matrix is actually the attention space let's say.

May 15
Exactly. One dimension is the input size (256 — the embedding dimension), and the other dimension is the output size (256 for single-head, or 32 per head for multi-head). When you multiply embedding vector times W_Q, you get an output vector in that output space. So yes, you can think of the columns of W_Q as defining the "attention space" — the directions that the model learns are important for queries.

Okay. I think for simplification, we can also sometimes just think about a single head attention mechanism. But now we are having a c... a training sequence of thousand twenty four tokens. And now we go... let's say we are, um, let's say with the example of hello world again. Or let's say that's our... that's all our tokens we have in the sequence, two tokens. So hello, world. Let's say world or... yeah. Let's say let's say we are currently at the world. So we're then actually comparing, um, the vector alignment of of our queue of world to the k of hello. And then we compute the attention. And let's say we find out This is similar because let's say we trained already with a lot of other sequences. Right? These are also kind of... we have already a trained network, and then we find out that in some way, one attention... way of of doing attention or or interpreting attention says, yeah. Hello. World is relevant to hello because it's the words that you focus on when you say hello. Hello to who, right, to the world. So let's say that's... I I know also good in grammar, so I don't know exactly the the, um, the proper word for it. But let's say we find this similarity, and now we are done with the attention. We get some outputs which signal something. So how is it going fur... further down then?

Show more
May 15
Right, so after attention computes that similarity and figures out "world should attend to hello," it weights hello's V vector heavily. Then it sums up all the attended V vectors — in this case mostly hello's V. That summed vector is the output of attention. It's still 256 dimensions. Then that output goes through a residual connection — it gets added back to the original embedding of "world." Then it goes into the feed-forward network, which applies two linear transformations

Wait. Wait. Sorry. But can you repeat that? I I didn't quite follow. So what what happens with the with the with the v vector?

May 15
Sorry, let me slow down. So after attention computes similarity scores between "world's" Q and all previous tokens' K's, it uses those scores as weights. "Hello" gets a high score, so you take a lot of "hello's" V. You multiply hello's V by that weight, getting a weighted version of hello's V. Since there's only one previous token in our example, that weighted V is basically the output of attention. In a longer sequence, you'd do this for all previous tokens and sum all the weighted V's together. That sum is the output of attention — still 256 dimensions. Does that make sense now?

Yeah. Okay. And what happens next?

May 15
You sum up all the weighted V's. In your two-token example, you'd have hello's weighted V plus world's weighted V (world attends to itself too because of the causal mask). That sum is a single 256-dimensional vector — the attention output. That's what flows out of the attention layer.

So in a sequence of two hundred, uh, of of thousand twenty four, um, tokens, we gonna just sum all the v's together.

May 15
Not quite. Each token gets its own attention output. Token at position 2 attends to tokens 1 and 2, computes weighted V's from those, sums them. Token at position 3 attends to tokens 1, 2, 3, sums those V's. And so on. So you end up with 1024 output vectors from attention — one per position in the sequence. Does that make sense?

Yes.

May 15
Okay good. So each of those 1024 attention output vectors then goes through the residual connection — it gets added to the original embedding vector for that position.

What's the residual connection?

May 15
Good question. A residual connection is a shortcut. Instead of replacing the embedding with the attention output, you add them together: output equals attention_output plus original_embedding. The idea is: if attention doesn't help, at least you still have the original information. It also helps gradients flow backward during training — information can skip layers. It's a safety net that prevents the model from losing information even if a layer isn't useful.

So in in another way, like, we keep the embedding information. That... that's the baseline. So if we add them, we can also say we take the embedding and we add the... have or the output of the attention layer is of... is thousand twenty four sums. Right?

May 15
Exactly. You take the original embedding for each token, and you add the attention output for that token. So for position one, you add embedding_1 plus attention_output_1. For position two, embedding_2 plus attention_output_2. And so on

Wait. Wait. But... oh, shit. So the attention... oh, you... It's not... yeah. Of course. We're we're just... we we added... the sums are vectors. Right?

May 15
Right, exactly. We're adding vectors to vectors. Attention_output_1 is a 256-dimensional vector. Embedding_1 is also a 256-dimensional vector. You add them element-wise: first dimension plus first dimension, second plus second, etc. The result is a new 256-dimensional vector. You do this for all 1024 positions. So you still have 1024 vectors of 256 dimensions coming out of that residual

So... okay. I'm I'm thinking about this, like, graphically. So what... I have my embeddings of the whole sequence, and I'm actually adding the attention some vectors for every single position of those embeddings to the embeddings. And these, um, these sums are initially basically random because, um, the the WV, um, matrix is also random. And so, basically, initially, I'm just adding random vectors to the to the, uh, to the embeddings. Um, but over time, these matrices are optimized for certain things, um, and this could probably be expressed of likelihood. No. Not likelihood because it's v. The likelihood is more the q q q k, and the v is basically the transformed meaning describing the relevance.

May 15
You're thinking about this really well. Yes, initially you're adding random noise basically. But over time, backprop adjusts W_Q, W_K, W_V so that the attention outputs become meaningful — they highlight relevant context. V specifically learns to extract and transform information from attended tokens that's useful for prediction. Q and K learn to create meaningful matches. So the random noise gradually becomes signal. And you're right that this is all optimized indirectly through the next-token prediction loss. If those additions help predict better, they get reinforced. If they hurt, they get adjusted.

Okay. So the outcome after residual is basically more driven by v, the sums, so the relationship values to all previous... to the previous to the previous tokens. So the first token in the sequence is probably just residual. Right? It's just the embedding. And then the second token is, um, some v. So the output of the of the residual is not the meaning of each word in the sequence anymore. It's the first one, yes, but all the other ones are Transformed a lot. Right? And what happens with that afterwards? That that's not the meaning of of words anymore in the sequence. It's more the... I don't know how to describe it.

May 15
Exactly. You've got it. After the first token, each vector is no longer just "the meaning of this word." It's the embedding plus information gathered from all previous tokens through attention. It's more like "the meaning of this word in context." The first token is just itself because there's nothing before it. The second token is itself plus what it learned from the first. The third is itself plus aggregated context from tokens 1 and 2. And so on. So by the time you reach the last token, it's heavily influenced by everything that came before. That transformed vector then goes into the feed-forward network.

Okay. I think let's let's continue with the feed forward. So what happens in feed forward?

May 15
Good. So the feed-forward network takes that 256-dimensional vector and does two things. First, it expands it — multiplies by W1 matrix that projects from 256 dimensions up to 1024 dimensions. Then it applies a nonlinearity called GELU, which is a smooth activation function that adds complexity. Then it contracts back down — multiplies by W2 matrix that projects from 1024 back down

Okay. It's... wait. Wait. Sorry. Can you just describe briefly what happens in the feed forward?

May 15
Sure. Feed-forward takes a 256-dimensional vector. Expands it to 1024 with W1.

No. No. I mean, what's the purpose of feed forward?

May 15
Oh, good question. Attention mixes information between tokens — it gathers context. Feed-forward doesn't look at other tokens. It just takes each token's vector independently and transforms it through nonlinear operations. It's like "thinking" or "processing" the context you just gathered. The expansion and nonlinearity let the model learn complex patterns and features that a single linear transformation couldn't capture. It's where the model refines and deepens its understanding of each token based on the context it just gathered from attention.

So the attention, um, data is going into the feed forward, and then the feed forward is basically making sense of that attention.

May 15
Exactly. Attention gathers the relevant context from previous tokens. Feed-forward takes that context-enriched vector and processes it — learns nonlinear transformations of it. So attention says "here's what you should look at," and feed-forward says "here's what that means." They're complementary: attention for horizontal mixing (across positions), feed-forward for vertical deepening (within each position).

And how does this feed forward path look like? I mean, is it... how is it structured?

May 15
So it's pretty simple structurally. You've got a 256-dimensional vector coming in. It gets multiplied by W1, a 256 by 1024 matrix, expanding to 1024 dimensions. Then GELU

What size of it, sir?

May 15
W1 is 256 by 1024. It takes 256 dimensions and projects to 1024. Then W2 is 1024 by 256 — it projects back down from 1024 to 256. So you expand, apply nonlinearity, then contract back to the original dimension.

So why do we expand and contract? So what's the purpose of, um, expanding to ten twenty four? And why ten twenty four?

May 15
Good question. The expansion gives the model more "room to think." With just one matrix going from 256 to 256, it's a pure linear transformation — limited expressivity. By expanding to 1024, then applying GELU nonlinearity, the model can learn more complex, nonlinear patterns. The contraction back to 256 means you still output in the right dimension to flow to the next block.

As for why 1024 specifically — that's just d_ff equals 4 times d_model. In your case, 4 times 256 is 1024. That ratio of 4x is pretty standard in transformer designs. Bigger models use it too. It's empirically chosen — researchers found that 4x works well. You could use 2x or 8x, but 4x seems to be the sweet spot for expressivity versus computational cost.

What's g e l u nonlinearity?

May 15
GELU is a smooth activation function. It's like a gate that lets some information through and blocks some. ReLU is a simpler one — it zeros out negative numbers. GELU is smoother — it gradually tapers down rather than cutting off sharply. It's based on the cumulative distribution function of a Gaussian, but honestly the math is less important than the intuition: it's a nonlinearity that lets the model learn curved, nonlinear patterns instead of just straight lines. Without it, stacking matrices would just be one big linear transformation, which is very limited. GELU lets the model learn complex relationships.

And there are two matrices. One is, um, one is, um, expanding, and the other one is contracting again. And the weights in this matrix... so we have two matrices. Um, these weights are also trained.

May 15
Yes, exactly. W1 and W2 are both trained during backprop. W1 has 256 times 1024 values — that's 262,144 numbers. W2 has 1024 times 256 values — also 262,144 numbers. So the feed-forward layer in one block has about half a million trainable parameters. You have four blocks, so that's roughly 2 million parameters just for feed-forward layers. Plus all the attention weights on top of that.

Another video here from China, Elon Musk is out of his mind. He's just on Kedaby. He's got so much Kedaby going through him. There's really not much else to say about it. Uh, and we have the reporting on it. We have videos similar to this where he just is out of it. How much of her video she has up this way? Really? Ten percent? I'm real. What's the word now? Five point seven trillion US.

May 15
Sorry, I'm not sure what you're referencing there — that seems like it got pasted in from somewhere else. Did you mean to send that, or should we keep going with the transformer architecture discussion?

Yeah. No. Sorry. We keep going with transformer. Uh, I was sidetracked by a video. Sorry. So, yeah, we expand. So we have two... these two matrices. They're also initialized randomly, I guess. And and then, uh, they will be... the weights will be, uh, amended, um, or or trained. So How do how do the vectors from attention, they go through? Like, okay, they go through the... for the whole training sequence, let's say, thousand twenty four tokens in the training sequence. They go through attention, and then they go through the expansion in the... with with w one and then through the contraction again. w two inside of the, uh, feed forward network, um, or feed forward part. And that... that's it. Feed forward is just those two matrices, not not more.

May 15
Yes, that's it for feed-forward. Input vector gets multiplied by W1, GELU is applied, then W2 is applied. One 256-dimensional vector in

Question

May 15
Yes, that's it for one feed-forward layer. Two matrices, W1 and W2, with GELU in between. All 1024 tokens in your sequence go through this independently — each token's vector gets expanded, nonlinearity applied, contracted. Then that output goes through another residual connection, adding

So... sorry. I'm I'm using, um, or I did some research and and some experiments with neural networks, and there, I've used, uh, Sigmund, uh, sig... sigma one. Sigma sigma... no. Sigmund. Sigmund activation functions, which are just very simple activation functions on values, actually, not even on vectors. Um, so the g e l u is just an activation function for for vectors. Is that correct?

May 15
Not quite. GELU is an activation function that operates element-wise on vectors — meaning it applies the same function to each element independently. So if your vector has 1024 elements, GELU applies to each one separately. It's similar to sigmoid or ReLU in that sense — they all work element-wise. GELU just happens to be smoother and empirically works better than sigmoid or ReLU in transformers. So yes, it's applied to vectors, but really it's 1024 independent applications of the same scalar function.

I got you. Okay. And now you you were saying for one layer, there is w one and w two. So there are more layers. Is that a... that is question.

May 15
No, there's just one feed-forward per block. Each of your four transformer blocks has one attention layer and one feed-forward layer. So W1 and W2 appear four times — once in block one, once in block two, etc. But they're different weights each time. Block one has its own W1 and W2, block two has different W1 and W2, and so on.

Okay. So after feed forward, staying in the first block, after feed forward, what happens then?

May 15
After feed-forward, you've got another residual connection — the feed-forward output gets added back to its input. Then you're done with block one. The 1024 vectors of 256 dimensions flow into block two, which does the exact same thing: attention layer plus feed-forward

Okay. So staying with the with the residual, the residual is, um, from the from the output, from the attention layer, not from the embeddings. Right?

May 15
Right, exactly. After attention, you add back the input to the attention layer — which is the output from the previous residual connection, or the embedding if it's the first layer. So the residual skips the attention, not back to the original embedding. Then after feed-forward, the residual skips the feed-forward, adding back to the output of the attention-plus-residual. So each block has two residual connections: one after attention, one after feed-forward.

Okay. And so now I understand there are multiple blocks in, uh, transformer blocks. So let's say we have only one transformer block. Why is there a need to have multiple transformer blocks? What's the effect of having multiple transformer blocks?

May 15
Good question. More blocks means more layers of abstraction and refinement. With one block, each token gets one round of attention-gathering and thinking. With four blocks, it gets four rounds. Each round can build on what came before — refine patterns, build longer-range dependencies. A one-block model can learn basic local patterns. A four-block model can track grammatical structure across longer distances. A 96-block GPT model can build very deep hierarchies of meaning. But there's a tradeoff: more blocks means more computation and more parameters to train. Your four blocks are a middle ground — small enough to train locally, but deep enough to learn non-trivial patterns.

So, um, the input for the second block is the output from the first block. So the attention and transform transform by attention layer, transformed by feed forward layer, and then this sequence output, let's say, the thousand twenty four sequence length, uh, the output that is computed from the previous feed forward is then fed into the second block as if those tokens or those vectors are the embeddings of the first block.

May 15
Exactly right. The output of block one becomes the input to block two. Block two treats those vectors the same way block one did — runs attention on them, then feed-forward. So yes, you can think of them as "new embeddings" for block two. They're still 256-dimensional vectors, one per position. Block two refines them further based on what block one computed. And so on through blocks three and four.

Okay. And, um, what's... so what's the

May 15
No. What's the what's the amount of blocks in DeepSeek v three?

May 15
I'm not sure off the top of my head. Let me search that for you.

Searched the web
Searched the web
DeepSeek V3 has 61 transformer blocks or layers. That's way bigger than your four. And it uses a mixture-of-experts approach where each block chooses which of 256 expert networks to route to, rather than just one feed-forward network. So it's way more complex, but the basic structure — attention plus processing, stacked many times — is the same.

Okay. Yeah. Okay. Let's let's complete the picture. So let's say we have en blocs of of transformer blocks. And then what happens if the sequence went through all those transformations?

May 15
Good. So after all four blocks, you've got 1024 vectors of 256 dimensions each. Those vectors have been refined through multiple rounds of attention and feed-forward processing. They're no longer just embeddings — they're rich contextual representations. Then those vectors go to the output head. The output head takes each vector and projects it from 256 dimensions to 8000 dimensions — one score per token in your vocabulary. Then you apply softmax to turn those 8000 scores into probabilities that sum to one. Those probabilities tell you which token comes next.

How can I imagine the output head?

May 15
The output head is really simple — it's just one more matrix multiplication. You've got a 256 by 8000 matrix called W_output. Each of your 1024 vectors gets multiplied by W_output, producing 8000 values. Those 8000 values are logits — raw scores. Then softmax converts them to probabilities. So logically, the output head is asking: "Given this 256-dimensional representation, what's the probability of each of the 8000 tokens being next?"

And w output is also initialized randomly.

May 15
Yes, exactly. W_output starts as random numbers too. 256 times 8000 equals about 2 million random values. During training, backprop updates those numbers so that the model learns to project the right contextual vectors into high probabilities for the correct next tokens.

So w output is responsible to predict the next token given, uh, all the sequence outputs from the block sequence.

May 15
Exactly. W_output takes the final 256-dimensional representation of each token's context and projects it to 8000 logits — one per token in the vocab. So for position 1024, the output

How can I imagine Lodgett's

May 15
Logits are just raw scores before softmax. Think of them as votes. Token 42 might have a logit of 2.5, token 89 might have 4.8, token 156 might have 0.3. Softmax converts those into probabilities — 4.8 becomes a high probability, 0.3 becomes low. Logits let the model express confidence — higher logit means "I'm more confident this token is next." Lower means "I'm less sure."

So the matrix, the w output matrix, um, basically converts a sequence of vectors that have two hundred fifty six. So let's let's be concrete. My sequence of ten twenty four, uh, of of vectors with two hundred fifty six dimensions, it converts that to one, um, vector with with eight thousand dimensions.

May 15
Not quite. W_output converts each vector independently. So you've got 1024 vectors of 256 dimensions. Each one gets multiplied by W_output separately, producing 1024 vectors of 8000 dimensions. So you end up with 1024 logit vectors, each with 8000 values. Does that make sense?

So w output is is a matrix. Right? And what's the dimensions of that matrix?

May 15
W_output is 256 by 8000. It takes a 256-dimensional input and produces an 8000-dimensional output. When you multiply a 256-dimensional vector by a 256 by 8000 matrix, you get an 8000-dimensional vector. You do this 1024 times — once for each position in your sequence.

So I get, uh, eight thousand dimension vector for every token in my sequence, actually.

May 15
Exactly. You've got 1024 tokens in your sequence, and each one gets an 8000-dimensional vector of logits. So you have 1024 by 8000 — about 8 million logits total. Then softmax normalizes each of those 1024 logit vectors independently into probabilities.

So softmax need to normalize a vector of eight thousand dimensions. Can you explain softmax to me? And, also, explain how this makes up a probability. I guess the eight thousand is because my vocabulary is eight thousand. So I guess every vector in this... every vector position in this vector is representing one of my words in the in the dictionary, in the vocabulary. So how does that, you know, like, link to the correct word?

May 15
Perfect. So softmax takes your 8000 logits and converts them to probabilities. The formula is: for each logit, compute e raised to that logit, then divide by the sum of all e raised to all logits. This guarantees all probabilities sum to one and are between zero and one.

So if logit for token 42 is 4.8 and logit for token 89 is 0.3, token 42's probability will be much higher because e^4.8 is way bigger than e^0.3. After softmax, maybe token 42 gets probability 0.8 and token 89 gets 0.1.

And yes — each position in that 8000-dimensional vector corresponds to one token in your vocabulary. Position zero is token zero, position 1 is token one, etc. So if softmax gives position 42 a probability of 0.8, that means "I predict token 42 — whatever word that is — comes next with 80 percent confidence."

What's the formula of Softmax?

May 15
The formula is: for each logit L_i, softmax(L_i) equals e raised to L_i, divided by the sum of e raised to all logits. Written as: softmax(L_i) = e^L_i / sum(e^L_j for all j). So you exponentiate each logit, then normalize by the sum of all exponentiated logits. This guarantees the output sums to one and all values are between zero and one.

Where is Softmax used in other applications?

May 15
Softmax is used anywhere you need to convert scores into probabilities that sum to one. Classification problems use it a lot — if you're classifying an image into ten categories, you use softmax to turn your ten output scores into ten probabilities. Attention also uses softmax — remember, we compute similarity scores between Q and K, then softmax converts those into attention weights. Multi-class neural networks use it. Anywhere you have competing options and need to pick one with a probability distribution, softmax is useful. It's really a general tool for turning raw scores into valid probabilities.

Okay. So now let's continue, um, with the training flow. So now my my... we we actually computed the softmax, and then we have probabilities for every single word for... sorry, for every single token. No. For... no. We have actually probabilities for all words in the vocabulary. And then what's... what happens then? How do we select a word that it should be used as the next word?

May 15
Good question. During training, you don't select a word. You compare the probabilities to the ground truth. The ground truth is the actual next token in your training data. Let's say the actual next token after your sequence is token 42. You look at the softmax probability for token 42 — maybe it's 0.1. You want it to be 1.0. So you compute a loss: cross-entropy loss, which is negative log of the probability. Negative log of 0.1 is about 2.3. That's your loss — it tells you how wrong the model was. Then backprop updates all the weights to make that probability higher next time. During inference, you'd sample from those probabilities or pick the highest one.

So what's the loss of, uh, a probability of zero dot zero zero zero one?

May 15
Negative log of 0.0001 is about 9.2. So the loss is very high. That means the model was very confident in the wrong answer — it gave the correct token almost zero probability. During backprop, that huge loss creates huge gradients that strongly update the weights to fix this mistake. Whereas if the model gave 0.9 probability to the correct token, the loss is only negative log of 0.9, which is about 0.1 — much smaller.

What's the difference of log and negative log?

May 15
Log is just the natural logarithm. Negative log is the negative of that. So log of 0.9 is about negative 0.1, and negative log of 0.9 is positive 0.1. We use negative log in the loss function because we want loss to be positive and higher when predictions are wrong. Log of probabilities between zero and one is always negative, so we negate it to get positive loss values.

And why does this need to be logarithmic?

May 15
Good question. Logarithm has a nice property: it heavily penalizes confident wrong predictions. If the model predicts 0.9 for the wrong token, log loss is small. If it predicts 0.5, log loss is bigger. If it predicts 0.0001, log loss is huge. Linear loss wouldn't have this property — it would penalize all mistakes roughly equally. With log, the model learns to be very confident in correct answers and very uncertain in wrong ones. Also, log transforms multiplication into addition, which makes the math cleaner for computing gradients across many tokens.

So the loss is always between zero and nine.

May 15
Not exactly. The loss depends on vocab size. With vocab 8000, the worst possible prediction is probability 1/8000, which is about 0.000125. Negative log of that is about 8.9. So the loss ranges from zero (perfect prediction) to about 8.9 (worst possible prediction). Your loss of 6.8 after 50k iterations is pretty good — it's in the middle-to-good range.

Okay. So now we have a loss calculated. And how do we then proceed?

May 15
Now you backpropagate. You compute gradients — how much each weight in the model contributed to that loss. Then you use those gradients to update the weights using an optimizer like AdamW. The optimizer decides how much to move each weight in the direction of the gradient. After updating, you grab the next batch of training data and repeat. That's one training step. After thousands of steps, the loss gets smaller because the model's weights are being adjusted to make better predictions.

How do we com... compute those gradients?

May 15
That's the chain rule from calculus. Each operation in the forward pass — embedding lookup, matrix multiplication, softmax, loss — has a local derivative. Backprop multiplies all those local derivatives together using the chain rule to get the gradient of loss with respect to each weight. So if loss depends on softmax, and softmax depends on W_output, and W_output is a weight, you compute dLoss/dSoftmax times dSoftmax/dW_output to get dLoss/dW_output. You repeat this for every weight in the model, working backwards from loss to embedding. It's automated in frameworks like PyTorch, but that's what's happening under the hood.

And what's what's the Adam w, uh, algorithm?

May 15
AdamW is an optimizer that updates weights based on gradients. Instead of just subtracting the gradient from the weight, it's smarter. It keeps running averages of gradients and squared gradients, which lets it adapt the step size per weight. Frequently-updated weights get smaller steps (already well-trained), rare weights get larger steps (need more movement). It also adds weight decay — a small penalty that shrinks weights toward zero. That prevents weights from growing unbounded. The algorithm is called "adaptive moment estimation with weight decay."

But item w then needs memory.

May 15
Yes, good point. AdamW stores two running averages per weight — the first moment (average gradient) and second moment (average squared gradient). So if you have 50 million parameters in your model, AdamW needs to store two 50-million-value arrays. That's extra memory on top of storing the weights themselves. So AdamW uses roughly three times the memory of just storing weights — the weights plus the two moment arrays. That's why some people use simpler optimizers like SGD when memory is tight, though AdamW usually trains better.

And in back prop... back propagation, we can actually calculate the gradients not just through all the blocks, through all the matrices like w one, w two, the expansion, the contraction matrix, and then the the matrices in the attention, the w q, w k, w v. So we can actually back back propagate and create gradients for all those, um, weights back to the origin, um, that are relatively relatively accurate.

May 15
Exactly. Backprop computes gradients for every single weight — W1, W2, W_Q, W_K, W_V, W_O, the embedding table, all of it. It traces back through all four blocks, through every attention head, every feed-forward layer, all the way to the embeddings. By the time you're done, you have a gradient for each weight telling you which direction to move it. The chain rule chains all those local derivatives together, and because matrices and activations are all differentiable, you get accurate gradients everywhere. That's the power of backpropagation — one backward pass through the entire model.

and Adam w optimizer is responsible of how much is the gradient used to do a correction.

May 15
Exactly. The gradient tells you the direction and magnitude of how wrong you are. AdamW takes that gradient and decides how much to actually use it for the weight update. It might use the full gradient, or scale it down, or scale it up, depending on the history of that weight's gradients. It also applies weight decay. So yes — AdamW is the gatekeeper between gradient and weight update. It's smarter than just blindly subtracting the gradient.

Okay. That's that's amazing. I think I understand it now the first time a little bit.

May 15
That's great! You've got the core flow down now. To recap: tokens get embedded, flow through attention and feed-forward blocks, get projected to logits, softmax turns them to probabilities, you compare to ground truth, compute loss, backprop computes gradients through everything, and AdamW updates all the weights intelligently. That's the full training loop. The magic is that by doing this thousands of times on millions of tokens, the model learns meaningful patterns. Does it feel less magical now, or still pretty wild?

Well, the mechanics, the architecture, and everything doesn't feel magic, but the outcome still does. And I think... but I think that's normal. Right? I mean, even, like, open AI, data scientists, engineers are still puzzled that something so seemingly intelligent can arise from just scale.

May 15
Yeah, that's exactly it. The mechanics are understandable — matrix math, backprop, optimization. But why those mechanics produce language understanding, reasoning, writing quality — that's still genuinely mysterious. You're training on next-token prediction, which is a simple signal. But somehow, at scale, that signal teaches the model grammar, facts, logic, even creativity. OpenAI researchers call it an "emergence" — abilities that weren't explicitly programmed but fell out of scaling. So you're in good company being puzzled by it. The honest answer is: we don't fully understand why yet.

Okay. So let's assume we do training, and then we have a sequence of thousand twenty four. So we train the first batch, um, with thousand twenty four tokens, and we do the attention... the embeddings, attention, and then the four forward feed, uh, feed forward. And We calculate the loss. We do back propagation one time. And now the next sequence comes. Right? So do we... for the first token in the next sequence, do we use anything from the first sequence, or is the second sequence, first token already a primer again, but with already the, uh, the corrected or... yeah. slightly corrected, um, weights.

May 15
Good question. Each sequence is independent during training. The second sequence starts fresh — its first token is just the embedding, with no context. But the weights have been slightly updated from the first sequence's backprop. So the second sequence uses the slightly improved weights. Over thousands of sequences, those tiny weight updates accumulate. The model gets better and better at predicting next tokens. Each batch is independent in terms of context — no hidden state carries over — but the weights accumulate learning across all batches.

And how do we select sequences in the training data? Is this just sequentially? We go from one sequence to the next to the next text.

May 15
It depends on your setup. You could go sequentially through your training data — read the first 1024 tokens, then the next 1024, and so on. Or you could shuffle the data and grab random chunks. Most people shuffle because it adds randomness, which helps the model generalize better — it doesn't overfit to the order of the original data. In practice, you usually break your 600 megabytes of training data into overlapping or non-overlapping windows of 1024 tokens and randomly sample them. Does your implementation shuffle, or do you go sequentially?

I actually don't know. I need to verify that. But, um, I have some more questions. So the sequence length,

May 15
Go ahead, what's your question about sequence length?

It does.

May 15
Sure, go ahead with your questions. What about sequence length?

So it does really matter in terms of, um, complex sentences, long sentences, or paragraphs with a lot of references. Um, then it's actually becoming important that the sequence length is quite long. Right? So, otherwise, the model would never train on very complex long sentences that has a lot of relationships. or paragraphs that are very long, which have a lot of interrelationships.

12:00 AM
Exactly right. With a max sequence length of 1024, your model can only see 1024 tokens of context at once. If a paragraph is longer than that, it gets cut off. The model never learns relationships that span beyond 1024 tokens. So it can't handle complex long-range dependencies — like a pronoun referring to something mentioned 2000 tokens ago. Bigger models like GPT-4 have much longer context windows — tens of thousands of tokens — so they can track relationships across entire documents. Your 1024 is decent for short stories from Cosmopedia, but for really complex reasoning or long documents, you'd need longer. That's one reason why longer context windows matter so much in practice.

Is that also what you read when a model card in... on Hogging Face, for example, says the model was natively trained with one million context window, for example, then this was actually the sequence length during training?

12:01 AM
Exactly. If a model card says "trained on 1 million context window," that means during training, sequences were 1 million tokens long. So the model learned relationships across a million-token span. That's massive — it can handle very long documents and complex cross-references. Your model with 1024 was trained on 1024-token sequences. So the "native context window" in the model card tells you the max sequence length used during training.

So if I have a model that was trained for thousand twenty four, um, tokens in a sequence... so the sequences during training were thousand twenty four. Um, but then I'm running the model with inference, and I actually was in the inference. I used more context. I use, uh, let's say, four thousand ninety six tokens. That means that the model starts to actually forget mid in... mid discussion. Right?

12:03 AM
Not exactly forget, but it can break down. During training, the model never saw sequences longer than 1024, so it has no learned patterns for how to handle 4096. The attention mechanism was trained on 1024-token relationships. Beyond that, performance degrades. Some models can extrapolate a bit — they might work okay up to 1.5x or 2x their training length — but it's unreliable. You might see the model losing track of context halfway through, repeating itself, or making inconsistent statements. That's why context window matters: if you want your model to handle long documents at inference, you need to train it on long sequences.

Okay. So now we have trained our model. So let's move to inference.

12:05 AM
Good. So you've trained your model — all the weights are now tuned from backprop. Now at inference time, you take a fresh prompt, tokenize it, and run it through the model to generate text. What would you like to know about inference?

So let's maybe go through an example. Let's say, um, I start inference by just... let's think of, like, story completion. So very simple. Like, uh, story completion, k... use case. So I basically just... the only prompt that the model gets during inference is, like, Once upon a time. Right. So let's take this example and go through all the steps piece by piece. Let's make a break between the steps.

12:06 AM
Perfect. So prompt is "Once upon a time." Let's go step by step.

Step 1: Tokenize the prompt

You tokenize "Once upon a time" using your BPE tokenizer. Let's say it becomes tokens: 123, 456, 789. Three tokens total.

What's next?

So I would assume these tokens or that sequence of tokens is then going through the embedding layer, and there, um, we actually get embeddings for those tokens.

12:06 AM
Exactly. The tokenizer gives you token IDs 123, 456, 789. The embedding layer looks them up in your embedding table. Token 123 becomes a 256-dimensional vector, token 456 becomes another 256-dimensional vector, token 789 becomes another. So you've got three vectors of 256 dimensions each. What happens next?

I would assume we go... we take the whole sequence and go through the attention layer. And then in the attention layer, we do all the attention computations. And in the end, we're gonna end up with the v sums.

12:08 AM
Right. So token 789 (the last one, "time") attends to tokens 123 and 456 (the previous ones, "Once" and "upon"). It computes Q for token 789, K and V for tokens 123 and 456. The attention scores tell you which previous tokens matter. Then you weight and sum the V's. So the output for token 789 is enriched with context from "Once upon." Token 456 attends to token 123. Token 123 attends to itself only. Then those attention outputs go through the residual and feed-forward. All three tokens flow through all four blocks. At the end, you've got three 256-dimensional vectors — one per input token.

Alright. And then it goes through the v... w output... sorry, w output. And there, we actually, um, we have probabilities for one of our tokens in the eight thousand, um, vocabulary. So we prob... we then have probabilities, um, and we can say which word has the highest probability. But then, I guess, we do not just choose the one with the highest probability. We probably have some noise function, random function, and then we wait, um, the output token possibilities from the vocabulary, and we choose the one where our weighted randomness lands.

12:10 AM
Exactly. You could pick the highest probability token — that's called greedy decoding. But it tends to be boring and repetitive. Instead, you sample from the probability distribution. You spin a weighted die where each token's face is weighted by its probability. Higher probability tokens are more likely to be picked, but lower ones still have a chance. This adds creativity and variety to generation. Different sampling strategies exist — temperature scaling makes the distribution sharper or flatter, top-k restricts to the top k tokens. But yes, weighted randomness is the idea.

So temperature is actually used, um, to scale the probabilities. Um, so that means if I have a temperature of zero, then I get exactly the probabilities. And if I have a temperature of one or two, then those, uh, probabilities are divided and rescaled or something like that so that the higher the temperature, the less the less influence from the probabilities. Is that true?

12:11 AM
Close, but not quite. Temperature scales the logits before softmax, not after. So with temperature zero, you're picking greedily — the highest probability. With temperature one, you use the probabilities as-is. With temperature greater than one, you divide the logits by the temperature before softmax. This flattens the distribution — makes it more uniform, so low-probability tokens become more likely. So higher temperature means more randomness, less confident in the model's predictions. Lower temperature means more confident, more deterministic.

So temperature of zero means automatic like a greedy selection. So the... we just pick the one that has the highest probability. end with temperature one. We have a normal random weighted probability situation. And with higher temperatures, we flatten the probabilities.

12:12 AM
Exactly. Temperature zero is greedy — pick the max. Temperature one is the raw probabilities from softmax. Higher temperatures flatten it out, making the distribution more uniform. So with temperature 2, even a token with 1 percent probability becomes much more likely. It adds creativity and unpredictability.

Okay. So then we have our next word. Then what I would expect is that this is just a loop. Right? So then, basically, we go back to the start, but with the new generated word at the end of the sequence, and we just run through it all over.

12:14 AM
Exactly. You append the sampled token to your sequence. Now you've got four tokens: 123, 456, 789, and the new one. You feed those four through the model again — embedding, attention, feed-forward through all four blocks, W_output, softmax, sample again. You keep looping until you hit a stopping condition — maybe you generate 100 tokens total, or you sample the special end-of-sequence token.

Like in my implementation, for example, I think I don't have an end of sequence token because the model is just keeps synthesizing until it runs out of context tokens.

12:15 AM
Right, so your model just generates until it hits the max sequence length of 1024. Once you've generated enough tokens that you're at 1024 total (prompt plus generated), you stop. That's a simple stopping condition. More sophisticated models have an explicit end-of-sequence token that signals "I'm done," so the model can choose to stop whenever it wants. But running until max length works too.

How do you train the model for a end of sequence token?

12:15 AM
Good question. You include it in your training data. When you're preparing training sequences, you append a special end-of-sequence token at the end of each text passage. Then the model trains on predicting that token naturally — it learns that after certain patterns, the next token should be end-of-sequence. Over time, it learns to predict end-of-sequence when it's actually done generating. You can also give it a low probability during inference — only sample it if the model is very confident it's done.

And how do you insert end of sequence tokens in training data? I mean, is it after a conversation or... yeah. How do you do that?

12:16 AM
It depends on your data format. If your training data is a collection of separate documents or stories, you add the end-of-sequence token after each one before feeding it to the model. So your training sequence might look like: "Once upon a time there was a cat [EOS] In another land [EOS] ..." where [EOS] is the special end token. This teaches the model "when you see the end of a story, predict [EOS]." If your data is one long corpus, you might add [EOS] at paragraph boundaries or sentence boundaries, depending on the granularity you want. The key is that the token appears in training data at natural stopping points, so the model learns to predict it there.

Okay. I think I have a pretty clear picture now. So I have some follow-up questions now. For example, how do special things... special architectures that evolve now come into play? For example, I heard that some models use very special attention, for example. Uh, I think, for example, There is... what's the name? Let me wait. Let me think. Let me think. Um, was it GLM, or was it was it a different open weight model? I can't remember, but let's take another example. Flash attention, I guess, is a property of inference. What's flash attention?

12:18 AM
Flash attention is an inference optimization. Standard attention computes all pairwise similarities between Q and K — that's quadratic in sequence length. With 1024 tokens, that's a million comparisons. Flash attention reorganizes the computation to be more cache-efficient on GPUs. Instead of loading everything into memory at once, it processes attention in tiles — small chunks that fit in fast cache. This doesn't change the math, just how it's computed. It's faster and uses less memory. It's still the same attention mechanism, just implemented cleverly to run faster on hardware.

Sorry. What what do you... can can you elaborate in pairwise similarities between q and k?

12:19 AM
Sure. In attention, you compute a similarity score between every Q and every K. So if you have 1024 tokens, you compute 1024 by 1024 similarity scores — that's over a million dot products. Each position's Q gets compared to every other position's K. That's what "pairwise" means — every pair. With a longer sequence, this becomes expensive. Flash attention doesn't reduce the number of comparisons, it just makes them faster through better memory access patterns on the GPU.

Wait. But is that true? I mean, we don't compare every q to every k. Right? It depends on the position in the sequence. We never compare forward. We only compare backward.

12:19 AM
You're absolutely right. Good catch. Because of the causal mask, token at position 1024 only compares its Q to K's from positions 1 through 1024 — not future positions. So it's not a full 1024 by 1024 matrix. It's a triangular matrix — token 1 compares to itself, token 2 compares to tokens 1 and 2, token 3 to tokens 1, 2, 3, and so on. So the total number of comparisons is 1 plus 2 plus 3 plus... plus 1024, which is roughly half a million instead of a million. But the point stands — it's still quadratic in sequence length, and flash attention optimizes it.

So can you explain to me how KV Cash works?

12:20 AM
Good question. During inference, you keep generating tokens one at a time. Without KV cache, each new token causes you to recompute K and V for all previous tokens again. That's wasteful — the K and V for token 1 never change, token 2's K and V never change, etc. KV cache says: store the K and V vectors from previous tokens. When token 1024 comes in, you only compute its Q, K, V. Then you attend to all previous tokens using their cached K and V. You don't recompute them. This drops the computation from quadratic to linear per new token during generation. So inference speeds up dramatically.

That's so confusing for an IT guy because key value cash is something that we do anyways. Like, for example, in Cloudflare, you have key value cash. You use just a key value store as a cash. But here, key v... key, uh, key v cash means actually key v vectors, and the cash is just to cash the previous sequence Very confusing. Right?

12:22 AM
Haha, yeah, that naming is confusing. You're right — in systems engineering, KV cache usually means a key-value store. But in transformer land, "KV cache" literally means caching the K and V vectors from previous tokens. It's the same concept — store past computations so you don't recompute them — but applied to attention vectors instead of a general data store. The name is unfortunate because it overloads "cache," but yes, it's just storing the K and V tensors from earlier positions so you can reuse them.

Oh, wait. So vectors with many dimension... dimensions are called tensors?

12:23 AM
Yes, kind of. A tensor is a generalization of vectors and matrices. A vector is a 1D tensor, a matrix is a 2D tensor. A 3D tensor would be like a cube of numbers. In deep learning, we talk about tensors loosely — your 256-dimensional embedding vector is technically a 1D tensor, your 256 by 1024 matrix is a 2D tensor, your batch of 32 sequences of 1024 tokens of 256 dimensions is a 4D tensor. It's the general term for any n-dimensional array of numbers.

So when some inference run times allow you to quantize k v cash, that means that just the cash is quantized not at the native, um, model weights, but at a lower precision. And then when the model requests it, it will be transformed again. to the native precision.

12:24 AM
Exactly. Quantizing the KV cache means storing K and V vectors in lower precision — maybe 8-bit integers instead of 32-bit floats. This saves memory because you're caching thousands of vectors per sequence. When the model needs to use them in attention, you dequantize back to full precision. This trades memory for a tiny bit of compute — you save cache memory but spend cycles dequantizing. It's a practical tradeoff for long sequences where KV cache memory becomes a bottleneck.

And how does Paged Attention work from some inference, uh, run times like SG Lang or VLLM?

12:25 AM
Paged attention is another memory optimization. Normally, you allocate a fixed chunk of memory for the KV cache upfront — enough for your max sequence length. But many

in in this learning text.

12:25 AM
Paged attention is another inference optimization. The problem is: when you're serving multiple requests, each one has its own KV cache growing as it generates tokens. Memory fragmentation becomes a problem — you allocate space for cache, but not all of it gets used efficiently. Paged attention treats the KV cache like virtual memory — it pages the cache to and from GPU memory, similar to how operating systems page RAM to disk. This lets you fit more concurrent requests on a GPU because you're not preallocating huge buffers. You only keep the active parts in fast memory. It's a systems-level optimization for serving many requests simultaneously, not a change to the model itself.

But it's not about sharing the KV cash with with many requests. It's more about offloading some of the unused cash.

12:26 AM
Good point. Yeah, paged attention pages out parts of the KV cache that aren't actively being used — maybe to slower GPU memory or even CPU memory — and pages them back in when needed. So you can keep more sequences alive on the GPU without running out of memory. It's dynamic allocation based on what's actually needed right now, rather than preallocating huge buffers for each sequence. So yes, more efficient use of memory by offloading unused cache temporarily.

So assuming one head of the attention layer of a very large language model is somehow optimizing itself for punctuation, uh, in text, why are so many language models using, uh, long dash e... EM dash in their outputs. I was always wondering because when I read textbooks or regular books, novels, scientific papers, whatever, I don't see that many e m dashes.

12:28 AM
That's a really interesting observation. I'm honestly not 100 percent sure why that is. It could be a few things: maybe the training data has more em-dashes than real books — like certain internet sources or forums use them more. Or maybe attention heads that specialize in punctuation learn that em-dashes are "safe" separators that work in many contexts, so they overuse them. Or it could be an artifact of how the model generates — it learns statistical patterns from training data that don't perfectly match natural writing. But I'm speculating. Do you want me to search for research on this?

Yeah.

12:28 AM

Searched the web
Searched the web
Interesting. So the research suggests a few things: Em dashes appear heavily in high-quality training data — literary classics, well-written blog posts, technical documentation, publications like The New Yorker. The model learned that em dashes signal "quality writing." So during generation, especially at lower temperatures where it's playing it safe, it defaults to em dashes as a "safe" punctuation choice. They're grammatically hard to get wrong. Also, em dashes require more cognitive load to parse, but the model doesn't feel that friction — it just learned the statistical pattern that em dashes appear in high-quality text. So it overuses them. It's not really about one attention head specializing — it's more a global bias in the training data that gets amplified during generation.

Wright

12:30 AM
Yeah, it's a neat example of how LLMs can pick up on statistical patterns in training data that don't actually match real human writing. The model learned "em dashes = quality," but it doesn't understand the cognitive load they

No. Sorry. I was asking something completely different that got kind of lost. Um, let me let me phrase again. So what I still... what I'm still so puzzled about is that these models, although they are only trained on text and next word prediction, they are so good in following instructions, and I can actually make them correct things. I can ask them to exchange a word in a long text with another word even if it doesn't make sense textually, and they never learned that. Or I can ask them to replace all m dashes and remove them and don't use them. And then the model will follow those instructions even though not a single text actually instructed this in the text and and then, uh, set, um, and then followed with a text that didn't contain that. Right? So it did not directly learn this.

12:32 AM
That's the real mystery right there, honestly. The model was only trained on next-token prediction — never explicitly trained on instruction-following or text manipulation. Yet somehow it can parse your instruction, understand what you want, and execute it. How?

Best theory right now: the model develops internal representations during training that capture high-level concepts — "remove punctuation," "replace word X with Y," "follow instructions." It learns these implicitly from patterns in the training data. For example, if the training data contains examples where someone says "remove the em dashes" and then shows text without em dashes, the model sees that pattern and learns the association. Or it learns general reasoning skills that let it apply instructions to new situations it's never seen.

But honestly? We don't really know. It's called "in-context learning" or "instruction following," and researchers are still trying to understand the mechanism. The model definitely learned something beyond pure next-token prediction, but what exactly and how it generalizes to novel instructions — that's still an open question. It's one of those emergent capabilities that arises from scale that nobody fully predicted or understands.

But still, um, an LLM learns so differently than us humans. I mean, an LLM has probably read a trillion times more texts, when not more, than even the smartest, well read human on Earth. but still a human is able to actually develop cognitive capabilities without reading all those books. And an LLM is purely depending on all that text on human communication to actually develop all those skills. So it's still kind of mimicking what we are doing by reading what we have communicated And I'm wondering... and probably not only me. I think that's the most puzzling thing about that whole thing, but it's so surreal that this happens in such a high quality and that pure scale of the training data makes the models more more intelligent, or they seem intelligent while they're actually just predicting next words. Right.

12:35 AM
Yeah, you've hit on the deepest question. We don't really know if the model is intelligent or just an extremely sophisticated pattern matcher. It reads a trillion tokens, learns statistical relationships, and can predict what comes next so well that it looks intelligent. But is that intelligence, or is it memorization at scale? Humans learn differently — we have embodied experience, we can reason abstractly, we generalize from far fewer examples. A child learns language from thousands of hours of exposure, not trillions. Yet we develop understanding in a way that feels deeper than next-token prediction.

The honest answer is: we don't know if there's a fundamental difference or if intelligence is just very good pattern matching at scale. Maybe intelligence emerges naturally from enough data and compute. Maybe the human brain is also doing something like next-token prediction, just in a different substrate. Or maybe LLMs are missing something essential about understanding. That's genuinely an open philosophical question, not just a technical one. And I think you're right to find it surreal and puzzling — because it kind of is.
