# Module 02 — Embeddings and the residual stream

Module 01 turned text into integers. This module turns integers into vectors and introduces the
single most useful mental model for how a transformer works: **the residual stream as a shared
workspace that every layer reads from and writes to.**

---

## Terms

| Term | Meaning |
|------|---------|
| **Embedding matrix** | `E: [vocab_size, d_model]`. One learned vector per token id. |
| **`d_model` / hidden size** | The width of the residual stream. Qwen3-1.7B: 2048. |
| **Token embedding** | `E[token_id]` — the vector for one token. |
| **Residual stream** | The running `[seq_len, d_model]` state carried through all layers. |
| **Residual connection** | `x = x + f(x)`. Every sublayer *adds* to the stream. |
| **Read / write** | A sublayer reads a subspace of the stream and writes to another. |
| **Unembedding / LM head** | `[d_model, vocab_size]`. Projects the final state back to logits. |
| **Weight tying** | Sharing the embedding and unembedding matrices. |
| **Layer norm / RMSNorm** | Normalisation applied before each sublayer. |
| **Pre-norm / post-norm** | Norm before the sublayer / after it. Modern models are pre-norm. |
| **Cosine similarity** | Directional similarity between two vectors. |
| **Anisotropy** | Embeddings clustering in a narrow cone rather than filling the space. |
| **Superposition** | More features represented than dimensions, using near-orthogonal directions. |
| **Logit lens** | Applying the unembedding to intermediate layers to see the prediction form. |
| **Contextual embedding** | The representation at a position after some layers — depends on context. |
| **Static embedding** | The pre-layer `E[token_id]` — same regardless of context. |
| **Sentence embedding** | A whole-text vector, from a model trained for retrieval (Module 21). |

---

## Concepts

### The residual stream

A transformer forward pass is not a chain of transformations. It is a **running sum**:

```
x_0 = E[tokens] + positional information
x_1 = x_0 + attention_1(norm(x_0))
x_2 = x_1 + mlp_1(norm(x_1))
x_3 = x_2 + attention_2(norm(x_2))
...
logits = unembed(norm(x_final))
```

Every sublayer adds a delta. Nothing overwrites. The consequences are worth stating explicitly
because they explain a lot of otherwise-mysterious behaviour:

1. **Gradients flow directly from the loss to every layer** through the `+`. This is why 100-layer
   transformers train at all — the residual path is a gradient highway.
2. **Layers communicate through a shared bus.** Layer 3 can write information that layer 20 reads,
   with the intermediate layers simply not touching that subspace.
3. **The stream has limited width.** `d_model` dimensions must carry everything. Different features
   occupy different (approximately orthogonal) directions — this is **superposition**, and it is why
   `d_model` scales with model size.
4. **You can decode the stream at any depth.** Apply the unembedding to `x_7` and you see what the
   model would predict if it stopped there. That is the logit lens (Module 24), and it works
   *because* the stream is in a consistent representational space throughout.

Point 4 is the one that makes the mental model earn its keep: it means "the residual stream" is not
a metaphor, it is something you can inspect.

### Embeddings are learned, not designed

`E` is initialised randomly and learned by gradient descent. Nothing forces structure onto it — the
structure that appears (similar tokens near each other, some directions carrying interpretable
meaning) is emergent, driven entirely by the prediction objective.

Two properties that surprise people:

- **Embeddings are anisotropic.** They do not fill the space uniformly; they cluster in a narrow
  cone. Raw cosine similarity between two token embeddings is almost always high (0.3–0.9), so
  absolute similarity numbers are close to meaningless. **Only relative comparisons are useful.**
- **Frequency dominates geometry.** Rare tokens end up further from the centroid than common ones,
  simply because they receive fewer gradient updates.

### Weight tying

Many models share `E` between the input embedding and the LM head (`unembed = E.T`).

| | Tied | Untied |
|---|------|--------|
| Parameters | Saves `vocab × d_model` — 300M+ for a large vocab | More capacity |
| Small models | Usually tied; it's a big fraction of parameters | — |
| Large models | Often untied | Preferred at scale |

For Qwen3-0.6B, a 151k vocab × 1024 dims is 155M parameters — a *quarter* of the whole model in one
matrix. That is why small models tie, and it is a good concrete answer to "why does vocabulary size
matter" beyond tokenization.

### Normalisation: RMSNorm and pre-norm

**Pre-norm** (`x + f(norm(x))`) rather than post-norm (`norm(x + f(x))`) is now universal, because
pre-norm keeps the residual path clean — the identity path from input to output has no
normalisation on it, so gradients pass through unattenuated. Post-norm transformers need careful
warmup to train at all; pre-norm ones are far more forgiving.

**RMSNorm** drops the mean-centering and the bias from LayerNorm:

```
LayerNorm: (x - mean) / std * g + b
RMSNorm:   x / sqrt(mean(x^2) + eps) * g
```

Fewer operations, no mean reduction, essentially the same quality. Every modern model uses it.

### Static versus contextual

`E["bank"]` is one fixed vector. After a few layers, the representation at the position of "bank"
in *"river bank"* and in *"bank account"* has diverged — attention has mixed in context.

This is worth being precise about because it is a common interview trip-up: **the embedding layer
is not where meaning lives.** It is the starting point. Meaning is built up the stack, and a token's
representation at layer 20 is a function of the entire preceding context.

It is also why the embeddings from a causal LM make poor *sentence* embeddings — that is a
different objective and a different model (Module 21).

---

## Where it's used

- **Every forward pass**, as the substrate.
- **Interpretability** — the logit lens, probing and activation patching all operate on the stream.
- **Tokenizer extension** — adding tokens means resizing `E`, and new rows start random.
- **LoRA target selection** (Module 12) — you adapt the matrices that read from and write to the
  stream.
- **Retrieval** (Module 21) — where sentence embeddings are the right tool and LM embeddings are
  not.

---

## Labs

### Lab 1 — Anatomy of the embedding matrix

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", torch_dtype=torch.float32)

E = model.get_input_embeddings().weight
print(f"embedding matrix: {tuple(E.shape)}  =  {E.numel():,} params")
print(f"total model params: {sum(p.numel() for p in model.parameters()):,}")
print(f"embeddings are {E.numel()/sum(p.numel() for p in model.parameters()):.1%} of the model")

head = model.get_output_embeddings().weight
print(f"weight tying: {torch.equal(E, head)}")
```

**Write down that percentage.** For a small model it is startling, and it explains the design
pressure toward smaller vocabularies and tied weights.

### Lab 2 — Anisotropy, measured

```python
import torch

idx = torch.randint(0, E.shape[0], (2000,))
V = torch.nn.functional.normalize(E[idx].float(), dim=-1)
sims = (V @ V.T)
mask = ~torch.eye(len(V), dtype=bool)

print(f"mean pairwise cosine of RANDOM token pairs: {sims[mask].mean():.3f}")

R = torch.nn.functional.normalize(torch.randn_like(V), dim=-1)
rs = (R @ R.T)
print(f"mean pairwise cosine of RANDOM VECTORS:     {rs[~torch.eye(len(R), dtype=bool)].mean():.3f}")
```

**The finding:** random vectors in high dimensions are nearly orthogonal (cosine ≈ 0), but real
token embeddings have a substantial positive mean cosine. They live in a cone. **Therefore an
absolute cosine of 0.6 tells you nothing** — you must compare against the baseline. This is the
lab that stops you misreading similarity numbers for the rest of your career.

### Lab 3 — Nearest neighbours (relative similarity works)

```python
def neighbours(word, k=8):
    ids = tok(word, add_special_tokens=False)["input_ids"]
    v = torch.nn.functional.normalize(E[ids[0]].float(), dim=-1)
    allv = torch.nn.functional.normalize(E.float(), dim=-1)
    sims = allv @ v
    top = torch.topk(sims, k + 1)
    return [(tok.decode([i]), round(s.item(), 3))
            for i, s in zip(top.indices[1:], top.values[1:])]

for w in [" king", " Paris", " python", " SELECT", " drift", " 2024"]:
    print(f"{w!r:>10} -> {neighbours(w)}")
```

Note what the neighbours actually are: often morphological variants and casing/whitespace variants,
not semantic relatives. **Static embeddings encode surface form more than meaning** — and that is
the honest finding, not the tidy king/queen story.

### Lab 4 — Watch the residual stream grow

```python
ids = tok("The capital of France is", return_tensors="pt")
out = model(**ids, output_hidden_states=True)

print(f"{'layer':>6}{'norm':>10}{'Δ from prev':>14}{'cos with prev':>15}")
prev = None
for i, h in enumerate(out.hidden_states):
    v = h[0, -1].float()
    d = (v - prev).norm().item() if prev is not None else 0.0
    c = torch.nn.functional.cosine_similarity(v, prev, dim=0).item() if prev is not None else 1.0
    print(f"{i:>6}{v.norm().item():>10.2f}{d:>14.2f}{c:>15.3f}")
    prev = v
```

**Two observations:** the norm grows monotonically (every layer adds), and consecutive states have
very high cosine similarity (each layer makes a small directional adjustment to a large existing
vector). That is what "a running sum with small deltas" looks like numerically.

### Lab 5 — Logit lens: the answer forming

```python
norm = model.model.norm
W_U = model.get_output_embeddings().weight

for i, h in enumerate(out.hidden_states):
    logits = norm(h[0, -1]) @ W_U.T
    p = torch.softmax(logits.float(), -1)
    top = torch.topk(p, 3)
    print(f"layer {i:>2}: " + "  ".join(
        f"{tok.decode(t).strip()!r}:{v:.3f}" for t, v in zip(top.indices, top.values)))
```

**Watch " Paris" appear.** Typically the early layers predict generic continuations, the answer
emerges somewhere in the middle, and the last layers sharpen the distribution. Run it on a
question the model gets *wrong* too — sometimes the wrong answer is locked in early, which tells
you it is a representational problem rather than a decoding accident (Module 24).

### Lab 6 — Static versus contextual

```python
import torch.nn.functional as F

def rep_at(text, target, layer):
    enc = tok(text, return_tensors="pt")
    ids = enc["input_ids"][0].tolist()
    t_id = tok(target, add_special_tokens=False)["input_ids"][0]
    pos = ids.index(t_id)
    hs = model(**enc, output_hidden_states=True).hidden_states
    return hs[layer][0, pos].float()

a = "I sat on the river bank and watched the water."
b = "I deposited the cheque at the bank this morning."

for layer in [0, 4, 8, 12, -1]:
    va, vb = rep_at(a, " bank", layer), rep_at(b, " bank", layer)
    print(f"layer {layer:>3}: cosine('bank' in the two sentences) = "
          f"{F.cosine_similarity(va, vb, dim=0).item():.3f}")
```

**Layer 0 is exactly 1.000** — the same token id, the same embedding row. It falls with depth as
attention mixes in context. **This is the cleanest possible demonstration that the embedding layer
is not where meaning lives**, and it is a 15-line experiment.

### Lab 7 — Resize the embedding matrix safely

```python
new_tokens = ["<|schema|>", "<|sql|>", "<|obs|>"]
n_added = tok.add_tokens(new_tokens)
model.resize_token_embeddings(len(tok))

E = model.get_input_embeddings().weight
mean_emb = E[:-n_added].mean(dim=0)
with torch.no_grad():
    for i in range(n_added):
        E[-(i + 1)] = mean_emb + torch.randn_like(mean_emb) * 0.01
```

**Why this matters:** newly added rows are initialised randomly by default, which places them far
outside the learned distribution and produces garbage until they get enough gradient. Initialising
from the mean embedding (or from the average of the tokens the new one replaces) starts them
somewhere sensible. If you add special tokens for an agent protocol, do this.

---

## Failure modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| Cosine similarities all look high | Anisotropy | Compare relatively, never absolutely |
| Nearest neighbours are morphological, not semantic | Static embeddings encode surface form | Use contextual reps, or a retrieval model |
| Added tokens produce garbage | Random init far from the learned distribution | Init from mean or from constituent tokens |
| Shape mismatch loading a checkpoint | Tokenizer extended, embeddings not resized (or vice versa) | Resize and save both together |
| Tiny model, most params in embeddings | Large vocab, untied weights | Tie weights; consider a smaller vocab |
| Deep model won't train | Post-norm | Pre-norm + RMSNorm |
| LM embeddings are poor at retrieval | Wrong objective | Use a sentence-embedding model (Module 21) |
| Logit lens output is nonsense | Forgot the final norm before unembedding | Apply `model.model.norm` first |

---

## Interview

**"Explain the residual stream."**
A transformer's forward pass is a running sum, not a chain — every sublayer reads a normalised copy
of the stream and *adds* a delta back, nothing overwrites. Three things follow. Gradients flow
straight from the loss to every layer through the additions, which is why very deep transformers
train at all. Layers communicate through a shared bus, so layer 3 can write something layer 20
reads with everything between ignoring it. And because the stream stays in a consistent
representational space, you can apply the unembedding at any depth and see what the model would
have predicted there — that's the logit lens, and it's a genuinely useful debugging tool rather
than just a nice picture.

**"Is the embedding layer where meaning lives?"**
No — it's the starting point. `E[token_id]` is the same vector regardless of context, so "bank" in
"river bank" and "bank account" start identical, cosine exactly 1.0. Attention mixes context in as
you go up, and by the middle layers those two representations have clearly diverged. Meaning is
built up the stack. It's also why embeddings from a causal LM make poor sentence embeddings —
that's a different objective and needs a model trained for it.

**"Why are cosine similarities between token embeddings always high?"**
Anisotropy — learned embeddings occupy a narrow cone rather than filling the space, so the mean
pairwise cosine between random tokens is substantially positive where random high-dimensional
vectors would be near zero. So an absolute similarity of 0.6 carries no information; only
comparisons against the baseline, or ranked nearest neighbours, are meaningful. It's an easy way to
fool yourself when eyeballing embedding numbers.

**"Why does vocabulary size matter beyond tokenization?"**
Parameters. The embedding matrix is `vocab × d_model`, and for a small model that's a big fraction
of everything — a 151k vocab at 1024 dimensions is 155M parameters, roughly a quarter of a 0.6B
model in one matrix. That's why small models tie the input embedding and the LM head, and why a
large multilingual vocabulary is a real architectural cost, not just a tokenization detail.

---

## Checkpoint

1. Explain the residual stream as a running sum and give the three consequences.
2. Report what fraction of your model the embeddings are, and whether weights are tied.
3. Report the anisotropy baseline and explain why absolute cosine is meaningless.
4. Show the norm growing and consecutive-layer cosine staying high.
5. Show the answer forming with the logit lens.
6. Show `cos = 1.000` at layer 0 for the same token in two contexts, and its decline with depth.
7. Explain how to add tokens without breaking the model.

---

**Next:** [03 — Attention](03-attention.md) ·
**Back:** [01 — Tokenization](01-tokenization.md) · [Syllabus](../SYLLABUS.md)
