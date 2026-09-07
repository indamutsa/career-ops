# B4 — Attention, from one head to many

Your bigram model sees one token of context. B1 Lab 7 showed that more context means lower loss,
and that n-gram counting cannot buy it — the table needs `V^n` parameters.

Attention is how a fixed parameter budget buys arbitrary context. This lesson derives it from the
problem it solves, builds it in four stages, and ends with the multi-head causal attention that goes
into your model in B5.

**You write** `code/b4_attention.py` · **you record in** `notes/b4-attention.md` ·
**reference companion** [M03 — Attention](../03-attention.md)

```bash
cp notes/_template.md notes/b4-attention.md
touch code/b4_attention.py
```

---

## Terms

| Term | Meaning |
|------|---------|
| **Context** | The tokens a position is allowed to use. |
| **Query (Q)** | What this position is looking for. |
| **Key (K)** | What each position offers. |
| **Value (V)** | What each position contributes if attended to. |
| **Attention score** | `q · k` — how well a query matches a key. |
| **Attention weight** | The score after softmax. Non-negative, sums to 1. |
| **Scaled dot-product attention** | `softmax(QKᵀ/√d)V`. |
| **Scaling factor `1/√d_head`** | Keeps the pre-softmax variance ~1. |
| **Causal mask** | Prevents a position attending to the future. |
| **`-inf` masking** | Set forbidden scores to `-inf` so softmax gives them exactly 0. |
| **Head** | One independent attention computation. |
| **Multi-head attention (MHA)** | Several heads in parallel, concatenated, projected. |
| **`d_head`** | Per-head dimension: `d_model / n_heads`. |
| **Output projection** | The final `W_o` that mixes head outputs. |
| **Attention matrix** | The `(T, T)` weight matrix. What you visualise. |
| **KV cache** | Reusing K and V across generation steps (B7, Module 19). |
| **Flash attention** | Fused kernel that never materialises the `(T, T)` matrix. |

---

## Concepts

### The problem, stated precisely

Position `t` needs to build a representation using information from positions `0…t`. Three obvious
approaches, and why each fails:

| Approach | Failure |
|----------|---------|
| Concatenate all previous vectors | Variable size; parameters would scale with context |
| Average all previous vectors | Fixed size, but every position weighted equally — "not" and "the" contribute the same |
| Learn a weight per position | Weights would be positional, not content-dependent |

**What you actually want is a weighted average where the weights depend on the content.** Position
`t` should look at what it needs, decide which earlier positions are relevant *to it*, and average
those.

That is attention. Everything else is mechanics.

### Q, K, V — the retrieval analogy, and its limit

Each position produces three vectors by linear projection of its own embedding:

```python
q = x @ W_q      # what I'm looking for
k = x @ W_k      # what I offer
v = x @ W_v      # what I'll contribute
```

Score every query against every key with a dot product — large when they point the same way.
Softmax over the scores gives weights that are non-negative and sum to 1. Average the values with
those weights.

```
attention(Q, K, V) = softmax(QKᵀ / √d_head) V
```

The retrieval analogy — query matches keys, returns values — is genuinely useful, but note where it
breaks: this is a *soft* lookup returning a blend of every value, not the single best match. A head
attending to twelve positions at weight 1/12 each is doing something a hard lookup cannot express,
and much of what attention does is exactly that kind of diffuse aggregation.

### Why `√d_head` and not something else

The scaling factor is the one piece of the formula people cannot justify, and the derivation is
short.

Let `q` and `k` have components that are independent with mean 0 and variance 1. Their dot product
is a sum of `d_head` such products:

```
Var(q · k) = Σᵢ Var(qᵢkᵢ) = d_head
```

So the standard deviation grows as `√d_head`. With `d_head = 32`, raw scores have a spread of about
±6; with `d_head = 128`, about ±11.

Feed scores that large into softmax and it saturates — one weight goes to ~1, the rest to ~0. That
is bad twice over: the average becomes a hard selection, and **the gradient through a saturated
softmax is nearly zero, so the head stops learning.**

Dividing by `√d_head` restores unit variance, keeps softmax in its responsive range, and keeps
gradients alive. You will measure exactly this in Lab 2.

### The causal mask

A language model must not see the future. Position 3 predicting token 4 cannot attend to token 4,
or the task is trivial and the model learns nothing useful.

Implemented by setting forbidden scores to `-inf` **before** softmax:

```python
scores = scores.masked_fill(mask == 0, float("-inf"))
weights = F.softmax(scores, dim=-1)          # exp(-inf) = 0, exactly
```

Two details that are easy to get wrong:

- **Mask before softmax, never after.** Zeroing weights afterwards leaves the remaining weights not
  summing to 1, so you get a scaled-down average, silently.
- **`-inf`, not a large negative number.** `-1e9` works in float32 and produces `nan` in float16,
  because `-1e9` is outside its range. Use `float("-inf")` or `torch.finfo(dtype).min`.

The signature of a broken causal mask is unmistakable: **training loss drops far below what the task
allows — often near zero — while generation is garbage.** The model has learned to copy the answer
it can see. B0's `ln(V)` check catches this at step 0.

### Multiple heads

One head computes one weighted average, so it can express one relationship — "attend to the
subject", say. Language needs many simultaneously: syntactic dependency, coreference, the previous
occurrence of this token, positional locality.

So run `n_heads` attentions in parallel, each in a `d_head = d_model / n_heads` subspace, and
concatenate:

```
d_model = 192, n_heads = 6  →  d_head = 32
```

**Multi-head is free.** Six heads of dimension 32 use exactly the parameters of one head of
dimension 192 — the projections are the same total size, just reshaped. You get six independent
relationships for the price of one.

The final `W_o` is not decoration: concatenated head outputs are six independent 32-dim blocks, and
`W_o` mixes them into a single 192-dim vector that the residual stream can absorb. Remove it and
heads cannot combine.

### Cost, and why context is expensive

The `(T, T)` attention matrix is the whole story:

| Quantity | Scaling |
|----------|---------|
| Attention matrix | `B × n_heads × T × T` |
| Compute | `O(T² · d)` |
| Memory | `O(T²)` per head |

Doubling context quadruples both. At `T=256, B=32, heads=6` in float32 that matrix is 32 × 6 × 256 ×
256 × 4 bytes ≈ **50 MB per layer** — larger than your entire model. This is why long context is
hard, why flash attention (which never materialises the matrix) matters, and why the KV cache is the
dominant memory cost at serving time.

You will measure the quadratic yourself in Lab 6.

---

## Where it's used

- Every transformer layer, in every model.
- The `(T,T)` matrix is what attention visualisations show, and what flash attention avoids.
- The causal mask is the difference between a language model and BERT.
- Cross-attention (Module 05a) is this with K and V from a different sequence.
- GQA/MQA (Module 07) reduce the number of K/V heads to shrink the cache.
- The KV cache (B7, Module 19) exists because K and V for past positions never change.

---

## Labs

### Lab 1 — The weighted average, three ways

Build up to attention from something you already understand. Start with a plain running average:

```python
import torch
torch.manual_seed(1337)
B, T, C = 4, 8, 32
x = torch.randn(B, T, C)

# v1: explicit loop — the definition
xbow = torch.zeros(B, T, C)
for b in range(B):
    for t in range(T):
        xbow[b, t] = x[b, :t+1].mean(dim=0)

# v2: matrix multiply with a lower-triangular averaging matrix
w = torch.tril(torch.ones(T, T))
w = w / w.sum(dim=1, keepdim=True)
xbow2 = w @ x

# v3: softmax over a masked matrix of zeros — the form attention uses
tril = torch.tril(torch.ones(T, T))
scores = torch.zeros(T, T).masked_fill(tril == 0, float("-inf"))
xbow3 = torch.softmax(scores, dim=-1) @ x

assert torch.allclose(xbow, xbow2, atol=1e-6)
assert torch.allclose(xbow, xbow3, atol=1e-6)
print(torch.softmax(scores, dim=-1))
```

**All three are identical, and v3 is attention with the scores set to zero.** Attention is a uniform
average where the model gets to choose the weights instead. Print that softmax matrix and look at
the lower triangle — row `t` has `1/(t+1)` in the first `t+1` slots.

### Lab 2 — Scaling, and what happens without it

```python
head_size = 32
q = torch.randn(B, T, head_size)
k = torch.randn(B, T, head_size)

raw = q @ k.transpose(-2, -1)
scaled = raw / head_size**0.5
print("raw var:", raw.var().item(), " scaled var:", scaled.var().item())   # ≈32 vs ≈1
```

Now watch softmax saturate:

```python
for name, s in [("scaled", scaled), ("unscaled", raw)]:
    w = torch.softmax(s, dim=-1)
    print(f"{name:9} max weight {w.max():.4f}  entropy {-(w * w.clamp_min(1e-9).log()).sum(-1).mean():.4f}")
```

Then the part that matters — the gradient:

```python
for name, s in [("scaled", scaled), ("unscaled", raw)]:
    s = s.clone().requires_grad_(True)
    torch.softmax(s, dim=-1).sum().backward()
    print(f"{name:9} grad magnitude {s.grad.abs().mean().item():.3e}")
```

**The unscaled gradient is orders of magnitude smaller.** That is the real argument: without
`1/√d_head`, softmax saturates, gradients vanish, and the head never learns. Repeat with
`head_size = 128` and watch it get worse — which is why the scaling depends on `d_head`, not
`d_model`.

### Lab 3 — Single-head causal attention

```python
import torch.nn as nn, torch.nn.functional as F

class Head(nn.Module):
    def __init__(self, d_model, d_head, block_size):
        super().__init__()
        self.key   = nn.Linear(d_model, d_head, bias=False)
        self.query = nn.Linear(d_model, d_head, bias=False)
        self.value = nn.Linear(d_model, d_head, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))
        self.d_head = d_head

    def forward(self, x):
        B, T, C = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)
        scores = q @ k.transpose(-2, -1) * self.d_head**-0.5      # (B,T,T)
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float("-inf"))
        w = F.softmax(scores, dim=-1)
        return w @ v, w
```

`register_buffer` — not a parameter, but moves with `.to(device)` and saves with the checkpoint.
Using a plain tensor is a classic device-mismatch bug.

Verify the causality property directly, which is far stronger than eyeballing the mask:

```python
h = Head(32, 16, 8)
x = torch.randn(1, 8, 32)
out1, w = h(x)
x2 = x.clone(); x2[0, 5:] = torch.randn(3, 32)        # change only the FUTURE
out2, _ = h(x2)
assert torch.allclose(out1[0, :5], out2[0, :5], atol=1e-6)
print("causality verified; row 3 weights:", w[0, 3].tolist())
```

**Row 3 must have exactly four non-zero entries.** If it has eight, your mask is not applied.

### Lab 4 — Break the mask on purpose

```python
class LeakyHead(Head):
    def forward(self, x):
        B, T, C = x.shape
        k, q, v = self.key(x), self.query(x), self.value(x)
        scores = q @ k.transpose(-2, -1) * self.d_head**-0.5
        w = F.softmax(scores, dim=-1)          # ← no mask at all
        return w @ v, w
```

Train a small model with each on tinyshakespeare for 2000 steps:

| Model | Train loss | Val loss | Generation |
|-------|------------|----------|------------|
| Causal | ~1.9 | ~2.0 | Shakespeare-ish |
| **Leaky** | **≪ 1.0** | ~1.9 | **Garbage** |

**The leaky model's training loss is dramatically better.** Nothing errors. This is the exact
signature to memorise: *implausibly low training loss with normal validation loss and bad
generation means information is leaking into the prediction.* You will meet the same signature in
Module 11 as a loss-masking bug and in Module 23 as evaluation contamination.

Also try masking *after* softmax and confirm the weights no longer sum to 1:

```python
w = F.softmax(scores, dim=-1).masked_fill(tril == 0, 0.0)
print(w.sum(dim=-1))          # not 1.0 — a silently scaled-down average
```

### Lab 5 — Multi-head, two ways

First the obvious way, to see what it means:

```python
class MultiHeadSlow(nn.Module):
    def __init__(self, d_model, n_heads, block_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(d_model, d_model // n_heads, block_size)
                                    for _ in range(n_heads)])
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        return self.proj(torch.cat([h(x)[0] for h in self.heads], dim=-1))
```

Then the real way — one fused projection, reshaped. This is what goes into B5:

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, block_size, dropout=0.0):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads, self.d_head = n_heads, d_model // n_heads
        self.qkv  = nn.Linear(d_model, 3 * d_model, bias=False)   # all three at once
        self.proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = dropout
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size))
                                        .view(1, 1, block_size, block_size))

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) → (B, n_heads, T, d_head)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) * self.d_head**-0.5
        scores = scores.masked_fill(self.tril[:, :, :T, :T] == 0, float("-inf"))
        out = F.softmax(scores, dim=-1) @ v                        # (B, nh, T, dh)

        out = out.transpose(1, 2).contiguous().view(B, T, C)       # concat heads
        return self.proj(out)
```

Verify parameter equivalence and the shape sequence:

```python
slow = MultiHeadSlow(192, 6, 256); fast = MultiHeadAttention(192, 6, 256)
p = lambda m: sum(x.numel() for x in m.parameters())
print(p(slow), p(fast))                       # both 4 × 192 × 192 = 147,456
```

**Print every intermediate shape.** `(B,T,C) → (B,T,3C) → 3×(B,T,C) → (B,nh,T,dh) → (B,nh,T,T) →
(B,nh,T,dh) → (B,T,C)`. That sequence is the thing to know cold; every attention implementation you
will ever read is a variation on it.

The `.contiguous()` before `.view()` is required after `transpose` — B1 Lab 1 made you meet that
error already.

Finally, swap in the fused kernel and confirm it agrees:

```python
out_flash = F.scaled_dot_product_attention(q, k, v, is_causal=True)
```

**It never materialises the `(T,T)` matrix** and is substantially faster. Use the manual version for
understanding and to inspect weights; use this one in B5.

### Lab 6 — Measure the quadratic

```python
import time
for T in (64, 128, 256, 512, 1024):
    mha = MultiHeadAttention(192, 6, T)
    x = torch.randn(8, T, 192)
    t0 = time.time()
    for _ in range(20):
        mha(x)
    dt = (time.time() - t0) / 20
    matrix_mb = 8 * 6 * T * T * 4 / 1e6
    print(f"T={T:>5}  {dt*1000:>7.1f} ms   attn matrix {matrix_mb:>7.1f} MB")
```

**Plot time against `T` and against `T²`.** The second is a straight line. Then note that at
`T=1024` the attention matrix alone is ~200 MB — over 50× your model's parameters. That single
observation explains flash attention, KV cache pressure, and why context length is the hardest thing
to scale.

### Lab 7 — Look at what a head learned

After B6 trains, come back and run this:

```python
_, w = head(x)                              # (B, T, T)
import matplotlib.pyplot as plt
plt.imshow(w[0].detach(), cmap="viridis"); plt.xlabel("key position"); plt.ylabel("query position")
```

**Expect an obvious lower-triangular structure, a strong diagonal-adjacent band (attend to the
previous token), and heavy weight on position 0.** That last one is the attention sink of Module 04
— appearing in your own model, unprompted, because softmax must put its mass somewhere and position
0 is the reliable dumping ground.

---

## Failure modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| Train loss ≪ possible, generation garbage | Causal mask missing or broken | Verify with the future-perturbation test |
| Attention weights don't sum to 1 | Masked after softmax | Mask scores with `-inf` before softmax |
| `nan` in float16 | Used `-1e9` instead of `-inf` | `float("-inf")` or `finfo.min` |
| Model won't learn; attention near-uniform | Missing `1/√d_head` → saturated softmax | Scale by `d_head**-0.5` |
| `view() not compatible` | Missing `.contiguous()` after `transpose` | Add it |
| Device mismatch on the mask | Mask stored as a plain tensor | `register_buffer` |
| OOM as context grows | The `(T,T)` matrix | `scaled_dot_product_attention`; shorter blocks |
| Heads produce identical outputs | `W_o` removed, or heads not split correctly | Check the reshape; keep the output projection |
| Scaled by `d_model` not `d_head` | Wrong variance correction | It is per-head |

---

## Interview

**"Derive attention."**
The problem is that position `t` needs a fixed-size representation built from a variable number of
earlier positions, weighted by relevance. A plain average is fixed-size but content-blind; learned
positional weights aren't content-dependent. So let each position emit a query for what it needs and
a key for what it offers, score every query against every key with a dot product, softmax the scores
into weights that are non-negative and sum to one, and use those to average the values. That's
`softmax(QKᵀ/√d)V`. The causal mask sets future scores to `-inf` before the softmax so they get
exactly zero weight.

**"Why divide by √d_head?"**
Variance. If query and key components are independent with unit variance, their dot product over
`d_head` dimensions has variance `d_head`, so scores spread as `√d_head`. Large scores saturate the
softmax into a near-one-hot distribution, and the gradient through a saturated softmax is nearly
zero, so the head stops learning. Dividing by `√d_head` restores unit variance and keeps gradients
alive. I measured it — the unscaled gradient magnitude is orders of magnitude smaller, and it gets
worse as `d_head` grows, which is why the correction is per-head rather than per-model.

**"Why multiple heads?"**
One head produces one weighted average, so it can express one relationship at a time. Language needs
several simultaneously — syntactic dependency, coreference, local position. Multi-head is
essentially free: six heads of dimension 32 use exactly the parameters of one head of dimension 192,
since it's the same projections reshaped. The output projection matters more than it looks — the
concatenated heads are independent blocks, and `W_o` is what mixes them into something the residual
stream can absorb.

**"How would you know your causal mask is broken?"**
Training loss implausibly low while validation loss and generation are normal-to-bad. The model has
learned to copy an answer it can see. The check I actually use is a perturbation test: change only
the future positions of the input and assert the earlier outputs are bit-identical. That's much
stronger than inspecting the mask, and it's the same class of check as verifying the input/target
shift, because they're the same bug wearing different clothes.

**"What's the real cost of attention?"**
The `T × T` matrix, per head per layer. Compute and memory are both quadratic in sequence length, so
doubling context quadruples both. On my own model at 256 tokens and batch 32 that matrix is about 50
MB per layer — bigger than the entire 3.4M-parameter model. At 1024 tokens it's 200 MB. That's the
whole motivation for flash attention, which computes the same result in tiles without ever
materialising the matrix, and it's why KV cache size dominates serving memory rather than weights.

---

## Checkpoint

1. Show the three equivalent weighted averages and explain why v3 is attention with zero scores.
2. Report score variance with and without scaling, and the gradient magnitude for both.
3. Pass the causality perturbation test; show row `t` has exactly `t+1` non-zeros.
4. Reproduce the leaky-mask run and state its signature in one sentence.
5. Show masking after softmax breaks the sum-to-1 property.
6. Write the full shape sequence for multi-head attention from memory.
7. Show slow and fused multi-head have identical parameter counts.
8. Plot latency against `T²` and report the attention-matrix size at `T=1024`.

---

**Next:** [B5 — The transformer, assembled](05-transformer.md) ·
**Back:** [B3 — Data](03-data.md) · [Build Track](README.md)
