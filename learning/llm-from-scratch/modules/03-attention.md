# Module 03 — Attention

The mechanism. If you understand this one properly, most of the rest of the curriculum is
bookkeeping around it.

**Why it matters for the interview:** every serving cost, every context-length limit, and every
architectural variant of the last three years (GQA, MQA, FlashAttention, PagedAttention, sliding
windows) exists to work around a property of attention. You cannot reason about inference economics
without it.

---

## Terms

| Term | Meaning |
|------|---------|
| **Query, Key, Value (Q/K/V)** | Three linear projections of the input. Query = "what am I looking for", Key = "what do I offer", Value = "what I contribute if selected". |
| **Scaled dot-product attention** | `softmax(QKᵀ / √d_k) V`. The whole mechanism. |
| **Scaling factor √d_k** | Divisor that keeps dot products from growing with dimension and saturating the softmax. |
| **Causal mask** | Upper-triangular −∞ mask preventing position *i* from attending to *j > i*. What makes the model autoregressive. |
| **Attention head** | One independent Q/K/V projection set. Heads specialise. |
| **MHA** (multi-head attention) | `n` heads, each with its own K and V. The original. |
| **MQA** (multi-query attention) | `n` query heads sharing **one** K/V head. Shrinks KV cache by `n×`; costs quality. |
| **GQA** (grouped-query attention) | `n` query heads sharing `g` K/V heads, `1 < g < n`. The modern default — most of MQA's savings, most of MHA's quality. |
| **KV cache** | Cached K and V for all previous positions, so each new token is O(n) instead of O(n²). The dominant memory cost in serving. |
| **Prefill** | Processing the prompt. All positions at once, compute-bound, parallel. |
| **Decode** | Generating tokens one at a time. Memory-bandwidth-bound, sequential. |
| **FlashAttention** | Exact attention computed tile-by-tile in SRAM without materialising the n×n matrix. Same maths, far less memory traffic. |
| **Sliding-window attention** | Each token attends only to the last `w` tokens. Linear instead of quadratic; loses long-range links. |
| **Attention sink** | The first token(s) receiving large attention mass regardless of content — a softmax pressure valve. Dropping them breaks streaming generation. |
| **Quadratic complexity** | Attention is O(n²·d) in sequence length. The reason context is expensive. |
| **Head dimension `d_k`** | `hidden_size / num_attention_heads`. Typically 64 or 128. |

---

## Concepts

### Deriving it, rather than quoting it

The problem: for token *i*, build a representation that incorporates relevant information from
tokens 1…*i*. "Relevant" has to be learned, and content-dependent.

A weighted sum is the obvious form:

```
out_i = Σ_j  w_ij · v_j
```

So: how do you compute `w_ij`? It must depend on both the token doing the looking and the token
being looked at. A dot product of two learned projections gives exactly that:

```
score_ij = q_i · k_j        where q_i = W_Q x_i ,  k_j = W_K x_j
```

Normalise the scores into weights with softmax so they sum to 1, and let each token contribute a
third learned projection instead of its raw embedding:

```
w_ij = softmax_j(score_ij)         v_j = W_V x_j
```

That is attention. Three learned projections and a softmax-weighted sum.

### Why divide by √d_k

`q · k` sums `d_k` products. If the components are roughly unit-variance and independent, the dot
product has variance ≈ `d_k`, so its magnitude grows like `√d_k`. Feed large values into softmax and
it saturates: one weight ≈ 1, the rest ≈ 0, and the gradient vanishes. Dividing by `√d_k` restores
unit variance and keeps the softmax in a responsive range.

**This is a favourite interview question** because it separates people who derived it from people
who memorised the formula. The answer is about gradient flow, not about "normalisation" in the
abstract.

### The causal mask

For autoregressive language modelling, position *i* must not see the future. Set `score_ij = −∞`
for `j > i` before the softmax; those weights become exactly 0.

The subtlety: **during training, all positions are computed in parallel** and the mask is what
makes that legitimate. Each position predicts its next token using only what precedes it — so one
forward pass yields a loss term for every position at once. That is teacher forcing (Module 06),
and it is why pretraining is efficient at all.

### Multi-head: why not one big head?

One attention pattern per layer is limiting — a token often needs several relations at once
(the syntactic subject, the coreferent, the matching bracket). Split `d_model` into `n` heads of
`d_k = d_model/n`, run attention independently in each, concatenate, project.

Total parameters are the same. You have bought **multiple simultaneous attention patterns** for
free, and empirically heads specialise: induction heads, positional heads, syntactic heads.

### The KV cache, and why it dominates serving

Generating token *n+1* needs Q for the new token, and K/V for **all** previous tokens. Those K/V
values do not change — so cache them.

Without the cache, generating *n* tokens is O(n³) total. With it, O(n²). This is not an
optimisation, it is what makes autoregressive generation viable.

The cost is memory:

```
kv_bytes = 2 × n_layers × n_kv_heads × d_k × seq_len × batch × bytes_per_elem
```

Work it for Qwen3-1.7B (28 layers, 8 KV heads, d_k=128, bf16), 8k context, batch 32:

```
2 × 28 × 8 × 128 × 8192 × 32 × 2  ≈  30 GB
```

**The KV cache is larger than the model.** That single fact drives GQA, KV-cache quantization,
PagedAttention, and every serving architecture decision in Module 19.

### MHA → MQA → GQA

The cache scales with `n_kv_heads`. So reduce it.

| Variant | Q heads | KV heads | Cache | Quality |
|---------|---------|----------|-------|---------|
| MHA | 32 | 32 | 1× | baseline |
| MQA | 32 | 1 | 1/32× | noticeably worse |
| GQA | 32 | 8 | 1/4× | ≈ MHA |

GQA is the compromise everyone converged on. Llama 3, Qwen3, Mistral all ship it. In a config,
`num_attention_heads: 32, num_key_value_heads: 8` means GQA with 4 query heads per KV head.

### Prefill vs decode — two different machines

| | Prefill | Decode |
|---|---|---|
| Input | whole prompt | one token |
| Parallelism | all positions | none, sequential |
| Bottleneck | compute (FLOPs) | memory bandwidth |
| Batching helps? | already saturated | enormously |
| Metric | TTFT | inter-token latency |

They have **opposite** hardware profiles, which is why modern servers schedule them separately
(chunked prefill, disaggregated serving). Decode is bandwidth-bound because each step reads the
entire weight matrix to produce a single token — arithmetic intensity is terrible. Batching
amortises that read across many sequences, which is why throughput scales with batch size while
latency barely moves. **This is continuous batching's entire justification** (Module 19).

### FlashAttention

Naive attention materialises the n×n score matrix in HBM. At n=8192 that is 64M floats per head per
layer — and the *memory traffic*, not the FLOPs, is the bottleneck.

FlashAttention tiles the computation, keeps tiles in SRAM, and uses the online-softmax trick to
accumulate correct results without ever holding the full matrix. **Mathematically identical
output**, several times faster, and memory linear in sequence length instead of quadratic.

Interview-relevant: it is an *exact* method, not an approximation. People confuse it with linear-
attention approximations, which do change the output.

---

## Where it's used

- **Context-length pricing.** Quadratic attention plus linear KV cache is why long context costs
  more than proportionally.
- **Serving capacity.** Max concurrent requests is usually KV-cache-bound, not weight-bound.
- **GQA in every modern config.** You will read `num_key_value_heads` in every model card.
- **Prefix caching.** Shared system prompts across requests reuse KV blocks — a large real saving.
- **Streaming.** Attention sinks explain why naive sliding-window streaming degrades and why
  keeping the first few tokens fixes it.

---

## Labs

### Lab 1 — Attention from scratch, verified against PyTorch

```python
import math
import torch
import torch.nn.functional as F


def attention(Q, K, V, causal: bool = True):
    """Q,K,V: (batch, heads, seq, d_k). Returns (out, weights)."""
    d_k = Q.size(-1)
    scores = (Q @ K.transpose(-2, -1)) / math.sqrt(d_k)      # (b,h,s,s)

    if causal:
        s = scores.size(-1)
        mask = torch.triu(torch.ones(s, s, device=Q.device, dtype=torch.bool), diagonal=1)
        scores = scores.masked_fill(mask, float("-inf"))

    weights = F.softmax(scores, dim=-1)
    return weights @ V, weights


torch.manual_seed(0)
b, h, s, d = 2, 4, 10, 16
Q, K, V = (torch.randn(b, h, s, d) for _ in range(3))

mine, w = attention(Q, K, V, causal=True)
theirs = F.scaled_dot_product_attention(Q, K, V, is_causal=True)

print("max abs diff:", (mine - theirs).abs().max().item())     # ~1e-7
assert torch.allclose(mine, theirs, atol=1e-5)

# Causality check: row i must have zero weight beyond column i.
print("row 3 weights:", w[0, 0, 3].tolist())
assert w[0, 0, 3, 4:].sum().item() == 0.0
print("causal mask verified")
```

### Lab 2 — Prove why √d_k is needed

```python
import torch
import torch.nn.functional as F

for d_k in [8, 64, 512, 4096]:
    q = torch.randn(2000, d_k)
    k = torch.randn(2000, d_k)
    raw = (q * k).sum(-1)
    scaled = raw / d_k ** 0.5

    # Softmax sharpness over a small candidate set
    probs_raw = F.softmax(raw[:16], dim=0)
    probs_scaled = F.softmax(scaled[:16], dim=0)
    print(f"d_k={d_k:>5}  std(raw)={raw.std():>8.2f}  std(scaled)={scaled.std():>5.2f}  "
          f"max_p raw={probs_raw.max():.3f}  scaled={probs_scaled.max():.3f}")
```

**Observe:** `std(raw)` grows as √d_k while `std(scaled)` stays ≈ 1. At d_k=4096 the unscaled
softmax puts nearly all mass on one element — saturated, and its gradient is ~0. That is the
vanishing-gradient failure the scaling prevents.

### Lab 3 — Real attention patterns

```python
import torch
from common.load import load

tok, model = load("Qwen/Qwen3-0.6B")

text = "The cat sat on the mat because it was tired."
ids = tok(text, return_tensors="pt").to(model.device)

with torch.no_grad():
    out = model(**ids, output_attentions=True)

attn = out.attentions                    # tuple: n_layers × (b, heads, seq, seq)
tokens = [tok.decode([i]) for i in ids["input_ids"][0]]
print(f"{len(attn)} layers, {attn[0].shape[1]} heads, seq {len(tokens)}")

def top_attended(layer: int, head: int, pos: int, k: int = 3):
    w = attn[layer][0, head, pos].float()
    vals, idx = w.topk(k)
    return [(tokens[i], round(v.item(), 3)) for v, i in zip(vals, idx)]

it_pos = tokens.index(" it") if " it" in tokens else len(tokens) - 4
print(f"\nposition {it_pos} = {tokens[it_pos]!r}")
for layer in [0, len(attn)//2, len(attn)-1]:
    for head in range(3):
        print(f"  L{layer:>2} H{head}: {top_attended(layer, head, it_pos)}")
```

**Observe:** early layers attend locally and to the first token. Middle layers show the interesting
structure. Look for a head where `" it"` attends to `" cat"` — a coreference head. Also note how
much mass sits on token 0 regardless of content: that is the **attention sink**.

### Lab 4 — Measure the KV cache wall

```python
def kv_cache_gb(n_layers, n_kv_heads, d_k, seq_len, batch=1, bytes_per=2):
    return 2 * n_layers * n_kv_heads * d_k * seq_len * batch * bytes_per / 1e9

configs = {
    # name:            (layers, kv_heads, d_k, weights_GB_bf16)
    "Qwen3-1.7B (GQA)": (28, 8, 128, 3.4),
    "Qwen3-1.7B as MHA": (28, 16, 128, 3.4),
    "Llama-3-8B (GQA)":  (32, 8, 128, 16.0),
    "Llama-3-8B as MHA": (32, 32, 128, 16.0),
}

print(f"{'model':<22}{'weights':>9}{'8k×1':>9}{'8k×32':>9}{'128k×1':>9}")
for name, (L, H, D, W) in configs.items():
    print(f"{name:<22}{W:>8.1f}G"
          f"{kv_cache_gb(L,H,D,8192):>8.1f}G"
          f"{kv_cache_gb(L,H,D,8192,32):>8.1f}G"
          f"{kv_cache_gb(L,H,D,131072):>8.1f}G")
```

**Observe:** at batch 32 / 8k the cache exceeds the weights. The MHA rows show what GQA bought:
a 4× cache reduction, which is a 4× increase in concurrent requests per GPU. That is the whole
commercial argument for GQA, and it is a good thing to be able to state in numbers.

### Lab 5 — Measure the cache's speedup directly

```python
import time
import torch
from common.load import load

tok, model = load("Qwen/Qwen3-0.6B")
prompt = "Write a short paragraph about distributed systems."
ids = tok(prompt, return_tensors="pt").to(model.device)

for use_cache in (True, False):
    torch.mps.synchronize() if model.device.type == "mps" else None
    t0 = time.perf_counter()
    with torch.no_grad():
        model.generate(**ids, max_new_tokens=64, do_sample=False,
                       use_cache=use_cache, pad_token_id=tok.eos_token_id)
    torch.mps.synchronize() if model.device.type == "mps" else None
    dt = time.perf_counter() - t0
    print(f"use_cache={use_cache!s:<5}  {dt:6.2f}s   {64/dt:6.1f} tok/s")
```

**Observe:** several times slower without the cache, and the gap *widens* with `max_new_tokens`
because the recompute is quadratic. Re-run at 128 and 256 tokens and plot the two curves — one is
linear, the other quadratic, and seeing it is worth more than reading it.

---

## Failure modes

| Symptom | Cause |
|---------|-------|
| OOM at high concurrency, plenty of weight headroom | KV cache, not weights. Reduce batch, quantize the cache, or shorten context. |
| Throughput scales with batch but latency does not | Correct and expected — decode is bandwidth-bound. |
| Quality collapse after switching MHA → MQA | Too few KV heads. Use GQA. |
| Streaming degrades after the window slides past the start | Attention sink evicted. Pin the first few tokens. |
| Training loss fine, generation broken | Causal mask wrong or absent — the model saw the future during training. |
| Long-context recall poor in the middle of the prompt | Lost-in-the-middle (Module 04), not an attention bug. |

---

## Interview

**"Why divide by √d_k?"**
Dot products of `d_k` roughly-independent unit-variance components have variance ~`d_k`, so
magnitude grows like √d_k. Large logits saturate the softmax into a near-one-hot distribution whose
gradient is ~0, so learning stalls. Dividing by √d_k restores unit variance. It is a gradient-flow
fix, not cosmetic normalisation.

**"What actually limits how many concurrent requests you can serve?"**
Usually KV cache, not weights. Weights are a fixed cost paid once; the cache is per-request and
grows linearly with sequence length. Give the formula, note the batch-32/8k case where the cache
exceeds the model, and name the levers: GQA, KV quantization, PagedAttention to remove
fragmentation, prefix caching for shared prompts.

**"Why is prefill compute-bound and decode memory-bound?"**
Prefill processes all prompt positions in parallel — large matmuls, high arithmetic intensity,
saturates FLOPs. Decode produces one token per step and must read the entire weight matrix to do
it, so arithmetic intensity is near-worst-case and HBM bandwidth is the limit. That asymmetry is
why continuous batching helps decode enormously and prefill hardly at all, and why servers now
schedule the two phases separately.

**"Is FlashAttention an approximation?"**
No — bit-comparable to naive attention up to floating-point reassociation. It reorders the
computation into SRAM-resident tiles with an online softmax, so it never materialises the n×n
matrix. The win is memory traffic. Contrast with Performer/Linformer-style linear attention, which
*are* approximations and do change outputs.

**"Explain GQA to someone who knows MHA."**
Query heads stay at *n*; K/V heads drop to *g*, with groups of query heads sharing a K/V head. The
KV cache shrinks by *n/g*, which multiplies serving concurrency by the same factor, and quality
stays near MHA — unlike MQA at g=1, which is measurably worse. It is a pure serving-economics
decision made at architecture time.

---

## Checkpoint

1. Implement attention from memory and match `F.scaled_dot_product_attention`.
2. Explain √d_k in terms of gradients.
3. Compute the KV cache size for a given config, context and batch, without notes.
4. State the prefill/decode bottleneck asymmetry and its serving consequence.
5. Explain GQA's trade in terms of concurrent requests per GPU.

---

**Next:** [04 — Positional information](04-positional.md) ·
**Back:** [02 — Embeddings and the residual stream](02-embeddings.md) · [Syllabus](../SYLLABUS.md)
