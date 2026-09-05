# Module 07 — Architecture variants

Module 05 built the standard block. This module covers the deviations you will actually meet — most
importantly **Mixture of Experts**, because the JD mentions the company's own open-source LLMs and
essentially every current open frontier model (including H Company's Holo3-35B-A3B, whose very name
encodes it) is an MoE.

Knowing what "35B-A3B" means and what it implies for serving is a small piece of knowledge with a
high signal-to-effort ratio.

---

## Terms

| Term | Meaning |
|------|---------|
| **Dense model** | Every parameter participates in every forward pass. |
| **MoE** | Mixture of Experts — only a subset of parameters activate per token. |
| **Expert** | One of several parallel MLPs in an MoE layer. |
| **Router / gate** | The small network choosing experts per token. |
| **Top-k routing** | Each token goes to its k highest-scoring experts (usually k=2 or 8). |
| **Total vs active parameters** | All experts vs those used per token. `35B-A3B` = 35B total, 3B active. |
| **Load balancing loss** | Auxiliary loss encouraging even expert usage. |
| **Expert collapse** | A few experts get all traffic; the rest are dead weight. |
| **Capacity factor** | Max tokens one expert accepts per batch; overflow is dropped or rerouted. |
| **Shared expert** | An expert every token always uses, alongside the routed ones. |
| **Expert parallelism** | Distributing experts across devices. |
| **MLA** | Multi-head Latent Attention (DeepSeek) — compresses the KV cache into a latent. |
| **SSM / Mamba** | State-space models: linear-time sequence modelling, constant-size state. |
| **Hybrid** | Interleaving attention and SSM layers. |
| **Encoder-decoder** | Two stacks with cross-attention (T5, original transformer). |
| **Decoder-only** | One causal stack. Every modern LLM. |
| **Multimodal / VLM** | Vision (or audio) encoder projected into the LM's residual stream. |
| **MTP** | Multi-Token Prediction — predicting several future tokens as an auxiliary objective. |

---

## Concepts

### Mixture of Experts

Replace a block's single MLP with `N` parallel MLPs plus a router. For each token, the router picks
the top `k` and combines their outputs weighted by the routing scores.

```
scores = softmax(W_router · x)            # [N]
top_k  = argtopk(scores, k)
y = Σ_{i in top_k}  scores_i · expert_i(x)
```

**The point:** parameters scale with `N`, but FLOPs per token scale with `k`. A model can hold far
more knowledge than it spends compute on.

| Model | Total | Active | Ratio |
|-------|-------|--------|-------|
| Mixtral 8×7B | 47B | 13B | 3.6× |
| DeepSeek-V3 | 671B | 37B | 18× |
| Qwen3-235B-A22B | 235B | 22B | 11× |
| Holo3-35B-A3B | 35B | 3B | 12× |

So "35B-A3B" reads as: **35B parameters must be in memory, but each token costs roughly what a 3B
dense model costs.** That asymmetry is the whole design, and it has a sharp operational consequence:

- **Memory is set by *total* parameters** — you need to hold all 35B.
- **Compute is set by *active* parameters** — decode is as fast as a 3B model.

Which means **MoE is a bad fit for memory-constrained single-user deployment and an excellent fit
for throughput-oriented serving**, where the memory is amortised across many concurrent requests
and the compute saving is realised on every one. Being able to state that trade-off cleanly is the
useful thing here.

### What goes wrong in MoE

**Expert collapse.** Nothing in the objective forces balanced usage, and it is self-reinforcing: an
expert that gets more traffic trains faster, becomes more useful, attracts more traffic. Left
alone, a few experts take everything and the rest are dead parameters.

The fix is an auxiliary **load-balancing loss** penalising the deviation from uniform usage. It is
a real term in the training objective, typically weighted small (0.01), and tuning it is a genuine
part of training an MoE. DeepSeek-V3 introduced an auxiliary-loss-free scheme using per-expert
bias adjustment instead, because the balancing loss trades off against quality.

**Capacity and dropping.** With a fixed per-expert capacity, tokens beyond it are dropped or
rerouted — so a token's output can depend on what *else* was in the batch. That breaks
per-request determinism in a way people find surprising when debugging.

**Serving complexity.** Experts may live on different devices (expert parallelism), so routing
becomes a communication pattern. All-to-all across nodes is expensive; MoE serving is meaningfully
harder than dense serving.

**Fine-tuning.** LoRA on an MoE typically targets attention and the router, or the shared expert.
Adapting all `N` experts loses much of the parameter efficiency, and routed experts see uneven
gradient depending on traffic.

### Multi-head Latent Attention

DeepSeek's answer to the KV cache wall (Module 03). Instead of caching K and V per head, project
them down to a shared low-dimensional latent, cache *that*, and reconstruct on use.

```
GQA:  cache 2 × layers × kv_heads × head_dim  per token
MLA:  cache 2 × layers × d_latent             per token   (d_latent ≪ kv_heads × head_dim)
```

Roughly an order of magnitude smaller than GQA, with quality reportedly at or above MHA. The cost
is extra compute to reconstruct K and V — which is the right trade, because decode is
bandwidth-bound (Module 19), so trading memory traffic for arithmetic is exactly the direction you
want.

### State-space models

Attention is `O(n²)` in sequence length. SSMs (S4, Mamba) are `O(n)` with a **constant-size**
recurrent state — no KV cache at all, so memory does not grow with context.

The trade: a fixed-size state cannot hold arbitrary detail from arbitrary positions. Attention can
look up any token exactly; an SSM has compressed everything into a bounded state. So SSMs are
strong on long sequences where a summary suffices and weaker on exact retrieval from far back —
precisely the needle-in-a-haystack task (Module 09).

Hence **hybrids**: mostly SSM layers for cheap long-range processing, with a few attention layers
retained for exact lookup. Jamba, Zamba and Nemotron-H take this shape. The framing to remember:
*attention for recall, SSM for context*.

### Decoder-only won, and why

| | Encoder-decoder | Decoder-only |
|---|---|---|
| Structure | Two stacks + cross-attention | One causal stack |
| Training | Needs paired data for its natural objective | Any text |
| Serving | More complex | Simple, one KV cache |
| Prevalence | T5, BART, translation | **Every modern LLM** |

Decoder-only won mostly on simplicity and data: any text is training data, one stack to scale, one
cache to serve. Encoder-only models (BERT) remain the right choice for *embeddings* and
classification, which is Module 21's territory — a bidirectional encoder is genuinely better at
producing a sentence vector than a causal LM is.

### Multimodal, briefly

The dominant pattern is simple: a vision encoder produces patch embeddings, a projector (an MLP)
maps them into the LM's `d_model`, and they are inserted into the token sequence as if they were
tokens. The LM is often frozen initially; only the projector trains.

The key insight for this curriculum: **it all lands in the same residual stream** (Module 02).
Images become vectors in the same space as tokens, and the rest of the model does not need to know
the difference.

---

## Where it's used

- **Reading model cards** — "A3B", "8×7B", "MLA" all become legible.
- **Serving decisions** — MoE memory-vs-compute changes hardware sizing entirely.
- **Fine-tuning MoE** — the target-module choice is different from dense.
- **The interview** — if the company trains its own models, this is the shared vocabulary.

---

## Labs

### Lab 1 — Implement an MoE layer

```python
import torch, torch.nn as nn, torch.nn.functional as F


class MoELayer(nn.Module):
    def __init__(self, d_model, d_ff, n_experts=8, k=2):
        super().__init__()
        self.k, self.n = k, n_experts
        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList([SwiGLU(d_model, d_ff) for _ in range(n_experts)])

    def forward(self, x):
        B, T, D = x.shape
        flat = x.reshape(-1, D)
        scores = F.softmax(self.router(flat), dim=-1)         # [BT, n]
        topv, topi = scores.topk(self.k, dim=-1)
        topv = topv / topv.sum(-1, keepdim=True)              # renormalise

        out = torch.zeros_like(flat)
        for e in range(self.n):
            mask = (topi == e)
            if not mask.any():
                continue
            rows = mask.any(dim=-1).nonzero(as_tuple=True)[0]
            w = (topv * mask).sum(-1)[rows].unsqueeze(-1)
            out[rows] += w * self.experts[e](flat[rows])

        usage = scores.mean(0)                                # for the balance loss
        return out.reshape(B, T, D), usage
```

Then compare parameters and FLOPs against a dense MLP of equal `d_ff`:

```python
dense = SwiGLU(512, 1365)
moe = MoELayer(512, 1365, n_experts=8, k=2)

pd = sum(p.numel() for p in dense.parameters())
pm = sum(p.numel() for p in moe.parameters())
print(f"dense params {pd:,}")
print(f"moe   params {pm:,}   ({pm/pd:.1f}x)")
print(f"moe   ACTIVE {pd*2:,}  ({pd*2/pd:.1f}x compute)")
```

**8× the parameters, 2× the compute.** That single comparison is the MoE pitch.

### Lab 2 — Watch expert collapse happen

```python
moe = MoELayer(128, 340, n_experts=8, k=2)
opt = torch.optim.AdamW(moe.parameters(), lr=1e-3)

for step in range(2000):
    x = torch.randn(8, 32, 128)
    y, usage = moe(x)
    loss = y.pow(2).mean()                    # dummy task
    loss.backward(); opt.step(); opt.zero_grad()
    if step % 400 == 0:
        u = usage.detach()
        print(f"step {step:>5} usage {[f'{v:.3f}' for v in u]}  "
              f"max/min {u.max()/u.min():.1f}x")
```

**Run it without a balance loss and watch the spread widen.** Then add one and watch it stay flat:

```python
def balance_loss(usage, n_experts):
    return n_experts * (usage * usage).sum()      # minimised when uniform

loss = task_loss + 0.01 * balance_loss(usage, moe.n)
```

**Report the max/min usage ratio with and without.** That ratio is the collapse metric and having
produced it yourself makes the explanation concrete.

### Lab 3 — Compare real MoE and dense configs

```python
from transformers import AutoConfig

for name in ["Qwen/Qwen3-1.7B", "Qwen/Qwen3-30B-A3B",
             "mistralai/Mixtral-8x7B-v0.1", "deepseek-ai/DeepSeek-V2-Lite"]:
    try:
        c = AutoConfig.from_pretrained(name, trust_remote_code=True)
        ne = getattr(c, "num_experts", getattr(c, "n_routed_experts", None))
        k  = getattr(c, "num_experts_per_tok", None)
        print(f"{name:<34} d={c.hidden_size:<5} L={c.num_hidden_layers:<3} "
              f"experts={ne}  top-k={k}  "
              f"{'MoE' if ne else 'dense'}")
    except Exception as e:
        print(f"{name:<34} {type(e).__name__}")
```

Then compute, for each MoE: total parameters, active parameters, memory to serve in bf16, and
memory to serve at 4-bit (Module 18). **The 4-bit row is what makes a large MoE deployable**, which
is why quantization and MoE go together.

### Lab 4 — Route inspection on a real MoE

```python
# On a small MoE (Qwen3-30B-A3B needs a rented GPU; use a smaller one locally if available)
routes = []

def hook(mod, inp, out):
    routes.append(out[1].detach() if isinstance(out, tuple) else None)

for layer in model.model.layers:
    if hasattr(layer.mlp, "gate"):
        layer.mlp.gate.register_forward_hook(hook)

for text in ["def fibonacci(n):", "SELECT * FROM orders",
             "The capital of France is", "総理大臣は"]:
    routes.clear()
    model(**tok(text, return_tensors="pt"))
    print(f"{text!r}: top experts per layer -> {[r.argmax(-1).tolist() for r in routes[:4]]}")
```

**Look for whether different content types route to different experts.** The honest finding is
usually that routing is far less interpretable than the "experts specialise by topic" story
suggests — specialisation is often positional or syntactic rather than semantic. Knowing that, and
saying it, is better than repeating the marketing.

### Lab 5 — MLA versus GQA cache arithmetic

```python
def kv_per_token(layers, kv_heads, head_dim, dtype=2):
    return 2 * layers * kv_heads * head_dim * dtype

def mla_per_token(layers, d_latent, dtype=2):
    return 2 * layers * d_latent * dtype

L = 28
print(f"MHA (32 heads):  {kv_per_token(L, 32, 128)/1024:>8.1f} KB/token")
print(f"GQA (8 kv):      {kv_per_token(L,  8, 128)/1024:>8.1f} KB/token")
print(f"MLA (d=512):     {mla_per_token(L, 512)/1024:>8.1f} KB/token")

for seq, batch in [(8192, 32), (32768, 8)]:
    print(f"\nseq={seq} batch={batch}")
    for name, per in [("MHA", kv_per_token(L, 32, 128)),
                      ("GQA", kv_per_token(L, 8, 128)),
                      ("MLA", mla_per_token(L, 512))]:
        print(f"  {name}: {per*seq*batch/1e9:>7.1f} GB")
```

**This table is why MLA exists.** Put it next to Module 19's capacity planning and the connection
to concurrency is direct: smaller KV per token means more concurrent sequences means more
throughput.

### Lab 6 — SSM versus attention on recall

If you can run a Mamba or hybrid model, run Module 09's needle-in-a-haystack on it and on a
comparable attention model.

**Expected:** the SSM handles long context cheaply and does worse at exact retrieval from deep in
the sequence — a bounded recurrent state cannot store arbitrary detail. That result is the concrete
justification for hybrid architectures, and having measured it is better than having read it.

---

## Failure modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| A few experts get all traffic | Expert collapse | Load-balancing loss, or bias-based balancing |
| MoE quality worse than dense at equal *total* params | Compare at equal *active* params | Correct the comparison |
| OOM serving an MoE that "acts like 3B" | Memory follows total, not active | Size for total; quantize |
| Output depends on batch composition | Capacity overflow dropping tokens | Raise the capacity factor |
| LoRA on MoE barely helps | Routed experts get uneven gradient | Target attention, router, or shared expert |
| MoE serving slow across nodes | All-to-all expert communication | Keep expert parallelism intra-node |
| SSM fails needle retrieval | Bounded state can't store arbitrary detail | Hybrid with some attention layers |
| Encoder-only used for generation | Wrong architecture class | BERT-family is for embeddings/classification |
| VLM ignores the image | Projector undertrained, or tokens misplaced | Check the insertion and train the projector |

---

## Interview

**"What does 35B-A3B mean and what does it imply?"**
35 billion total parameters, about 3 billion active per token — a mixture of experts with roughly a
12× ratio. The implication is that memory and compute decouple: you need to hold all 35B in memory,
but each token costs roughly what a 3B dense model costs. So it's a poor fit for a
memory-constrained single-user deployment and an excellent one for throughput-oriented serving,
where the memory is amortised across many concurrent requests and the compute saving lands on every
one. It also pairs naturally with 4-bit quantization, because quantization attacks exactly the term
that's binding.

**"What's the hard part about training an MoE?"**
Load balancing. Nothing in the objective encourages even expert usage and the dynamics are
self-reinforcing — an expert getting more traffic trains faster, gets more useful, attracts more
traffic — so without intervention a few experts take everything and the rest are dead parameters.
The standard fix is an auxiliary balancing loss, weighted small, but it trades off against quality,
which is why DeepSeek-V3 moved to an auxiliary-loss-free scheme using per-expert bias adjustment.
There's also a subtlety people hit in debugging: with a fixed capacity factor, tokens beyond an
expert's capacity get dropped or rerouted, so a token's output can depend on what else was in the
batch.

**"How would you fine-tune an MoE?"**
Not by putting LoRA on every expert — that loses most of the parameter efficiency, and routed
experts get very uneven gradient depending on traffic, so some adapters barely train. I'd target
the attention projections, the router, and the shared expert if there is one. If the goal is
behavioural adaptation, that's usually sufficient, since routing and attention carry a lot of the
behaviour. And I'd watch expert usage during fine-tuning, because a small dataset can shift routing
sharply.

**"Why hasn't Mamba replaced attention?"**
Because the trade is real, not free. SSMs are linear in sequence length with a constant-size state,
so no KV cache and cheap long context — but a bounded state can't hold arbitrary detail from
arbitrary positions, and attention can look up any token exactly. That shows up directly on
needle-in-a-haystack: SSMs handle long sequences fine and do worse at exact retrieval from deep in
them. Which is why the interesting models are hybrids — mostly SSM layers for cheap context, a few
attention layers retained for exact recall. Attention for recall, SSM for context.

**"What's MLA?"**
DeepSeek's approach to the KV cache wall. Rather than caching keys and values per head, project
them into a shared low-dimensional latent, cache that, and reconstruct on use — roughly an order of
magnitude smaller than GQA with quality reportedly at or above full MHA. The cost is extra
arithmetic to reconstruct, which is the right direction to trade, because decode is
memory-bandwidth bound, so spending FLOPs to save memory traffic is exactly what you want. And
smaller KV per token translates directly into more concurrent sequences, which is where serving
throughput comes from.

---

## Checkpoint

1. Implement an MoE layer and report the parameter/compute ratio.
2. Reproduce expert collapse and fix it with a balance loss; report the usage ratio both ways.
3. Compute total, active, and 4-bit serving memory for three real MoE models.
4. Inspect routing and report honestly what specialisation you see.
5. Produce the MHA/GQA/MLA KV table at two context settings.
6. Explain the SSM trade-off in terms of a specific measurable task.

---

**Next:** [08 — Decoding and sampling](08-decoding.md) ·
**Back:** [06 — Pretraining](06-pretraining.md) · [Syllabus](../SYLLABUS.md)
