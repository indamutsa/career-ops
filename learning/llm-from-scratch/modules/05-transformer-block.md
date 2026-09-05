# Module 05 — The transformer block

Modules 02–04 covered the pieces. This module assembles them into the repeated unit that *is* the
model, and ends with you building a working transformer from scratch and loading real weights into
it. That exercise is the single best way to stop treating the architecture as a black box.

---

## Terms

| Term | Meaning |
|------|---------|
| **Block / layer** | One attention sublayer + one MLP sublayer, each with a residual connection. |
| **MLP / FFN** | The position-wise feed-forward network. |
| **`d_ff` / intermediate size** | The MLP's hidden width. Typically 4× `d_model`, or ~8/3× for gated. |
| **Expansion ratio** | `d_ff / d_model`. |
| **GELU / SiLU / Swish** | Smooth activation functions. |
| **GLU** | Gated Linear Unit — one projection gates another. |
| **SwiGLU** | `SiLU(W_gate·x) ⊙ (W_up·x)`, then `W_down`. Three matrices. The modern standard. |
| **Pre-norm / post-norm** | Norm before / after the sublayer. |
| **RMSNorm** | Normalise by root-mean-square; no mean-centering, no bias. |
| **QKV projection** | Producing queries, keys and values from the stream. |
| **Output projection** | `o_proj`, writing attention's result back to the stream. |
| **Parameter count** | The arithmetic that tells you what a model is made of. |
| **Depth vs width** | Number of layers vs `d_model`. |
| **Weight init** | How parameters start. Scaled by depth in deep models. |

---

## Concepts

### The block

```
def block(x):
    x = x + attention(rmsnorm_1(x))     # communication across positions
    x = x + mlp(rmsnorm_2(x))           # computation within a position
    return x
```

That is the whole thing. Stack it `n_layers` times, add an embedding at the bottom and a norm plus
LM head at the top, and you have a modern LLM.

The division of labour is worth internalising:

- **Attention moves information between positions.** It is the only operation in the model that
  does. Remove it and every position is processed independently.
- **The MLP computes on each position independently.** It is applied identically at every position
  and never looks sideways.

"Attention communicates, the MLP computes" is a compact and correct summary, and it is the right
frame for reasoning about what a change to either will affect.

### The MLP, and where the parameters actually are

Classic FFN:

```
mlp(x) = W_down( GELU( W_up(x) ) )        # 2 matrices, d_ff = 4·d_model
```

Modern gated (SwiGLU):

```
mlp(x) = W_down( SiLU(W_gate(x)) ⊙ W_up(x) )   # 3 matrices, d_ff ≈ (8/3)·d_model
```

The gating lets the network modulate one projection by another — a multiplicative interaction that
a plain MLP cannot express cheaply. It costs a third matrix, so `d_ff` is reduced to ~8/3× to keep
the parameter count comparable. Empirically it is a consistent quality win, which is why every
current model uses it.

**Parameter accounting per layer** (with GQA and SwiGLU):

| Component | Parameters |
|-----------|-----------|
| `q_proj` | `d_model × d_model` |
| `k_proj` | `d_model × (kv_heads × head_dim)` |
| `v_proj` | `d_model × (kv_heads × head_dim)` |
| `o_proj` | `d_model × d_model` |
| `gate_proj` | `d_model × d_ff` |
| `up_proj` | `d_model × d_ff` |
| `down_proj` | `d_ff × d_model` |

With `d_ff ≈ 2.67 × d_model`, the MLP holds **roughly 8/3 × 3 = 8 units** against attention's ~2–4.
So **about two-thirds of a transformer's non-embedding parameters are in the MLPs.**

That number matters practically: it is why LoRA target selection should usually include the MLP
projections (Module 12), and it is the basis for the observation that factual knowledge appears to
live disproportionately in MLP layers.

### Why pre-norm won

```
post-norm:  x = norm(x + f(x))     # norm is ON the residual path
pre-norm:   x = x + f(norm(x))     # residual path is CLEAN
```

In pre-norm the identity path from input to output passes through no normalisation, so gradients
reach early layers unattenuated. Post-norm transformers need careful learning-rate warmup and often
will not train at all past moderate depth without it.

The cost: the residual stream's magnitude grows with depth (you saw this in Module 02's Lab 4),
which is why a final norm before the LM head is required.

### Depth versus width

For a fixed parameter budget:

- **Deeper** — more sequential composition; better at multi-step reasoning; harder to parallelise;
  more latency (layers are inherently sequential at decode).
- **Wider** — more capacity per layer; better hardware utilisation; more parallelisable.

In practice the ratio is remarkably consistent: `d_model / n_layers ≈ 100–150` for most well-tuned
models. Qwen3-1.7B: 2048/28 ≈ 73. Llama-3-8B: 4096/32 = 128. It is not a law, but a design far
outside that band usually indicates a specific constraint (latency targets push wider; research on
reasoning pushes deeper).

### Reading a config file

Everything above is visible in `config.json`, and being able to read one fluently — deriving
parameter count, KV cache size and memory footprint from it — is a genuinely useful skill and a
reasonable thing for an interviewer to test.

```
hidden_size            -> d_model
num_hidden_layers      -> n_layers
num_attention_heads    -> query heads
num_key_value_heads    -> KV heads (GQA if fewer)
intermediate_size      -> d_ff
vocab_size             -> embedding rows
max_position_embeddings-> trained context
rope_theta             -> RoPE base (Module 04)
tie_word_embeddings    -> shared E and LM head (Module 02)
```

---

## Where it's used

- **Every model.** This is the object.
- **LoRA targeting** (Module 12) — you choose which of these matrices get adapters.
- **Memory estimation** (Modules 17, 19) — parameter and KV arithmetic comes straight from here.
- **Debugging** — hooks, activation inspection and interpretability all address these modules by
  name.

---

## Labs

### Lab 1 — Derive the parameter count, then check it

```python
def count_params(d_model, n_layers, n_heads, n_kv_heads, d_ff, vocab, tied=True):
    head_dim = d_model // n_heads
    attn = (d_model * d_model                       # q
            + d_model * n_kv_heads * head_dim       # k
            + d_model * n_kv_heads * head_dim       # v
            + d_model * d_model)                    # o
    mlp = 3 * d_model * d_ff                        # gate, up, down
    norms = 2 * d_model
    per_layer = attn + mlp + norms
    emb = vocab * d_model
    total = per_layer * n_layers + emb + (0 if tied else emb) + d_model
    return {"per_layer": per_layer, "attn": attn, "mlp": mlp,
            "layers_total": per_layer * n_layers, "embeddings": emb, "total": total}


from transformers import AutoConfig, AutoModelForCausalLM
c = AutoConfig.from_pretrained("Qwen/Qwen3-0.6B")
p = count_params(c.hidden_size, c.num_hidden_layers, c.num_attention_heads,
                 c.num_key_value_heads, c.intermediate_size, c.vocab_size,
                 tied=c.tie_word_embeddings)

m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
actual = sum(x.numel() for x in m.parameters())

for k, v in p.items():
    print(f"{k:<16}{v:>14,}")
print(f"{'ACTUAL':<16}{actual:>14,}   error {abs(p['total']-actual)/actual:.2%}")
print(f"\nMLP share of a layer: {p['mlp']/p['per_layer']:.1%}")
```

**Get the error under 1%.** Then read the last line — that MLP share is the number behind "most of
the model is MLP", and it is the argument for including MLP projections in your LoRA targets.

### Lab 2 — Build a transformer from scratch

The centrepiece. Write it yourself; do not copy a reference implementation.

```python
import torch, torch.nn as nn, torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.w = nn.Parameter(torch.ones(d)); self.eps = eps

    def forward(self, x):
        return self.w * x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)


class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up   = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Attention(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads):
        super().__init__()
        self.nh, self.nkv = n_heads, n_kv_heads
        self.hd = d_model // n_heads
        self.q = nn.Linear(d_model, n_heads * self.hd, bias=False)
        self.k = nn.Linear(d_model, n_kv_heads * self.hd, bias=False)
        self.v = nn.Linear(d_model, n_kv_heads * self.hd, bias=False)
        self.o = nn.Linear(n_heads * self.hd, d_model, bias=False)

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.q(x).view(B, T, self.nh, self.hd).transpose(1, 2)
        k = self.k(x).view(B, T, self.nkv, self.hd).transpose(1, 2)
        v = self.v(x).view(B, T, self.nkv, self.hd).transpose(1, 2)

        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)     # from Module 04

        rep = self.nh // self.nkv                                   # GQA: repeat KV heads
        k = k.repeat_interleave(rep, dim=1)
        v = v.repeat_interleave(rep, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.o(out.transpose(1, 2).reshape(B, T, -1))


class Block(nn.Module):
    def __init__(self, d_model, n_heads, n_kv_heads, d_ff):
        super().__init__()
        self.n1 = RMSNorm(d_model); self.attn = Attention(d_model, n_heads, n_kv_heads)
        self.n2 = RMSNorm(d_model); self.mlp = SwiGLU(d_model, d_ff)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin)      # <- residual, pre-norm
        x = x + self.mlp(self.n2(x))
        return x


class TinyLM(nn.Module):
    def __init__(self, vocab, d_model, n_layers, n_heads, n_kv_heads, d_ff, tied=True):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads, n_kv_heads, d_ff)
                                     for _ in range(n_layers)])
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        if tied:
            self.head.weight = self.emb.weight

    def forward(self, ids):
        x = self.emb(ids)
        cos, sin = rope_cache(x.shape[1], self.blocks[0].attn.hd, x.device)
        for b in self.blocks:
            x = b(x, cos, sin)
        return self.head(self.norm(x))
```

### Lab 3 — Load real weights into your implementation

The proof that Lab 2 is correct.

```python
from transformers import AutoModelForCausalLM
import torch

hf = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", torch_dtype=torch.float32)
c = hf.config
mine = TinyLM(c.vocab_size, c.hidden_size, c.num_hidden_layers, c.num_attention_heads,
              c.num_key_value_heads, c.intermediate_size, c.tie_word_embeddings)

sd = hf.state_dict()
mine.emb.weight.data = sd["model.embed_tokens.weight"]
mine.norm.w.data     = sd["model.norm.weight"]
if not c.tie_word_embeddings:
    mine.head.weight.data = sd["lm_head.weight"]

for i, b in enumerate(mine.blocks):
    p = f"model.layers.{i}."
    b.n1.w.data = sd[p + "input_layernorm.weight"]
    b.n2.w.data = sd[p + "post_attention_layernorm.weight"]
    b.attn.q.weight.data = sd[p + "self_attn.q_proj.weight"]
    b.attn.k.weight.data = sd[p + "self_attn.k_proj.weight"]
    b.attn.v.weight.data = sd[p + "self_attn.v_proj.weight"]
    b.attn.o.weight.data = sd[p + "self_attn.o_proj.weight"]
    b.mlp.gate.weight.data = sd[p + "mlp.gate_proj.weight"]
    b.mlp.up.weight.data   = sd[p + "mlp.up_proj.weight"]
    b.mlp.down.weight.data = sd[p + "mlp.down_proj.weight"]

ids = torch.randint(0, 1000, (1, 16))
with torch.no_grad():
    a, b_ = mine(ids), hf(ids).logits
print("max abs difference:", (a - b_).abs().max().item())
```

**Target: under 1e-3.** If it is larger, the bug is almost always one of three things — RoPE
applied to V, the GQA repeat done wrong (`repeat_interleave` vs `repeat`), or `q_norm`/`k_norm` in
newer Qwen configs that you have not implemented. Chase it down. Debugging this to convergence
teaches more than any amount of reading, and "I reimplemented the architecture and matched HF logits
to 1e-4" is a strong, checkable claim.

### Lab 4 — Ablate the components

```python
import copy

def ablate(model, what):
    m = copy.deepcopy(model)
    for b in m.blocks:
        if what == "no_attn":
            b.attn.forward = lambda x, c, s: torch.zeros_like(x)
        elif what == "no_mlp":
            b.mlp.forward = lambda x: torch.zeros_like(x)
        elif what == "no_norm":
            b.n1.forward = b.n2.forward = lambda x: x
    return m

text_ids = tok("The capital of France is", return_tensors="pt")["input_ids"]
for what in ["baseline", "no_attn", "no_mlp", "no_norm"]:
    m = mine if what == "baseline" else ablate(mine, what)
    with torch.no_grad():
        top = m(text_ids)[0, -1].topk(3)
    print(f"{what:<10} -> {[tok.decode(i) for i in top.indices]}")
```

**Predict each result before running it.** `no_attn` should destroy anything requiring context —
every position becomes independent. `no_mlp` should degrade badly but differently. `no_norm` will
likely produce numerical chaos in a deep model. Getting your predictions right means you understand
the division of labour.

### Lab 5 — Depth-versus-width ratios in the wild

```python
from transformers import AutoConfig

for name in ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B", "Qwen/Qwen3-8B",
             "meta-llama/Llama-3.2-1B", "meta-llama/Llama-3.1-8B",
             "mistralai/Mistral-7B-v0.3", "google/gemma-2-2b"]:
    try:
        c = AutoConfig.from_pretrained(name)
        print(f"{name:<32} d={c.hidden_size:>5} L={c.num_hidden_layers:>3} "
              f"H={c.num_attention_heads:>3} KV={getattr(c,'num_key_value_heads','-'):>3} "
              f"ff={c.intermediate_size:>6} ff/d={c.intermediate_size/c.hidden_size:.2f} "
              f"d/L={c.hidden_size/c.num_hidden_layers:>5.0f}")
    except Exception as e:
        print(f"{name:<32} {type(e).__name__}")
```

**Look at the last two columns.** `ff/d` clusters near 2.67 (SwiGLU) or 4.0 (classic FFN). `d/L`
clusters in the 70–150 band. Note any model that breaks the pattern and hypothesise why.

### Lab 6 — Train the tiny model on something real

```python
# 4 layers, d_model=128, on tinyshakespeare or your own SQL trajectories
model = TinyLM(vocab=tok.vocab_size, d_model=128, n_layers=4,
               n_heads=4, n_kv_heads=2, d_ff=336)
```

Train it for a few thousand steps and watch the loss fall and the samples become less random.
**Having trained a transformer you wrote yourself, on data you chose, is a different kind of
understanding from having called `Trainer.train()`** — and it takes an afternoon.

---

## Failure modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| Your logits don't match HF | RoPE applied to V | Q and K only |
| Your logits don't match HF | GQA repeat wrong | `repeat_interleave`, not `repeat` |
| Your logits don't match HF | Missing `q_norm`/`k_norm` | Check the config; newer Qwen has them |
| Loss won't go down | Post-norm without warmup | Pre-norm |
| Activations explode with depth | No final norm | Norm before the LM head |
| Parameter count off by ~2× | Assumed 2 MLP matrices instead of 3 | SwiGLU has gate, up, down |
| KV cache larger than expected | Used `n_heads` instead of `n_kv_heads` | Read `num_key_value_heads` |
| Deep model won't train | Init not depth-scaled | Scale residual-branch init by `1/√(2·n_layers)` |
| MLP LoRA "did nothing" | Targeted attention only | Two-thirds of params are in the MLP |

---

## Interview

**"Describe a transformer block."**
Two sublayers, each with a pre-norm and a residual connection: `x = x + attention(norm(x))` then
`x = x + mlp(norm(x))`. Attention is the only operation that moves information between positions;
the MLP computes on each position independently and identically. Modern blocks use RMSNorm rather
than LayerNorm — no mean-centering, no bias, same quality and fewer ops — and SwiGLU rather than a
plain FFN, which is three matrices instead of two with `d_ff` around 8/3 of `d_model` to keep the
parameter count comparable.

**"Where are the parameters?"**
Roughly two-thirds of the non-embedding parameters are in the MLPs. With SwiGLU the MLP is three
`d_model × d_ff` matrices at `d_ff ≈ 2.67 d_model`, so about 8 units per layer, against attention's
2 to 4 depending on how aggressive the GQA ratio is. That matters practically — it's why LoRA
targets should usually include `gate_proj`, `up_proj` and `down_proj` rather than just the
attention projections, and it's consistent with factual knowledge appearing to live
disproportionately in MLP layers.

**"Why pre-norm?"**
Because it keeps the residual path clean. In post-norm the normalisation sits on the identity path,
so gradients get attenuated on the way to early layers and deep models need careful warmup or won't
train at all. Pre-norm leaves an unobstructed path from the loss to every layer. The cost is that
the residual stream's magnitude grows with depth, which is why you need a final norm before the LM
head.

**"How do you know how much memory a model needs, given only its config?"**
Read `hidden_size`, `num_hidden_layers`, `num_attention_heads`, `num_key_value_heads`,
`intermediate_size` and `vocab_size`, and the arithmetic falls out. Parameters: per layer it's four
attention projections plus three MLP matrices, times layers, plus the embedding — and doubled if
weights aren't tied. Training memory is roughly 16 bytes per parameter with AdamW. KV cache is
2 × layers × kv_heads × head_dim × dtype_bytes per token, and that's where `num_key_value_heads`
matters enormously — using `num_attention_heads` there is a common mistake that overestimates by
the GQA ratio.

---

## Checkpoint

1. Derive parameter count from a config to within 1% and state the MLP share.
2. Implement `RMSNorm`, `SwiGLU`, GQA attention and the block from scratch.
3. Load real weights and match HF logits to under 1e-3.
4. Predict then verify the three ablations.
5. Report `ff/d` and `d/L` across six models and note the clustering.
6. Train your own tiny transformer and show the loss curve.

---

**Next:** [05a — The encoder–decoder transformer](05a-encoder-decoder.md) ·
**Back:** [04 — Positional encoding](04-positional.md) · [Syllabus](../SYLLABUS.md)
