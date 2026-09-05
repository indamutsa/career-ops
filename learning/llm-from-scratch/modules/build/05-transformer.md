# B5 — The transformer, assembled

Everything converges here. You have a tokenizer, a data pipeline, and multi-head causal attention.
This lesson adds the four remaining pieces — RMSNorm, RoPE, SwiGLU, and the residual structure that
holds them together — and produces a complete 3.4M-parameter language model whose every parameter
you can account for.

The model you build here is architecturally the same as Llama and Qwen. Not "similar to" — the same
design, at 1/2000th the size. When you read `modeling_qwen3.py` after this, you will recognise every
line.

Reference companions: [Module 02](../02-embeddings.md), [Module 04](../04-positional.md),
[Module 05](../05-transformer-block.md).

---

## Terms

| Term | Meaning |
|------|---------|
| **Residual stream** | The `(B,T,C)` tensor every layer reads from and writes into. |
| **Residual connection** | `x = x + f(x)`. The addition, not the function. |
| **Pre-norm** | Normalise inside the branch: `x + f(norm(x))`. |
| **LayerNorm** | Subtract mean, divide by std, scale and shift. |
| **RMSNorm** | Divide by root-mean-square, scale. No mean subtraction, no bias. |
| **RoPE** | Rotary Position Embedding — rotate Q and K by an angle proportional to position. |
| **`theta` / base** | RoPE's frequency base, usually 10,000. |
| **SwiGLU** | `(Swish(xW_gate) ⊙ xW_up) W_down`. Three matrices. |
| **Gated activation** | One projection modulates another elementwise. |
| **`d_ff`** | MLP hidden width. ≈ `8/3 × d_model` for SwiGLU. |
| **Weight tying** | Sharing the embedding matrix with the output head. |
| **`lm_head`** | Final `(C → V)` projection producing logits. |
| **Init scale** | Standard deviation of the initial random weights. |
| **Depth-scaled init** | Shrinking residual-output layers by `1/√(2·n_layers)`. |

---

## Concepts

### The residual stream is the architecture

The most useful mental model, and the one that makes everything else obvious.

A transformer maintains one `(B, T, C)` tensor from embeddings to logits. Every layer **reads** from
it, computes something, and **adds** the result back. Nothing overwrites; everything accumulates.

```python
x = embed(tokens)                    # (B, T, C)
for block in blocks:
    x = x + attn(norm(x))            # read, compute, add
    x = x + mlp(norm(x))             # read, compute, add
logits = lm_head(norm(x))            # (B, T, V)
```

Four consequences worth holding onto:

- **Gradients reach layer 1 unimpeded.** The `+` passes gradient through unchanged (B0's addition
  rule), so the path from loss to embeddings is a clean sum. This is why 100-layer transformers
  train at all.
- **Layers communicate by writing to a shared workspace**, not by passing messages in sequence.
- **The dimension `C` is fixed everywhere** — that invariant is why you can stack any number of
  layers.
- **Removing a middle layer degrades the model rather than breaking it** (Module 16's depth
  pruning). Layers contribute additively.

### Attention communicates, the MLP computes

Two sublayers with genuinely different jobs:

| | Attention | MLP |
|---|-----------|-----|
| Mixes across | **Positions** | **Channels** |
| Sees other tokens | Yes | **No** — applied independently per position |
| Parameters (yours) | 147,456/layer | **294,912/layer** |

The MLP is applied to each position separately and identically. It cannot see other tokens at all.
So: attention moves information *between* positions, the MLP processes information *within* a
position — and **two-thirds of your non-embedding parameters are in the MLP**, which surprises
people who assume attention is where the model lives.

### RMSNorm

LayerNorm subtracts the mean, divides by standard deviation, then scales and shifts:

```
LN(x) = γ · (x − μ) / σ + β
```

RMSNorm drops the mean subtraction and the bias:

```
RMSNorm(x) = γ · x / sqrt(mean(x²) + ε)
```

Simpler, ~10–15% faster, no measurable quality loss — so every recent model uses it. The empirical
finding behind the change was that re-centering contributes little; the re-scaling is what matters.

```python
class RMSNorm(nn.Module):
    def __init__(self, d, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d))

    def forward(self, x):
        # compute in float32 even when the model is bf16 — this matters
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.to(dtype) * self.weight)
```

The float32 upcast is not optional in mixed precision: `x.pow(2)` overflows bf16's range for
activations that are perfectly normal in a trained model.

### RoPE

Attention is permutation-invariant — shuffle the tokens and the same set of Q·K products appears.
Position must be injected explicitly.

The original added a positional vector to the embedding. RoPE instead **rotates** the query and key
vectors by an angle proportional to their position, in 2D pairs of dimensions:

```
θ_k = base^(−2k/d_head),  base = 10000
position m rotates pair k by angle m·θ_k
```

The property that makes it work — and the one to be able to state:

```
⟨R_m q, R_n k⟩ = ⟨q, R_(n−m) k⟩
```

**The dot product depends only on the relative distance `n − m`, never on absolute positions.** You
inject absolute position and attention sees relative position, for free.

Different dimension pairs rotate at different rates: high-frequency pairs (small `k`) complete a
rotation every few tokens and encode fine local order; low-frequency pairs turn slowly and encode
coarse long-range position. It is a positional signal at every scale simultaneously.

```python
def build_rope_cache(d_head, max_len, base=10000.0, device="cpu"):
    inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2, device=device).float() / d_head))
    t = torch.arange(max_len, device=device).float()
    freqs = torch.outer(t, inv_freq)              # (max_len, d_head/2)
    return torch.cos(freqs), torch.sin(freqs)

def apply_rope(x, cos, sin):
    """x: (B, n_heads, T, d_head)"""
    T = x.shape[-2]
    cos, sin = cos[:T][None, None], sin[:T][None, None]
    x1, x2 = x[..., 0::2], x[..., 1::2]           # even and odd dimensions as 2D pairs
    return torch.stack([x1 * cos - x2 * sin,
                        x1 * sin + x2 * cos], dim=-1).flatten(-2)
```

**RoPE is applied to Q and K only, never to V.** Position should influence *which* positions are
attended to, not *what* they contribute. Applying it to V is one of the two classic bugs — the model
trains, badly.

### SwiGLU

The classic FFN is `W₂ · ReLU(W₁x)` — two matrices, expansion ratio 4.

SwiGLU uses three, with a gate:

```python
class SwiGLU(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up   = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))
```

The `gate` branch, squashed through SiLU, multiplies the `up` branch elementwise. That
multiplication is the point — it lets the network suppress or pass features **conditionally on the
input**, which a fixed pointwise nonlinearity cannot do.

Three matrices instead of two means `d_ff` shrinks to keep parameters constant: `4 · d_model`
becomes `8/3 · d_model`. For `d_model = 192` that is 512, which is what your config says.

The honest summary: the quality gain is real but modest, consistent, and free at equal parameter
count — which is why it won.

### Weight tying

The embedding is `(V, C)`: token ID → vector. The output head is `(C, V)`: vector → logits over
tokens. They are transposes of the same relationship, so share the matrix:

```python
self.lm_head.weight = self.embed.weight
```

For you that saves 786,432 parameters — **23% of the model, for one line.** At 3.4M that is
enormous; at 7B it is noise, which is why large models often untie them. It also acts as a
regulariser, and it is why an untrained tied model's logits are already correlated with token
identity.

### Initialisation, and the depth correction

Standard init: normal with std 0.02. But there is a subtlety that matters at depth.

Every layer *adds* to the residual stream, so after `n_layers` blocks the variance has accumulated
across `2 · n_layers` additions. Left uncorrected, activations grow with depth and training gets
unstable.

The fix — from GPT-2, and used by essentially everyone since — is to scale the weights of layers
that *write into* the residual stream (`attn.proj` and `mlp.down`) by `1/√(2·n_layers)`:

```python
for name, p in model.named_parameters():
    if name.endswith(("proj.weight", "down.weight")):
        torch.nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * n_layers))
```

You will measure the difference in Lab 5. It is small at 6 layers and decisive at 48.

---

## Where it's used

- This is the architecture. Llama, Qwen, Mistral, Gemma — RMSNorm, RoPE, SwiGLU, pre-norm, tied or
  untied head. The differences are GQA, MoE, and scale.
- Reading any modern `modeling_*.py` file becomes possible after this lesson.
- The parameter accounting is what you use to size a model to a memory budget (Module 17).
- The residual stream is the object Module 24's logit lens inspects.

---

## Labs

### Lab 1 — Parameter count, derived before it is measured

Do the arithmetic on paper first. This is the lab that most separates people who understand the
architecture from people who have used it.

```
vocab=4096, d_model=192, n_layers=6, n_heads=6, d_ff=512, tied head

embeddings                    4096 × 192              =   786,432
per layer:
  qkv          192 × (3×192)                          =   110,592
  attn.proj    192 × 192                              =    36,864
  mlp.gate     192 × 512                              =    98,304
  mlp.up       192 × 512                              =    98,304
  mlp.down     512 × 192                              =    98,304
  2 × RMSNorm  2 × 192                                =       384
                                          per layer   =   442,752
× 6 layers                                            = 2,656,512
final norm                                            =       192
lm_head (tied — already counted)                      =         0
                                              TOTAL   = 3,443,136
```

Then verify to the digit:

```python
def count_params(model, non_embedding=False):
    n = sum(p.numel() for p in model.parameters())
    if non_embedding:
        n -= model.embed.weight.numel()
    return n

print(f"{count_params(m):,}")                    # must equal your hand calculation
print(f"non-embedding: {count_params(m, True):,}")
```

**If they disagree, find the discrepancy before continuing.** It is always a bias you forgot, a
norm, or double-counting the tied head — and finding it is how you learn to size models by
inspection.

### Lab 2 — The full model

```python
import math, torch, torch.nn as nn, torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class Config:
    vocab_size: int = 4096
    d_model:    int = 192
    n_layers:   int = 6
    n_heads:    int = 6
    d_ff:       int = 512
    block_size: int = 256
    dropout:  float = 0.0

class Attention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.d_head  = cfg.d_model // cfg.n_heads
        self.qkv  = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        split = lambda t: t.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        q, k, v = split(q), split(k), split(v)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)   # NOT v
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.proj(out.transpose(1, 2).contiguous().view(B, T, C))

class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.n1, self.attn = RMSNorm(cfg.d_model), Attention(cfg)
        self.n2, self.mlp  = RMSNorm(cfg.d_model), SwiGLU(cfg.d_model, cfg.d_ff)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.n1(x), cos, sin)      # pre-norm: norm inside the branch
        x = x + self.mlp(self.n2(x))
        return x

class TinyLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg    = cfg
        self.embed  = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.norm_f = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight                  # tie

        cos, sin = build_rope_cache(cfg.d_model // cfg.n_heads, cfg.block_size)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)

        self.apply(self._init)
        for name, p in self.named_parameters():                  # depth-scaled init
            if name.endswith(("proj.weight", "down.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layers))

    def _init(self, mod):
        if isinstance(mod, nn.Linear):
            nn.init.normal_(mod.weight, mean=0.0, std=0.02)
        elif isinstance(mod, nn.Embedding):
            nn.init.normal_(mod.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequence {T} exceeds block {self.cfg.block_size}"
        x = self.embed(idx)
        for block in self.blocks:
            x = block(x, self.cos, self.sin)
        x = self.norm_f(x)
        logits = self.lm_head(x)
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
```

```python
m = TinyLM(Config())
print(f"{count_params(m):,} parameters")
xb, yb = get_batch("train")
logits, loss = m(xb, yb)
print(logits.shape, loss.item(), math.log(4096))
```

**Three assertions before you train it:**

```python
assert logits.shape == (32, 256, 4096)
assert abs(loss.item() - math.log(4096)) < 0.3, "initial loss must be ≈ ln(vocab)"
assert m.lm_head.weight.data_ptr() == m.embed.weight.data_ptr(), "weights not actually tied"
```

That third one catches a real bug: assigning `.weight` before `apply(self._init)` runs means the
init can silently untie them by replacing the tensor. Check the pointer, not the values.

### Lab 3 — Verify RoPE's defining property

```python
d_head, base = 32, 10000.0
cos, sin = build_rope_cache(d_head, 256)
q, k = torch.randn(1, 1, 256, d_head), torch.randn(1, 1, 256, d_head)
qr, kr = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

scores = (qr @ kr.transpose(-2, -1))[0, 0]
# same relative distance → same dot product, given the same underlying vectors
for (m_, n_), (m2, n2) in [((10, 15), (100, 105)), ((3, 4), (200, 201))]:
    print(f"Δ={n_-m_}: {scores[m_, n_]:.4f} vs {scores[m2, n2]:.4f}")
```

Those will differ because `q` and `k` are random per position. Do it properly by holding the
vectors fixed and varying only position:

```python
qv, kv = torch.randn(d_head), torch.randn(d_head)
def rotated_dot(m_, n_):
    Q = qv.view(1,1,1,-1).expand(1,1,m_+1,-1).clone()
    K = kv.view(1,1,1,-1).expand(1,1,n_+1,-1).clone()
    return (apply_rope(Q, cos, sin)[0,0,m_] @ apply_rope(K, cos, sin)[0,0,n_]).item()

print(rotated_dot(10, 15), rotated_dot(100, 105), rotated_dot(200, 205))   # all equal
```

**Those three numbers must match to floating-point precision.** That is `⟨R_m q, R_n k⟩ =
⟨q, R_(n−m) k⟩`, demonstrated on your own implementation, and it is a much better interview answer
than reciting the formula.

Then plot the frequency spectrum:

```python
inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))
print("wavelengths (tokens):", (2 * math.pi / inv_freq).tolist())
```

**The fastest pair cycles every ~6 tokens; the slowest every ~60,000.** Positions beyond the slowest
wavelength are indistinguishable — which is exactly why extrapolation past the training context
fails, and what YaRN and NTK-aware scaling manipulate (Module 04).

### Lab 4 — Ablate every component

Train each variant for 2000 steps on the same data and compare val loss. This table is the empirical
justification for every design decision above.

| Variant | Change | Expect |
|---------|--------|--------|
| Baseline | — | reference |
| No RoPE | Remove position entirely | **Much worse** — bag of words |
| RoPE on V too | `apply_rope(v, …)` | Worse; the classic bug |
| Post-norm | `x = norm(x + f(x))` | Similar at 6 layers, unstable at 24 |
| LayerNorm | Instead of RMSNorm | ≈ equal, slower |
| ReLU MLP, `d_ff=768` | Instead of SwiGLU | Slightly worse at equal params |
| No residual | `x = f(norm(x))` | **Fails to train** |
| Untied head | Separate `lm_head` | +786k params, marginal gain |
| No depth-scaled init | Plain 0.02 | Similar at 6 layers |

**Two rows carry the lesson.** "No residual" failing outright is the strongest evidence for the
residual stream being the architecture rather than a trick. And "no RoPE" collapsing shows how much
of language is word order — the model becomes a bag of words with a fixed budget.

### Lab 5 — Depth-scaled init, where it matters

```python
for n_layers in (6, 12, 24, 48):
    for scaled in (True, False):
        m = TinyLM(Config(n_layers=n_layers, d_model=192))
        if not scaled:
            m.apply(m._init)                       # undo the depth scaling
        x = torch.randint(0, 4096, (4, 256))
        with torch.no_grad():
            h = m.embed(x)
            norms = []
            for b in m.blocks:
                h = b(h, m.cos, m.sin)
                norms.append(h.norm(dim=-1).mean().item())
        print(f"L={n_layers:>2} scaled={scaled}  first {norms[0]:.2f}  last {norms[-1]:.2f}  "
              f"ratio {norms[-1]/norms[0]:.2f}")
```

**Without scaling, the residual-stream norm grows steadily with depth, and the ratio blows up as
layers increase.** At 6 layers it barely matters; at 48 it is the difference between training and
not. This is also the norm-growth phenomenon Module 02 discusses, observed in your own model.

### Lab 6 — Load real weights into your implementation

The definitive proof that what you built is the real architecture. Map Qwen3-0.6B's state dict onto
your class and check the logits match.

```python
from transformers import AutoModelForCausalLM
ref = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", torch_dtype=torch.float32)
cfg = Config(vocab_size=ref.config.vocab_size, d_model=ref.config.hidden_size,
             n_layers=ref.config.num_hidden_layers, n_heads=ref.config.num_attention_heads,
             d_ff=ref.config.intermediate_size, block_size=512)
```

You will need three changes, and each is a lesson:

1. **GQA** — Qwen3 has fewer KV heads than Q heads, so `qkv` splits unevenly and K/V are repeated
   with `repeat_interleave`, **not** `repeat` (Module 07).
2. **QK-norm** — Qwen3 applies RMSNorm to Q and K before RoPE. Omit it and logits diverge.
3. **Untied head** at this size.

```python
x = torch.randint(0, 1000, (1, 16))
with torch.no_grad():
    assert (mine(x)[0] - ref(x).logits).abs().max() < 1e-3
```

**When that assert passes, you have implemented a production LLM.** The three bugs above are the
three everyone hits; having hit them yourself is worth more than reading about them.

### Lab 7 — Where the parameters and the FLOPs actually are

```python
from collections import defaultdict
groups = defaultdict(int)
for n, p in m.named_parameters():
    key = ("embed" if "embed" in n else
           "attn"  if "attn"  in n else
           "mlp"   if "mlp"   in n else "norm")
    groups[key] += p.numel()
total = sum(groups.values())
for k, v in sorted(groups.items(), key=lambda kv: -kv[1]):
    print(f"{k:<6} {v:>10,}  {v/total:>6.1%}")
```

**Expect roughly: embed 23%, mlp 51%, attn 26%, norm ~0%.** Then compute forward FLOPs as
`≈ 2 · N · T` and compare against measured time to get a crude MFU. Two facts to carry: the MLP
holds twice the parameters of attention, and embeddings are a quarter of a small model and
negligible in a large one.

---

## Failure modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| Initial loss ≠ `ln(vocab)` | Init scale, or label leakage | Check both; B0 Lab 2 |
| Loss ~0 immediately | `is_causal=False` or mask broken | B4's perturbation test |
| Model ignores word order | RoPE not applied, or applied after the head split incorrectly | Apply to Q and K after reshaping to `(B,nh,T,dh)` |
| Trains but noticeably worse | RoPE applied to V | Q and K only |
| Parameter count off by 786k | Tying broken by later re-init | Assert `data_ptr()` equality |
| `nan` in bf16 | RMSNorm computed in low precision | Upcast to float32 inside the norm |
| Diverges at depth ≥ 24 | No depth-scaled init, or post-norm | Scale residual-writing layers by `1/√(2L)` |
| Won't train at all | Residual connections missing | `x = x + f(norm(x))`, not `x = f(norm(x))` |
| Logits don't match HF | `repeat` instead of `repeat_interleave`; missing QK-norm | Both, in that order |
| `assert T <= block_size` fires | RoPE cache too short | Build the cache for the max length you'll use |

---

## Interview

**"Describe the architecture you'd implement."**
Decoder-only, pre-norm, with a residual stream that runs unbroken from embeddings to logits. Each
block is two sublayers: RMSNorm then multi-head causal attention added back into the stream, and
RMSNorm then a SwiGLU MLP added back in. Positions come from RoPE applied to queries and keys inside
attention, not from an additive input encoding. Final norm, then a linear head to vocabulary, tied to
the embedding matrix. That's Llama, Qwen, Mistral — the differences at scale are GQA for KV cache
size and MoE for capacity, not the block structure.

**"Why pre-norm, and why RMSNorm?"**
Pre-norm keeps the residual path clean — every layer adds into an unnormalised stream, so gradients
reach layer one without passing through N normalisations. Post-norm, which is what the original
paper's figure shows, interrupts that path at every layer and needs a warmup schedule to train at
depth. RMSNorm drops the mean subtraction and the bias from LayerNorm; the empirical finding was that
re-centering contributes little and the rescaling does all the work, so you get 10–15% speed for no
measurable quality cost. The implementation detail that bites is computing it in float32 even in a
bf16 model, because squaring activations overflows bf16's range.

**"Explain RoPE."**
Attention is permutation-invariant, so position has to be injected. RoPE rotates query and key
vectors by an angle proportional to their position, in 2D pairs of dimensions, with each pair
rotating at a different frequency. The property that makes it work is that the dot product of a
rotated query and a rotated key depends only on the difference of the positions — you inject
absolute position and attention sees relative position for free. It goes on Q and K only, never V,
because position should determine *which* positions are attended to, not what they contribute. And
the frequency spectrum explains extrapolation failure directly: the slowest-rotating pair has a
wavelength of tens of thousands of tokens, so positions beyond it are indistinguishable, which is
what the context-extension methods manipulate.

**"Where are the parameters?"**
Two-thirds of the non-embedding parameters are in the MLP, not attention — the MLP has three
matrices of `d_model × d_ff` against attention's four of `d_model × d_model`, and with SwiGLU's
`8/3` ratio the MLP wins. That surprises people who assume attention is where the model lives.
Embeddings are the other big block and their share is entirely scale-dependent: on my 3.4M model a
4096-vocabulary embedding is 23% of everything, so tying it to the output head saves a quarter of the
model for one line. At 7B the same matrix is noise, which is why large models often untie.

**"How would you know your implementation is correct?"**
Load real weights into it and match logits. I mapped Qwen3-0.6B onto my own class and asserted max
absolute logit difference under 1e-3. Three things break on the way and they're the same three
every time: GQA needs `repeat_interleave` rather than `repeat` to expand KV heads, Qwen3 applies
RMSNorm to Q and K before RoPE which is easy to miss, and the head isn't tied at that size. Before
that, cheaper checks: initial loss equals `ln(vocab)`, a causality perturbation test, and a hand
parameter count matching the measured one to the digit.

---

## Checkpoint

1. Hand-derive the 3,443,136 parameter count and match it exactly in code.
2. Initial loss within 0.3 of `ln(4096)`; assert the head is genuinely tied via `data_ptr`.
3. Demonstrate `⟨R_m q, R_n k⟩` equal for equal `n−m` on your implementation.
4. Report RoPE's fastest and slowest wavelengths and what they imply about extrapolation.
5. Produce the nine-row ablation; explain why "no residual" fails outright.
6. Show residual-norm growth with depth, with and without depth-scaled init.
7. Match Qwen3-0.6B logits to <1e-3 and name the three bugs you hit.
8. Report the parameter split and state where the majority sits.

---

**Next:** [B6 — Training it for real](06-training.md) ·
**Back:** [B4 — Attention](04-attention.md) · [Build Track](README.md)
