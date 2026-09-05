# Module 04 — Positional encoding and long context

Attention (Module 03) is permutation-invariant: shuffle the tokens and the raw attention
computation gives the same answer. Position has to be injected deliberately. How it is injected
determines what happens when you feed the model a sequence longer than it was trained on — which
is the entire long-context story.

---

## Terms

| Term | Meaning |
|------|---------|
| **Positional encoding** | Any mechanism that makes attention position-aware. |
| **Absolute PE** | Position `i` gets a vector added to its embedding. |
| **Sinusoidal PE** | The original fixed sin/cos encoding. |
| **Learned absolute PE** | A trainable table of position vectors (GPT-2, BERT). |
| **Relative PE** | Attention depends on `i − j`, not on `i` and `j` separately. |
| **RoPE** | Rotary Position Embedding. Rotates Q and K by an angle proportional to position. |
| **ALiBi** | Adds a linear distance penalty to attention scores. No embedding at all. |
| **`rope_theta` / base** | The base of RoPE's frequency geometric series. Usually 10000. |
| **Wavelength** | `2π/frequency` for a RoPE dimension pair. |
| **Extrapolation** | Working beyond the trained length with no changes. |
| **Interpolation** | Compressing positions to fit longer input into the trained range. |
| **PI (Position Interpolation)** | Linear position scaling. |
| **NTK-aware scaling** | Scale `rope_theta` instead of positions; preserves high frequencies. |
| **YaRN** | Frequency-dependent interpolation; the current standard. |
| **Dynamic NTK** | Scaling adjusted per request by actual length. |
| **Long-context continued pretraining** | Actually training on long sequences after scaling. |
| **Effective context** | The span the model genuinely uses well. Usually below the window. |
| **Attention sink** | The first token absorbing large attention mass. |
| **StreamingLLM** | Keep the sink tokens + a sliding window for unbounded streaming. |

---

## Concepts

### Why position must be injected

```
attention(Q, K, V) = softmax(QK^T / √d) V
```

There is nothing in that expression referencing `i` or `j`. Permute the rows and the output permutes
identically — the operation is a set operation, not a sequence operation. So "the cat sat" and "sat
the cat" are indistinguishable without added position information.

### RoPE — how it actually works

RoPE does not add anything to the embeddings. It **rotates** the query and key vectors by an angle
proportional to their position, in 2D pairs of dimensions.

Split each head's `d_head` dimensions into `d_head/2` pairs. Pair `k` gets frequency:

```
θ_k = base^(-2k/d_head)          base = rope_theta, typically 10000
```

At position `m`, pair `k` is rotated by angle `m · θ_k`:

```
[x_2k  ]  ->  [cos(m·θ_k)  -sin(m·θ_k)] [x_2k  ]
[x_2k+1]      [sin(m·θ_k)   cos(m·θ_k)] [x_2k+1]
```

**The property that makes it work:** the dot product of a rotated query at position `m` and a
rotated key at position `n` depends only on `m − n`.

```
⟨R_m q, R_n k⟩ = ⟨q, R_(n−m) k⟩
```

So RoPE is applied *absolutely* (each position rotated by its own index) but *behaves* relatively.
That is the elegant part and the thing to be able to state in one sentence.

Practical consequences:

- Applied to **Q and K only**, never to V.
- Applied **inside every attention layer**, not once at the input.
- The frequency spectrum spans many scales: low-`k` pairs rotate fast (wavelength of a few tokens,
  encoding local order), high-`k` pairs rotate slowly (wavelength of thousands of tokens, encoding
  coarse position).

### Why naive extrapolation fails

Train on 4k tokens and the model has only ever seen rotation angles up to `4096 · θ_k`. At position
10000 the high-frequency dimensions have wrapped around many times into angle combinations that
never appeared in training. The model has no learned response to them, and quality collapses —
usually abruptly, not gracefully.

### The scaling ladder

| Method | Idea | Cost | Quality |
|--------|------|------|---------|
| Naive extrapolation | Change nothing | Free | Breaks past the trained length |
| **PI** | Divide positions by `s` so `L·s` fits in `L` | Free, or brief fine-tune | Works; blurs local detail |
| **NTK-aware** | Scale `rope_theta` instead | Free | Better — preserves high frequencies |
| **YaRN** | Interpolate low frequencies, extrapolate high, + attention temperature | Short fine-tune | Best. The standard. |
| **Dynamic NTK** | Scale per request by actual length | Free | Good; no penalty on short inputs |
| **Long-context CPT** | Actually train on long sequences | Expensive | Best, and the only real answer |

The insight behind NTK-aware and YaRN: **frequencies should not all be treated the same.**
High-frequency dimensions (short wavelength) encode local ordering and are the ones that break when
compressed — so extrapolate those. Low-frequency dimensions encode coarse position and interpolate
safely. PI compresses everything uniformly, which is why it blurs local detail; YaRN treats each
band according to its wavelength.

**None of these actually teach the model to *use* long context.** They make it not break. A model
scaled to 128k but never trained past 8k will produce coherent text at 100k and still fail to
retrieve from the middle. Scaling is necessary and not sufficient — and saying that plainly is the
honest position.

### ALiBi

No positional embedding at all. Add a linear penalty to attention scores proportional to distance:

```
score(i, j) = q_i · k_j − m_h · |i − j|
```

with a per-head slope `m_h`. Distant tokens are penalised more, and different heads have different
slopes so some are local and some are global.

Extrapolates gracefully by construction — nothing wraps around, the penalty just keeps growing. But
it imposes a strong recency prior, which hurts tasks needing genuine long-range retrieval. RoPE +
YaRN has won in practice; ALiBi is worth knowing as the contrast case that illustrates the
trade-off.

### Attention sinks

Models put large attention mass on the first token, regardless of what it is. The mechanism:
softmax must sum to 1, so when a head has nothing relevant to attend to it needs somewhere to dump
probability mass. The first token is visible to every position under a causal mask, so it becomes
the default dump.

**The practical consequence is important.** If you implement a sliding window by simply dropping
the oldest tokens, you eventually drop the sink — and quality collapses immediately and
catastrophically. StreamingLLM's fix: always keep the first ~4 tokens plus a sliding window. That
gives unbounded streaming with stable quality, for a few lines of code.

This is a genuinely good thing to know. It sounds obscure, it is easy to demonstrate, and it is the
kind of detail that signals you have looked at attention maps rather than read about them.

---

## Where it's used

- **Every transformer**, in every attention layer.
- **Long-context deployment** — the `rope_scaling` config field is where this lives.
- **Agent trajectories** (Module 22) — long trajectories are long contexts.
- **Serving config** — `max_model_len` beyond the trained length requires scaling.

---

## Labs

### Lab 1 — Implement RoPE and verify the relative property

```python
import torch, math

def rope_freqs(d_head, base=10000.0):
    return 1.0 / (base ** (torch.arange(0, d_head, 2).float() / d_head))

def apply_rope(x, pos, base=10000.0):
    """x: [..., d_head]; pos: scalar or [seq]"""
    d = x.shape[-1]
    freqs = rope_freqs(d, base)
    angles = pos.unsqueeze(-1) * freqs if torch.is_tensor(pos) else pos * freqs
    cos, sin = angles.cos(), angles.sin()
    x1, x2 = x[..., 0::2], x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


torch.manual_seed(0)
d = 64
q, k = torch.randn(d), torch.randn(d)

print("dot product of rotated q@m and rotated k@n, for equal (m-n):")
for m, n in [(5, 3), (10, 8), (100, 98), (1000, 998)]:
    val = (apply_rope(q, torch.tensor(float(m))) @ apply_rope(k, torch.tensor(float(n)))).item()
    print(f"  m={m:>5} n={n:>5} (m-n={m-n})  ->  {val:.6f}")
```

**All four values are identical.** That is the entire theoretical claim of RoPE, demonstrated in
ten lines: absolute rotation, relative behaviour. Then vary `m-n` and watch the value change.

### Lab 2 — The frequency spectrum

```python
d_head = 128
for base in (10000.0, 1000000.0):
    f = rope_freqs(d_head, base)
    wl = 2 * math.pi / f
    print(f"\nbase={base:g}")
    for i in [0, 8, 16, 32, 48, 63]:
        print(f"  pair {i:>3}: freq {f[i]:.3e}  wavelength {wl[i]:>12.1f} tokens")
```

**Read the wavelength column.** Pair 0 has a wavelength of a few tokens — that dimension encodes
"which of my immediate neighbours is which". Pair 63 has a wavelength of tens of thousands — coarse
position. Raising `base` from 10k to 1M stretches every wavelength, which is exactly what NTK-aware
scaling does and why it works.

### Lab 3 — Watch extrapolation break

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B",
                                             torch_dtype=torch.float32).to("mps")
print("trained max position:", model.config.max_position_embeddings)
print("rope_theta:", model.config.rope_theta)

text = "The quarterly report describes operational metrics in detail. " * 400

for n in [256, 1024, 2048, 4096, 8192, 16384, 32768]:
    ids = tok(text, return_tensors="pt")["input_ids"][:, :n].to("mps")
    if ids.shape[1] < n:
        break
    with torch.no_grad():
        loss = model(ids, labels=ids).loss.item()
    print(f"len {n:>6}  ppl {math.exp(loss):>12.2f}")
```

**Find the cliff.** Perplexity stays flat up to the trained length and then explodes — often by
orders of magnitude within a factor of two in length. This is not gradual degradation; it is a
wall. Plot it.

### Lab 4 — Fix it with scaling

```python
for cfg in [None,
            {"type": "linear", "factor": 4.0},
            {"type": "dynamic", "factor": 4.0},
            {"rope_type": "yarn", "factor": 4.0,
             "original_max_position_embeddings": model.config.max_position_embeddings}]:
    m = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-0.6B", torch_dtype=torch.float32,
        rope_scaling=cfg, trust_remote_code=True).to("mps")
    name = "none" if cfg is None else cfg.get("type", cfg.get("rope_type"))
    row = []
    for n in (2048, 8192, 16384):
        ids = tok(text, return_tensors="pt")["input_ids"][:, :n].to("mps")
        with torch.no_grad():
            row.append(math.exp(m(ids, labels=ids).loss.item()))
    print(f"{name:<10} " + "  ".join(f"{n}:{p:>9.2f}" for n, p in zip((2048, 8192, 16384), row)))
    del m
```

**Two things to record.** Scaling removes the cliff — perplexity stays finite. And scaling
*slightly worsens* short-context perplexity, which is the cost, and the reason dynamic scaling
exists (it applies no scaling when the input is short).

### Lab 5 — Perplexity is not comprehension

Take the best-scaled model from Lab 4 and run Module 09's needle-in-a-haystack at 2k, 8k and 16k.

**The expected and important result:** perplexity looks healthy at 16k while needle retrieval has
collapsed. The model produces fluent text and cannot find the fact. **This is the empirical basis
for "scaling makes it not break; it doesn't make it work"**, and it is a strong thing to have
measured yourself rather than read.

### Lab 6 — Find the attention sink

```python
out = model(**tok("The quick brown fox jumps over the lazy dog near the river bank.",
                  return_tensors="pt").to("mps"), output_attentions=True)

for layer in [0, 4, 8, 11]:
    a = out.attentions[layer][0]              # [heads, q, k]
    to_first = a[:, :, 0].mean().item()
    print(f"layer {layer:>2}: mean attention to token 0 = {to_first:.3f} "
          f"(uniform would be ~{1/a.shape[-1]:.3f})")
```

Then look at *which* token position 0 is — usually BOS or an unremarkable word. **The mass has
nothing to do with its meaning.** It is a softmax pressure valve.

### Lab 7 — Break and fix streaming

Implement a naive sliding window over a long generation, dropping the oldest tokens from the KV
cache. Watch quality collapse when token 0 falls out. Then apply StreamingLLM: always keep the
first 4 tokens plus the last `w`.

```python
def evict(kv_len, window, n_sink=4):
    """Return the indices to KEEP."""
    if kv_len <= window + n_sink:
        return list(range(kv_len))
    return list(range(n_sink)) + list(range(kv_len - window, kv_len))
```

Compare generated text under `n_sink=0` and `n_sink=4`. **The difference is dramatic and the fix is
four tokens.** This is the module's best demonstration.

---

## Failure modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| Output degenerates past a specific length | Naive extrapolation past trained length | RoPE scaling (YaRN) |
| Short-context quality dropped after scaling | Static scaling applied always | Dynamic scaling |
| Perplexity fine at 32k, retrieval broken | Scaling without long-context training | Continued pretraining on long data |
| Streaming collapses after a while | Sliding window evicted the attention sink | Keep the first ~4 tokens |
| Long-context serving OOMs | KV cache is linear in length | GQA, KV quantization, shorter context |
| Fine-tuned model lost long-context ability | Fine-tuned only on short examples | Mix in long examples |
| Two models disagree at the same length | Different `rope_theta` or scaling config | Check both configs |
| Reordering the prompt changed the answer a lot | Lost-in-the-middle | Put critical content at the ends |

---

## Interview

**"Explain RoPE."**
It rotates the query and key vectors by an angle proportional to their position, in 2D pairs of
dimensions, with each pair getting a different frequency from a geometric series based on
`rope_theta`. The key property is that the dot product between a query rotated by `m` and a key
rotated by `n` depends only on `m − n` — so it's applied absolutely but behaves relatively, which
is what you want for attention. It's applied to Q and K only, never V, and inside every attention
layer rather than once at the input.

**"Why can't you just feed a longer sequence than it was trained on?"**
Because the model has never seen those rotation angles. It trained on positions up to some length,
so the high-frequency dimensions — the ones with wavelengths of a few tokens — have wrapped around
into combinations that never occurred. There's no learned response, and the failure is a cliff, not
a slope: perplexity is flat and then explodes by orders of magnitude within a factor of two in
length.

**"How do you extend a model to 128k?"**
RoPE scaling, and YaRN is the current standard because it treats frequencies differently by
wavelength — interpolate the low-frequency dimensions that encode coarse position, extrapolate the
high-frequency ones that encode local ordering. Uniform position interpolation compresses
everything and blurs local detail. But I'd be clear that scaling only stops it breaking: a model
scaled to 128k without long-context continued pretraining will produce fluent text at 100k and
still fail needle retrieval in the middle. You can measure exactly that — perplexity healthy,
retrieval collapsed. Real long context needs training on long sequences, and that's expensive.

**"What's an attention sink and why should I care?"**
Models dump large attention mass on the first token regardless of what it is, because softmax has
to sum to one and a head with nothing relevant to attend to needs somewhere to put the mass — the
first token is visible from every position under a causal mask, so it becomes the default. You care
because if you implement streaming with a naive sliding window, you eventually evict it and quality
collapses immediately. StreamingLLM's fix is to always keep the first four tokens plus the window,
which is a few lines of code for unbounded stable streaming.

**"RoPE or ALiBi?"**
RoPE with YaRN scaling, which is what's won in practice. ALiBi extrapolates gracefully by
construction — it's just a linear distance penalty on the attention scores, so nothing wraps
around — but that penalty is a strong recency prior, and it hurts on tasks that need genuine
long-range retrieval rather than fluent continuation. It's a useful contrast because it makes the
trade-off explicit: guaranteed extrapolation bought with a built-in bias toward the recent.

---

## Checkpoint

1. Implement RoPE and demonstrate that equal `m − n` gives equal dot products.
2. Report the wavelength spectrum and explain what raising `rope_theta` does.
3. Plot the perplexity cliff and name where it falls.
4. Show scaling removing the cliff and the short-context cost.
5. Show perplexity healthy while needle retrieval fails.
6. Measure attention to token 0 against the uniform baseline.
7. Break streaming by evicting the sink and fix it with four tokens.

---

**Next:** [05 — The transformer block](05-transformer-block.md) ·
**Back:** [03 — Attention](03-attention.md) · [Syllabus](../SYLLABUS.md)
