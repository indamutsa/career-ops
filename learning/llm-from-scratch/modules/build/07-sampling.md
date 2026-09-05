# B7 — Sampling from your own logits

Your model outputs a `(B, T, V)` tensor of logits. Everything a user perceives as the model's
"personality", "creativity", or "reliability" is a decision made **after** that tensor, in code you
write. This lesson is where you stop treating generation as a black box.

You will implement greedy, temperature, top-k, top-p and min-p decoding from scratch, build a KV
cache and measure the speedup, and see repetition and degeneration appear in your own outputs.

Reference companion: [Module 08](../08-decoding.md).

---

## Terms

| Term | Meaning |
|------|---------|
| **Logits** | Unnormalised scores over the vocabulary, one per token. |
| **Greedy decoding** | Always take `argmax`. Deterministic. |
| **Temperature** | Divide logits by `T` before softmax. `T<1` sharpens, `T>1` flattens. |
| **Top-k** | Keep the `k` highest-probability tokens, renormalise, sample. |
| **Top-p / nucleus** | Keep the smallest set whose cumulative probability ≥ `p`. |
| **Min-p** | Keep tokens with `prob ≥ min_p × max_prob`. Adaptive to confidence. |
| **Repetition penalty** | Divide logits of already-generated tokens. |
| **Degeneration** | The fluent-but-empty looping text greedy decoding produces. |
| **KV cache** | Stored K and V from previous positions, so each new token is O(T) not O(T²). |
| **Prefill** | Processing the prompt — one parallel forward pass. |
| **Decode** | Generating tokens one at a time, memory-bandwidth-bound. |
| **Stop condition** | EOS token, max length, or a stop string. |

---

## Concepts

### Why greedy is worse than it sounds

`argmax` maximises the probability of each token *given the prefix*, which is not the same as
maximising the probability of the sequence — that would be a search over exponentially many
sequences, and greedy is the cheapest possible approximation.

The practical consequence is **degeneration**: greedy decoding falls into loops. "The little girl
went to the park. The little girl went to the park." This is not a bug in your model; it happens to
frontier models at temperature 0 too. The mechanism is a positive feedback loop — once a phrase
appears, its tokens become more likely because the model has now conditioned on seeing them, so they
appear again, and the effect compounds.

**Human text is not the argmax of its own distribution.** Real language is surprising at a fairly
steady rate; always taking the single most likely token produces text with far too little surprise,
which reads as flat and then as looping.

### Temperature is one line and it does everything

```python
probs = F.softmax(logits / temperature, dim=-1)
```

- `T → 0` approaches greedy (the largest logit dominates).
- `T = 1` samples from the model's actual distribution.
- `T → ∞` approaches uniform over the whole vocabulary.

Note the connection back to B0: temperature exploits softmax's *scale* sensitivity — the same
property that made the max-subtraction trick safe, since softmax is shift-invariant but not
scale-invariant.

`T = 1` is not the "correct" setting despite being the honest one. A model trained on 100MB of text
has a distribution with a heavy, badly-estimated tail: thousands of tokens each carrying tiny
probability, which collectively hold a great deal of mass. Sampling at `T=1` over the full vocabulary
means you draw from that garbage tail regularly. **Truncation is what fixes this, not temperature** —
which is why every production sampler truncates first and applies temperature second.

### Top-k, top-p, min-p

**Top-k** keeps a fixed number of candidates. Simple, but the right `k` depends on context: after
"The capital of France is" the distribution is a spike and `k=50` admits 49 wrong answers; mid-story,
`k=50` may be too few.

**Top-p (nucleus)** keeps the smallest set whose cumulative probability reaches `p`. It adapts: a
spike gives a nucleus of one or two tokens, a flat distribution gives hundreds. This is why it became
the default.

**Min-p** keeps tokens whose probability is at least `min_p × max_prob`. Also adaptive, and it
degrades more gracefully at high temperature — it is defined relative to the model's own confidence
rather than to an absolute mass target, which is why it has become popular for creative generation
where people run `T > 1` deliberately.

```python
def sample_next(logits, temperature=1.0, top_k=None, top_p=None, min_p=None):
    logits = logits.clone()
    if temperature <= 0:
        return logits.argmax(-1, keepdim=True)          # greedy
    logits = logits / temperature

    if top_k is not None:
        kth = torch.topk(logits, min(top_k, logits.size(-1)))[0][..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))

    if min_p is not None:
        probs = F.softmax(logits, dim=-1)
        logits = logits.masked_fill(probs < min_p * probs.max(-1, keepdim=True).values,
                                    float("-inf"))

    if top_p is not None:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True, dim=-1)
        sorted_probs = F.softmax(sorted_logits, dim=-1)
        cum = sorted_probs.cumsum(-1)
        remove = (cum - sorted_probs) > top_p            # keep the token that CROSSES p
        sorted_logits = sorted_logits.masked_fill(remove, float("-inf"))
        logits = torch.full_like(logits, float("-inf")).scatter(-1, sorted_idx, sorted_logits)

    return torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
```

**The shifted comparison in the top-p block is the subtle part.** You must keep the token that
*crosses* the threshold, not drop it — otherwise a distribution whose top token already exceeds
probability `p` yields an empty candidate set and `multinomial` throws. The off-by-one here is the
most common sampler bug, and it only fires on confident predictions, so it survives casual testing.

### The KV cache

Naive generation re-runs the whole prefix for every new token: to produce token 100 you recompute
attention for tokens 1–99, whose keys and values have not changed. That is O(T²) work to generate T
tokens.

The cache stores K and V per layer and appends one column per step:

```python
class KVCache:
    def __init__(self, n_layers):
        self.k = [None] * n_layers
        self.v = [None] * n_layers

    def update(self, layer, k, v):
        if self.k[layer] is None:
            self.k[layer], self.v[layer] = k, v
        else:
            self.k[layer] = torch.cat([self.k[layer], k], dim=2)   # (B, nh, T, dh)
            self.v[layer] = torch.cat([self.v[layer], v], dim=2)
        return self.k[layer], self.v[layer]
```

Two consequences that define modern inference:

- **Generation splits into two regimes.** *Prefill* processes the whole prompt in one parallel,
  compute-bound pass. *Decode* generates one token at a time and is **memory-bandwidth-bound** — you
  read the entire model and cache per token to do a trivial amount of arithmetic. Different
  bottlenecks, optimised differently (Module 19).
- **The cache is the memory problem.** Its size is
  `2 × n_layers × n_heads × d_head × seq_len × batch × bytes`. For your model at length 256 that is
  kilobytes. For a 70B model at 128k context it exceeds the weights — which is the entire motivation
  for GQA, MQA, MLA and PagedAttention.

With a cache, RoPE must be applied at the **absolute** position of the new token, not at index 0 of
the one-token input. That is the second classic bug: output that is fine for the prompt and drifts
into nonsense immediately after it.

---

## Where it's used

- Every `generate()` call in every framework is this code, plus batching and scheduling.
- Sampling parameters are the first thing to check when a deployed model "gets worse".
- Evaluation harnesses use greedy/`T=0` for reproducibility — a benchmark run at `T=0.7` is not
  comparable to one at `T=0`.
- Structured output and tool calling use *constrained* decoding: mask logits to a grammar, which is
  exactly the `masked_fill` you just wrote.

---

## Labs

### Lab 1 — Generation with the full sampler

```python
@torch.no_grad()
def generate(model, prompt, max_new=200, **kw):
    model.eval()
    idx = torch.tensor([tok.encode(prompt)], device=device)
    for _ in range(max_new):
        idx_cond = idx[:, -model.cfg.block_size:]           # crop to context
        logits, _ = model(idx_cond)
        nxt = sample_next(logits[:, -1, :], **kw)
        if nxt.item() == EOS:
            break
        idx = torch.cat([idx, nxt], dim=1)
    return tok.decode(idx[0].tolist())
```

Generate 200 tokens from "Once upon a time" under each setting and read the outputs side by side:

| Setting | Expect |
|---------|--------|
| `temperature=0` (greedy) | **Loops within ~50 tokens.** |
| `T=1.0`, no truncation | Fluent, then a nonsense token derails it. |
| `T=0.8, top_k=50` | Coherent. |
| `T=0.8, top_p=0.9` | Coherent, slightly more varied. |
| `T=1.2, min_p=0.05` | Creative, still grammatical. |
| `T=2.0`, no truncation | Word salad. |

**Paste the greedy output into your notes.** Seeing your own model loop is what makes degeneration
permanent knowledge rather than a term you can define.

### Lab 2 — Measure the nucleus

Instrument the sampler to record, at each step, how many tokens survive truncation, then histogram
the counts for `top_p=0.9`.

**Expect a heavily right-skewed distribution: a median of a handful, with a long tail into the
hundreds.** After a token like `"the"` the nucleus is wide; mid-word or after a name it is one or
two. That histogram *is* the argument for nucleus over top-k in a single picture — no fixed `k` can
be right for both ends of it.

### Lab 3 — Entropy over a generation

```python
probs = F.softmax(logits[:, -1, :], dim=-1)
entropy = -(probs * probs.clamp_min(1e-9).log()).sum(-1).item()
```

Plot entropy per generated token. **It is low mid-word and at forced continuations, and spikes at the
start of a sentence or clause** — the decision points. The maximum possible is `ln(4096) = 8.32`; a
trained model averages far below it.

Worth stating the connection once, because it is the same quantity: cross-entropy is this measured
against known targets, predictive entropy is this measured against unknown ones.

### Lab 4 — KV cache, implemented and measured

Add cache support to `Attention.forward`, then:

```python
for T in (64, 128, 256, 512):
    t_nocache = time_generate(model, max_new=T, use_cache=False)
    t_cache   = time_generate(model, max_new=T, use_cache=True)
    print(f"{T:>4}  no-cache {t_nocache:.2f}s  cache {t_cache:.2f}s  "
          f"speedup {t_nocache/t_cache:.1f}×")
```

**The speedup must grow with length — that is the signature of turning O(T²) into O(T).** A constant
factor means your cache is being built but not actually reused.

Then assert correctness, which people skip:

```python
torch.manual_seed(0); a = generate(model, "Once", max_new=50, temperature=0, use_cache=False)
torch.manual_seed(0); b = generate(model, "Once", max_new=50, temperature=0, use_cache=True)
assert a == b, "cached generation diverges from uncached"
```

Greedy makes this deterministic, so any difference is a real bug — almost always RoPE applied at the
wrong position.

### Lab 5 — Cache memory scaling

```python
def kv_bytes(n_layers, n_heads, d_head, seq, batch=1, bytes_per=2):
    return 2 * n_layers * n_heads * d_head * seq * batch * bytes_per

print(kv_bytes(6, 6, 32, 256) / 1e3, "KB   (yours)")
print(kv_bytes(80, 64, 128, 128_000, batch=32) / 1e9, "GB  (70B, 128k, batch 32)")
```

**Your cache is kilobytes; the 70B number is larger than the model weights.** Now recompute the
second line with 8 KV heads instead of 64 — that single change is GQA, and the ratio you get is the
reason every model above ~7B uses it.

### Lab 6 — Repetition penalty, and its cost

```python
def apply_repetition_penalty(logits, generated, penalty=1.1):
    for t in set(generated):
        logits[0, t] = logits[0, t] / penalty if logits[0, t] > 0 else logits[0, t] * penalty
    return logits
```

Sweep `1.0, 1.05, 1.1, 1.3, 1.5` under greedy decoding.

**Low values break loops; high values make the model refuse to reuse necessary words** — articles and
the protagonist's name disappear and the text becomes strained. The lesson is that repetition penalty
is a blunt instrument that treats "the" and a looping phrase identically; better sampling usually
beats a stronger penalty.

### Lab 7 — Batched generation

Generate for 8 prompts at once with left-padding and a per-sequence finished mask.

Three things you must get right, each a real bug in the wild:

1. **Left-pad, not right-pad** — a causal model must see the prompt ending at the last position.
2. **Mask padding out of attention**, or padding tokens contribute to the prefix.
3. **Freeze finished sequences** — once a sequence emits EOS, stop updating it, or it generates past
   its own ending and you see text after the stop token.

**Compare total wall clock against 8 sequential generations.** The speedup is close to linear,
because decode is memory-bandwidth-bound: you read the same weights either way, so extra sequences
are nearly free. That single fact is the foundation of continuous batching and of everything in
Module 19.

---

## Failure modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| Output loops | Greedy/low temperature | Sample; truncate; small repetition penalty |
| Occasional nonsense token | `T=1` over the full tail | `top_p=0.9` or `top_k=50` |
| Word salad | Temperature too high without truncation | Truncate first, then temperature |
| `multinomial` errors on confident steps | Top-p off-by-one drops every candidate | Keep the crossing token |
| Cached output ≠ uncached | RoPE at index 0 instead of absolute position | Pass the true position |
| Cache gives a constant speedup | Cache not actually reused | Verify it grows one column per step |
| Generation never stops | No EOS in training data | B3 — the `<eos>` appended per document |
| Text continues after EOS | Finished sequences not frozen | Per-sequence done mask |
| Batched ≠ single output | Right-padding, or padding unmasked | Left-pad and mask |
| Slower than expected | Recomputing the prompt each step | Prefill once, then decode |

---

## Interview

**"Walk me through decoding strategies."**
Greedy takes the argmax and is deterministic, but it degenerates into loops because it maximises
per-token probability rather than sequence probability, and human text simply isn't the argmax of its
own distribution. Temperature rescales logits before softmax to control sharpness. The truncation
methods matter more: top-k keeps a fixed number of candidates, top-p keeps the smallest set reaching
cumulative probability p and so adapts to the model's confidence, and min-p keeps tokens above a
fraction of the max probability, which holds up better at high temperature. In practice you truncate
first and apply temperature second, because the reason `T=1` over the full vocabulary fails is the
badly-estimated tail, not the sharpness.

**"Why does greedy decoding loop?"**
Positive feedback. Once a phrase is generated the model conditions on having seen it, which raises
the probability of those same tokens, and greedy takes that maximum every time with no mechanism to
escape. I measured it on my own 3.4M model: greedy output loops within about fifty tokens, while the
same checkpoint at `T=0.8, top_p=0.9` stays coherent for hundreds. Same weights, same prompt — the
entire difference is in the sampler.

**"Explain the KV cache and its cost."**
Without it, generating token N re-attends over all N−1 previous tokens whose keys and values haven't
changed, so producing T tokens costs O(T²). Caching K and V per layer makes each step O(T), and the
speedup grows with length — if you measure a constant-factor gain, the cache isn't being used. The
cost is memory: `2 × layers × heads × d_head × seq × batch × bytes`. On my model that's kilobytes; on
a 70B model at 128k context and batch 32 it exceeds the weights, which is exactly why GQA, MQA and
MLA exist and why PagedAttention manages it as pages rather than contiguous buffers.

**"How would you debug a model that never stops generating?"**
Check whether EOS is in the training data at all — the most common cause is a pipeline that
concatenated documents without a separator, so the model has literally never seen a sequence end. If
EOS exists, check that the sampler tests for it, and that truncation can't mask it out: an EOS with
low but non-zero probability gets dropped by an aggressive top-k. In batched generation, check that
finished sequences are frozen, because otherwise you see text continuing past a stop token that was
in fact emitted correctly.

**"Output quality dropped after a deploy but the weights didn't change. Where do you look?"**
Sampling parameters first — a temperature or top-p change is the most common cause and costs nothing
to check. Then the prompt template and whether special tokens are applied consistently with training.
Then quantization, if inference moved. Weights being identical is precisely why the sampler is the
first suspect: everything a user perceives as quality is decided after the logits.

---

## Checkpoint

1. All five strategies in one `sample_next`; six-row output comparison recorded.
2. Greedy loop reproduced and pasted verbatim into your notes.
3. Nucleus-size histogram for `top_p=0.9`; report median and max.
4. Entropy plotted per token; explain the spikes.
5. KV cache implemented, speedup measured at four lengths, greedy-equality asserted.
6. Cache size for your model vs 70B/128k; recompute with 8 KV heads and state the ratio.
7. Repetition-penalty sweep, with the failure at high values described.
8. Batched generation matching single-sequence output, with a wall-clock comparison.

---

**Next:** [B8 — Scaling laws on models you trained](08-scaling.md) ·
**Back:** [B6 — Training it for real](06-training.md) · [Build Track](README.md)
