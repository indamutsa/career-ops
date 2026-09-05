# Module 08 — Decoding and sampling

Everything up to here produces a probability distribution over the next token. Decoding is how you
turn that distribution into text. It is the cheapest lever in the whole stack and the one most
often misconfigured — and in RL post-training, **the sampling temperature is not a preference, it
is a correctness requirement**. That last point is the reason this module sits in the first day's
reading.

---

## Terms

| Term | Meaning |
|------|---------|
| **Logits** | Raw pre-softmax scores over the vocabulary, one per token. |
| **Softmax** | Turns logits into a probability distribution. |
| **Greedy decoding** | Always take `argmax`. Deterministic. |
| **Temperature `T`** | Divide logits by `T` before softmax. `T<1` sharpens, `T>1` flattens. |
| **Top-k** | Keep the k highest-probability tokens, renormalise, sample. |
| **Top-p / nucleus** | Keep the smallest set whose cumulative probability ≥ p. |
| **Min-p** | Keep tokens with `p ≥ min_p × p_max`. Adapts to distribution sharpness. |
| **Typical sampling** | Keep tokens whose surprisal is near the distribution's entropy. |
| **Repetition penalty** | Divide logits of already-generated tokens. |
| **Presence / frequency penalty** | Subtract a constant / a count-scaled amount from seen tokens. |
| **No-repeat n-gram** | Hard ban on repeating any n-gram. |
| **Beam search** | Keep b partial sequences, expand all, keep best b. Maximises sequence likelihood. |
| **Length penalty** | Beam-search correction so long sequences aren't unfairly penalised. |
| **Contrastive search** | Penalise tokens too similar to existing context representations. |
| **Speculative decoding** | A small draft model proposes k tokens; the big model verifies in one pass. |
| **Logit bias** | Manual additive adjustment to specific token logits. |
| **Constrained decoding** | Masking logits so only grammar-valid tokens can be sampled. |
| **Stop sequence** | A string that halts generation when produced. |
| **EOS** | End-of-sequence token. The model's own decision to stop. |
| **Logprob** | `log p(token)`. The unit of every training objective in Modules 14–15. |
| **Perplexity** | `exp(mean negative logprob)`. Lower = less surprised. |
| **Degeneration** | Repetitive, looping, or empty output. |

---

## Concepts

### Temperature, precisely

```
p_i = softmax(logit_i / T)
```

- `T → 0`: distribution collapses to a point mass on the argmax. Equivalent to greedy.
- `T = 1`: the model's own calibrated distribution, unchanged.
- `T > 1`: flattened; low-probability tokens become reachable.

The key intuition: temperature does not add information. It only redistributes probability mass
that the model already assigned. A token with logit far below the top can only be reached at high
`T`, and reaching it usually means incoherence.

### Truncation: top-k, top-p, min-p

The problem temperature alone cannot solve is the **long tail**. A 150k-vocabulary distribution has
tens of thousands of tokens with tiny but nonzero probability. Sample enough tokens and you will
eventually draw one, and one absurd token derails everything after it (this is exposure bias from
Module 11, appearing at inference).

| Method | Rule | Weakness |
|--------|------|----------|
| Top-k | Fixed k tokens | Wrong when the model is very confident (k=50 admits junk) *and* when it is very uncertain (k=50 truncates real options) |
| Top-p | Smallest set with cumulative ≥ p | Adapts to sharpness. The standard default. |
| Min-p | `p ≥ min_p × p_max` | Adapts more directly; good at high temperature |

Order of operations in most implementations: **temperature first, then truncation, then sample.**
This matters — truncating before scaling gives a different result.

**Sensible defaults:**

| Use | Settings |
|-----|----------|
| Deterministic extraction, classification | `T=0` (greedy) |
| Factual QA, code | `T=0.2–0.4`, `top_p=0.9` |
| General chat | `T=0.7`, `top_p=0.9–0.95` |
| Creative | `T=0.9–1.1`, `top_p=0.95` |
| **RL rollout generation** | **`T=1.0`** — see below |

### Why RL rollouts require T=1.0

This is the highest-value paragraph in the module for your interview.

Policy-gradient methods (Module 15) estimate a gradient by sampling from the policy `π_θ` and
weighting by advantage. The estimator is only unbiased if your samples actually come from `π_θ`.

Sampling at `T=0.7` with `top_p=0.9` means you sampled from a *different, truncated, sharpened*
distribution — call it `q`. But the loss then computes `log π_θ(token)` and treats those tokens as
if they were drawn from `π_θ`. They were not. Your gradient is biased, and specifically it is
biased **against exploration**: the low-probability actions that GRPO exists to discover and
reinforce were never sampled in the first place.

Symptom in practice: training reward barely moves, group variance is tiny, and it looks like "GRPO
doesn't work on my task". The actual bug is three characters in a config.

The same applies to the group diversity GRPO needs. If all 8 rollouts in a group are near-identical
because you sampled greedily, the advantage is zero for all of them and there is no gradient — the
zero-variance problem from Module 15, self-inflicted.

**Rule: rollouts for training at `T=1.0`, no top-p truncation (or a very permissive one).
Evaluation at whatever you will actually deploy.** And keep those two configs in separate,
clearly-named places so nobody unifies them "for consistency".

### Beam search, and why chat models don't use it

Beam search maximises total sequence log-likelihood. That is exactly right for machine translation
and summarisation, where there is a single best rendering.

For open-ended generation it produces bland, repetitive text. The reason is that the
highest-likelihood sequence is not the highest-quality one — human text is *not* the maximum of the
model's own distribution. Humans are surprising in a way that maximum-likelihood decoding is not.
That is the "curious case of neural text degeneration" result and the reason nucleus sampling
became standard.

Beam search is also slow (b× the compute) and does not compose with streaming.

### Repetition control

| Method | Mechanism | Risk |
|--------|-----------|------|
| `repetition_penalty` (~1.05–1.15) | Divide seen-token logits | >1.2 breaks required repetition (code, names, JSON keys) |
| `presence_penalty` | Flat subtraction once seen | Milder |
| `frequency_penalty` | Scaled by count | Better for long generations |
| `no_repeat_ngram_size` | Hard ban | **Dangerous** — will break valid repeated syntax |

For structured output and code, prefer low or zero repetition penalty. `no_repeat_ngram_size=3` on
JSON output will forbid a legitimately repeated key pattern and produce invalid JSON.

### Speculative decoding

Decode is memory-bandwidth bound (Module 03): one token per full weight read. Speculative decoding
exploits the fact that verifying k tokens costs about the same as generating one.

1. A small draft model generates k tokens cheaply.
2. The target model scores all k **in a single forward pass**.
3. Accept the longest prefix consistent with the target's distribution; reject the rest.

The acceptance test is designed so the output distribution is **exactly** the target model's — this
is lossless, not an approximation. Typical speedup 2–3× depending on how well the draft agrees.

Variants: **Medusa** (extra prediction heads instead of a separate model), **EAGLE** (drafting in
feature space), **n-gram / prompt lookup** (draft by copying from the prompt — remarkably effective
for summarisation and RAG where output repeats input).

### Constrained decoding

Mask invalid tokens' logits to `-inf` before sampling, according to a grammar or JSON schema.
Guarantees syntactic validity by construction. Covered properly in Module 10; note here that it is
a *decoding-time* intervention, not a prompting one — which is why it cannot fail.

---

## Where it's used

- **Every inference call**, whether you set the parameters or accept defaults.
- **RL rollout generation** — the correctness requirement above.
- **Eval reproducibility** — greedy for anything you want comparable across runs.
- **Serving cost** — speculative decoding is the main latency lever after batching.
- **Structured output** — constrained decoding is how you stop parse failures.
- **Self-consistency** — sample n at `T>0`, majority-vote the answers. Only works with sampling on.

---

## Labs

### Lab 1 — Watch temperature move the distribution

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B",
                                             torch_dtype=torch.bfloat16).to("mps")

prompt = "The capital of France is"
ids = tok(prompt, return_tensors="pt").to("mps")
with torch.no_grad():
    logits = model(**ids).logits[0, -1].float()

for T in [0.1, 0.5, 0.7, 1.0, 1.5, 2.0]:
    p = torch.softmax(logits / T, dim=-1)
    top = torch.topk(p, 5)
    ent = -(p * p.clamp_min(1e-12).log()).sum().item()
    toks = [tok.decode(i).strip() for i in top.indices]
    print(f"T={T:<4} entropy {ent:5.2f}  " +
          "  ".join(f"{t!r}:{v:.3f}" for t, v in zip(toks, top.values)))
```

**Record the entropy column.** Entropy is the honest measure of how much randomness you injected —
far more informative than the temperature number itself.

### Lab 2 — Nucleus size varies wildly with context

```python
def nucleus_size(prompt, T=1.0, p=0.9):
    ids = tok(prompt, return_tensors="pt").to("mps")
    with torch.no_grad():
        logits = model(**ids).logits[0, -1].float()
    probs = torch.softmax(logits / T, dim=-1)
    s, _ = torch.sort(probs, descending=True)
    return int((s.cumsum(0) < p).sum().item()) + 1

for prompt in [
    "The capital of France is",
    "2 + 2 =",
    "Once upon a time,",
    "def fibonacci(n):",
    "My favourite colour is",
    "SELECT * FROM",
]:
    print(f"{nucleus_size(prompt):>5} tokens in nucleus  <- {prompt!r}")
```

**The point:** the nucleus is 1–2 tokens after "The capital of France is" and hundreds after "Once
upon a time,". A fixed `top_k` cannot be right for both. This is the argument for top-p in one
table.

### Lab 3 — Reproduce degeneration, then fix it

```python
def gen(prompt, **kw):
    ids = tok(prompt, return_tensors="pt").to("mps")
    out = model.generate(**ids, max_new_tokens=120, pad_token_id=tok.eos_token_id, **kw)
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)

p = "Write a short paragraph about the sea."

print("--- GREEDY ---");            print(gen(p, do_sample=False))
print("--- BEAM 4 ---");            print(gen(p, num_beams=4, do_sample=False))
print("--- T=1.0 no trunc ---");    print(gen(p, do_sample=True, temperature=1.0, top_k=0, top_p=1.0))
print("--- T=2.0 ---");             print(gen(p, do_sample=True, temperature=2.0, top_k=0, top_p=1.0))
print("--- T=0.8 p=0.95 ---");      print(gen(p, do_sample=True, temperature=0.8, top_p=0.95))
print("--- greedy + rep 1.15 ---"); print(gen(p, do_sample=False, repetition_penalty=1.15))
```

**Expected:** greedy loops. Beam search is bland and may also loop. `T=2.0` is word salad. The
nucleus setting is the only readable one. Repetition penalty rescues greedy at some cost in
fluency. Paste the six outputs into your notes — this is the module's evidence.

### Lab 4 — The RL sampling bug, demonstrated

This is the lab that earns its place in an interview answer.

```python
import torch

def rollouts(prompt, n=8, **kw):
    ids = tok(prompt, return_tensors="pt").to("mps")
    out = model.generate(**ids, max_new_tokens=40, do_sample=True,
                         num_return_sequences=n, pad_token_id=tok.eos_token_id, **kw)
    return [tok.decode(o[ids["input_ids"].shape[1]:], skip_special_tokens=True) for o in out]

p = "Q: How many orders were placed in March?\nA:"

for name, kw in [
    ("T=1.0 untruncated (correct for RL)", dict(temperature=1.0, top_k=0, top_p=1.0)),
    ("T=0.7 top_p=0.9 (typical serving)",  dict(temperature=0.7, top_p=0.9)),
    ("T=0.3 top_p=0.8 (over-sharpened)",   dict(temperature=0.3, top_p=0.8)),
]:
    outs = rollouts(p, n=8, **kw)
    uniq = len(set(outs))
    print(f"{name:<38} unique {uniq}/8")
```

**Then connect it to GRPO.** A group of 8 rollouts with `unique = 1` has zero reward variance,
therefore zero advantage, therefore **zero gradient**. Run this against your real Lab 01 SQL agent
tasks and record `unique/8` per configuration. That number is the mechanism behind "GRPO isn't
learning", and being able to name it on the spot is a strong signal.

### Lab 5 — Speculative decoding

```python
import time
from transformers import AutoModelForCausalLM

target = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B",
                                              torch_dtype=torch.bfloat16).to("mps")
draft  = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B",
                                              torch_dtype=torch.bfloat16).to("mps")

ids = tok("Explain the difference between data drift and concept drift.",
          return_tensors="pt").to("mps")

def timed(**kw):
    t0 = time.time()
    out = target.generate(**ids, max_new_tokens=200, do_sample=False,
                          pad_token_id=tok.eos_token_id, **kw)
    dt = time.time() - t0
    n = out.shape[1] - ids["input_ids"].shape[1]
    return dt, n / dt

d1, tps1 = timed()
d2, tps2 = timed(assistant_model=draft)
print(f"baseline    {d1:.2f}s  {tps1:.1f} tok/s")
print(f"speculative {d2:.2f}s  {tps2:.1f} tok/s   speedup {tps2/tps1:.2f}x")
```

Then verify losslessness: run both greedily and diff the strings. They should be identical. If they
are not, you have found a real bug — the acceptance test is supposed to guarantee this.

### Lab 6 — Self-consistency needs sampling

```python
from collections import Counter

def answer(prompt, n, **kw):
    outs = rollouts(prompt, n=n, **kw)
    return Counter(o.strip().split("\n")[0] for o in outs)

q = "Q: If a train travels 60 km in 45 minutes, what is its speed in km/h?\nA:"
print("greedy   :", answer(q, 1, temperature=1e-4, top_k=1))
print("sampled  :", answer(q, 8, temperature=0.8, top_p=0.95).most_common(3))
```

Self-consistency (sample n, majority vote) beats greedy on reasoning tasks. It **cannot work at
`T=0`** — every sample is identical, so the vote is a single opinion. Cost is n× compute for a
typically several-point accuracy gain.

---

## Failure modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| Output loops the same phrase | Greedy or beam on open-ended text | Nucleus sampling, or mild repetition penalty |
| Occasional incoherent word ruins a good answer | Long tail sampled | Add `top_p=0.9` |
| Bland, generic, safe output | Temperature too low, or beam search | Raise T, drop beams |
| Word salad | Temperature too high, no truncation | Lower T, add top-p |
| **GRPO makes no progress** | **Rollouts sampled at serving temperature** | **`T=1.0`, no truncation, for training rollouts** |
| Eval numbers not reproducible | Sampling enabled in eval | Greedy, fixed seed |
| Invalid JSON despite good prompt | Sampling can always pick an invalid token | Constrained decoding (Module 10) |
| Code output missing repeated syntax | `no_repeat_ngram_size` or high repetition penalty | Set both to off for code |
| Generation never stops | No EOS, no stop sequence, no max tokens | Set all three |
| Speculative decoding gives different text | Acceptance test bug | Must be lossless — investigate |
| Self-consistency does nothing | Sampling was off | `T>0` is a precondition |

---

## Interview

**"What temperature would you use to generate rollouts for GRPO, and why?"**
1.0, with no top-p truncation. Policy gradient assumes samples are drawn from the policy you're
computing logprobs under. Sample at 0.7 with top-p 0.9 and you've drawn from a different,
truncated distribution while the loss pretends otherwise — the estimator is biased, and biased
specifically against the low-probability actions RL exists to discover. It also collapses group
diversity, so the advantage within a group goes to zero and you get no gradient at all. The
symptom is "GRPO isn't learning" and the cause is a sampling config. I keep the training rollout
config and the serving config in separate files for exactly that reason.

**"Top-k or top-p?"**
Top-p, almost always. The size of the plausible next-token set varies enormously with context —
after "The capital of France is" it's essentially one token, after "Once upon a time," it's
hundreds. A fixed k is simultaneously too permissive in the first case and too restrictive in the
second. Top-p adapts to the distribution's actual sharpness. Min-p is a further refinement that
holds up better at high temperature.

**"Why don't chat models use beam search?"**
Because it maximises sequence likelihood, and the highest-likelihood sequence isn't the
highest-quality one for open-ended text. Human writing isn't the mode of the distribution — it's
surprising in ways maximum-likelihood decoding never is, so beam output comes out bland and
repetitive. It's also b× the compute and doesn't stream. It remains correct for translation and
summarisation, where there genuinely is one best rendering.

**"How does speculative decoding give a speedup for free?"**
Decode is memory-bandwidth bound — you read the entire weight matrix to produce one token, so the
GPU is mostly idle. But verifying k proposed tokens is one forward pass, roughly the cost of
generating one. So a cheap draft model proposes k, the target verifies them all at once, and you
accept the longest valid prefix. The acceptance test is constructed so the output distribution is
exactly the target model's — it's lossless, not an approximation. Two to three times, depending on
how often the draft agrees.

**"Your model outputs invalid JSON 3% of the time. Fix it."**
Prompting won't fix it, because sampling can always pick an invalid token no matter how good the
prompt is — 3% is exactly what a tail looks like. Constrained decoding: mask every token the
grammar forbids to `-inf` before sampling, so invalid output is unreachable by construction.
Lowering temperature reduces the rate but never to zero, and it costs quality elsewhere.

---

## Checkpoint

1. Write the temperature formula and explain what it does and does not do.
2. Explain top-p over top-k using your Lab 2 numbers.
3. State the RL sampling rule and the bias argument behind it.
4. Report `unique/8` for three sampling configs on your own agent.
5. Explain speculative decoding, including why it is lossless.
6. Name the only decoding-time guarantee of valid JSON.

---

**Next:** [09 — Chat templates and prompting](09-prompting.md) ·
**Back:** [07 — Architecture variants](07-architectures.md) · [Syllabus](../SYLLABUS.md)
