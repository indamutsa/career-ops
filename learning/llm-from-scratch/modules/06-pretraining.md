# Module 06 — Pretraining

You will not pretrain a frontier model, and no interviewer expects you to have. What they expect is
that you understand **what pretraining produced**, because everything you do in post-training is
constrained by it: SFT selects behaviour the base model already has, RL amplifies capabilities that
already exist, and no amount of either installs knowledge that pretraining did not.

For a company that trains its own open-source LLMs, being fluent about the base-model stage is also
just table stakes for the conversation.

---

## Terms

| Term | Meaning |
|------|---------|
| **Pretraining** | Self-supervised next-token prediction on a very large corpus. |
| **Base model** | The result. Completes text; does not follow instructions. |
| **Causal LM objective** | Predict token `t+1` from tokens `≤ t`. |
| **Corpus** | The training data. Common Crawl, code, books, papers, curated web. |
| **Token budget** | Total tokens seen during training. |
| **Compute budget** | Total FLOPs. `C ≈ 6 · N · D` for `N` params and `D` tokens. |
| **Scaling laws** | Loss as a predictable power law in compute, parameters and data. |
| **Chinchilla-optimal** | ~20 tokens per parameter for a *fixed compute* budget. |
| **Over-training** | Training well past compute-optimal to get a smaller, cheaper-to-serve model. |
| **Emergence** | Capabilities appearing sharply with scale. Partly a metric artefact. |
| **Curriculum** | Ordering or reweighting data through training. |
| **Data mixture** | Proportions of each source. |
| **Quality filtering** | Classifier or heuristic scoring to drop low-quality documents. |
| **Deduplication** | Removing repeated documents. Large measured quality effect. |
| **Continued pretraining (CPT)** | More pretraining on a new domain or longer context. |
| **Annealing / decay phase** | Final training on a high-quality mixture at decaying LR. |
| **Loss spike** | Sudden divergence during a long run. |
| **Checkpoint rewind** | Recovering from a spike by restarting from an earlier checkpoint. |
| **Muon / Adam / AdamW** | Optimizers; Muon is a recent second-order-ish alternative. |

---

## Concepts

### What pretraining produces

A base model is a very good next-token predictor and nothing else. It will:

- Complete text in the style of its input.
- Continue a list, a code block, a table.
- Do few-shot tasks, because a prompt with examples *looks like* text where the pattern continues.

It will not reliably answer a question, follow an instruction, or stop when it should. Ask a base
model "What is the capital of France?" and a very plausible completion is "What is the capital of
Germany? What is the capital of Spain?" — because that is what a list of questions looks like.

**Instruction-following is entirely a post-training artefact.** That framing is worth stating
crisply: the capability is in the base model; the *interface* is what SFT and preference
optimisation build.

### Scaling laws, and the correction that matters

The Kaplan (2020) laws showed loss falls as a power law in compute, and were read as favouring
large models on relatively little data. GPT-3 at 175B parameters and 300B tokens reflects that.

Hoffmann et al. (2022) — Chinchilla — corrected it: for a **fixed compute budget**, parameters and
data should scale roughly **equally**, about **20 tokens per parameter**. Chinchilla at 70B/1.4T
beat Gopher at 280B/300B using the same compute.

**The part that is most often misstated:** Chinchilla-optimal is about *training* compute, not
about the best model to deploy. If you will serve a model to millions of users, inference cost
dominates lifetime cost, so it is rational to train a *smaller* model far past 20 tokens/param —
spending more training compute to get a cheaper-to-serve model. Llama 3 8B saw ~15T tokens, nearly
2,000 tokens per parameter, roughly 100× "optimal". That is a deliberate, correct decision, and
being able to explain why is a good signal.

```
C ≈ 6 · N · D          FLOPs for N params over D tokens (fwd+bwd)
```

That formula plus a GPU's FLOPs and an MFU estimate lets you compute a training run's cost in a
minute. Do that in Lab 1.

### Data is the differentiator

At a fixed architecture and budget, data quality dominates. What actually moves the needle:

| Intervention | Effect |
|--------------|--------|
| **Deduplication** | Large. Repeated documents waste budget and encourage memorisation. |
| **Quality filtering** | Large. A classifier trained on "good" reference text, applied at scale. |
| **Mixture weights** | Large. Code improves reasoning even for non-code tasks. |
| **Decontamination** | Essential for honest evals (Module 13). |
| **Annealing on high-quality data** | Notable. Final phase on curated data at decaying LR. |
| **Toxicity/PII filtering** | Necessary for deployment; small quality effect. |

Two facts worth carrying into an interview: **code in the pretraining mixture improves general
reasoning**, and **the annealing phase punches far above its token share** — the last few percent of
training on curated data matters disproportionately, which is a bridge to why post-training works
at all.

### Emergence, honestly

Some capabilities appear to jump sharply at a scale threshold. The nuance: much of the sharpness is
a **metric artefact**. Exact-match accuracy on multi-step arithmetic is a step function by
construction — getting 4 of 5 digits right scores zero. Measure with a continuous metric (per-token
probability of the correct answer) and improvement is smooth.

The honest position: **capability improves smoothly; discrete metrics make it look sudden.** Some
genuinely discontinuous behaviour may remain, but "emergence" claims should be checked against the
metric before being believed. This is a good answer because it is nuanced without being contrarian.

### Continued pretraining — the one you might actually do

CPT is realistic for an applied team and is the right tool in three cases:

1. **Domain adaptation** where the domain has genuinely different language (legal, clinical, a
   proprietary codebase) — this is the case where SFT is *not* sufficient, because you need new
   knowledge rather than new behaviour.
2. **Context extension** — the training half of Module 04's RoPE scaling.
3. **New language** — adding a language the base model saw little of, usually with tokenizer
   extension (Module 02's Lab 7).

Rules that matter:

- **Lower LR than pretraining** (often 10×) — you are adapting, not learning from scratch.
- **Mix in original-distribution data**, typically 5–30%, or you get catastrophic forgetting at the
  pretraining scale, which is much worse than SFT-scale forgetting.
- **Re-run post-training afterwards.** CPT on an instruct model damages its instruction-following;
  CPT belongs on the base, followed by SFT and preference optimisation again.

That last rule is the one people get wrong: they continue-pretrain the chat model and are surprised
when it stops following instructions.

### What goes wrong in a long run

| Problem | Mechanism | Response |
|---------|-----------|----------|
| **Loss spike** | A bad batch, or an instability that has been building | Rewind to an earlier checkpoint, skip the batch, lower LR |
| Slow divergence | LR too high for the current phase | Adjust the schedule |
| Throughput degradation | Node failure, thermal throttling, a straggler rank | Monitor per-rank throughput |
| Silent data corruption | A shard failed to load; the run trains on less data than logged | Checksum shards; assert token counts |
| Loss looks fine, model is bad | Evaluating loss, not capability | Downstream evals throughout, not just at the end |

Loss spikes are so routine at scale that checkpoint-rewind is standard operating procedure rather
than an emergency. Knowing that is a small marker of familiarity.

---

## Where it's used

- **Understanding your base model's ceiling.** Everything downstream is bounded by it.
- **Model selection** — which base to start post-training from, and why.
- **Domain adaptation** — CPT versus SFT versus RAG is a real decision with a real answer.
- **Cost conversations** — "what would it take to train our own" comes up constantly.

---

## Labs

### Lab 1 — Cost a training run

```python
def training_cost(n_params, n_tokens, tflops_per_gpu, mfu=0.4, gpu_hour_usd=2.0):
    flops = 6 * n_params * n_tokens
    gpu_seconds = flops / (tflops_per_gpu * 1e12 * mfu)
    gpu_hours = gpu_seconds / 3600
    return {"PFLOPs": flops / 1e15, "gpu_hours": gpu_hours,
            "usd": gpu_hours * gpu_hour_usd,
            "days_on_512_gpus": gpu_hours / 512 / 24}

runs = [
    ("Chinchilla-optimal 7B", 7e9, 140e9),
    ("Llama-3-style 8B",      8e9, 15e12),
    ("70B, 15T tokens",      70e9, 15e12),
    ("Your 1.7B CPT, 10B tok", 1.7e9, 10e9),
]
for name, n, d in runs:
    c = training_cost(n, d, 990)      # H100 bf16
    print(f"{name:<26} {c['gpu_hours']:>12,.0f} GPU-h  ${c['usd']:>13,.0f}  "
          f"{c['days_on_512_gpus']:>6.1f} d on 512 GPUs")
```

**Two things to take away.** The ratio between the first two rows is the over-training decision,
quantified. And the last row — a domain CPT on a small model — is affordable, which is why CPT is
the pretraining-adjacent thing an applied team actually does.

### Lab 2 — Reproduce a scaling law on a laptop

```python
# Train several TinyLM configs (Module 05) on the same corpus, fixed steps.
configs = [
    (64,  2, 2),   # d_model, n_layers, n_heads
    (128, 4, 4),
    (192, 6, 6),
    (256, 8, 8),
]
for d, L, H in configs:
    m = TinyLM(vocab=tok.vocab_size, d_model=d, n_layers=L,
               n_heads=H, n_kv_heads=max(1, H // 2), d_ff=int(d * 2.67))
    n = sum(p.numel() for p in m.parameters())
    final_loss = train(m, steps=2000)
    print(f"params {n:>10,}  final loss {final_loss:.4f}")
```

Plot `log(loss)` against `log(params)`. **You will get a straight line**, on a laptop, over three
orders of magnitude less scale than the papers. Scaling laws are not a large-model phenomenon; they
are a property of the objective, and seeing that yourself is worth an afternoon.

### Lab 3 — Base versus instruct, side by side

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

prompts = [
    "What is the capital of France?",
    "Write a SQL query to count orders.",
    "Explain data drift in one sentence.",
]

for name in ["Qwen/Qwen3-0.6B-Base", "Qwen/Qwen3-0.6B"]:
    t = AutoTokenizer.from_pretrained(name)
    m = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.bfloat16).to("mps")
    print("=" * 70, "\n", name)
    for p in prompts:
        ids = t(p, return_tensors="pt").to("mps")
        out = m.generate(**ids, max_new_tokens=60, do_sample=False,
                         pad_token_id=t.eos_token_id)
        print(f"\n  {p}\n  -> {t.decode(out[0][ids['input_ids'].shape[1]:], skip_special_tokens=True)!r}")
```

**The base model will often continue with more questions rather than answer.** That single output
is the best possible illustration of what post-training actually adds, and it is worth pasting into
your notes verbatim.

Then repeat with 3 few-shot examples in the prompt and watch the base model suddenly comply —
demonstrating that the *capability* was there and only the interface was missing.

### Lab 4 — Deduplication's effect, at small scale

```python
import random

base_corpus = load_corpus()                      # ~50k documents
duped = base_corpus + random.choices(base_corpus, k=len(base_corpus) // 2)  # 33% duplicates

for name, corpus in [("clean", base_corpus), ("33% duped", duped)]:
    m = TinyLM(...)
    train(m, corpus, steps=3000)                 # SAME number of steps, not epochs
    print(f"{name:<12} held-out loss {evaluate(m, heldout):.4f} "
          f"memorisation {memorisation_rate(m, corpus):.3f}")
```

Measure memorisation as: fraction of training documents whose continuation the model reproduces
verbatim given a 20-token prefix. **The duped run will show worse held-out loss and higher
memorisation at equal compute** — that is the deduplication result, reproduced small.

### Lab 5 — Continued pretraining with and without replay

```python
domain = load_sql_corpus()                       # schemas, queries, docs
general = load_general_corpus()

runs = {
    "domain only":        domain,
    "90% domain + 10% general": mix(domain, general, 0.9),
    "70% domain + 30% general": mix(domain, general, 0.7),
}
for name, data in runs.items():
    m = load_base()
    cpt(m, data, lr=2e-5, steps=2000)            # 10x lower than pretraining LR
    print(f"{name:<26} domain ppl {ppl(m, domain_eval):>8.2f}  "
          f"general ppl {ppl(m, general_eval):>8.2f}")
```

**The forgetting is much more severe than at SFT scale**, because CPT touches all the weights over
many more tokens. Record the three-by-two table; it is the empirical argument for the replay
fraction.

### Lab 6 — Emergence is partly a metric artefact

```python
# Evaluate the same capability with a discrete and a continuous metric across model sizes.
for d, L in [(64, 2), (128, 4), (192, 6), (256, 8), (384, 12)]:
    m = trained_model(d, L)
    exact = exact_match_accuracy(m, arithmetic_tasks)        # discrete
    logp  = mean_logprob_of_correct(m, arithmetic_tasks)     # continuous
    print(f"d={d:>4} L={L:>3}  exact-match {exact:>6.2%}  mean logprob {logp:>8.3f}")
```

**Plot both.** Exact match will look like a step function; mean logprob will be a smooth line
through the same models. That contrast is the whole argument, produced from your own runs.

---

## Failure modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| Base model won't answer questions | It's a base model | Use the instruct version, or few-shot, or SFT |
| CPT destroyed general ability | No replay data | Mix 5–30% original distribution |
| CPT broke instruction-following | Continued pretraining an *instruct* model | CPT the base, then redo post-training |
| Loss spike mid-run | Bad batch or building instability | Rewind to a checkpoint; skip; lower LR |
| Model worse than a smaller one at equal compute | Not Chinchilla-optimal | Rebalance params and tokens |
| Serving cost too high | Trained compute-optimal, not inference-optimal | Over-train a smaller model |
| Benchmarks great, product bad | Contamination | Decontaminate the pretraining corpus too |
| Memorises training data | Insufficient deduplication | Dedup before training, not after |
| Trained on fewer tokens than logged | A shard silently failed to load | Checksum shards; assert token counts |

---

## Interview

**"What does pretraining actually give you?"**
A next-token predictor with essentially all the knowledge and most of the raw capability, and no
interface. A base model asked "What is the capital of France?" will very plausibly continue with
"What is the capital of Germany?" — because that's what a list of questions looks like as text. The
capability is there; you can get at it with few-shot prompting. What post-training adds is the
interface: SFT selects the answering behaviour, preference optimisation shapes it, RL improves
outcomes on tasks. But none of it installs knowledge the base doesn't have, which is why a
hallucinating fine-tune needs retrieval rather than more SFT.

**"Explain Chinchilla, and the way people get it wrong."**
For a fixed *training* compute budget, parameters and tokens should scale roughly equally — about
20 tokens per parameter — and Chinchilla at 70B/1.4T beat Gopher at 280B/300B on the same compute
to demonstrate it. Where people go wrong is treating that as the optimal model to *deploy*. If
you're serving millions of requests, inference cost dominates lifetime cost, so it's rational to
train a smaller model far past 20 tokens per parameter — Llama 3 8B saw about 15T tokens, roughly
2,000 per parameter, around 100× "optimal". That's not a mistake, it's trading training compute for
serving cost.

**"Would you continue-pretrain for a domain?"**
Only if the domain needs new *knowledge* or genuinely different language — clinical text, legal
text, a large proprietary codebase. If it needs different behaviour, SFT is cheaper and better. If
it needs current facts, retrieval is the right answer. When CPT is genuinely warranted: a learning
rate around 10× below pretraining, 5–30% of the original distribution mixed in or you get
forgetting far worse than at SFT scale, and it goes on the *base* model with post-training redone
afterwards — continuing pretraining on an instruct model damages instruction-following, and that's
the mistake I'd expect to see.

**"Are emergent capabilities real?"**
Partly, and less than the plots suggest. A lot of the sharpness is a metric artefact: exact-match
accuracy on multi-step arithmetic is a step function by construction, since four correct digits out
of five scores zero. Measure the same models with a continuous metric — mean log-probability of the
correct answer — and the curve is smooth. So my position is that capability improves smoothly and
discrete metrics make it look sudden. Some genuinely discontinuous behaviour may survive that, but
I'd want to see the continuous metric before believing an emergence claim.

**"What would it cost us to pretrain our own 8B?"**
Roughly 6 × N × D FLOPs, so 8B parameters over 15T tokens is about 7×10²³ FLOPs. On H100s at
990 TFLOPS bf16 and a realistic 40% MFU that's on the order of half a million GPU-hours — a few
weeks on 512 GPUs, low seven figures at commodity rental rates, before failed runs and data work,
which in practice are most of the cost. That's why almost everyone starts from an open base and
spends the budget on post-training and data instead, where the marginal return per dollar is far
higher.

---

## Checkpoint

1. Compute a training run's cost from parameters and tokens.
2. Reproduce a scaling-law line from your own tiny models.
3. Show a base model failing to answer and then complying under few-shot.
4. Report the deduplication effect on held-out loss and memorisation.
5. Report the CPT replay-fraction table.
6. Show the same capability as a step function and a smooth curve.

---

**Next:** [07 — Architecture variants](07-architectures.md) ·
**Back:** [05a — The encoder–decoder transformer](05a-encoder-decoder.md) · [Syllabus](../SYLLABUS.md)
