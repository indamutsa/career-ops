# B8 — Scaling laws on models you trained

Scaling laws are usually taught as quotations: "Chinchilla says 20 tokens per parameter." That is
worth very little in an interview, because everyone can recite it and almost no one can say what was
actually measured, what the exponents mean, or why the number changed.

You have three trained models. In this lesson you fit a power law to your own data, extrapolate it,
and then test the extrapolation by training a fourth model — which is the entire scientific content
of the scaling-laws literature, at a size where you can run it in an afternoon.

Reference companion: [Module 06](../06-pretraining.md).

---

## Terms

| Term | Meaning |
|------|---------|
| **Scaling law** | Loss as a smooth power-law function of scale. |
| **`N`** | Non-embedding parameter count. |
| **`D`** | Training tokens seen. |
| **`C`** | Compute, `≈ 6ND` FLOPs. |
| **Power law** | `L = A · N^(−α) + E`. Straight line on log-log axes. |
| **Irreducible loss `E`** | The entropy of the data. No model beats it. |
| **Compute-optimal** | The `(N, D)` split minimising loss at fixed `C`. |
| **Kaplan (2020)** | Concluded model size matters most. Held LR schedule fixed. |
| **Chinchilla (2022)** | Corrected it: scale `N` and `D` together, ≈20 tokens/param. |
| **Over-training** | Deliberately exceeding 20:1 to make inference cheaper. |
| **IsoFLOP** | Curve of loss vs `N` at fixed compute. Has a visible minimum. |
| **Emergence** | Capability appearing sharply with scale — often a metric artifact. |

---

## Concepts

### The empirical fact

Across many orders of magnitude, test loss follows

```
L(N) = A · N^(−α) + E
```

and the same form holds in `D` and in `C`. On log-log axes it is a **straight line** — and the
remarkable part is that this holds across seven orders of magnitude of compute, well enough that
labs commit hundred-million-dollar training runs to its extrapolation.

Three components, each meaning something distinct:

- **`E`** — irreducible loss, the entropy of the data itself. A perfect model still cannot predict
  genuinely random continuations. It is why loss curves flatten rather than reaching zero.
- **`α`** — the exponent, typically ~0.05–0.1 for `N` in language modelling. **Small exponents mean
  brutal returns:** at α ≈ 0.07, a 10× larger model reduces the reducible loss by only about 15%.
- **`A`** — a scale constant, set by the architecture and data.

Two things this predicts that are worth saying out loud, because they explain the industry's
behaviour: progress requires *exponential* investment for linear gains, and the curve is smooth, so
"we tried it small and it didn't work" is only evidence if it lands on the line.

### Kaplan vs Chinchilla — the part people get wrong

Kaplan et al. (2020) concluded that at a fixed compute budget you should spend most of it on
parameters and train on relatively few tokens. GPT-3 followed this: 175B parameters, 300B tokens —
about 1.7 tokens per parameter.

Hoffmann et al. (2022, Chinchilla) redid the analysis and got a different answer: `N` and `D` should
scale **roughly equally**, around 20 tokens per parameter. Chinchilla, at 70B parameters and 1.4T
tokens, beat the 280B Gopher trained on the same compute.

**Why the disagreement, which is the actual interview question:** Kaplan's sweeps used a *fixed*
learning-rate schedule rather than one matched to each run's length. As you saw in B6, a cosine
schedule stopped early leaves the model mid-bounce at a high LR, so the shorter-trained (larger)
models were systematically under-evaluated in the wrong direction. Correct the schedule per run and
the optimum moves.

The methodological lesson is worth more than the number: **an experimental artifact in a
hyperparameter protocol shifted the entire industry's model sizes for two years.**

### Why nobody trains Chinchilla-optimal any more

Chinchilla minimises training loss for a training budget. It says nothing about inference.

If a model is served to millions, inference cost dwarfs training cost, and a *smaller* model trained
far past 20:1 is cheaper for the whole lifecycle even though it cost more to train. Llama 3 8B saw
15T tokens — about **1875 tokens per parameter**, roughly 90× Chinchilla.

**So "Chinchilla-optimal" is the answer to a question that is no longer the one being asked.** Being
able to say that cleanly, with the Llama 3 ratio as evidence, is a much stronger answer than reciting
20:1.

### What emergence is and isn't

Some capabilities appear to arrive suddenly with scale. The 2023 counter-argument (Schaeffer et al.)
is that this is often an artifact of the **metric**: exact-match accuracy on a multi-step task is a
step function over a continuously improving underlying quantity. Measure per-token log-probability of
the correct answer instead, and the "emergent" jump becomes a smooth curve.

Not all claimed emergence dissolves this way, but enough does that the right posture is: **check
whether the metric is discontinuous before concluding the capability is.** You can demonstrate this
on your own three models in Lab 6.

---

## Where it's used

- Deciding model size and token budget before committing compute.
- Predicting the final loss of a large run from small pilot runs — routine practice.
- Justifying a training budget: the fitted curve is the argument.
- Diagnosing a run: a large model landing *above* your fitted line means something is wrong, and the
  line tells you before the run finishes.

---

## Labs

### Lab 1 — Assemble the dataset

From B6 you have three points. Record them precisely:

| Model | `N` (non-emb) | `D` tokens | `C = 6ND` | Final val loss |
|-------|---------------|------------|-----------|----------------|
| `tiny` d=128 L=4 | ~0.4M | ? | ? | ? |
| `base` d=192 L=6 | ~2.7M | ? | ? | ? |
| `wide` d=320 L=8 | ~9.8M | ? | ? | ? |

**Use non-embedding parameters.** Embeddings scale with vocabulary, not with capacity, and including
them at these sizes distorts the fit badly — at 3.4M they are 23% of the total. This is the single
most common mistake when people fit their own scaling curves.

All three runs must use the **same data, same tokenizer, and a schedule matched to each run's
length.** Otherwise you are fitting Kaplan's bug.

### Lab 2 — Fit the power law

```python
import numpy as np
from scipy.optimize import curve_fit

def power_law(N, A, alpha, E):
    return A * N**(-alpha) + E

N = np.array([...]); L = np.array([...])
(A, alpha, E), _ = curve_fit(power_law, N, L, p0=[10.0, 0.1, 1.0], maxfev=20000)
print(f"L(N) = {A:.3f} · N^(-{alpha:.4f}) + {E:.3f}")
```

Then plot on log-log axes with the fit line.

**Three points is a thin fit and `E` will be poorly constrained** — say so rather than pretending
otherwise. That honesty is itself the lesson: the published laws use dozens of runs across orders of
magnitude, and knowing why your fit is weak is knowing what makes theirs strong. Add a fourth and
fifth model if you have the time; the fit tightens visibly.

### Lab 3 — Predict, then test

This is the lab that matters. Everything else is curve-fitting.

1. Use your fit to **predict** the val loss of a `d_model=256, n_layers=7` model (~6M params) at the
   matched token budget. Write the number down **before training.**
2. Train it.
3. Compare.

**Within ~5% is a good result at this scale.** Then state what you have actually done: predicted the
outcome of a training run you had not performed, from a law fitted to smaller runs. That is exactly
what a lab does before committing to a large run, and it is a far more credible thing to describe in
an interview than any citation.

If the prediction is badly off, the diagnosis is usually one of: embeddings included in `N`, an LR
schedule not matched to run length, or the largest model being data-limited rather than
capacity-limited.

### Lab 4 — Find your compute-optimal ratio

Fix a compute budget `C` and train several `(N, D)` splits that all consume it: a small model on many
tokens, a large model on few, and points between. Plot loss against `N` at fixed `C`.

**You should see a U — an IsoFLOP curve with a visible minimum.** That minimum is your empirical
compute-optimal point, and the `D/N` ratio at it is your Chinchilla number.

Expect it to disagree with 20. At this scale, with a 4096-token vocabulary and a narrow corpus, that
is the correct outcome and not a failed lab. **Say why:** the constant depends on data quality, data
diversity, vocabulary, and architecture. The *form* transfers across scales; the *constant* does not.

### Lab 5 — Over-training, and the inference argument

Take `tiny` and train it to 10× its Chinchilla token budget.

- Loss keeps improving, with clearly diminishing returns.
- Now compute inference cost for `tiny`-overtrained vs `base` at equal loss: parameters, KB of KV
  cache at length 256, and measured tokens/sec from B7.

**The over-trained small model wins on every inference metric at equal quality.** You have now
derived Llama 3's training decision from your own measurements — that is the whole argument for
over-training, and it is much stronger stated as a result than as a citation.

### Lab 6 — Emergence as a metric artifact

Define a task your models can partly do — say, "the story's protagonist name is consistent between
its first and last mention."

Score all three models two ways:

1. **Exact match** — binary per sample, averaged.
2. **Log-probability** of the correct continuation.

**Expect exact-match to look like a step and log-probability to look like a smooth curve, over the
same three checkpoints.** That is Schaeffer et al.'s argument, reproduced on models you trained. It
is one of the most compelling things you can bring to an interview about scaling, because almost
nobody has run it themselves.

### Lab 7 — Extrapolate to absurdity, and find where it breaks

Use your fit to predict the loss of a 1B-parameter model on your corpus.

The number will be implausibly low. **Explain why the extrapolation is invalid**, and this is the
substance of the lab:

- **You would run out of data.** At 20:1 a 1B model wants 20B tokens; TinyStories has ~500M. The law
  assumes fresh data, and repeating it changes the regime.
- **`E` is fitted, not measured.** With three points it is barely constrained, and it is the term
  that dominates at scale.
- **The corpus caps the achievable loss.** TinyStories' irreducible entropy is low but real; a large
  model saturates it and the curve flattens for reasons the fit never saw.

Knowing where a scaling law stops applying is the difference between using one and quoting one.

---

## Failure modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| Fit is poor / `E` absurd | Too few points; `E` unconstrained | More models; fix `E` to a plausible value and refit |
| Points don't fall on a line | Embeddings included in `N` | Use non-embedding parameters |
| Larger model does worse | LR not retuned; `lr ∝ 1/√d_model` | Scale the LR down with size |
| Largest model plateaus early | Data-limited, not capacity-limited | More tokens, or acknowledge the regime |
| Your ratio ≠ 20 | Different data, vocab, architecture | Expected. The form transfers, not the constant |
| Prediction off by 20%+ | Schedule not matched per run | Rerun with per-run cosine length |
| "Emergence" claimed | Discontinuous metric | Re-score with log-probability |
| Extrapolation looks amazing | Outside the fitted range | State the data limit and the `E` uncertainty |

---

## Interview

**"What are scaling laws and why do they matter?"**
Test loss follows a power law in parameters, data and compute — `L = A·N^(−α) + E` — which is a
straight line on log-log axes and holds across many orders of magnitude. It matters because it makes
large training runs predictable: you fit the curve on small pilot runs and extrapolate to decide
whether a budget is worth committing. The exponent is small, around 0.05–0.1, which is the part
people underrate — it means a 10× bigger model buys maybe 15% of the reducible loss, so progress
costs exponentially more for linear gains. I fitted this on three models I trained myself, then
predicted a fourth model's loss before training it and landed within a few percent.

**"Kaplan versus Chinchilla — what changed?"**
Kaplan concluded you should spend compute mostly on parameters, and GPT-3 followed that at about 1.7
tokens per parameter. Chinchilla redid the analysis and found `N` and `D` should scale roughly
together, around 20 to 1, and demonstrated it by beating the 280B Gopher with a 70B model at equal
compute. The reason for the disagreement is the interesting bit: Kaplan used a fixed learning-rate
schedule instead of one matched to each run's length, so shorter runs were evaluated mid-cosine at a
high LR and systematically penalised. A hyperparameter protocol artifact set the industry's model
sizes for two years.

**"So should I train Chinchilla-optimal?"**
Usually not. Chinchilla minimises loss for a *training* budget and ignores inference entirely. If
you're serving a model at volume, inference dominates lifetime cost, and a smaller model trained well
past the optimum is cheaper overall at equal quality. Llama 3 8B saw 15T tokens, roughly 1875 tokens
per parameter — about 90× Chinchilla. I verified the mechanism on my own models: an over-trained
small model matched a larger one's loss with fewer parameters, a smaller KV cache, and higher
throughput.

**"Is emergence real?"**
Partly, and less often than claimed. Schaeffer et al. showed that many emergent jumps are artifacts
of discontinuous metrics — exact-match accuracy on a multi-step task is a step function over a
smoothly improving quantity, so the jump is in the measurement, not the model. I reproduced this on
my own three checkpoints: scored by exact match a consistency task looks like a step, scored by
log-probability of the correct continuation it's a smooth curve. That doesn't dissolve every case,
but it means the first question is always whether the metric is discontinuous.

**"When does a scaling law stop applying?"**
When you leave the regime it was fitted in. The three that bite: you run out of unique data, so the
law's assumption of fresh tokens breaks and repetition changes the curve; the irreducible term is
fitted rather than measured, and it's the term that dominates at scale, so an under-constrained `E`
makes long extrapolations meaningless; and the architecture or data distribution changes, which moves
the constants even though the functional form survives. When I extrapolated my own fit to a billion
parameters it predicted a loss far below my corpus's entropy — which is a clean illustration that the
line is only a line inside its range.

---

## Checkpoint

1. Table of `N`, `D`, `C`, and final loss for three models, `N` non-embedding.
2. Fitted `A`, `α`, `E`, with the log-log plot; state honestly how constrained `E` is.
3. A prediction written down *before* training the fourth model, and the measured error.
4. IsoFLOP U-curve with your empirical `D/N`, plus why it differs from 20.
5. Over-training comparison at equal loss: parameters, KV bytes, tokens/sec.
6. Emergence demonstrated as a metric artifact on your own checkpoints.
7. The 1B extrapolation, with three specific reasons it is invalid.

---

**Next:** [B9 — Instruction tuning your own model](09-instruct.md) ·
**Back:** [B7 — Sampling from your own logits](07-sampling.md) · [Build Track](README.md)
