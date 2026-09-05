# Module 16 — Distillation and model compression

Distillation is how a small model learns to behave like a large one. For an applied research team
it is the standard route from "a frontier model can do this" to "we can afford to serve this" —
and it is the technique behind most small models that punch above their size.

It sits after preference optimisation in the syllabus because the most useful modern form of it —
distilling reasoning traces — depends on everything in Modules 13–15.

---

## Terms

| Term | Meaning |
|------|---------|
| **Teacher / student** | The large source model / the small target model. |
| **Knowledge distillation (KD)** | Training the student to match the teacher. |
| **Hard labels** | The teacher's sampled output text. |
| **Soft labels / soft targets** | The teacher's full probability distribution per position. |
| **Dark knowledge** | The information in the teacher's *non-argmax* probabilities. |
| **KD temperature** | Softening applied to both distributions before the KL. Distinct from sampling temperature. |
| **Forward KL** | `KL(teacher ‖ student)` — mode-covering; student spreads mass. |
| **Reverse KL** | `KL(student ‖ teacher)` — mode-seeking; student commits. |
| **Sequence-level KD** | SFT on teacher-generated sequences. The common case. |
| **On-policy distillation** | Student generates; teacher scores the student's own outputs. |
| **GKD** | Generalised KD — on-policy sequences with teacher distributions. |
| **Rationale distillation** | Distilling chain-of-thought, not just answers. |
| **Reasoning distillation** | Training a small model on a reasoning model's traces (r1-style). |
| **Self-distillation** | Teacher and student are the same size or the same model. |
| **Pruning** | Removing weights, heads, or whole layers. |
| **Structured / unstructured pruning** | Removing whole units / individual weights. |
| **Depth pruning / layer dropping** | Removing entire transformer blocks. |
| **Width pruning** | Reducing `d_model`, heads, or `d_ff`. |
| **Healing** | Brief retraining after pruning to recover quality. |

---

## Concepts

### Why soft targets beat hard labels

Train on the teacher's sampled text and the student learns "the answer was token X". Train on the
teacher's *distribution* and the student learns "X at 0.7, Y at 0.2, Z at 0.05, everything else
near zero".

That second signal is far richer. The relative probabilities of the wrong answers encode the
teacher's similarity structure — its uncertainty, and which alternatives it considered plausible.
That is Hinton's **dark knowledge**, and it is why distillation with soft targets converges faster
and to a better student than SFT on the same generated text.

```
L = α · KL(softmax(z_t / T) ‖ softmax(z_s / T)) · T²  +  (1 − α) · CE(y, z_s)
```

The `T²` is not cosmetic: softening by `T` shrinks the gradients by roughly `1/T²`, so the factor
restores their scale and keeps `α` meaningful across temperatures.

**The catch, and it is a big one:** soft targets require the teacher's full vocabulary distribution
per position. Through an API you get, at best, top-k logprobs. So **soft-target distillation
requires an open-weights teacher.** For a company with its own open models — which the JD
describes — that is exactly the setup, and it is worth noting as a real advantage of their
position.

### Forward versus reverse KL

| | Forward `KL(T‖S)` | Reverse `KL(S‖T)` |
|---|---|---|
| Behaviour | **Mode-covering** — student must put mass everywhere the teacher does | **Mode-seeking** — student concentrates on one mode |
| Failure | Student hedges, produces vague averaged output | Student is confident but loses diversity |
| Needs | Teacher probs at teacher-sampled positions | Sampling from the student |
| Use | Standard KD | On-policy / GKD |

The intuition: forward KL punishes the student for assigning low probability where the teacher
assigns high — so the student must cover everything, and covering a multi-modal teacher means
sitting between the modes, which is the "vague output" failure. Reverse KL punishes the student for
putting mass where the teacher does not, so it picks one mode and commits.

For instruction-following where there are many valid answers, **reverse KL / on-policy is usually
better** — you want a student that produces one good answer, not the average of several.

### The distribution-mismatch problem, and on-policy distillation

Standard sequence-KD trains the student on the *teacher's* outputs. But at inference the student
generates its own, and it makes mistakes the teacher never made — so it is conditioning on prefixes
it never trained on.

This is exactly Module 11's exposure bias, and the fix is the same: **train on the student's own
outputs.**

On-policy distillation:

1. The **student** generates a sequence.
2. The **teacher** scores that sequence, providing distributions at each position.
3. The student is trained toward the teacher's distributions on its own trajectory.

More expensive — the teacher runs on every student sample — but it fixes the mismatch, and it is
what GKD formalises. For agents, where trajectories are long and errors compound (Module 22), the
gap between off-policy and on-policy distillation is largest.

### Reasoning distillation

The most consequential recent form. Take a strong reasoning model, generate long chain-of-thought
traces for a task set, filter to traces whose final answer is *verified correct*, and SFT a small
model on the full traces.

The finding that surprised people: **small models distilled from a reasoning model's traces
substantially outperform the same small models trained with RL directly.** The reasoning
*structure* transfers, and imitating good reasoning is much easier than discovering it.

The practical implication for your project: **before running GRPO, check whether distilling from a
stronger model gets you there for a fraction of the cost.** That is the same instinct as Module
15's "try rejection-sampling SFT first" and it is the kind of question a research engineer is
supposed to ask before spending a GPU budget.

Note that this is rejection sampling (Module 13) with a stronger model as the generator — the same
pipeline, a different source. And the licence question from Module 13 applies with full force.

### Pruning

| Type | What goes | Speedup | Quality cost |
|------|-----------|---------|--------------|
| Unstructured | Individual weights | None without sparse kernels | Low |
| 2:4 semi-structured | 2 of every 4 weights | ~1.5–2× on Ampere+ | Moderate |
| Head pruning | Whole attention heads | Modest | Low — many heads are redundant |
| Width pruning | `d_ff` / `d_model` slices | Proportional | Moderate |
| **Depth pruning** | Whole layers | **Proportional and real** | Moderate, and recoverable |

**Depth pruning is the one that gives a real speedup for simple engineering.** Layers are not
equally important — the middle layers of a deep model are often surprisingly removable, while the
first and last are not. Remove a contiguous block of middle layers, then **heal** with a short
distillation from the unpruned model as teacher. Recovery is usually most of the loss.

**Always pair pruning with distillation.** Pruning alone leaves quality on the table that a few
hundred healing steps recover almost for free.

### Choosing an approach

| Situation | Approach |
|-----------|----------|
| Open-weights teacher, same tokenizer | Soft-target KD — best signal available |
| API teacher, licence permits | Sequence-level KD on generated text |
| Long generations, agents | On-policy / GKD |
| Reasoning tasks | Distil verified reasoning traces |
| Model too slow, architecture fixed | Depth-prune + heal |
| Model too big for memory | Quantize (Module 18) before pruning |

**Quantization before pruning**, generally: 4-bit weights cost less quality per byte saved than
removing 40% of the layers.

---

## Where it's used

- **Shipping a cheap model** that behaves like an expensive one.
- **Small reasoning models** — the entire r1-distill family.
- **Latency-bound products** where the big model's TPOT misses the SLO.
- **On-device** — distil, then quantize to GGUF (Module 18).
- **Your agent project** — distilling a strong model's trajectories is a serious alternative to RL
  and belongs in the comparison.

---

## Labs

### Lab 1 — Soft-target KD, implemented

```python
import torch, torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

tok     = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
teacher = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B",
                                               torch_dtype=torch.bfloat16).to("mps").eval()
student = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B",
                                               torch_dtype=torch.bfloat16).to("mps")

assert teacher.config.vocab_size == student.config.vocab_size, "same tokenizer required"


def kd_loss(s_logits, t_logits, labels, T=2.0, alpha=0.9):
    s = F.log_softmax(s_logits / T, dim=-1)
    t = F.softmax(t_logits / T, dim=-1)
    kl = F.kl_div(s, t, reduction="batchmean") * (T ** 2)
    ce = F.cross_entropy(s_logits.view(-1, s_logits.size(-1)),
                         labels.view(-1), ignore_index=-100)
    return alpha * kl + (1 - alpha) * ce, kl.item(), ce.item()
```

Train the student on your SFT set with `alpha` at 0.0, 0.5, 0.9 and 1.0 and compare held-out task
accuracy. **`alpha=0` is plain SFT** — that is your baseline, and the gap to `alpha=0.9` is the
value of dark knowledge, measured.

### Lab 2 — See the dark knowledge

```python
ids = tok("The capital of France is", return_tensors="pt").to("mps")
with torch.no_grad():
    logits = teacher(**ids).logits[0, -1].float()

for T in (1.0, 2.0, 4.0):
    p = F.softmax(logits / T, dim=-1)
    top = p.topk(8)
    print(f"T={T}: " + "  ".join(f"{tok.decode(i).strip()!r}:{v:.4f}"
                                 for i, v in zip(top.indices, top.values)))
```

**At T=1 the distribution is nearly a point mass — almost no signal beyond the argmax. At T=4 the
alternatives are visible.** That is precisely why KD uses a temperature: it exposes the structure
that hard labels discard. Print the entropy at each T as the one-number summary.

### Lab 3 — Forward versus reverse KL

```python
import torch

# A bimodal teacher and a unimodal student, in 1D — the clearest possible demonstration.
def demo(direction, steps=2000):
    x = torch.linspace(-5, 5, 200)
    t = 0.5 * torch.exp(-(x - 2) ** 2) + 0.5 * torch.exp(-(x + 2) ** 2)
    t = t / t.sum()
    mu = torch.tensor(0.0, requires_grad=True)
    ls = torch.tensor(0.0, requires_grad=True)
    opt = torch.optim.Adam([mu, ls], lr=0.02)
    for _ in range(steps):
        s = torch.exp(-((x - mu) ** 2) / (2 * torch.exp(ls) ** 2))
        s = s / s.sum()
        loss = (t * (t / (s + 1e-10)).log()).sum() if direction == "forward" \
               else (s * (s / (t + 1e-10)).log()).sum()
        opt.zero_grad(); loss.backward(); opt.step()
    return mu.item(), torch.exp(ls).item()

print("forward KL(T||S):", demo("forward"))     # wide, centred between the modes
print("reverse KL(S||T):", demo("reverse"))     # narrow, sitting on one mode
```

**Forward KL gives a wide distribution centred at 0 — between both modes, covering neither well.
Reverse KL picks one mode and fits it tightly.** That is mode-covering versus mode-seeking, in
twenty lines, and it makes the abstract distinction permanent.

### Lab 4 — On-policy distillation for your agent

```python
# 1. Student generates trajectories on the task set (T=1.0, Module 08)
student_rollouts = generate(student, tasks, n=4, temperature=1.0)

# 2. Teacher scores the STUDENT's trajectories
with torch.no_grad():
    teacher_logits = [teacher(**r["inputs"]).logits for r in student_rollouts]

# 3. Train the student toward the teacher on its own outputs
for r, tl in zip(student_rollouts, teacher_logits):
    sl = student(**r["inputs"]).logits
    loss, _, _ = kd_loss(sl, tl, r["labels"])
    loss.backward()
```

Compare against off-policy KD (student trained on *teacher*-generated trajectories) on held-out
tasks. **Expect on-policy to win, and by more on longer trajectories** — that is exposure bias
appearing again, and having the two curves as a function of trajectory length is a genuinely good
artefact.

### Lab 5 — Reasoning distillation, versus RL

The comparison that matters for your project.

| Run | Method | Cost |
|-----|--------|------|
| A | Base Qwen3-0.6B | — |
| B | SFT on rejection-sampled 0.6B rollouts | Low |
| C | **SFT on verified Qwen3-8B trajectories** | Medium (teacher inference) |
| D | GRPO on the 0.6B | High |
| E | C, then GRPO | Highest |

Same held-out task set, same eval protocol, confidence intervals (Module 23).

**The likely and important finding is that C is close to D at a fraction of the cost, and E beats
both.** Reasoning structure transfers well; imitating good reasoning is easier than discovering it.
Producing that five-row table is one of the strongest single artefacts you can bring to an
interview, because it is a real methodological result rather than a tutorial reproduction.

### Lab 6 — Depth pruning and healing

```python
import copy, torch

def prune_layers(model, drop_idx):
    m = copy.deepcopy(model)
    keep = [l for i, l in enumerate(m.model.layers) if i not in drop_idx]
    m.model.layers = torch.nn.ModuleList(keep)
    m.config.num_hidden_layers = len(keep)
    return m

n = model.config.num_hidden_layers
for name, drop in [("first 4", list(range(4))),
                   ("middle 4", list(range(n//2 - 2, n//2 + 2))),
                   ("last 4", list(range(n - 4, n)))]:
    p = prune_layers(model, drop)
    print(f"drop {name:<10} ppl {perplexity(p, evalset):>10.2f}")
```

**The middle rows will be far less damaging than the first or last.** Then heal the best-pruned
model with 500 steps of KD from the unpruned original as teacher, and report perplexity and task
accuracy before and after healing. The recovery is usually large, which is the argument for never
pruning without healing.

---

## Failure modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| KD no better than SFT | `alpha` too low, or T=1 (no dark knowledge) | Raise T to 2–4, alpha to 0.9 |
| Can't compute the KL | Teacher and student have different tokenizers | Same tokenizer family, or sequence-level KD |
| Student output is vague and hedged | Forward KL mode-covering a multi-modal teacher | Reverse KL / on-policy |
| Student good on teacher data, bad live | Distribution mismatch | On-policy distillation |
| Loss scale changes with temperature | Missing the `T²` factor | Multiply the KL term by `T²` |
| Distillation from an API teacher blocked | Terms forbid training competing models | Check *before* generating (Module 13) |
| Pruned model unusable | Removed early or late layers | Prune the middle; always heal |
| Pruning gave no speedup | Unstructured sparsity without sparse kernels | Structured or depth pruning |
| Student learned the teacher's mistakes | No verification filter | Keep only verified-correct traces |

---

## Interview

**"How would you get a 70B model's behaviour into a 7B?"**
Sequence-level distillation as the baseline: generate with the teacher, filter to verified-correct
outputs, SFT the student. If the teacher is open-weights and shares the tokenizer — which for a
company with its own models it would be — soft-target KD is strictly better, because the teacher's
full distribution carries the relative probabilities of the alternatives it considered, and that
signal is discarded entirely by training on sampled text. For anything with long generations,
especially agents, I'd move to on-policy: the student generates, the teacher scores the student's
own trajectories. Otherwise you get the same exposure-bias problem as SFT — the student is trained
on prefixes it will never produce.

**"Forward or reverse KL?"**
Depends on whether the target is multi-modal. Forward KL is mode-covering: the student is penalised
for putting low probability where the teacher puts high, so it has to cover everything, and
covering a bimodal teacher means sitting between the modes and producing vague averaged output.
Reverse KL is mode-seeking — the student picks one mode and commits. For instruction-following,
where many different answers are valid, reverse KL or on-policy is usually what you want, because
you want one good answer rather than the average of several.

**"Would you distil or run RL?"**
I'd try distillation first and I'd want the comparison rather than an opinion. The r1-distill
results were striking: small models trained on a reasoning model's verified traces substantially
beat the same models trained with RL directly, because reasoning structure transfers and imitating
good reasoning is much easier than discovering it. So the experiment I'd run is base, versus
rejection-sampled self-SFT, versus SFT on a stronger teacher's verified trajectories, versus GRPO,
versus distil-then-GRPO — five rows, same eval, with confidence intervals. My expectation is that
distillation lands close to RL at a fraction of the cost and the combination beats both, but that's
a hypothesis to test rather than assert.

**"What's the constraint on soft-target distillation?"**
You need the teacher's full vocabulary distribution at every position, which means an open-weights
teacher — an API gives you top-k logprobs at best, and often not even that. You also need a shared
tokenizer, since the distributions have to be over the same vocabulary. That's why distillation is
much more powerful inside a company that trains its own models than for someone consuming an API,
and it's a genuine structural advantage for a team in that position.

**"When would you prune instead?"**
When the architecture is fixed and I need latency rather than memory. Depth pruning is the version
that gives a real speedup for simple engineering — remove a contiguous block of middle layers,
which are far less critical than the first and last, then heal with a few hundred steps of
distillation from the unpruned model as teacher. Recovery is usually most of the quality loss.
Unstructured pruning is mostly pointless without sparse kernels. And if the constraint is memory
rather than latency I'd quantize first — 4-bit weights cost less quality per byte saved than
removing 40% of the layers.

---

## Checkpoint

1. Implement the KD loss, including the `T²` factor, and explain why it's there.
2. Show the dark knowledge appearing as `T` rises; report entropy at each.
3. Reproduce mode-covering vs mode-seeking and state which you'd use where.
4. Report on-policy versus off-policy KD as a function of trajectory length.
5. Produce the five-row distillation-versus-RL table with confidence intervals.
6. Report layer-pruning sensitivity by position and the healing recovery.

---

**Next:** [17 — Distributed training](17-distributed.md) ·
**Back:** [15a — DeepSeek-R1 and the RL turn](15a-deepseek-r1.md) · [Syllabus](../SYLLABUS.md)
