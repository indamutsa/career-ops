# Module 14 — Preference optimization

How you get from "imitates demonstrations" to "produces output people prefer".

**Why it matters for this role:** the JD says *"materially better at complex decision-making."* SFT
cannot do that — it can only imitate. This module and the next are where that capability comes from.

---

## Terms

| Term | Meaning |
|------|---------|
| **RLHF** | Reinforcement Learning from Human Feedback. The three-stage pipeline: SFT → reward model → RL. |
| **Reward model (RM)** | A model trained on human preference pairs to score outputs. |
| **Bradley-Terry** | The statistical model turning pairwise comparisons into scalar scores. `P(a ≻ b) = σ(r_a − r_b)`. |
| **Preference pair** | `(prompt, chosen, rejected)`. The unit of preference data. |
| **PPO** | Proximal Policy Optimization. Actor-critic RL with a clipped objective. |
| **Policy** | The model being trained, viewed as a distribution over actions. |
| **Value head / critic** | A head predicting expected future reward from a state. Used as the baseline in PPO. |
| **Advantage** | `reward − baseline`. How much better an action was than expected. |
| **GAE** | Generalized Advantage Estimation. Variance-reduced advantage over multiple steps. |
| **Clipping** | Limiting how far the policy ratio may move in one update. PPO's stability mechanism. |
| **KL penalty** | Loss term penalising divergence from the reference (pre-RL) model. |
| **Reference model** | A frozen copy of the starting policy. The anchor for the KL term. |
| **DPO** | Direct Preference Optimization. Preference learning with no reward model and no RL loop. |
| **Implicit reward** | In DPO, `β log(π/π_ref)` — the reward the loss behaves as if it were optimising. |
| **Beta (β)** | DPO's KL-strength knob. Low β = more drift from reference; high β = more conservative. |
| **ORPO** | Odds-Ratio Preference Optimization. SFT and preference in one stage, no reference model. |
| **KTO** | Kahneman-Tversky Optimization. Learns from unpaired thumbs-up/down labels. |
| **SimPO** | Reference-free, length-normalised preference optimization. |
| **IPO** | Identity Preference Optimization. Fixes a DPO overfitting pathology on deterministic preferences. |
| **Length bias** | Preference models and judges systematically favour longer answers. |
| **Alignment tax** | Capability lost on benchmarks as a side effect of alignment. |
| **Reward overoptimization** | Policy exploits reward-model error; true quality falls while measured reward rises. |

---

## Concepts

### Why SFT is not enough

SFT maximises the likelihood of demonstration tokens. Three consequences:

1. **It cannot express "worse".** Every training token is a positive example. There is no mechanism
   to say "this output was fluent but wrong".
2. **It is bounded by the demonstrator.** The model converges to the demonstration distribution.
   If the demonstrations are mediocre, so is the model.
3. **No credit assignment.** Every token in the demonstration carries the same weight. A trajectory
   that was correct for 11 steps and fatal on the 12th trains all 12 equally.

Preference learning fixes (1) and (2). Only RL on trajectories fixes (3) — that is Module 15.

### The classic RLHF pipeline

```
1. SFT             base model + demonstrations           -> a model that follows instructions
2. Reward model    preference pairs, Bradley-Terry loss  -> a scorer
3. RL (PPO)        maximise RM score - β·KL(π ‖ π_ref)   -> the aligned policy
```

Reward-model loss:

```
L_RM = −log σ( r(x, y_chosen) − r(x, y_rejected) )
```

Note it only ever sees *differences*. The reward scale is arbitrary — which is exactly why the KL
term is essential: without an anchor, the policy walks off into whatever region the RM scores
highest, including gibberish.

### Why PPO is painful

You must hold **four models** in memory: policy, reference, reward model, value head. It is
sensitive to hyperparameters, expensive, and awkward to debug. That cost is the reason DPO
took over.

### DPO — the key insight

The RLHF objective has a closed-form optimal policy:

```
π*(y|x)  ∝  π_ref(y|x) · exp( r(x,y) / β )
```

Invert it: the reward is recoverable from the optimal policy.

```
r(x,y) = β log( π*(y|x) / π_ref(y|x) ) + const
```

Substitute that into the Bradley-Terry loss and the reward model **disappears**. What remains is a
supervised loss over preference pairs:

```
L_DPO = −log σ( β [ log π(y_c|x)/π_ref(y_c|x)  −  log π(y_r|x)/π_ref(y_r|x) ] )
```

Two models instead of four, no RL loop, no sampling during training. Same theoretical objective.

**Read the loss.** It raises the log-prob of chosen *relative to* rejected, both measured against
the reference. It is a *contrastive* objective, not a likelihood objective — which is why DPO can
lower the absolute probability of the chosen response while still reducing the loss.

### The DPO failure that surprises people

Both chosen and rejected log-probs often fall during training. The *margin* grows — the loss is
satisfied — but the model is becoming less likely to produce either. Push far enough and you get
degenerate output that is technically preferred over the rejected sample.

**Always log both log-probs separately, not just the margin.** If both are falling steeply, raise
β or stop earlier. This is the single most useful thing to know about running DPO, and a strong
interview answer.

### The variants, and when each is right

| Method | Ref model? | Data needed | Pick it when |
|--------|-----------|-------------|--------------|
| **PPO** | yes + RM + critic | pairs → RM | You need online exploration and can afford the complexity |
| **DPO** | yes | pairs | Default. Offline pairs, simple, well understood |
| **IPO** | yes | pairs | Preferences are near-deterministic and DPO overfits |
| **ORPO** | **no** | pairs | You want one stage from base model — no separate SFT |
| **KTO** | yes | **unpaired** labels | You have thumbs up/down, not comparisons — i.e. real product feedback |
| **SimPO** | **no** | pairs | Length bias is your problem; you want no reference model in memory |

**KTO deserves attention for enterprise work.** Production feedback is almost never paired — users
click 👍 or 👎 on one response. Constructing pairs from that is lossy. KTO consumes the signal you
actually have.

### Length bias

Human raters and LLM judges both prefer longer answers, roughly independent of quality. The reward
model learns "longer = better", the policy learns to pad, and your measured reward rises while
usefulness falls.

Detection: correlate reward against response length on a held-out set. A strong positive
correlation means your RM learned length, not quality.

Mitigations: length-normalise (SimPO does this in the objective), balance lengths in the preference
data, or add an explicit length penalty. Do not skip the detection step — this is present by
default, not occasionally.

---

## Where it's used

- **Every deployed assistant.** Instruct models are SFT + preference optimization.
- **Domain agents.** Preference data from your own domain experts is often the highest-value
  data an enterprise has, and it is unpaired — hence KTO.
- **Safety.** Refusal behaviour is trained here, and the helpfulness/harmlessness trade-off is
  tuned here.
- **Style and format.** Response length, tone, structure — usually preference-trained, not prompted.

---

## Labs

### Lab 1 — Build preference pairs from your own model

You need pairs before you can train. Generate them with the verifier you already have.

```python
"""Turn verified rollouts into preference pairs: correct = chosen, incorrect = rejected."""
import json, random, itertools, collections

records = [json.loads(l) for l in open("data/baseline.jsonl")]
by_task = collections.defaultdict(list)
for r in records:
    by_task[r["task"]].append(r)

pairs = []
for task, rows in by_task.items():
    good = [r for r in rows if r["passed"] and r["sql"]]
    bad = [r for r in rows if not r["passed"] and r["sql"]]
    for c, rj in itertools.product(good, bad):
        pairs.append({"task": task, "chosen": c["sql"], "rejected": rj["sql"]})

random.shuffle(pairs)
print(f"{len(pairs)} pairs from {len(by_task)} tasks")
print(f"tasks with both outcomes: {sum(1 for t, r in by_task.items() if any(x['passed'] for x in r) and any(not x['passed'] for x in r))}")
```

**Note what just happened:** tasks where every sample passed, or every sample failed, produce **zero
pairs**. Only the mixed-outcome band is usable. That is the same 20–60% constraint that governs
GRPO, showing up one module early — and it is not a coincidence. Both methods learn from
*contrast*, and a task with no variance contains none.

### Lab 2 — DPO by hand

Implement the loss before using a library. It is six lines and understanding it is the point.

```python
import torch
import torch.nn.functional as F


def dpo_loss(pi_chosen_lp, pi_rejected_lp, ref_chosen_lp, ref_rejected_lp, beta=0.1):
    """All args: (batch,) summed log-probs of the response tokens."""
    pi_logratio = pi_chosen_lp - pi_rejected_lp
    ref_logratio = ref_chosen_lp - ref_rejected_lp
    logits = beta * (pi_logratio - ref_logratio)
    loss = -F.logsigmoid(logits).mean()

    chosen_reward = beta * (pi_chosen_lp - ref_chosen_lp).detach()
    rejected_reward = beta * (pi_rejected_lp - ref_rejected_lp).detach()
    return loss, chosen_reward, rejected_reward


# Sanity: identical policy and reference -> logits 0 -> loss = -log(0.5)
z = torch.zeros(4)
loss, _, _ = dpo_loss(z, z, z, z)
print(loss.item(), "expected", -torch.log(torch.tensor(0.5)).item())

# Policy already prefers chosen more than reference does -> lower loss
loss2, cr, rr = dpo_loss(torch.tensor([2.0]), torch.tensor([-2.0]),
                         torch.tensor([0.0]), torch.tensor([0.0]))
print(f"loss={loss2.item():.4f}  chosen_r={cr.item():.3f}  rejected_r={rr.item():.3f}")
```

Then sweep β and see what it controls:

```python
for beta in [0.01, 0.1, 0.5, 1.0]:
    l, _, _ = dpo_loss(torch.tensor([1.0]), torch.tensor([-1.0]),
                       torch.tensor([0.0]), torch.tensor([0.0]), beta=beta)
    print(f"beta={beta:<5} loss={l.item():.4f}")
```

**Observe:** larger β makes the same policy-reference gap produce a much lower loss — the gradient
saturates sooner, so the policy moves less. β is the KL leash.

### Lab 3 — Run DPO and watch it go wrong

```python
"""DPO with TRL, logging the diagnostics that matter."""
from datasets import Dataset
from trl import DPOConfig, DPOTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch, json

MODEL = "Qwen/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16)

pairs = [json.loads(l) for l in open("data/pairs.jsonl")]
ds = Dataset.from_list([
    {"prompt": p["task"], "chosen": p["chosen"], "rejected": p["rejected"]}
    for p in pairs
])

cfg = DPOConfig(
    output_dir="out/dpo",
    beta=0.1,
    learning_rate=5e-6,          # 10-50x lower than SFT. DPO overfits fast.
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    logging_steps=5,
    bf16=True,
    report_to="none",
)

trainer = DPOTrainer(model=model, args=cfg, train_dataset=ds, processing_class=tok)
trainer.train()
```

**Watch these four in the logs, not just `loss`:**

| Metric | Healthy | Trouble |
|--------|---------|---------|
| `rewards/margins` | rising | flat = no learning |
| `rewards/chosen` | flat or slightly rising | **falling steeply = degeneration** |
| `rewards/rejected` | falling | — |
| `rewards/accuracies` | rising toward ~0.7–0.9 | pinned at 1.0 early = overfitting |

Now **induce the failure deliberately**: set `beta=0.01` and `num_train_epochs=3`. Watch
`rewards/chosen` collapse alongside `rewards/rejected`. Generate from the resulting model — it will
be degraded. You have now seen the failure that most people only read about, and you can describe
it from experience.

### Lab 4 — Detect length bias

```python
import json
from scipy.stats import pearsonr

pairs = [json.loads(l) for l in open("data/pairs.jsonl")]
c = [len(p["chosen"]) for p in pairs]
r = [len(p["rejected"]) for p in pairs]

longer = sum(1 for a, b in zip(c, r) if a > b)
print(f"chosen longer in {longer}/{len(pairs)} = {longer/len(pairs):.1%}")
print(f"mean chosen {sum(c)/len(c):.0f} chars, rejected {sum(r)/len(r):.0f}")
```

**If "chosen longer" is far from 50%, your preference data has length baked in.** Whatever you
train on it will learn length as a proxy for quality. Fix the data or normalise in the objective;
do not proceed and hope.

---

## Failure modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| Both chosen and rejected log-probs falling | β too low, or too many epochs | Raise β, stop earlier, use IPO |
| Reward rises, held-out quality falls | Reward overoptimization | Stronger KL, early stop against a held-out set |
| Outputs get longer and worse | Length bias in preference data | Length-balance, or use SimPO |
| Model becomes bland and repetitive | Over-alignment / mode collapse | Lower training strength; check diversity metrics |
| Benchmarks drop after alignment | Alignment tax | Expected to a degree; keep a capability regression suite |
| DPO does nothing | Pairs too easy or too similar | Check reward accuracy — near 1.0 at step 0 means no signal |
| Zero usable pairs | All tasks fully passed or fully failed | Same variance problem as GRPO — fix task difficulty |

---

## Interview

**"Walk me through DPO versus PPO."**
PPO is the RLHF pipeline: train a reward model on preference pairs, then optimise the policy against
it with a KL leash to the reference. Four models in memory, an online sampling loop, hyperparameter
sensitive. DPO uses the closed-form optimal policy of that same objective to express the reward in
terms of the policy itself, which cancels the reward model out — leaving a contrastive supervised
loss on preference pairs. Two models, no sampling, far easier to run. The trade: DPO is offline, so
it only ever sees the pairs you collected. PPO can explore. That is exactly why GRPO — on-policy,
but critic-free — became the choice for agentic tasks.

**"How would you know DPO is going wrong?"**
Log chosen and rejected rewards separately, not just the margin. The characteristic failure is both
falling: the margin grows, the loss drops, and the model is becoming less likely to produce either
response. Also check reward accuracy — pinned at 1.0 within a few hundred steps means overfitting —
and correlate reward with length to catch length bias.

**"You have thumbs up/down from production. How do you use it?"**
KTO. Production feedback is unpaired, and manufacturing pairs from it — pairing a random 👍 against
a random 👎 on a different prompt — introduces confounds. KTO takes the unpaired signal directly.
I would also check the feedback distribution first: it is usually heavily skewed toward 👎, since
satisfied users rarely click, and that skew needs handling.

**"What is the alignment tax and do you care?"**
Capability loss on benchmarks from alignment training — some real, some an artefact of aligned
models refusing benchmark-style prompts. I care insofar as I keep a capability regression suite
running alongside alignment evals, so a helpfulness gain that costs 5 points of domain accuracy is
a decision someone makes rather than something we discover later.

---

## Checkpoint

1. Derive why the reward model cancels in DPO.
2. State what β controls and what happens at both extremes.
3. Name the diagnostic that catches DPO degeneration, and why the margin alone hides it.
4. Choose between DPO, KTO and ORPO given a described dataset.
5. Detect length bias in a preference set in three lines of code.

---

**Next:** [15 — RL with verifiable rewards](15-rlvr-grpo.md) ·
**Back:** [13 — Datasets](13-datasets.md) · [Syllabus](../SYLLABUS.md)
