# Module 15a — DeepSeek-R1 and the RL turn

Module 15 taught GRPO as a technique. This module is about where it came from and what it changed,
because "how did DeepSeek change post-training?" is a question you will be asked by a team building
an applied research capability, and the difference between a good and a bad answer is precision
about **what was actually new**.

The lazy answer — "DeepSeek introduced RL to LLMs" — is wrong, and an interviewer who knows the
field will notice. RLHF shipped in 2022. What changed in early 2025 was the *reward*, the *pipeline*,
and the *openness*.

---

## Terms

| Term | Meaning |
|------|---------|
| **RLHF** | RL from human feedback. Learned reward model over human preferences. Aligns behaviour. |
| **RLVR** | RL with verifiable rewards. Programmatic reward — a checker, not a model. Improves capability. |
| **GRPO** | Group Relative Policy Optimization. From **DeepSeekMath (2024)**, not R1. |
| **R1-Zero** | A base model trained with GRPO and **no SFT at all**. |
| **R1** | The shipped model: cold-start SFT → reasoning RL → rejection sampling → final RL. |
| **Cold-start SFT** | A small, high-quality SFT run before RL, to fix format and readability. |
| **Emergent CoT** | Chain-of-thought appearing without being trained for, because it raises reward. |
| **Response-length growth** | Output length rising over RL steps on its own. The signature result. |
| **"Aha moment"** | Self-correction mid-trace ("wait, let me reconsider") appearing unprompted. |
| **Language mixing** | Traces switching between languages — R1-Zero's characteristic defect. |
| **Language-consistency reward** | An auxiliary reward term penalising that mixing. |
| **Rejection sampling** | Generate many, keep only verified-correct, use as SFT data (Module 13). |
| **Reasoning distillation** | SFT'ing small models on R1's traces (Module 16). |
| **Test-time compute** | Spending more tokens at inference to get better answers. |

---

## Concepts

### The lineage, stated precisely

| | RLHF (2022) | RLVR (2025) |
|---|---|---|
| Reward | Learned model over human preferences | **Programmatic checker** |
| Optimises | Helpfulness, harmlessness, style | **Correctness** |
| Ceiling | The reward model's judgement | The verifier's coverage |
| Hackable by | Sycophancy, length, formatting | Verifier gaps |
| Effect | Alignment | **Capability** |

RLHF made models pleasant. RLVR made them *better at things that can be checked*. That is the axis
that moved, and it is why the technique lands on maths, code, and — directly relevant to your
project — SQL, where a result set can be compared to gold.

**GRPO itself predates R1.** It was introduced in the DeepSeekMath paper in 2024 as a critic-free
alternative to PPO: sample a group, use the group mean as the baseline, drop the value model
(Module 15). Attributing GRPO to R1 is a small error that signals second-hand knowledge.

### R1-Zero: the actually surprising result

Take a **base** model — no SFT, no instruction tuning, no chat template — and run GRPO directly,
with a reward that is essentially:

```
reward = accuracy_reward(answer)  +  format_reward(uses <think>…</think>)
```

Prevailing practice said this shouldn't work. SFT was understood as the necessary step that makes a
base model follow instructions at all, and RL as a refinement on top of it.

Three things happened instead:

- **Reasoning ability rose substantially** on maths benchmarks, from RL alone.
- **Response length grew over training without being rewarded for length.** The model discovered
  that thinking longer produced correct answers more often, and that showed up as a monotonically
  rising length curve. This is the plot everyone reproduces, and it is genuinely striking because
  nothing in the reward mentions length.
- **Self-correction emerged** — traces containing "wait, that's not right, let me reconsider",
  never demonstrated in any training example.

The honest framing, and the one that reads as competent: **the capability was latent in the base
model from pretraining; RL didn't install reasoning, it selected for deploying it.** That is
consistent with Module 11's "SFT selects behaviour, it doesn't teach knowledge" and with the
observation that RLVR gains are largest where the base model already had some pass@k.

**And R1-Zero was not shippable.** Its traces mixed languages mid-sentence, had poor readability,
and formatted inconsistently. This matters because it explains the entire structure of R1: the Zero
result is a scientific finding, not a product.

### R1: the four-stage pipeline, and the fact that you already know every stage

| Stage | What | Why | Module |
|-------|------|-----|--------|
| 1 | **Cold-start SFT** on a few thousand curated long-CoT examples | Fix readability and format so RL starts from a sane policy | 11 |
| 2 | **Reasoning RL** (GRPO) + a language-consistency reward | The capability gain, without the language mixing | 15 |
| 3 | **Rejection sampling** from stage 2 → ~800k SFT examples, plus general data | Broaden beyond reasoning; recover writing and general tasks | 13 |
| 4 | **Final RL** over both verifiable and preference rewards | Helpfulness and harmlessness on top of capability | 14, 25 |

Two observations worth making in an interview:

**Stage 1 exists because of stage 2's failure mode, not because SFT was needed for capability.** The
Zero run proved SFT wasn't required to get reasoning. Cold-start SFT is there to make the output
usable and to give RL a better starting policy — a small amount of data doing a specific job.

**Stage 3 is rejection-sampling SFT — the "cheap baseline" Module 15 tells you to beat before
committing to RL.** Here it appears *inside* the pipeline, using the RL'd model as the generator.
Capability is created by RL, then consolidated back into SFT data, then polished. That
RL → distil-into-SFT → RL loop is the structural idea worth carrying, more than any single
hyperparameter.

### The distillation result

R1's traces were used to SFT small dense models (1.5B–70B). Those distilled models **beat the same
small models trained with RL directly**, at a fraction of the cost.

This is Module 16's Lab 5, and its implication is the practical one for a team with a budget: *if a
stronger model already reasons well, distilling its verified traces is usually the better first
move.* Run the comparison before spending on RL.

### What was new, and what wasn't

Be able to separate these cleanly — it is the whole test.

**Not new:** RL on language models (RLHF, 2022). Chain-of-thought (2022). Process/outcome reward
distinctions. Test-time compute scaling (o1 had shipped months earlier). GRPO itself
(DeepSeekMath, 2024).

**New, or newly demonstrated:**

- **Pure RL from a base model works** — SFT is not a prerequisite for reasoning.
- **Long CoT emerges from an outcome reward alone**, including self-correction and length growth.
- **A complete, reproducible recipe published openly**, with open weights — which is why the entire
  field reproduced it within weeks and why GRPO is now in TRL, verl, and OpenRLHF.
- **Reasoning distils into small models better than RL trains them.**

The openness is not a footnote. The reason you can run Module 15's labs on a rented GPU with a
library API is that this recipe was published rather than described.

### Limits, and saying them out loud

- **Verifiable domains only.** It works where a checker exists. Open-ended writing, strategy,
  judgement — no verifier, no RLVR.
- **Reward hacking is still the default** (Module 15, Module 25). A checker with gaps gets gamed.
- **Benchmark contamination** questions apply to every model reporting maths benchmarks, including
  this one; treat published numbers with the same suspicion you'd apply to your own (Module 23).
- **Length is not free.** Longer traces cost tokens and latency; the accuracy/cost curve is a real
  serving decision (Module 19).
- **Cost claims deserve care.** The widely quoted training figure is a final-run number that
  excludes research, failed runs, and infrastructure. Repeating it uncritically is a tell.

### Why this matters for the role you're interviewing for

The JD describes fine-tuning open LLMs and building agentic systems for enterprises. The R1 pipeline
is the reference architecture for exactly that:

- Enterprise tasks with checkable outcomes — SQL correctness, API call validity, retrieval
  groundedness, test pass rate — are **verifiable domains**. RLVR applies directly.
- The four stages map onto four modules you have already built labs for.
- The distillation finding tells you which experiment to run *first* on a fixed budget.
- The Zero result tells you that a base model's latent capability, not your SFT set, is the ceiling
  — so model selection matters more than data volume.

---

## Where it's used

- Every reasoning model trained since early 2025.
- GRPO in TRL / verl / OpenRLHF — the code path from Module 15's labs.
- Agent post-training wherever the outcome can be checked programmatically.
- Small reasoning models across the open ecosystem, nearly all distilled from R1-family traces.

---

## Labs

### Lab 1 — R1-Zero in miniature

The single most valuable lab in this module: reproduce the *shape* of the result, not the scale.

```python
# Base model. NOT the instruct version — that is the point.
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B-Base")

SYSTEM = ("Reason inside <think></think>, then give the final answer inside <answer></answer>.")

def reward(completion, gold):
    r = 0.0
    if re.search(r"<think>.*?</think>.*?<answer>.*?</answer>", completion, re.S):
        r += 0.2                                     # format
    ans = re.search(r"<answer>(.*?)</answer>", completion, re.S)
    if ans and normalize(ans.group(1)) == normalize(gold):
        r += 1.0                                     # correctness — nothing about length
    return r
```

Train with GRPO on GSM8K-style arithmetic, or on your SQL agent's verifiable tasks. **Log mean
completion length every step and plot it.**

| Step | Mean reward | **Mean completion tokens** | Format compliance |
|------|-------------|----------------------------|-------------------|
| 0 | | | |
| 200 | | | |
| 500 | | | |

**The length column rising while nothing in the reward mentions length is the R1-Zero result,
reproduced on your own machine.** That plot is worth more in an interview than any amount of
summarising the paper, because you can say you watched it happen.

At 0.6B expect the curve to be noisy and the absolute gain modest. Report it honestly — the shape
is the finding.

### Lab 2 — Look for the aha moment

Dump 50 completions from step 0 and 50 from the final step, and grep for self-correction:

```python
MARKERS = ["wait", "actually", "let me reconsider", "that's not right",
           "hold on", "on second thought", "recheck"]
rate = lambda comps: sum(any(m in c.lower() for m in MARKERS) for c in comps) / len(comps)
print("before:", rate(step0), "after:", rate(final))
```

Then read ten of the final traces yourself. **Report both the rate and your honest read of whether
the self-correction is real reasoning or surface mimicry of the pretraining distribution.** At 0.6B
it is often the latter, and saying so is a better answer than claiming emergence you didn't observe.

### Lab 3 — Why cold-start SFT exists

Reproduce the failure that motivated stage 1.

```python
def language_mixing_rate(completions):
    """Fraction of completions containing more than one script/language."""
    return sum(len(detect_scripts(c)) > 1 for c in completions) / len(completions)
```

| Run | Accuracy | Language mixing | Format compliance | Readability (your 1–5) |
|-----|----------|-----------------|-------------------|------------------------|
| A — Zero (base → GRPO) | | | | |
| B — Cold-start SFT (500 examples) → GRPO | | | | |
| C — Cold-start SFT only | | | | |

**Expect A to be competitive with B on accuracy and clearly worse on every usability column.** That
is the argument for stage 1 in one table, and it makes the point that the pipeline's structure is
driven by product constraints, not capability.

### Lab 4 — The full four-stage pipeline

Run the whole thing end-to-end on the SQL agent, small.

```bash
python stage1_coldstart_sft.py   --n 500                    # readability
python stage2_reasoning_rl.py    --steps 500                # GRPO + language reward
python stage3_rejection_sample.py --n 8 --keep-verified     # → SFT set
python stage4_final_sft_rl.py                               # consolidate + preference pass
```

Evaluate after **every** stage on the same held-out set with confidence intervals (Module 23):

| After stage | Task accuracy | Parse-failure rate | Mean tokens | Readability |
|-------------|---------------|--------------------|-------------|-------------|
| Base | | | | |
| 1 | | | | |
| 2 | | | | |
| 3 | | | | |
| 4 | | | | |

**A four-row-plus-base ablation of a published pipeline, on your own task, is a genuinely strong
portfolio artefact** — it's the difference between having read the paper and having run it.

### Lab 5 — Does RL create capability or select it?

The question the field is still arguing about, and you can get evidence cheaply.

Measure **pass@k** on the base model and on the RL'd model, for k = 1, 4, 16, 64 (Module 23):

| k | Base pass@k | RL'd pass@k |
|---|-------------|-------------|
| 1 | | |
| 4 | | |
| 16 | | |
| 64 | | |

**The pattern reported in the literature: RL improves pass@1 substantially and pass@64 much less,
sometimes not at all** — consistent with RL sharpening the distribution toward solutions the base
model could already reach, rather than creating new ones. If you see that, say so with the numbers.
If you see the opposite, that is more interesting still. Either way you are answering with data,
which is the posture the role wants.

### Lab 6 — Distil versus RL, decisively

Cross-reference Module 16 Lab 5 and complete the table. Same held-out set, confidence intervals,
plus the cost column people forget:

| Run | Accuracy | GPU-hours |
|-----|----------|-----------|
| Base | | 0 |
| Rejection-sampled self-SFT | | |
| **SFT on Qwen3-8B verified traces** | | |
| GRPO | | |
| Distil → GRPO | | |

**Include GPU-hours.** The R1 distillation finding is an economic result as much as a technical
one, and a table without cost misses the point.

### Lab 7 — Reward hacking the verifier

Give the SQL reward a deliberate gap — string-match the query text rather than execute it — and run
GRPO for 300 steps. Read twenty rollouts.

**The reward curve will look excellent and the rollouts will be gaming the string match.** Then
switch to execution-based verification (Module 23) and diff. This is Module 25's lesson arriving
through the R1 pipeline: RLVR's ceiling is your verifier's coverage.

---

## Failure modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| "DeepSeek invented RL for LLMs" | Conflating RLHF with RLVR | RLHF 2022 aligns; RLVR 2025 improves capability |
| "R1 introduced GRPO" | Wrong paper | GRPO is DeepSeekMath, 2024 |
| Zero run produces unusable output | Working as documented | That's why cold-start SFT exists |
| Length never grows | Zero-variance groups; task too hard or too easy | Module 15 — check the 20–60% pass band |
| RL gains don't hold on real tasks | Verifier gap | Execution-based verification; read rollouts |
| Claiming emergence you didn't measure | Reciting the paper | Run Lab 2 and report what you saw |
| Quoting the training cost as total cost | It's a final-run figure | Say what it excludes |
| Applying RLVR to open-ended writing | No verifier exists | Preference optimisation instead (Module 14) |

---

## Interview

**"How did DeepSeek change post-training?"**
I'd separate the claim carefully, because "they introduced RL to LLMs" isn't right — RLHF had been
standard since 2022. What changed was the reward. RLHF uses a learned reward model over human
preferences and it optimises for helpfulness and style; RLVR uses a programmatic verifier and it
optimises for correctness, which means it improves capability rather than alignment. The striking
demonstration was R1-Zero: take a base model, no SFT at all, run GRPO with an accuracy-plus-format
reward, and reasoning improves substantially. Response length grew over training even though
nothing in the reward mentioned length — the model discovered that thinking longer paid — and
self-correction appeared in traces that were never demonstrated. I'd also note GRPO itself came
from DeepSeekMath the year before, not from R1. And the openness mattered as much as the result:
the recipe was reproducible, so the field had it in libraries within weeks.

**"Why does R1 have a cold-start SFT stage if Zero proved SFT wasn't needed?"**
Because they solve different problems. Zero showed SFT isn't needed for *capability* — the
reasoning gain came from RL alone. But Zero's output mixed languages mid-trace and was poorly
formatted, so it was a research result rather than a product. Cold-start SFT is a few thousand
curated long-CoT examples doing a narrow job: fix readability and give RL a saner starting policy.
Then stage two is the reasoning RL with an added language-consistency reward, stage three is
rejection sampling from that model to build a large SFT set — which is exactly the cheap baseline
you're supposed to try before RL, except here it's used to consolidate what RL created — and stage
four is a final RL pass for helpfulness and safety. The structural idea I take from it is the loop:
RL creates capability, you distil it back into SFT data, then polish.

**"Does RL create new capability or surface what's already there?"**
The honest answer is that the evidence points more toward surfacing, and I'd want to measure rather
than assert. The test is pass@k: RL reliably improves pass@1, and at high k the gap to the base
model narrows considerably, which is what you'd expect if RL is sharpening the distribution toward
solutions the base could already reach at some k rather than creating new ones. That's consistent
with the general principle that post-training selects behaviour rather than installing knowledge.
It has a practical consequence I care about: your base model choice sets the ceiling, so on a fixed
budget I'd rather start from a stronger base than run more RL on a weaker one.

**"You have a budget and an enterprise agent task. RL or not?"**
I'd run the comparison rather than decide up front, and I'd expect distillation to win on cost. The
R1 distillation result was that small models SFT'd on a strong model's verified traces beat the same
models trained with RL directly, at a fraction of the price. So the order is: rejection-sampled
self-SFT as the cheap baseline, then SFT on a stronger teacher's verified trajectories, then GRPO,
then distil-then-GRPO — same eval, confidence intervals, and GPU-hours in the table, because this
is an economic decision as much as a technical one. RLVR is worth the money when the task is
genuinely verifiable and the cheaper options have plateaued. And I'd be clear about the limit: this
whole family of methods needs a checker, so it applies to SQL correctness or test pass rate, and
not to open-ended writing.

**"What are the limits?"**
The verifier is the ceiling — RLVR is only as good as what you can check, and any gap in the checker
gets found and exploited, which is why I read rollouts rather than trusting the reward curve. It
doesn't transfer to non-verifiable domains. Longer traces cost real tokens and latency, so the
accuracy-versus-cost curve is a serving decision, not a free win. And the published benchmark
numbers deserve the same contamination scrutiny I'd apply to my own evals — as does the widely
quoted training cost, which is a final-run figure that excludes the research, the failed runs, and
the infrastructure.

---

## Checkpoint

1. State the RLHF/RLVR difference in one sentence each, with the right years.
2. Attribute GRPO to the right paper.
3. Say what R1-Zero demonstrated and what it failed at, and connect the failure to stage 1.
4. Name all four R1 stages and the module each maps to.
5. Produce your own response-length-growth curve from Lab 1.
6. Report the self-correction rate before/after, with your honest read of it.
7. Produce the pass@k table and say what it implies about creation versus selection.
8. Produce the distil-versus-RL table **with GPU-hours**.
9. List three limits of RLVR without prompting.

---

**Next:** [16 — Distillation and compression](16-distillation.md) ·
**Back:** [15 — RLVR and GRPO](15-rlvr-grpo.md) · [Syllabus](../SYLLABUS.md)
