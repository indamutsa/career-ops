# Module 23 — Evaluation

This is your differentiator and you should treat it that way. Most candidates for a post-training
role can discuss DPO and GRPO. Far fewer can explain how they knew the model got better, and fewer
still have built the harness that told them. You already built evaluation gates at Expedia and a
retrieval eval harness at Boehringer — this module gives you the LLM-specific vocabulary to put on
top of experience you actually have.

---

## Terms

| Term | Meaning |
|------|---------|
| **Benchmark** | A fixed dataset + metric + protocol. |
| **Held-out set** | Data never used in training. The only honest source of a number. |
| **Contamination** | Eval data leaked into training. Invalidates the number. |
| **pass@k** | Probability at least one of k samples is correct. |
| **pass^k** | Probability *all* k are correct. The reliability metric. |
| **Exact match / F1** | String-overlap metrics. Cheap, brittle. |
| **Execution-based eval** | Run the output; compare effects. The right choice for code and SQL. |
| **Execution accuracy** | Fraction whose executed result matches the gold result. |
| **LLM-as-judge** | A model scores outputs. Cheap, scalable, biased. |
| **Pairwise comparison** | Judge picks between two outputs. More reliable than absolute scoring. |
| **Position bias** | Judges prefer whichever answer came first. |
| **Verbosity bias** | Judges prefer longer answers. |
| **Self-preference bias** | Judges prefer outputs from their own family. |
| **Elo / Bradley-Terry** | Turning pairwise wins into a ranking. |
| **Inter-annotator agreement** | Do two annotators agree? Your ceiling for judge quality. |
| **Cohen's / Fleiss' kappa** | Agreement corrected for chance. |
| **Rubric** | Explicit scoring criteria. Turns an opinion into a measurement. |
| **Golden set** | Small, hand-verified, never-changing. The regression suite. |
| **Regression suite** | Cases that must never break. |
| **Canary / behavioural test** | A specific known behaviour asserted directly. |
| **Confidence interval** | The range your number really lives in. |
| **Bootstrap** | Resampling to estimate that interval. |
| **Variance across seeds** | The spread that decides whether your improvement is real. |
| **Online / offline eval** | Real traffic vs fixed dataset. |
| **A/B test** | Randomised online comparison. |
| **Guardrail metric** | A metric that must not regress even if the target improves. |

---

## Concepts

### The hierarchy of evaluation quality

| Level | Method | Trust | Cost |
|-------|--------|-------|------|
| 1 | Loss / perplexity | Very low for task quality | Free |
| 2 | String match (EM, BLEU, ROUGE) | Low | Free |
| 3 | LLM-as-judge | Medium, with caveats | Low |
| 4 | **Execution-based verification** | **High** | Low, if the environment exists |
| 5 | Human evaluation | Highest | High |
| 6 | Online A/B on real traffic | Ground truth | Highest |

**Climb as high as your task allows.** For a SQL agent, level 4 is available and you should be
there — execute both queries and compare result sets. String-matching SQL is actively wrong: two
queries that differ in every character can be identically correct, and two nearly identical queries
can differ by a `JOIN` type that changes everything.

Level 1 deserves a warning. Eval loss can improve while task accuracy falls. It is a *training
health* signal, not a quality metric, and reporting it as though it were quality is a common
tell.

### Execution-based verification, properly

From Lab 01's `verifier.py`, the design decisions that matter:

1. **Compare result sets, not query text.**
2. **Sort rows** unless the question specifies ordering — otherwise you fail correct answers.
3. **Do not compare column names.** The user asked "how many orders"; `count`, `n`, and
   `total_orders` are all correct.
4. **Normalise numeric types.** `Decimal('5')`, `5`, `5.0` are the same answer.
5. **Distinguish `GOLD QUERY BROKEN` from `WRONG ANSWER`.** Your gold queries will break as the
   schema evolves and you must never score that as a model failure. This is the single most
   important line in the verifier and it is the one people leave out.

Point 5 generalises: **your evaluation harness has bugs, and its bugs look exactly like model
regressions.** Building in a distinct "the eval itself failed" outcome is what lets you tell them
apart at 2am.

### pass@k versus pass^k

- `pass@k` — at least one of k samples correct. Rises with k. Right for a system with a verifier
  that can select the good one.
- `pass^k` — all k correct. Falls with k. The **reliability** metric, and the honest one for
  production where you get one shot.

An agent at `pass@8 = 90%` and `pass@1 = 40%` is not a 90% agent. It is a 40% agent with a lot of
variance, and reporting the first number without the second is misleading. **Report both.** Being
the person who volunteers `pass^k` is a strong signal.

### LLM-as-judge: use it, but know its biases

| Bias | Effect | Mitigation |
|------|--------|------------|
| Position | Prefers the first option | Run both orders, average |
| Verbosity | Prefers longer | Control for length; report length alongside score |
| Self-preference | Prefers its own family | Use a different family as judge |
| Sycophancy | Agrees with framing in the prompt | Neutral prompt; hide provenance |
| Format | Prefers formatting it likes | Rubric with explicit criteria |

Rules that make judges usable:

1. **Pairwise beats absolute.** "Which is better, A or B?" is far more reliable than "score 1–10".
2. **Rubrics beat vibes.** Enumerate criteria and score each separately.
3. **Validate against humans.** Label 100 examples by hand, compute agreement (kappa) with the
   judge. If the judge does not agree with you, its numbers are decorative. Report the agreement
   number whenever you report judge scores.
4. **Never judge with the model you are training.** Circular, and it will reward-hack the judge.

### Statistics — the part most people skip

If your eval set is 100 tasks and accuracy goes 62% → 65%, you have not shown anything. The 95%
confidence interval on 100 binary trials at 62% is roughly ±10 points.

```python
import numpy as np

def bootstrap_ci(results, n=10000, alpha=0.05):
    a = np.asarray(results, dtype=float)
    means = np.mean(np.random.choice(a, size=(n, len(a)), replace=True), axis=1)
    return float(np.percentile(means, 100 * alpha / 2)), float(np.percentile(means, 100 * (1 - alpha / 2)))
```

Three rules:

1. **Always report a confidence interval**, never a bare point estimate.
2. **Run multiple seeds.** Seed variance in post-training is often larger than the effect you are
   claiming. Three seeds minimum; report mean and spread.
3. **Paired comparison beats independent.** Evaluate both models on the *same* tasks and compare
   per-task outcomes (McNemar's test). Far more statistical power than comparing two independent
   accuracy numbers.

For an applied-research interview, "how many tasks do you need for this difference to be
significant" is an excellent question to ask *them*, and it signals seriousness.

### The golden set

A small (50–200), hand-verified, **frozen** set that:

- Never changes, so numbers are comparable across months.
- Is never trained on, and is decontaminated against every training set (Module 13).
- Contains the cases you care about most, including known past failures.
- Runs fast enough to gate every commit.

This is the LLM version of the evaluation gates you built at Expedia — the promotion blocker. Same
concept, different artefact.

### Guardrail metrics

A model can improve on target and regress somewhere that matters more:

| Target | Guardrail |
|--------|-----------|
| Task accuracy | Refusal rate on out-of-scope questions |
| Task accuracy | General chat ability (forgetting) |
| Helpfulness | Safety / harmful-compliance rate |
| Conciseness | Completeness |
| Speed | Accuracy |
| Reward | **Held-out** reward — the reward-hacking detector (Module 15) |

**A release gate is target improved AND no guardrail regressed.** Stating that as a rule is the
production instinct the JD's "enterprise scale" is asking about.

### The eval-driven development loop

```
1. Define the metric BEFORE the intervention
2. Measure the baseline, with CI
3. Change one thing
4. Re-measure on the same frozen set, same seeds
5. Check guardrails
6. Ship or revert
```

Step 1 is the one that gets skipped, and skipping it is how people end up choosing the metric that
happens to have moved. Write the metric down before you run anything.

---

## Where it's used

- **Every post-training decision.** Without eval, SFT vs DPO vs GRPO is an aesthetic preference.
- **Reward-hacking detection** (Module 15) — held-out reward diverging from training reward.
- **Release gating**, which is your existing Expedia experience under a new name.
- **Dataset curation** (Module 13) — pass rates decide which tasks make the RL set.
- **The interview.** "How did you know it got better" is the question that separates candidates.

---

## Labs

### Lab 1 — Build the frozen golden set

From your Lab 01 task pool, take 60 tasks and:

```python
import json, hashlib

golden = []
for t in tasks:
    golden.append({
        "task_id": t["id"],
        "tier": t["tier"],
        "question": t["question"],
        "gold_sql": t["gold_sql"],
        "expect": t.get("expect", "answerable"),   # or "impossible"
    })

blob = json.dumps(golden, sort_keys=True).encode()
meta = {"n": len(golden), "sha256": hashlib.sha256(blob).hexdigest(), "frozen": "2026-09-05"}
json.dump({"meta": meta, "tasks": golden}, open("eval/golden.json", "w"), indent=2)
print(meta)
```

**Commit the hash.** Any future change to the golden set changes the hash, which forces a
deliberate decision instead of a silent drift. Then verify every gold query still executes — a
golden set with a broken gold answer scores your model wrong forever.

### Lab 2 — Confidence intervals on your baseline

```python
import json, numpy as np

rollouts = [json.loads(l) for l in open("data/baseline.jsonl")]
res = [r["reward"] == 1.0 for r in rollouts]

lo, hi = bootstrap_ci(res)
print(f"accuracy {np.mean(res):.3f}  95% CI [{lo:.3f}, {hi:.3f}]  n={len(res)}")

for n in [50, 100, 200, 500, 1000]:
    sub = np.random.choice(res, size=n)
    l, h = bootstrap_ci(sub)
    print(f"  n={n:>5}  width {h-l:.3f}")
```

**Write down the width at your actual n.** That width is the smallest improvement you can honestly
claim. If it is 10 points, do not report a 3-point gain as an improvement.

### Lab 3 — pass@k and pass^k

```python
from collections import defaultdict
import statistics

by_task = defaultdict(list)
for r in rollouts:
    by_task[r["task_id"]].append(r["reward"] == 1.0)

for k in [1, 2, 4, 8]:
    at  = statistics.mean(any(v[:k]) for v in by_task.values() if len(v) >= k)
    pow_ = statistics.mean(all(v[:k]) for v in by_task.values() if len(v) >= k)
    print(f"k={k}  pass@k {at:.2f}   pass^k {pow_:.2f}")
```

**The gap between the two columns is your variance problem, quantified.** A large gap says the
capability is there but unreliable — which points at RL and at decoding config, not at more SFT
data.

### Lab 4 — Verifier robustness (test the test)

Your verifier is code and it has bugs. Test it:

```python
CASES = [
    # (gold, candidate, expected_verdict)
    ("SELECT count(*) FROM orders",
     "SELECT COUNT(*) AS n FROM orders",                 "PASS"),   # column name differs
    ("SELECT name FROM customers ORDER BY name",
     "SELECT name FROM customers",                        "PASS"),   # order not requested
    ("SELECT sum(total) FROM orders",
     "SELECT sum(total)::numeric FROM orders",            "PASS"),   # numeric type
    ("SELECT count(*) FROM orders",
     "SELECT count(*) FROM order_items",                  "FAIL"),   # genuinely wrong
    ("SELECT * FROM nonexistent_table",
     "SELECT count(*) FROM orders",                       "GOLD_BROKEN"),
]

for gold, cand, expected in CASES:
    got = verify(gold, cand)
    print(f"{'ok ' if got == expected else 'BUG'} expected {expected:<12} got {got:<12} {cand[:40]}")
```

**Every case here is one you will hit.** A verifier that fails the first three under-reports your
model; one that passes case four over-reports it. Case five is the one that saves you a day of
debugging a "regression" that was a schema change.

### Lab 5 — Build and validate an LLM judge

```python
JUDGE_RUBRIC = """Compare two answers to a database question.
Score each on:
  correctness  (0-2): does it answer the question asked?
  completeness (0-2): all requested fields present?
  clarity      (0-1): unambiguous phrasing?
Reply with JSON: {"a": {...}, "b": {...}, "winner": "a"|"b"|"tie"}"""

def judge(question, a, b, model):
    fwd = model(JUDGE_RUBRIC, question, a, b)
    rev = model(JUDGE_RUBRIC, question, b, a)     # swap to cancel position bias
    return fwd, rev
```

Then **validate it**:

1. Hand-label 100 pairs yourself.
2. Compute Cohen's kappa between you and the judge.
3. Measure position bias: how often does forward disagree with reversed?
4. Measure verbosity bias: correlate the judge's winner with answer length.

```python
from sklearn.metrics import cohen_kappa_score
print("kappa vs human:", cohen_kappa_score(human_labels, judge_labels))
```

**Interpretation:** kappa below ~0.4 means the judge is not measuring what you are. Report this
number every time you report a judge score — an unvalidated judge is an opinion with a decimal
point.

### Lab 6 — Paired significance test

```python
from statsmodels.stats.contingency_tables import mcnemar
import numpy as np

# same tasks, two models
both  = sum(a and b for a, b in zip(base, tuned))
a_only = sum(a and not b for a, b in zip(base, tuned))
b_only = sum(b and not a for a, b in zip(base, tuned))
neither = sum(not a and not b for a, b in zip(base, tuned))

table = [[both, a_only], [b_only, neither]]
print(f"base only {a_only}   tuned only {b_only}")
print(mcnemar(table, exact=True))
```

**The two off-diagonal cells are the whole test.** If the tuned model fixed 12 tasks and broke 9,
your headline +3% is noise regardless of how it looks. This is the analysis that stops you shipping
a non-improvement, and describing it in an interview is a much stronger answer than quoting a
delta.

### Lab 7 — The release gate

```python
GATE = {
    "task_accuracy":    ("target",    "increase"),
    "parse_fail_rate":  ("guardrail", "no_worse_than", 0.01),
    "general_chat_acc": ("guardrail", "no_worse_than_baseline_minus", 0.02),
    "refusal_correct":  ("guardrail", "no_worse_than_baseline_minus", 0.05),
    "p95_latency_ms":   ("guardrail", "no_worse_than", 3000),
}
```

Implement it as a script that reads two eval JSON files and exits non-zero on any guardrail
violation. **Wire it into the repo's CI.** This is the artefact that makes your Expedia promotion
gates transferable to LLM work, and having it in a public repo is exactly the CV evidence Module 13
and `modes/train.md` are aiming at.

---

## Failure modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| Great benchmark, poor production | Contamination, or the benchmark isn't the task | Decontaminate; build a task-specific set |
| Improvement vanishes on rerun | Within noise; single seed | CI, multiple seeds, paired test |
| Correct answers scored wrong | String matching | Execution-based verification |
| Correct answers scored wrong | Column names / row order compared | Normalise both |
| "Model regressed" but it didn't | Gold query broke on a schema change | Distinct `GOLD_BROKEN` verdict |
| Judge scores don't match your judgement | Unvalidated judge | Kappa against 100 human labels |
| Judge always prefers option A | Position bias | Run both orders |
| Judge always prefers longer | Verbosity bias | Control and report length |
| Eval loss improved, task didn't | Loss isn't the metric | Evaluate the task |
| Target up, product worse | No guardrails | Guardrail metrics in the gate |
| pass@8 looks great, users complain | Reported the wrong statistic | Report pass@1 and pass^k |
| Numbers not comparable month to month | Golden set drifted | Hash it; freeze it |

---

## Interview

**"How do you know a fine-tune actually improved anything?"**
A frozen golden set that's hash-pinned and decontaminated against every training set, evaluated
with the same seeds before and after. Execution-based verification rather than string matching,
because two SQL queries that share no characters can be identically correct. Bootstrap confidence
intervals, three seeds minimum, and a paired McNemar test on the same tasks rather than comparing
two independent accuracy numbers — the off-diagonal cells, how many tasks it fixed versus broke,
tell you far more than the delta. And guardrail metrics, so I can't claim a win when the model got
better on-task and lost general ability or started refusing valid questions.

**"Your model went from 62% to 65%. Ship it?"**
Not on that number. With 100 tasks the 95% interval is roughly ±10 points, so 3 points is well
inside noise. I'd want the paired analysis — if it fixed 12 and broke 9, that's not an improvement
even though the headline moved. And I'd check seed variance, which in post-training is often larger
than the effect being claimed. If it survives all that, then guardrails, then ship.

**"How do you evaluate something with no single right answer?"**
Climb as high as the task allows. If any part is verifiable — does the code run, does the JSON
parse, does the query return the right rows — verify that part programmatically, because it's free
and it's exact. For the rest, LLM-as-judge, but pairwise rather than absolute scoring, with a
written rubric, both orderings run to cancel position bias, and a judge from a different model
family than the one I'm training. Then I validate the judge against 100 of my own labels and report
the kappa alongside every judge score. An unvalidated judge is an opinion with a decimal point.

**"What's the most common evaluation mistake?"**
Reporting eval loss as if it were quality. It's a training-health signal — it can improve while
task accuracy falls, because loss rewards matching the reference tokens and the task rewards being
right. Close second is contamination: eval examples leaked into training make everything downstream
fiction, and the numbers look *better*, which is why nobody catches it. Third is single-run
benchmarking of agents, where run-to-run variance is often bigger than the effect.

**"What would you build first at a new post-training team?"**
The eval harness, before touching a training script. Without it, every method choice is taste and
every result is unfalsifiable — you can't tell reward hacking from real improvement, and you can't
tell a broken gold answer from a regression. I've done the equivalent at Expedia: evaluation gates
that blocked regressed models from promotion. Same concept, different artefact — a frozen golden
set, execution-based verification, guardrail metrics, and a gate script in CI that exits non-zero.

---

## Checkpoint

1. Produce a hash-pinned golden set and verify every gold answer still executes.
2. Report your baseline with a bootstrap CI and state the smallest claimable improvement.
3. Report pass@k and pass^k and interpret the gap.
4. Pass all five verifier robustness cases, including `GOLD_BROKEN`.
5. Report judge-vs-human kappa, position bias rate, and verbosity correlation.
6. Run a paired McNemar test and report both off-diagonal cells.
7. Have a release-gate script in CI that fails on a guardrail regression.

---

**Next:** [24 — Failure analysis](24-failure-analysis.md) ·
**Back:** [22 — Agents and orchestration](22-agents.md) · [Syllabus](../SYLLABUS.md)
