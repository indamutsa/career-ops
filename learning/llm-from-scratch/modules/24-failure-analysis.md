# Module 24 — Failure analysis and model forensics

Module 23 tells you *whether* the model got better. This module tells you *why it fails*, which is
what determines what you do next. It is also the module where your existing production experience
transfers most directly: at Expedia you separated data drift, concept drift and prediction shift as
distinct failure modes and added SHAP to explain individual errors. That instinct — **an aggregate
metric is a symptom, not a diagnosis** — is exactly what this module formalises for LLMs.

---

## Terms

| Term | Meaning |
|------|---------|
| **Failure taxonomy** | A closed set of mutually exclusive failure categories. |
| **Error attribution** | Assigning a failure to a stage in the pipeline. |
| **Root cause** | The change that would have prevented it. |
| **Confusion matrix** | Predicted vs actual, cross-tabulated. |
| **Slice analysis** | Metrics broken down by a data attribute. |
| **Regression** | A previously-passing case that now fails. |
| **Flaky case** | Passes sometimes. Symptom of variance, not capability. |
| **Hallucination** | Confident output not grounded in the input or in fact. |
| **Intrinsic / extrinsic hallucination** | Contradicts the source / unverifiable from the source. |
| **Calibration** | Do stated/implied confidences match actual accuracy? |
| **ECE** | Expected Calibration Error — mean gap between confidence and accuracy. |
| **Reliability diagram** | Accuracy plotted against confidence bin. |
| **Logprob-based confidence** | Using mean token logprob as a confidence proxy. |
| **Perplexity spike** | A sharp rise in surprisal — often marks the point of derailment. |
| **Attention analysis** | Inspecting what the model attended to. Suggestive, not proof. |
| **Logit lens** | Decoding intermediate layers to see the prediction forming. |
| **Activation patching** | Swapping activations between runs to establish causality. |
| **Circuit** | A minimal subgraph implementing a behaviour. |
| **Probing** | Training a classifier on activations to test what is encoded. |
| **Data drift / concept drift / prediction shift** | Input distribution changes / input→output relation changes / output distribution changes. |
| **Golden regression case** | A specific past failure, frozen as a test. |

---

## Concepts

### The taxonomy is the deliverable

"60% accuracy" is not actionable. This is:

| Category | Share | Fix |
|----------|-------|-----|
| Parse failure | 4% | Constrained decoding (M10) |
| Wrong tool selected | 9% | Better tool descriptions, or RL |
| Right tool, wrong arguments | 11% | Data coverage, or RL |
| Correct execution, wrong answer | 8% | Reasoning — RL, or a bigger model |
| Step limit hit | 5% | Raise horizon, or reduce steps needed |
| Gave up when answerable | 2% | Reward shaping |
| Didn't give up when impossible | 1% | Reward shaping |
| **Gold answer wrong** | **3%** | **Fix the eval** |

**Two things make this table valuable.** First, each row has a *different* fix, so the shape of the
distribution tells you where to spend the next two weeks. Second, the last row exists: some
fraction of every failure set is your own eval being wrong, and a team that has never found any is
not looking.

Build the taxonomy before you build the training loop. It is what makes every subsequent result
interpretable.

### Categories must be mutually exclusive and jointly exhaustive

If a trajectory can land in two buckets, your percentages do not add up and your prioritisation is
garbage. Enforce it in code:

```python
def classify(rollout) -> str:
    if rollout["terminal"] is None:
        return "step_limit"
    if any(s.get("parse_failed") for s in rollout["steps"]):
        return "parse_failure"
    if rollout["terminal"] == "gave_up":
        return "gave_up_correctly" if rollout["task_impossible"] else "gave_up_wrongly"
    if rollout["gold_broken"]:
        return "eval_bug"
    if rollout["reward"] == 1.0:
        return "success"
    if not rollout["used_required_tools"]:
        return "wrong_tool"
    if rollout["sql_error"]:
        return "bad_arguments"
    return "wrong_reasoning"
```

Order matters — the first matching rule wins, so put the unambiguous cases first. Then assert
that categories sum to 100% of rollouts. That assertion catches taxonomy drift, which happens
whenever someone adds a new tool.

### Read fifty failures by hand

There is no substitute. Automated categorisation reproduces the categories you already thought of;
reading the raw trajectories finds the ones you did not.

Do it like this: sample 50 failures stratified across categories, read the full trajectory of each,
and write one line per failure describing what went wrong in your own words. Then cluster your own
sentences. Almost always you will find two or three patterns your taxonomy has no bucket for — and
those are the interesting ones.

**This is the single most valuable half-day in a post-training project**, and saying so in an
interview reads as experience rather than theory.

### Slice analysis

Aggregate metrics hide systematic failure:

| Slice by | Reveals |
|----------|---------|
| Task tier / difficulty | Where the capability boundary is |
| Number of tables involved | Whether joins are the problem |
| Question length | Long-context handling |
| Whether the answer requires aggregation | A specific skill gap |
| Whether a column name is ambiguous | Discovery-strategy failure |
| Trajectory length | Compounding (M22) |
| Time of day / user cohort (production) | Drift |

A model at 60% overall might be at 85% on single-table and 20% on multi-table. That is one finding
worth more than a month of undirected tuning, and it is invisible in the headline.

### Hallucination, split usefully

| Type | Description | Detection |
|------|-------------|-----------|
| Intrinsic | Contradicts the provided source | Check against the source |
| Extrinsic | Unverifiable from the source | Attribution / groundedness scoring |
| Fabricated entity | Invents a column, table, function, citation | **Schema/registry check — trivial and exact** |
| Confident error | Wrong with no hedging | Calibration analysis |

For a database agent the third row is nearly free to detect and should be a hard metric: parse the
generated SQL, extract identifiers, check every one against the real schema. **Fabricated-column
rate** is a specific, cheap, unambiguous number, and having it is the difference between "the model
hallucinates sometimes" and a measurement.

### Calibration

A model that is 90% confident should be right 90% of the time. Use mean token logprob of the final
answer as a confidence proxy, bin it, and plot accuracy per bin.

```python
import numpy as np

def ece(confidences, correct, n_bins=10):
    conf, corr = np.asarray(confidences), np.asarray(correct, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.sum():
            total += m.mean() * abs(corr[m].mean() - conf[m].mean())
    return total
```

Why it matters operationally: **calibration is what lets you route**. If confidence predicts
correctness, you escalate low-confidence queries to a bigger model or to a human, and you get a
large reliability win without touching the model. If it does not, routing on confidence is
superstition.

Known effect worth mentioning: **RLHF degrades calibration.** Base models are often better
calibrated than their aligned versions, because alignment training pushes toward confident,
assertive phrasing. If you post-train and your ECE worsens, that is expected, and it is a guardrail
metric you should be tracking (Module 23).

### Perplexity spikes locate the derailment

For a failed trajectory, compute per-token logprob across the whole thing and find where surprisal
jumps. That point is usually where it went wrong — a misread column name, a bad observation, an
unexpected error message. It turns "the trajectory is wrong" into "step 4, token 31" and it is a
genuinely useful debugging technique that few people use.

### Interpretability: what is worth your time

| Technique | Effort | Payoff for applied work |
|-----------|--------|-------------------------|
| Attention maps | Low | Suggestive only — attention is not attribution |
| **Logit lens** | Low | Good — see the prediction form across layers |
| Probing classifiers | Medium | Good for "is X encoded at all?" |
| Activation patching | High | The only one that establishes causality |
| SAEs / circuits | Very high | Research programme, not a debugging tool |

**Be honest about the ceiling.** For an applied research engineer, the taxonomy, slice analysis and
calibration work will produce ten times the value of circuit-level interpretability. Knowing the
vocabulary and knowing when *not* to reach for it is the mature position — and if the team does
serious interpretability, saying "I'd want to learn it from people doing it properly rather than
claim it" is a better answer than bluffing.

### Drift, in your own vocabulary

The Expedia framing maps onto LLM systems cleanly:

| Classic ML | LLM system | Detection |
|-----------|------------|-----------|
| Data drift | Query distribution changes (new users, new topics) | Embedding-distribution monitoring; OOD rate |
| Concept drift | Correct answer changes (schema evolves, policy changes) | Golden-set failures on unchanged inputs |
| Prediction shift | Output distribution changes (longer, more refusals) | Length, refusal rate, tool-mix monitoring |
| Upstream change | Base model version, tokenizer, template, provider | **Pin everything and alert on change** |

That last row is the LLM-specific one and it bites hardest: a provider silently updating a model
behind an endpoint changes your system with no code change on your side. Pin versions and record
the base model sha in every eval run.

---

## Where it's used

- **After every eval run** — the taxonomy is what turns a number into a decision.
- **Reward design** (M15) — the failure distribution tells you which cases the reward must
  distinguish.
- **Dataset curation** (M13) — you add data for the categories that dominate.
- **Production monitoring** — the same taxonomy, running online.
- **The interview** — "walk me through debugging a model that got worse" is a standard question.

---

## Labs

### Lab 1 — Build and validate the taxonomy

Implement `classify()` above over `data/baseline.jsonl`, then:

```python
from collections import Counter
import json

rollouts = [json.loads(l) for l in open("data/baseline.jsonl")]
counts = Counter(classify(r) for r in rollouts)
n = len(rollouts)

assert sum(counts.values()) == n, "categories are not exhaustive"
for cat, c in counts.most_common():
    print(f"{cat:<22}{c:>5}{c/n:>8.1%}")
```

**Deliverable:** the table. Then write, for each category, the one intervention you would try. If
two categories share an intervention, they should probably be one category.

### Lab 2 — Read fifty failures

```python
import random
random.seed(0)

fails = [r for r in rollouts if r["reward"] != 1.0]
by_cat = {}
for r in fails:
    by_cat.setdefault(classify(r), []).append(r)

sample = []
for cat, rs in by_cat.items():
    sample += random.sample(rs, min(8, len(rs)))     # stratified

for r in sample[:50]:
    print("=" * 78)
    print(f"[{classify(r)}] task {r['task_id']}: {r['question']}")
    for i, s in enumerate(r["steps"], 1):
        obs = str(s.get("observation", ""))[:160]
        print(f"  {i}. {s['tool']}({str(s['args'])[:70]}) -> {obs}")
    print(f"  terminal: {r['terminal']}  reward: {r['reward']}")
    print("  MY NOTE: ")
```

Fill in `MY NOTE` by hand for all 50. Then cluster your notes. **Write down every pattern that has
no bucket in your taxonomy** — that list is the lab's real output and it is the most credible
artefact you can bring to a technical interview.

### Lab 3 — Slice analysis

```python
import statistics
from collections import defaultdict

def slices(r):
    return {
        "tier": r["tier"],
        "n_tables": "1" if r["gold_n_tables"] == 1 else "2" if r["gold_n_tables"] == 2 else "3+",
        "aggregation": "yes" if r["gold_has_aggregate"] else "no",
        "q_len": "short" if len(r["question"].split()) < 12 else "long",
        "traj_len": "<=4" if len(r["steps"]) <= 4 else ">4",
    }

acc = defaultdict(list)
for r in rollouts:
    for k, v in slices(r).items():
        acc[(k, v)].append(r["reward"] == 1.0)

for (k, v), vals in sorted(acc.items()):
    print(f"{k:<12}{v:<8}{statistics.mean(vals):>7.2f}  (n={len(vals)})")
```

**Find the worst slice and state the hypothesis it implies.** "20% on 3+ tables versus 85% on
single-table" is a finding; "60% overall" is not.

### Lab 4 — Fabricated identifier rate

```python
import re

schema_idents = set()   # populate from information_schema
for tbl, cols in real_schema.items():
    schema_idents.add(tbl.lower())
    schema_idents |= {c.lower() for c in cols}

SQL_KEYWORDS = {"select", "from", "where", "group", "by", "order", "join", "on",
                "count", "sum", "avg", "min", "max", "as", "and", "or", "limit"}

fab = tot = 0
for r in rollouts:
    for s in r["steps"]:
        if s["tool"] != "run_sql":
            continue
        for ident in re.findall(r"\b[a-z_][a-z0-9_]*\b", s["args"]["q"].lower()):
            if ident in SQL_KEYWORDS:
                continue
            tot += 1
            fab += ident not in schema_idents

print(f"fabricated identifier rate: {fab}/{tot} = {fab/tot:.2%}")
```

Then correlate: do trajectories with a fabricated identifier fail more often? And does the rate
fall after SFT on verified rollouts? **That before/after number is a clean, quantified result.**

### Lab 5 — Calibration and reliability diagram

```python
import numpy as np

conf = [np.exp(np.mean(r["final_answer_logprobs"])) for r in rollouts]
corr = [r["reward"] == 1.0 for r in rollouts]

print(f"ECE: {ece(conf, corr):.3f}\n")
edges = np.linspace(0, 1, 11)
for lo, hi in zip(edges[:-1], edges[1:]):
    m = [(c > lo) and (c <= hi) for c in conf]
    if sum(m):
        a = np.mean([c for c, k in zip(corr, m) if k])
        mc = np.mean([c for c, k in zip(conf, m) if k])
        print(f"[{lo:.1f},{hi:.1f}]  n={sum(m):>4}  conf {mc:.2f}  acc {a:.2f}  gap {a-mc:+.2f}")
```

Then the operational question: **if you routed everything below 0.5 confidence to a bigger model,
what accuracy and cost would you get?** Compute it. That is a shippable proposal, not an
observation.

### Lab 6 — Locate the derailment with perplexity

```python
import numpy as np

def derail_point(rollout):
    lp = np.array(rollout["token_logprobs"])
    if len(lp) < 20:
        return None
    win = 10
    roll = np.convolve(lp, np.ones(win) / win, mode="valid")
    return int(np.argmin(roll)) + win // 2      # most surprised region

for r in [x for x in rollouts if x["reward"] != 1.0][:10]:
    i = derail_point(r)
    print(f"task {r['task_id']}: derailed near token {i} -> "
          f"{r['tokens'][max(0,i-8):i+8]}")
```

Read the surrounding tokens for each. In most failed trajectories the spike lands on the exact
misstep — a hallucinated column, a misread error message, a wrong table choice.

### Lab 7 — Logit lens

```python
import torch

model.config.output_hidden_states = True
out = model(**tok("The capital of France is", return_tensors="pt").to("mps"),
            output_hidden_states=True)

W_U = model.get_output_embeddings().weight     # unembedding
for layer, h in enumerate(out.hidden_states):
    logits = model.model.norm(h[0, -1]) @ W_U.T
    top = torch.topk(torch.softmax(logits.float(), -1), 3)
    print(f"layer {layer:>2}: " +
          "  ".join(f"{tok.decode(i).strip()!r}:{v:.3f}" for i, v in zip(top.indices, top.values)))
```

**Watch the answer form.** Early layers predict syntactic continuations; the factual answer usually
emerges in the middle-to-late layers and then sharpens. Do it once for a case the model gets right
and once for a case it gets wrong — sometimes the wrong answer is already locked in surprisingly
early, which tells you the problem is representational, not a decoding accident.

### Lab 8 — Regression suite from real failures

Every failure you diagnose becomes a permanent test:

```python
import json, datetime

def freeze_regression(rollout, diagnosis, fix):
    return {
        "task_id": rollout["task_id"],
        "question": rollout["question"],
        "gold_sql": rollout["gold_sql"],
        "failed_on": rollout["model_sha"],
        "diagnosis": diagnosis,
        "fix": fix,
        "frozen": datetime.date.today().isoformat(),
    }
```

Append to `eval/regressions.json` and run it in the release gate from Module 23. **A bug you fixed
without a regression test is a bug you will ship again**, and that sentence is the entire argument
for this lab.

---

## Failure modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| Don't know what to work on next | No taxonomy | Build it before the training loop |
| Categories overlap; percentages don't add up | Not mutually exclusive | Ordered first-match rules + a sum assertion |
| Fixed the wrong thing | Optimised the biggest number, not the biggest fixable one | Pair each category with its intervention |
| "The model hallucinates" with no number | No measurement | Fabricated-identifier rate |
| Routing on confidence doesn't help | Model isn't calibrated | Measure ECE before designing the router |
| Calibration worse after post-training | Expected effect of alignment | Track ECE as a guardrail |
| Same bug returns two months later | No regression test | Freeze every diagnosed failure |
| Overall metric stable, users unhappy | A slice regressed | Slice analysis on every run |
| Production degraded, offline evals fine | Drift, or an upstream model change | Pin versions; monitor input distribution |
| Weeks lost in interpretability | Reached for circuits before taxonomy | Taxonomy, slices, calibration first |
| Failures blamed on the model | Some are eval bugs | Keep an explicit `eval_bug` category |

---

## Interview

**"Your agent is at 60%. What do you do?"**
Not tune hyperparameters. I'd break the 40% into a mutually exclusive taxonomy — parse failures,
wrong tool, wrong arguments, wrong reasoning, step limit, incorrect give-up, and eval bugs — because
each has a different fix and the shape tells me where the next two weeks go. If it's mostly parse
failures that's constrained decoding, a day's work. If it's mostly wrong reasoning, that's RL or a
bigger model. Then slice it: 60% overall might be 85% single-table and 20% multi-table, which is a
finding the aggregate hides completely. And I'd read fifty failed trajectories by hand, because
automated categorisation only ever finds the categories I already thought of.

**"How do you measure hallucination in a database agent?"**
For that domain it's mostly exact and nearly free. Parse the generated SQL, extract every
identifier, check each against the real schema — fabricated table and column names give you a hard,
unambiguous rate rather than an impression. Then correlate it with failure, and track it before and
after training. For the softer cases — an answer that's syntactically fine and factually
unsupported — I'd fall back to groundedness checking against the returned rows, but I'd start with
the part that's exactly measurable.

**"Walk me through debugging a model that got worse after training."**
First establish it's real: same golden set, same seeds, confidence interval, paired test — a
surprising number of regressions are noise or a broken gold answer, which is why I keep an
explicit eval-bug category. If it's real, compare failure taxonomies before and after; a
regression usually concentrates in one or two categories rather than spreading evenly, and that
localises it. Check guardrails — often the target improved and something else broke, like refusal
behaviour or general chat. Then check the usual training suspects: reward hacking if held-out
reward diverged from training reward, forgetting if there was no replay data, DPO degeneration if
both chosen and rejected logprobs fell. And I'd check the boring things first — did the base model
version, tokenizer or chat template change under me.

**"How much interpretability do you use in practice?"**
Honestly, the cheap end. Logit lens to see where a prediction forms across layers, and perplexity
spikes to locate where a trajectory derailed — that one turns "the trajectory is wrong" into "step
4, token 31", which is genuinely useful. Attention maps I treat as suggestive, not as attribution.
Activation patching is the only technique on that list that establishes causality and it's
expensive. For applied post-training, failure taxonomy, slice analysis and calibration produce
maybe ten times the value of circuit-level work. If the team does serious interpretability I'd want
to learn it from people doing it properly rather than claim it.

**"What's your background in this?"**
I built model failure diagnosis and monitoring from scratch at Expedia — separating data drift,
concept drift and prediction shift as distinct failure modes rather than one alert, and adding SHAP
so individual errors were explainable rather than just counted. The LLM version is the same
instinct with different instruments: the aggregate metric is a symptom, the taxonomy is the
diagnosis. And the drift categories transfer almost directly — query distribution shifting is data
drift, schema evolution is concept drift, output getting longer or more refusal-heavy is prediction
shift. The one that's new is upstream change: a provider updating a model behind an endpoint
changes your system with no code change on your side, so everything gets pinned and every eval run
records the base model sha.

---

## Checkpoint

1. Produce the failure taxonomy table with a sum assertion and an intervention per row.
2. Read 50 failures by hand and list the patterns your taxonomy missed.
3. Produce the slice table and name your worst slice.
4. Report fabricated-identifier rate and its correlation with failure.
5. Report ECE, the reliability diagram, and the confidence-routing proposal with numbers.
6. Locate the derailment point in ten failed trajectories.
7. Show the answer forming across layers with the logit lens.
8. Have a regression suite wired into the release gate.

---

**Next:** [25 — Safety and alignment](25-safety.md) ·
**Back:** [23 — Evaluation](23-evaluation.md) · [Syllabus](../SYLLABUS.md)
