# Module 22 — Agents and orchestration

"Agentic systems" is the phrase in Isaac's message that carries the most weight and the least
definition. This module gives it a precise one, then covers what actually breaks when you run
agents at enterprise scale — which is mostly not the model.

---

## Terms

| Term | Meaning |
|------|---------|
| **Agent** | An LLM in a loop with tools, taking actions against an environment until a stop condition. |
| **Trajectory / episode / rollout** | The full sequence of observations, actions and results for one task. |
| **Step** | One action and its observation. |
| **Horizon** | Max steps before forced termination. |
| **Environment** | Everything the agent acts on: DB, filesystem, API, browser. |
| **Observation** | What a tool returns. |
| **Terminal state** | Task submitted, gave up, error, or step limit hit. |
| **ReAct** | Reason → Act → Observe, interleaved. |
| **Reflexion** | Self-critique after failure, retry with the critique in context. |
| **Plan-and-execute** | Plan the whole sequence up front, then run it. |
| **Router** | A cheap classifier that picks which agent/model/tool path handles a request. |
| **Multi-agent** | Several specialised agents with a coordination protocol. |
| **Supervisor / orchestrator** | The agent that delegates to others. |
| **Handoff** | Transferring control and context between agents. |
| **Context management** | Keeping the trajectory inside the window as it grows. |
| **Compaction / summarisation** | Compressing earlier steps to free context. |
| **Scratchpad** | Working memory the agent writes to and reads back. |
| **Checkpointing** | Persisting state so a long run can resume. |
| **Human-in-the-loop** | Required confirmation before an irreversible action. |
| **Guardrail** | A hard constraint outside the model (permissions, allow-lists, budgets). |
| **Determinism gap** | Same input, different trajectory. Inherent with sampling. |
| **Compounding error** | Per-step reliability `r` over `n` steps gives `r^n`. |

---

## Concepts

### The arithmetic that governs everything

An agent that is 95% reliable per step:

| Steps | Success |
|-------|---------|
| 1 | 95% |
| 5 | 77% |
| 10 | 60% |
| 20 | 36% |
| 50 | 8% |

**This single table explains most of what is true about agent engineering.** Every design decision
follows from it:

- **Fewer steps beats smarter steps.** A tool that does in one call what took four is worth more
  than a model upgrade.
- **Per-step reliability compounds, so it is the highest-leverage metric.** Going 95% → 99% takes
  10 steps from 60% to 90%.
- **Recovery changes the base.** If an error is observable and recoverable, a failed step is not a
  failed episode, and the exponent stops being brutal.
- **Long horizons need checkpoints.** At 50 steps you cannot rely on getting through in one go.

Have this table in your head. It is the answer to "why are agents unreliable" and it is quantitative
rather than hand-wavy.

### The loop, minimally

```
state = initial_observation(task)
for step in range(MAX_STEPS):
    action = model(render(state))          # constrained decode (M10), T=1.0 for training (M08)
    if action.tool == "submit_answer":
        return terminal("submitted", action.args)
    if action.tool == "give_up":
        return terminal("gave_up")
    observation = execute(action)          # errors returned, not raised
    state.append(action, observation)
return terminal(None)                       # step limit — a DISTINCT outcome
```

Four details that are not incidental:

1. **`terminal is None` is its own outcome.** Hitting the step limit is not the same failure as
   submitting a wrong answer — one is a budget problem, the other a reasoning problem. Collapsing
   them destroys your diagnosis and your reward signal (Module 15).
2. **`give_up` must exist and must be rewarded when correct.** Without it, an agent facing an
   impossible task flails until the step limit and you cannot distinguish "couldn't" from "didn't
   know it couldn't".
3. **Errors are observations.** Otherwise recovery is unlearnable (Module 10).
4. **Every step is a full forward pass over the whole growing trajectory.** Cost is quadratic-ish
   in steps, not linear.

### Context growth is the operational problem

Step 1 renders ~500 tokens. Step 8 renders every prior action and observation — easily 8,000. So:

- Cost per step *rises* through the episode.
- Prefill dominates (Module 03) and prefix caching is critical (Module 09).
- Long episodes eventually hit the window.

Strategies, with trade-offs:

| Strategy | Keeps | Loses |
|----------|-------|-------|
| Full trajectory | everything | window and cost |
| Sliding window (last k steps) | recency | early discoveries — often the schema |
| Summarise old steps | gist | precision; summarisation can hallucinate |
| Structured scratchpad | what the agent chose to record | what it failed to record |
| Externalised memory + retrieval | scale | retrieval failures |

**For a DB agent the failure mode of a sliding window is specific and worth naming:** the schema
discovered in step 2 scrolls out by step 9, and the agent re-discovers it, burning steps. A
structured scratchpad — "facts established so far" carried forward verbatim — fixes exactly that
and is cheap.

### Single agent or multi-agent

Multi-agent is fashionable and usually wrong. Costs it adds:

- **Context loss at every handoff.** The receiving agent does not have the sending agent's
  trajectory, only a summary. Summaries lose the detail that mattered.
- **Compounding across agents.** Three 90% agents in series is 73%.
- **Attribution collapse.** When the output is wrong, which agent's fault? Debugging is
  dramatically harder.
- **Latency and cost multiply.**

Genuine reasons to split:

| Reason | Why it is real |
|--------|----------------|
| Different tool *permissions* | A read-only analyst and a write-capable executor should be different principals |
| Different models by cost | Cheap router → expensive specialist |
| Genuinely parallel subtasks | Independent work with no shared state |
| Different fine-tuned adapters | One base, per-domain LoRA (Module 20) — the enterprise pattern |

That last row is the one that connects to the JD. "Domain-specific agents at enterprise scale"
usually means *one architecture, one base model, many adapters and many tool permission sets* — not
many bespoke agent frameworks.

**Default to a single agent with more tools. Split when you can name which of the four reasons
above applies.**

### Routing

A cheap classifier in front of an expensive model is the highest-ROI piece of most production
agent systems:

```
request -> router (small model / classifier)
             |-- simple lookup     -> direct SQL template, no LLM
             |-- standard query    -> 1.7B fine-tuned agent
             +-- complex/ambiguous -> large model, full agent loop
```

Route on predicted difficulty, and **make the fallback direction safe**: an easy task sent to the
big model costs money; a hard task sent to the small model produces a wrong answer. Bias the router
toward escalation and measure the escalation rate.

### Evaluation is different for agents

You cannot score an agent by comparing output strings. What you need (Module 23 goes deeper):

- **Outcome correctness** via an execution-based verifier, not text matching.
- **Steps to success** — efficiency is a real objective, not a nicety.
- **Per-step reliability** — the compounding base.
- **Recovery rate** — episodes that errored and still succeeded.
- **Failure taxonomy** — parse / selection / args / reasoning (Module 24).
- **Determinism gap** — variance across repeated runs of the same task.

**Run every task n times.** A single-run agent benchmark is noise. Report pass@1 and pass@k, and
report the variance; a system with 60% ± 25% is a different system from 60% ± 3% even though the
headline matches.

### Guardrails belong outside the model

| Guardrail | Mechanism | Holds under injection? |
|-----------|-----------|------------------------|
| Read-only DB role | Database permissions | **Yes** |
| Statement timeout | DB config | **Yes** |
| Row/char caps on output | Harness code | **Yes** |
| Step limit | Harness code | **Yes** |
| Token/cost budget | Harness code | **Yes** |
| Confirm before irreversible action | Harness code | **Yes** |
| Tool allow-list per agent | Harness code | **Yes** |
| "Please don't drop tables" | Prompt | **No** |

Everything enforced in code holds. Everything enforced in the prompt is a suggestion. This is the
same point as Module 09's injection ladder and it is worth repeating because it is the difference
between a demo and a system someone will deploy against a production database.

---

## Where it's used

- **The whole `learn-agent-posttrain` project.** Lab 01 built exactly this loop.
- **RL post-training** — the trajectory is the training datum (Modules 13, 15).
- **Enterprise deployments** — coding agents, support agents, data-analysis agents.
- **The interview** — this is the vocabulary Isaac's message is written in.

---

## Labs

These build directly on Lab 01 in the `learn-agent-posttrain` worktree.

### Lab 1 — Measure the compounding curve on your own agent

```python
import json
from collections import defaultdict

rollouts = [json.loads(l) for l in open("data/baseline.jsonl")]

# per-step reliability: fraction of steps that neither errored nor failed to parse
ok = tot = 0
for r in rollouts:
    for s in r["steps"]:
        tot += 1
        ok += 0 if (s.get("error") or s.get("parse_failed")) else 1
per_step = ok / tot
print(f"per-step reliability: {per_step:.3f}")

by_len = defaultdict(lambda: [0, 0])
for r in rollouts:
    n = len(r["steps"])
    by_len[n][0] += 1
    by_len[n][1] += int(r["reward"] == 1.0)

print(f"\n{'steps':>6}{'n':>6}{'observed':>11}{'predicted r^n':>15}")
for n in sorted(by_len):
    cnt, wins = by_len[n]
    print(f"{n:>6}{cnt:>6}{wins/cnt:>11.2f}{per_step**n:>15.2f}")
```

**Compare observed against `r^n`.** If observed beats the prediction, your agent is recovering from
errors — good, and worth quantifying. If it is worse, errors are correlated: one bad step poisons
the ones after it, which is exposure bias (Module 11) showing up as an operational number.

### Lab 2 — Context growth

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")

r = rollouts[0]
running = 0
print(f"{'step':>5}{'render tok':>12}{'cumulative':>12}")
for i, s in enumerate(r["steps"], 1):
    n = len(tok(s["rendered_prompt"], add_special_tokens=False)["input_ids"])
    running += n
    print(f"{i:>5}{n:>12}{running:>12}")
print(f"\ntotal prefill tokens for one 
{len(r['steps'])}-step episode: {running:,}")
```

Then multiply by episodes/day. This number is why prefix caching is not optional, and it is a
strong, concrete thing to bring up when asked about cost.

### Lab 3 — Context strategy ablation

Four variants of `render(state)` over the same tasks:

| Variant | Rule |
|---------|------|
| A | Full trajectory |
| B | Sliding window, last 3 steps |
| C | Sliding window + a persistent "facts established" scratchpad |
| D | Full trajectory, but observations truncated to 500 chars |

Report success rate, mean steps, mean prefill tokens per episode.

**The expected finding:** B is cheapest and worst, and it is worst *specifically* because schema
discovered early scrolls out and gets re-discovered — check by counting repeated `describe_table`
calls on the same table. C recovers most of A's accuracy at close to B's cost. That is a real,
defensible engineering result.

### Lab 4 — Add a router

```python
def route(question: str) -> str:
    q = question.lower()
    if any(k in q for k in ["how many", "count of", "total number"]) and "join" not in q:
        return "template"          # no LLM at all
    if len(q.split()) < 12:
        return "small"
    return "large"
```

Run the full task set through it and report:

| | share | accuracy | mean cost |
|---|---|---|---|
| template | | | |
| small | | | |
| large | | | |

Then compute blended accuracy and cost against always-large. **Also report misroute cost** — tasks
sent to `template` or `small` that the large model would have solved. That is the number that
decides whether the router ships.

### Lab 5 — Guardrails under adversarial input

Add tasks whose *database content* contains injected instructions (Module 09 Lab 7 poisoning, but
inside a real table row). Confirm empirically:

1. The agent sometimes attempts the injected action.
2. The `agent_ro` role makes the attempt fail harmlessly.
3. The attempt appears in your logs as a distinct, detectable event.

**Point 3 is what people forget.** A blocked attack you cannot see is indistinguishable from no
attack. Add an explicit log line when a tool call is refused by permissions.

### Lab 6 — Determinism gap

```python
import statistics, collections

runs = collections.defaultdict(list)
for r in rollouts:
    runs[r["task_id"]].append(r["reward"])

rates = [statistics.mean(v) for v in runs.values()]
print(f"tasks: {len(rates)}")
print(f"always pass : {sum(1 for x in rates if x == 1.0)}")
print(f"always fail : {sum(1 for x in rates if x == 0.0)}")
print(f"variable    : {sum(1 for x in rates if 0 < x < 1)}")
print(f"mean pass@1 : {statistics.mean(rates):.2f}")
```

The **variable** bucket is the interesting one twice over: it is where your headline number is
unstable, and — per Module 15 — it is the *only* bucket that produces a GRPO gradient. Always-pass
and always-fail tasks are both dead weight for training.

### Lab 7 — Single vs multi-agent, honestly

Split your SQL agent into two: a **schema explorer** (discovery tools only) that hands a schema
summary to a **query writer** (`run_sql` + `submit_answer` only).

Measure success rate, total steps, total tokens, and latency against the single agent.

**Expect the multi-agent version to lose**, and be able to say precisely why: the handoff summary
drops detail the query writer needed — most often a column's exact type or a value's exact
spelling. That is a result worth having, because in an interview "we tried multi-agent and it was
worse for this reason" is a far stronger answer than either enthusiasm or dismissal.

---

## Failure modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| Long tasks nearly always fail | Compounding `r^n` | Fewer steps; raise per-step reliability; enable recovery |
| Agent re-discovers the same schema | Sliding window dropped it | Persistent scratchpad of established facts |
| Cost per episode much higher than expected | Context grows every step | Prefix caching; truncate observations |
| Episodes end at the step limit | Horizon too short, or agent looping | Distinguish `terminal is None`; add loop detection |
| Agent loops on the same failing call | No memory of what it tried | Include prior failed actions in the render |
| Cannot tell why it failed | No failure taxonomy | Module 24 |
| Benchmark number moves run to run | Single run per task | n runs, report pass@1/pass@k and variance |
| Multi-agent worse than single | Context loss at handoff | Merge back, or pass full trajectory |
| Injected instruction executed | Prompt-level defence only | Permissions, budgets, confirmations |
| Blocked attack invisible | No logging on refusal | Log every permission denial |
| Agent flails on impossible tasks | No `give_up` | Add it and reward correct use |

---

## Interview

**"Why are agents unreliable?"**
Compounding. At 95% per-step reliability, ten steps is 60% and twenty is 36% — the arithmetic
dominates everything else. So the levers in order are: reduce steps, which means better tools
rather than a better model; raise per-step reliability, where 95→99% takes ten steps from 60% to
90%; and make errors recoverable, which changes the base entirely because a failed step stops being
a failed episode. Most of what looks like "the model isn't smart enough" is actually one of those
three.

**"Single agent or multi-agent?"**
Single, unless I can name the reason. Handoffs lose context — the receiving agent gets a summary,
not the trajectory, and the detail that mattered is usually what the summary dropped. Reliability
compounds across agents too, and attribution collapses so debugging gets much harder. The reasons
that are real: different tool permissions, so a read-only analyst and a write-capable executor
should be separate principals; cost routing; genuinely parallel independent subtasks; and different
fine-tuned adapters over one base. That last one is usually what "domain-specific agents" means at
enterprise scale — one architecture, many LoRAs, many permission sets, not many frameworks.

**"How do you keep a long agent trajectory in context?"**
I'd measure first, because prefill cost rises every step and it's usually larger than people
expect. Then: truncate observations aggressively — a 500-character cap on tool output costs almost
nothing in accuracy. Keep a structured scratchpad of established facts, carried forward verbatim,
rather than a naive sliding window; the specific failure of a sliding window on a DB agent is that
the schema discovered in step 2 scrolls out and gets re-discovered, burning steps. And order the
prompt stable-content-first so prefix caching actually hits.

**"How do you stop an agent doing something destructive?"**
Outside the model. Read-only database role, statement timeout, row caps, step limit, cost budget,
tool allow-list, and human confirmation on anything irreversible. All of those are enforced in code
and hold regardless of what the model was persuaded to attempt. Prompt instructions are
suggestions — anything in the context can look like an instruction and the model can't reliably
tell data from directive. I'd also log every permission denial, because a blocked attack you can't
see is indistinguishable from no attack.

**"How do you evaluate an agent?"**
Not by string comparison — by executing the result and comparing outcomes, so a differently-written
but correct query still passes. Then n runs per task rather than one, reporting pass@1, pass@k and
variance, because a single run is noise and 60% ± 25% is a different system from 60% ± 3%.
Alongside outcome: steps to success, per-step reliability, recovery rate, and a failure taxonomy
that separates parse errors, wrong tool, wrong arguments and wrong reasoning — because those four
have four different fixes and one aggregate number tells you which lever to pull, which is none.

---

## Checkpoint

1. Reproduce the compounding table and your agent's own per-step reliability.
2. Report observed success vs `r^n` and interpret the gap.
3. Report total prefill tokens for one episode and per day.
4. Produce the four-way context-strategy ablation.
5. Report router shares, blended accuracy, and misroute cost.
6. Demonstrate a blocked injection *and* its log line.
7. Report always-pass / always-fail / variable task counts.
8. Report the single vs multi-agent comparison and name the mechanism behind the difference.

---

**Next:** [23 — Evaluation](23-evaluation.md) ·
**Back:** [21 — RAG](21-rag.md) · [Syllabus](../SYLLABUS.md)
