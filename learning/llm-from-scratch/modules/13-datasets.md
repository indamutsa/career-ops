# Module 13 — Fine-tuning dataset construction

This is the module that separates people who *ran* a fine-tune from people who *own* a fine-tuning
programme. Almost every disappointing fine-tune is a data problem, not a method problem, and almost
every interview for a post-training role probes this even when the JD does not say so.

---

## Terms

| Term | Meaning |
|------|---------|
| **Instruction dataset** | `(instruction, [input], output)` triples used for SFT. |
| **Conversation dataset** | Multi-turn `messages` lists, the native format for chat models. |
| **Trajectory dataset** | Full agent episodes: observations, actions, rewards. The format for Modules 14–15. |
| **Seed set** | A small, hand-written, high-quality set used to bootstrap generation. |
| **Self-Instruct** | Bootstrap a large set by prompting a model with a few seed examples. |
| **Evol-Instruct** | Iteratively make instructions harder (deepen, add constraints, complicate reasoning). |
| **Distillation data** | Outputs from a stronger model used as targets for a weaker one. |
| **Rejection sampling** | Sample k completions, keep only those a verifier accepts. |
| **Best-of-n** | Sample n, keep the highest-scoring one. |
| **Deduplication** | Removing near-identical examples. MinHash/LSH for scale, embeddings for semantics. |
| **Decontamination** | Removing training examples that overlap your eval set. |
| **n-gram overlap** | The standard contamination test: shared 13-grams between train and eval. |
| **Loss masking** | Computing loss only on assistant tokens, not on prompts. |
| **Packing** | Concatenating short examples into full-length sequences for throughput. |
| **Data mixture** | The proportions of each source in the final blend. |
| **Replay / rehearsal** | Mixing in general data to prevent forgetting. |
| **Quality filter** | A model or heuristic that scores and drops low-quality examples. |
| **LIMA hypothesis** | ~1k excellent examples beat 50k mediocre ones for style alignment. |
| **Provenance** | Recorded origin of each example. Non-negotiable for enterprise. |
| **Licence contamination** | Training on outputs whose licence forbids that use. |

---

## Concepts

### The dominant fact: quality beats quantity, sharply

The LIMA result — 1,000 carefully curated examples producing a strong instruction-follower — is not
a curiosity. It reflects what SFT actually does: it does not teach knowledge, it *selects a
behaviour* the pretrained model already contains. Selecting a behaviour needs consistency and
coverage, not volume.

Consequences you should be able to state:

- 1,000 excellent examples usually beat 50,000 scraped ones.
- **One systematic error repeated 200 times is learned as a rule.** A bad annotator, a buggy
  template, a mis-parsed field — the model absorbs it faithfully.
- Adding more data of the same kind past saturation adds nothing but training cost.
- The marginal value is in *coverage of cases you currently fail*, not in more of what you pass.

### The generation ladder

From cheapest/weakest to most expensive/strongest:

| Method | Cost | Quality | Use when |
|--------|------|---------|----------|
| Templated from structured data | trivial | rigid but perfectly consistent | Format/schema training |
| Self-Instruct from seeds | low | uneven; needs filtering | Bootstrapping breadth |
| Evol-Instruct on existing set | low | raises difficulty | Your set is too easy |
| Distillation from a strong model | medium | high; licence-bound | Allowed and you need scale |
| **Rejection sampling with a verifier** | medium | **high and self-verifying** | You have a checker |
| Human-written | high | highest | The last mile |

**For an agent task with a verifier, rejection sampling is the right default and you should say so
in the interview.** The loop is:

1. Run the current policy on tasks, sampling k rollouts each at temperature ~1.0.
2. Keep only trajectories the verifier marks correct.
3. Optionally keep only the shortest correct one per task (biases toward efficiency).
4. SFT on those.
5. Repeat.

This is "STaR" / rejection-sampling SFT. It needs no reward model, no RL machinery, no KL term, and
it captures a large fraction of what GRPO would give you. It is also the honest baseline any RL
result must beat — see Module 15's Lab 5.

### The three things that will silently ruin the run

**1. Contamination.** If your eval examples leak into training, your numbers are fiction and you
will ship a regression believing it is an improvement. Standard test: 13-gram overlap between each
training example and each eval example; drop training examples that hit. Do this *before* you
train, every time, and record it.

**2. Duplication.** Near-duplicates over-weight whatever they contain. Two passes:
- **Exact / near-exact:** MinHash + LSH over shingles, Jaccard ≥ 0.8.
- **Semantic:** embed and drop pairs above cosine ~0.95.

**3. Loss masking.** If you compute loss over the prompt tokens as well as the completion, you are
training the model to *generate user turns*. On a small dataset this is a real quality hit and it
is invisible — the loss curve looks fine. Every serious trainer masks prompts to `-100`.

### Format, and why it must match the tokenizer exactly

The chat template (Module 09) is part of the data. If you train with a hand-rolled
`"### Human:\n...\n### Assistant:\n"` format and serve behind a `<|im_start|>` template, the model
sees out-of-distribution input at inference and quality collapses in a way that looks like "the
fine-tune didn't work".

**Always** build the training string with `tokenizer.apply_chat_template`. Never hand-write it.

### Mixture design

A realistic SFT mixture for a domain agent:

| Slice | Share | Purpose |
|-------|-------|---------|
| Target task, correct trajectories | 50–60% | The behaviour you want |
| Target task, hard/near-miss cases | 10–15% | The boundary |
| Refusal / out-of-scope / "I don't know" | 5–10% | Not answering when it shouldn't |
| Format and tool-call adherence | 5–10% | Parseability |
| General instruction data (replay) | 15–25% | Prevents forgetting |

The replay slice is the one people skip and then discover their SQL agent can no longer hold a
conversation. Catastrophic forgetting is real and cheap to prevent.

### Provenance — the enterprise requirement

For an enterprise buyer, "where did this training data come from" is a procurement question, not a
nicety. Every example should carry:

```json
{
  "id": "...",
  "source": "rejection_sampling | human | distill:<model> | template",
  "created": "2026-09-04",
  "licence": "internal | apache-2.0 | cc-by-sa | proprietary-customer",
  "task_id": "...",
  "verifier_passed": true,
  "generator_model": "Qwen3-1.7B@<sha>",
  "review": "unreviewed | approved | rejected"
}
```

Two hard rules:
- **Never distill from a model whose terms forbid using outputs to train competing models.** This
  is a legal exposure, and being the person who raises it unprompted is a strong signal.
- **Customer data needs an explicit contractual basis** to enter a training set. PII scrubbing is
  necessary, not sufficient.

### Packing and loss masking interact

Packing concatenates short examples to fill the context window — a big throughput win when your
examples average 300 tokens and your window is 4096. But naive packing lets attention flow across
the boundary between two unrelated examples, so example B conditions on example A. Use a trainer
that applies **block-diagonal attention** across packed segments (TRL's `packing=True` with
FlashAttention position-ids handling), or accept the noise knowingly on short runs.

---

## Where it's used

- **Every SFT run.** Module 11 is the mechanics; this is what you feed it.
- **Preference data.** Module 14's pairs come from the same rollout pool — chosen and rejected
  must be responses *to the same prompt*, ideally from the same policy.
- **RL task sets.** Module 15's tasks need the 20–60% pass-rate band, which is a data-curation
  decision made here.
- **Evaluation.** Your eval set is built by the same discipline and must be frozen and quarantined
  from all of the above.

---

## Labs

### Lab 1 — Build a rejection-sampled SFT set from your agent

This uses `data/baseline.jsonl` from Lab 01 in the `learn-agent-posttrain` worktree.

```python
import json
from collections import defaultdict

rollouts = [json.loads(l) for l in open("data/rollouts.jsonl")]

by_task = defaultdict(list)
for r in rollouts:
    by_task[r["task_id"]].append(r)

kept, stats = [], {"all_fail": 0, "all_pass": 0, "mixed": 0}
for task_id, rs in by_task.items():
    correct = [r for r in rs if r["reward"] == 1.0]
    if not correct:
        stats["all_fail"] += 1
        continue
    if len(correct) == len(rs):
        stats["all_pass"] += 1
    else:
        stats["mixed"] += 1
    # shortest correct trajectory: biases toward efficient tool use
    best = min(correct, key=lambda r: len(r["steps"]))
    kept.append({
        "task_id": task_id,
        "messages": best["messages"],
        "meta": {
            "source": "rejection_sampling",
            "verifier_passed": True,
            "n_sampled": len(rs),
            "n_correct": len(correct),
            "steps": len(best["steps"]),
        },
    })

print(stats)
print(f"kept {len(kept)} of {len(by_task)} tasks")
with open("data/sft.jsonl", "w") as f:
    for k in kept:
        f.write(json.dumps(k) + "\n")
```

**The number that matters:** `all_fail`. Those tasks contribute nothing — no SFT example, and in
Module 15, no gradient either. They are your curriculum's upper boundary. If `all_fail` is most of
your set, the tasks are too hard for the current policy and you must add easier ones before either
SFT or RL will move.

### Lab 2 — Decontaminate

```python
import json, re
from collections import defaultdict

def ngrams(text, n=13):
    toks = re.findall(r"\w+", text.lower())
    return {" ".join(toks[i:i + n]) for i in range(max(0, len(toks) - n + 1))}

train = [json.loads(l) for l in open("data/sft.jsonl")]
evalset = [json.loads(l) for l in open("data/eval.jsonl")]

eval_grams = set()
for e in evalset:
    eval_grams |= ngrams(json.dumps(e))

clean, dropped = [], []
for t in train:
    if ngrams(json.dumps(t)) & eval_grams:
        dropped.append(t)
    else:
        clean.append(t)

print(f"kept {len(clean)}  dropped {len(dropped)} contaminated")
```

Run this on a set you *know* is contaminated (copy three eval tasks into train) and confirm it
catches exactly those three. A decontamination script you have not tested against a planted
positive is not a decontamination script.

### Lab 3 — Deduplicate

```python
from datasketch import MinHash, MinHashLSH
import json, re

def mh(text, num_perm=128):
    m = MinHash(num_perm=num_perm)
    toks = re.findall(r"\w+", text.lower())
    for i in range(max(1, len(toks) - 4)):
        m.update(" ".join(toks[i:i + 5]).encode())
    return m

data = [json.loads(l) for l in open("data/sft.jsonl")]
lsh = MinHashLSH(threshold=0.8, num_perm=128)
keep, dupes = [], 0
for i, d in enumerate(data):
    sig = mh(json.dumps(d))
    if lsh.query(sig):
        dupes += 1
        continue
    lsh.insert(str(i), sig)
    keep.append(d)
print(f"kept {len(keep)}  dropped {dupes} near-duplicates")
```

### Lab 4 — Prove loss masking matters

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
msgs = [
    {"role": "system", "content": "You are a SQL agent."},
    {"role": "user", "content": "How many orders were placed in March?"},
    {"role": "assistant", "content": '{"tool":"run_sql","args":{"q":"SELECT ..."}}'},
]

full = tok.apply_chat_template(msgs, tokenize=False)
prefix = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)

ids = tok(full, add_special_tokens=False)["input_ids"]
n_prefix = len(tok(prefix, add_special_tokens=False)["input_ids"])

labels = list(ids)
for i in range(n_prefix):
    labels[i] = -100

print(f"total {len(ids)}  masked {n_prefix}  supervised {len(ids) - n_prefix}")
print(f"fraction of loss on prompt if unmasked: {n_prefix / len(ids):.1%}")
```

**Record that percentage.** On short assistant turns it is routinely 70–85%, meaning an unmasked
run spends most of its gradient learning to write user questions. Then train the same 200 examples
both ways and compare held-out task accuracy. The gap is the lab's deliverable.

### Lab 5 — Mixture ablation

Build four mixtures from your rejection-sampled set and train identical LoRA configs on each:

| Run | Mixture |
|-----|---------|
| A | 100% target task |
| B | 80% target + 20% general instruction (replay) |
| C | 70% target + 10% refusal/out-of-scope + 20% replay |
| D | B, but with the 2× hardest tasks upsampled |

Evaluate each on **two** suites: your held-out agent tasks *and* a small general chat set.

Expected shape: A wins on-task and loses badly on general; B recovers general at a small on-task
cost; C adds the ability to decline; D helps if and only if your hard tasks were under-represented.
Write the four-by-two number table — that table is a genuine interview artefact.

### Lab 6 — Provenance-stamp everything

```python
import json, hashlib, datetime, subprocess

def stamp(example, source, generator=None, licence="internal"):
    payload = json.dumps(example, sort_keys=True).encode()
    example["meta"] = {
        "id": hashlib.sha1(payload).hexdigest()[:16],
        "source": source,
        "generator_model": generator,
        "licence": licence,
        "created": datetime.date.today().isoformat(),
        "code_rev": subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"]).decode().strip(),
        "review": "unreviewed",
    }
    return example
```

Apply it in every generation script from the start. Retrofitting provenance onto a dataset you
already trained on is not possible, and "we can't say where this came from" is the answer that
loses an enterprise deal.

---

## Failure modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| Great eval numbers, bad in production | Contamination | 13-gram decontamination before every run |
| Model generates user turns / talks to itself | Loss not masked to assistant tokens | Mask prompt tokens to `-100` |
| Fine-tune "did nothing" | Training format ≠ serving chat template | Always `apply_chat_template` |
| On-task good, everything else broken | No replay slice | Mix 15–25% general data |
| One weird behaviour reproduced exactly | A systematic error repeated in the data | Sample and read 50 examples by hand |
| More data stopped helping | Saturation; adding more of the same | Add coverage of failures, not volume |
| Examples bleeding into each other | Packing without block-diagonal attention | Trainer-side segment masking |
| Legal escalation post-launch | Distilled from a model that forbids it | Check terms *before* generating |
| Zero preference pairs from rollouts | All-pass or all-fail tasks | Curate to the 20–60% band |

---

## Interview

**"You have a budget for 10,000 examples. How do you spend it?"**
Not on 10,000 of the same thing. First I'd build a few hundred by hand to fix the format and define
what good looks like, then use those as seeds and as the verifier's spec. Then rejection sampling
against the verifier for volume, which is cheap and self-filtering. Then I'd look at *what still
fails* and spend human effort only there — the marginal value is in coverage of failures, not
volume. And I'd hold maybe 15–25% for general replay data so the model doesn't lose the abilities
it came with. Realistically I might ship 3,000 and have better results than 10,000 unfiltered.

**"How do you know your eval isn't contaminated?"**
13-gram overlap between every training example and every eval example, run before every training
job, with the drop count logged as part of the run's metadata. And I test the decontamination
script itself by planting known-contaminated examples and confirming it catches them — an untested
filter is a filter you're trusting on faith. Beyond that, the eval set is frozen and generated
separately from anything the training pipeline touches.

**"What's the most common cause of a fine-tune that doesn't work?"**
Format mismatch between training and serving. The model was trained on a hand-written prompt format
and served behind the tokenizer's chat template, so at inference it sees something it never saw in
training. The loss curve looks perfectly healthy, which is why it fools people. Second most common
is loss computed over the prompt as well as the completion — on short assistant turns that's most
of your gradient going into learning to write user messages.

**"Would you distil from a frontier model?"**
Depends entirely on the licence, and I'd check that before writing the generation script rather
than after. Several major providers' terms forbid using outputs to train competing models, and for
an enterprise product that's real legal exposure, not a technicality. Where it's permitted it's an
efficient way to get scale. Where it isn't, rejection sampling against a verifier gets you most of
the way with no licence question at all — which for an enterprise buyer is worth something on its
own.

**"Your agent passes 95% of your task set. What do you do?"**
Recognise the task set is the problem, not a success. At 95% there's almost no signal left — for
SFT nearly every rollout is already correct so there's nothing to learn, and for GRPO nearly every
group has zero variance and therefore zero gradient. I'd add harder tasks until the pass rate sits
in the 20–60% band, and separately check the 5% failures for whether they're genuinely hard or just
broken gold answers.

---

## Checkpoint

1. State the LIMA finding and why it follows from what SFT does.
2. Run decontamination against a planted positive and confirm it fires.
3. Report the fraction of tokens masked in a typical agent turn.
4. Produce the four-by-two mixture ablation table.
5. Explain the licence question in distillation without being prompted.
6. Explain why all-pass and all-fail tasks are both worthless.

---

**Next:** [14 — Preference optimisation](14-preference.md) ·
**Back:** [12 — PEFT](12-peft.md) · [Syllabus](../SYLLABUS.md)
