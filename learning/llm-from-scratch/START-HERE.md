# START HERE

You are going to build a language model from an empty file, train it on this laptop, and then use
it as the subject of every concept an AI Research Engineer interview can ask about.

**This is the only file you need to read to begin.** Everything else is reachable from here.

---

## How this is organised

The same content is two things: a folder of markdown in this repository, and a website with a
sidebar. On the website every page carries a short id — `L0`, `B4`, `M15`, `LG`. Those ids are the
map, so here is what they mean.

There are **two tracks**, shown as tabs at the top of the sidebar:

| Track | Ids | Pages | What it is |
|-------|-----|-------|------------|
| **Recall** | `00`, `A`–`G` | 43 | Maths → classical ML → deep learning → transformers → MLOps → mock rounds. Reading and recall, no code to write. A separate course; not part of this one. |
| **Build LLM** | `L`, `B`, `M`, `LG` | 44 | **This track.** You write the code. |

Within the Build LLM track, the first letter tells you what kind of page you are looking at:

| Prefix | Sidebar group | What it is | In the repo |
|--------|---------------|------------|-------------|
| **L** | `H · LLM track` | Orientation. Four documents that tell you what to do and when. | the four files at the track root |
| **B** | `I · Build labs` | The eleven build lessons, `B0` → `B10`. **This is the curriculum.** | `modules/build/` |
| **M** | `J · LLM concepts` | 28 reference modules, `M00` → `M25`. Read alongside the labs, never instead of them. | `modules/` |
| **LG** | `K · Reference` | The glossary — 175 terms. | `GLOSSARY.md` |

The `H`–`K` group letters just continue the Recall track's `A`–`G` so the two tabs share one
ordering. They carry no other meaning.

### The four `L` pages

`L` is a reading order, not a sequence of work. You are reading `L0` now.

| Id | File | The question it answers | When to read it |
|----|------|-------------------------|-----------------|
| **L0** | [`START-HERE.md`](START-HERE.md) | What am I doing, in what order, and where do the files go? | Now, once. |
| **L1** | [`PLAN.md`](PLAN.md) | Which lesson on which day, and what must be true at the end of each week? | After this file, then every Monday. |
| **L2** | [`SYLLABUS.md`](SYLLABUS.md) | What is in the 28 reference modules? | Step 5 — **not before.** |
| **L3** | [`modules/build/README.md`](modules/build/README.md) | Why build rather than read? What exactly are we building, and will this laptop take it? | After B0, when you want the whole shape at once. |

Everything else you will meet is a `B` lesson. Start at `B0` and go in order.

---

## The one rule

**The lessons are written. You write the code.**

Every code block in every lesson is meant to be *typed into your own editor and run* — not copied.
The lessons are the navigator; you are the driver. The bugs you make while typing are the actual
curriculum; a copied file teaches nothing.

Every lesson ends with a **Checkpoint**: a list of numbers you must have produced. If you cannot
state what changed, you have not finished that lesson.

---

## Step 1 — Make the workspace (1 minute)

Run this once. It creates the four folders every lesson writes into, so nothing later fails on a
missing directory:

```bash
cd learning/llm-from-scratch
mkdir -p code notes data checkpoints
ls
```

You should see `code  checkpoints  data  modules  notes` alongside the markdown files. What each
one is for:

| Folder | Holds | In git? |
|--------|-------|---------|
| `code/` | **Your code.** One file per lesson — `b0_autograd.py`, `b1_bigram.py`, … | Yes — it is the work. |
| `notes/` | **Your numbers.** One file per lesson: losses, timings, what broke. | Yes — it is the proof. |
| `data/` | Corpora and `.bin` token files. Hundreds of MB. | No — ignored. |
| `checkpoints/` | `ckpt_best.pt`, `ckpt_instruct.pt`, the tiny/base/wide ladder. | No — ignored. |

`.gitignore` already excludes `data/` and `checkpoints/` — they are large and regenerable. It does
**not** exclude `code/` or `notes/`, and that is deliberate: those two folders are the artifact you
show someone.

---

## Step 2 — Check the machine (2 minutes)

```bash
python3 code/check_env.py
```

It prints a PASS or FAIL line per requirement and exits non-zero if anything is missing. Install
whatever it names:

```bash
pip install -r requirements.txt
```

Then re-run `python3 code/check_env.py` until every line says PASS.

As of the last run on this machine: **Python 3.13.13, torch 2.13.0, MPS available, bf16 autocast
working.**

> **Prefer an isolated environment?** `python3 -m venv .venv && source .venv/bin/activate` then
> `pip install -r requirements.txt`. `.venv/` is gitignored. The curriculum has no dependencies
> beyond that file — no HuggingFace `transformers` until B5 Lab 6, no `trl`, no `peft`, nothing.

---

## Step 3 — Open the first lesson

```bash
open modules/build/00-foundations.md      # or read B0 on the site
```

Read it top to bottom, then do the labs in it. That is the whole instruction.

**Do not start with `SYLLABUS.md` (L2).** It is a reference index of 28 deep-dive modules, and
reading it first is how people spend an evening feeling productive without writing a line of code.
Come back to it in Step 5.

---

## Step 4 — Work the Build Track, B0 → B10

Eleven lessons. Each one leaves you something that runs.

| | Lesson | Time | You end with |
|---|--------|------|--------------|
| **B0** | [Foundations](modules/build/00-foundations.md) | 3–4 h | A scalar autograd engine you wrote, verified against PyTorch to 1e-6 |
| **B1** | [Tensors + first model](modules/build/01-first-model.md) | 2–3 h | A trained bigram LM, and a baseline loss to beat |
| **B2** | [Tokenizer from scratch](modules/build/02-tokenizer.md) | 3–4 h | Your own 4096-token BPE, trained on your corpus |
| **B3** | [Corpus → batches](modules/build/03-data.md) | 2–3 h | A memmapped token stream and a batch loader |
| **B4** | [Attention](modules/build/04-attention.md) | 3–4 h | Multi-head causal attention you implemented |
| **B5** | [The transformer](modules/build/05-transformer.md) | 4–5 h | A 3,443,136-parameter model, counted to the digit |
| **B6** | [Training](modules/build/06-training.md) | 3 h + 45 min run | **A model that writes coherent English** |
| **B7** | [Sampling](modules/build/07-sampling.md) | 2–3 h | Every decoding strategy + a KV cache, implemented |
| **B8** | [Scaling laws](modules/build/08-scaling.md) | 3 h + runs | A power law fitted to models *you* trained, and a tested prediction |
| **B9** | [Instruction tuning](modules/build/09-instruct.md) | 3 h | A base model **and** an instruct model, both yours |
| **B10** | [Teaching it to reason](modules/build/10-reasoning.md) | 4–5 h | CoT + GRPO against a verifier you wrote |

### The two files each lesson creates

Every lesson opens with the same two lines: the code file you are writing, and the notes file you
record into. Create them at the start of the lesson, not the end.

| Lesson | Create in your editor | Record your numbers in | Also produces |
|--------|-----------------------|------------------------|---------------|
| **B0** | `code/b0_autograd.py` | `notes/b0-foundations.md` | — |
| **B1** | `code/b1_bigram.py` | `notes/b1-first-model.md` | — |
| **B2** | `code/b2_tokenizer.py` | `notes/b2-tokenizer.md` | `data/tokenizer.json` |
| **B3** | `code/b3_data.py` | `notes/b3-data.md` | `data/train.bin`, `data/val.bin` |
| **B4** | `code/b4_attention.py` | `notes/b4-attention.md` | — |
| **B5** | `code/b5_model.py` | `notes/b5-transformer.md` | — |
| **B6** | `code/b6_train.py` | `notes/b6-training.md` | `checkpoints/ckpt_best.pt` |
| **B7** | `code/b7_sample.py` | `notes/b7-sampling.md` | — |
| **B8** | `code/b8_scaling.py` | `notes/b8-scaling.md` | `checkpoints/ckpt_{tiny,base,wide}.pt` |
| **B9** | `code/b9_instruct.py` | `notes/b9-instruct.md` | `checkpoints/ckpt_instruct.pt` |
| **B10** | `code/b10_reasoning.py` | `notes/b10-reasoning.md` | `checkpoints/ckpt_reason.pt` |

Starting B0, then, is exactly this:

```bash
touch code/b0_autograd.py notes/b0-foundations.md
open modules/build/00-foundations.md
```

The names are a convention, not an import contract — no lesson imports another lesson's file, so
nothing breaks if you rename one. Keep them anyway: `notes/` is read back in B8 and B9, and a
consistent tree is the difference between a repository someone reads and one they close.

**The shape of the whole thing** — the same three stages as every frontier model, four orders of
magnitude smaller:

```
[ Phase 1: Pretraining ]                          B0 - B8
       |  grammar, vocabulary, structure from raw text.
       |  Result: a completion engine. Ask it to solve a problem
       |  and it writes more problems.
       v
[ Phase 2: Instruction tuning + chain of thought ] B9 - B10
       |  request -> answer, and reasoning before answering:
       |  <prompt>...</prompt><thought>...</thought><response>...</response>
       v
[ Phase 3: RL from a verifiable reward ]           B10
       |  programmatic verifier. No reward model, no human labels.
       v
[ A small reasoning LLM that is entirely yours ]
```

### The schedule

Interview: **Wednesday 30 September 2026.** 25 days, four full weekends — enough for the full track
with no cutting.

**→ [`PLAN.md`](PLAN.md) (L1)** has the dated breakdown: four sprints, a hard gate at the end of
each, which reference modules to read alongside, and the five things you should be able to say on
the day.

Short version:

| Sprint | Dates | Gate |
|--------|-------|------|
| 1 | Sat 5 – Sun 13 Sep | B0–B5: parameter count derived by hand = `count_params()` |
| 2 | Mon 14 – Sun 20 Sep | B6–B8: **a model that writes English** + a tested scaling prediction |
| 3 | Mon 21 – Sun 27 Sep | B9–B10: instruct + reasoning checkpoints, length-growth plot |
| 4 | Mon 28 – Wed 30 Sep | **No new material** — write-up, drill, CV |

If you fall behind, say so on the day. The triage that works is dropping B0's autograd engine and
B2's tokenizer (use `tiktoken`) — never cutting B6 or B10.

---

## Step 5 — Read the reference modules alongside

[`SYLLABUS.md`](SYLLABUS.md) (L2) indexes 28 deep-dive modules — `M00`–`M25`, plus `M05a`
encoder-decoder and `M15a` DeepSeek-R1. Each has the same shape: **Terms · Concepts · Where it's
used · Labs · Failure modes · Interview Q&A · Checkpoint.**

Read them **in parallel with the Build Track, not before it.** Each build lesson names its
reference companion in its header block — read that module after finishing the lesson, while the
code is still in your head:

| Lesson | Reference companion |
|--------|---------------------|
| B0 | — B0 derives its own maths from nothing |
| B1 | [M02 Embeddings and the residual stream](modules/02-embeddings.md) |
| B2 | [M01 Tokenization](modules/01-tokenization.md) |
| B3 | [M13 Fine-tuning dataset construction](modules/13-datasets.md) · [M06 Pretraining](modules/06-pretraining.md) |
| B4 | [M03 Attention](modules/03-attention.md) |
| B5 | [M02 Embeddings](modules/02-embeddings.md) · [M04 Positional encoding](modules/04-positional.md) · [M05 The transformer block](modules/05-transformer-block.md) |
| B6 | [M06 Pretraining](modules/06-pretraining.md) · [M17 Distributed training](modules/17-distributed.md) |
| B7 | [M08 Decoding and sampling](modules/08-decoding.md) |
| B8 | [M06 Pretraining](modules/06-pretraining.md) — the scaling-law half |
| B9 | [M11 Supervised fine-tuning](modules/11-sft.md) · [M12 Parameter-efficient fine-tuning](modules/12-peft.md) |
| B10 | [M15 RL with verifiable rewards](modules/15-rlvr-grpo.md) · [M15a DeepSeek-R1](modules/15a-deepseek-r1.md) · [M14 Preference optimisation](modules/14-preference.md) |

[`GLOSSARY.md`](GLOSSARY.md) (LG) — 175 terms, each with what it is *and* where you'd meet it in
production. Read it end to end the night before the interview. Any term you can't place is a gap;
go to its module.

---

## Where things go

This is the tree after Step 1, with a couple of lessons done:

```
learning/llm-from-scratch/
  START-HERE.md          L0 — this file
  PLAN.md                L1 — the dated schedule
  SYLLABUS.md            L2 — index of the 28 reference modules
  GLOSSARY.md            LG — 175 terms
  README.md              the repository front door
  requirements.txt       every dependency the curriculum has
  .gitignore             excludes data/, checkpoints/, .venv/

  modules/
    build/               B0-B10 — the Build Track. Start at 00-foundations.md
      README.md          L3 — why build rather than read
    00-25 + 05a, 15a     M00-M25 — the reference modules

  code/                  ← YOUR code. One file per lesson.
    README.md            the naming convention
    check_env.py         the only lab file I wrote
    b0_autograd.py       yours, from here down
    b1_bigram.py
  notes/                 ← YOUR numbers. One file per lesson.
    README.md            what a good note contains
    _template.md         copy this at the start of each lesson
    b0-foundations.md
    b1-first-model.md
  data/                  corpora, tokenizer.json, .bin token files   (gitignored)
  checkpoints/           ckpt_best.pt, ckpt_instruct.pt, the ladder  (gitignored)

  build-site.mjs         regenerates the website from this markdown
```

**`notes/` is not optional.** Every lesson's checkpoint asks for measurements, and three later
lessons consume earlier ones directly: B8 fits a scaling law to the three loss curves from B6, and
B9's before/after comparison needs the exact base-model output you recorded in B6 Lab 8. If you
don't write them down, you re-run the training.

A `notes/b6-training.md` that says *"step 0 loss 8.31 (ln 4096 = 8.32 ✓), final val 1.58, 43 min,
grad norm settled ~0.3, no-warmup run plateaued at 1.71"* is worth more at interview time than any
paragraph you could write from memory.

---

## Working with me on this

I write lessons and review your code. I do not write your lab code — that was the deal, and it is
the reason this works.

Useful things to say:

- *"I'm on B4 Lab 3, my causality test fails"* — paste the code, I'll find it
- *"explain the √d_head scaling again, I typed it but I don't believe it"*
- *"here are my B6 numbers"* — I'll tell you whether the run is healthy
- *"drill me on attention"* — I'll ask the Interview questions from the module and grade you
- *"I have 5 days, re-plan"*

---

## Ground rules

- **Type the code.** Not copy — type.
- **CPU first, MPS second** for anything new. CPU is easier to debug and fast enough under 1M params.
- **Seed everything.** `torch.manual_seed(1337)`. You will be comparing runs constantly.
- **Commit after every lesson.** You will want a working state to go back to.
- **Save every loss curve.** B8 needs them and so does every ablation.
- **`torch.mps.synchronize()` before reading any clock.** MPS dispatch is async — without it your
  timing numbers are fiction, and this catches almost everyone once.

---

## Right now

```bash
cd learning/llm-from-scratch
mkdir -p code notes data checkpoints
pip install -r requirements.txt
python3 code/check_env.py
touch code/b0_autograd.py notes/b0-foundations.md
open modules/build/00-foundations.md
```

B0 starts at *"what is a probability distribution over sequences"* and assumes nothing after that.
