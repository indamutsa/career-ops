# START HERE

You are going to build a language model from an empty file, train it on this laptop, and then use
it as the subject of every concept an AI Research Engineer interview can ask about.

**This is the only file you need to read to begin.** Everything else is reachable from here.

---

## The one rule

**The lessons are written. You write the code.**

Every code block in every lesson is meant to be *typed into your own editor and run* — not copied.
The lessons are the navigator; you are the driver. The bugs you make while typing are the actual
curriculum; a copied file teaches nothing.

Every lesson ends with a **Checkpoint**: a list of numbers you must have produced. If you cannot
state what changed, you have not finished that lesson.

---

## Step 1 — Check the machine (2 minutes)

```bash
cd learning/llm-from-scratch
python3 code/check_env.py
```

As of the last run on this machine: **Python 3.13.13, torch 2.13.0, MPS available, bf16 autocast
working.** Two packages were missing:

```bash
pip install matplotlib datasets
```

Then re-run `python3 code/check_env.py` until every line says PASS.

> If you prefer an isolated environment: `python3 -m venv .venv && source .venv/bin/activate`
> then `pip install torch numpy matplotlib datasets regex`. The curriculum has no other
> dependencies — no HuggingFace `transformers` until B5 Lab 6, no `trl`, no `peft`, nothing.

---

## Step 2 — Open the first lesson

```
modules/build/00-foundations.md
```

Read it top to bottom, then do the labs in it. That is the whole instruction.

**Do not start with `SYLLABUS.md`.** It is a reference index of 28 deep-dive modules, and reading it
first is how people spend an evening feeling productive without writing a line of code. Come back to
it in Step 4.

---

## Step 3 — Work the Build Track, B0 → B10

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

**→ [`PLAN.md`](PLAN.md)** has the dated breakdown: four sprints, a hard gate at the end of each,
which reference modules to read alongside, and the five things you should be able to say on the day.

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

## Step 4 — Read the reference modules alongside

`SYLLABUS.md` indexes 28 deep-dive modules (00–25, plus `05a` encoder-decoder and `15a`
DeepSeek-R1). Each has the same shape: **Terms · Concepts · Where it's used · Labs · Failure
modes · Interview Q&A · Checkpoint.**

Read them **in parallel with the Build Track, not before it.** Each build lesson names its
reference companion at the top — read that one after finishing the build lesson, while the code
is still in your head.

`GLOSSARY.md` — 175 terms, each with what it is *and* where you'd meet it in production. Read it
end to end the night before the interview. Any term you can't place is a gap; go to its module.

---

## Where things go

```
learning/llm-from-scratch/
  START-HERE.md          this file
  SYLLABUS.md            index of the 28 reference modules
  GLOSSARY.md            175 terms
  modules/
    build/               ← the Build Track, B0-B10. Start at 00-foundations.md
    00-25 + 05a, 15a     the reference modules
  code/                  ← YOUR code. One file per lesson: b0_autograd.py, b1_bigram.py, ...
    check_env.py         the only file I wrote here
  data/                  corpora and .bin token files (large; keep out of git)
  checkpoints/           ckpt_best.pt, ckpt_instruct.pt, the tiny/base/wide ladder
  notes/                 ← your numbers. One file per lesson.
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
pip install matplotlib datasets
python3 code/check_env.py
open modules/build/00-foundations.md
```

B0 starts at *"what is a probability distribution over sequences"* and assumes nothing after that.
