# `notes/` — your measurements

One file per lesson, named after the lesson: `b0-foundations.md`, `b1-first-model.md`, and so on
down to `b10-reasoning.md`. Copy [`_template.md`](_template.md) at the *start* of each lesson and
fill it in as you go — writing it afterwards from memory is how numbers get invented.

## Why this folder is tracked in git

`code/` proves you can write it. `notes/` proves you ran it and understood what came back. The
second is rarer and harder to fake: a fitted scaling law with a prediction you tested, an ablation
table where the no-warmup run is visibly worse, a parameter count you derived on paper before the
code agreed with you.

## Why three lessons will not work without it

These are not optional records; later lessons read them back.

| Lesson | Needs, from an earlier note |
|--------|-----------------------------|
| **B6** | The `ln(4096) = 8.32` expected initial loss from B0 Lab 2 — the number that tells you the run is healthy at step 0 rather than at step 20,000 |
| **B8** | The three loss curves from B6. A scaling law needs three points; without them you retrain for three hours |
| **B9** | The exact base-model output from B6 Lab 8. The before/after is the whole demonstration, and once the weights are fine-tuned the "before" is gone |

## What a good note looks like

Numbers, with the context that makes them mean something:

> step 0 loss 8.31 (ln 4096 = 8.32 ✓), final val 1.58, 43 min on MPS, grad norm settled ~0.3,
> no-warmup run plateaued at 1.71

Not: *"trained the model, it worked."*

Record the failures too. "Loss went to NaN at step 340 with lr=3e-3; 1e-3 was stable" is a better
interview answer than any success, because it is the sentence that proves you were driving.
