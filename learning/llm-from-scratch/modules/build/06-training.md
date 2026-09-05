# B6 — Training it for real (Phase 1: pretraining)

This is Phase 1 of the two-stage pipeline: a pure next-token predictor learns grammar, spelling,
syntax and association from raw text. It will not follow instructions when this lesson ends — that is
the expected result, not a failure, and B9 is where it changes.

You have a model. Now make it produce English. Everything here is a knob you will turn on real
models: warmup, cosine decay, gradient clipping, weight decay on the right subset of parameters,
mixed precision, checkpointing, and the discipline of knowing whether a run is healthy at step 200
instead of finding out at step 20,000.

Target: **val loss ≈ 1.5–1.8 bits/token on TinyStories, coherent multi-sentence output, ~45 minutes
on your M3 Pro.**

Reference companions: [Module 06](../06-pretraining.md), [Module 17](../17-distributed.md).

---

## Terms

| Term | Meaning |
|------|---------|
| **Step** | One optimizer update. |
| **Warmup** | Linear LR ramp from ~0 over the first few hundred steps. |
| **Cosine decay** | LR following a cosine curve from peak to a floor (usually `0.1 × peak`). |
| **Gradient clipping** | Rescaling the gradient when its global norm exceeds a threshold. |
| **Grad norm** | `‖g‖₂` over all parameters — your single best health metric. |
| **Weight decay** | L2-style pull toward zero, applied to matrices but **not** norms/biases/embeddings. |
| **Gradient accumulation** | Summing grads over micro-batches to simulate a larger batch. |
| **Effective batch** | `micro_batch × accum_steps × block_size` tokens per update. |
| **Mixed precision** | Compute in bf16/fp16, keep master weights in fp32. |
| **MPS** | Metal Performance Shaders — PyTorch's Apple Silicon GPU backend. |
| **Unified memory** | Apple Silicon's shared CPU/GPU RAM. No host↔device copy. |
| **MFU** | Model FLOPs Utilisation — achieved FLOPs ÷ hardware peak. |
| **Checkpoint** | Serialised model + optimizer + step + config + RNG state. |
| **Grokking-ish plateau** | Loss flat then dropping sharply — normal early in tiny-model training. |

---

## Concepts

### The learning-rate schedule is not decoration

Three phases, each solving a specific problem:

**Warmup (first ~2–5% of steps).** At step 0 every parameter is random, so gradients are large and
uninformative, and Adam's second-moment estimate `v` is still near zero — which makes the effective
step size `m/√v` enormous. Warmup ramps the LR linearly from ~0 while Adam's moment estimates
stabilise. Skip it and the first hundred steps can permanently damage the model: you get a loss that
recovers to a *worse* plateau than a warmed-up run, which is the confusing part — it doesn't diverge,
it just quietly ends up worse.

**Peak.** For a model this size, `3e-4` to `1e-3`. The scaling relationship is roughly
`lr ∝ 1/√d_model` — bigger models need smaller LRs, which is why you cannot copy GPT-3's `6e-5`
onto a 3M model and expect anything.

**Cosine decay to `0.1 × peak`.** Late in training you want small steps to settle into a minimum
rather than bounce around it. Decaying to exactly zero wastes the final steps; the `0.1` floor is
the convention that stuck.

```python
def get_lr(step, warmup=200, max_steps=5000, peak=1e-3, floor_frac=0.1):
    if step < warmup:
        return peak * (step + 1) / warmup
    if step > max_steps:
        return peak * floor_frac
    prog = (step - warmup) / (max_steps - warmup)
    return peak * (floor_frac + (1 - floor_frac) * 0.5 * (1 + math.cos(math.pi * prog)))
```

**The schedule is tied to `max_steps`.** Stopping a cosine run early leaves the model at a high LR
and mid-bounce — its loss is worse than the same compute spent on a correctly-sized schedule. This is
why you cannot honestly compare a run you killed at 60% against one that finished.

### Gradient clipping is the cheapest insurance in deep learning

One bad batch — an unusual document, a rare token — can produce a gradient orders of magnitude larger
than typical. One such step can undo thousands of good ones.

```python
norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

It rescales the whole gradient vector to norm 1.0 when it exceeds it, preserving direction and
discarding only magnitude. Costs one extra pass over the gradients; prevents the class of failure
where the loss spikes to 8 and never recovers.

**Log the returned norm.** It is the single most informative training signal you have:

| Grad-norm behaviour | Meaning |
|---------------------|---------|
| Starts ~1–5, decays to ~0.1–0.5, stable | Healthy |
| Constantly clipped after warmup | LR too high |
| Spikes to 100+ at one step | Bad batch or data bug |
| Collapses to ~0 with loss flat | Dead — check init, LR, `zero_grad` |
| Slowly climbing across the run | Instability building; expect divergence |

### Weight decay does not belong on every parameter

AdamW's decay pulls weights toward zero. That is a sensible prior for a weight *matrix*. It is
actively wrong for:

- **RMSNorm gains** — decaying `γ` toward 0 shrinks activations, which the network must fight.
- **Biases** — one number per feature, no capacity to overfit.
- **Embeddings** — arguable, and commonly excluded in small models where they are a quarter of the
  parameters.

The idiom, worth being able to write from memory:

```python
def configure_optimizer(model, lr, weight_decay=0.1, betas=(0.9, 0.95)):
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (decay if p.dim() >= 2 else no_decay).append(p)
    groups = [{"params": decay, "weight_decay": weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=lr, betas=betas, eps=1e-8, fused=False)
```

**`p.dim() >= 2` is the whole trick**: matrices are 2D, norms and biases are 1D. Note
`betas=(0.9, 0.95)` — LLM training uses 0.95 rather than the PyTorch default 0.999, because the
longer-memory second moment adapts too slowly to the loss landscape here.

### Batch size, and what "effective batch" means

Tokens per update = `micro_batch × accum_steps × block_size`. For `32 × 1 × 256` that is 8,192.

Small batches give noisy gradients (which regularise, up to a point); large batches give clean ones
and better hardware utilisation. When memory bounds you, **gradient accumulation** decouples the two:

```python
for micro in range(accum_steps):
    xb, yb = get_batch("train")
    _, loss = model(xb, yb)
    (loss / accum_steps).backward()     # divide, or your effective LR scales with accum_steps
optimizer.step()
optimizer.zero_grad(set_to_none=True)
```

**The `/ accum_steps` is the bug everyone writes once.** Omit it and gradients sum rather than
average, so your effective learning rate is multiplied by `accum_steps` — the run diverges and the
cause is invisible in the code.

### Apple Silicon specifics

Your M3 Pro's unified memory is a genuine architectural advantage: there is no host→device copy, so
`.to(device)` on Apple Silicon is far cheaper than the equivalent CUDA transfer, and a dataset that
fits in RAM is already "on the GPU".

```python
device = ("mps" if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available() else "cpu")
```

Four things that are specific to this backend and will cost you time if you don't know them:

- **bf16 works, fp16 mostly isn't worth it.** Use `torch.autocast("mps", dtype=torch.bfloat16)`.
  `torch.cuda.amp.GradScaler` is CUDA-only and unnecessary with bf16 anyway (bf16 has fp32's
  exponent range, so there is nothing to scale).
- **`torch.compile` support is partial.** Try it, measure, and fall back without ceremony if it
  errors or gets slower — on this backend it often does.
- **MPS is asynchronous.** Timings are meaningless without `torch.mps.synchronize()` before reading
  the clock, which is the classic "my model trains in 0.1ms" mistake.
- **A 3.4M model is far too small to saturate this GPU.** Expect low MFU and CPU-side overhead to
  dominate. That is fine and it is worth knowing *why* — dataloading and Python launch overhead, not
  the math.

### Reading a loss curve

For your model on TinyStories with a 4096 vocab:

| Step | Expected loss | What just happened |
|------|---------------|--------------------|
| 0 | **8.32** = `ln(4096)` | Uniform. If not, stop — you have a bug. |
| ~50 | ~6.0 | Unigram frequencies learned. Predicts common tokens. |
| ~200 | ~4.5 | Bigram-level structure; spacing and common words. |
| ~1000 | ~2.5 | Words are real; grammar emerging. |
| ~3000 | ~1.9 | Sentences parse. Names stay consistent within a story. |
| ~5000 | ~1.6 | Coherent short stories. Occasional logic errors. |

**Val loss below train loss** means your split leaks (B3 Lab 5). **Val flat while train falls** means
overfitting — at 3.4M params on 100MB of text you should not see this, so if you do, you are training
on a subset far smaller than you think.

---

## Where it's used

- Every pretraining run at every scale is this loop. Llama 3's differs in parallelism, not structure.
- The grad-norm table is how on-call engineers triage a 1000-GPU run at 3am.
- Checkpoint/resume correctness is what makes multi-day runs survivable.
- Modules 11–15 all continue training *from* the checkpoint this lesson produces.

---

## Labs

### Lab 1 — The training loop, complete

```python
import math, time, os, torch

cfg = Config()
model = TinyLM(cfg).to(device)
opt   = configure_optimizer(model, lr=1e-3)

MAX_STEPS, WARMUP, EVAL_EVERY, ACCUM = 5000, 200, 250, 1
best_val, t0 = float("inf"), time.time()

for step in range(MAX_STEPS + 1):
    lr = get_lr(step, WARMUP, MAX_STEPS, peak=1e-3)
    for g in opt.param_groups:
        g["lr"] = lr

    if step % EVAL_EVERY == 0:
        losses = estimate_loss(model, iters=50)
        el = time.time() - t0
        print(f"step {step:>5} | train {losses['train']:.4f} | val {losses['val']:.4f} "
              f"| lr {lr:.2e} | {el/60:.1f}m")
        if losses["val"] < best_val:
            best_val = losses["val"]
            torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                        "step": step, "cfg": cfg, "val": best_val,
                        "rng": torch.get_rng_state()}, "ckpt_best.pt")

    for _ in range(ACCUM):
        xb, yb = get_batch("train")
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=(device != "cpu")):
            _, loss = model(xb, yb)
        (loss / ACCUM).backward()

    gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad(set_to_none=True)

    if step % 100 == 0:
        print(f"  step {step:>5}  loss {loss.item():.4f}  |g| {gnorm:.3f}")
```

**Before committing to 5000 steps, run 20 and check three things:** step-0 loss is 8.32 ± 0.3, loss
is falling by step 20, and the grad norm is order 1 rather than order 100. Those three checks cost 30
seconds and catch nearly every bug in this lesson.

### Lab 2 — Warmup ablation

Three runs, 1500 steps each, identical except warmup: **0, 50, 200**.

Plot all three. **The no-warmup run's loss spikes in the first ~30 steps and settles onto a visibly
higher plateau than the others** — it does not diverge, which is precisely what makes it dangerous:
without the comparison you would never know you had lost anything.

Log the grad norm for step 0–50 of each and report the peak. That number is the mechanism.

### Lab 3 — Learning-rate sweep

`3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2`, 1000 steps each.

| LR | Expect |
|----|--------|
| 3e-5 | Falls, far too slowly. Compute wasted. |
| 1e-4 | Safe, slow. |
| **3e-4 – 1e-3** | **Best final loss.** |
| 3e-3 | Fast then unstable; grad norm clipped constantly. |
| 1e-2 | Diverges or plateaus around 6. |

Plot final loss against `log(lr)`. **The U-shape you get is the real reason the "LR range test"
exists** — and having drawn it on your own model, you will never again treat a learning rate as a
number copied from a config.

### Lab 4 — Grad-norm forensics

Deliberately corrupt one batch in 500 with a pathological input (e.g. a single token repeated 256
times) and log the norm. **You will see the spike, and see clipping absorb it.** Then disable
clipping and rerun — the loss jumps and takes hundreds of steps to recover, or does not.

This is what "we had a data bug at step 41,000" looks like from the inside.

### Lab 5 — Checkpoint and resume, verified

Naive resume is silently wrong. Prove yours is right:

```python
# run A: 500 steps, save
# run B: load at 500, run 100 more
# run C: 600 steps straight through, same seed
assert abs(loss_B - loss_C) < 1e-3, "resume is not exact"
```

**To pass you must restore four things: model state, optimizer state (Adam's `m` and `v`), the step
counter (the LR schedule depends on it), and the RNG state (batch order).** Miss the optimizer state
and the model takes ~100 steps to recover — a rounding error at this scale, and days of compute at
production scale. Miss the step counter and the LR schedule silently restarts at warmup.

### Lab 6 — Throughput and MFU

```python
torch.mps.synchronize() if device == "mps" else None
t0 = time.time()
for _ in range(50):
    xb, yb = get_batch("train")
    _, loss = model(xb, yb)
    loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
torch.mps.synchronize() if device == "mps" else None
dt = time.time() - t0

tok_per_step = xb.numel()
print(f"{50*tok_per_step/dt:,.0f} tokens/sec")
flops_per_token = 6 * count_params(model, non_embedding=True)
print(f"{flops_per_token * 50 * tok_per_step / dt / 1e12:.3f} TFLOP/s")
```

Compare `cpu` vs `mps`, then `bf16` vs `fp32`, then batch 8/32/128. **Expect MFU in the low single
digits** — a 3.4M model cannot saturate this GPU, and the bottleneck is Python and dataloading, not
matmuls. Report where the time actually goes; that instinct transfers directly to real runs.

The `6N` constant is worth remembering: ≈2N FLOPs forward, ≈4N backward, per token.

### Lab 7 — Train the ladder

Three real runs. This is the artifact the rest of the track uses.

| Config | Params | Steps | Expected val | Wall clock |
|--------|--------|-------|--------------|------------|
| `tiny` d=128 L=4 | ~1.1M | 3000 | ~2.1 | ~10 min |
| **`base` d=192 L=6** | **~3.4M** | **5000** | **~1.6** | **~45 min** |
| `wide` d=320 L=8 | ~12M | 8000 | ~1.3 | ~3 h |

Keep all three checkpoints and their loss curves. **B8 fits a scaling law to exactly these three
points**, and B9 instruction-tunes the `base` checkpoint.

### Lab 8 — Confirm Phase 1's limit

The point of this lab is to see, concretely, what pretraining does **not** give you.

```python
for p in ["Once upon a time", "The little girl", "Can you solve this math problem?"]:
    print(f"--- {p!r}\n{generate(model, p, max_new=100)}\n")
```

The first two continue plausibly. **The third produces more questions, or a story about someone
asking a question — it does not answer.** The model has no concept of a request; it completes text.

Write down its actual output. In B9 you will run the identical prompt against the instruction-tuned
checkpoint and put the two side by side. That contrast is the clearest single demonstration of what
instruction tuning is, and it is a strong thing to be able to describe from your own model.

---

## Failure modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| Step-0 loss ≠ `ln(V)` | Init or data bug | Fix before training anything |
| Loss spikes early, settles worse | No warmup | 200 warmup steps |
| Loss → `nan` | LR too high; fp16 without scaler; norm in low precision | Clip, lower LR, bf16, upcast norms |
| Loss stuck at ~`ln(V)` | LR ~0; `zero_grad` before `step`; grads not flowing | Print `p.grad.norm()` for a few params |
| Grad norm always clipped | LR too high | Halve it |
| Val < train | Leaky split | B3 Lab 5 |
| Loss jumps on resume | Optimizer state not restored | Save and load `opt.state_dict()` |
| LR restarts after resume | Step counter not restored | Save `step` |
| Diverges only with accumulation | Missing `/ accum_steps` | Divide the loss |
| "0.1 ms/step" on MPS | Async dispatch not synchronised | `torch.mps.synchronize()` |
| Slower on GPU than CPU | Batch too small; overhead-bound | Raise batch; this is expected at 3.4M |
| Output is fluent nonsense | Working as intended at this scale | B9 |

---

## Interview

**"Walk me through your training loop."**
Cosine LR with linear warmup over the first few hundred steps, AdamW with betas 0.9/0.95, weight
decay 0.1 on 2D parameters only, global grad-norm clipping at 1.0, bf16 autocast, and checkpointing
on best validation loss with model, optimizer, step and RNG state so resume is bit-exact. I verify
resume by comparing a resumed run against a straight-through run at the same seed and asserting the
losses match to 1e-3. The metric I watch is not the loss, it's the gradient norm — loss falling
smoothly while the grad norm climbs means the run is going to diverge and you have maybe a thousand
steps of warning.

**"Why warmup?"**
At step 0 the weights are random and Adam's second-moment estimate is near zero, so the effective
step `m/√v` is enormous exactly when the gradients are least informative. Warmup ramps the LR while
those estimates stabilise. What makes it worth insisting on is the failure mode: without warmup the
run usually doesn't diverge, it just settles on a permanently worse plateau. I ablated it on my own
3M model — no warmup versus 200 steps of warmup, same seed — and the no-warmup run's loss was
visibly higher for the rest of training after a spike in the first thirty steps.

**"Why exclude some parameters from weight decay?"**
Decay is a prior that a parameter should be near zero, which is reasonable for a weight matrix and
wrong for a normalisation gain — pulling RMSNorm's γ toward zero shrinks activations the network then
has to fight. Same for biases, which have no capacity to overfit. The implementation is a one-liner:
decay parameters with `dim() >= 2`, don't decay the rest. Embeddings are the arguable case; on a
small model where they're a quarter of the parameters I leave them out.

**"How would you debug a loss that goes to nan at step 3000?"**
First, is it reproducible at the same step with the same seed — if yes it's data, if no it's numeric.
For data, dump the batch at 2999 and look for degenerate input. For numerics, check the grad norm
just before: a climb over the preceding hundred steps means the LR is too high and clipping was
masking it. Then check precision — fp16 without a loss scaler, or an RMSNorm computed in bf16, both
produce exactly this. Cheapest immediate mitigation is to resume from the last good checkpoint with
a lower LR and tighter clipping, then fix the actual cause.

**"What did you learn from training at this scale that transfers?"**
That the diagnostics are scale-invariant even though the numbers aren't. Step-0 loss must equal
`ln(vocab)`; the grad norm tells you about the run's health before the loss does; a validation loss
below training loss means your split leaks, not that you got lucky. I also learned where the time
actually goes — at 3.4M on an M3 Pro, MFU is a few percent because Python and dataloading dominate,
not matmuls, and that intuition about being overhead-bound versus compute-bound is the same question
you ask on a large cluster.

---

## Checkpoint

1. `base` trained to val ≤ 1.8, curve saved, wall-clock reported.
2. Warmup ablation, three curves, with the step-0–50 peak grad norm for each.
3. LR sweep plotted as final loss vs `log(lr)`; identify the U.
4. Grad-norm forensics with and without clipping.
5. Bit-exact resume asserted to 1e-3; name the four things you had to restore.
6. Tokens/sec and MFU for `cpu`/`mps` × `fp32`/`bf16`; explain the number.
7. All three ladder checkpoints saved with curves — B8's input.
8. The Phase-1 limit recorded verbatim: what your model does with *"Can you solve this math
   problem?"* B9 compares against it.

---

**Next:** [B7 — Sampling from your own logits](07-sampling.md) ·
**Back:** [B5 — The transformer, assembled](05-transformer.md) · [Build Track](README.md)
