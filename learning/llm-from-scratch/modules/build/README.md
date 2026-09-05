# The Build Track — one LLM, built from nothing, carrying everything

The reference modules (`../00` … `../25`) explain how language models work and borrow Qwen3 to
demonstrate it. This track does the opposite: **you build a complete language model from an empty
file, and then every concept in the curriculum is demonstrated on the model you built.**

That difference matters more than it sounds. When you quantize Qwen3 you are trusting that
something you did not write got smaller without breaking. When you quantize *your* model you know
what every one of its 3.4 million numbers is, where it came from, and what happens when you round
it. There is no layer of "it just works" left anywhere.

**Nothing is assumed.** Not gradient descent, not backpropagation, not autograd, not tensor shapes.
Lesson B0 starts at "what is a probability distribution over sequences" and derives everything from
there.

---

## The shape of the whole track

```
[ Phase 1: Pretraining ]                                   B0 - B8
       |  Learns grammar, vocabulary, structure from raw text.
       |  Result: a text completion engine. Ask it to solve a problem
       |  and it writes more problems.
       v
[ Phase 2: Instruction tuning + chain of thought ]          B9 - B10
       |  Learns the request -> answer contract, and to emit reasoning
       |  before answering, using strict structural anchors:
       |  <prompt> ... </prompt><thought> ... </thought><response> ... </response>
       v
[ Phase 3: RL from a verifiable reward ]                    B10
       |  Programmatic verifier, no reward model, no human labels.
       |  Optimises for being *right*, not for imitating.
       v
[ A small reasoning LLM that is entirely yours ]
```

Phase 1 builds the capability; Phase 2 makes it addressable; Phase 3 improves it. The same three
stages, in the same order, as every frontier model — the only difference is four orders of magnitude.

## Your hardware

An M3 Pro with 36 GB of unified memory is heavily over-specified for this, which is exactly what you
want while learning: nothing here will ever be blocked on memory.

- A 3.4M-parameter model is **~14 MB of weights**; with gradients and Adam state, ~55 MB. Add the
  activations and a full training run sits comfortably under 500 MB.
- Unified memory means there is no host-to-device copy — `.to("mps")` is far cheaper than the CUDA
  equivalent, and a dataset that fits in RAM is already reachable by the GPU.
- Use `bf16` autocast, not `fp16` — bf16 has fp32's exponent range, so no `GradScaler` is needed
  (and `torch.cuda.amp.GradScaler` is CUDA-only anyway).
- **MPS is asynchronous.** Call `torch.mps.synchronize()` before reading the clock or every timing
  you take is fiction. B6 Lab 6 covers this.
- Expect low MFU. A 3.4M model cannot saturate this GPU; Python and dataloading dominate. Knowing
  *why* you are overhead-bound rather than compute-bound is itself a transferable result.

```python
device = ("mps" if torch.backends.mps.is_available()
          else "cuda" if torch.cuda.is_available() else "cpu")
```

The headroom also means the ceiling is not this laptop: the same machine will fine-tune a quantized
7-8B model later if you want it to.

---

## The rule

**I write the lesson. You write the code.** Every code block here is meant to be typed into your
own editor and run, not copied blindly. The labs are the curriculum; the prose is scaffolding
around them.

Every lesson ends with a number you produced. If you cannot state what changed, you have not
finished it.

---

## The model you are going to build

| Hyperparameter | Value | Why this value |
|----------------|-------|----------------|
| `vocab_size` | 4096 | Small enough that embeddings don't dominate; big enough for real BPE |
| `d_model` | 192 | Divisible by 6 heads; `d_head = 32` is the classic size |
| `n_layers` | 6 | Deep enough for the depth/width ratio to be sane (`192/6 = 32`) |
| `n_heads` | 6 | `d_head = 192/6 = 32` |
| `d_ff` | 512 | SwiGLU, ≈ `8/3 × d_model` (Module 05) |
| `context` | 256 | Enough for a TinyStories story; cheap attention |
| **Parameters** | **≈ 3.4 M** | |

The arithmetic, which you will verify yourself in B5:

```
embeddings (tied with output head):  4096 × 192              =   786,432
per layer:
  attention  4 × (192 × 192)                                 =   147,456
  SwiGLU MLP 3 × (192 × 512)                                 =   294,912
  2 × RMSNorm                                                =       384
                                                    per layer =   442,752
6 layers                                                      = 2,656,512
                                                        TOTAL ≈ 3,442,944
```

**The scaling ladder.** You will train three sizes, because three points is what a scaling law
needs (B8):

| Name | `d_model` | `n_layers` | Params | Rough train time on M3 Pro |
|------|-----------|------------|--------|----------------------------|
| `tiny` | 128 | 4 | ≈ 1.1 M | ~10 min |
| **`base`** | **192** | **6** | **≈ 3.4 M** | **~45 min** |
| `wide` | 320 | 8 | ≈ 12 M | ~3 h |

All three fit in memory with room to spare. None of them needs a rented GPU. That is the point.

## The corpus

Two options, and you should use both:

| Corpus | Size | Use |
|--------|------|-----|
| **tinyshakespeare** | 1.1 MB | The 10-minute smoke test. Debug the pipeline here. |
| **TinyStories** | ~100 MB subset | The real run. Simple synthetic children's stories with a deliberately small vocabulary — the corpus that proved models this size can produce *grammatical, coherent* English. |

TinyStories is what makes this track work. A 3M-parameter model trained on Wikipedia produces
noise. The same model trained on TinyStories writes real sentences, because the corpus was
constructed to use only words a four-year-old knows. You get genuine language-model behaviour at a
scale you can train over lunch.

For the post-training half (B9 onward) you will also generate a **synthetic verifiable task set** —
something with a programmatic checker, so that SFT, DPO and GRPO are real rather than performative.

---

## The lessons

| | Lesson | You end with |
|---|--------|-------------|
| **B0** | [Foundations: probability, loss, gradients, backprop](00-foundations.md) | A scalar autograd engine you wrote, verified against PyTorch |
| **B1** | [Tensors, and your first language model](01-first-model.md) | A trained bigram model generating (bad) text, and your baseline loss |
| **B2** | [The tokenizer, from scratch](02-tokenizer.md) | Your own 4096-token BPE, trained on your corpus |
| **B3** | [Data: corpus to batches](03-data.md) | A memmapped token stream and a batch loader |
| **B4** | [Attention, from one head to many](04-attention.md) | Multi-head causal attention you implemented |
| **B5** | [The transformer, assembled](05-transformer.md) | The 3.4M model, parameter count verified to the digit |
| **B6** | [Training it for real](06-training.md) | Coherent English, and a loss curve you can read |
| **B7** | [Sampling from your own logits](07-sampling.md) | Every decoding strategy, implemented |
| **B8** | [Scaling laws on your own models](08-scaling.md) | A power law fitted to three models you trained |
| **B9** | [Instruction tuning (Phase 2)](09-instruct.md) | A base model *and* an instruct model, both yours |
| **B10** | [Teaching it to reason](10-reasoning.md) | CoT supervision + GRPO against a verifier you wrote |

---

## How this feeds the reference track

After B10 you own three checkpoints — base, instruct, and reasoning — all of them yours. Every
later module gets a better lab:

| Reference module | Now demonstrated on |
|------------------|---------------------|
| 01 Tokenization | The BPE **you trained** — including deliberately retraining it wrong |
| 02 Embeddings | Your embedding matrix, whose every row you can trace to a merge rule |
| 03/04 Attention, RoPE | Your attention implementation — swap RoPE out and watch it fail |
| 05/05a Block, encoder-decoder | Your block; add an encoder and cross-attention to it |
| 06 Pretraining | Your scaling law, your dedup experiment, your annealing run |
| 08 Decoding | Your logits |
| 11 SFT | Your instruct model, from your base model |
| 12 PEFT | LoRA on a model small enough to print the whole `ΔW` |
| 14/15/15a Preference, GRPO, R1 | **R1-Zero reproduced on a base model you pretrained yourself** |
| 16 Distillation | Qwen3-1.7B as teacher, your 3.4M as student |
| 18 Quantization | Quantize weights you understand; watch exactly which layers break |
| 23/24 Eval, failure analysis | A model whose failures you can trace to specific training decisions |

**The strongest version of the R1-Zero lab is one where you also pretrained the base model.** Very
few candidates can say that.

---

## After B10

You have a base model, an instruct model, and a reasoning model, plus the loss curves, ablations and
scaling fit that produced them. Three directions from here, in order of value:

1. **Re-run the reference modules' labs on your own model.** Modules 12 (LoRA on a model small
   enough to print the whole `ΔW`), 18 (quantize weights you understand and see exactly which layers
   break), 23–24 (failure analysis traceable to specific training decisions) are all strictly better
   with your checkpoints than with a borrowed one.
2. **Write it up.** The artifact is not the model, it's the record: parameter derivation, ablation
   tables, the scaling fit and its tested prediction, the length-growth plot with the reward function
   beside it showing no length term. That document is the thing worth showing.
3. **Scale one axis.** A 50M model on a larger corpus, or the same 3.4M model with GQA and a 1024
   context. One change at a time, measured against the curves you already have.

---

## Ground rules

- **Type the code.** Not copy — type. The bugs you make are the lesson.
- **Run on CPU first, MPS second.** CPU is easier to debug and fast enough at 1M params.
- **Commit after every lesson.** You will want to go back to a working state.
- **Seed everything.** `torch.manual_seed(1337)`. Reproducibility is not optional when you're
  comparing runs.
- **Save every loss curve.** B8 needs them, and so does every ablation later.

---

**Start:** [B0 — Foundations](00-foundations.md) · [Curriculum syllabus](../../SYLLABUS.md)
