# B9 — Instruction tuning your own model (Phase 2)

At the end of B6 you recorded what your base model did with *"Can you solve this math problem?"* It
wrote more questions. It has no concept of a request — it completes text, and a question is text that
is usually followed by more text like it.

This lesson performs the mindset shift. Same weights, same architecture, same optimizer, **~30
minutes of extra training on a few thousand formatted examples**, and the model starts answering
instead of continuing. Nothing about its knowledge changes; what changes is which *distribution* it
believes it is sampling from.

That is the single most important thing to understand about post-training, and you are about to
observe it directly on a model you built.

Reference companions: [Module 11](../11-sft.md), [Module 12](../12-peft.md).

---

## Terms

| Term | Meaning |
|------|---------|
| **SFT** | Supervised fine-tuning — next-token prediction on curated (prompt, response) pairs. |
| **Instruction tuning** | SFT whose data teaches the request→answer format. |
| **Chat template** | The exact string layout wrapping turns. Must match between train and inference. |
| **Special/control tokens** | Structural anchors (`<prompt>`, `<thought>`, `<response>`) the model learns as boundaries. |
| **Loss masking** | Setting prompt-token labels to `-100` so loss is computed on the completion only. |
| **`-100`** | PyTorch's `ignore_index` for `F.cross_entropy`. |
| **Completion-only loss** | Training the model to *produce* answers, not to *predict* questions. |
| **Packing vs padding** | Concatenating short examples vs padding each to a fixed length. |
| **Format collapse** | The model emits the template's tags as content. |
| **Catastrophic forgetting** | Fluency degrading during fine-tuning. |
| **Base vs instruct** | Two checkpoints from one pretraining run. Keep both. |

---

## Concepts

### Instruction tuning does not add knowledge

The strongest evidence for this is the well-known result that a very small number of high-quality
examples — the LIMA paper used 1,000 — is enough to produce a usable assistant from a strong base
model. If instruction tuning were teaching *content*, a thousand examples could not possibly do it.

What it teaches is **format and intent**: that a request should be followed by a compliant answer
rather than by more requests; where responses begin and end; what register to use; and when to stop.
Everything the model actually knows was learned during pretraining. That's the framing to carry:
**pretraining builds the capability, instruction tuning makes it addressable.**

On your model specifically, that means the ceiling is what B6 achieved. A 3.4M model instruction-tuned
on TinyStories-scale data will follow simple instructions in its domain and fail outside it — which
is the correct and expected result, and a useful thing to have seen at a scale where you can inspect
every part of it.

### The template is a contract

Your template, following the structure you specified:

```
<prompt>Instruction: Solve the puzzle.
Input: Box A is heavier than Box B. Box B is heavier than Box C. Which is lightest?</prompt>
<thought>Track the weights. A > B. B > C. So A > B > C. The lightest is C.</thought>
<response>Box C is the lightest.</response><eos>
```

Three properties this needs, and each has a failure attached:

- **Unambiguous boundaries.** The model must learn that `</prompt>` means "stop listening, start
  producing". This is what turns a completion engine into a responder.
- **Consistency between training and inference.** A single missing newline or a different tag order
  at inference time puts the model off-distribution, and the symptom is *quality degradation with no
  error message* — the hardest class of bug in this whole track to notice. Generate your inference
  prompt with the same function that generated your training data. Not a similar function. The same
  one.
- **Tags as real tokens, unreachable by the tokenizer.** Add `<prompt>`, `<thought>`, `<response>`
  as special tokens (B2). If a user can type a literal `<response>` and have it tokenize to the
  control token, they can forge an assistant turn — the same unreachability property you asserted in
  B2, now doing security work rather than roundtrip work.

The `<thought>` block is what makes this more than formatting: the model learns that after
`</prompt>` it must produce reasoning tokens **before** it is permitted to produce an answer. B10
develops that into a real reasoning capability with generated logic data.

### Loss masking, the mechanism that matters

Naive SFT computes loss over the whole sequence, which trains the model to *generate the question*
as well as the answer. Correct SFT masks the prompt:

```python
def build_example(prompt_ids, completion_ids, max_len):
    ids    = prompt_ids + completion_ids
    labels = [-100] * len(prompt_ids) + completion_ids[:]     # mask the prompt
    ids, labels = ids[:max_len], labels[:max_len]
    pad = max_len - len(ids)
    return (ids + [PAD] * pad, labels + [-100] * pad)          # padding masked too
```

`F.cross_entropy(..., ignore_index=-100)` then skips those positions entirely.

**Note that the input still contains the prompt** — the model attends to it fully, it simply is not
scored on predicting it. That distinction is the thing to be able to state crisply: masking changes
what is *learned*, not what is *seen*.

Two consequences worth knowing before you hit them:

- **Your SFT loss is not comparable to your pretraining loss.** It is computed over a different
  (smaller, easier, more templated) set of positions. A 0.8 SFT loss does not mean the model got
  better than its 1.6 pretraining loss.
- **Padding must be masked too**, or the model spends most of its gradient learning to predict `PAD`
  after short answers — a spectacular way to waste a run, and one whose symptom (trailing padding in
  generations) looks like a sampler bug.

### Hyperparameters change completely

| | Pretraining (B6) | SFT (here) |
|---|------------------|------------|
| LR | 1e-3 | **1e-5 – 5e-5** |
| Steps | 5,000 | **200–1,000** |
| Epochs | <1 (fresh data) | **1–3** |
| Warmup | 200 | **10–20** |
| Data | 100MB raw | **1–10MB curated** |
| Failure | Underfitting | **Overfitting, forgetting** |

**The LR drops by roughly two orders of magnitude, and this is the mistake everyone makes once.**
Fine-tuning at the pretraining LR destroys the base model in a few dozen steps: it produces fluent
template-shaped output with no content, because you have overwritten the pretrained weights rather
than adjusting them.

More than 2–3 epochs on a small SFT set produces memorisation: verbatim training responses to
paraphrased prompts. Watch validation loss, and stop when it turns.

---

## Where it's used

- Every `-Instruct` / `-Chat` model is this step applied to a base checkpoint.
- Chat-template mismatch is the most common cause of "the model is worse in my app than in the demo".
- The masking code here is identical to what TRL's `SFTTrainer` does under `completion_only_loss`.
- DPO (Module 14) and RLVR (Module 15) start from an SFT checkpoint, not from a base model.
- B10 extends the `<thought>` block into trained reasoning.

---

## Labs

### Lab 1 — Build the dataset

3,000–5,000 examples, generated programmatically. Because your model's competence is TinyStories,
the instructions must live in that domain — this is the constraint people miss when they fine-tune a
small model on general-purpose instruction data and conclude that SFT "doesn't work".

| Type | Example instruction | Count |
|------|--------------------|-------|
| Continue | "Continue this story: Once upon a time…" | 1000 |
| Summarise | "Summarise this story in one sentence." | 800 |
| Question | "Who is the main character in this story?" | 800 |
| Rewrite | "Rewrite this story so it ends happily." | 600 |
| Logic | "Box A > Box B > Box C. Which is lightest?" | 800 |

Generate the last group with a template and a solver so the `<thought>` block is always correct by
construction — that is B10's method, previewed here.

Hold out 200 examples. You need them in Lab 5.

### Lab 2 — Tokenizer surgery

```python
SPECIALS = ["<prompt>", "</prompt>", "<thought>", "</thought>",
            "<response>", "</response>"]
tok.add_special_tokens(SPECIALS)          # ids 4096..4101

model.embed.weight.data = torch.cat([
    model.embed.weight.data,
    model.embed.weight.data.mean(0, keepdim=True).repeat(len(SPECIALS), 1)
        + 0.02 * torch.randn(len(SPECIALS), model.cfg.d_model),
])
model.lm_head.weight = model.embed.weight   # re-tie after resizing
```

Three things to verify, each a real bug:

1. **New embeddings initialised near the mean of existing ones**, not at zero and not at full random
   scale — random-scale rows produce huge initial logits for tokens that should be rare.
2. **Re-tie the head after resizing**, or you have silently created 6 untied rows.
3. `assert tok.encode("<response>") != [tok.special_id("<response>")]` — a literal typed by a user
   must not become the control token.

### Lab 3 — Fine-tune

```python
sft_opt = configure_optimizer(model, lr=3e-5, weight_decay=0.0)

for epoch in range(2):
    for ids, labels in sft_loader:
        logits, _ = model(ids)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                               labels.view(-1), ignore_index=-100)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        sft_opt.step(); sft_opt.zero_grad(set_to_none=True)
```

Before the loop, assert the masking is real:

```python
ids, labels = next(iter(sft_loader))
assert (labels == -100).sum() > 0, "nothing is masked"
assert (labels[0][:10] == -100).all(), "prompt tokens are not masked"
```

**~30 minutes. Save as `ckpt_instruct.pt`, and keep `ckpt_best.pt` untouched.** Two checkpoints from
one pretraining run is how every model family ships, and you need both for Lab 4.

### Lab 4 — The before/after, side by side

Run the *identical* prompts through base and instruct.

```python
for p in ["Can you solve this math problem?",
          "Summarise this story in one sentence: ...",
          "Who is the main character in this story: ...",
          "Box A is heavier than Box B. Box B is heavier than Box C. Which is lightest?"]:
    print("BASE    :", generate(base_model, p, max_new=80, temperature=0.8, top_p=0.9))
    print("INSTRUCT:", generate(instr_model, wrap(p), max_new=80, temperature=0.8, top_p=0.9))
```

**This is the deliverable of the lesson.** Same weights three thousand examples ago; the base model
continues, the instruct model responds. Write both outputs down verbatim next to the B6 Lab 8
recording — that three-way comparison, on your own model, is a better answer to "what does
instruction tuning do?" than any explanation.

### Lab 5 — Masking ablation

Train an identical run with the mask removed (loss over the whole sequence).

**Expect the unmasked model to sometimes generate a new instruction instead of answering, or to
answer and then continue into another `<prompt>` block.** It learned the prompt distribution because
you trained it to. Score both models on the 200 held-out examples: fraction that produce a
well-formed `<response>` block, and fraction that stop cleanly at `<eos>`.

### Lab 6 — Learning-rate destruction

Fine-tune at `1e-3` — the pretraining LR — for 200 steps.

**The model will emit correctly-formatted, fluent, contentless output.** Perfect tags, empty
substance. Measure it: perplexity on the *pretraining* validation set, before and after. It rises
sharply, which is catastrophic forgetting quantified rather than asserted, and it explains in one
number why fine-tuning LRs are two orders of magnitude below pretraining LRs.

### Lab 7 — Overfitting curve

Train for 1, 2, 3, 5, 10 epochs, checkpointing each.

| Epochs | Expect |
|--------|--------|
| 1 | Format learned, some drift |
| **2–3** | **Best held-out quality** |
| 5 | Repetitive; template phrasing everywhere |
| 10 | Verbatim training answers to paraphrased prompts |

At 10 epochs, feed a *paraphrase* of a training prompt and check whether the exact training response
comes back. **When it does, you have memorisation you can point at**, and the held-out loss curve
turning upward is where you should have stopped.

### Lab 8 — Template mismatch, the silent killer

Take the correctly-trained instruct model and generate with three deliberately wrong prompt formats:

1. Correct template.
2. Missing the trailing newline before `<response>`.
3. Raw instruction text with no tags at all.

**Quality degrades progressively and nothing raises an error.** Version-3 output looks like the base
model again. This is the mechanism behind most "it worked in the demo" reports in production, and
the fix is structural: generate inference prompts with the same function that built training data.

---

## Failure modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| Model generates new questions | Loss not masked | `-100` on prompt tokens |
| Fluent but empty answers | LR far too high | 1e-5 – 5e-5 |
| Verbatim training answers | Too many epochs | Stop at 2–3; watch held-out loss |
| Trailing padding in output | Padding not masked | `-100` on pad positions too |
| Tags emitted as content | Specials not registered as tokens | Add to tokenizer, verify ids |
| Quality drops in production only | Template mismatch | Share the formatting function |
| Never stops | No `<eos>` after `</response>` | Append it in every example |
| Base fluency lost | Catastrophic forgetting | Lower LR, fewer epochs, or mix in pretraining data |
| Head silently untied | Resized embedding without re-tying | Re-assign `lm_head.weight` after resize |
| Users can forge assistant turns | Special tokens reachable from text | B2's unreachability assert |

---

## Interview

**"What does instruction tuning actually do?"**
It teaches format and intent, not knowledge. The evidence is that on a strong base model a thousand
or so well-chosen examples is enough — if it were teaching content, that couldn't work. What the
model learns is that a request is followed by a compliant answer rather than by more requests, where
a response begins and ends, and when to stop. I did this on a 3.4M model I pretrained myself: the
base checkpoint answers "Can you solve this math problem?" by writing more math problems, and after
about thirty minutes of SFT the same weights answer it. Nothing was added to what the model knows;
the capability became addressable.

**"Why mask the prompt in the loss?"**
Because otherwise you train the model to generate prompts as well as responses, and you can see it in
the output — an unmasked run will sometimes answer and then continue into a new instruction, because
that's the distribution it was fit to. Masking sets prompt-token labels to `-100` so cross-entropy
skips them. The distinction worth being precise about is that the model still *attends* to the whole
prompt; masking changes what's learned, not what's seen. Padding needs masking too, or short answers
teach the model to predict pad tokens.

**"How do fine-tuning hyperparameters differ from pretraining?"**
The learning rate drops by about two orders of magnitude — 1e-3 to something like 3e-5 — and the run
is hundreds of steps instead of thousands, over one to three epochs on far less data. The failure
mode inverts too: pretraining underfits, fine-tuning overfits and forgets. I demonstrated the LR
point directly by fine-tuning at the pretraining LR: the model produced perfectly formatted,
completely contentless output, and perplexity on the pretraining validation set jumped. That's
catastrophic forgetting as a number rather than a phrase.

**"A model performs well in evaluation but poorly in production, same weights. Why?"**
Chat-template mismatch, first guess. If the inference-time prompt differs from the training format by
even a newline or a tag order, the model is off-distribution and quality degrades with no error
raised at all — which is what makes it so hard to catch. I reproduced it deliberately: correct
template, template missing one newline, and raw untagged text, and the third case makes the instruct
model behave like the base model again. The structural fix is to build inference prompts with the
same function that built the training data, not a reimplementation of it.

**"Why keep the base checkpoint?"**
Because they're different products. The base model is the right starting point for any further
training — DPO, RLVR, a differently-targeted SFT — and starting those from an instruct checkpoint
compounds its formatting biases. It's also the honest baseline for measuring what post-training
actually bought you. Every model family ships both for exactly this reason.

---

## Checkpoint

1. 3,000+ examples generated across five instruction types, 200 held out.
2. Special tokens added, embeddings resized and re-tied, unreachability asserted.
3. `ckpt_instruct.pt` trained with masking asserted before the loop; `ckpt_best.pt` untouched.
4. Base vs instruct on identical prompts, recorded verbatim alongside the B6 Lab 8 output.
5. Masking ablation scored on held-out data: well-formed-response rate and clean-stop rate.
6. LR-destruction run with pretraining-set perplexity before and after.
7. Epoch sweep with the held-out curve turning, plus a demonstrated memorised response.
8. Three-way template-mismatch comparison showing silent degradation.

---

**Next:** [B10 — Teaching it to reason](10-reasoning.md) ·
**Back:** [B8 — Scaling laws](08-scaling.md) · [Build Track](README.md)
