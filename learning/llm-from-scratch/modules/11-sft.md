# Module 11 — Supervised fine-tuning

SFT is the step you have already done at Boehringer. This module makes sure you can defend every
choice inside it, because in a post-training interview SFT is the ground everyone stands on before
they argue about DPO and GRPO.

---

## Terms

| Term | Meaning |
|------|---------|
| **SFT** | Supervised Fine-Tuning. Next-token cross-entropy on curated `(prompt, response)` pairs. |
| **Instruction tuning** | SFT specifically on instruction-following data. |
| **Behavioural cloning** | The RL name for the same thing: imitate demonstrations. |
| **Teacher forcing** | Conditioning on the *ground-truth* prefix at every position, not the model's own output. |
| **Loss masking** | Setting prompt-token labels to `-100` so loss is computed on the completion only. |
| **Label smoothing** | Softening one-hot targets to reduce over-confidence. |
| **Warmup** | Ramping LR from ~0 over the first few percent of steps. |
| **Cosine schedule** | LR decaying along a cosine from peak to ~0. |
| **Gradient accumulation** | Summing grads over micro-batches to emulate a larger batch. |
| **Effective batch size** | `per_device_bs × grad_accum × n_devices`. |
| **Gradient clipping** | Rescaling the global grad norm to a max (typically 1.0). |
| **Grad norm** | The health metric of a training run. |
| **Epoch** | One pass over the dataset. For SFT: usually 1–3, rarely more. |
| **Catastrophic forgetting** | Losing prior capabilities while learning the new task. |
| **Exposure bias** | Train on ground-truth prefixes, infer on self-generated ones. The gap RL closes. |
| **Packing** | Concatenating examples to fill the sequence length. |
| **NEFTune** | Adding noise to embeddings during SFT; a cheap, real regulariser. |

---

## Concepts

### What SFT actually optimises

Exactly the pretraining objective — next-token cross-entropy — restricted to curated data and,
crucially, masked to the response tokens:

```
L = - (1/|R|) * sum over response positions t of  log p(y_t | y_<t, prompt)
```

Two things follow, and both matter in interviews:

**1. SFT does not teach knowledge; it selects behaviour.** The pretrained model already contains
the capability. SFT raises the probability of the response *style* and *format* you demonstrated.
This is why 1,000 good examples work (Module 13) and why SFT cannot make a model know a fact it
never saw in pretraining. If your fine-tune is hallucinating domain facts, more SFT will not fix
it — retrieval will (Module 21).

**2. SFT can only imitate, never exceed, its demonstrations.** The ceiling is the data. If your
demonstrations solve tasks in eight tool calls, the model learns eight tool calls; it has no
mechanism to discover that three would do. That ceiling is the entire argument for preference
optimisation and RL, and it is the cleanest way to answer "why isn't SFT enough".

### Teacher forcing and exposure bias

During training the model always sees the *correct* prefix. At inference it sees its own output.
One early mistake compounds, and it is now conditioned on a prefix distribution it never trained
on.

For a single-turn answer this is mild. For an **eight-step agent trajectory it is severe** — step 3
conditions on the model's own steps 1–2, which may be wrong. This is precisely why agentic tasks
benefit disproportionately from RL: RL trains on the model's *own* rollouts, so the training
distribution matches deployment.

Say that sentence in the interview. It connects Module 11 to Module 15 in one line and it is the
technically correct reason agents need more than SFT.

### The hyperparameters that actually matter

| Knob | Typical | Why |
|------|---------|-----|
| Learning rate | 1e-5–5e-5 full FT; **1e-4–3e-4 LoRA** | LoRA tolerates ~10× higher; adapter is small and zero-init |
| Epochs | 1–3 | Beyond 3, memorisation and degeneration |
| Effective batch | 32–128 sequences | Smaller is noisier; larger wastes compute at this scale |
| Warmup | 3–10% of steps | Prevents an early destructive update |
| Schedule | cosine to ~0 | Stable, standard; linear is fine |
| Max grad norm | 1.0 | Clips the rare exploding batch |
| Weight decay | 0.0–0.1 | Modest effect at these scales |
| Seq length | fit p95 of your data | Truncation silently deletes your answers |

**The one most often wrong is learning rate for LoRA.** People carry over a full-fine-tuning 2e-5,
see nothing happen, and conclude LoRA is weak. The adapter starts at zero and is tiny; it needs
1e-4 or more.

**The second most often wrong is sequence length.** If `max_seq_length` truncates below your p95
example length, you are training on cut-off answers and teaching the model to stop mid-thought. Do
the token-length histogram before the run, not after.

### Reading the loss curve

| Observation | Meaning |
|-------------|---------|
| Loss falls smoothly, eval falls too | Healthy |
| Train falls, eval rises | Overfitting — fewer epochs, more data, or more regularisation |
| Loss flat from step 0 | LR too low, or gradients not reaching trainable params |
| Loss spikes to NaN | LR too high, fp16 overflow, or a bad batch — use bf16, clip, inspect |
| Loss drops in a step at each epoch boundary | Memorisation |
| Grad norm growing steadily | Instability building; lower LR before it explodes |

**Log grad norm.** It moves before the loss does, so it is your earliest warning.

### Full fine-tune vs LoRA

| | Full FT | LoRA |
|---|---------|------|
| Memory | ~16 B/param | ~2 B/param + small adapter |
| Quality | slightly better | very close for task adaptation |
| Forgetting | worse | milder — base weights untouched |
| Serving | one model per task | one base, many adapters |
| New knowledge | can absorb some | limited by low rank |

Default to LoRA. Reach for full fine-tuning when you are shifting behaviour broadly or continuing
pretraining on a new domain.

### Where SFT sits in the pipeline

```
pretrained base
      |  SFT  (Module 11)  <- format, instruction-following, tool syntax
      v
   SFT model
      |  DPO/KTO (Module 14)  <- style, preference, tone
      v
 aligned model
      |  GRPO/RLVR (Module 15)  <- outcome-level task performance
      v
 deployed policy
```

Each stage assumes the previous one. **Running DPO on a base model that has not been SFT'd is a
classic mistake** — the reference policy is not yet producing the right format, so the preference
signal is dominated by formatting noise rather than the quality difference you care about.

---

## Where it's used

- **Every post-training pipeline**, as step one.
- **Format and tool-call compliance** — the fastest fix for an agent emitting unparseable actions
  is SFT on correct action strings, not prompt engineering.
- **Domain adaptation** of tone, register, terminology.
- **Rejection-sampling SFT** (Module 13) — SFT on verified self-generated rollouts, the cheapest
  method that closes most of the gap to RL.
- **Distillation** (Module 16) — SFT where the targets come from a stronger model.

---

## Labs

### Lab 1 — Length histogram before anything else

```python
import json
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
data = [json.loads(l) for l in open("data/sft.jsonl")]

lens = []
for d in data:
    s = tok.apply_chat_template(d["messages"], tokenize=False)
    lens.append(len(tok(s, add_special_tokens=False)["input_ids"]))

lens.sort()
n = len(lens)
for p in (50, 90, 95, 99, 100):
    print(f"p{p:<4} {lens[min(n - 1, p * n // 100)]:>6} tokens")
print("mean", sum(lens) // n)
```

Set `max_seq_length` at or above p99. Anything below p95 and you are training on truncated answers.

### Lab 2 — Minimal SFT loop by hand

Write it once without TRL so the masking is not magic.

```python
import json, torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, get_cosine_schedule_with_warmup

MODEL = "Qwen/Qwen3-0.6B"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16).to("mps")


class SFTData(Dataset):
    def __init__(self, path, max_len=1024):
        self.rows = [json.loads(l) for l in open(path)]
        self.max_len = max_len

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        msgs = self.rows[i]["messages"]
        full = tok.apply_chat_template(msgs, tokenize=False)
        prefix = tok.apply_chat_template(msgs[:-1], tokenize=False, add_generation_prompt=True)

        ids = tok(full, add_special_tokens=False)["input_ids"][: self.max_len]
        n_pre = len(tok(prefix, add_special_tokens=False)["input_ids"])

        labels = list(ids)
        for j in range(min(n_pre, len(labels))):
            labels[j] = -100                      # <- the whole point
        return {"input_ids": ids, "labels": labels}


def collate(batch):
    m = max(len(b["input_ids"]) for b in batch)
    pad = tok.pad_token_id or tok.eos_token_id
    return {
        "input_ids": torch.tensor(
            [b["input_ids"] + [pad] * (m - len(b["input_ids"])) for b in batch]),
        "attention_mask": torch.tensor(
            [[1] * len(b["input_ids"]) + [0] * (m - len(b["input_ids"])) for b in batch]),
        "labels": torch.tensor(
            [b["labels"] + [-100] * (m - len(b["labels"])) for b in batch]),
    }


dl = DataLoader(SFTData("data/sft.jsonl"), batch_size=2, shuffle=True, collate_fn=collate)
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-5)
sched = get_cosine_schedule_with_warmup(opt, int(0.05 * len(dl)), len(dl))

model.train()
for step, batch in enumerate(dl):
    batch = {k: v.to("mps") for k, v in batch.items()}
    out = model(**batch)
    out.loss.backward()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step(); sched.step(); opt.zero_grad()
    if step % 10 == 0:
        print(f"step {step:>4} loss {out.loss.item():.4f} "
              f"grad_norm {gn:.3f} lr {sched.get_last_lr()[0]:.2e}")
```

**Deliverable:** the loss/grad-norm/LR trace. You should see grad norm large and jumpy in the first
few steps, then settling — that is warmup doing its job.

### Lab 3 — The masking ablation

Run Lab 2 twice, identical except: the second run deletes the `labels[j] = -100` line.

Compare:
1. Loss curves (the unmasked one starts *lower* — prompt tokens are easy to predict).
2. Held-out task accuracy.
3. Generate 10 samples from each and read them. The unmasked model will show a tendency to
   continue past its answer into a fabricated next user turn.

**This is the single most convincing lab in the module.** It produces a story you can tell.

### Lab 4 — LoRA SFT with TRL

```python
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig
from datasets import load_dataset

cfg = SFTConfig(
    output_dir="runs/sft-lora",
    num_train_epochs=2,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,          # effective 16
    learning_rate=2e-4,                     # LoRA-scale, not 2e-5
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    max_grad_norm=1.0,
    logging_steps=5,
    bf16=True,
    max_seq_length=2048,                    # from Lab 1
    packing=False,
    eval_strategy="steps",
    eval_steps=50,
    save_strategy="steps",
    save_steps=50,
    load_best_model_at_end=True,
)

peft_cfg = LoraConfig(
    r=16, lora_alpha=32, lora_dropout=0.05,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    task_type="CAUSAL_LM",
)

ds = load_dataset("json", data_files={"train": "data/sft_train.jsonl",
                                      "test": "data/sft_eval.jsonl"})

SFTTrainer(model="Qwen/Qwen3-1.7B", args=cfg, peft_config=peft_cfg,
           train_dataset=ds["train"], eval_dataset=ds["test"]).train()
```

TRL masks prompts for you when the dataset is in `messages` form. Verify it rather than assume:
pull one batch from `trainer.get_train_dataloader()` and check that early label positions are
`-100`.

### Lab 5 — Learning-rate sweep, LoRA

```
1e-5   2e-5   5e-5   1e-4   2e-4   5e-4   1e-3
```

Same data, same seed, 200 steps each. Plot final eval loss against LR. Expect a clear U: too low
and nothing moves; too high and it diverges or degenerates. Record where your minimum lands — that
number is a real answer to "how do you pick a learning rate for LoRA".

### Lab 6 — Measure catastrophic forgetting

Before and after your SFT run, evaluate on something entirely unrelated (a small MMLU slice, or 20
general chat prompts scored by hand).

| Run | Agent task acc | General acc |
|-----|----------------|-------------|
| base | | |
| SFT, 100% target data | | |
| SFT, 80% target + 20% replay | | |

The middle row is the one that scares people. The bottom row is the fix, and it costs almost
nothing.

---

## Failure modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| Nothing changes | LoRA LR set at full-FT scale | Raise to 1e-4–3e-4 |
| Loss → NaN | fp16 overflow, or LR too high | bf16, clip at 1.0, lower LR |
| Model continues past its answer | Prompt tokens unmasked | Mask to `-100` |
| Answers cut off mid-sentence | `max_seq_length` below p95 | Length histogram first |
| Great train loss, poor real use | Overfitting, or format ≠ serving template | Fewer epochs; `apply_chat_template` |
| Task learned, everything else lost | No replay data | 15–25% general mix |
| Agent good on step 1, drifts after | Exposure bias | Rejection-sampling SFT, then RL |
| Step drops at each epoch boundary | Memorising | Stop at 1–2 epochs |
| Eval loss improves, task accuracy doesn't | Loss is not your metric | Evaluate the task, not the loss |

---

## Interview

**"Walk me through an SFT run end to end."**
Data first: curate, decontaminate against the eval set, dedupe, format with the tokenizer's own
chat template, mask prompt tokens. Then a length histogram to set sequence length. LoRA r=16,
α=32 on all attention and MLP projections, LR 2e-4 cosine with 5% warmup, effective batch around
16–32, one to two epochs, grad clip at 1.0, bf16. I log loss, eval loss, grad norm and LR, and I
evaluate the *task*, not the loss — eval loss can improve while task accuracy doesn't. Then a
forgetting check on unrelated data, because a fine-tune that wins on-task and loses general
ability is not a win.

**"Why isn't SFT enough for an agent?"**
Two reasons and they compound. SFT can only imitate its demonstrations, so the demonstrations are
a hard ceiling — if they solve tasks in eight steps the model learns eight steps and has no way to
discover three would do. And SFT is trained with teacher forcing on ground-truth prefixes, while at
inference the model conditions on its own prior steps. For a one-shot answer that gap is small;
across an eight-step trajectory it's severe, because step 3 conditions on possibly-wrong steps 1
and 2. RL trains on the model's own rollouts, so the training distribution matches deployment.
That's the real argument.

**"Your fine-tune is hallucinating domain facts. More SFT?"**
No. SFT selects behaviour the pretrained model already has; it doesn't install knowledge. If the
facts weren't in pretraining, more demonstrations mostly teach it to hallucinate *confidently* in
the right format. The fix is retrieval, or continued pretraining on the domain corpus if the scale
justifies it — but that's a different and much more expensive intervention.

**"How many epochs?"**
One to two for most SFT, three at the outside. Past that you see the step-shaped loss drop at
epoch boundaries that means memorisation, and the model starts degenerating. If two epochs aren't
enough, that's usually a data problem — coverage, not repetition.

---

## Checkpoint

1. Write the masked loss and say why masking matters.
2. Explain teacher forcing and why exposure bias is worse for agents.
3. Give LoRA and full-FT learning rates and why they differ by 10×.
4. Produce the masking-ablation comparison with generated samples.
5. Produce the LR sweep U-curve and name your minimum.
6. Produce the forgetting table with a replay row.

---

**Next:** [12 — PEFT and LoRA](12-peft.md) ·
**Back:** [10 — Structured output and tool calling](10-structured-output.md) · [Syllabus](../SYLLABUS.md)
