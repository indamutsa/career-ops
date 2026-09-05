# Module 12 — Parameter-efficient fine-tuning

You have already written publicly about LoRA. This module is the depth behind that post: the
mechanics, the hyperparameters that actually matter, and the failure modes.

---

## Terms

| Term | Meaning |
|------|---------|
| **PEFT** | Parameter-Efficient Fine-Tuning. Update a small fraction of parameters, freeze the rest. |
| **LoRA** | Low-Rank Adaptation. Freeze `W`, learn `ΔW = BA` where `B: d×r`, `A: r×k`, `r ≪ min(d,k)`. |
| **Rank `r`** | The bottleneck dimension. Controls adapter capacity. Typically 8–64. |
| **Alpha `α`** | Scaling numerator. The update is `(α/r)·BA`. |
| **Scaling factor** | `α/r`. What actually multiplies the adapter output. |
| **Target modules** | Which weight matrices get adapters (`q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj`). |
| **Merge** | Folding `BA` into `W` so inference has zero adapter overhead. |
| **QLoRA** | LoRA over a 4-bit quantized frozen base. |
| **NF4** | 4-bit NormalFloat — quantization levels matched to a normal distribution. |
| **Double quantization** | Quantizing the quantization constants themselves. Saves ~0.4 bits/param. |
| **Paged optimizer** | Optimizer state paged to CPU on memory spikes, using unified memory. |
| **DoRA** | Weight-Decomposed LoRA. Splits `W` into magnitude and direction, adapts them separately. |
| **rsLoRA** | Rank-stabilised LoRA: scale by `α/√r` instead of `α/r`, so high ranks stay trainable. |
| **LoRA+** | Different learning rates for `A` and `B`. |
| **Prefix tuning** | Learn virtual key/value vectors prepended to every attention layer. |
| **Prompt tuning** | Learn continuous embeddings prepended to the input. Fewest parameters, weakest. |
| **IA³** | Learn per-channel rescaling vectors for keys, values and FFN activations. |
| **Trainable-parameter ratio** | Trainable ÷ total. LoRA is typically 0.1–1%. |

---

## Concepts

### The memory argument (this is the whole motivation)

From Module 00, full fine-tuning with AdamW costs roughly:

```
weights 2 + gradients 2 + adam m 4 + adam v 4 + fp32 master 4  ≈  16 bytes/param
```

7B model → ~112 GB before activations. Multiple A100s.

With LoRA, the base is frozen: no gradients, no optimizer state for it. Only the adapter needs all
five. At r=16 on the attention projections, that adapter is maybe 20M parameters.

```
base (frozen, bf16)      14   GB
adapter weights+states    0.3 GB
activations               ~4  GB
                        --------
                         ~18  GB   -> one 24 GB GPU
```

With QLoRA (4-bit base) it drops to ~8 GB — a consumer card.

**The optimizer state, not the weights, is what forces the sharding.** That sentence is the whole
insight, and it is the right answer to "why does LoRA exist".

### Why low rank works

The hypothesis: the *update* needed to adapt a pretrained model to a task has low intrinsic rank,
even though `W` itself is full rank. You are not re-learning language — you are applying a small,
structured correction.

So parameterise the update as a product of two thin matrices:

```
W' = W + (α/r)·B·A
     ^frozen   ^trainable
```

`A` is initialised random (Gaussian), `B` is initialised **zero**. So `BA = 0` at step 0 and the
adapted model is *exactly* the base model — training starts from a known-good point with no shock.
That zero-init is not incidental; initialise both randomly and the first steps are chaos.

### Alpha and rank — the confusion worth clearing up

The update is scaled by `α/r`. So:

- Doubling `r` while holding `α` **halves** the effective scale.
- The common convention `α = 2r` keeps the scale constant at 2 as you sweep rank.
- Tuning `α` and `r` independently means you are changing two things at once.

**Practical rule:** set `α = 2r` and treat `r` as your single capacity knob. If you need a stronger
update at fixed rank, raise `α` — but know you are raising an effective learning rate, and it can
destabilise.

`rsLoRA` scales by `α/√r` instead, which keeps gradients well-scaled at high rank. If r=128+ seems
to underperform r=32, this is likely why.

### Which modules to target

| Target set | Params | Quality | When |
|-----------|--------|---------|------|
| `q_proj, v_proj` | smallest | decent | The original paper; still a fine default |
| all attention (`q,k,v,o`) | ~2× | better | Common default |
| attention + MLP (`gate,up,down`) | ~4× | best | When quality matters more than memory |
| MLP only | medium | task-dependent | Knowledge-heavy adaptation |

Modern practice targets everything. The MLP holds most of the parameters and most of the factual
knowledge, so excluding it caps what you can change.

### QLoRA

Three components:

1. **NF4** — 4-bit levels placed at the quantiles of a normal distribution, which is how pretrained
   weights are actually distributed. Better than uniform int4 at the same width.
2. **Double quantization** — the per-block scaling constants are themselves quantized.
3. **Paged optimizers** — spill optimizer state on spikes rather than OOM.

The base is dequantized to bf16 **per-block during the forward pass**, so compute is bf16 while
*storage* is 4-bit. That is why QLoRA is slower per step than LoRA but fits far more.

Quality loss versus 16-bit LoRA is small — usually within noise on downstream tasks. The trade is
almost always worth it when memory is the binding constraint.

### Merging, and when you must not

```
W_merged = W + (α/r)·B·A
```

After merging, inference is identical in cost to the base model — LoRA adds **zero** latency in
production. This is a genuine advantage over adapter methods that insert extra layers.

**But:** merged means baked in. To serve twenty customer-specific adapters you must keep them
*unmerged* and swap per request. That is Module 20, and it is the architecture the JD's
"domain-specific agents" phrase implies. Merging is for the single-adapter case.

You also cannot merge cleanly into a quantized base — dequantize, merge, requantize, and accept
some error.

---

## Where it's used

- **Essentially all fine-tuning.** Full fine-tuning is now the exception, reserved for large
  behavioural change or continued pretraining.
- **Multi-tenant serving.** One base, many adapters — the economic argument for domain agents.
- **Rapid iteration.** An adapter is tens of MB; you can version, ship and roll back cheaply.
- **RL post-training.** GRPO over a LoRA adapter is standard for keeping RL affordable.

---

## Labs

### Lab 1 — LoRA from scratch

Implement it before importing it.

```python
import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Wrap a frozen Linear with a trainable low-rank update."""

    def __init__(self, base: nn.Linear, r: int = 8, alpha: int = 16):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False

        self.r = r
        self.scaling = alpha / r
        self.A = nn.Parameter(torch.randn(r, base.in_features) * 0.01)
        self.B = nn.Parameter(torch.zeros(base.out_features, r))   # zero init!

    def forward(self, x):
        return self.base(x) + (x @ self.A.T @ self.B.T) * self.scaling

    def merged_weight(self):
        return self.base.weight.data + (self.B @ self.A) * self.scaling


torch.manual_seed(0)
base = nn.Linear(512, 512, bias=False)
lora = LoRALinear(base, r=8, alpha=16)
x = torch.randn(4, 512)

# Step 0: adapter is a no-op because B is zero.
print("delta at init:", (lora(x) - base(x)).abs().max().item())     # 0.0

# Parameter accounting
base_p = base.weight.numel()
lora_p = lora.A.numel() + lora.B.numel()
print(f"base {base_p:,}  lora {lora_p:,}  ratio {lora_p/base_p:.2%}")

# Merge equivalence
lora.B.data = torch.randn_like(lora.B) * 0.02
merged = nn.Linear(512, 512, bias=False)
merged.weight.data = lora.merged_weight()
print("merge error:", (lora(x) - merged(x)).abs().max().item())     # ~1e-6
```

**Three things proven:** zero-init means training starts at the base model; the adapter is ~3% of
the layer at r=8; merging is exact up to float error.

### Lab 2 — Rank sweep on a real model

```python
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM
import torch

for r in [4, 8, 16, 32, 64, 128]:
    m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", torch_dtype=torch.bfloat16)
    cfg = LoraConfig(
        r=r, lora_alpha=2 * r, lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
    )
    pm = get_peft_model(m, cfg)
    trainable = sum(p.numel() for p in pm.parameters() if p.requires_grad)
    total = sum(p.numel() for p in pm.parameters())
    print(f"r={r:<4} trainable {trainable:>12,}  {trainable/total:>7.3%}  "
          f"adapter {trainable*2/1e6:>6.1f} MB bf16")
    del m, pm
```

**Observe:** trainable parameters scale linearly with `r`. Even r=128 stays around 1–2%. Then run
the same downstream task at each rank and plot quality against rank — you will typically see it
plateau by r=16–32, which is why higher ranks are usually wasted memory.

### Lab 3 — Target-module ablation

```python
sets = {
    "q,v only":      ["q_proj", "v_proj"],
    "all attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
    "mlp only":      ["gate_proj", "up_proj", "down_proj"],
    "everything":    ["q_proj", "k_proj", "v_proj", "o_proj",
                      "gate_proj", "up_proj", "down_proj"],
}
for name, mods in sets.items():
    m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", torch_dtype=torch.bfloat16)
    pm = get_peft_model(m, LoraConfig(r=16, lora_alpha=32, target_modules=mods,
                                      task_type="CAUSAL_LM"))
    t = sum(p.numel() for p in pm.parameters() if p.requires_grad)
    print(f"{name:<16} {t:>11,} trainable")
    del m, pm
```

Note how much larger the MLP targets are — that is where the parameters live. Train each and
compare; the ordering will tell you what your task actually needs changed.

### Lab 4 — Prove the alpha/rank interaction

```python
import torch, torch.nn as nn

torch.manual_seed(0)
x = torch.randn(8, 256)
base = nn.Linear(256, 256, bias=False)

print(f"{'r':>5}{'alpha':>7}{'scale':>8}{'delta norm':>13}")
for r, alpha in [(8, 16), (16, 16), (32, 16), (16, 32), (32, 64)]:
    l = LoRALinear(nn.Linear(256, 256, bias=False), r=r, alpha=alpha)
    l.B.data = torch.randn_like(l.B) * 0.02
    delta = (l(x) - l.base(x)).norm().item()
    print(f"{r:>5}{alpha:>7}{alpha/r:>8.2f}{delta:>13.4f}")
```

**Observe:** `(8,16)`, `(16,32)` and `(32,64)` share scale 2.0 and produce comparable update
magnitudes. `(16,16)` and `(32,16)` have smaller scales and correspondingly weaker updates. If you
sweep rank without adjusting alpha, you are confounding capacity with update strength — and your
"higher rank was worse" conclusion is an artefact.

### Lab 5 — QLoRA memory measurement

```python
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig

def mem_gb():
    return torch.mps.current_allocated_memory() / 1e9 if torch.backends.mps.is_available() else 0

m16 = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B", torch_dtype=torch.bfloat16)
print("bf16 weights GB:", sum(p.numel() * p.element_size() for p in m16.parameters()) / 1e9)
del m16

# bitsandbytes 4-bit is CUDA-only; on macOS reason it analytically:
n = 1_720_000_000
for name, bits in [("bf16", 16), ("int8", 8), ("nf4", 4), ("nf4+dq", 4.13)]:
    print(f"{name:<8} {n * bits / 8 / 1e9:>6.2f} GB")
```

The `nf4+dq` row shows double quantization's overhead being small — the constants add back a
fraction of a bit rather than a whole one.

---

## Failure modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| Training loss barely moves | Effective scale too small, or targets too narrow | Raise `α`, widen target modules |
| Loss drops then output degenerates | Effective LR too high (often via large `α`) | Lower `α` or LR |
| High rank underperforms low rank | `α/r` scaling shrank the update | Use `α = 2r`, or rsLoRA |
| Merged model differs from adapter model | Merged into a quantized base | Merge in bf16, then requantize |
| Adapter helps the task, breaks everything else | Catastrophic forgetting | Mix in general data; keep a regression suite |
| Cannot swap adapters at serving time | They were merged | Keep unmerged; see Module 20 |
| QLoRA much slower per step | Per-block dequantization in the forward pass | Expected — it is a memory/speed trade |
| Adapter has no effect at inference | Adapter not loaded, or loaded onto a different base revision | Verify the base sha matches training |

---

## Interview

**"Why does LoRA save so much memory when the base is still in memory?"**
Because the base is frozen. Frozen parameters need no gradient buffer and no optimizer state. With
AdamW that is ~14 of the ~16 bytes per parameter — the weights themselves are the small part. Only
the adapter, typically well under 1% of parameters, carries the full training footprint.

**"What do rank and alpha do, and how do you tune them?"**
Rank is adapter capacity; alpha divided by rank is the scale applied to the update. They interact,
so I fix `α = 2r` and sweep rank alone — otherwise a rank sweep silently changes update strength
too and you draw the wrong conclusion. Quality usually plateaus around r=16–32 for task adaptation;
higher ranks mostly cost memory. At very high rank I would switch to rsLoRA, which scales by `α/√r`
and keeps gradients well-conditioned.

**"When would you not use LoRA?"**
Continued pretraining or a large behavioural shift — LoRA's low-rank assumption is about *task
adaptation*, and it does not hold when you are genuinely teaching new knowledge at scale. Also
when the base is small enough that full fine-tuning fits comfortably, since full FT is usually a
little better and simpler to reason about.

**"How would you serve fifty customer-specific fine-tunes?"**
One base model, fifty unmerged LoRA adapters, per-request adapter selection — vLLM's multi-LoRA or
S-LoRA-style punica kernels. Merging would mean fifty full model copies in memory; unmerged
adapters are tens of MB each. The cost is a small per-request overhead from the extra low-rank
matmuls, and cold-start latency when an adapter is not resident. That is the architecture behind
"domain-specific agents" as a product.

---

## Checkpoint

1. Explain the memory saving in terms of optimizer state, not weights.
2. Explain why `B` is zero-initialised.
3. Predict the effect of doubling `r` at fixed `α`.
4. State when you must not merge.
5. Compute the adapter size for a given rank and target set.

---

**Next:** [13 — Dataset construction](13-datasets.md) ·
**Back:** [11 — SFT](11-sft.md) · [Syllabus](../SYLLABUS.md)
