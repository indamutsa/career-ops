# Module 00 — Environment and tooling

Short module. Get this right once and no later lab wastes your time.

---

## Terms

| Term | Meaning | Where you meet it |
|------|---------|-------------------|
| **Model card** | The README on a Hugging Face model repo: architecture, training data, licence, intended use. | Every model you download. Licence matters commercially — Qwen3 is Apache 2.0, Llama is a custom licence with conditions. |
| **safetensors** | Tensor serialisation format that cannot execute code on load, unlike pickle-based `.bin`. | Default for all modern checkpoints. If you are offered a `.bin`, ask why. |
| **Checkpoint shard** | Large models split into `model-00001-of-00004.safetensors` etc., with an index JSON. | Any model over ~5 GB. Partial downloads leave you with a broken shard set. |
| **Revision pinning** | Loading `revision="<commit sha>"` instead of `main`. | Reproducibility. A model repo can change under you; a pinned sha cannot. |
| **Device map** | How `transformers` places layers across devices (`"mps"`, `"auto"`, `"cpu"`). | `device_map="auto"` will silently offload to CPU when VRAM runs out, and your throughput collapses without an error. |
| **dtype** | Numerical precision of weights: `float32`, `bfloat16`, `float16`. | bf16 is the default for training and inference on modern hardware. fp16 overflows more easily; fp32 doubles memory for no quality gain at inference. |
| **HF cache** | `~/.cache/huggingface/hub`, shared across all projects. | Fills up fast. A 4B model in bf16 is ~8 GB. |

---

## Memory accounting — learn this now

You should be able to answer "will this fit?" without trying it.

```
weights_bytes  =  n_params  ×  bytes_per_param
```

| Precision | Bytes/param | 1.7B model |
|-----------|-------------|------------|
| fp32 | 4 | 6.8 GB |
| bf16 / fp16 | 2 | 3.4 GB |
| int8 | 1 | 1.7 GB |
| int4 | 0.5 | 0.85 GB |

**Inference** needs weights + KV cache + activations. Rule of thumb: weights × 1.2.

**Full fine-tuning** needs weights + gradients + optimizer state. With AdamW in bf16 with fp32
master weights, that is roughly:

```
weights (2) + gradients (2) + adam m (4) + adam v (4) + fp32 master (4)  ≈  16 bytes/param
```

A 1.7B model full-fine-tuned needs ~27 GB before activations. On 36 GB unified memory that is
tight-to-impossible. **This single calculation is why LoRA exists** — see Module 12.

---

## Setup

```bash
# Python 3.11 or 3.12. 3.13 works but some kernels lag.
uv venv .venv --python 3.12
source .venv/bin/activate

uv pip install torch transformers accelerate datasets
uv pip install "psycopg[binary]"          # for the agent labs
uv pip install matplotlib pandas          # for the measurement labs
```

Verify MPS is real:

```python
import torch
print(torch.__version__)
print("mps available:", torch.backends.mps.is_available())
print("mps built:", torch.backends.mps.is_built())

x = torch.randn(4096, 4096, device="mps", dtype=torch.bfloat16)
print((x @ x).mean().item())      # should not error
```

If `mps available` is False you are on a CPU build of torch and every lab will be 20× slower.

---

## Download the working models

```python
from huggingface_hub import snapshot_download

for repo in ["Qwen/Qwen3-0.6B", "Qwen/Qwen3-1.7B"]:
    snapshot_download(repo)
    print("ok", repo)
```

```bash
du -sh ~/.cache/huggingface/hub/*     # know what you are storing
```

---

## A load helper you will reuse in every lab

`common/load.py`:

```python
"""Shared model loading. Every lab imports this."""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

SMALL = "Qwen/Qwen3-0.6B"      # plumbing
MAIN = "Qwen/Qwen3-1.7B"       # quality


def device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def load(model_id: str = MAIN, dtype=torch.bfloat16):
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype).to(device())
    model.eval()
    return tok, model


def param_count(model) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
```

Sanity check:

```python
from common.load import load, param_count, device
tok, model = load("Qwen/Qwen3-0.6B")
print(device(), param_count(model))
print(model.config)
```

Read the printed config. `hidden_size`, `num_hidden_layers`, `num_attention_heads`,
`num_key_value_heads`, `intermediate_size`, `vocab_size`, `max_position_embeddings`,
`rope_theta`. Every one of those is a module in this curriculum. By Module 07 you will be able to
predict the parameter count from that config alone, to within a percent.

---

## Interview

**"How much memory to fine-tune a 7B model?"**
Full fine-tune with AdamW: ~16 bytes/param ≈ 112 GB, plus activations — so multiple 80 GB GPUs with
ZeRO/FSDP sharding. LoRA at r=16: base weights in bf16 (14 GB) or 4-bit (3.5 GB), gradients and
optimizer state only for the adapter (tens of MB) — a single 24 GB card. The interesting part of
the answer is that the optimizer state, not the weights, is what forces the sharding.

---

**Next:** [01 — Tokenization](01-tokenization.md)
