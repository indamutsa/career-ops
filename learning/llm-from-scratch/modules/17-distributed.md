# Module 17 — Distributed training

This is one of your four named gaps. You will not run a multi-node job on an M3 Pro, so the goal
here is different from other modules: **understand the memory arithmetic well enough to size a job
and choose a strategy correctly**, then validate the mechanics on two simulated ranks locally and
one real multi-GPU run on rented hardware.

Being able to say "for a 7B full fine-tune you need roughly 112 GB before activations, so that's
FSDP across at least two 80 GB cards, or one card with LoRA" — with the arithmetic behind it — is
worth more in an interview than having babysat someone else's cluster.

---

## Terms

| Term | Meaning |
|------|---------|
| **Data parallel (DP)** | Replicate the model on each device, split the batch. |
| **DDP** | PyTorch's DistributedDataParallel. All-reduce gradients each step. |
| **Model parallel** | Split the model itself across devices. |
| **Tensor parallel (TP)** | Split individual matrices across devices. High communication; keep intra-node. |
| **Pipeline parallel (PP)** | Split by layer; devices form a pipeline. |
| **Pipeline bubble** | Idle time while the pipeline fills and drains. |
| **Sequence / context parallel** | Split the sequence dimension. For very long contexts. |
| **Expert parallel** | Distribute MoE experts across devices. |
| **ZeRO** | DeepSpeed's sharding of optimizer state / gradients / parameters. |
| **ZeRO-1 / 2 / 3** | Shard optimizer states / +gradients / +parameters. |
| **FSDP** | PyTorch-native equivalent of ZeRO-3. |
| **FSDP2 / DTensor** | The current rewrite, per-parameter sharding. |
| **Sharding** | Splitting a tensor across ranks; each holds a slice. |
| **All-reduce / all-gather / reduce-scatter** | Collective ops: combine to all / gather to all / combine and split. |
| **NCCL** | NVIDIA's collective communication library. |
| **Rank / world size / local rank** | Process id / total processes / id within a node. |
| **Gradient accumulation** | More micro-batches per optimizer step. |
| **Activation checkpointing** | Discard activations in the forward pass, recompute in the backward. |
| **CPU offload** | Move optimizer state or parameters to host RAM. |
| **Mixed precision** | bf16/fp16 compute with an fp32 master copy. |
| **Global batch size** | `micro_bs × grad_accum × world_size`. |
| **MFU** | Model FLOPs Utilisation — achieved FLOPs ÷ theoretical peak. |
| **Communication/computation overlap** | Hiding collectives behind compute. |

---

## Concepts

### The memory equation — know this cold

For AdamW mixed-precision full fine-tuning, per parameter:

| Component | Bytes |
|-----------|-------|
| bf16 weights | 2 |
| bf16 gradients | 2 |
| fp32 master weights | 4 |
| Adam `m` (fp32) | 4 |
| Adam `v` (fp32) | 4 |
| **Total** | **16** |

Plus activations, which depend on batch, sequence length, layers and hidden size — and which
activation checkpointing trades away for ~30% extra compute.

| Model | States (16 B/param) | Fits on |
|-------|--------------------|---------|
| 0.6B | ~10 GB | one 24 GB card |
| 1.7B | ~27 GB | one 40 GB card |
| 7B | ~112 GB | 2× 80 GB with FSDP |
| 13B | ~208 GB | 4× 80 GB |
| 70B | ~1.1 TB | 16× 80 GB minimum |

**The optimizer state, not the weights, forces the sharding.** 14 of those 16 bytes are training
overhead. This is the same sentence as Module 12 and it is the answer to two different questions:
why LoRA exists, and why ZeRO exists.

### ZeRO / FSDP, staged

Start from DDP: every rank holds a full copy of everything. Redundant by a factor of `world_size`.

| Stage | Shards | Memory per rank | Extra communication |
|-------|--------|-----------------|---------------------|
| DDP | nothing | 16 B/param | all-reduce gradients |
| ZeRO-1 | optimizer states | 4 + 12/N | + reduce-scatter |
| ZeRO-2 | + gradients | 2 + 14/N | same volume, rearranged |
| ZeRO-3 / FSDP | + parameters | 16/N | + all-gather params in fwd **and** bwd |

ZeRO-3 gives near-linear memory scaling but all-gathers parameters twice per step. On fast
interconnect (NVLink, InfiniBand) that overlaps with compute and costs little. On slow interconnect
(PCIe-only, Ethernet) it dominates and your MFU collapses.

**The decision rule:** ZeRO-2 if the model fits; ZeRO-3 only when it does not. Do not reach for
stage 3 by default — you are paying communication for memory you may not need.

### Choosing a parallelism strategy

```
Does the model + optimizer fit on one device?
├── yes -> DDP.  Simplest, fastest. Stop here.
└── no
    ├── Can LoRA make it fit?          -> LoRA + DDP.  Usually the right answer.
    └── no
        ├── Fits with sharded states?  -> FSDP / ZeRO-2 or 3
        └── Model alone doesn't fit on one device
            ├── within one node        -> + tensor parallel (needs NVLink)
            └── across nodes           -> + pipeline parallel
```

Real large-scale runs combine these — "3D parallelism" is TP × PP × DP. Two placement rules that
sound like experience:

- **Tensor parallel stays inside a node.** It communicates per layer; over Ethernet it is
  unusable.
- **Pipeline parallel crosses nodes.** It only communicates activations at stage boundaries, so it
  tolerates slower links.

### Activation checkpointing

Do not store activations for the backward pass; recompute them from checkpointed boundaries.

- Memory: from O(layers) to roughly O(√layers) with optimal placement.
- Compute: ~30% more (one extra forward).

**Almost always worth it** when it lets you raise the batch size, because larger batches improve
throughput more than the recompute costs. It is the first knob to turn on an OOM, before reaching
for more parallelism.

### Global batch size is the thing that must stay constant

```
global_batch = micro_batch × grad_accum × world_size
```

When you scale from 1 GPU to 8, keeping `micro_batch` and `grad_accum` fixed **multiplies your
effective batch by 8** — a different optimisation problem with different optimal hyperparameters.
Your carefully tuned LR is now wrong and the run behaves differently for reasons that have nothing
to do with distribution.

**Fix `global_batch` and derive `grad_accum` from world size.** Then scaling is a throughput change
only. This is a specific, practical thing that trips people up and mentioning it signals you have
actually done it.

Related: the linear scaling rule says LR should scale roughly with batch size, but it breaks down
at large batch. For post-training, keeping global batch fixed is simpler and safer than
re-tuning LR every time the cluster shape changes.

### MFU — the number that tells you whether it is working

```
MFU = achieved_FLOPs / peak_FLOPs
achieved_FLOPs ≈ 6 × params × tokens / seconds        (fwd+bwd, dense)
```

| MFU | Verdict |
|-----|---------|
| >50% | Excellent |
| 35–50% | Good |
| 20–35% | Investigate |
| <20% | Something is badly wrong |

Common causes of low MFU: communication not overlapping, dataloader starving the GPU, too much
padding, activation checkpointing without a compensating batch increase, or a batch too small to
saturate the device.

**Report MFU, not just tokens/second.** Tokens/second is not comparable across hardware; MFU is.
That habit reads as someone who has optimised a real training job.

### What actually goes wrong in practice

| Problem | Symptom | Cause |
|---------|---------|-------|
| One rank OOMs | Job dies at a random step | Uneven sequence lengths in that rank's batch |
| Hang with no error | Job sits forever | Ranks disagree on collective order — usually a conditional branch |
| Loss differs across ranks | Divergence | Different seeds, or dataloader not sharded |
| Throughput far below linear | Poor scaling | Communication not overlapping; check interconnect |
| Checkpoint won't load | Shape mismatch | Sharded checkpoint loaded with a different world size |
| NaN on some ranks only | Instability | fp16 overflow; use bf16 |
| Slow start every run | Recompilation / NCCL init | Expected; amortise over a long run |

The hang is the one worth understanding: collectives are **synchronous and ordered**. If rank 0
takes an `if` branch that calls an all-reduce and rank 1 does not, rank 0 waits forever with no
error message. Any data-dependent control flow around a collective is a latent deadlock.

---

## Where it's used

- **Any full fine-tune above ~1B**, and any LoRA run above ~13B.
- **RL post-training**, which is worse: you hold a policy, a reference model, and rollout
  generation simultaneously. Memory pressure is higher than SFT at the same model size.
- **The JD's "own open-source LLMs"** — a team training its own models runs multi-node.

---

## Labs

Labs 1–4 run on your laptop. Labs 5–7 are `[GPU]` and belong on a rented multi-GPU box; budget
one session.

### Lab 1 — The memory calculator

```python
def training_memory(n_params, dtype_bytes=2, optimizer="adamw",
                    zero_stage=0, world_size=1, lora_frac=None):
    if lora_frac is not None:
        frozen = n_params * dtype_bytes
        trainable = n_params * lora_frac
        states = trainable * (2 + 2 + 4 + 4 + 4)
        return (frozen + states) / 1e9

    per_param = {"adamw": 16, "sgd_momentum": 10, "sgd": 8, "adafactor": 8}[optimizer]
    if zero_stage == 0:
        total = n_params * per_param
    elif zero_stage == 1:
        total = n_params * (2 + 2) + n_params * 12 / world_size
    elif zero_stage == 2:
        total = n_params * 2 + n_params * 14 / world_size
    else:
        total = n_params * per_param / world_size
    return total / 1e9


for name, n in [("Qwen3-0.6B", 0.6e9), ("Qwen3-1.7B", 1.7e9),
                ("Qwen3-8B", 8e9), ("Llama-3-70B", 70e9)]:
    print(f"\n{name}")
    print(f"  full FT, 1 GPU        {training_memory(n):>8.1f} GB")
    for st in (1, 2, 3):
        print(f"  ZeRO-{st}, 8 GPUs        {training_memory(n, zero_stage=st, world_size=8):>8.1f} GB/rank")
    print(f"  LoRA r=16 (~0.5%)     {training_memory(n, lora_frac=0.005):>8.1f} GB")
```

**Deliverable:** a table you can reproduce from memory in an interview. Then answer, with numbers:
*what is the largest model I can full-fine-tune on 8×80 GB, and on 8×40 GB?*

### Lab 2 — Measure activation memory empirically

```python
import torch
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B",
                                             torch_dtype=torch.bfloat16).to("mps")

def peak(bs, seqlen, ckpt=False):
    model.gradient_checkpointing_enable() if ckpt else model.gradient_checkpointing_disable()
    torch.mps.empty_cache()
    base = torch.mps.current_allocated_memory()
    ids = torch.randint(0, 1000, (bs, seqlen), device="mps")
    out = model(ids, labels=ids)
    out.loss.backward()
    p = torch.mps.current_allocated_memory() - base
    model.zero_grad(set_to_none=True)
    return p / 1e9

print(f"{'bs':>4}{'seq':>6}{'no ckpt GB':>13}{'ckpt GB':>10}{'saving':>9}")
for bs in (1, 2, 4):
    for seq in (512, 1024, 2048):
        a, b = peak(bs, seq), peak(bs, seq, ckpt=True)
        print(f"{bs:>4}{seq:>6}{a:>13.2f}{b:>10.2f}{1-b/a:>8.0%}")
```

**Observe:** activation memory grows with `bs × seqlen`; checkpointing cuts it substantially.
Time both to measure the compute cost, and confirm it lands near 30%.

### Lab 3 — Simulate two ranks with gloo on CPU

You can run real collectives locally. This teaches the mechanics without a GPU.

```python
# save as ddp_demo.py, run:  torchrun --nproc_per_node=2 ddp_demo.py
import os, torch, torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

dist.init_process_group("gloo")
rank, world = dist.get_rank(), dist.get_world_size()
torch.manual_seed(0)

model = DDP(nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 8)))
opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

torch.manual_seed(100 + rank)                 # different data per rank
x, y = torch.randn(8, 64), torch.randn(8, 8)

for step in range(3):
    loss = ((model(x) - y) ** 2).mean()
    loss.backward()
    g = model.module[0].weight.grad[0, :3].clone()
    opt.step(); opt.zero_grad()
    print(f"rank {rank} step {step} loss {loss.item():.4f} grad[:3] {g.tolist()}")

dist.barrier()
if rank == 0:
    print("\nNote: losses differ (different data) but grads are IDENTICAL (all-reduced).")
dist.destroy_process_group()
```

**The point:** each rank computes a different loss on different data, but after the backward pass
the gradients are identical because DDP all-reduced them. That is the entire idea of data
parallelism, demonstrated in 20 lines.

### Lab 4 — Reproduce the collective deadlock

Deliberately create the hang so you recognise it:

```python
# ADD THIS to ddp_demo.py and watch it hang
if rank == 0:
    dist.all_reduce(torch.zeros(1))     # only rank 0 calls it
print("never reached")
```

Run it, watch it hang with no error, and kill it. **Now you will recognise this instantly in
production**, which is worth more than reading about it. The general rule: never put a collective
behind data-dependent control flow.

### Lab 5 `[GPU]` — Real FSDP run

On a 2× or 4× GPU box:

```bash
accelerate config     # answer: multi-GPU, FSDP, FULL_SHARD, bf16
accelerate launch --config_file fsdp.yaml train_sft.py
```

```yaml
# fsdp.yaml
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
mixed_precision: bf16
num_processes: 4
fsdp_config:
  fsdp_sharding_strategy: FULL_SHARD
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_transformer_layer_cls_to_wrap: Qwen3DecoderLayer
  fsdp_state_dict_type: SHARDED_STATE_DICT
  fsdp_use_orig_params: true
  fsdp_activation_checkpointing: true
```

Measure per-rank memory and step time at world sizes 1, 2, 4. **Keep `global_batch` constant** by
adjusting `grad_accum`. Plot tokens/second against world size and compute scaling efficiency.

### Lab 6 `[GPU]` — Compute MFU

```python
def mfu(n_params, tokens, seconds, peak_tflops, world_size):
    achieved = 6 * n_params * tokens / seconds
    return achieved / (peak_tflops * 1e12 * world_size)

# A100 bf16 ~312 TFLOPS, H100 bf16 ~990 TFLOPS (dense, no sparsity)
print(f"MFU: {mfu(1.7e9, tokens_seen, elapsed, 312, 4):.1%}")
```

Then try to raise it: bigger micro-batch, longer sequences, fewer padding tokens, more dataloader
workers. **Record what moved the number.** That list is a real answer to "how do you optimise a
training job".

### Lab 7 `[GPU]` — Break the global-batch invariant on purpose

Run the same job twice at world size 4:

| Run | micro_bs | grad_accum | global batch |
|-----|----------|------------|--------------|
| A | 4 | 8 | 128 |
| B | 4 | 32 | 512 |

Same LR, same data, same steps. **They will not match**, and the difference has nothing to do with
distribution — B is a different optimisation problem. This is exactly the confound that appears
when someone scales from 1 GPU to 8 without adjusting `grad_accum`, and having reproduced it
deliberately means you will never lose a week to it.

---

## Failure modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| OOM at a random step | Long sequence landed on one rank | Length-grouped batching; cap seq length |
| Job hangs, no error | Ranks disagree on collective order | No collectives behind data-dependent branches |
| Results change when scaling GPUs | Global batch changed | Fix global batch; derive grad_accum |
| Throughput scales sub-linearly | Communication not overlapping | Check interconnect; ZeRO-2 instead of 3 |
| MFU under 20% | Dataloader, padding, or tiny batch | Profile before adding parallelism |
| NaN on some ranks | fp16 overflow | bf16 |
| Checkpoint won't load | World size changed | `SHARDED_STATE_DICT`, or consolidate on save |
| ZeRO-3 slower than ZeRO-2 | Model fit anyway | Use the lowest stage that fits |
| Loss diverges only when distributed | Dataloader not sharded; ranks see the same data | Distributed sampler |
| TP across nodes is glacial | TP needs NVLink | Keep TP intra-node, PP inter-node |

---

## Interview

**"How would you fine-tune a 70B model?"**
First question is whether it needs to be a full fine-tune, because LoRA on 70B is roughly 140 GB
of frozen bf16 weights plus a small adapter — that's 2×80 GB, versus about 1.1 TB of optimizer
states for full AdamW, which is 16 cards minimum. Assuming full FT is genuinely required: FSDP with
full sharding, activation checkpointing, bf16, transformer-layer auto-wrap, and sharded
checkpoints. Tensor parallel inside a node where there's NVLink, pipeline parallel across nodes
since it only communicates at stage boundaries. And I'd fix the global batch size and derive
grad_accum from world size, so changing the cluster shape is a throughput change and not a
different optimisation problem.

**"Why does a model that's 140 GB in bf16 need over a terabyte to train?"**
Because the weights are the small part. With AdamW mixed precision you carry bf16 weights, bf16
gradients, an fp32 master copy, and Adam's two fp32 moment buffers — about 16 bytes per parameter,
of which only 2 are the weights. That's the whole reason ZeRO and FSDP exist, and it's the same
reason LoRA works: freeze the base and 14 of those 16 bytes disappear for it.

**"ZeRO-2 or ZeRO-3?"**
The lowest stage that fits. ZeRO-3 all-gathers parameters in both the forward and backward pass, so
on fast interconnect it overlaps with compute and costs little, but on PCIe-only or Ethernet it
dominates and MFU collapses. If the model fits with ZeRO-2 there's no reason to pay for stage 3.
People reach for stage 3 by default and then wonder why scaling is sub-linear.

**"Your job hangs with no error. What do you check?"**
Collective mismatch first — collectives are synchronous and ordered, so if one rank calls an
all-reduce and another doesn't, the first waits forever with no message. That almost always means a
collective behind data-dependent control flow, like an `if` on batch contents or an early-exit on
one rank. After that: NCCL init and network config, a rank that OOMed and died silently while the
others wait, and dataloader deadlocks from worker processes. `TORCH_DISTRIBUTED_DEBUG=DETAIL` and
NCCL's async error handling will usually name it.

**"How do you know your training job is efficient?"**
MFU, not tokens per second — tokens per second isn't comparable across hardware. Roughly 6 × params
× tokens over elapsed time, divided by peak FLOPs times world size. Above 50% is excellent, below
20% means something structural is wrong, and at that point I profile before adding parallelism. The
usual culprits are dataloader starvation, excessive padding from unsorted batches, and communication
that isn't overlapping with compute.

**"How honest are you about your experience here?"**
Direct: I've sized and reasoned about these jobs and run FSDP on rented multi-GPU, but I haven't
operated a large multi-node cluster in production. What I do bring is the production side of it —
at Expedia I built the pipeline orchestration and the evaluation gates around training, which is
the part that usually decides whether a training programme is reproducible. The parallelism
arithmetic I know cold; the operational scars at 512-GPU scale I'd be learning from the team.

---

## Checkpoint

1. Reproduce the 16 bytes/param breakdown from memory.
2. Produce the memory table across models, ZeRO stages, and LoRA.
3. Report activation memory with and without checkpointing, plus the compute cost.
4. Run the two-rank gloo demo and explain why losses differ but gradients don't.
5. Reproduce the collective deadlock deliberately.
6. `[GPU]` Report FSDP scaling efficiency at world size 1/2/4 with global batch held constant.
7. `[GPU]` Report MFU and what raised it.
8. State the parallelism decision tree without looking.

---

**Next:** [18 — Quantization](18-quantization.md) ·
**Back:** [16 — Distillation](16-distillation.md) · [Syllabus](../SYLLABUS.md)
