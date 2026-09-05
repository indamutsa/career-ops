# Module 18 — Quantization

Quantization is how a model that does not fit, fits. It sits between Module 17 (training memory)
and Module 19 (serving memory) and it answers a question you will be asked in some form: *how do
you run this on hardware you can actually afford?*

---

## Terms

| Term | Meaning |
|------|---------|
| **Quantization** | Representing weights/activations in fewer bits. |
| **fp32 / fp16 / bf16** | 32-bit float / 16-bit float / 16-bit "brain float" with fp32's exponent range. |
| **fp8 (E4M3 / E5M2)** | 8-bit floats; two exponent/mantissa splits. |
| **int8 / int4** | 8- and 4-bit integers. |
| **NF4** | 4-bit NormalFloat — levels at the quantiles of a normal distribution. |
| **Scale / zero-point** | The affine map from integer to real: `x ≈ s·(q − z)`. |
| **Symmetric / asymmetric** | Zero-point fixed at 0 / learned. |
| **Per-tensor / per-channel / per-group** | Granularity of the scale. Finer = more accurate, more overhead. |
| **Group size** | Weights sharing one scale (typically 64 or 128). |
| **PTQ** | Post-Training Quantization. No retraining. |
| **QAT** | Quantization-Aware Training. Simulate quantization during training. |
| **Calibration set** | Sample data used to choose scales. |
| **GPTQ** | Second-order PTQ, layer-by-layer error compensation. |
| **AWQ** | Activation-aware; protects the ~1% of channels with large activations. |
| **GGUF** | llama.cpp's quantized format (`Q4_K_M`, `Q5_K_S`, …). |
| **bitsandbytes** | On-the-fly 8-/4-bit for training (QLoRA) — CUDA only. |
| **SmoothQuant** | Shifts activation outliers into weights so both quantize well. |
| **Outlier channels** | The few dimensions with huge activation magnitudes. The whole difficulty. |
| **KV cache quantization** | Storing K/V in int8/fp8. Directly buys concurrency. |
| **Weight-only vs W&A** | Quantize weights only / weights and activations. |
| **Dequantization overhead** | Cost of converting back to compute dtype. |
| **Perplexity delta** | The standard quality-loss measure. |

---

## Concepts

### The affine map

```
q = round(x / s) + z          # quantize
x̂ = s · (q − z)               # dequantize
```

`s` (scale) and `z` (zero-point) come from the observed range. Error is bounded by `s/2`, so
**anything that widens the range widens the error for every value in the group** — which is the
entire outlier problem in one sentence.

Finer granularity shrinks the range each scale must cover:

| Granularity | Scales | Accuracy | Overhead |
|-------------|--------|----------|----------|
| Per-tensor | 1 | Worst | Negligible |
| Per-channel | one per output channel | Good | Small |
| Per-group (128) | one per 128 weights | Best | ~0.1–0.25 bits/weight |

Modern 4-bit methods are per-group with group size 64–128. That is why "4-bit" is really ~4.25
bits/weight in practice.

### Why weights quantize easily and activations do not

**Weights** are static, roughly Gaussian, and known ahead of time. You can measure their
distribution exactly and pick optimal levels. NF4 does exactly this — its levels are the quantiles
of a normal distribution, which fits pretrained weights better than uniform int4 at the same width.

**Activations** are input-dependent and have severe outliers: in large models a handful of channels
carry values 10–100× the rest, consistently. A per-tensor scale sized for those outliers destroys
resolution everywhere else.

Three responses:

- **LLM.int8()** — keep outlier channels in fp16, quantize the rest to int8. Correct, slow.
- **SmoothQuant** — migrate the difficulty: divide activations by a per-channel factor and multiply
  the corresponding weight columns by it. Mathematically equivalent output, both sides now
  quantizable.
- **AWQ** — do not quantize the ~1% of weight channels that correspond to large activations, and
  scale the rest. Weight-only, so no activation quantization needed at all.

**Weight-only quantization is the pragmatic default for serving**, because decode is
bandwidth-bound (Module 19) — shrinking the weights is exactly what you want, and the activations
stay in bf16 where they are safe.

### PTQ methods compared

| Method | Bits | Needs | Quality | Speed | Use |
|--------|------|-------|---------|-------|-----|
| bitsandbytes NF4 | 4 | nothing | Good | Slow (per-block dequant) | **Training** (QLoRA) |
| GPTQ | 4/3 | calibration | Very good | Fast kernels | Serving |
| AWQ | 4 | calibration | Very good, often best at 4-bit | Fast | Serving |
| GGUF `Q4_K_M` | ~4.5 | nothing | Good | Fast on CPU/Metal | Local, llama.cpp |
| fp8 | 8 | nothing (H100+) | Near-lossless | Very fast | Serving on modern NVIDIA |
| int8 W&A | 8 | calibration | Good with SmoothQuant | Fast | Throughput-bound serving |

**The practical split:** bitsandbytes NF4 for QLoRA training because it needs no calibration and
works on an unmodified checkpoint; AWQ or GPTQ for serving because they have fast inference
kernels. Using bitsandbytes in production serving is a common and costly mistake — it is optimised
for training memory, not inference throughput.

### The quality/size curve

Typical perplexity degradation on a 7B model:

| Precision | Size | Δ perplexity | Verdict |
|-----------|------|--------------|---------|
| bf16 | 14 GB | 0 | Baseline |
| fp8 | 7 GB | ~0.01 | Effectively free |
| int8 | 7 GB | ~0.05 | Effectively free |
| 4-bit (AWQ/GPTQ) | 4 GB | ~0.15–0.4 | Usually acceptable |
| 3-bit | 3 GB | ~1.0+ | Noticeable |
| 2-bit | 2 GB | large | Research territory |

**8-bit is close to free. 4-bit is the sweet spot. Below 4-bit, quality falls off a cliff.**

Two important caveats:

1. **Bigger models tolerate quantization better.** A 4-bit 70B usually beats a bf16 13B at similar
   memory. "Quantize a bigger model" often dominates "run a smaller one at full precision" — that
   is a genuinely useful deployment heuristic.
2. **Perplexity understates the damage on hard tasks.** Reasoning, long-context and structured
   output degrade more than perplexity suggests. Always evaluate on your *task* (Module 23), not
   on perplexity alone.

### KV cache quantization

Often more valuable than weight quantization for long-context serving, because at high concurrency
the KV cache is larger than the weights (Module 03's Qwen3-1.7B example: ~30 GB of KV against 3.4
GB of weights).

int8 KV halves it and roughly doubles concurrency, at small quality cost. fp8 KV on H100 is close
to free. **If you are memory-bound at serving time, quantize the KV cache before you quantize the
weights** — that ordering is a good thing to volunteer.

### Where quantization interacts with fine-tuning

- **QLoRA** (Module 12): 4-bit frozen base, bf16 LoRA adapter. Compute is bf16; only *storage* is
  4-bit.
- **Merging into a quantized base is lossy.** Dequantize → merge → requantize, and accept error.
  For multi-adapter serving you keep adapters unmerged anyway (Module 20).
- **Quantize after training, not before**, unless you are doing QAT. Train in bf16 (or QLoRA), then
  quantize the result for serving, then re-evaluate. That last step is not optional — the
  quantized model is a different model.

---

## Where it's used

- **Training on constrained hardware** — QLoRA is why a 7B fine-tune fits on a consumer GPU.
- **Serving cost** — 4-bit weights mean more concurrency per dollar.
- **Long-context serving** — KV quantization is the lever.
- **Edge/local** — GGUF is why models run on laptops.
- **Your laptop** — everything you run locally is quantized in some form.

---

## Labs

### Lab 1 — Implement quantization from scratch

```python
import torch

def quantize_per_group(w: torch.Tensor, bits=4, group=128, symmetric=False):
    orig = w.shape
    w = w.reshape(-1, group)
    qmax = 2 ** bits - 1
    if symmetric:
        s = w.abs().amax(dim=1, keepdim=True) / (qmax // 2)
        z = torch.zeros_like(s)
    else:
        mn, mx = w.amin(dim=1, keepdim=True), w.amax(dim=1, keepdim=True)
        s = (mx - mn).clamp_min(1e-8) / qmax
        z = torch.round(-mn / s)
    q = torch.clamp(torch.round(w / s) + z, 0, qmax)
    return q, s, z, orig

def dequantize(q, s, z, orig):
    return (s * (q - z)).reshape(orig)


from transformers import AutoModelForCausalLM
m = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B", torch_dtype=torch.float32)
W = m.model.layers[10].mlp.down_proj.weight.data

print(f"{'bits':>5}{'group':>7}{'rel err':>12}{'bits/weight':>14}")
for bits in (8, 4, 3, 2):
    for group in (32, 128, W.shape[1]):
        q, s, z, o = quantize_per_group(W, bits=bits, group=group)
        err = (dequantize(q, s, z, o) - W).norm() / W.norm()
        overhead = 2 * 32 / group          # scale + zero-point, fp32
        print(f"{bits:>5}{group:>7}{err:>12.4f}{bits + overhead:>14.2f}")
```

**Two findings to write down:** error rises sharply below 4 bits, and per-group beats per-tensor
substantially — but the overhead column shows what finer grouping costs. That table is the whole
accuracy/size trade in one place.

### Lab 2 — Find the outlier channels

```python
import torch
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
acts = {}

def hook(name):
    def f(mod, inp, out):
        acts[name] = inp[0].detach().abs()
    return f

h = m.model.layers[10].mlp.down_proj.register_forward_hook(hook("down_proj"))
m(**tok("The quarterly revenue report shows significant growth.", return_tensors="pt"))
h.remove()

a = acts["down_proj"].reshape(-1, acts["down_proj"].shape[-1])
per_ch = a.amax(dim=0)
print(f"median channel max : {per_ch.median():.3f}")
print(f"largest channel max: {per_ch.max():.3f}")
print(f"ratio              : {per_ch.max()/per_ch.median():.1f}x")
top = torch.topk(per_ch, 10)
print("outlier channels:", top.indices.tolist())
```

**Run it on several different inputs.** The same channel indices will keep appearing — the outliers
are a property of the model, not of the input. That persistence is precisely what SmoothQuant and
AWQ exploit, and seeing it yourself makes the explanation concrete.

### Lab 3 — Quantization error is not uniform across layers

```python
for i in range(0, m.config.num_hidden_layers, 4):
    layer = m.model.layers[i]
    for name in ["self_attn.q_proj", "mlp.down_proj", "mlp.gate_proj"]:
        W = layer.get_submodule(name).weight.data
        q, s, z, o = quantize_per_group(W, bits=4, group=128)
        err = (dequantize(q, s, z, o) - W).norm() / W.norm()
        print(f"layer {i:>2} {name:<18} rel err {err:.4f}")
```

**Look for which layers and projections are hardest.** Typically the first and last layers are more
sensitive — which is why most quantization schemes keep the embedding and LM head at higher
precision. That is a detail worth knowing rather than reciting.

### Lab 4 — GGUF quantization end to end

```bash
pip install llama-cpp-python
git clone https://github.com/ggerganov/llama.cpp && cd llama.cpp && make

python convert_hf_to_gguf.py ~/models/Qwen3-1.7B --outfile qwen-f16.gguf --outtype f16
for q in Q8_0 Q5_K_M Q4_K_M Q3_K_M Q2_K; do
  ./llama-quantize qwen-f16.gguf qwen-$q.gguf $q
done
ls -lh qwen-*.gguf
```

Then measure perplexity for each:

```bash
./llama-perplexity -m qwen-Q4_K_M.gguf -f wiki.test.raw
```

**Produce the size / perplexity / tokens-per-second table across all five.** Then — and this is the
part that matters — run your **SQL agent task set** through each and report task accuracy. Compare
the two curves. Perplexity will degrade gently; task accuracy on structured output will degrade
faster. That divergence is the module's key empirical result.

### Lab 5 — Serve a quantized model and measure the real trade

```bash
python -m vllm.entrypoints.openai.api_server \
  --model TheBloke/Qwen-1.8B-AWQ --quantization awq --gpu-memory-utilization 0.9
```

Report, against the bf16 baseline from Module 19: weights memory, KV budget, max concurrency,
throughput, and task accuracy.

**The interesting result is usually that 4-bit weights buy you more concurrency than they cost in
quality** — the freed memory becomes KV cache, and more KV cache means bigger batches, which
(Module 19) is where throughput comes from. Quantization is a throughput lever, not just a
fitting-it-in lever.

### Lab 6 — KV cache quantization

```bash
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-1.7B \
  --kv-cache-dtype fp8 --max-model-len 8192
```

Compare against `--kv-cache-dtype auto`: KV capacity from the startup log, max concurrency,
throughput, and task accuracy. At long context this is usually a larger win than weight
quantization, and being able to say which you would reach for first — with numbers — is the point.

---

## Failure modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| Big quality drop at 4-bit | Per-tensor scales | Per-group, group size 128 |
| Quality collapses below 4-bit | Genuine information loss | Stay at 4-bit; quantize a bigger model instead |
| int8 activations destroy quality | Outlier channels | SmoothQuant, or weight-only |
| Quantized model slower than bf16 | Dequantization overhead; wrong kernel | Use AWQ/GPTQ kernels, not bitsandbytes, for serving |
| Perplexity fine, task broken | Perplexity understates structured-output damage | Evaluate the task |
| Merged LoRA into 4-bit base lost quality | Merging into a quantized base is lossy | Merge in bf16, or keep unmerged |
| OOM despite 4-bit weights | KV cache, not weights, was the constraint | Quantize the KV cache |
| Calibration-dependent results vary | Unrepresentative calibration set | Calibrate on in-domain data |
| Works locally, not in vLLM | Format mismatch (GGUF is llama.cpp only) | AWQ/GPTQ for vLLM |

---

## Interview

**"How do you fit a 70B model on two 80 GB cards for serving?"**
4-bit weight-only quantization — AWQ or GPTQ — takes it from about 140 GB to 35–40, which leaves
most of the memory for KV cache, and KV cache is what determines concurrency. Weight-only rather
than weights-and-activations because decode is bandwidth-bound, so shrinking the weights is exactly
the win you want and the activations stay in bf16 where the outlier channels can't hurt you. I'd
then re-evaluate on the actual task, not just perplexity, because structured output and reasoning
degrade faster than perplexity suggests.

**"Why are activations harder to quantize than weights?"**
Weights are static and roughly Gaussian, so you can measure the distribution exactly and pick
levels that fit it — NF4 literally places its levels at the quantiles of a normal distribution.
Activations are input-dependent and have persistent outlier channels carrying values 10 to 100×
the rest, and a scale sized for those destroys resolution everywhere else. The fixes either keep
the outliers in high precision, as LLM.int8() does, or migrate the difficulty into the weights,
which is SmoothQuant — divide activations by a per-channel factor and multiply the matching weight
columns by it, so the output is unchanged and both sides quantize cleanly.

**"4-bit 70B or bf16 13B, same memory?"**
Usually the quantized 70B. Larger models tolerate quantization noticeably better — there's more
redundancy to absorb the error — so 4-bit 70B typically beats bf16 13B on most benchmarks at
similar footprint. I'd verify on the specific task rather than assume, particularly if it involves
long context or strict structured output, since those are where 4-bit degrades faster than
perplexity implies.

**"You're memory-bound serving long contexts. What do you quantize first?"**
The KV cache, not the weights. At high concurrency with long sequences the KV cache is larger than
the model — for a 1.7B model at 8k context and batch 32 it's around 30 GB against 3.4 GB of
weights. int8 KV roughly doubles concurrency, and fp8 on H100 is nearly free. People reach for
weight quantization by reflex, but if KV is the binding constraint that's optimising the wrong
term.

**"Which quantization method?"**
Depends on the stage. For training, bitsandbytes NF4 — it's what QLoRA uses, it needs no
calibration and it works on an unmodified checkpoint. For serving, AWQ or GPTQ, because they have
fast inference kernels; using bitsandbytes for production serving is a common mistake since it's
optimised for training memory and its per-block dequantization makes inference slow. For local or
CPU, GGUF. And fp8 on H100 or newer, where it's close to lossless and very fast.

---

## Checkpoint

1. Implement quantize/dequantize and produce the bits × group-size error table.
2. Find the outlier channels and show they persist across inputs.
3. Report which layers quantize worst.
4. Produce the GGUF size / perplexity / speed table *and* the task-accuracy curve, and explain the
   divergence.
5. Report concurrency and throughput gains from 4-bit serving.
6. Report the KV-quantization win and say when it beats weight quantization.

---

**Next:** [19 — High-throughput serving](19-serving.md) ·
**Back:** [17 — Distributed training](17-distributed.md) · [Syllabus](../SYLLABUS.md)
