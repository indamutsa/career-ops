# Module 19 — High-throughput serving

Your second infrastructure gap, and the one that maps most directly onto "enterprise scale" in the
JD. The good news: unlike distributed training, you can run a real vLLM server on your laptop and
measure everything in this module for yourself.

The central fact carries over from Module 03: **decode is memory-bandwidth bound, not
compute-bound.** Every serving optimisation that matters is a consequence of that.

---

## Terms

| Term | Meaning |
|------|---------|
| **Throughput** | Tokens or requests per second, aggregate. |
| **Latency** | Time for one request. |
| **TTFT** | Time To First Token. Dominated by prefill. |
| **TPOT / ITL** | Time Per Output Token / Inter-Token Latency. Dominated by decode. |
| **Goodput** | Throughput that meets an SLO. The metric that actually matters. |
| **Static batching** | Wait for a full batch; all requests finish together. |
| **Continuous batching** | Add and retire requests every iteration. |
| **Iteration-level scheduling** | Scheduling decisions each forward pass, not per request. |
| **PagedAttention** | KV cache in fixed-size non-contiguous blocks, like OS paging. |
| **KV block** | The unit of PagedAttention allocation (typically 16 tokens). |
| **Fragmentation** | Wasted memory from variable-length contiguous allocation. |
| **Prefix caching** | Reusing KV for a shared prompt prefix. |
| **RadixAttention** | SGLang's prefix tree over cached prefixes. |
| **Chunked prefill** | Splitting long prefills so decodes aren't starved. |
| **Preemption / swapping** | Evicting a request's KV under pressure; recompute or swap to CPU. |
| **`gpu_memory_utilization`** | Fraction of VRAM vLLM claims for weights + KV. |
| **`max_num_seqs`** | Max concurrent sequences. |
| **Speculative decoding** | Draft-and-verify (Module 08). |
| **Tensor parallel (serving)** | Split weights across GPUs to fit or to speed up. |
| **Multi-LoRA** | Many adapters over one base (Module 20). |
| **Roofline** | Whether a kernel is compute- or bandwidth-bound. |
| **Arithmetic intensity** | FLOPs per byte moved. |

---

## Concepts

### Prefill and decode are two different workloads

| | Prefill | Decode |
|---|---------|--------|
| Processes | All prompt tokens at once | One token per step |
| Parallelism | High — a big matmul | None across the sequence |
| Bound by | Compute | **Memory bandwidth** |
| Arithmetic intensity | High | ~1 FLOP per byte |
| Determines | TTFT | TPOT |
| Batching helps | Somewhat | **Enormously** |

The decode asymmetry is stark: to generate one token you read **every weight in the model** from
HBM. For a 1.7B bf16 model that is 3.4 GB per token. At 400 GB/s that is ~8.5 ms per token
regardless of how little arithmetic you did.

**The consequence:** batching 32 requests reads those same weights *once* and produces 32 tokens.
Throughput goes up ~32× while per-request latency barely moves. Batching is not an optimisation
here — it is the entire economics of LLM serving.

### Continuous batching

Static batching wastes enormously: a batch finishes when its *longest* member finishes, so a
request wanting 20 tokens sits idle while one wanting 500 completes.

Continuous batching schedules at **iteration** level. Every forward pass:

1. Retire finished sequences.
2. Admit waiting ones if there is KV space.
3. Run one decode step for everything currently active.

Typical gain: 2–4× throughput at equal or better latency, purely from not idling. This was the
key idea in Orca and it is why vLLM exists.

### PagedAttention

The KV cache is the scarce resource, and its size is unknown in advance because you do not know how
long the output will be.

Naive approach: allocate `max_model_len` per request contiguously. If a request uses 200 of 4096
reserved tokens, 95% is wasted. Measured fragmentation waste in pre-vLLM systems ran 60–80%.

PagedAttention borrows OS virtual memory: fixed-size blocks (16 tokens), a per-sequence block
table, non-contiguous physical storage. Waste drops to under 4% — internal fragmentation in the
last block only.

Two things it enables beyond memory efficiency:

- **Copy-on-write sharing.** Parallel samples from one prompt (Module 08's `n>1`, and every RL
  rollout group) share the prompt's KV blocks physically. For GRPO with 8 rollouts per prompt this
  is a large, direct saving.
- **Prefix caching.** Shared prefixes are shared blocks across *different* requests.

**Why it matters more KV memory is more concurrency**, and more concurrency is more throughput. The
chain is: PagedAttention → less waste → more sequences resident → bigger batches → better
amortisation of the weight read.

### Sizing the KV cache

From Module 03:

```
kv_bytes = 2 (K,V) × layers × kv_heads × head_dim × 2 (bf16) × seq_len × batch
```

GQA is what makes long-context serving viable — `kv_heads` is 4 or 8 instead of 32, an 8× cut
straight off the KV cache.

Compute your own budget:

```
KV budget = total_VRAM × gpu_memory_utilization − model_weights
max_concurrent ≈ KV budget ÷ (kv_bytes_per_token × avg_seq_len)
```

**This is the capacity-planning calculation** and being able to do it live is a strong signal.

### Chunked prefill

A long prefill occupies the GPU for many milliseconds, during which every decoding request stalls —
their inter-token latency spikes. Chunked prefill splits it into pieces and interleaves decode
steps between them.

Trade-off: slightly worse TTFT for the prefilling request, much better and more *consistent* TPOT
for everyone else. For an interactive product, p99 inter-token latency usually matters more than
one request's TTFT, so chunked prefill is normally on.

### Preemption

When KV runs out, a running request must be evicted:

- **Recompute** — drop its KV, re-prefill later. Cheap in memory, costs compute.
- **Swap** — move KV to CPU RAM, copy back later. Costs PCIe bandwidth.

Either way, **preemption is a signal you are over-subscribed.** vLLM logs it. A production
deployment with constant preemption needs lower `max_num_seqs` or more memory — watch that counter.

### Goodput, not throughput

Raising batch size raises throughput and worsens per-request latency. Past your SLO, extra
throughput is worthless: requests that miss the deadline do not count.

```
goodput = requests/sec that met the SLO
```

Tune `max_num_seqs` for goodput at your actual SLO. A system doing 2,000 tok/s with p99 TTFT of 8
seconds may be strictly worse than one doing 1,200 tok/s at 900 ms, depending on the product.
**Defining the SLO before tuning is the discipline** — otherwise you optimise the number that is
easiest to move.

### vLLM vs SGLang vs TGI vs TensorRT-LLM

| | Strength | Choose when |
|---|----------|-------------|
| **vLLM** | Best all-round, huge model coverage, multi-LoRA | Default |
| **SGLang** | RadixAttention prefix tree, structured output, fast | Heavy prefix sharing; agent workloads |
| **TGI** | HuggingFace-integrated, production-hardened | Already in the HF ecosystem |
| **TensorRT-LLM** | Fastest on NVIDIA, ahead-of-time compiled | Max performance, fixed model, willing to pay in complexity |
| **llama.cpp / Ollama** | CPU/Metal, tiny footprint | Local dev, edge |

For an agent workload — long shared system prompt and tool schemas on every step (Module 22) —
**SGLang's RadixAttention is a genuine advantage** because the prefix tree shares across requests
automatically. Being able to say which you would pick and why is more useful than knowing all four.

---

## Where it's used

- **Any deployment.** Serving cost is the recurring bill of an LLM product.
- **RL rollout generation** — vLLM generates the rollouts; it is a *training* dependency too.
  Speeding up rollouts speeds up your RL loop directly.
- **Evaluation** — running a 1,000-task eval through a vLLM server versus a `generate` loop is the
  difference between minutes and hours.
- **Multi-tenant domain agents** (Module 20).

---

## Labs

Runs on macOS via vLLM's CPU/Metal path or in Docker; small models throughout. `[GPU]` marks what
needs rented hardware.

### Lab 1 — Prove decode is bandwidth-bound

```python
import torch, time
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B",
                                             torch_dtype=torch.bfloat16).to("mps")

weight_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
print(f"model weights: {weight_bytes/1e9:.2f} GB")

ids = tok("Hello", return_tensors="pt").to("mps")
model.generate(**ids, max_new_tokens=5)          # warmup

t0 = time.time()
out = model.generate(**ids, max_new_tokens=100, do_sample=False,
                     pad_token_id=tok.eos_token_id)
dt = time.time() - t0
n = out.shape[1] - ids["input_ids"].shape[1]

print(f"{n} tokens in {dt:.2f}s -> {dt/n*1000:.1f} ms/token")
print(f"implied bandwidth: {weight_bytes*n/dt/1e9:.0f} GB/s")
```

**Compare the implied bandwidth to your hardware's spec** (M3 Pro is ~150 GB/s unified). If you are
within a factor of two, you have demonstrated that decode is reading the weights and little else —
the model's arithmetic is essentially free by comparison.

### Lab 2 — Batching is nearly free

```python
prompts = ["Explain data drift in one sentence."] * 32

for bs in (1, 2, 4, 8, 16, 32):
    batch = tok(prompts[:bs], return_tensors="pt", padding=True).to("mps")
    t0 = time.time()
    out = model.generate(**batch, max_new_tokens=64, do_sample=False,
                         pad_token_id=tok.eos_token_id)
    dt = time.time() - t0
    total = bs * 64
    print(f"bs={bs:>3}  {dt:>6.2f}s  {dt/64*1000:>7.1f} ms/step  "
          f"{total/dt:>8.1f} tok/s  {dt:>5.2f}s/request")
```

**The finding:** ms/step rises far more slowly than batch size. Total throughput scales close to
linearly. Plot both columns — that plot is the economic argument for continuous batching and it is
worth having in your notes.

### Lab 3 — vLLM server, real measurements

```bash
pip install vllm
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-0.6B \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.85 \
  --enable-prefix-caching \
  --port 8000
```

Read the startup log. It prints the **KV cache size and how many tokens it can hold** — that number
is your concurrency budget. Write it down.

```bash
python -m vllm.entrypoints.cli.benchmark.serve \
  --model Qwen/Qwen3-0.6B --num-prompts 200 --request-rate 10
```

Record: throughput, mean/p50/p99 TTFT, mean/p99 TPOT. Then sweep `--request-rate` at 1, 5, 10, 20,
50 and **plot latency against load**. You will see the knee where the system saturates and latency
goes vertical. That knee is your capacity.

### Lab 4 — Prefix caching, measured properly

This is the Module 09 lab done for real.

```python
import requests, time

SYS = open("prompts/agent_system.txt").read()      # long, stable
QS  = ["How many orders in March?", "Top 5 customers?", "Average order value?"] * 10

def call(prompt):
    t0 = time.time()
    requests.post("http://localhost:8000/v1/completions",
                  json={"model": "Qwen/Qwen3-0.6B", "prompt": prompt, "max_tokens": 32})
    return time.time() - t0

stable_first = [call(f"{SYS}\n\nUser: {q}\nAssistant:") for q in QS]
variable_first = [call(f"User: {q}\n\n{SYS}\nAssistant:") for q in QS]

print(f"stable-first   first {stable_first[0]:.3f}s  rest mean {sum(stable_first[1:])/len(stable_first[1:]):.3f}s")
print(f"variable-first first {variable_first[0]:.3f}s  rest mean {sum(variable_first[1:])/len(variable_first[1:]):.3f}s")
```

**Expected:** stable-first shows a large first call and much faster subsequent ones. Variable-first
shows no improvement. Also check vLLM's logged prefix cache hit rate. **This is a free, large win
that costs nothing but prompt ordering**, and it is a good thing to raise unprompted.

### Lab 5 — Capacity planning by hand, then verified

```python
def kv_bytes_per_token(layers, kv_heads, head_dim, dtype_bytes=2):
    return 2 * layers * kv_heads * head_dim * dtype_bytes

cfg = model.config
per_tok = kv_bytes_per_token(cfg.num_hidden_layers,
                             getattr(cfg, "num_key_value_heads", cfg.num_attention_heads),
                             cfg.hidden_size // cfg.num_attention_heads)

for vram, util in [(24, 0.9), (40, 0.9), (80, 0.9)]:
    weights = weight_bytes / 1e9
    kv_budget = vram * util - weights
    for seq in (1024, 4096, 16384):
        n = int(kv_budget * 1e9 / (per_tok * seq))
        print(f"{vram}GB, seq={seq:>6}: KV budget {kv_budget:>5.1f} GB -> ~{n:>5} concurrent")
```

**Then verify against vLLM's own startup log.** If your arithmetic matches within ~10%, you can do
capacity planning live in an interview. If it does not, find out why — that is the lab.

### Lab 6 — Goodput under an SLO

```python
SLO_TTFT_MS, SLO_TPOT_MS = 1000, 50

def goodput(results):
    ok = [r for r in results if r["ttft_ms"] < SLO_TTFT_MS and r["tpot_ms"] < SLO_TPOT_MS]
    return len(ok) / results[-1]["end_s"], len(ok) / len(results)

for max_seqs in (8, 16, 32, 64, 128, 256):
    # restart the server with --max-num-seqs {max_seqs}, run the benchmark
    gp, frac = goodput(run_benchmark(rate=20))
    print(f"max_num_seqs={max_seqs:>4}  goodput {gp:>6.1f} req/s  SLO met {frac:>6.1%}")
```

**Find the goodput maximum and note that it is not the throughput maximum.** Being able to say
"raw throughput peaked at 256 concurrent but goodput peaked at 32, so I shipped 32" is a
production answer.

### Lab 7 — Serve the SQL agent's rollouts through vLLM

Replace `env/model.py`'s local `generate` with a vLLM client and re-run Lab 01's baseline.

```python
from vllm import LLM, SamplingParams

llm = LLM(model="Qwen/Qwen3-1.7B", enable_prefix_caching=True, max_model_len=8192)
params = SamplingParams(temperature=1.0, top_p=1.0, n=8, max_tokens=256)   # RL settings (M08)

outputs = llm.generate(prompts, params)
```

Measure wall-clock time for the full baseline both ways. **The speedup is your RL iteration
speed** — and since GRPO's cost is dominated by rollout generation, this is directly how long each
experiment takes. Report the number; "rollout generation went from 40 minutes to 6" is a concrete
engineering result.

Also note the `n=8` interaction with PagedAttention: the eight rollouts share the prompt's KV
blocks copy-on-write, so group sampling is much cheaper than eight separate requests.

### Lab 8 `[GPU]` — Tensor parallel and speculative decoding

```bash
python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-8B \
  --tensor-parallel-size 2

python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen3-8B \
  --speculative-model Qwen/Qwen3-0.6B --num-speculative-tokens 5
```

Measure TTFT, TPOT and throughput for each against the single-GPU baseline. Note that TP helps
latency *and* capacity but scales sub-linearly, and that speculative decoding helps TPOT but
consumes memory that would otherwise be KV cache — so it can *reduce* max concurrency. That
trade-off is the interesting part.

---

## Failure modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| Low throughput despite a fast GPU | Not batching | Continuous batching; raise `max_num_seqs` |
| OOM at startup | `gpu_memory_utilization` too high, or `max_model_len` too long | Lower both |
| OOM under load | KV exhausted | Lower `max_num_seqs` or `max_model_len` |
| Constant preemption in logs | Over-subscribed | Lower concurrency, or add memory |
| p99 inter-token latency spikes | Long prefills blocking decode | Enable chunked prefill |
| Prefix cache never hits | Variable content first in the prompt | Stable content first |
| Throughput great, users complain | Optimised throughput, not goodput | Define the SLO, tune for it |
| Long-context serving impossible | KV cache dominated by MHA | GQA model, or quantized KV cache |
| Speculative decoding reduced capacity | Draft model ate KV memory | Measure both; it's a trade |
| Different output than local `generate` | Different default sampling params | Pin them explicitly |
| First request very slow | Compilation / graph capture | Warm up before serving traffic |

---

## Interview

**"Why does batching help LLM inference so much?"**
Because decode is memory-bandwidth bound, not compute bound. To generate one token you read every
weight in the model from HBM — for a 1.7B bf16 model that's 3.4 GB per token, and at 400 GB/s
that's about 8.5 ms no matter how little arithmetic you actually did. Batch 32 requests and you
read those same weights once and produce 32 tokens. Throughput goes up almost 32× while per-request
latency barely moves. That asymmetry is the entire economics of serving, and it's why continuous
batching rather than static batching is the single biggest win — static batching idles the whole
batch waiting for its longest member.

**"What does PagedAttention actually solve?"**
Fragmentation. You don't know how long a response will be, so the naive approach reserves
`max_model_len` of contiguous KV per request, and a request that uses 200 of 4096 reserved tokens
wastes 95%. Measured waste in pre-vLLM systems was 60–80%. PagedAttention allocates fixed 16-token
blocks with a per-sequence block table, exactly like OS paging, so waste drops under 4% — just the
last partial block. And because blocks are shared by reference, parallel samples from one prompt
share the prompt's KV copy-on-write. That matters directly for RL: a GRPO group of 8 rollouts
shares one prompt's cache rather than duplicating it eight times.

**"How would you size a deployment?"**
Arithmetically, then verify. KV per token is 2 × layers × kv_heads × head_dim × dtype_bytes — GQA
is what makes this tractable, since kv_heads is 4 or 8 instead of 32. KV budget is total VRAM times
utilisation minus model weights. Divide by KV-per-token times expected sequence length and you have
max concurrency. Then I check it against vLLM's startup log, which prints the actual KV capacity,
and I'd expect to be within about 10%. From there I'd tune `max_num_seqs` for *goodput* at a
defined SLO rather than raw throughput — 2,000 tokens/second with an 8-second p99 TTFT can be
strictly worse than 1,200 at 900 ms, depending on the product.

**"Which serving stack?"**
vLLM as the default — best coverage, mature multi-LoRA, good enough at everything. SGLang if the
workload has heavy prefix sharing, which agent workloads do: every step re-sends the system prompt
and tool schemas, and RadixAttention shares those across requests via a prefix tree rather than
just within one. TensorRT-LLM if I need maximum performance on NVIDIA with a fixed model and I'm
willing to pay in build complexity. For the agent post-training work I'd start on vLLM and benchmark
SGLang, because that's a workload where the difference could be real rather than theoretical.

**"How does serving affect training?"**
For RL it *is* training infrastructure. GRPO's wall-clock cost is dominated by rollout generation —
you're sampling n completions per prompt for every prompt in every batch — so the generation stack
determines how long an experiment takes. Moving rollouts from a naive `generate` loop to vLLM with
prefix caching and grouped sampling is a large multiple, and that translates directly into how many
experiments you can run in a week. It's also why the sampling parameters have to be set explicitly
rather than inherited: serving defaults are 0.7 with top-p, and RL needs 1.0 untruncated.

---

## Checkpoint

1. Report your implied decode bandwidth and compare it to hardware spec.
2. Plot ms/step and total throughput against batch size.
3. Report throughput, p50/p99 TTFT and TPOT from a real vLLM benchmark, and find the latency knee.
4. Demonstrate the prefix-caching win and the ordering that defeats it.
5. Compute max concurrency by hand and verify against vLLM's startup log.
6. Report the goodput maximum and show it differs from the throughput maximum.
7. Report the speedup from serving your agent's rollouts through vLLM.

---

**Next:** [20 — Multi-LoRA serving](20-multi-lora.md) ·
**Back:** [18 — Quantization](18-quantization.md) · [Syllabus](../SYLLABUS.md)
