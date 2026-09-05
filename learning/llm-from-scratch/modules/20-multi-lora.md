# Module 20 — Multi-LoRA serving

This is the shortest module and one of the most strategically important, because it is the concrete
architecture behind the phrase in Isaac's message: **"domain-specific agents at enterprise scale"**.
When a company says that, they almost always mean *one base model, many adapters, many permission
sets* — not many bespoke models.

---

## Terms

| Term | Meaning |
|------|---------|
| **Multi-LoRA serving** | One base model in memory, many adapters, per-request selection. |
| **Adapter swapping** | Loading/unloading adapters at request time. |
| **Unmerged inference** | Keeping `BA` separate so it can be swapped (vs merged into `W`). |
| **S-LoRA** | Serving thousands of adapters on one base; unified paging for adapter weights. |
| **Punica** | Batched kernels applying *different* adapters within one batch. |
| **SGMV** | Segmented Gather Matrix-Vector — the kernel that makes heterogeneous batching work. |
| **Heterogeneous batching** | One batch containing requests using different adapters. |
| **Adapter cold start** | Latency to load an adapter not currently resident. |
| **Adapter registry** | The mapping from tenant/domain to adapter artefact and version. |
| **`max_loras`** | Adapters resident concurrently in the serving engine. |
| **`max_lora_rank`** | Max rank the engine will accept. Sized at startup. |
| **Tenant isolation** | Guaranteeing one customer's adapter and data never touch another's. |

---

## Concepts

### The economics

Serving 50 domain-specialised 7B models:

| Approach | Memory | Notes |
|----------|--------|-------|
| 50 full fine-tunes | 50 × 14 GB = **700 GB** | 9 × 80 GB cards, mostly duplicated weights |
| 50 merged LoRAs | 50 × 14 GB = **700 GB** | Merging throws away the entire advantage |
| **1 base + 50 unmerged LoRAs** | 14 GB + 50 × 0.04 GB ≈ **16 GB** | **One card** |

That is a ~44× reduction, and it is the whole reason this architecture exists. The adapters are
noise next to the base.

**The catch that people miss: merging destroys it.** Module 12 says merging gives zero inference
overhead, and that is true for the single-adapter case. For multi-tenant serving you must keep them
unmerged, and pay a small per-request cost instead. Knowing *when* each is right is the point.

### How heterogeneous batching works

The hard problem: continuous batching (Module 19) wants to batch requests together, but requests in
one batch may want *different* adapters. Naively you would have to batch per adapter, which
destroys the batching win — the thing that made serving economical in the first place.

The base computation is shared:

```
y = W·x  +  (α/r)·B_i·A_i·x
    ^^^^     ^^^^^^^^^^^^^^^
    shared   per-request adapter
```

`W·x` is one big batched matmul over the whole batch, exactly as normal. The adapter term is a
*gather* over per-request adapter weights followed by two small matmuls — which is what SGMV
(Punica) implements as a single kernel.

So: **one batch, many adapters, one pass.** Overhead is typically 5–15% throughput versus the base
model alone. That is the number to know.

### What it costs

| Cost | Magnitude | Mitigation |
|------|-----------|------------|
| Extra matmuls per request | 5–15% throughput | Inherent; usually acceptable |
| Adapter memory | ~40 MB each at r=16 | Trivial |
| Cold start when not resident | 10s–100s of ms | LRU cache; pin hot adapters |
| Rank must be ≤ `max_lora_rank` | Sized at startup | Standardise on one rank across adapters |
| Scheduling complexity | Some | Handled by the engine |

**Standardising the rank across all adapters is a real operational decision.** Mixed ranks force
`max_lora_rank` to the maximum and waste kernel capacity. Pick r=16 or r=32 as a house standard and
hold to it.

### When to use it (and when not)

**Use it when:**
- Many domains/tenants share one base model.
- Each needs modest behavioural adaptation, not new knowledge.
- Adapters are updated independently and often.
- You want per-tenant rollback and versioning.

**Do not use it when:**
- There is only one adapter → merge it, take the zero overhead.
- Domains need genuinely different base models (different sizes, different licences).
- The adaptation needs new knowledge rather than behaviour → RAG (Module 21), not LoRA.

That last line matters. A LoRA adapter is a low-rank *behavioural* correction (Module 12). If the
finance team's agent needs to know finance *facts*, an adapter is the wrong tool and retrieval is
the right one. Being clear about that boundary is what separates someone who has deployed this from
someone who has read about it.

### The operational picture

```
request  ->  auth / tenant resolution
         ->  adapter registry lookup  (tenant -> adapter id @ version)
         ->  vLLM  (base + LoRARequest(adapter_id))
         ->  tool permissions scoped to that tenant   <- Module 22
         ->  response
```

Four things this architecture must get right, and they are all outside the model:

1. **Tenant → adapter mapping is security-relevant.** Serving tenant A's adapter to tenant B is a
   data-leak-shaped incident even if no data moved, because the adapter encodes their domain
   behaviour.
2. **Adapter versioning and rollback.** Adapters are ~40 MB artefacts — version them like code,
   pin the base model sha they were trained against, and be able to roll one tenant back without
   touching others. **This is the single biggest operational advantage over full fine-tunes.**
3. **Per-tenant evaluation.** Each adapter needs its own golden set (Module 23). A shared eval
   cannot tell you that tenant 12's adapter regressed.
4. **Base model upgrades invalidate every adapter.** When the base moves, all adapters must be
   retrained and re-evaluated. Plan the migration; do not discover it.

Point 4 is the one that bites teams and it is a good thing to raise unprompted — it turns a base
model upgrade from a config change into a programme of work.

---

## Where it's used

- **Enterprise multi-tenant LLM products** — the JD's phrase, decoded.
- **Per-domain agents** — legal, finance, support, each with its own adapter and tool permissions.
- **A/B testing** — two adapter versions served side by side against live traffic.
- **RL post-training** — serve the policy adapter and the reference adapter over one base to halve
  memory during rollout generation.

---

## Labs

### Lab 1 — Train three domain adapters

Using Module 11's pipeline, train three r=16 adapters over the same Qwen3-1.7B base, differing only
in data:

| Adapter | Data |
|---------|------|
| `sql-analyst` | The Lab 01 SQL agent trajectories |
| `terse` | Same tasks, answers rewritten short and factual |
| `explainer` | Same tasks, answers rewritten with reasoning spelled out |

Keep everything else identical — same rank, same LR, same steps. **Same base, same rank, different
behaviour** is the setup the rest of the labs need.

```bash
ls -lh adapters/*/adapter_model.safetensors    # ~40 MB each
```

### Lab 2 — Serve all three at once

```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-1.7B \
  --enable-lora \
  --max-loras 4 \
  --max-lora-rank 16 \
  --lora-modules sql-analyst=./adapters/sql-analyst \
                 terse=./adapters/terse \
                 explainer=./adapters/explainer
```

```python
import requests

for adapter in ["Qwen/Qwen3-1.7B", "sql-analyst", "terse", "explainer"]:
    r = requests.post("http://localhost:8000/v1/completions", json={
        "model": adapter,
        "prompt": "User: How many orders were placed in March?\nAssistant:",
        "max_tokens": 120, "temperature": 0.0,
    })
    print(f"\n--- {adapter} ---\n{r.json()['choices'][0]['text']}")
```

**Four distinct behaviours, one base model in memory.** Confirm with `nvidia-smi` (or vLLM's log)
that memory did not grow by three model copies.

### Lab 3 — Measure the overhead

```python
import time, requests, statistics

def bench(model, n=50):
    lat = []
    for _ in range(n):
        t0 = time.time()
        requests.post("http://localhost:8000/v1/completions", json={
            "model": model, "prompt": "User: Top 5 customers?\nAssistant:",
            "max_tokens": 64, "temperature": 0.0})
        lat.append(time.time() - t0)
    return statistics.mean(lat), statistics.quantiles(lat, n=20)[18]   # mean, p95

for m in ["Qwen/Qwen3-1.7B", "sql-analyst"]:
    mean, p95 = bench(m)
    print(f"{m:<22} mean {mean*1000:>7.1f} ms   p95 {p95*1000:>7.1f} ms")
```

**Expect 5–15% overhead.** Write down your number — "we measured 9% on our workload" is a much
better interview answer than "there's some overhead".

### Lab 4 — Heterogeneous batching, the actual claim

The critical test: send concurrent requests to **different** adapters and confirm they batch
together rather than serialising.

```python
import concurrent.futures as cf, time, requests

def call(model):
    t0 = time.time()
    requests.post("http://localhost:8000/v1/completions", json={
        "model": model, "prompt": "User: Average order value?\nAssistant:",
        "max_tokens": 64, "temperature": 0.0})
    return time.time() - t0

models = ["sql-analyst", "terse", "explainer"] * 8      # 24 mixed requests

t0 = time.time()
with cf.ThreadPoolExecutor(24) as ex:
    lat = list(ex.map(call, models))
mixed_wall = time.time() - t0

t0 = time.time()
with cf.ThreadPoolExecutor(24) as ex:
    lat_same = list(ex.map(call, ["sql-analyst"] * 24))
same_wall = time.time() - t0

print(f"mixed adapters : {mixed_wall:.2f}s wall, mean {sum(lat)/len(lat)*1000:.0f} ms")
print(f"same adapter   : {same_wall:.2f}s wall, mean {sum(lat_same)/len(lat_same)*1000:.0f} ms")
print(f"ratio          : {mixed_wall/same_wall:.2f}x")
```

**If the ratio is near 1.0, heterogeneous batching is working.** If it is near 3.0, requests are
serialising by adapter and something is misconfigured — usually `max_loras` set below the number of
distinct adapters in flight. Deliberately set `--max-loras 1` and re-run to see the failure mode;
recognising it from the ratio alone is the skill.

### Lab 5 — Cold start

```python
# start the server WITHOUT --lora-modules, then load dynamically
requests.post("http://localhost:8000/v1/load_lora_adapter",
              json={"lora_name": "explainer", "lora_path": "./adapters/explainer"})
```

Time: first request to a freshly loaded adapter, versus a warm one, versus one evicted and
reloaded. Then compute what an LRU cache of size `max_loras` costs at your adapter count — with 200
tenants and `max_loras=8`, most requests are cold. **That is a capacity-planning number**, and the
answer is usually to pin the hot tenants and accept cold starts on the long tail.

### Lab 6 — Per-tenant evaluation and rollback

```python
REGISTRY = {
    "acme":   {"adapter": "sql-analyst", "version": "v3", "base_sha": "abc123"},
    "globex": {"adapter": "terse",       "version": "v1", "base_sha": "abc123"},
}
```

Build a per-tenant golden set (Module 23) and a gate script that evaluates **each adapter
separately**. Then simulate a regression: deploy a deliberately bad `sql-analyst` v4, confirm the
gate fails for `acme` only, and roll that tenant back to v3 without touching `globex`.

**That rollback is the operational argument for this architecture** and it is exactly the kind of
thing the Expedia promotion-gate experience maps onto. Have the script in the repo.

---

## Failure modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| Memory grows per adapter as if full models | Adapters merged into the base | Keep unmerged |
| Mixed-adapter requests serialise | `max_loras` below concurrent distinct adapters | Raise it |
| Adapter rejected at load | Rank above `max_lora_rank` | Standardise rank; size at startup |
| High p99 on some tenants | Cold starts on non-resident adapters | Pin hot adapters; LRU |
| Adapter has no effect | Trained against a different base revision | Pin and record base sha per adapter |
| Wrong tenant's behaviour served | Registry mapping bug | Treat the mapping as security-critical; test it |
| All adapters break after an upgrade | Base model changed | Retrain and re-evaluate all; plan the migration |
| One tenant's regression invisible | Shared eval set | Per-tenant golden sets |
| Adapter helps behaviour, not knowledge | LoRA is a behavioural correction | Use RAG for facts |

---

## Interview

**"How would you serve fifty domain-specific fine-tunes?"**
One base model, fifty unmerged LoRA adapters, per-request adapter selection — vLLM's multi-LoRA
with Punica-style SGMV kernels. Fifty merged fine-tunes is fifty full model copies, about 700 GB
for a 7B; one base plus fifty adapters is about 16 GB, because a rank-16 adapter is roughly 40 MB.
The base matmul is shared across the whole batch and the adapter term is a gather plus two small
matmuls, so requests using different adapters still batch together in one pass — that's the part
that matters, because without heterogeneous batching you'd be batching per adapter and losing the
throughput that made serving economical. Overhead is typically 5–15%.

**"When would you merge instead?"**
When there's exactly one adapter for that deployment. Merging folds `BA` into `W` and gives
literally zero inference overhead, so for a single-tenant model it's strictly better. The moment
you need more than one, merging costs you the entire memory advantage — and you also can't roll
back a merged adapter without redeploying the whole model.

**"What's the operational risk in this architecture?"**
Two things, and both are outside the model. The tenant-to-adapter mapping is security-relevant —
serving tenant A's adapter to tenant B is an incident even though no data moved, because the
adapter encodes their domain behaviour. And a base model upgrade invalidates every adapter
simultaneously: they were all trained against a specific base revision, so moving the base means
retraining and re-evaluating all of them. That turns what looks like a config change into a
programme of work, and it needs planning rather than discovery. I'd pin the base sha in every
adapter's metadata and have the registry refuse a mismatch.

**"How do you know an adapter update is safe?"**
Per-tenant golden sets and a per-adapter gate. A shared eval can't tell you that tenant 12
regressed, and with fifty adapters that's exactly the failure you'll get. Each adapter is a ~40 MB
versioned artefact pinned to a base sha, so I can evaluate one, gate one, and roll one back without
touching the other forty-nine. That per-tenant rollback is the real operational advantage over full
fine-tunes, more than the memory saving.

**"When is a LoRA adapter the wrong answer for a domain?"**
When the domain needs new *knowledge* rather than different *behaviour*. LoRA is a low-rank
correction — it's very good at tone, format, tool-use patterns and domain conventions, and it's
weak at installing facts that weren't in pretraining. If the finance agent needs to know current
finance facts, that's retrieval, not an adapter. Fine-tuning the behaviour and retrieving the
knowledge is usually the right split, and conflating them is how you get a confident,
well-formatted, wrong answer.

---

## Checkpoint

1. Train three adapters over one base and confirm their size.
2. Serve all three concurrently and show memory did not triple.
3. Report your measured per-request overhead.
4. Report the mixed-vs-same adapter wall-clock ratio and interpret it.
5. Reproduce the serialisation failure by setting `max_loras=1`.
6. Report cold-start latency and the LRU capacity implication at your tenant count.
7. Demonstrate a per-tenant gate failure and a single-tenant rollback.

---

**Next:** [21 — RAG](21-rag.md) ·
**Back:** [19 — High-throughput serving](19-serving.md) · [Syllabus](../SYLLABUS.md)
