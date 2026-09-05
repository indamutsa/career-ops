# LLM: first principles to advanced

A complete curriculum for the Senior AI Research Engineer interview — and, more usefully, for
actually being able to do the job.

**Nothing is assumed.** This curriculum has two tracks, and you should do them together.

**Track A — [the Build Track](modules/build/README.md).** You build a complete language model from
an empty file: probability and backpropagation derived from scratch, your own BPE tokenizer, your
own attention, a ~3.4M-parameter transformer you pretrain, instruction-tune, and then teach to
reason with GRPO against a verifier you wrote. It starts at "what is a probability distribution over
sequences" and writes an autograd engine before it writes a model — then follows the same three
stages every frontier model does: pretrain, instruction-tune with chain of thought, RL from a
verifiable reward.

**Track B — the reference modules below.** Depth on every concept, with the production context,
failure modes, and interview answers. They assume you can read PyTorch; the Build Track is what
makes that true.

The point of Track A is that when you quantize a model you built, you know what every one of its
3.4 million numbers is. There is no "it just works" left anywhere — which is the standard this
whole curriculum is held to.

**Format of every module:**

| Section | What it gives you |
|---------|-------------------|
| **Terms** | Every piece of vocabulary the module introduces, defined precisely |
| **Concepts** | Why the thing exists, what it replaced, what it costs |
| **Where it's used** | Which real systems use it, and where you'd meet it in production |
| **Lab** | Runnable code. Small models, local hardware, verifiable output |
| **Interview** | What gets asked, and the answer that shows hands-on experience |
| **Failure modes** | How it breaks, and the signature that identifies it |

Terms are also collected in [`GLOSSARY.md`](GLOSSARY.md) — one line each, searchable, with a pointer
back to the module that explains it.

---

## Hardware and ground rules

Apple M3 Pro, 36 GB. Everything here runs locally except where flagged `[GPU]`.

- **Default models:** `Qwen/Qwen3-0.6B` for plumbing, `Qwen/Qwen3-1.7B` for anything where quality
  matters. `Qwen/Qwen3-4B` is the ceiling for comfortable local training.
- **Docker for services only.** No GPU passthrough on macOS — training runs native on MPS.
- **Prove the pipeline on the smallest model that can run it.** Scale last.
- **Every lab produces a number.** If you cannot state what changed, the lab is not finished.

---

## Part 0 — Setup

**[00 — Environment and tooling](modules/00-setup.md)**
Python env, `transformers`, `torch` on MPS, model downloads, HF cache layout, memory accounting.
Terms: *model card, safetensors, checkpoint shard, revision pinning, device map, dtype.*

---

## Part I — How a language model actually works

**[01 — Tokenization](modules/01-tokenization.md)**
The layer everyone skips and then gets bitten by.
Terms: *token, vocabulary, BPE, byte-level BPE, WordPiece, SentencePiece, Unigram, merge rules,
pre-tokenizer, special token, BOS/EOS/PAD, chat template tokens, token healing, fertility,
out-of-vocabulary, tokenizer mismatch.*
Lab: measure fertility across languages; find where tokenization breaks arithmetic; corrupt a
generation with a tokenizer mismatch on purpose.

**[02 — Embeddings and the residual stream](modules/02-embeddings.md)**
Terms: *embedding matrix, tied embeddings, unembedding / LM head, hidden state, residual stream,
d_model, logits, weight tying, embedding scaling.*
Lab: extract hidden states layer by layer; watch the residual stream evolve; cosine-similarity
probes.

**[03 — Attention](modules/03-attention.md)**
The core mechanism, derived rather than quoted.
Terms: *query/key/value, scaled dot-product attention, softmax temperature in attention, causal
mask, attention head, multi-head attention (MHA), multi-query attention (MQA), grouped-query
attention (GQA), attention sink, KV cache, prefill, decode, FlashAttention, sliding-window
attention, quadratic complexity.*
Lab: implement attention from scratch, verify against PyTorch; visualise attention maps; measure
KV-cache growth and the memory wall.

**[04 — Positional information](modules/04-positional.md)**
Terms: *absolute positional embedding, learned vs sinusoidal, relative position, RoPE (rotary),
rotation frequency / theta base, ALiBi, NoPE, context length, context extension, position
interpolation, NTK-aware scaling, YaRN, long-context degradation, lost-in-the-middle.*
Lab: implement RoPE; extend a model's context by rescaling theta and measure where it breaks.

**[05 — The transformer block, assembled](modules/05-transformer-block.md)**
Terms: *feed-forward network (FFN/MLP), expansion ratio, activation function, GELU, SwiGLU,
layer normalization, RMSNorm, pre-norm vs post-norm, residual connection, depth vs width,
parameter count accounting.*
Lab: build a complete block; count parameters analytically and verify against the real model.

**[05a — The encoder–decoder transformer, and reading Figure 1](modules/05a-encoder-decoder.md)**
Terms: *encoder stack, decoder stack, cross-attention, memory/encoder output, masked
self-attention, shifted right, teacher forcing, post-norm vs pre-norm, warmup, Add & Norm,
encoder-only, seq2seq.*
Lab: implement cross-attention and show the rectangular T×S attention matrix; ablate it away and
watch the decoder ignore its input; reproduce post-norm divergence at depth without warmup.

**[06 — Pretraining](modules/06-pretraining.md)**
Terms: *causal language modelling, next-token prediction, cross-entropy loss, perplexity,
teacher forcing, context window packing, document masking, corpus curation, deduplication,
data mixture, epochs vs tokens, scaling laws, Chinchilla-optimal, compute budget, emergent
abilities, base model vs instruct model.*
Lab: pretrain a tiny model on a small corpus; watch loss and perplexity; verify the scaling
relationship holds in miniature.

**[07 — Architecture variants](modules/07-architectures.md)**
Terms: *dense model, mixture of experts (MoE), router, expert, top-k routing, active vs total
parameters, load balancing loss, expert capacity, A3B-style naming, encoder-decoder vs
decoder-only, state-space models (Mamba), hybrid architectures.*
Lab: inspect a real MoE checkpoint; trace routing decisions; compute active-parameter cost.

---

## Part II — Getting output out of it

**[08 — Decoding and sampling](modules/08-decoding.md)**
Terms: *greedy decoding, beam search, temperature, top-k, top-p / nucleus, min-p, typical
sampling, repetition penalty, frequency/presence penalty, logit bias, constrained decoding,
guided generation, speculative decoding, draft model, acceptance rate, self-consistency,
best-of-n, entropy.*
Lab: sweep temperature and measure output diversity and pass rate; implement speculative decoding
and measure the speedup.

**[09 — Chat templates and prompting](modules/09-prompting.md)**
Terms: *chat template, Jinja template, role (system/user/assistant/tool), turn, special-token
delimiters, generation prompt, few-shot prompting, chain of thought, zero-shot CoT, reasoning
traces, thinking mode, prompt injection, context engineering, prompt caching.*
Lab: hand-render a chat template; break a model by getting the template subtly wrong.

**[10 — Structured output and tool calling](modules/10-structured-output.md)**
Terms: *function calling, tool schema, JSON mode, grammar-constrained decoding, GBNF, outlines /
XGrammar, parse failure rate, tool choice, parallel tool calls, tool result message.*
Lab: compare a free-text JSON protocol against grammar-constrained decoding; measure parse
failure rate on a 0.6B model.

---

## Part III — Adapting the model

**[11 — Supervised fine-tuning](modules/11-sft.md)**
Terms: *SFT, instruction tuning, completion-only loss, prompt masking, packing, sample packing
with attention isolation, learning rate schedule, warmup, cosine decay, gradient accumulation,
effective batch size, catastrophic forgetting, overfitting signature, eval loss divergence.*
Lab: SFT a 0.6B model; deliberately induce catastrophic forgetting; fix it.

**[12 — Parameter-efficient fine-tuning](modules/12-peft.md)**
Terms: *PEFT, LoRA, rank r, alpha, scaling factor, target modules, adapter, merge, QLoRA,
NF4, double quantization, paged optimizer, DoRA, rsLoRA, LoRA+, prefix tuning, prompt tuning,
IA³, trainable-parameter ratio.*
Lab: LoRA at r=4/16/64, measure quality against memory; merge an adapter and verify equivalence.

**[13 — Fine-tuning dataset construction](modules/13-datasets.md)**
Terms: *trajectory, rollout, rejection sampling, distillation data, synthetic data, self-instruct,
deduplication (exact/near/MinHash/semantic), decontamination, quality filtering, difficulty
stratification, curriculum, train/eval split, leakage, benchmark contamination, data mixture ratio,
annotation guideline, inter-annotator agreement.*
Lab: build a dataset from your own agent rollouts; measure contamination against your eval set.

**[14 — Preference optimization](modules/14-preference.md)**
Terms: *RLHF, reward model, Bradley-Terry, preference pair, chosen/rejected, PPO, policy, value
head, critic, advantage, GAE, clipping, KL penalty, reference model, DPO, implicit reward, beta,
ORPO, KTO, SimPO, IPO, length bias, verbosity bias, alignment tax, reward model overoptimization.*
Lab: DPO on preference pairs; watch the implicit reward margin; induce and diagnose length bias.

**[15 — RL with verifiable rewards](modules/15-rlvr-grpo.md)**
Terms: *RLVR, verifiable reward, GRPO, group, group-relative advantage, critic-free, rollout
budget, reward shaping, dense vs sparse reward, credit assignment, on-policy vs off-policy,
policy collapse, entropy collapse, reward hacking, KL divergence budget, curriculum matching,
zero-variance group, RLOO, REINFORCE, process vs outcome reward, PRM/ORM.*
Lab: GRPO on the agent environment from `learn-agent-posttrain`; produce a reward-hacking run
deliberately and identify it from the curves.

**[15a — DeepSeek-R1 and the RL turn](modules/15a-deepseek-r1.md)**
Terms: *RLHF vs RLVR, R1-Zero, cold-start SFT, emergent chain-of-thought, response-length growth,
"aha moment", language mixing, language-consistency reward, rejection sampling, reasoning
distillation, test-time compute.*
Lab: reproduce R1-Zero in miniature from a base model and plot response length rising with no
length term in the reward; run the full four-stage pipeline on the SQL agent with per-stage evals.

**[16 — Distillation](modules/16-distillation.md)**
Terms: *knowledge distillation, teacher/student, soft targets, temperature in distillation,
logit matching, sequence-level KD, on-policy distillation, reverse KL, mode collapse vs mode
covering, speculative-decoding draft training.*
Lab: distil a 1.7B teacher into a 0.6B student; measure the quality/latency trade.

---

## Part IV — Systems

**[17 — Distributed training](modules/17-distributed.md)** `[GPU]`
Terms: *data parallelism, DDP, gradient all-reduce, ZeRO stages 1/2/3, FSDP, sharding strategy,
parameter/gradient/optimizer-state sharding, tensor parallelism, pipeline parallelism, bubble,
sequence parallelism, context parallelism, 3D parallelism, activation checkpointing /
recomputation, gradient accumulation, mixed precision, bf16 vs fp16, loss scaling, communication
overlap, NCCL, memory accounting.*
Lab: FSDP on rented GPUs; compute the memory budget analytically first, then verify.

**[18 — Quantization](modules/18-quantization.md)**
Terms: *quantization, post-training quantization (PTQ), quantization-aware training (QAT),
INT8/INT4, FP8, calibration set, per-tensor vs per-channel vs group-wise, outlier feature,
GPTQ, AWQ, SmoothQuant, bitsandbytes, GGUF, k-quants, perplexity degradation, KV-cache
quantization.*
Lab: quantize to 4-bit three ways; measure perplexity delta, memory, and tokens/sec.

**[19 — Inference and serving](modules/19-serving.md)**
Terms: *prefill, decode, time-to-first-token (TTFT), inter-token latency (ITL), throughput vs
latency, static vs dynamic vs continuous batching, PagedAttention, KV-cache block, prefix caching,
chunked prefill, tensor parallel serving, request scheduling, preemption, vLLM, SGLang, TensorRT-LLM,
goodput, SLO.*
Lab: vLLM locally; measure TTFT and throughput against batch size; find the knee.

**[20 — Multi-adapter serving](modules/20-multi-lora.md)**
Terms: *multi-LoRA, adapter hot-swap, S-LoRA, punica kernels, adapter routing, per-request
adapter selection, base-model sharing, cold start, adapter registry.*
Lab: serve several LoRA adapters over one base model; measure the cost of a swap.

---

## Part V — Applied and evaluated

**[21 — Retrieval-augmented generation](modules/21-rag.md)**
You already ship this. Included for completeness and for the parts that are usually missing.
Terms: *chunking, overlap, embedding model, bi-encoder vs cross-encoder, reranker, hybrid search,
BM25, dense retrieval, HNSW, IVF, recall@k, MRR, NDCG, context stuffing, lost-in-the-middle,
grounding, attribution, hallucination rate, retrieval ceiling.*
Lab: measure the retrieval ceiling — the score an oracle retriever would get. Everything above it
is unreachable by generation improvements.

**[22 — Agents and tool use](modules/22-agents.md)**
Terms: *agent, trajectory, ReAct, planning, reflection, scratchpad, state, memory, termination
criterion, step budget, tool selection, error recovery, cascading failure, computer use,
multi-agent, handoff, MCP, sandbox.*
Lab: extend the SQL agent with planning and reflection; measure whether either actually helps.

**[23 — Evaluation](modules/23-evaluation.md)**
Terms: *benchmark, MMLU, GSM8K, HumanEval, SWE-bench, τ-bench / tau2-bench, BFCL, WebArena,
OSWorld, pass@k, majority voting, LLM-as-judge, judge bias (position/length/self-preference),
rubric, inter-rater agreement, contamination, held-out set, statistical significance,
bootstrapping, confidence interval, eval harness, regression suite.*
Lab: build an eval harness with bootstrapped confidence intervals; demonstrate that a 2-point
benchmark difference is noise.

**[24 — Failure analysis and interpretability](modules/24-failure-analysis.md)**
Terms: *error taxonomy, trace analysis, failure clustering, logit lens, logprob analysis,
calibration, ECE, uncertainty, entropy as a confidence signal, attention attribution, activation
patching, probing classifier, sparse autoencoder, feature, circuit.*
Lab: cluster your agent's failures into a taxonomy; find the step where a trajectory goes wrong
using per-token logprobs.

**[25 — Safety, alignment and deployment risk](modules/25-safety.md)**
Terms: *alignment, helpfulness/harmlessness trade-off, refusal, jailbreak, prompt injection,
indirect prompt injection, data exfiltration, guardrail, input/output filtering, red teaming,
constitutional AI, RLAIF, model card, eval for harm, PII leakage, memorization, extraction attack.*
Lab: prompt-inject your own SQL agent through a tool result; then defend it.

---

## Interview coverage map

Everything in the Isaac role's brief, mapped to where it's covered.

| Their words | Modules |
|-------------|---------|
| "strong ML fundamentals" | **B0–B8**, 01–07, 05a |
| "hands-on LLM fine-tuning" | **B9**, 11, 12, 13 |
| "reasoning / RL post-training" | **B10**, 14, 15, 15a |
| "fine-tuning domain-specific agents" | 12, 15, 20, 22 |
| "materially better at complex decision-making" | 14, 15, **15a** |
| "understanding why models fail" | 23, 24 |
| "turning research into reliable production systems" | 17, 18, 19, 20 |
| "their own open-source language models" | 06, 07, **15a**, 16 |
| "enterprise scale" | 17, 19, 20 |

---

## Order to actually do this in

Twenty-eight reference modules plus an eleven-lesson Build Track. Do not go 00 → 25
linearly. Do this:

0. **[The Build Track, B0–B10](modules/build/README.md)** — in parallel with everything below. It is
   the spine: by B10 you own a base model, an instruct model and a reasoning model you trained
   yourself, and every lab after that gets a better subject than a borrowed Qwen3.

1. **00, 01, 03, 08** — enough to reason about any model. One day.
2. **09, 10, 22** — the agent layer. Feeds `learn-agent-posttrain` Lab 01 directly.
3. **11, 12, 13** — fine-tuning you partly know; 13 is new.
4. **14, 15, 15a** — the centre of gravity for this role. Slow down here. 15a is where "how did
   DeepSeek change post-training?" gets answered precisely rather than from headlines.
5. **23, 24** — the differentiator. Nobody prepares for these and they are half the job.
6. **17, 18, 19, 20** — systems. Fastest for you; it's your existing domain wearing new names.
7. **02, 04, 05, 05a, 06, 07, 16, 21, 25** — fill in as time allows. Do **05a** before any
   whiteboard round: Figure 1 of *Attention Is All You Need* is the most-shown image in the field,
   and two of its boxes are absent from every model you will actually touch.

Steps 1–5 are the interview. Step 6 is where you will out-argue every pure researcher in the room.
