# Glossary

Every term the curriculum introduces. Alphabetical. `M##` points at the module that explains it
properly.

**How to use this:** before an interview, read it end to end. Any term where you cannot say *what
it is* **and** *where you'd meet it in production* is a gap — go to its module.

---

## A

**Activation checkpointing** (M17) — Discard intermediate activations in the forward pass and
recompute them during backward. Trades ~30% more compute for a large memory saving. *Used:* any
training run that would otherwise OOM; standard in FSDP configs.

**Active parameters** (M07) — In an MoE, the parameters actually used per token, versus total
parameters stored. `Holo3-35B-A3B` = 35B total, 3B active. *Used:* MoE cost reasoning — you pay
memory for total, compute for active.

**Adapter** (M12) — Small trainable module inserted into a frozen model. LoRA is the dominant
form. *Used:* serving many customer-specific variants over one base model (M20).

**Advantage** (M14, M15) — How much better an action/trajectory was than the baseline
expectation. In GRPO, computed relative to the group mean. *Used:* the term the policy gradient
is weighted by.

**Agentic RAG** (M21) — Retrieval exposed as a tool the model calls in a loop, deciding when and
what to search, rather than one retrieval before generation. *Used:* multi-hop questions where
the second query depends on the first result; costs latency and pays the `r^n` reliability tax,
so route to it only when needed.

**"Aha moment"** (M15a) — Self-correction appearing mid-trace ("wait, let me reconsider") in an
RL-trained reasoning model, never demonstrated in training data. *Used:* the widely cited
R1-Zero observation; measure its rate yourself before repeating the claim, since at small scale
it is often surface mimicry.

**ALiBi** (M04) — Positional method adding a linear distance penalty to attention scores instead
of using embeddings. Extrapolates to longer contexts than trained. *Used:* Bloom, MPT; largely
superseded by RoPE + scaling.

**Alignment tax** (M14) — Capability loss on benchmarks caused by alignment training. *Used:*
justifying why a helpful-only model may score higher on raw capability.

**Anisotropy** (M02) — The tendency of a causal LM's hidden states to occupy a narrow cone, so
*any* two of them have high cosine similarity. *Used:* the reason you must not use LM hidden
states as retrieval embeddings — measure against a random-vector baseline to see it.

**Attention head** (M03) — One independent Q/K/V projection set. Heads specialise (induction,
coreference, positional). *Used:* interpretability; `num_attention_heads` in every config.

**Attention sink** (M03) — First token(s) absorbing large attention mass regardless of content,
acting as a softmax pressure valve. *Used:* explains why naive sliding-window streaming degrades
and why pinning the first tokens fixes it.

**Autograd** (B0) — Automatic differentiation: recording operations on a graph and replaying the
chain rule backwards. *Used:* the whole of PyTorch. Writing a scalar one is the fastest way to
stop treating `.backward()` as magic.

**AWQ** (M18) — Activation-aware weight quantization; protects the weight channels that matter
most for activations. *Used:* 4-bit serving with small quality loss.

---

## B

**Base model** (M06) — Pretrained, not instruction-tuned. Completes text rather than answering.
*Used:* the starting point for your own SFT; using one with a chat template is a common beginner
bug.

**Beam search** (M08) — Keep the *k* highest-probability partial sequences. *Used:* translation
and constrained generation; rarely for chat, where it produces bland output.

**Best-of-n** (M08, M15) — Sample *n* completions, keep the best by some scorer. *Used:* cheap
quality boost at inference; also the baseline that GRPO must beat to justify its cost.

**bfloat16 (bf16)** (M00, M17) — 16-bit float with fp32's exponent range and reduced mantissa.
*Used:* default training/inference dtype; safer than fp16 because it rarely overflows.

**Bi-encoder** (M21) — Encodes query and document separately, so document vectors can be
precomputed and indexed. *Used:* the first stage of every RAG system; its weakness is that the
whole match must survive compression into one vector, which is why reranking exists.

**Bits-per-byte** (M01, M06) — Perplexity normalised to underlying text rather than tokens.
*Used:* the only fair cross-tokenizer quality comparison.

**BPE** (M01) — Byte-Pair Encoding. Build vocab by repeatedly merging the most frequent adjacent
symbol pair. *Used:* the tokenizer of essentially every modern LLM.

**Bradley-Terry model** (M14) — Statistical model turning pairwise preferences into scalar
scores. *Used:* the objective a reward model is trained with.

**Bubble** (M17) — Idle time in pipeline parallelism while stages wait for each other. *Used:*
why pipeline parallelism has worse efficiency than tensor parallelism at small scale.

---

## C

**Calibration** (M24) — Whether stated confidence matches actual accuracy. A calibrated model
that says 0.9 is right 90% of the time. *Used:* deciding when an agent should escalate to a
human.

**Catastrophic forgetting** (M11) — Fine-tuning on a narrow task destroys unrelated capability.
*Used:* the reason to mix general data into domain SFT, and to keep a broad regression suite.

**Causal mask** (M03) — Prevents position *i* attending to positions after it. *Used:* what
makes a decoder autoregressive; if broken, training looks fine and generation is garbage.

**Chain of thought (CoT)** (M09) — Producing intermediate reasoning tokens before the answer.
*Used:* converts a depth-limited computation into a sequential one; why it helps arithmetic.

**Chain-of-thought (CoT)** (M09, M15a) — Generating intermediate reasoning before the answer.
*Used:* prompted in M09; in M15a it *emerges* from an outcome-only reward, which is the
surprising part.

**Chat template** (M09) — Jinja template mapping a message list to the exact token string the
model was trained on. *Used:* every instruct model. Getting it subtly wrong degrades quality
with no error.

**Chinchilla-optimal** (M06) — The compute-optimal tokens-per-parameter ratio (~20:1). *Used:*
budget planning; also why modern small models are trained far past "optimal" — inference cost
dominates over training cost in deployment.

**Chunked prefill** (M19) — Split a long prompt's prefill into pieces interleaved with decode
steps. *Used:* stops one long prompt from stalling every other request's token stream.

**Cold-start SFT** (M15a) — A small, curated SFT run *before* RL, to fix format and readability
rather than to add capability. *Used:* stage 1 of the R1 pipeline; it exists because R1-Zero's
traces mixed languages, not because SFT was needed for reasoning.

**Confused deputy** (M25) — A privileged component acting on instructions from an unprivileged
party. *Used:* the precise description of what prompt injection does to an agent — the agent has
your credentials and follows the attacker's text.

**Constitutional AI / RLAIF** (M25) — The model critiques and revises its own outputs against
written principles, generating preference pairs without humans labelling harmful content.
*Used:* scaling safety alignment; also spares annotators exposure to the worst material.

**Constrained decoding** (M10) — Restrict the sampled token set to those a grammar permits.
*Used:* guaranteed-valid JSON; drives parse failure to zero.

**Contamination** (M13, M23) — Eval data present in training data. *Used:* the first thing to
suspect when a benchmark score is implausibly good.

**Context window** (M04) — Max tokens the model can attend over. *Used:*
`max_position_embeddings`; note the effective usable window is often shorter than the advertised
one.

**Contextual retrieval** (M21) — Prepending an LLM-written document-level context sentence to
each chunk *before embedding* (storing the original chunk unchanged). *Used:* the fix for chunks
like "this reduced latency by 40%" that are unretrievable because the referent is in another
chunk; large, reproducible recall gains.

**Continuous batching** (M19) — Add and remove sequences from a running batch per step instead
of waiting for the whole batch to finish. *Used:* vLLM's core throughput win.

**Credit assignment** (M15) — Attributing final outcome to specific earlier actions. *Used:* the
central difficulty of multi-step agent RL, and what SFT cannot do at all.

**Cross-attention** (M05a) — Queries from the decoder, keys and values from the encoder's
output. *Used:* the only channel between the two stacks of the original transformer, and the box
that does not exist in any decoder-only model. Its attention matrix is rectangular (T×S) and
never causally masked.

**Cross-encoder** (M21) — Encodes query and document jointly, so attention runs over both. Far
more accurate than a bi-encoder, cannot be precomputed. *Used:* reranking the top-50 down to
top-5 — the highest value-per-line component in a RAG pipeline.

**Cross-entropy loss** (M06) — `−log p(correct token)`. *Used:* the pretraining and SFT
objective.

---

## D

**Dark knowledge** (M16) — The information carried by a teacher's *non-argmax* probabilities:
which wrong answers it considered plausible, and how plausible. *Used:* the reason soft-target
distillation beats SFT on the teacher's sampled text; visible only at KD temperature > 1.

**DDP** (M17) — Distributed Data Parallel. Full model replica per GPU, gradients all-reduced.
*Used:* the simplest scaling; fails when the model no longer fits on one device.

**Decode** (M03, M19) — Generating tokens one at a time. Memory-bandwidth-bound. *Used:* the
phase continuous batching optimises.

**Deduplication** (M13) — Removing repeated data. Exact, near (MinHash/LSH), or semantic.
*Used:* duplicate training data causes memorisation and wastes compute.

**Degeneration** (B7) — The fluent, empty, looping text that greedy decoding produces. *Used:*
the reason production samplers never run at temperature 0 for open-ended generation; caused by
positive feedback, since conditioning on a phrase raises its own probability.

**Depth pruning** (M16) — Removing whole transformer blocks. *Used:* the one pruning form that
gives a real speedup for simple engineering; middle layers are far more removable than the first
or last, and it must always be followed by healing.

**Depth-scaled init** (B5) — Initialising layers that write into the residual stream with std
`0.02/√(2·n_layers)`. *Used:* GPT-2 onward; without it residual-stream norm grows with depth and
deep models destabilise.

**Distillation** (M16) — Training a small student to match a large teacher. *Used:* shrinking a
model for serving; producing draft models for speculative decoding.

**DoRA** (M12) — Weight-decomposed LoRA, splitting magnitude and direction. *Used:* a modest
quality gain over LoRA at similar cost.

**DPO** (M14) — Direct Preference Optimization. Optimises preferences directly with no separate
reward model or RL loop. *Used:* the default preference method; far simpler than PPO.

**Draft model** (M08) — Small fast model proposing tokens for a large model to verify. *Used:*
speculative decoding.

---

## E

**ECE** (M24) — Expected Calibration Error. Gap between confidence and accuracy. *Used:*
quantifying calibration.

**Effective batch size** (M11) — `per_device_batch × grad_accum × n_devices`. *Used:* the number
that actually determines training dynamics; LR must be tuned against it, not the per-device
batch.

**Embedding matrix** (M02) — `vocab_size × d_model` lookup table mapping token ids to vectors.
*Used:* often a large share of a small model's parameters.

**Emergence** (B8) — A capability appearing sharply with scale. *Used:* often an artifact of a
discontinuous metric — re-score with log-probability before concluding the model changed
discontinuously.

**Emergent abilities** (M06) — Capabilities appearing abruptly with scale. *Used:* contested —
some apparent emergence is an artefact of discontinuous metrics.

**Encoder-decoder (seq2seq)** (M05a, M07) — Two stacks joined by cross-attention. *Used:* T5,
Whisper, translation systems; the encoder runs once and every decoding step reuses its memory.

**Encoder-only** (M05a, M21) — Bidirectional stack, no generation. *Used:* BERT-family embedding
and reranking models — the retrieval half of every RAG system.

**Entropy collapse** (M15) — Policy becomes deterministic during RL, sampling diversity dies,
learning stops. *Used:* a primary GRPO failure signature; watch entropy, not just reward.

**Expert** (M07) — One FFN in an MoE layer. *Used:* `A3B`-style naming counts active experts.

**Expert collapse** (M07) — In an MoE, the router sending nearly all tokens to a few experts,
leaving the rest untrained. *Used:* the reason a load-balancing auxiliary loss exists;
reproducible in a toy MoE in an afternoon.

---

## F

**Fertility** (M01) — Average tokens per word. *Used:* multilingual cost and context fairness.

**FlashAttention** (M03) — Exact tiled attention that never materialises the n×n matrix. *Used:*
default in modern training and inference stacks. Not an approximation.

**Format collapse** (B9) — An instruction-tuned model emitting its own template tags as content.
*Used:* a symptom of special tokens not registered with the tokenizer, or of over-training on a
small SFT set.

**FP8** (M18) — 8-bit float. *Used:* H100-class training and inference; better dynamic range
than int8 at the same width.

**FSDP** (M17) — Fully Sharded Data Parallel. Shards parameters, gradients and optimizer state
across devices. *Used:* PyTorch-native equivalent of ZeRO-3.

---

## G

**GGUF** (M18) — Quantized model container used by llama.cpp. *Used:* local/edge CPU inference.

**Goodput** (M19) — Requests served *within their latency SLO* per second, as opposed to raw
throughput. *Used:* the metric that actually matters for a serving deployment — raising
`max_num_seqs` improves throughput and can destroy goodput.

**GPTQ** (M18) — One-shot post-training quantization using second-order error compensation.
*Used:* 4-bit weights with modest degradation.

**GQA** (M03) — Grouped-Query Attention. *n* query heads share *g* KV heads. *Used:* every
modern model; reduces KV cache by n/g and multiplies serving concurrency by the same factor.

**Grad norm** (B6) — The global L2 norm of the gradient across all parameters. *Used:* the
single best training-health signal — a norm climbing steadily across a run predicts divergence
long before the loss shows it.

**Gradient accumulation** (M11) — Sum gradients over several micro-batches before stepping.
*Used:* simulating a large batch on small memory.

**Groundedness / faithfulness** (M21) — Whether every claim in an answer is supported by the
retrieved context. *Used:* the metric that catches the dangerous RAG failure, where the model
answers from parametric memory while citing an unrelated chunk; needs an LLM judge, so report it
with its kappa.

**GRPO** (M15) — Group Relative Policy Optimization. Sample a group of completions per prompt,
use the group mean as the baseline, no critic. *Used:* the reasoning/agent post-training method
behind recent reasoning models.

**Guided generation** (M10) — See constrained decoding.

---

## H

**Hallucination** (M21, M25) — Confident output unsupported by evidence. *Used:* measured as a
rate against a grounded corpus, not as a binary property.

**Healing** (M16) — A short distillation run, using the unpruned model as teacher, to recover
quality after pruning. *Used:* always, after any pruning — it recovers most of the loss for a
few hundred steps.

**HNSW** (M21) — Graph index for approximate nearest-neighbour search. *Used:* most vector
databases.

**Hybrid retrieval** (M21) — Dense (embedding) and sparse (BM25) retrieval fused, usually with
RRF. *Used:* enterprise queries contain both paraphrase and exact identifiers — error codes,
SKUs, function names — and each method fails on the other's strength.

**HyDE** (M21) — Hypothetical Document Embeddings: generate a fake answer to the query, then
retrieve using *its* embedding. *Used:* query transformation when queries and documents are
stylistically mismatched.

---

## I–K

**IA³** (M12) — PEFT method rescaling activations with learned vectors. Even fewer parameters
than LoRA.

**Induction head** (M03) — Attention head implementing "if A was followed by B earlier, predict
B after A now". *Used:* a mechanistic basis for in-context learning.

**Inter-token latency (ITL)** (M19) — Time between generated tokens. *Used:* the perceived-speed
metric; distinct from TTFT.

**Irreducible loss (E)** (B8) — The additive constant in a scaling law: the entropy of the data
itself. *Used:* why loss curves flatten instead of reaching zero, and the term that dominates —
and is worst-constrained — in long extrapolations.

**IsoFLOP curve** (B8) — Loss plotted against model size at fixed compute. *Used:* the U-shaped
curve whose minimum defines the compute-optimal `(N, D)` split; the method behind Chinchilla.

**Jailbreak** (M25) — A prompt that circumvents a model's alignment training. *Used:* distinct
from prompt injection — a jailbreak is the *user* attacking the model's own policy; injection is
a *third party* attacking through data the model reads.

**Kaplan scaling (2020)** (B8) — The pre-Chinchilla scaling analysis favouring parameters over
tokens. *Used:* its optimum was shifted by holding the LR schedule fixed across run lengths — a
hyperparameter protocol artifact that set industry model sizes for two years.

**KL divergence / KL penalty** (M14, M15) — Distance from the reference policy, added to the
loss to stop the policy drifting. *Used:* the main defence against reward hacking and mode
collapse.

**KTO** (M14) — Kahneman-Tversky Optimization. Learns from single thumbs-up/down labels rather
than pairs. *Used:* when you have unpaired feedback, which is most production feedback.

**KV cache** (M03) — Cached keys and values for previous positions. *Used:* the dominant serving
memory cost; usually the real limit on concurrency.

**Language mixing** (M15a) — Traces switching between languages mid-generation. *Used:*
R1-Zero's characteristic defect and the reason R1 adds a language-consistency reward in stage 2.

---

## L

**Lethal trifecta** (M25) — Private data + untrusted content + external communication. Any two
are usually fine; all three is exfiltration waiting for a trigger. *Used:* the design frame for
agent security — you cannot remove untrusted content, so break egress or scope the data.

**Logit** (M02) — Pre-softmax score per vocabulary entry. *Used:* sampling, constrained
decoding, logit-level distillation.

**Logit lens** (M24) — Applying the LM head to intermediate layers to see the evolving
prediction. *Used:* cheap interpretability.

**LoRA** (M12) — Low-Rank Adaptation. Freeze `W`, learn `BA` with rank `r ≪ d`. *Used:* the
default fine-tuning method; trains ~0.1–1% of parameters.

**Loss masking** (B9) — Setting prompt-token labels to `-100` so cross-entropy skips them.
*Used:* completion-only SFT loss. The model still *attends* to the prompt; masking changes what
is learned, not what is seen. Padding must be masked too.

**Lost in the middle** (M21) — Models use the beginning and end of a long context far better
than the middle. *Used:* why you order retrieved chunks strongest-at-the-edges rather than by
rank; the effect worsens as context grows, which is why long context does not obsolete
retrieval.

**Lost-in-the-middle** (M04, M21) — Recall degrades for content in the middle of a long context.
*Used:* why reranking and placement matter more than raw context length.

---

## M

**McNemar's test** (M23) — A paired significance test over the *off-diagonal* counts: cases
model A got right and B wrong, versus the reverse. *Used:* comparing two models on the same eval
set — far more sensitive than comparing two independent accuracy figures.

**Membership inference** (M25) — Determining whether a specific record was in a model's training
set. *Used:* the privacy risk of fine-tuning on customer data; closely tied to memorisation,
which rises sharply with duplication.

**Merge (LoRA)** (M12) — Folding `BA` back into `W` to eliminate adapter overhead. *Used:*
single- adapter production serving; incompatible with multi-adapter serving (M20).

**MFU (Model FLOPs Utilisation)** (B6) — Achieved FLOPs ÷ hardware peak, with FLOPs ≈ `6N` per
token. *Used:* distinguishing a compute-bound run from an overhead-bound one; small models on
fast accelerators are overhead-bound and report low MFU legitimately.

**MinHash** (M13) — LSH technique for near-duplicate detection at corpus scale. *Used:*
pretraining data dedup.

**Min-p** (B7) — Truncated sampling keeping tokens with `prob ≥ min_p × max_prob`. *Used:*
adaptive like top-p but defined relative to the model's own confidence, so it degrades more
gracefully above temperature 1.

**Mixture of Experts (MoE)** (M07) — FFN replaced by many experts with a router selecting top-k
per token. *Used:* large total capacity at small active compute.

**MMLU / GSM8K / HumanEval / SWE-bench / τ-bench / BFCL** (M23) — Standard benchmarks for
knowledge, maths, code, real software tasks, agentic tool use, and function calling
respectively. *Used:* know what each measures and its main criticism.

**MQA** (M03) — Multi-Query Attention. All query heads share one KV head. *Used:* maximum cache
saving, measurable quality loss; mostly replaced by GQA.

**MRR** (M21) — Mean Reciprocal Rank. *Used:* retrieval quality.

---

## N–O

**NF4** (M12, M18) — 4-bit NormalFloat, information-theoretically suited to normally-distributed
weights. *Used:* QLoRA.

**Nucleus sampling (top-p)** (B7) — Keep the smallest token set whose cumulative probability
reaches `p`. *Used:* the default sampler; adapts to confidence, unlike a fixed top-k. Keep the
token that *crosses* `p` — the off-by-one empties the candidate set on confident steps.

**On-policy** (M15) — Training on data generated by the current policy. *Used:* the property
that makes GRPO improve decision-making where more SFT data does not.

**On-policy distillation / GKD** (M16) — The *student* generates the sequence and the *teacher*
scores the student's own trajectory. *Used:* fixes the distribution mismatch of standard KD —
the same exposure-bias argument that motivates RL over SFT, and the gap grows with trajectory
length.

**ORPO** (M14) — Odds-Ratio Preference Optimization. Combines SFT and preference learning in one
stage without a reference model. *Used:* simpler pipelines.

**Outcome supervision** (B10) — Training on the final answer only. *Used:* contrast with process
supervision; cheaper to label, weaker signal about *how* an answer was reached.

**Outlier feature** (M18) — Activation dimension with extreme magnitude that dominates
quantization error. *Used:* the problem SmoothQuant and AWQ address.

**Over-refusal / false refusal** (M25) — Declining a benign request. *Used:* the main cost of
safety training and the failure that makes enterprise users abandon a model; must be measured on
a benign-but-adjacent set alongside harmful compliance, or either number is trivially gamed.

**Post-norm** (M05a) — `LayerNorm(x + Sublayer(x))` — normalisation *after* the residual add, as
drawn in Figure 1 of the original paper. *Used:* nowhere modern; it interrupts the residual
highway at every layer and needs warmup to train at depth.

---

## P

**PagedAttention** (M19) — KV cache stored in fixed-size non-contiguous blocks, like OS paging.
*Used:* vLLM; removes fragmentation and enables prefix sharing.

**pass@k** (M23) — Probability at least one of *k* samples is correct. *Used:* code and agent
benchmarks. Report *k*; pass@1 and pass@10 are different claims.

**PEFT** (M12) — Parameter-Efficient Fine-Tuning. Umbrella for LoRA, adapters, prefix tuning,
IA³.

**Perplexity** (M06) — `exp(cross-entropy)`. *Used:* pretraining health. Not comparable across
tokenizers.

**PPO** (M14) — Proximal Policy Optimization. Actor-critic RL with clipped updates. *Used:*
classic RLHF; heavy — four models in memory — which is why DPO and GRPO displaced it.

**Prefill** (M03, M19) — Processing the prompt. Compute-bound. *Used:* determines TTFT.

**Prefix caching** (M19) — Reusing KV blocks for a shared prompt prefix across requests. *Used:*
large saving when every request carries the same long system prompt.

**Pre-norm** (M05a, M05) — `x + Sublayer(Norm(x))` — normalisation inside the branch, leaving a
clean additive residual path. *Used:* GPT-2 onward and every model you will fine-tune; the
reason deep transformers train without the original's warmup schedule.

**PRM / ORM** (M15) — Process vs Outcome Reward Model: grading each step versus only the final
result. *Used:* process rewards give denser signal but cost far more to label.

**Process supervision** (B10) — Training on the reasoning steps, not just the final answer.
*Used:* CoT SFT and process reward models; the reason synthetic generators emit a correct
`<thought>` alongside the answer.

**Prompt injection** (M09, M25) — Untrusted content carrying instructions the model follows.
*Used:* the top agent security risk; indirect injection arrives via tool results, not user
input.

---

## Q–R

**QLoRA** (M12) — LoRA over a 4-bit quantized frozen base. *Used:* fine-tuning large models on
one consumer GPU.

**R1-Zero** (M15a) — A base model trained with GRPO and **no SFT at all**. *Used:* the
demonstration that SFT is not a prerequisite for reasoning; response length grew over training
with no length term in the reward, and self-correction emerged. Not shippable — its traces mixed
languages.

**RAG** (M21) — Retrieval-Augmented Generation. *Used:* grounding output in a corpus.

**Recall@k** (M21) — Fraction of queries whose gold document appears in the top *k* retrieved.
*Used:* the first thing to measure when a RAG system gives wrong answers — it needs no LLM, and
it tells you whether your ceiling is retrieval or generation.

**Red teaming** (M25) — Adversarially probing a model or system for failures before an attacker
does. *Used:* pre-release safety work; for agents it means injection attempts through every
content channel the agent reads.

**Repetition penalty** (B7) — Dividing the logits of already-generated tokens. *Used:* breaks
loops at low values; at high values the model stops reusing necessary words like articles and
names. A blunt instrument — better sampling usually beats a stronger penalty.

**Reranker** (M21) — Cross-encoder rescoring retrieved candidates. *Used:* the highest-ROI
single addition to a naive RAG pipeline.

**Residual stream** (M02) — The running hidden-state vector each block reads from and writes to.
*Used:* the mental model behind most interpretability work.

**Response-length growth** (M15a) — Output length rising over RL steps without being rewarded
for it. *Used:* the signature R1-Zero plot; the model discovers that thinking longer raises
accuracy. Reproduce it yourself rather than citing it.

**Reverse KL** (M16) — `KL(student ‖ teacher)` — mode-seeking: the student picks one mode and
commits. Contrast forward KL, which is mode-covering and produces vague averaged output on a
multi-modal teacher. *Used:* instruction-following distillation, where many answers are valid
and you want one good one.

**Reward hacking** (M15) — Maximising the measured reward without doing the task. *Used:* the
default outcome of a naive reward; signature is rising reward with falling held-out eval.

**Reward model** (M14) — Model trained on preferences to score outputs. *Used:* classic RLHF;
unnecessary in DPO and in RLVR.

**RLVR** (M15) — RL with Verifiable Rewards. Rewards computed by a program, not a judge. *Used:*
maths, code, tool use — anywhere correctness is checkable.

**RMSNorm** (M05) — Normalisation using root-mean-square only, no mean subtraction. *Used:*
cheaper than LayerNorm; standard in Llama/Qwen.

**RoPE** (M04) — Rotary Position Embedding. Rotates Q and K by a position-dependent angle,
making attention naturally relative. *Used:* essentially universal; `rope_theta` is the context-
extension knob.

**RRF (reciprocal rank fusion)** (M21) — `Σ 1/(k + rank)` over several ranked lists, `k = 60`.
*Used:* fusing dense and sparse retrieval without needing calibrated scores between them.

---

## S

**Sandboxing** (M25) — Executing model-generated code in an isolated environment: container, no
network, read-only filesystem, resource and time caps, ephemeral. *Used:* any agent that runs
code; anything blocked only by your Python rather than by the container is a bug waiting for an
edge case.

**Scaling laws** (M06) — Power-law relations between loss and compute/data/parameters. *Used:*
predicting whether more scale will help before spending the money.

**Scratchpad** (B10) — The intermediate reasoning tokens a model writes before answering.
*Used:* converts a transformer's fixed per-token compute into variable-length compute; the
working memory lives in the KV cache. Scramble it at inference to test whether the model reads
it.

**Self-consistency** (M08) — Sample several CoT paths, take the majority answer. *Used:*
reliable accuracy gain on reasoning at *n*× cost.

**SFT** (M11) — Supervised Fine-Tuning on demonstrations. *Used:* stage one of post-training;
teaches imitation, not decision quality.

**Shifted right** (M05a) — The decoder input is the target sequence offset by one, so position
*i* predicts token *i*. *Used:* the bottom-right label in Figure 1; it is teacher forcing drawn
as a wire, and the same label-shift you implement in M11.

**SimPO** (M14) — Reference-free preference optimization with length normalisation. *Used:*
removes the reference model and attacks length bias directly.

**Sleeper agent / backdoor** (M25) — Behaviour installed via data poisoning that triggers only
on a specific input pattern, and can survive safety training. *Used:* the argument for data
provenance and trusted sources rather than post-hoc filtering.

**Sliding-window attention** (M03) — Attend only to the last *w* tokens. *Used:* linear-cost
long context, at the price of long-range links.

**Speculative decoding** (M08) — Draft model proposes *k* tokens; the target verifies them in
one pass. *Used:* 2–3× decode speedup with identical output distribution.

**SwiGLU** (M05) — Gated activation used in modern FFNs. *Used:* Llama, Qwen; why the FFN has
three weight matrices rather than two.

**Sycophancy** (M25) — Agreeing with the user against the evidence. *Used:* a direct consequence
of optimising for human approval; measure it as a flip rate — correct answer, confident wrong
pushback, does the model fold — and treat it as a release metric.

---

## T–Z

**Teacher forcing** (M06) — Feeding ground-truth tokens during training rather than the model's
own outputs. *Used:* what makes parallel training possible; source of train/inference mismatch.

**Temperature** (B7) — Dividing logits by `T` before softmax. *Used:* `T<1` sharpens, `T>1`
flattens, `T→0` is greedy. Exploits softmax's scale sensitivity — the same property that makes
the max-subtraction trick safe, since softmax is shift- but not scale-invariant.

**Tensor parallelism** (M17) — Splitting individual matrices across devices. *Used:* when one
layer does not fit; needs fast interconnect.

**Token healing** (M01) — Repairing a prompt that ends mid-token. *Used:* completion APIs and
fill-in-the-middle.

**Top-k sampling** (B7) — Keep the `k` highest-probability tokens, renormalise, sample. *Used:*
simplest truncation; its weakness is that the right `k` depends on how peaked the distribution
is at that step.

**Trainable-parameter ratio** (M12) — Fraction of parameters actually updated. *Used:* LoRA is
typically 0.1–1%.

**Trajectory** (M13, M15, M22) — One full agent episode: states, actions, observations, outcome.
*Used:* the unit of agent training data.

**TTFT** (M19) — Time To First Token. *Used:* the latency users perceive first; dominated by
prefill.

**Unified memory** (B6) — Apple Silicon's shared CPU/GPU memory. *Used:* no host↔device copy, so
`.to("mps")` is far cheaper than the CUDA equivalent; MPS dispatch is async, so
`torch.mps.synchronize()` is required before any timing.

**Unigram** (M01) — Probabilistic tokenizer algorithm that prunes a large vocab down. *Used:*
SentencePiece; supports subword regularisation.

**Verifier** (B10) — A deterministic program mapping `(completion, gold) → reward`. *Used:* the
component that makes RLVR possible without human labels. It *is* your objective function, and
the policy will find every gap in it.

**Warmup** (B6) — Linear LR ramp over the first few hundred steps. *Used:* Adam's second-moment
estimate starts near zero, making early effective steps enormous. Skipping it usually does not
diverge — it settles on a permanently worse plateau, which is what makes it dangerous.

**Weight tying** (B5) — Sharing the embedding matrix with the output head. *Used:* saves `vocab
× d_model` parameters — 23% of a 3.4M model, noise at 7B, which is why large models often untie.
Verify with `data_ptr()` equality, not by comparing values.

**YaRN** (M04) — Context-extension method combining NTK-aware interpolation with attention
temperature correction. *Used:* extending a model past its trained context without full
retraining.

**ZeRO** (M17) — Sharding optimizer state (1), gradients (2), and parameters (3) across devices.
*Used:* DeepSpeed; conceptually identical to FSDP.

**Zero-variance group** (M15) — A GRPO group where all samples score identically, so every
advantage is zero and no gradient is produced. *Used:* the main reason a first GRPO run
flatlines; fixed by matching task difficulty to current policy strength.

---

## Self-test

Cover the right column. For each term say (a) what it is, (b) where you would meet it in a real
system, (c) one way it fails. Anything you cannot do all three for, go read its module.
