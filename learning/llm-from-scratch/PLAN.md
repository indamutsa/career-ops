# Plan — 25 days to 30 September 2026

Interview: **Wednesday 30 September 2026.** Senior AI Research Engineer (LLM fine-tuning + agentic
systems, their own open-source models, enterprise scale).

Today is Saturday 5 September. Four full weekends before the date. That is enough for the **full
Build Track with no triage** — you do not have to cut anything.

**Budget assumed:** ~2.5 h on weekdays, ~6 h on weekend days ≈ **90 h total.** The Build Track needs
~38 h. The rest goes to reference modules, the agent artifact, and drilling. If your real budget is
half that, tell me and I will re-cut this rather than let you discover the gap on the 25th.

---

## The four sprints

| Sprint | Dates | Deliverable | Gate |
|--------|-------|-------------|------|
| **1** | Sat 5 – Sun 13 Sep | B0 → B5 | A 3,443,136-param transformer, hand-derived count matching code |
| **2** | Mon 14 – Sun 20 Sep | B6 → B8 | **A model that writes English**, + a fitted scaling law with a tested prediction |
| **3** | Mon 21 – Sun 27 Sep | B9 → B10 + agent artifact | Instruct + reasoning checkpoints; RL length-growth plot |
| **4** | Mon 28 – Wed 30 Sep | **No new material** | Drill, glossary, write-up, CV |

The gates are hard. If a gate slips, the next sprint absorbs it — do not start B6 without a model
whose parameter count you can derive on a whiteboard.

---

## Sprint 1 — Sat 5 to Sun 13 September

**Goal: a transformer, assembled and understood. No training yet.**

| Day | Work | Done when |
|-----|------|-----------|
| Sat 5 | Setup + **B0** foundations | Scalar autograd matches `torch.autograd` to 1e-6 |
| Sun 6 | **B0** finish + **B1** first model | Bigram trained; loss within ~0.02 of the counted optimum |
| Mon 7 – Wed 9 | **B2** tokenizer | 4096-vocab BPE; roundtrip passes on emoji and Japanese |
| Thu 10 – Fri 11 | **B3** data | Memmap verified; the leaky split proven to give a *lower* val loss |
| Sat 12 | **B4** attention | Causality perturbation test passes; leaky-mask breakage reproduced |
| Sun 13 | **B5** transformer | **Gate:** hand count = `count_params()`, exactly |

**Sunday 13 is the parameter-derivation gate.** Do it on paper before you run the code. If they
disagree, find the discrepancy — it is always a bias, a norm, or double-counting the tied head, and
finding it is the point.

Also this week, in background time (it is mostly waiting on a run): the **`learn-agent-posttrain`
Lab 01 baseline** you still owe four numbers for — per-tier pass rate, parse-failure rate, mean
steps/episode, give-up behaviour. Those numbers are the baseline everything in Sprint 3 improves on.

---

## Sprint 2 — Mon 14 to Sun 20 September

**Goal: a model that writes English, and a scaling law you tested.**

| Day | Work | Done when |
|-----|------|-----------|
| Mon 14 – Tue 15 | **B6** training loop | 20-step smoke test: loss 8.32±0.3, falling, grad norm ~1 |
| Wed 16 | **B6** `base` run (~45 min) + warmup ablation | Val ≤ 1.8; no-warmup run visibly worse |
| Thu 17 | **B6** LR sweep + resume test | U-curve plotted; resume bit-exact to 1e-3 |
| Fri 18 | **B6** ladder: `tiny`, `wide` | Three checkpoints + three loss curves saved |
| Sat 19 | **B7** sampling + KV cache | Speedup grows with length; cached == uncached at `T=0` |
| Sun 20 | **B8** scaling laws | **Gate:** prediction written down *before* training model #4 |

**Record B6 Lab 8 verbatim** — what the base model does with *"Can you solve this math problem?"*
B9 puts its output next to yours and that contrast is one of your strongest interview moments.

**End of Sprint 2 → first CV pass.** You now have a real artifact. `/career-ops train` places it in
Personal Projects. Do not wait until the 29th.

---

## Sprint 3 — Mon 21 to Sun 27 September

**Goal: post-training, on your model and on the agent environment.**

| Day | Work | Done when |
|-----|------|-----------|
| Mon 21 – Tue 22 | **B9** instruction tuning | Base vs instruct, same prompts, recorded side by side |
| Wed 23 | **B9** ablations | Masking ablation + LR-destruction run, both measured |
| Thu 24 – Fri 25 | **B10** CoT + synthetic logic data | Leakage baselines run **before** training |
| Sat 26 | **B10** GRPO | **Gate:** length-growth plot, with the reward function shown to have no length term |
| Sun 27 | Same pipeline → agent environment | SFT → GRPO on the SQL agent env, against the Sprint-1 baseline |

Sunday 27 is where the two threads join: B9/B10 teach SFT → GRPO with a programmatic verifier, and
the agent project *is* that applied to a real environment. Same machinery, second domain — which is
exactly the "fine-tuning domain-specific agents" line in the JD.

**Reference modules, read alongside** (each build lesson names its companion at the top). Priority
order given what they asked for:

1. `15a-deepseek-r1.md` + `15-rlvr-grpo.md` — they train their own models with RL
2. `22-agents.md` + `20-multi-lora.md` — agentic systems, serving many adapters
3. `06-pretraining.md` + `16-distillation.md` — their own open-source LLMs
4. `17-distributed.md` + `19-serving.md` — enterprise scale
5. `23-evaluation.md` + `24-failure-analysis.md` — "understanding why models fail"

---

## Sprint 4 — Mon 28 to Wed 30 September

**No new material.** Anything not learned by the 27th is not going to be learned well by the 30th,
and cramming displaces what you already know.

| Day | Work |
|-----|------|
| Mon 28 | **Write-up.** Parameter derivation, ablation tables, scaling fit + tested prediction, length-growth plot. This is the artifact, not the model. `git init` the repo and push it. |
| Tue 29 | **Drill.** Ask me for the Interview section of any module and I'll grade you. Then `GLOSSARY.md` end to end — any term you can't place is a gap. |
| Wed 30 | Read your own notes. Nothing new. |

**Final CV pass on Mon 28**, once the write-up exists and has a URL.

---

## The five things you should be able to say on the 30th

Not "I studied this" — *I did this, and here is the number*:

1. **"I pretrained a 3.4M transformer from scratch."** Parameter count derived by hand, RoPE's
   relative-position property verified on my own implementation, and real Qwen3-0.6B weights loaded
   into my class with logits matching to 1e-3.
2. **"I fitted a scaling law to three models I trained, predicted a fourth before training it, and
   landed within X%."** Plus why Kaplan and Chinchilla disagreed — an LR-schedule artifact — and why
   nobody trains Chinchilla-optimal any more.
3. **"I reproduced R1-Zero's length growth at 3.4M parameters."** Mean completion went from ~40 to
   ~95 tokens with no length term anywhere in the reward.
4. **"I ran the same SFT → GRPO pipeline on an agent environment with a programmatic verifier"** —
   and here is the baseline it beat.
5. **"Here is what I haven't done."** Multi-node training at real scale. Naming a limit and saying
   what you'd do about it reads as senior; the alternative is detectable.

---

## Checking in

Tell me your numbers as you get them — I'll say whether a run is healthy before you waste a day on
it. Say *"drill me on X"* any time and I'll run the module's Interview section on you.

If you fall behind, say so on the day, not on the 27th. The triage that actually works is dropping
B0's autograd engine and B2's tokenizer (use `tiktoken`) — losing depth on two topics to keep the
trained model. Cutting B6 or B10 to keep everything else is the wrong trade every time.

---

**Back:** [START-HERE.md](START-HERE.md)
