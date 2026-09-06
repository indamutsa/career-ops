# Handoff — read this first

**Written:** 5 September 2026 · **Interview:** Wednesday 30 September 2026 · **25 days left.**

Everything built for the interview lives under `learning/`. There are two tracks there and they are
not alternatives. One teaches you LLMs by making you build one; the other makes you able to explain
everything — including the parts you will not have time to build — out loud, under pressure, to a
stranger.

| Folder | What it is | Who does the work |
|--------|------------|-------------------|
| `learning/llm-from-scratch/` | The **build track**. Eleven lessons, empty file → a small reasoning LLM you trained. | **You write every line of code.** I write the lessons and review what you produce. |
| `learning/ml-interview-prep/` | The **recall track**. A 43-module HTML course: maths → classical ML → deep learning → transformers → MLOps → mock rounds → six system-design walkthroughs. | Read it, then close it and say the answers out loud. |

Read this note in that order — build track first, course second — because that is the order of
priority. The build track produces the artifacts and the numbers you will be asked about. The
course is what turns those artifacts into answers, and covers the ground you will not have built.

Everything else in this repository is career-ops, the job-search pipeline. It is unrelated to the
two tracks and untouched by them — **Part 3** below is the working cheat sheet for it.

---

# Part 1 — `learning/llm-from-scratch/`, the build track

## The one rule

> **The lessons are written. You write the code.**

That was the agreement and it is the reason the track is worth anything. A lesson gives you the
concept, the algorithm, and the step-by-step instructions. It does not give you a file to run. If
you type it yourself and it breaks, you learn where it breaks — which is exactly the thing an
interviewer is probing for.

## Where to start

```bash
cd $HOME/PROJECT-INITIATIVE/InterviewGuide/resume/career-ops/learning/llm-from-scratch
python3 code/check_env.py
open START-HERE.md
```

`START-HERE.md` is the entry point — not `README.md`. It walks you through the environment check,
then hands you to `modules/build/00-foundations.md`.

**Reading it away from the laptop:** the same markdown is published as HTML at
**https://ml-interview-prep-taupe.vercel.app**, in the sidebar groups `H · LLM track` through
`K · Reference`. It is generated, not hand-written — `learning/llm-from-scratch/build-site.mjs`
converts the markdown and writes the pages into the course folder, so the markdown stays the single
source of truth and nothing has to be edited twice. Regenerate and publish in one step:

```bash
cd $HOME/PROJECT-INITIATIVE/InterviewGuide/resume/career-ops/learning/llm-from-scratch
node build-site.mjs --deploy
```

`notes/`, `code/`, `data/` and `checkpoints/` are excluded from the build on purpose: the URL is
public and those are your own measurements.

## What is in the folder

```
learning/llm-from-scratch/
  START-HERE.md       ← begin here. The map, the schedule, the ground rules.
  PLAN.md             the dated 25-day plan: four sprints, a hard gate at the end of each
  SYLLABUS.md         index of the 28 reference modules
  GLOSSARY.md         175 terms, each with what it is and where you meet it in production
  modules/
    build/            the Build Track, B0–B10 — this is the spine
    00–25, 05a, 15a   the 28 reference modules (deep dives, read alongside)
  code/               ← YOUR code goes here. One file per lesson.
    check_env.py      one of only two files I wrote in here
  data/               corpora and .bin token files
  checkpoints/        trained model weights
  notes/              ← YOUR measurements. One file per lesson. Not optional.
```

## The eleven build lessons

Each leaves you something that runs.

| | Lesson | Time | You end with |
|---|--------|------|--------------|
| **B0** | `00-foundations.md` | 3–4 h | A scalar autograd engine, verified against PyTorch to 1e-6 |
| **B1** | `01-first-model.md` | 2–3 h | A trained bigram model, and a baseline loss to beat |
| **B2** | `02-tokenizer.md` | 3–4 h | Your own 4,096-token BPE tokenizer, trained on your corpus |
| **B3** | `03-data.md` | 2–3 h | A memory-mapped token stream and a batch loader |
| **B4** | `04-attention.md` | 3–4 h | Multi-head causal attention you implemented |
| **B5** | `05-transformer.md` | 4–5 h | A 3,443,136-parameter model, counted to the digit |
| **B6** | `06-training.md` | 3 h + 45 min run | **A model that writes coherent English** |
| **B7** | `07-sampling.md` | 2–3 h | Every decoding strategy plus a KV cache |
| **B8** | `08-scaling.md` | 3 h + runs | A power law fitted to models *you* trained, and a tested prediction |
| **B9** | `09-instruct.md` | 3 h | A base model **and** an instruct model, both yours |
| **B10** | `10-reasoning.md` | 4–5 h | Chain of thought plus GRPO against a verifier you wrote |

The shape is the same three stages every frontier model goes through, four orders of magnitude
smaller: pretraining (B0–B8) → instruction tuning and chain of thought (B9–B10) → reinforcement
learning from a verifiable reward (B10).

## The schedule and its gates

`PLAN.md` has the dated version. The short form:

| Sprint | Dates | Gate — you do not move on until this holds |
|--------|-------|---------------------------------------------|
| 1 | Sat 5 – Sun 13 Sep | B0–B5: your hand-derived parameter count equals `count_params()` |
| 2 | Mon 14 – Sun 20 Sep | B6–B8: **a model that writes English**, plus a scaling prediction you tested |
| 3 | Mon 21 – Sun 27 Sep | B9–B10: instruct and reasoning checkpoints, length-growth plot |
| 4 | Mon 28 – Wed 30 Sep | **No new material.** Write-up, drill, CV. |

The gates are hard on purpose. If one slips, the next sprint absorbs it — say so on the day rather
than carrying the slip silently. The triage that works is dropping B0's autograd engine and B2's
tokenizer (use `tiktoken` instead). Never cut B6 or B10: B6 is the only lesson that produces a model
that visibly works, and B10 is the only one that puts you on the DeepSeek-R1 conversation.

## The reference modules

28 deep dives indexed in `SYLLABUS.md`, all with the same shape — Terms · Concepts · Where it's
used · Labs · Failure modes · Interview Q&A · Checkpoint.

Read them **in parallel with the build track, not before it.** Each build lesson names its reference
companion at the top; read that one after finishing the build lesson, while the code is still fresh.
Reading them first is the failure mode — it feels productive and teaches you almost nothing.

`GLOSSARY.md` is for the night before: 175 terms, read end to end. Any term you cannot place is a
gap, and the glossary entry tells you which module to go to.

## `notes/` is the part people skip

Every lesson's checkpoint asks for measurements, and three later lessons consume earlier ones
directly. B8 fits a scaling law to the three loss curves from B6. B9's before/after comparison needs
the exact base-model output you recorded in B6 Lab 8. If you do not write them down you re-run the
training.

A `notes/b6-training.md` that reads *"step 0 loss 8.31 (ln 4096 = 8.32 ✓), final val 1.58, 43 min,
grad norm settled ~0.3, no-warmup run plateaued at 1.71"* is worth more on interview day than any
paragraph you could reconstruct from memory.

## Honest status

**Nothing has been run yet.** `code/` contains only the two scripts I wrote (`check_env.py`,
`verify_curriculum.py`). `checkpoints/` and `notes/` are empty. The lessons, the plan, the syllabus
and the glossary are complete and waiting.

Your environment was recorded as Python 3.13.13, torch 2.13.0, MPS available, bf16 autocast — so
`check_env.py` should pass on the first run.

**The next action is B0, today.** Sprint 1 started Saturday 5 September, which is today.

## Ground rules worth repeating

- **Type the code.** Not copy — type.
- **CPU first, MPS second** for anything new. CPU is easier to debug and fast enough under 1M params.
- **Seed everything** (`torch.manual_seed(1337)`). You will be comparing runs constantly.
- **Commit after every lesson.** You will want a working state to return to.
- **`torch.mps.synchronize()` before reading any clock.** MPS dispatch is asynchronous; without it
  your timing numbers are fiction. This catches almost everyone once.

## Working with me on this track

Useful things to say: *"I'm on B4 Lab 3, my causality test fails"* (paste the code) · *"explain the
√d scaling again, I typed it but I don't believe it"* · *"here are my B6 numbers"* (I will tell you
whether the run is healthy) · *"drill me on attention"* · *"I have 5 days, re-plan"*.

---

# Part 2 — `learning/ml-interview-prep/`, the recall track

## What it is and why it sits next to the build track

The build track goes deep on one thing. The interview will not. You will be asked about gradient
boosting on tabular fraud data, about PR-AUC under class imbalance, about canary rollouts and PSI
drift, about the dual form of an SVM — none of which you will build in 25 days, all of which you are
expected to explain without hesitation.

This course is the ladder underneath and the production layer above. It has no code in the teaching
modules by design: analogies, algorithms and formulas only, because you are being trained to *say*
these things, not to type them. The one exception is the config fragments in the walkthroughs.
There is no DSA round here — that is covered elsewhere.

## How to open it

**From anywhere — it is deployed:**

> **https://ml-interview-prep-taupe.vercel.app**

One site, both tracks: 43 hand-written course modules (groups `Start` to `G`) plus the 44 generated
pages of the build track (groups `H` to `K`), behind one nav and one search box. Searching
*attention* returns the course module, the build lab and the reference module together.

Redeploy with `node build-site.mjs --deploy` from `learning/llm-from-scratch/` — that regenerates
the build track first, which `vercel deploy --prod` on its own would not. If you only touched the
hand-written course files, `vercel deploy --prod` from the course folder is enough. The URL is
public — Vercel's Deployment Protection can put a password in front of it if you would rather it
were not. Full deployment notes, including how to relink the project if `vercel` ever offers to
create a new one, are in `learning/ml-interview-prep/README.md`.

**Locally:**

```bash
cd $HOME/PROJECT-INITIATIVE/InterviewGuide/resume/career-ops/learning/ml-interview-prep
python3 -m http.server 8000
# then open http://localhost:8000
```

Serve it if you can — over `http(s)://` the shell fetches each module and injects it, which gives you
**search and drill mode across all 87 modules at once**. Double-clicking `index.html` also works:
Chrome blocks `fetch()` on `file://`, so the shell falls back to an iframe per module and tells you
so in a one-line banner. Everything still renders; search and drill are just scoped to the open
module instead of all of them.

No build step, no bundler, no CDN. The only network call in the whole thing is a Google Fonts
`@import` — kill your wifi and it still works, with substituted fonts.

## The shape of every module

Once you have read two modules you know the shape of all 44, which is the point:

1. **Term cards** — the name, a one-sentence plain definition, the analogy, and *where you actually
   meet it in production*.
2. **Concept sections** — intuition first, algorithm second, formula last, with a hand-drawn SVG
   animation wherever motion explains something a static diagram cannot.
3. **Question accordions** — the interview question; inside, a gold **"SAY THIS FIRST"** line that
   is the 15-second version, then the full answer, then a ⚠️ trap box: the follow-up that catches
   people.
4. **A checkpoint** — you may move on when you can say X out loud without notes.

Difficulty pips mark each question BASIC / MID / SENIOR. At Senior AI Research Engineer level the
BASIC ones are the dangerous ones: they are the questions you are expected to answer instantly and
cleanly, and a fumbled "what is a logit" costs more than a shaky answer on speculative decoding.

## Controls

| Key | Does |
|-----|------|
| `/` | Jump to search (search box lives in the top bar) |
| `j` / `k` | Next / previous module |
| `d` | **Drill mode** — collapses every answer, leaves only the questions |
| `t` | Toggle theme (chalkboard / paper) |

"mark done" records progress per module; it persists in `localStorage`, so your progress bar
survives closing the browser. Drill mode is the one that matters — reading answers feels like
learning and is not. Turn it on, work down the questions out loud, open an accordion only after you
have committed to an answer.

## The ladder

| Part | Modules | Covers |
|------|---------|--------|
| **0** | `00` | How to use it, the four-sentence answer formula, what to do in the last week |
| **A — Maths from zero** | `10–14` | Linear algebra · calculus and what a minimum actually is · probability · information theory (why cross-entropy *is* the loss) · optimisation including Lagrange multipliers, KKT, and the Laplace approximation |
| **B — Classical ML** | `20–26` | The learning problem and leakage · linear models · trees and gradient boosting · SVM/kNN/Bayes (the dual, as the payoff for Lagrange) · unsupervised and anomaly detection · metrics, ROC-AUC vs PR-AUC under imbalance, calibration · interpretability and when SHAP lies to you |
| **C — Deep learning** | `30–34` | Neural nets and backprop as the chain rule · CNNs · RNN → the fixed-vector bottleneck → attention as its fix · classical NLP and embeddings · RL, which walks straight into RLHF |
| **D — Transformers & GPT** | `40–46` | Attention and why √d · the block, RoPE, RMSNorm, counting parameters by hand · model families and why decoder-only won · pretraining and Chinchilla · post-training, DPO, GRPO · inference: KV cache, GQA, quantisation, LoRA, speculative decoding · RAG, agents, evaluation |
| **E — MLOps & production** | `50–55` | Kafka, backpressure, sync vs async, delivery semantics · Feast and point-in-time correctness · MLflow and Kubeflow · canary, blue-green, shadow, bandits · drift, PSI, KS, delayed labels, SLOs · governance, fairness, cost |
| **F — Mock rounds** | `60–64`, `66` | Five subject rounds and a behavioural round mapped to the STAR bank |
| **G — Walkthroughs** | `70–75` | Six end-to-end system designs (below) |

## The six walkthroughs — the highest-value hours in the course

Each one is a full 45-minute system-design answer with the timing broken out minute by minute, a
compressed 5-minute version you can deliver verbatim, and a callback table mapping every stage back
to the module that taught it.

| | Prompt |
|---|--------|
| `70` | **Fraud detection**, end to end — Kafka ingestion and backpressure, feature store and point-in-time correctness, class imbalance, a threshold from a cost matrix, canary rollout, PSI monitoring, the delayed-label feedback loop |
| `71` | **RAG assistant** — retrieval quality, chunking, hybrid search, grounding and citation, the injection surface |
| `72` | **Recommendations** — the funnel with per-stage latency budgets, two-tower retrieval, negative sampling, position bias and inverse-propensity weighting, why offline NDCG gains vanish online |
| `73` | **LLM serving** — three pools by SLO, KV-cache arithmetic to the byte, prefill vs decode, continuous batching, paged attention, cost per million tokens, the degradation ladder |
| `74` | **Post-training pipeline** — why post-training is the *last* option, curation over volume, decontamination, SFT → preference tuning → RL with a verifier, and four evaluation suites |
| `75` | **Agent system** — why 0.95²⁰ ≈ 36 % is the whole problem, the tool gateway as a real service, idempotency, sandboxing, prompt injection, staged expansion of authority |

If you only have one evening for the course, spend it on `70` and `73`. If you have two, add `74`.

## Honest status

The course is complete and verified: 43 modules, 284 question accordions, 23 embedded
visualisations. Every module was loaded in a browser this session — no 404s, no blank panes, one
navigation footer each, no console errors, no horizontal overflow at desktop or at 430 px, and
search resolves correctly across all modules.

Two things are deliberately unresolved and are your call:

- **The light/paper theme.** The chalkboard palette is copied exactly from the calculus page. The
  light theme is mine, not from that reference. Keep it or drop it.
- **The Google Fonts `@import`.** It is the only network dependency. Everything degrades gracefully
  without it, but if you want the course to be genuinely 100 % offline, the fonts have to be
  vendored or dropped.

## How to review it

Open it, press `d` for drill mode, and go down Part A. If a question makes you hesitate, that module
is your evening. Nothing here needs proofreading from you — it needs testing against your own
recall, which is a different activity and the only one that moves the number.

---

# Part 3 — career-ops itself, the short version

The repository is large — 120 scripts and roughly 60 modes — but you only ever touch a handful. The
rest is machinery those few call. This section is the working set plus the two commands that let you
find everything else without reading source.

## Start here when you forget what exists

```
/career-ops
```

With no arguments it prints the full command menu. That is the canonical index and it is generated
from the router, so it cannot go stale. Two written indexes back it up, both readable without an
agent:

```bash
sed -n '/^| If the user/,/^$/p' AGENTS.md     # every mode and what triggers it
sed -n '/^| File | Function |/,/^$/p' AGENTS.md   # every script and what it does
node find.mjs --help                          # and most scripts answer --help
```

`modes/README.md` explains how a mode file works if you ever want to write or edit one — which you
are meant to: the modes are prompts in Markdown, not code.

## The daily loop

| Command | What it does |
|---------|--------------|
| *paste a JD or URL, no command* | **auto-pipeline** — evaluates, writes the report, generates the CV, updates the tracker. The single most-used path. |
| `/career-ops scan` | Sweeps the configured portals for new roles. Zero LLM cost — it hits the Greenhouse/Ashby/Lever APIs directly. |
| `/career-ops triage` | Fast first pass over a batch before spending a full evaluation on any of them |
| `/career-ops pipeline` | Works through the URLs you parked in `data/pipeline.md` |
| `/career-ops pdf` | Tailored, ATS-optimised CV for one role |
| `/career-ops cover` · `email` | Cover letter · formal application email. Both draft-only. |
| `/career-ops apply` | Live form assistant — reads the form, drafts the answers. Stops before Submit, always. |
| `/career-ops tracker` | Where everything stands |
| `/career-ops followup` | Who is overdue a nudge, and the draft to send |
| `/career-ops outcome` | Record the result and archive the artifacts |

**The parking habit worth forming:** when a role looks interesting but you have no time, append the
URL to `data/pipeline.md` and move on. `/career-ops pipeline` drains the queue later in one pass.
`/career-ops agent-inbox` does the same for requests to me rather than for URLs.

## The ones that matter for your situation right now

| Command | What it does |
|---------|--------------|
| `/career-ops train` | Your standing loop: classify a JD into have-it-written / have-it-unwritten / don't-have-it, then teach, build, and place. Also takes a bare topic: `/career-ops train GRPO`. |
| `/career-ops interview-prep` | Company-specific prep document |
| `/career-ops interview/drill` | Deep JD-specific question set — baseline answer vs. staff-level answer, plus the traps |
| `/career-ops interview/ready` | Ranks what to study next from your real question-bank history rather than from a guess |
| `/career-ops interview/practice` · `debrief` | One question at a time with feedback · post-interview gap closing |
| `/career-ops upskill` | Skill gaps aggregated across every role you have evaluated |
| `/career-ops interview-redflag` | Is this company safe to join |

## Zero-token utilities

These are plain Node scripts. No model call, no cost, safe to run whenever.

```bash
node doctor.mjs --json          # is the system set up; what is still template content
node verify-pipeline.mjs        # health check: broken links, bad statuses, duplicates, orphans
node stats.mjs --summary        # lifetime funnel, scan totals, portal coverage
node find.mjs <company|report#> # locate a report, tracker row and artifacts from any fragment
node merge-tracker.mjs          # merge queued tracker additions — run after every batch
node set-status.mjs <ref> <State> --note "..."   # the ONLY safe way to change a status
node analyze-patterns.mjs       # where applications actually die
node funnel-velocity.mjs --summary   # your funnel against market benchmarks, and stage velocity
node company-history.mjs <company> --summary   # everything known about one company
node jd-skill-gap.mjs <jd-file> --summary      # a JD's skills vs. cv.md: have / supported / gap
node interview-readiness.mjs --summary         # what to study next, measured vs. untested topics
node weekly-digest.mjs --summary               # this week's interview sessions rolled up
node scan-ats-full.mjs --resume  # keyword-first sweep of full public ATS datasets, no company list
```

## Three rules that prevent most of the damage

1. **Never hand-edit `data/applications.md` to add a row.** Write a TSV into
   `batch/tracker-additions/` and run `node merge-tracker.mjs`. To change a status, use
   `set-status.mjs` — it validates, locks, and writes atomically.
2. **Never run `node update-system.mjs apply` in this fork.** Its apply is a raw overwrite of exactly
   the system files this fork has modified on purpose. Sync with upstream by merging
   `upstream/main` on a branch instead.
3. **Personalisation goes in `modes/_profile.md`, `modes/_custom.md` or `config/profile.yml`** —
   never in `modes/_shared.md`, which system updates overwrite.

## Customising it

The whole point of the design is that you say what you want changed and it gets changed: archetypes
and targeting in `modes/_profile.md`, house rules and workflow preferences in `modes/_custom.md`,
companies and search keywords in `portals.yml`, CV design in `templates/cv-template.html` or
`.tex`. `modes/train.md` is an example of the pattern — a workflow of yours turned into a mode so it
runs the same way every session.

---

# What to do next

1. **Today (Sat 5 Sep):** `python3 code/check_env.py`, then `modules/build/00-foundations.md`.
   Sprint 1 starts today and its gate is nine days out.
2. **Every lesson:** write the checkpoint numbers into `notes/`, commit, move on.
3. **Evenings and dead time:** the course, in drill mode, in ladder order. Part A and Part B first —
   they are the ones you are furthest from using daily.
4. **The night before:** `GLOSSARY.md` end to end, then walkthrough `70` out loud, timed.

## Version control

Both tracks are committed in this fork. `learning/` used to sit in `.git/info/exclude`, which kept
it out of fork syncs with upstream `santifer/career-ops` but also meant none of it was committed
anywhere; that line was removed on 5 September 2026 and both tracks were merged to `main`. Your lab
code in `code/` and your measurements in `notes/` now show up in `git status` like anything else —
which matters, because "commit after every lesson" is one of the ground rules.

`learning/` is a path upstream never touches, so tracking it adds no merge-conflict risk on a fork
sync. The finished write-ups still get extracted into their own public repositories, because a CV
link has to point at a clean, standalone repo.

## Loose ends

- **Inside `.claude/worktrees/llm-curriculum/`** there are leftovers from authoring: a duplicate
  `GLOSSARY.md`, a duplicate `modules/` tree, and three verification screenshots. That directory is
  git-excluded, so none of it reaches the fork — but the worktree and its branch are both still on
  disk if you want the space back.
