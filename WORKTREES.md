# Worktrees — learning and build tracks

Index of the git worktrees used for interview preparation and CV artifacts.

Each learning topic gets its own worktree so that `main` stays clean and syncable with upstream
`santifer/career-ops`. The worktrees are workbenches, not features — a branch is merged to `main`
only to deliver finished learning material into `learning/`, never to carry work in progress.
Artifacts destined for the CV are extracted into their **own public repositories** when they are
finished, because a CV link must point at a clean, standalone repo.

**Start here:** [`NOTE.md`](NOTE.md) at the repository root is the reading guide to both learning
tracks — what each one is, where to start, and what state it is in.

> This file is `WORKTREES.md` rather than `README.md` on purpose: `README.md` is an upstream file
> and editing it would conflict on every fork sync.

---

## Active worktrees

| Worktree | Branch | Purpose | Status |
|----------|--------|---------|--------|
| `.claude/worktrees/learn-agent-posttrain` | `worktree-learn-agent-posttrain` | **Project A** — agentic post-training pipeline. Covers agentic orchestration, dataset construction, post-training (SFT → DPO → GRPO), distributed training. | Lab 01 written; awaiting baseline run |
| `.claude/worktrees/llm-curriculum` | `worktree-llm-curriculum` | **Curriculum authoring** — where the LLM curriculum was written. Content is now delivered to `learning/llm-from-scratch/`; this worktree is the authoring history only. | Delivered — see below |

---

## Where the learning actually happens

Two tracks, both in this checkout rather than in a worktree, both tracked in git:

| Track | What it is |
|-------|------------|
| `learning/llm-from-scratch/` | The **build track** — you write the code, B0→B10 |
| `learning/ml-interview-prep/` | The **recall track** — a 44-module offline HTML course |

### `learning/llm-from-scratch/` — the build track

Start at [`learning/llm-from-scratch/START-HERE.md`](learning/llm-from-scratch/START-HERE.md).
That file is the single entry point: environment check, first lesson, and the order for everything
else. Do not start from a `README.md` and hunt.

```
learning/llm-from-scratch/
  START-HERE.md      the entry point — read this first
  SYLLABUS.md        index of 28 reference modules (00-25, 05a, 15a)
  GLOSSARY.md        175 terms, each with where it is met in production
  modules/build/     Build Track B0-B10: an LLM built from an empty file
  modules/           the reference modules
  code/              your lab code (check_env.py is the only file provided)
  data/ checkpoints/ notes/
```

### `learning/ml-interview-prep/` — the recall track

The complement to the build track: the ladder underneath it (maths, classical ML, deep learning)
and the production layer above it (MLOps), plus the mock rounds and six end-to-end system-design
walkthroughs. 44 modules, 294 question accordions, 23 visualisations, no build step, works offline.

```bash
cd learning/ml-interview-prep && python3 -m http.server 8000
```

Serve it if you can — over `http://` search and drill mode work across all modules at once. Opening
`index.html` directly also works; it falls back to an iframe per module.

**Both tracks are tracked in this fork.** `learning/` used to sit in `.git/info/exclude`, which kept
it out of fork syncs but also meant none of it was committed anywhere; that line was removed on
5 Sep 2026 and both tracks now live on `main`. The write-ups still get extracted into their own
public repositories when they are finished — a CV link must point at a clean, standalone repo.

---

## What lives where

### `learn-agent-posttrain`

The buildable artifact for the Senior AI Research Engineer application.

```
learning/lab-01-agent-environment.md   verifiable SQL agent env + baseline harness
agent-posttrain/                        the code (written by hand, from the lab)
  docker/                               Postgres fixture — services only, no GPU
  env/                                  tools, rollout loop, verifier, tasks, model
  scripts/                              baseline and training entrypoints
```

Destination: its own public repo, then `## Selected Engineering Projects` in `cv.md`.

### `llm-curriculum`

The knowledge base behind the artifact. Two tracks, done in parallel.

```
modules/build/         Track A — B0..B10, an LLM built from nothing
  00-foundations       scalar autograd, written by hand, verified against torch
  01..04               tensors, BPE tokenizer, memmapped data, attention
  05                   the 3.4M transformer (RMSNorm, RoPE, SwiGLU, tied head)
  06..08               pretraining, sampling + KV cache, scaling laws
  09..10               instruction tuning, then CoT + GRPO against a verifier
modules/00..25         Track B — 28 reference modules, incl. 05a encoder-decoder
                       and 15a DeepSeek-R1
GLOSSARY.md            175 terms, each with where it is met in production
```

Every module has the same shape: **Terms · Concepts · Where it's used · Labs · Failure modes ·
Interview Q&A · Checkpoint.** Each lab ends with a number you produced; the checkpoint is what you
must be able to state before moving on.

The Build Track follows the three real post-training stages in order — pretrain (B0–B8), instruction
tuning with chain of thought (B9–B10), RL from a verifiable reward (B10) — at a scale that trains on
an M3 Pro in under an hour per model.

**Delivered to `learning/llm-from-scratch/` in this checkout** (see above). The worktree retains the
authoring history; work happens in `learning/`.

Destination: the Build Track's write-up (parameter derivation, ablation tables, fitted scaling law
with its tested prediction, RL length-growth plot) is a standalone artifact worth its own public
repo. Selected write-ups may become blog posts on indamutsa.com, which feed
`## Selected Technical Writing` in `cv.md`.

---

## Working with them

```bash
git worktree list                                    # what exists
cd .claude/worktrees/<name>                          # enter one
git -C .claude/worktrees/<name> status               # check from anywhere
```

Inside a Claude Code session, `EnterWorktree` / `ExitWorktree` switch the session's working
directory. Only one worktree session is active at a time; exit with `keep` before entering another.

**Never `git stash` bare** — the stash stack is shared across every worktree and the main checkout,
and another session can pop your entry. Use a WIP commit instead.

---

## Machine constraints that shaped these

Apple M3 Pro, 36 GB, macOS.

- **Docker has no GPU passthrough on macOS.** Containers run in a Linux VM with no Metal access, so
  Docker holds services (Postgres, eval harnesses) and training runs natively on MPS.
- **Small models by default.** Qwen3-0.6B / 1.7B class proves the pipeline. Scale the model last;
  never debug plumbing on a big model.
- **Rent a GPU only for final runs.** Real GRPO training at useful scale does not belong on a laptop.

---

## Related

- `modes/train.md` — the detect → teach → build → place workflow (`/career-ops train`)
- `modes/_custom.md` — CV posture, placement rules, and the "How we learn" standing rules
- `cv.md` — `## Current Focus` tracks what these worktrees are closing
