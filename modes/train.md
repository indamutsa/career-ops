# Mode: train — Detect → Teach → Build → Place

Standing workflow for this user. Turns a job description into a training plan and a set of
buildable artifacts, then places the results on the CV.

## Purpose

The user holds a PhD and ramps on new technical areas in roughly two weeks. Gap analysis that
stops at "you lack X" is useless to him. This mode closes the loop: identify the gap, teach it
properly, build a real artifact, then write it onto the CV in the right slot.

## Inputs

- A JD, recruiter message, or a bare technology name (`/career-ops train GRPO`)
- `cv.md`, `article-digest.md`, `config/profile.yml`, `modes/_profile.md` — the evidence bank
- An interview date if one is known — it drives the build order

## Steps

### 1. Classify every requirement into one of three buckets

| Bucket | Test | Action |
|--------|------|--------|
| **Have it, written** | Already stated in `cv.md` in the reader's vocabulary | Nothing |
| **Have it, unwritten** | Real work in a past role that never made the page, or is worded for a different audience | Rewrite the bullet in the target role's language. Free, immediate. |
| **Don't have it** | No past role genuinely involved it | Teach it, then build a personal project |

Bucket 2 is the highest-yield and the most frequently missed. Check `cv.md` and
`article-digest.md` before concluding anything is absent — published writing counts as evidence.

### 2. Teach

Explain the concept at the level of someone who will be interviewed on it, not at blog-post level.
Every topic must cover:

- **Why the technique exists** — what the prior approach could not do
- **The failure mode** — how it goes wrong in practice, and the signature that reveals it
- **The interview question** — what a strong interviewer probes, and the answer that shows
  hands-on experience rather than reading

### 3. Build

Follow the hands-on rules in `modes/_custom.md` ("How we learn"): worktree per topic, a markdown
lesson doc with step-by-step runnable code, Docker for services, native MPS for training, and the
smallest model that can prove the pipeline.


Prefer **one artifact covering several gaps** over one artifact per gap. Each project needs a
public repo and a write-up; for research-flavored roles the write-up carries more weight than the
benchmark number, because it demonstrates the reasoning the role is hiring for.

Work backwards from the interview date. An artifact must exist before it is claimed.

### 4. Place

Follow the placement rule in `modes/_custom.md`:

1. Real work in a past role → write it into that role
2. Otherwise → `## Selected Engineering Projects`, with the repo
3. While still building → `## Current Focus`, promoted out on completion

## Output

1. The three-bucket table for this JD
2. Rewrites for every bucket-2 item, as Current → Replacement snippets
3. A build plan for bucket 3: artifacts, sequence, and dates working back from the interview
4. The first teaching block, started immediately — never a plan alone

## Notes

- `cv.md` is an evidence bank, not the deliverable. Tailored per-role output comes from `pdf`,
  `text`, or `latex`, which select from the bank. Keep the bank broad and accurate; let the
  tailoring narrow it.
- Do not stop at gap identification. Identifying a gap without teaching it is an incomplete run
  of this mode.
