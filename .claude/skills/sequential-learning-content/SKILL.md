---
name: sequential-learning-content
description: >-
  Standard for writing and refining educational, technical, mathematical,
  scientific, ML/data, system-design, or interview-preparation content —
  especially the pages under learning/ml-interview-prep/parts/*.html in this
  repo, but general enough for any similar content page. Use this whenever
  writing a new content page, adding a new module or section, refurbishing or
  rewriting an existing learning/interview-prep page, explaining a concept for
  a course, drafting worked examples or derivations, or adding any new
  interactive element (accordion, tab, modal, worked-math block, comparison
  block, etc.) to one of these pages — even if the user doesn't say "skill" or
  reference this by name. Trigger on requests like "write the probability
  page", "add a section on X", "make this page less verbose", "add a
  worked example for Y", or "this page feels too dense/shallow".
---

# Sequential Learning Content

The objective is content that is concise, sequential, self-contained,
mathematically explicit, example-driven, interview-friendly, easy to scan, and
deep on demand.

The central rule: **the learner must never need to infer a missing
prerequisite, unexplained term, hidden calculation, or intermediate result.**

Conciseness must come from removing repetition and filler, **not from
removing reasoning steps.**

## Worktree first — mandatory

Before inspecting target content, drafting changes, editing files, or running
task-specific tests, create a dedicated feature branch and git worktree for
the learning-content task. Minimal read-only repository discovery needed to
identify the repository, current branch, status, and existing worktrees may
happen first; creating the worktree must be the first mutating action.

Perform the entire task from that worktree. Never edit learning content in
the primary checkout, even when it is clean or already on a feature branch.
Use a task-specific branch and a sibling worktree location that follows the
repository's existing convention. If the intended branch or worktree already
exists, verify that it belongs to this task and use it; otherwise choose a
new task-specific name without deleting or overwriting anything.

Do not use `git stash`, destructive git commands, or deletion to relocate
existing changes. If task-owned changes already exist outside the worktree,
preserve them as a patch, apply the patch inside the worktree, verify that
the transferred files match byte-for-byte, and only then remove the
transferred patch from the original checkout with a non-destructive reverse
apply. If ownership is ambiguous or unrelated changes overlap, stop and ask
the user before moving them.

## Git writes always need explicit, per-action approval

Completing a learning-content task — including under an active goal/"complete
this" instruction that authorizes working autonomously through the writing,
QA, and file-editing steps — never by itself authorizes `git commit`,
`git push`, `gh pr create`, `gh pr merge`, or any other git/gh write
operation. Autonomy granted over the content work (drafting, rewriting,
running QA passes, fixing findings) does not extend to version control.
Reaching "done" means stopping with the finished, QA'd files sitting
uncommitted and reporting that back — never committing, pushing, or merging
on your own judgment that the task is complete. Ask first, every time,
regardless of how the task was framed or how much autonomy was granted for
everything up to that point. This applies to every agent or sub-agent
touching this skill's content, including QA/verification passes that were
scoped as read-only: a read-only QA brief is not itself authorization to
then commit what it reviewed.

## Before editing an existing page

Read the whole page first. Almost every task here is refurbishing a page
that already committed to a running example (rule 7) and a dependency order
(rule 1) — find both before writing anything. Extend the example that's
already there and slot new material into its existing sequence; don't
introduce a second example or a term the page hasn't reached yet just
because it's convenient for the new section. If the new content genuinely
doesn't fit the existing order, that's a sign the surrounding sections need
reordering too, not that the rule can be skipped for this one section.

## 1. Enforce prerequisite-safe sequencing

Concepts must be introduced in dependency order. Before using a concept,
term, symbol, operation, or abbreviation, make sure the learner already knows
it — never assume understanding merely because a concept is common in the
field. For every new item, ask: "Has this already been introduced and
explained?" If not, either explain it before continuing, or expose its
explanation through the page's existing `?` / expandable-detail mechanism.
Never require the learner to reverse-engineer meaning from context.

Bad: "Lasso minimizes the loss with an L1 penalty" — if the page hasn't yet
explained loss, coefficient, regularization, penalty, norm, L1.

Better sequence: model prediction → prediction error → loss → model
coefficients → overfitting → regularization → penalty → norms → L1 norm →
Lasso. The learner should always understand the current node before moving to
the next one.

## 2. No unexplained terminology

Every technical term must be: already explained, OR immediately explained, OR
have a clickable/expandable explanation available. This applies to
mathematical terminology, ML terminology, statistics terminology, programming
terminology, symbols, abbreviations, notation, operations, and assumptions.
Don't write "we minimize the regularized objective" unless the learner
already understands minimize, objective, and regularization. The learner
should never think "when did we learn this?"

## 3. No numbers without meaning

Never throw numbers into an example just to make it look mathematical. Every
number needs a clear role — if using a vector of values, state what each
value means (e.g. size in m², bedrooms, age in years). The learner must
understand the data before operations are performed on it. Don't use
arbitrary values when meaningful ones can be used instead.

## 4. Results must always follow visible reasoning

Never jump from an input straight to a result when intermediate reasoning
exists. Forbidden: showing a matrix, then asserting `det(A) = 5` with no work
shown. Required: state the formula/rule, substitute the values, show each
intermediate calculation, then the result, then the interpretation.

The rule is: **input → rule/formula → substitution → intermediate steps →
result → interpretation.** Never: input → result.

## 5. Mathematical derivations must be reproducible

A derivation is complete only when the learner could reproduce it
independently. For every worked calculation: show the starting values, show
the formula or rule, explain the symbols if necessary, substitute the values,
show intermediate calculations, show the final result, and explain what it
means. Don't hide meaningful arithmetic or algebra inside prose, and don't
write "after calculating, we get..." when the calculation itself is part of
what the learner needs to understand.

## 6. Render mathematics with the site's math markup, never as plain prose

Important mathematical objects must be rendered as mathematics, not
compressed into a sentence — but this site has no MathJax/KaTeX/LaTeX
renderer, so "proper notation" means the site's own markup, not `$$...$$`
syntax. Writing raw LaTeX here ships as literal broken text on the page.

Use what `10-linear-algebra.html` already uses: `<span class="m">…</span>`
for an inline math expression, `<span class="matrix">…</span>` (add
`vector` for a single column) with one nested `<span>` per entry for a
matrix or vector, `<div class="eq">…</div>` for a standalone display
equation, real `<sub>…</sub>` tags for subscripts, and the Unicode
characters directly for superscripts and symbols (`Vᵀ`, `ft²`, `λ₁`, `Σ`,
`√`, `⁻¹`) rather than `<sup>` tags or LaTeX commands. Equations get their
own `.eq`/`.math-work` block rather than being buried in a sentence, and
multi-step derivations show one step per line (see rule 5), never one
compressed inline expression.

If a future page genuinely needs real LaTeX rendering, that's a new shared
capability (add MathJax/KaTeX to the shared assets per rule 17), not
something to reach for ad hoc on one page.

## 7. Use one coherent running example

Choose one domain or analogy for the page and reuse it throughout — don't
introduce an unrelated scenario for every concept. The specific domain
doesn't matter (movies, houses, exam scores, ecommerce, images, delivery
routes, music — anything works); consistency does. The chosen example should
be able to carry the full dependency chain the page teaches (e.g. for linear
algebra: features/vectors → dot product → similarity → matrices → matrix
multiplication → norms → dimensionality reduction → models). This way the
learner is only ever absorbing one new thing at a time — a new concept, not a
new concept *and* a new scenario.

## 8. Extend the running example progressively

Don't reveal the whole example at once. Introduce a single item first and
explain every component; reuse that same object for the next operation with
another already-introduced object; combine multiple already-understood
objects into the next larger structure; then operate on that structure. The
learner should meet complexity only after its components are already
familiar.

## 9. Introduce one conceptual jump at a time

Avoid sections that simultaneously introduce a new formula, new notation, a
new dataset, a new algorithm, and a new operation all at once. Prefer: known
example + one new concept + worked example = new understanding, then move to
the next concept.

## 10. Every concept follows the same learning pattern

For each concept, in this order: **What is it** (concise definition) → **why
do we need it** (the problem it addresses) → **intuition** (the simplest
useful mental model) → **formula/formal representation** (proper notation) →
**worked example** (using the page's running example where possible) →
**step-by-step solution** (every meaningful intermediate step) →
**interpretation** (what the result tells us) → **when would I use this**
(practical usage) → **interview takeaway** (what to remember or be able to
explain). Keep this structure consistent across the whole page.

## 11. Separate concise reading from deeper explanation

The main page must stay concise. Don't solve comprehensiveness by putting
every possible explanation directly in the main text — use progressive
disclosure instead (a term shown inline with a `?` / expandable that opens
into: formal definition, why it's defined that way, the intuitive picture, a
worked calculation, and its relationship to nearby concepts). This gives
"concise by default, comprehensive on demand," not "concise because
important explanations were cut."

## 12. Never use future concepts to explain current concepts

If concept B requires concept A, teach A first — never explain A by leaning
on unexplained knowledge of B. E.g. don't explain norms by saying "these are
what Ridge and Lasso use" before norms have been explained; teach vector →
magnitude → norm → L1/L2 → regularization → Ridge/Lasso in that order, then
connect them.

## 13. Explicitly connect concepts after both are understood

Once two concepts have each been independently explained, show their
relationship as an explicit chain (e.g. L1 norm → measures sum of absolute
coefficient values → used as a regularization penalty → pushes some
coefficients to exactly zero → Lasso). Connections should strengthen
understanding, never introduce an unexplained dependency.

## 14. Prefer examples over repeated prose

If another paragraph would just restate an idea, replace it with a worked
example instead: full substitution, every intermediate step, then one
sentence on what the final number means. That teaches more than several
paragraphs of abstract discussion.

## 15. Balance example complexity

Examples must be complex enough to show the real mechanism but simple enough
to work by hand. Avoid examples so trivial the operation becomes invisible
(e.g. an identity matrix for a determinant example). Avoid examples so large
that arithmetic obscures the concept. Choose the smallest example that
exposes the important behavior.

## 16. Never compress away the thing being taught

It's fine to skip mechanical steps the learner already knows. It is not fine
to skip the operation currently being taught. While teaching determinants,
the expansion must be visible; while teaching matrix multiplication, the
row-by-column products must be visible; while teaching gradient descent, at
least one parameter-update calculation must be visible; while teaching cosine
similarity, the dot product, norms, substitution, and division must all be
visible. Remove irrelevant detail, never instructional detail.

## 17. Reuse existing components before building new ones

Before adding any interactive element — accordion, tab, modal, button,
popup/glossary term, live-recompute figure, comparison block, worked-math
block, etc. — check `learning/ml-interview-prep/parts/10-linear-algebra.html`
and the shared assets it draws from
(`learning/ml-interview-prep/assets/app.css`, `code.js`, `viz.js`,
`glossary.js`, `app.js`) for an existing pattern, and reuse that class,
markup, or JS hook verbatim rather than inventing a new one.

`10-linear-algebra.html` is the fullest reference implementation of the
site's shared components:

- `.concept-brief` — what / why / when
- `.deep` + `.deep-q` (`?`) — progressive-disclosure accordion
- `.math-work` + `.step` + `.meaning` + `.legend` — step-by-step derivation block
- `.m` — inline math expression; `.matrix` / `.matrix.vector` — a matrix or vector as nested spans; `.eq` — a standalone display equation
- `.bridge` — an explicit dependency chain
- `.box` variants (`.analogy`, `.mem`, `.trap`) — callouts
- `.interview-line` — the closing takeaway
- `.terms.two` / `.term` — side-by-side comparison of two options
- `.pip` (basic / mid / senior) + `.tldr` — the Questions section
- `.checkpoint` — closing recall list
- `.gl` — glossary popup
- `.partnav` — prev/next navigation

Only introduce a genuinely new UI pattern if nothing in the shared assets
covers the need — and when you do, add it to the shared `assets/` files, not
inline in one page, so the next page can reuse it too.

This rule governs markup/UI reuse only. It doesn't change rules 1–16, which
are markup-agnostic and apply no matter how a given page is structured.

## Mandatory content invariants

Content isn't finished when it reads well — it's finished when you've walked
this list against the actual text you wrote and every box holds. Treat an
unchecked item as a required fix, not a note to self:

- [ ] Concepts appear in prerequisite order.
- [ ] No unexplained technical term is used.
- [ ] No unexplained symbol or notation is introduced.
- [ ] New concepts do not depend on concepts introduced later.
- [ ] Numbers have contextual meaning.
- [ ] Important results are derived rather than asserted.
- [ ] Mathematical operations include meaningful intermediate steps.
- [ ] Mathematics uses the site's math markup (`.m`/`.matrix`/`.eq`), not raw LaTeX or plain prose.
- [ ] Matrices and vectors are visually represented as matrices and vectors.
- [ ] One coherent running example is reused where practical.
- [ ] The example becomes progressively more sophisticated rather than constantly changing.
- [ ] Each section explains what the concept means, why it matters, and when to use it.
- [ ] Main-page content remains concise.
- [ ] Additional depth is available through `?` / expandable explanations.
- [ ] No verbosity is being used as a substitute for a good example.
- [ ] No conciseness is being achieved by skipping reasoning.
- [ ] Any new interactive element reuses an existing shared-asset pattern, or was added to the shared assets rather than inlined.

## Final standard

A learner should never encounter: "Where did that term come from?", "What
does that symbol mean?", "Why did they choose that number?", "How did they
get that answer?", "Why are we doing this?", "When would I ever use this?",
"How does this connect to what I just learned?"

The content should make the progression feel inevitable: known idea → new
idea → worked example → step-by-step derivation → interpretation →
application.

The governing principle: **never make the learner infer what the teaching
material itself should have taught.**
