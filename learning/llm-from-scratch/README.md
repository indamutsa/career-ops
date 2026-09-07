# LLM from scratch — a build-first curriculum

Build a ~3.4M-parameter language model from an empty file, train it on an M3 Pro, instruction-tune
it, and teach it to reason with RL against a verifier you wrote. Then use it as the subject of
every concept in 28 reference modules.

## → [**START-HERE.md**](START-HERE.md) ←

That file is the entry point: what the page ids mean, how to set up the workspace, the first lesson
to open, and the order to do everything in. Read it first.

```bash
cd learning/llm-from-scratch
mkdir -p code notes data checkpoints
pip install -r requirements.txt
python3 code/check_env.py
```

### The rest of the tree

| Path | What it is |
|------|------------|
| [`START-HERE.md`](START-HERE.md) | The entry point. Start here. |
| [`PLAN.md`](PLAN.md) | The dated schedule — four sprints, a hard gate at the end of each |
| [`modules/build/`](modules/build/README.md) | **The curriculum.** Eleven lessons, B0 → B10 |
| [`SYLLABUS.md`](SYLLABUS.md) | Index of the 28 deep-dive reference modules in `modules/` |
| [`GLOSSARY.md`](GLOSSARY.md) | 175 terms, each with where you'd meet it in production |
| [`code/`](code/README.md) | Lab code, one file per lesson. Written by hand, not generated. |
| [`notes/`](notes/README.md) | Measurements, one file per lesson. B8 and B9 read these back. |
| `data/`, `checkpoints/` | Corpora and weights — gitignored, regenerable |

## Reading it as a website

The same markdown is published at **https://ml-interview-prep-taupe.vercel.app** (nav groups `H` to
`K`), alongside the recall-track course, so it is readable from a phone. `build-site.mjs` generates
it; the markdown here stays the source of truth.

```bash
node build-site.mjs              # regenerate the HTML into ../ml-interview-prep/
node build-site.mjs --deploy     # regenerate, then vercel deploy --prod
node build-site.mjs --check      # dry run: convert everything, write nothing
```

`notes/`, `code/`, `data/` and `checkpoints/` are excluded on purpose — the site is public and those
are your own lab measurements. Deployment details, including how to relink the Vercel project on a
fresh clone, are in [`../ml-interview-prep/README.md`](../ml-interview-prep/README.md).
