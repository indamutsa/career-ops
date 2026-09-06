# LLM from scratch — a build-first curriculum

Build a ~3.4M-parameter language model from an empty file, train it on an M3 Pro, instruction-tune
it, and teach it to reason with RL against a verifier you wrote. Then use it as the subject of
every concept in 28 reference modules.

## → [**START-HERE.md**](START-HERE.md) ←

That file is the entry point: environment check, the first lesson to open, and the order to do
everything in. Read it first.

Two other files, both reference — not starting points:

- [`SYLLABUS.md`](SYLLABUS.md) — index of the 28 deep-dive modules
- [`GLOSSARY.md`](GLOSSARY.md) — 175 terms, each with where you'd meet it in production

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
