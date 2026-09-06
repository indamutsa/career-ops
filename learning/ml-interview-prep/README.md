# ML Interview Prep

43 modules: maths from zero -> classical ML -> deep learning -> transformers/GPT ->
MLOps & production -> 6 mock interview rounds -> six end-to-end system-design walkthroughs.
284 question accordions, 23 embedded visualisations, no build step, works offline.

## Three ways to read it

**1. Live, from anywhere** — deployed as a static Vercel site:

-> **https://ml-interview-prep-taupe.vercel.app** <-

**2. Local server** — same behaviour as the live site:

```bash
cd learning/ml-interview-prep && python3 -m http.server 8000
# then open http://localhost:8000
```

**3. Double-click `index.html`** — works, but degraded. Chrome blocks `fetch()` on `file://`, so the
shell falls back to one iframe per module and says so in a banner. Search and drill mode are then
scoped to the open module instead of all of them.

Over `http(s)://` the shell fetches each part and injects it into the page, which is what makes
search and drill mode work **across all modules at once**. That is the only reason to prefer a
server, and it is why the Vercel deployment needs no changes to the HTML.

---

## Deploying

Static files, no framework, no build step. Vercel serves the folder as-is.

```bash
cd learning/ml-interview-prep
vercel deploy --prod          # redeploy after any edit
```

| | |
|---|---|
| Project | `ml-interview-prep` (Vercel account `indamutsa`) |
| Production URL | https://ml-interview-prep-taupe.vercel.app |
| Config | `.vercelignore` only — no `vercel.json`, nothing to build |

### If `vercel deploy` asks to set up a new project

`.vercel/` holds the project link and is gitignored, so it is absent on a fresh clone. Deploying
then creates a **second, unrelated project** instead of updating this one. Relink first:

```bash
vercel link --yes --project ml-interview-prep
```

### First deployment, for the record

```bash
vercel deploy --prod --yes --name ml-interview-prep
```

### What `.vercelignore` excludes

- `parts/65-dsa.html` — the DSA round was removed from the course (covered elsewhere). The file is
  still on disk but no longer referenced by the nav, so it is kept out of the deployment and 404s
  on the live site.
- `README.md`, `src` — repo-only, not part of the served site.

### Notes

- The production URL is **public**; anyone with the link can read it. Vercel's Deployment Protection
  can put a password or your Vercel login in front of it if that is not wanted.
- The only network dependency in the course itself is a Google Fonts `@import`. Everything else is
  local, so the site degrades to substituted fonts rather than breaking.
