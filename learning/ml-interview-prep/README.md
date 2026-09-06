# ML Interview Prep

**87 modules across two tracks, one site.**

- **The recall track (43 modules, hand-written HTML)** — maths from zero -> classical ML -> deep
  learning -> transformers/GPT -> MLOps & production -> 6 mock interview rounds -> six end-to-end
  system-design walkthroughs. 284 question accordions, 23 embedded visualisations.
- **The build track (44 pages, generated)** — `learning/llm-from-scratch/` rendered to HTML by that
  folder's `build-site.mjs`. Nav groups `H` to `K`.

Both share one nav and one search box, so searching *attention* returns the recall module, the
build lab, and the reference module together.

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

Static files, no framework, nothing for Vercel to build — it serves the folder as-is.

**The one ordering rule: regenerate the build track before deploying, or the site ships the
previous version of those 44 pages.**

```bash
cd learning/llm-from-scratch
node build-site.mjs --deploy       # generate, then vercel deploy --prod
```

`--deploy` exists so the order cannot be forgotten. The two halves separately, if you need them:

```bash
node learning/llm-from-scratch/build-site.mjs   # regenerate parts/llm-*.html + assets/build-modules.js
cd learning/ml-interview-prep && vercel deploy --prod
```

Editing only hand-written parts (`parts/[0-9]*.html`, `assets/app.*`)? `vercel deploy --prod` on its
own is enough — the generator is idempotent, so re-running it changes nothing.

### The build track is generated — do not edit it here

`parts/llm-*.html` and `assets/build-modules.js` are output. The source is the markdown in
`learning/llm-from-scratch/`; edit that and re-run the generator. The generator writes only into
this folder and never deletes: rename a markdown file and it prints the now-stale HTML page it left
behind, for you to remove by hand if you want it gone.

`index.html` loads `assets/build-modules.js` before `app.js`, and `app.js` appends
`window.MLIP_BUILD` to its own `MODULES` table. If the file is missing, the shell simply shows the
43 hand-written modules.

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
