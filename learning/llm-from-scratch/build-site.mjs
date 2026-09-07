#!/usr/bin/env node
/* ===================================================================
   build-site.mjs — render learning/llm-from-scratch/*.md as HTML parts
   inside the ml-interview-prep course, so one deployment serves both
   tracks behind one nav and one search box.

   The markdown stays the single source of truth. Everything this
   script writes is disposable output: delete it and re-run.

   Usage
     node build-site.mjs              build
     node build-site.mjs --deploy     build, then `vercel deploy --prod`
     node build-site.mjs --check      build into memory only, report

   Outputs (into ../ml-interview-prep/)
     parts/llm-*.html          one standalone page per markdown file
     assets/build-modules.js   the nav/search registry the shell reads

   Never writes outside ../ml-interview-prep/ and never deletes.
   Sources under notes/, code/, data/ and checkpoints/ are excluded on
   purpose — they are private lab measurements and the site is public.
   =================================================================== */

import { readFileSync, writeFileSync, readdirSync, existsSync, mkdirSync } from 'node:fs';
import { dirname, join, basename, relative, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { execFileSync } from 'node:child_process';

const SRC  = dirname(fileURLToPath(import.meta.url));
const SITE = resolve(SRC, '..', 'ml-interview-prep');
const PARTS = join(SITE, 'parts');
const ARGS = new Set(process.argv.slice(2));
const DRY = ARGS.has('--check');

/* -------------------------------------------------------------------
   1. WHICH FILES, IN WHICH ORDER, UNDER WHICH NAV GROUP
   ------------------------------------------------------------------- */

const NAV_TRACK    = 'H · LLM track';
const NAV_LABS     = 'I · Build labs';
const NAV_CONCEPTS = 'J · LLM concepts';
const NAV_REF      = 'K · Reference';

const mdIn = (dir) =>
  existsSync(join(SRC, dir))
    ? readdirSync(join(SRC, dir)).filter((f) => f.endsWith('.md')).sort()
    : [];

/** `05a-encoder-decoder.md` -> `05a`; `00-foundations.md` -> `00`. */
const numPrefix = (f) => (f.match(/^(\d+[a-z]?)-/) || [, ''])[1];
/** `04-attention.md` -> `attention` (used in the output filename). */
const slugOf = (f) => basename(f, '.md').replace(/^\d+[a-z]?-/, '').toLowerCase();

const sources = [];
const push = (g, id, rel) => sources.push({ g, id, rel, abs: join(SRC, rel) });

push(NAV_TRACK, 'L0', 'START-HERE.md');
push(NAV_TRACK, 'L1', 'PLAN.md');
push(NAV_TRACK, 'L2', 'SYLLABUS.md');
push(NAV_TRACK, 'L3', 'modules/build/README.md');

for (const f of mdIn('modules/build')) {
  if (f === 'README.md') continue;                       // already placed above
  push(NAV_LABS, 'B' + Number(numPrefix(f)), `modules/build/${f}`);
}
for (const f of mdIn('modules')) {
  push(NAV_CONCEPTS, 'M' + numPrefix(f), `modules/${f}`);
}

push(NAV_REF, 'LG', 'GLOSSARY.md');

for (const s of sources) {
  if (!existsSync(s.abs)) { console.error(`missing source: ${s.rel}`); process.exit(1); }
  s.out = `llm-${s.id.toLowerCase()}-${slugOf(basename(s.rel))}.html`;
}

/** source path (repo-relative to SRC) -> nav id, for rewriting .md links. */
const byPath = new Map(sources.map((s) => [s.rel, s]));

/* -------------------------------------------------------------------
   2. MARKDOWN -> HTML

   A small, deliberate subset: exactly what these 45 files use. Verified
   against the corpus — no nested lists, no task lists, no raw HTML
   outside code fences, no LaTeX.
   ------------------------------------------------------------------- */

const esc = (s) =>
  s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');

const STOP = new Set(('the a an and or of to in on for is are was were be been it its this that with '
  + 'as at by from you your we our what why how when where which not but if then than so do does '
  + 'can will one two three into out up down over under about after before more most some any each '
  + 'no yes his her their they them he she i me my us also only just very much many'
).split(' '));

/** Resolve a markdown link target to shell HTML, or plain text if unmapped. */
function linkHtml(text, href, fromRel) {
  if (/^https?:\/\//i.test(href))
    return `<a href="${href}" target="_blank" rel="noopener">${text}</a>`;
  if (href.startsWith('#')) return text;

  const path = href.split('#')[0];
  const target = relative(SRC, resolve(SRC, dirname(fromRel), path)).split('\\').join('/');
  const hit = byPath.get(target);
  if (hit) return `<a href="../index.html#${hit.id}" target="_top">${text}</a>`;

  // code/, data/, notes/ — real files in the repo, deliberately not deployed.
  return `<span class="faint" title="${esc(target)}">${text}</span>`;
}

function inline(s, fromRel) {
  const codes = [];
  s = s.replace(/`([^`]+)`/g, (_m, c) => `\u0000${codes.push(c) - 1}\u0000`);
  s = esc(s);
  s = s.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, (_m, t, h) => linkHtml(t, h, fromRel));
  s = s.replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>');
  s = s.replace(/(^|[^*\w])\*([^*\n]+)\*(?![*\w])/g, '$1<em>$2</em>');
  s = s.replace(/\u0000(\d+)\u0000/g, (_m, i) => `<code>${esc(codes[+i])}</code>`);
  return s;
}

function splitRow(line) {
  return line.replace(/^\||\|$/g, '')
    .split(/(?<!\\)\|/)
    .map((c) => c.replace(/\\\|/g, '|').trim());
}

/**
 * Convert one markdown document.
 * Returns { title, lede, body, headings } — the first `# ` line becomes the
 * page title and the paragraph after it becomes the lede, matching how the
 * hand-written course parts are laid out.
 */
function convert(md, src) {
  const lines = md.replace(/\r\n?/g, '\n').split('\n');
  const out = [];
  const headings = [];
  let title = null, lede = null;
  let i = 0, section = 0;

  const para = [];
  const flushPara = () => {
    if (!para.length) return;
    const text = inline(para.join(' ').trim(), src.rel);
    para.length = 0;
    if (title && lede === null) { lede = text; return; }   // first paragraph -> lede
    out.push(`<p>${text}</p>`);
  };

  while (i < lines.length) {
    const line = lines[i];

    /* fenced code */
    if (/^```/.test(line)) {
      flushPara();
      const lang = line.slice(3).trim().replace(/[^a-z0-9+-]/gi, '');
      const buf = [];
      i++;
      while (i < lines.length && !/^```/.test(lines[i])) buf.push(lines[i++]);
      i++;
      const cls = lang ? ` class="lang-${lang}"` : '';
      out.push(`<pre><code${cls}>${esc(buf.join('\n'))}</code></pre>`);
      continue;
    }

    /* heading */
    const h = line.match(/^(#{1,6})\s+(.*)$/);
    if (h) {
      flushPara();
      const text = inline(h[2].trim(), src.rel);
      if (h[1].length === 1 && !title) { title = text; i++; continue; }
      let level = Math.max(2, h[1].length);
      if (level === 2) {
        section++;
        headings.push(h[2].trim());
        out.push(`<h2 data-n="${src.id}.${section}">${text}</h2>`);
      } else {
        headings.push(h[2].trim());
        out.push(`<h${level}>${text}</h${level}>`);
      }
      i++;
      continue;
    }

    /* horizontal rule */
    if (/^(-{3,}|\*{3,}|_{3,})\s*$/.test(line)) {
      flushPara(); out.push('<hr>'); i++; continue;
    }

    /* table */
    if (/^\s*\|/.test(line) && i + 1 < lines.length && /^\s*\|[\s:|-]+\|?\s*$/.test(lines[i + 1])) {
      flushPara();
      const head = splitRow(line);
      i += 2;
      const rows = [];
      while (i < lines.length && /^\s*\|/.test(lines[i])) rows.push(splitRow(lines[i++]));
      const th = head.map((c) => `<th>${inline(c, src.rel)}</th>`).join('');
      const tb = rows.map((r) =>
        '<tr>' + r.map((c) => `<td>${inline(c, src.rel)}</td>`).join('') + '</tr>').join('\n');
      out.push(`<div class="tw"><table>\n<thead><tr>${th}</tr></thead>\n<tbody>\n${tb}\n</tbody></table></div>`);
      continue;
    }

    /* blockquote -> the course's note box */
    if (/^>\s?/.test(line)) {
      flushPara();
      const buf = [];
      while (i < lines.length && /^>\s?/.test(lines[i])) buf.push(lines[i++].replace(/^>\s?/, ''));
      out.push(`<div class="box note"><p>${inline(buf.join(' ').trim(), src.rel)}</p></div>`);
      continue;
    }

    /* lists */
    const isUl = /^\s*[-*+]\s+/.test(line);
    const isOl = /^\s*\d+[.)]\s+/.test(line);
    if (isUl || isOl) {
      flushPara();
      const tag = isUl ? 'ul' : 'ol';
      const re  = isUl ? /^\s*[-*+]\s+/ : /^\s*\d+[.)]\s+/;
      const items = [];
      while (i < lines.length) {
        if (re.test(lines[i])) { items.push(lines[i].replace(re, '')); i++; continue; }
        // a continuation line is an indented, non-blank line under the last item
        if (items.length && /^\s+\S/.test(lines[i]) && !/^\s*[-*+]\s|^\s*\d+[.)]\s/.test(lines[i])) {
          items[items.length - 1] += ' ' + lines[i].trim(); i++; continue;
        }
        break;
      }
      out.push(`<${tag}>\n` +
        items.map((t) => `  <li>${inline(t.trim(), src.rel)}</li>`).join('\n') +
        `\n</${tag}>`);
      continue;
    }

    /* blank line / paragraph text */
    if (!line.trim()) { flushPara(); i++; continue; }
    para.push(line.trim());
    i++;
  }
  flushPara();

  return { title: title || src.id, lede, body: out.join('\n\n'), headings };
}

/* -------------------------------------------------------------------
   3. PAGE TEMPLATE — identical skeleton to the hand-written parts
   ------------------------------------------------------------------- */

const stripTags = (s) => s.replace(/<[^>]+>/g, '');

/** Titles restate their own number ("B4 — Attention…", "Module 03 — Attention");
    the nav already shows the id, so drop the prefix rather than say it twice. */
const label = (s) =>
  esc(stripTags(s.title)
    .replace(new RegExp('^(?:Module\\s+)?' + s.id.replace(/^[A-Z]/, '') +
                        '\\s*[\u2014\u2013-]\\s*', 'i'), '')
    .replace(new RegExp('^' + s.id + '\\s*[\u2014\u2013-]\\s*', 'i'), ''));

function page(src, doc, prev, next) {
  const nav = [];
  if (prev) nav.push(`  <a href="../index.html#${prev.id}" target="_top">` +
    `<span class="k">Back</span>← ${prev.id} · ${label(prev)}</a>`);
  if (next) nav.push(`  <a href="../index.html#${next.id}" target="_top">` +
    `<span class="k">Next</span>${next.id} · ${label(next)} →</a>`);

  return `<!doctype html>
<html lang="en" data-theme="dark">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${esc(stripTags(doc.title))} — ML Interview Prep</title>
<link rel="stylesheet" href="../assets/app.css">
</head>
<body>

<!-- GENERATED by learning/llm-from-scratch/build-site.mjs from ${src.rel}
     Edit the markdown, not this file. Re-run the script to regenerate. -->

<p class="crumb">${esc(src.g)} · ${esc(src.id)} · <code>${esc(src.rel)}</code></p>
<h1>${doc.title}${doc.lede ? `\n  <span class="lede">${doc.lede}</span>` : ''}
</h1>

${doc.body}

<div class="partnav">
${nav.join('\n')}
</div>

<script src="../assets/code.js"></script>
<script src="../assets/viz.js"></script>\n<script src="../assets/glossary.js"></script>
<script src="../assets/app.js"></script>
</body>
</html>
`;
}

/* -------------------------------------------------------------------
   4. BUILD
   ------------------------------------------------------------------- */

const docs = sources.map((s) => ({ src: s, doc: convert(readFileSync(s.abs, 'utf8'), s) }));
for (let n = 0; n < docs.length; n++) docs[n].src.title = docs[n].doc.title;

if (!DRY && !existsSync(PARTS)) mkdirSync(PARTS, { recursive: true });

let bytes = 0;
for (let n = 0; n < docs.length; n++) {
  const { src, doc } = docs[n];
  const html = page(src, doc, docs[n - 1]?.src, docs[n + 1]?.src);
  bytes += Buffer.byteLength(html);
  if (!DRY) writeFileSync(join(PARTS, src.out), html);
}

/* search keywords: title + every heading, minus stopwords and short tokens */
function keywords(doc) {
  const seen = new Set();
  const words = (stripTags(doc.title) + ' ' + doc.headings.join(' '))
    .toLowerCase().split(/[^a-z0-9+]+/);
  for (const w of words) if (w.length > 2 && !STOP.has(w)) seen.add(w);
  return Array.from(seen).slice(0, 70).join(' ');
}

/** The sidebar is narrow; long lesson titles are trimmed at a word boundary. */
function navTitle(src) {
  const t = label(src);
  if (t.length <= 52) return t;
  const cut = t.slice(0, 52);
  return cut.slice(0, cut.lastIndexOf(' ')).replace(/[,;:—–-]$/, '') + '…';
}

const registry = docs.map(({ src, doc }) =>
  `  { g: ${JSON.stringify(src.g)}, id: ${JSON.stringify(src.id)}, ` +
  `f: ${JSON.stringify(src.out)},\n    t: ${JSON.stringify(navTitle(src))}, ` +
  `k: ${JSON.stringify(keywords(doc))} }`).join(',\n');

const modulesJs = `/* ===================================================================
   GENERATED by learning/llm-from-scratch/build-site.mjs — do not edit.
   Nav + search entries for the LLM-from-scratch track. index.html loads
   this before app.js; app.js appends it to its own MODULES table.
   =================================================================== */
window.MLIP_BUILD = [
${registry}
];
`;

if (!DRY) writeFileSync(join(SITE, 'assets', 'build-modules.js'), modulesJs);

/* orphan check — this script never deletes, so say what went stale */
const expected = new Set(docs.map((d) => d.src.out));
const orphans = existsSync(PARTS)
  ? readdirSync(PARTS).filter((f) => f.startsWith('llm-') && !expected.has(f))
  : [];

console.log(`${docs.length} pages, ${(bytes / 1024).toFixed(0)} KB${DRY ? ' (not written)' : ''}`);
for (const g of [NAV_TRACK, NAV_LABS, NAV_CONCEPTS, NAV_REF])
  console.log(`  ${g.padEnd(18)} ${docs.filter((d) => d.src.g === g).length}`);
if (orphans.length)
  console.log(`\nstale, no longer generated (remove by hand if you want them gone):\n  ` +
    orphans.join('\n  '));

if (ARGS.has('--deploy')) {
  console.log('\nvercel deploy --prod');
  execFileSync('vercel', ['deploy', '--prod'], { cwd: SITE, stdio: 'inherit' });
}
