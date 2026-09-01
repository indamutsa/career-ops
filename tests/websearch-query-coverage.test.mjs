// tests/websearch-query-coverage.test.mjs — a `websearch` entry in portals.yml
// used to hand the agent its hand-written `scan_query` verbatim, and nothing
// ever reconciled that string with `title_filter.positive`. The two drift: a
// title_filter covering MLOps / LLM Engineer / ML Platform / Data Engineer
// still produced `"Machine Learning" OR "AI Infrastructure" OR "Research
// Engineer"`, so entire configured categories were never searched at any
// websearch company (#3612).
//
// The failure this guards is silent — a too-narrow query still returns
// results, so it looks like it worked. Hence the coverage assertion below is
// exhaustive over title_filter.positive rather than spot-checking a keyword.
// The legacy cases are paired with it because the fix must not change
// behaviour for configs that have no title_filter or no resolvable site scope.

import { pass, fail } from './helpers.mjs';
import { buildWebSearchQueries, extractSiteScope, siteScopeFromUrl } from '../scan.mjs';

const eq = (actual, expected, msg) => {
  const a = JSON.stringify(actual);
  const e = JSON.stringify(expected);
  if (a === e) pass(msg);
  else fail(`${msg}\n     got  ${a}\n     want ${e}`);
};

// ── site scope extraction ──────────────────────────────────────────
eq(extractSiteScope('site:openai.com/careers "ML" OR x'), 'site:openai.com/careers', 'extractSiteScope: single site: token');
eq(extractSiteScope('site:a.com OR site:b.com foo'), 'site:a.com OR site:b.com', 'extractSiteScope: preserves multi-site OR group');
eq(extractSiteScope('site:a.com foo site:a.com'), 'site:a.com', 'extractSiteScope: dedupes repeated token');
eq(extractSiteScope('no scope here'), '', 'extractSiteScope: none present');
eq(extractSiteScope(null), '', 'extractSiteScope: null-safe');

eq(siteScopeFromUrl('https://careers.factorialhr.com/'), 'site:careers.factorialhr.com', 'siteScopeFromUrl: strips trailing slash');
eq(siteScopeFromUrl('https://openai.com/careers'), 'site:openai.com/careers', 'siteScopeFromUrl: keeps path');
eq(siteScopeFromUrl('not-a-url'), '', 'siteScopeFromUrl: unparseable input');
eq(siteScopeFromUrl(''), '', 'siteScopeFromUrl: empty input');

// ── legacy behaviour must be byte-identical ────────────────────────
eq(buildWebSearchQueries({ scan_query: 'site:x.com foo' }, undefined), ['site:x.com foo'],
  'legacy: no title_filter returns the authored query unchanged');
eq(buildWebSearchQueries({ scan_query: 'site:x.com foo' }, { positive: [] }), ['site:x.com foo'],
  'legacy: empty title_filter.positive returns the authored query unchanged');
eq(buildWebSearchQueries({ careers_url: 'not-a-url', scan_query: 'plain text' }, { positive: ['MLOps'] }), ['plain text'],
  'legacy: unresolvable scope never invents an unscoped query');
eq(buildWebSearchQueries({}, { positive: ['MLOps'] }), [],
  'entry with neither scan_query nor careers_url yields no queries');

// ── query construction ─────────────────────────────────────────────
eq(buildWebSearchQueries({ careers_url: 'https://x.com/j' }, { positive: ['word:MLOps', 'AI Engineer'] }),
  ['site:x.com/j MLOps OR "AI Engineer"'],
  'word: prefix stripped (matcher syntax, not search syntax); phrases quoted');
eq(buildWebSearchQueries({ careers_url: 'https://x.com/j' }, { positive: ['stem:Architect'] }),
  ['site:x.com/j Architect'],
  'stem: prefix stripped');
eq(buildWebSearchQueries({ careers_url: 'https://x.com/j' }, { positive: ['a', 'b', 'c'] }, { chunkSize: 2 }),
  ['site:x.com/j a OR b', 'site:x.com/j c'],
  'keywords chunked so search engines do not truncate a long OR-chain');
eq(buildWebSearchQueries({ scan_query: 'site:authored.com/x "Old Keyword"', careers_url: 'https://ignored.com' },
  { positive: ['MLOps'] }),
  ['site:authored.com/x MLOps'],
  'authored site: scope wins over careers_url (it can encode board paths)');

// ── the actual regression: exhaustive coverage ─────────────────────
const titleFilter = {
  positive: ['Machine Learning Engineer', 'ML Engineer', 'MLOps', 'ML Platform', 'AI Platform',
             'AI Infrastructure', 'ML Infrastructure', 'LLM Engineer', 'Applied AI',
             'AI Engineer', 'ML Architect', 'Research Engineer', 'Data Engineer'],
};
const entry = { name: 'OpenAI', careers_url: 'https://openai.com/careers',
                scan_query: 'site:openai.com/careers "Machine Learning" OR "AI Infrastructure" OR "Research Engineer"' };
const joined = buildWebSearchQueries(entry, titleFilter).join(' ');
const uncovered = titleFilter.positive.filter(k => !joined.includes(k));
if (uncovered.length === 0) pass('every title_filter.positive keyword reaches the emitted websearch queries');
else fail(`title_filter keywords never searched for: ${uncovered.join(', ')}`);

// The pre-fix query is the thing that must NOT survive.
if (!joined.includes('"Machine Learning" OR "AI Infrastructure" OR "Research Engineer"')) {
  pass('the stale hand-written keyword subset no longer defines the search');
} else {
  fail('emitted queries still carry the drifted hand-written keyword subset verbatim');
}
