/* ===================================================================
   code.js — syntax highlighting + a copy button for every code block.

   Hand-written, no dependency, no CDN: the rest of this course works
   with the network off and a highlighter is not a good enough reason to
   change that. It covers the five languages the two tracks actually
   use — python, bash, json, yaml, sql — and leaves anything else as
   plain text with the copy button still attached.

   Exposes window.MLIPCode.init(root); app.js calls it from initPage(),
   so it runs in the shell and on standalone part pages alike.
   =================================================================== */
(function () {
  'use strict';

  /* ---------------------------------------------------------------
     Tokenisers. Each is an ordered list of [class, regex]; the first
     pattern that matches at the cursor wins, so strings and comments
     must come before keywords or a keyword inside a string lights up.
     --------------------------------------------------------------- */

  var KW = {
    python: ('False None True and as assert async await break class continue def del elif else ' +
      'except finally for from global if import in is lambda nonlocal not or pass raise return ' +
      'try while with yield match case self').split(' '),
    bash: ('if then else elif fi for while do done case esac function return in export local ' +
      'source echo cd set unset trap exit read shift').split(' '),
    sql: ('select from where group by order having join left right inner outer on as insert into ' +
      'values update set delete create table drop alter index with distinct limit offset union all ' +
      'case when then else end and or not null is asc desc count sum avg min max over partition')
      .split(' '),
    json: ['true', 'false', 'null'],
    yaml: ['true', 'false', 'null', 'yes', 'no']
  };

  var BUILTIN = ('print len range enumerate zip open sum min max abs int float str list dict set ' +
    'tuple bool type isinstance super property staticmethod classmethod sorted reversed map filter ' +
    'any all round pow divmod repr format hash id iter next getattr setattr hasattr').split(' ');

  function words(list) {
    return new RegExp('^(?:' + list.join('|') + ')\\b');
  }

  var RULES = {
    python: [
      ['c', /^#[^\n]*/],
      ['s', /^(?:[rbfu]|rb|br|fr|rf)?"""[\s\S]*?"""/i],
      ['s', /^(?:[rbfu]|rb|br|fr|rf)?'''[\s\S]*?'''/i],
      ['s', /^(?:[rbfu]|rb|br|fr|rf)?"(?:\\.|[^"\\\n])*"/i],
      ['s', /^(?:[rbfu]|rb|br|fr|rf)?'(?:\\.|[^'\\\n])*'/i],
      ['d', /^@[A-Za-z_][\w.]*/],
      ['k', words(KW.python)],
      ['b', words(BUILTIN)],
      ['f', /^[A-Za-z_]\w*(?=\s*\()/],
      ['n', /^\d[\w.]*/],
      ['o', /^(?:\*\*|\/\/|[-+*/%=<>!&|^~@]+)/]
    ],
    bash: [
      ['c', /^#[^\n]*/],
      ['s', /^"(?:\\.|[^"\\])*"/],
      ['s', /^'[^']*'/],
      ['v', /^\$(?:\{[^}]*\}|[A-Za-z_]\w*|[0-9@*#?])/],
      ['k', words(KW.bash)],
      ['f', /^(?:[a-z][\w.-]*)(?=\s|$)/],
      ['n', /^\d+\b/],
      ['o', /^(?:&&|\|\||[|<>=;&]+)/]
    ],
    json: [
      ['a', /^"(?:\\.|[^"\\])*"(?=\s*:)/],
      ['s', /^"(?:\\.|[^"\\])*"/],
      ['k', words(KW.json)],
      ['n', /^-?\d[\d.eE+-]*/],
      ['o', /^[{}[\],:]/]
    ],
    yaml: [
      ['c', /^#[^\n]*/],
      ['a', /^[A-Za-z_][\w.-]*(?=\s*:)/],
      ['s', /^"(?:\\.|[^"\\])*"/],
      ['s', /^'[^']*'/],
      ['k', words(KW.yaml)],
      ['n', /^-?\d[\d.]*\b/],
      ['o', /^[-:|>]/]
    ],
    sql: [
      ['c', /^--[^\n]*/],
      ['s', /^'(?:''|[^'])*'/],
      ['s', /^"(?:[^"])*"/],
      ['k', new RegExp('^(?:' + KW.sql.join('|') + ')\\b', 'i')],
      ['n', /^\d[\d.]*\b/],
      ['o', /^(?:<>|[-+*/%=<>(),;.]+)/]
    ]
  };

  var ALIAS = { py: 'python', sh: 'bash', shell: 'bash', zsh: 'bash', console: 'bash',
                yml: 'yaml', jsonl: 'json', text: null, txt: null };

  var esc = function (s) {
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  };

  function highlight(src, lang) {
    var rules = RULES[lang];
    if (!rules) return esc(src);
    var out = '', i = 0;
    while (i < src.length) {
      var rest = src.slice(i), matched = false;
      for (var r = 0; r < rules.length; r++) {
        var m = rules[r][1].exec(rest);
        if (m && m[0].length) {
          out += '<span class="t' + rules[r][0] + '">' + esc(m[0]) + '</span>';
          i += m[0].length;
          matched = true;
          break;
        }
      }
      if (matched) continue;
      // no rule applies: emit one character and move on, so an unknown
      // construct degrades to plain text instead of stalling the loop.
      out += esc(src[i]);
      i++;
    }
    return out;
  }

  /* ---------------------------------------------------------------
     Copy button. The clipboard API needs a secure context, which
     https and localhost both are; opening index.html straight off the
     disk is not, so fall back to the old selection trick there.
     --------------------------------------------------------------- */
  function copy(text) {
    if (navigator.clipboard && window.isSecureContext)
      return navigator.clipboard.writeText(text);
    return new Promise(function (resolve, reject) {
      var ta = document.createElement('textarea');
      ta.value = text;
      ta.setAttribute('readonly', '');
      ta.style.cssText = 'position:fixed;top:-1000px;opacity:0';
      document.body.appendChild(ta);
      ta.select();
      var ok = false;
      try { ok = document.execCommand('copy'); } catch (e) { ok = false; }
      document.body.removeChild(ta);
      ok ? resolve() : reject(new Error('copy blocked'));
    });
  }

  function decorate(pre) {
    if (pre.dataset.ready) return;
    pre.dataset.ready = '1';

    var code = pre.querySelector('code');
    if (!code) return;

    var raw = code.textContent;
    var cls = (code.className.match(/lang(?:uage)?-([\w+#-]+)/) || [, ''])[1].toLowerCase();
    var lang = ALIAS.hasOwnProperty(cls) ? ALIAS[cls] : cls;

    if (RULES[lang]) {
      code.innerHTML = highlight(raw, lang);
      pre.classList.add('hl');
    }

    var bar = document.createElement('div');
    bar.className = 'codebar';

    if (lang) {
      var tag = document.createElement('span');
      tag.className = 'lang';
      tag.textContent = lang;
      bar.appendChild(tag);
    }

    var btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'copy';
    btn.textContent = 'copy';
    btn.setAttribute('aria-label', 'Copy this code block');
    btn.addEventListener('click', function () {
      copy(raw).then(function () {
        btn.textContent = 'copied';
        btn.classList.add('ok');
      }, function () {
        btn.textContent = 'select it';
        btn.classList.add('bad');
      });
      setTimeout(function () {
        btn.textContent = 'copy';
        btn.classList.remove('ok', 'bad');
      }, 1400);
    });
    bar.appendChild(btn);

    pre.appendChild(bar);
  }

  window.MLIPCode = {
    init: function (root) {
      var pres = (root || document).querySelectorAll('pre');
      for (var i = 0; i < pres.length; i++) decorate(pres[i]);
    }
  };
})();
