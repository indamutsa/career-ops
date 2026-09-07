/* ===================================================================
   ML Interview Prep — shell + shared page behaviour.
   One file, loaded by index.html AND by every parts/*.html.
   It detects which it is by looking for #nav.
   No dependencies, no network, works from file:// and http://.
   =================================================================== */
(function () {
  'use strict';

  /* ---------------------------------------------------------------
     THE MODULE TABLE — single source of truth for nav + search.
     --------------------------------------------------------------- */
  var MODULES = [
    { g: 'Start',       id: '00', f: '00-start.html',              t: 'How to use this',
      k: 'orientation answer formula spaced repetition schedule plan' },

    { g: 'A · Maths',   id: '10', f: '10-linear-algebra.html',     t: 'Linear algebra',
      k: 'vector matrix dot product transformation rank eigenvector eigenvalue svd norm l1 l2 cosine similarity' },
    { g: 'A · Maths',   id: '11', f: '11-calculus.html',           t: 'Calculus & minima',
      k: 'derivative slope gradient minimum maximum saddle chain rule jacobian hessian convex second derivative' },
    { g: 'A · Maths',   id: '12', f: '12-probability.html',        t: 'Probability & stats',
      k: 'random variable distribution expectation variance bayes likelihood mle map independence clt stochastic sampling gaussian bernoulli' },
    { g: 'A · Maths',   id: '13', f: '13-information.html',        t: 'Information theory',
      k: 'entropy cross entropy kl divergence js divergence mutual information perplexity surprise bits nats' },
    { g: 'A · Maths',   id: '14', f: '14-optimization.html',       t: 'Optimisation',
      k: 'gradient descent sgd momentum adam learning rate convex lagrange multiplier kkt dual laplace approximation ridge lasso constraint' },

    { g: 'B · Classical', id: '20', f: '20-learning-problem.html', t: 'The learning problem',
      k: 'supervised unsupervised self-supervised split generalisation bias variance overfitting cross validation curse dimensionality leakage' },
    { g: 'B · Classical', id: '21', f: '21-linear-models.html',    t: 'Linear & logistic',
      k: 'linear regression logistic sigmoid odds glm ridge lasso elastic net multicollinearity coefficients' },
    { g: 'B · Classical', id: '22', f: '22-trees-ensembles.html',  t: 'Trees & ensembles',
      k: 'decision tree cart gini entropy bagging random forest boosting adaboost gradient boosting xgboost lightgbm' },
    { g: 'B · Classical', id: '23', f: '23-svm-knn-bayes.html',    t: 'SVM, kNN, Naive Bayes',
      k: 'support vector machine margin hinge kernel trick dual knn naive bayes laplace smoothing' },
    { g: 'B · Classical', id: '24', f: '24-unsupervised.html',     t: 'Unsupervised & anomaly',
      k: 'kmeans gmm em pca eigen tsne umap dbscan isolation forest anomaly outlier clustering silhouette' },
    { g: 'B · Classical', id: '25', f: '25-metrics.html',          t: 'Metrics & imbalance',
      k: 'confusion matrix precision recall f1 roc auc pr auc calibration log loss brier threshold cost sensitive smote imbalance' },
    { g: 'B · Classical', id: '26', f: '26-interpretability.html', t: 'Interpretability',
      k: 'shap lime feature importance permutation pdp ice counterfactual global local shapley' },

    { g: 'C · Deep',    id: '30', f: '30-neural-nets.html',        t: 'Neural networks',
      k: 'perceptron activation relu sigmoid tanh gelu universal approximation backpropagation initialisation vanishing exploding batchnorm layernorm dropout residual' },
    { g: 'C · Deep',    id: '31', f: '31-cnn.html',                t: 'Convolutional nets',
      k: 'convolution kernel filter stride padding pooling receptive field equivariance resnet vgg alexnet efficientnet' },
    { g: 'C · Deep',    id: '32', f: '32-rnn-attention.html',      t: 'RNN, LSTM, attention',
      k: 'recurrent rnn lstm gru gate seq2seq encoder decoder bottleneck bahdanau attention teacher forcing' },
    { g: 'C · Deep',    id: '33', f: '33-nlp-classic.html',        t: 'Classic NLP',
      k: 'tokenisation bag of words tfidf ngram word2vec glove embedding cosine similarity ner pos stemming lemmatisation' },
    { g: 'C · Deep',    id: '34', f: '34-rl.html',                 t: 'Reinforcement learning',
      k: 'mdp state action reward policy value bellman q learning dqn policy gradient reinforce actor critic ppo exploration exploitation' },

    { g: 'D · Transformers', id: '40', f: '40-attention.html',     t: 'Attention',
      k: 'query key value scaled dot product softmax multi head self cross causal mask quadratic' },
    { g: 'D · Transformers', id: '41', f: '41-block.html',         t: 'The transformer block',
      k: 'embedding positional encoding sinusoidal rope alibi residual stream layernorm rmsnorm ffn swiglu prenorm postnorm parameter count' },
    { g: 'D · Transformers', id: '42', f: '42-families.html',      t: 'BERT, GPT, T5',
      k: 'bert masked language model gpt causal t5 encoder decoder bidirectional autoregressive why decoder only' },
    { g: 'D · Transformers', id: '43', f: '43-pretraining-scaling.html', t: 'Pretraining & scaling',
      k: 'pretraining corpus bpe next token scaling laws kaplan chinchilla compute optimal emergence context length' },
    { g: 'D · Transformers', id: '44', f: '44-posttraining.html',  t: 'SFT, RLHF, DPO, GRPO',
      k: 'instruction tuning sft reward model rlhf ppo dpo rlvr grpo chain of thought distillation alignment' },
    { g: 'D · Transformers', id: '45', f: '45-inference.html',     t: 'Inference & efficiency',
      k: 'kv cache gqa mqa quantisation int8 int4 lora qlora flashattention speculative decoding prefill decode continuous batching throughput latency' },
    { g: 'D · Transformers', id: '46', f: '46-llm-systems.html',   t: 'RAG, agents, eval',
      k: 'rag retrieval vector database chunking embedding agent tool use react hallucination guardrail llm as judge evaluation' },

    { g: 'E · MLOps',   id: '50', f: '50-lifecycle-data.html',     t: 'Data & ingestion',
      k: 'kafka backpressure synchronous asynchronous streaming batch cdc data contract idempotency exactly once partition consumer lag' },
    { g: 'E · MLOps',   id: '51', f: '51-feature-store.html',      t: 'Feature stores',
      k: 'feast online offline store point in time correctness training serving skew feature view entity materialisation' },
    { g: 'E · MLOps',   id: '52', f: '52-tracking-orchestration.html', t: 'MLflow & Kubeflow',
      k: 'mlflow tracking registry kubeflow airflow dag dvc reproducibility experiment artifact lineage' },
    { g: 'E · MLOps',   id: '53', f: '53-deployment.html',         t: 'Deployment patterns',
      k: 'canary blue green shadow ab test bandit rollback autoscaling container model server triton kserve latency budget' },
    { g: 'E · MLOps',   id: '54', f: '54-monitoring.html',         t: 'Monitoring & drift',
      k: 'data drift concept drift psi population stability index ks kolmogorov smirnov kl divergence delayed label alerting slo observability' },
    { g: 'E · MLOps',   id: '55', f: '55-governance.html',         t: 'Governance & risk',
      k: 'model card lineage fairness bias privacy gdpr adversarial robustness cost audit' },

    { g: 'F · Rounds',  id: '60', f: '60-round1-math.html',        t: 'Round 1 — Maths',
      k: 'interview round mock rapid fire maths' },
    { g: 'F · Rounds',  id: '61', f: '61-round2-classical.html',   t: 'Round 2 — Classical ML',
      k: 'interview round mock classical metrics' },
    { g: 'F · Rounds',  id: '62', f: '62-round3-deep.html',        t: 'Round 3 — Deep learning',
      k: 'interview round mock deep learning cnn rnn' },
    { g: 'F · Rounds',  id: '63', f: '63-round4-transformers.html',t: 'Round 4 — Transformers',
      k: 'interview round mock transformer llm gpt' },
    { g: 'F · Rounds',  id: '64', f: '64-round5-mlops.html',       t: 'Round 5 — MLOps',
      k: 'interview round mock mlops production' },
    { g: 'F · Rounds',  id: '66', f: '66-behavioral.html',         t: 'Behavioural & judgment',
      k: 'behavioural star story judgment tradeoff disagreement failure' },

    { g: 'G · Walkthrough', id: '70', f: '70-fraud-detection.html', t: 'Fraud detection, end to end',
      k: 'system design fraud detection kafka feast mlflow canary psi latency cost matrix imbalance feedback loop' },
    { g: 'G · Walkthrough', id: '71', f: '71-rag-assistant.html', t: 'RAG assistant over internal docs',
      k: 'system design rag retrieval augmented generation chunking embedding vector database hybrid bm25 rerank permissions citation hallucination evaluation' },
    { g: 'G · Walkthrough', id: '72', f: '72-recommendations.html', t: 'Recommendation & ranking',
      k: 'system design recommender recommendation ranking candidate generation two tower embedding retrieval implicit feedback position bias cold start ab test' },
    { g: 'G · Walkthrough', id: '73', f: '73-llm-serving.html', t: 'LLM serving platform',
      k: 'system design llm serving inference platform kv cache continuous batching vllm gpu autoscaling multi tenancy cost per token routing quantisation' },
    { g: 'G · Walkthrough', id: '74', f: '74-posttraining-pipeline.html', t: 'Domain post-training pipeline',
      k: 'system design post training fine tuning sft dpo grpo rlvr preference data eval harness regression suite model registry release' },
    { g: 'G · Walkthrough', id: '75', f: '75-agent-system.html', t: 'Production agent with tools',
      k: 'system design agent tool use function calling error compounding sandbox permissions trace evaluation cost latency human in the loop' }
  ];

  /* The llm-from-scratch track is generated, not hand-written: its nav entries
     live in assets/build-modules.js, written by that track's build-site.mjs.
     Absent (a part page, or a build that was never run) it is simply skipped.
     Every module carries `tr`, the track it belongs to, which drives the tabs. */
  var BUILT = window.MLIP_BUILD || [];
  MODULES.forEach(function (m) { m.tr = 'course'; });
  BUILT.forEach(function (m) { m.tr = 'llm'; });
  if (BUILT.length) MODULES = MODULES.concat(BUILT);

  var TRACKS = [
    { k: 'course', label: 'Recall',    hint: 'maths → production, no code to write' },
    { k: 'llm',    label: 'Build LLM', hint: 'the from-scratch track, you write the code' }
  ];

  var LS = {
    get: function (k, d) { try { var v = localStorage.getItem(k); return v === null ? d : v; }
                           catch (e) { return d; } },
    set: function (k, v) { try { localStorage.setItem(k, v); } catch (e) {} }
  };

  /* ---------------------------------------------------------------
     THEME — applied on every page, shell or part.
     --------------------------------------------------------------- */
  /* The button shows where the click *goes*, not where you are: on a dark
     board it offers the sun. A word in a bar of icons reads as a label. */
  var THEME_ICON = {
    dark:  '<svg viewBox="0 0 24 24" aria-hidden="true"><circle cx="12" cy="12" r="4"/>' +
           '<path d="M12 2.6v2.3M12 19.1v2.3M4.3 4.3l1.6 1.6M18.1 18.1l1.6 1.6' +
           'M2.6 12h2.3M19.1 12h2.3M4.3 19.7l1.6-1.6M18.1 5.9l1.6-1.6"/></svg>',
    light: '<svg viewBox="0 0 24 24" aria-hidden="true">' +
           '<path d="M20.3 14.4A8.5 8.5 0 0 1 9.6 3.7 8.5 8.5 0 1 0 20.3 14.4z"/></svg>'
  };
  function applyTheme(t) {
    document.documentElement.setAttribute('data-theme', t);
    var b = document.getElementById('themebtn');
    if (b) b.innerHTML = THEME_ICON[t] || '';
  }
  var theme = LS.get('mlip.theme', 'dark');
  applyTheme(theme);

  /* ---------------------------------------------------------------
     PAGE BEHAVIOUR — runs on shell content AND standalone parts.
     --------------------------------------------------------------- */
  function initTabs(root) {
    var groups = root.querySelectorAll('.tabs');
    for (var i = 0; i < groups.length; i++) {
      (function (tabs) {
        if (tabs.dataset.ready) return;
        tabs.dataset.ready = '1';
        var btns = tabs.querySelectorAll('.tabbar button');
        var pans = tabs.querySelectorAll('.panel');
        function show(n) {
          for (var j = 0; j < btns.length; j++)
            btns[j].setAttribute('aria-selected', j === n ? 'true' : 'false');
          for (var k = 0; k < pans.length; k++)
            pans[k].classList.toggle('on', k === n);
        }
        for (var j = 0; j < btns.length; j++)
          (function (n) { btns[n].addEventListener('click', function () { show(n); }); })(j);
        show(0);
      })(groups[i]);
    }
  }

  /* ---------------------------------------------------------------
     ACCORDIONS — one open at a time, page-wide.

     Reading two answers side by side is how you end up recognising an
     answer instead of recalling it, which is the opposite of what this
     is for. `toggle` does not bubble, so listen in the capture phase.
     The expand-all button sets `bulk` to opt out for one pass.
     --------------------------------------------------------------- */
  var bulk = false;
  document.addEventListener('toggle', function (e) {
    var d = e.target;
    if (bulk || !d || d.tagName !== 'DETAILS' || !d.open) return;
    var all = document.querySelectorAll('details');
    for (var i = 0; i < all.length; i++)
      if (all[i] !== d && all[i].open && !all[i].contains(d)) all[i].open = false;
  }, true);

  /** Open or close many at once without the one-at-a-time rule firing. */
  function setAll(nodes, open) {
    bulk = true;
    for (var i = 0; i < nodes.length; i++) nodes[i].open = open;
    setTimeout(function () { bulk = false; }, 0);
  }

  function initPage(root) {
    initTabs(root);
    if (window.MLIPCode) window.MLIPCode.init(root);
    if (window.MLIPViz) window.MLIPViz.init(root);
    if (window.MLIPGloss) window.MLIPGloss.init(root);
  }

  /* ---------------------------------------------------------------
     Are we the shell, or a standalone part?
     --------------------------------------------------------------- */
  var nav = document.getElementById('nav');

  function initStandaloneCompletion() {
    if (window.parent !== window) return;
    if (document.body.dataset.completionReady) return;
    var filename = decodeURIComponent(location.pathname.split('/').pop() || '');
    var module = null;
    for (var i = 0; i < MODULES.length; i++) {
      if (MODULES[i].f === filename) { module = MODULES[i]; break; }
    }
    if (!module) return;
    document.body.dataset.completionReady = '1';

    var modal = document.createElement('div');
    modal.id = 'modal';
    modal.hidden = true;
    modal.innerHTML = '<div class="sheet" role="dialog" aria-modal="true" aria-labelledby="mtitle">' +
      '<h3 id="mtitle">Mark it done?</h3>' +
      '<p id="mbody" class="muted"></p>' +
      '<div class="row"><button id="mno" type="button">Cancel</button>' +
      '<button id="myes" type="button" class="primary">Mark done</button></div></div>';
    document.body.appendChild(modal);
    modal.querySelector('#mbody').textContent = '\u201c' + module.t + '\u201d gets a tick in the ' +
      'course sidebar and counts toward the track total. You can undo it there any time.';

    var prompted = false;
    function savedDone() {
      try { return new Set(JSON.parse(LS.get('mlip.done', '[]'))); }
      catch (e) { return new Set(); }
    }
    function close() { modal.hidden = true; }
    function checkBottom() {
      var root = document.scrollingElement || document.documentElement;
      var range = root.scrollHeight - root.clientHeight;
      if (prompted || range <= 40 || range - root.scrollTop > 8 || savedDone().has(module.id)) return;
      prompted = true;
      modal.hidden = false;
      modal.querySelector('#myes').focus();
    }
    modal.querySelector('#mno').addEventListener('click', close);
    modal.querySelector('#myes').addEventListener('click', function () {
      var done = savedDone();
      done.add(module.id);
      LS.set('mlip.done', JSON.stringify(Array.from(done)));
      close();
    });
    modal.addEventListener('click', function (event) { if (event.target === modal) close(); });
    document.addEventListener('keydown', function (event) {
      if (event.key === 'Escape' && !modal.hidden) close();
    });
    window.addEventListener('scroll', checkBottom, { passive: true });
    checkBottom();
  }

  if (!nav) {
    /* ----- standalone part page ----- */
    document.body.classList.add('standalone');
    function startStandalone() { initPage(document); initStandaloneCompletion(); }
    document.addEventListener('DOMContentLoaded', startStandalone);
    if (document.readyState !== 'loading') startStandalone();

    // let the shell drive theme + drill when we are inside its iframe
    window.addEventListener('message', function (e) {
      var d = e.data || {};
      if (d.mlip === 'theme') applyTheme(d.value);
      if (d.mlip === 'drill') document.body.classList.toggle('drill', !!d.value);
      if (d.mlip === 'expand') setAll(document.querySelectorAll('details.q, details.deep'), !!d.value);
    });
    return;
  }

  /* ================================================================
     SHELL
     ================================================================ */
  var IFRAME_MODE = (location.protocol === 'file:');
  document.body.classList.add('shell');

  var wrap    = document.getElementById('wrap');
  var frame   = document.getElementById('frame');
  var banner  = document.getElementById('banner');
  var qbox    = document.getElementById('q');
  var progEl  = document.getElementById('prog');
  var titleEl = document.getElementById('ptitle');

  if (IFRAME_MODE && banner) banner.hidden = false;

  /* ---------- build nav ---------- */
  var lastGroup = null, navLinks = {};
  MODULES.forEach(function (m) {
    if (m.g !== lastGroup) {
      var h = document.createElement('div');
      h.className = 'grp'; h.textContent = m.g; h.dataset.grp = m.g;
      nav.appendChild(h); lastGroup = m.g;
    }
    var a = document.createElement('a');
    a.href = '#' + m.id;
    a.dataset.id = m.id;
    a.dataset.k = (m.t + ' ' + m.k + ' ' + m.g).toLowerCase();
    a.innerHTML = '<span class="id">' + m.id + '</span><span class="tt"></span>';
    a.querySelector('.tt').textContent = m.t;
    nav.appendChild(a);
    navLinks[m.id] = a;
  });

  /* ---------- track tabs ---------- */
  var tbar     = document.getElementById('tracks');
  var hintEl   = document.getElementById('trackhint');
  var trackBtn = {};
  var track    = LS.get('mlip.track', 'course');
  if (!BUILT.length) track = 'course';

  if (tbar && BUILT.length) {
    TRACKS.forEach(function (t) {
      var n = MODULES.filter(function (m) { return m.tr === t.k; }).length;
      var b = document.createElement('button');
      b.type = 'button';
      b.setAttribute('role', 'tab');
      b.title = t.hint;
      b.innerHTML = '<span class="tl"></span><span class="tn">' + n + '</span>';
      b.querySelector('.tl').textContent = t.label;
      b.addEventListener('click', function () { setTrack(t.k, true); });
      tbar.appendChild(b);
      trackBtn[t.k] = b;
    });
  } else if (tbar) {
    tbar.hidden = true;
  }

  /* `jump` means the click came from the tab bar: land on that track's first
     module. Following a cross-track link switches the tab without jumping. */
  function setTrack(k, jump) {
    track = k;
    LS.set('mlip.track', k);
    for (var t in trackBtn)
      trackBtn[t].setAttribute('aria-selected', t === k ? 'true' : 'false');
    filterNav(qbox ? qbox.value : '');
    paintProgress();
    if (!jump) return;
    var cur = byId(current);
    if (cur && cur.tr === k) return;
    var first = MODULES.filter(function (m) { return m.tr === k; })[0];
    if (first) location.hash = '#' + first.id;
  }

  /* ---------- progress ---------- */
  function doneSet() {
    try { return new Set(JSON.parse(LS.get('mlip.done', '[]'))); } catch (e) { return new Set(); }
  }
  function saveDone(s) { LS.set('mlip.done', JSON.stringify(Array.from(s))); }
  function paintProgress() {
    var s = doneSet();
    MODULES.forEach(function (m) { navLinks[m.id].classList.toggle('done', s.has(m.id)); });
    var mine = MODULES.filter(function (m) { return m.tr === track; });
    var n = mine.filter(function (m) { return s.has(m.id); }).length;
    if (progEl) progEl.textContent = n + '/' + mine.length + ' done';
    var fab = document.getElementById('fabdone');
    if (fab) {
      var cur = (location.hash || '#00').slice(1), on = s.has(cur);
      fab.classList.toggle('on', on);
      fab.setAttribute('aria-pressed', on ? 'true' : 'false');
      fab.title = on ? 'Done \u2014 click to clear' : 'Mark this module done';
    }
  }

  /* ---------- loading a module ---------- */
  var textCache = {};   // id -> plain text, for search
  var current = null;
  var searchRevealUntil = 0;

  function byId(id) {
    for (var i = 0; i < MODULES.length; i++) if (MODULES[i].id === id) return MODULES[i];
    return null;
  }

  function show(id) {
    var m = byId(id);
    if (!m) { m = MODULES[0]; id = m.id; }
    current = id;
    if (m.tr !== track) setTrack(m.tr, false);
    for (var k in navLinks) navLinks[k].classList.toggle('on', k === id);
    if (titleEl) titleEl.textContent = m.g + '  •  ' + m.t;
    document.title = m.t + ' — ML Interview Prep';
    document.body.classList.remove('navopen');

    if (IFRAME_MODE) {
      wrap.hidden = true; frame.hidden = false;
      frame.src = 'parts/' + m.f;
      frame.onload = function () {
        pushToFrame();
        try {
          var doc = frame.contentDocument;
          var scroller = doc.scrollingElement || doc.documentElement;
          revealSearchMatch(doc.body, qbox ? qbox.value : '');
          var frameBottom = function () {
            var range = scroller.scrollHeight - scroller.clientHeight;
            maybePromptCompletion(range > 40 && range - scroller.scrollTop <= 8);
          };
          scroller.addEventListener('scroll', frameBottom, { passive: true });
          frameBottom();
        } catch (e) {}
      };
    } else {
      frame.hidden = true; wrap.hidden = false;
      wrap.innerHTML = '<p class="faint">Loading…</p>';
      fetch('parts/' + m.f).then(function (r) {
        if (!r.ok) throw new Error(r.status + ' ' + r.statusText);
        return r.text();
      }).then(function (html) {
        var doc = new DOMParser().parseFromString(html, 'text/html');
        wrap.innerHTML = doc.body.innerHTML;
        textCache[id] = (wrap.textContent || '').replace(/\s+/g, ' ');
        initPage(wrap);
        applyDrillLocal();
        document.getElementById('main').scrollTop = 0;
        revealSearchMatch(wrap, qbox ? qbox.value : '');
        paintScroll();
      }).catch(function (e) {
        wrap.innerHTML = '<h1>Not written yet</h1><p class="lead">' +
          '<code>parts/' + m.f + '</code> could not be loaded.</p>' +
          '<div class="box warn"><span class="lbl">Error</span><p>' + e.message + '</p>' +
          '<p class="small">If every module fails, you are probably not serving the folder. ' +
          'Run <code>python3 -m http.server 8000</code> here and open ' +
          '<code>http://localhost:8000</code>.</p></div>';
      });
    }
    paintProgress();
  }

  function pushToFrame() {
    if (!IFRAME_MODE || !frame.contentWindow) return;
    try {
      frame.contentWindow.postMessage({ mlip: 'theme', value: theme }, '*');
      frame.contentWindow.postMessage({ mlip: 'drill', value: drill }, '*');
    } catch (e) {}
  }

  /* ---------- drill mode ---------- */
  var drill = LS.get('mlip.drill', '0') === '1';
  function applyDrillLocal() { document.body.classList.toggle('drill', drill); }
  function setDrill(v) {
    drill = v; LS.set('mlip.drill', v ? '1' : '0');
    applyDrillLocal();
    var b = document.getElementById('drillbtn');
    if (b) { b.classList.toggle('on', v); }
    if (v)    // collapse everything so nothing is pre-revealed
      setAll(IFRAME_MODE ? [] : wrap.querySelectorAll('details.q'), false);
    pushToFrame();
  }

  /* ---------- search ---------- */
  function clearSearchHighlights(root) {
    if (!root) return;
    var marks = root.querySelectorAll('mark[data-search-hit]');
    for (var i = 0; i < marks.length; i++) {
      var mark = marks[i], parent = mark.parentNode;
      parent.replaceChild(mark.ownerDocument.createTextNode(mark.textContent), mark);
      parent.normalize();
    }
  }

  function revealSearchMatch(root, rawTerm) {
    if (!root) return 0;
    clearSearchHighlights(root);
    var term = (rawTerm || '').trim();
    if (term.length < 3) return 0;

    var doc = root.ownerDocument || document;
    var win = doc.defaultView || window;
    var walker = doc.createTreeWalker(root, win.NodeFilter.SHOW_TEXT);
    var nodes = [], node;
    while ((node = walker.nextNode())) {
      var parent = node.parentElement;
      if (!parent || parent.closest('script, style, svg, button, input, textarea, select, ' +
                                    'mark, [hidden], [aria-hidden="true"]')) continue;
      if (node.data.toLowerCase().indexOf(term.toLowerCase()) >= 0) nodes.push(node);
    }

    var first = null, count = 0, needle = term.toLowerCase();
    nodes.forEach(function (textNode) {
      if (count >= 80 || !textNode.parentNode) return;
      var source = textNode.data, lower = source.toLowerCase(), at = 0, hit;
      var fragment = doc.createDocumentFragment();
      while (count < 80 && (hit = lower.indexOf(needle, at)) >= 0) {
        if (hit > at) fragment.appendChild(doc.createTextNode(source.slice(at, hit)));
        var mark = doc.createElement('mark');
        mark.dataset.searchHit = '1';
        mark.textContent = source.slice(hit, hit + term.length);
        if (!first) { first = mark; mark.classList.add('current'); }
        fragment.appendChild(mark);
        count++;
        at = hit + term.length;
      }
      if (at < source.length) fragment.appendChild(doc.createTextNode(source.slice(at)));
      textNode.parentNode.replaceChild(fragment, textNode);
    });

    if (first) {
      var details = first.closest('details');
      if (details) details.open = true;
      searchRevealUntil = Date.now() + 250;
      win.requestAnimationFrame(function () {
        win.requestAnimationFrame(function () {
          first.scrollIntoView({ block: 'center', inline: 'nearest' });
        });
      });
    }
    return count;
  }

  function highlightCurrentSearch() {
    var term = qbox ? qbox.value : '';
    if (IFRAME_MODE) {
      try { revealSearchMatch(frame.contentDocument.body, term); } catch (e) {}
    } else {
      revealSearchMatch(wrap, term);
    }
  }

  var allFetched = false;
  function fetchAllText() {
    if (allFetched || IFRAME_MODE) return Promise.resolve();
    allFetched = true;
    return Promise.all(MODULES.map(function (m) {
      if (textCache[m.id] !== undefined) return null;
      return fetch('parts/' + m.f).then(function (r) { return r.ok ? r.text() : ''; })
        .then(function (h) {
          var d = new DOMParser().parseFromString(h, 'text/html');
          textCache[m.id] = (d.body.textContent || '').replace(/\s+/g, ' ');
        }).catch(function () { textCache[m.id] = ''; });
    }));
  }

  /* A search deliberately escapes the active tab: cross-track hits are the
     whole point of one search box over both tracks. The group headers still
     say which track each hit came from. */
  function filterNav(term) {
    var t = (term || '').trim().toLowerCase();
    var shownGroups = {};
    MODULES.forEach(function (m) {
      var a = navLinks[m.id];
      var hit = (t || m.tr === track) &&
                (!t || a.dataset.k.indexOf(t) >= 0 ||
                 (textCache[m.id] || '').toLowerCase().indexOf(t) >= 0);
      a.classList.toggle('hidden', !hit);
      if (hit) shownGroups[m.g] = 1;
    });
    var hs = nav.querySelectorAll('.grp');
    for (var i = 0; i < hs.length; i++)
      hs[i].classList.toggle('hidden', !shownGroups[hs[i].dataset.grp]);
    if (hintEl) hintEl.hidden = !(t && BUILT.length);
    for (var k in trackBtn) trackBtn[k].classList.toggle('muted', !!t);
  }

  if (qbox) {
    qbox.addEventListener('input', function () {
      var t = qbox.value;
      filterNav(t);
      if (!t.trim()) highlightCurrentSearch();
      // the box lives in the topbar but filters the sidebar, so on narrow screens
      // the results sit behind a closed drawer — open it while a term is active
      if (window.matchMedia('(max-width: 900px)').matches)
        document.body.classList.toggle('navopen', !!t.trim());
      if (t.length >= 3) fetchAllText().then(function () { filterNav(qbox.value); });
    });
    nav.addEventListener('click', function (event) {
      var link = event.target.closest('a[data-id]');
      if (link && link.dataset.id === current)
        setTimeout(highlightCurrentSearch, 0);
    });
  }

  /* ---------- buttons ---------- */
  function bind(id, fn) { var e = document.getElementById(id); if (e) e.addEventListener('click', fn); }

  bind('themebtn', function () {
    theme = (theme === 'dark') ? 'light' : 'dark';
    LS.set('mlip.theme', theme); applyTheme(theme); pushToFrame();
  });
  bind('drillbtn', function () { setDrill(!drill); });
  /* ---------- confirm sheet ----------
     Progress is the one piece of state the reader cannot undo by scrolling,
     and the button now sits under the thumb where it is easy to brush. So it
     asks once, in the reader's own words, before it changes anything. */
  var modal = document.getElementById('modal');
  var onYes = null;

  function ask(title, body, label, fn) {
    if (!modal) { fn(); return; }
    document.getElementById('mtitle').textContent = title;
    document.getElementById('mbody').textContent = body;
    document.getElementById('myes').textContent = label;
    onYes = fn;
    modal.hidden = false;
    document.getElementById('myes').focus();
  }
  function closeAsk() { if (modal) modal.hidden = true; onYes = null; }

  bind('mno', closeAsk);
  bind('myes', function () { var f = onYes; closeAsk(); if (f) f(); });
  if (modal) modal.addEventListener('click', function (e) { if (e.target === modal) closeAsk(); });

  function setDone(id, v) {
    var s = doneSet();
    if (v) s.add(id); else s.delete(id);
    saveDone(s); paintProgress();
  }

  function promptMarkDone(id) {
    var m = byId(id), name = m ? '\u201c' + m.t + '\u201d' : 'this module';
    ask('Mark it done?',
        name + ' gets a tick in the sidebar and counts toward the track total. You can undo it any time.',
        'Mark done', function () { setDone(id, true); });
  }

  bind('fabdone', function () {
    var m = byId(current), name = m ? '\u201c' + m.t + '\u201d' : 'this module';
    var id = current;
    if (doneSet().has(id))
      ask('Clear this one?',
          name + ' is marked done. Clearing it removes the tick and drops it from the count.',
          'Clear it', function () { setDone(id, false); });
    else
      promptMarkDone(id);
  });

  /* ---------- reading progress ---------- */
  var rbar = document.querySelector('#rprog i');
  var mainEl = document.getElementById('main');
  var bottomPrompted = {};
  function maybePromptCompletion(atBottom) {
    if (!atBottom || !current || bottomPrompted[current] ||
        Date.now() < searchRevealUntil || doneSet().has(current) ||
        (modal && !modal.hidden)) return;
    bottomPrompted[current] = true;
    promptMarkDone(current);
  }
  function paintScroll() {
    if (!rbar || !mainEl) return;
    var h = mainEl.scrollHeight - mainEl.clientHeight;
    rbar.style.width = (h > 40 ? Math.min(100, (mainEl.scrollTop / h) * 100) : 0) + '%';
    if (!IFRAME_MODE) maybePromptCompletion(h > 40 && h - mainEl.scrollTop <= 8);
  }
  if (mainEl) mainEl.addEventListener('scroll', paintScroll, { passive: true });
  bind('expandbtn', function () {
    if (IFRAME_MODE) {
      try { frame.contentWindow.postMessage({ mlip: 'expand', value: true }, '*'); } catch (e) {}
      return;
    }
    var qs = wrap.querySelectorAll('details.q, details.deep');
    var anyClosed = false;
    for (var i = 0; i < qs.length; i++) if (!qs[i].open) anyClosed = true;
    setAll(qs, anyClosed);
  });
  bind('menu', function () { document.body.classList.toggle('navopen'); });
  bind('prevbtn', function () { step(-1); });
  bind('nextbtn', function () { step(1); });

  function step(d) {
    var list = MODULES.filter(function (m) { return m.tr === track; });
    var i = 0;
    for (var j = 0; j < list.length; j++) if (list[j].id === current) i = j;
    var n = Math.min(list.length - 1, Math.max(0, i + d));
    if (list[n]) location.hash = '#' + list[n].id;
  }

  /* ---------- keyboard ---------- */
  document.addEventListener('keydown', function (e) {
    var tag = (e.target.tagName || '').toLowerCase();
    if (tag === 'input' || tag === 'textarea') {
      if (e.key === 'Escape') { e.target.value = ''; filterNav(''); e.target.blur(); }
      return;
    }
    if (e.key === 'Escape' && modal && !modal.hidden) { closeAsk(); return; }
    if (e.metaKey || e.ctrlKey || e.altKey) return;
    if (e.key === '/') { e.preventDefault(); if (qbox) qbox.focus(); }
    else if (e.key === 'j') step(1);
    else if (e.key === 'k') step(-1);
    else if (e.key === 'd') setDrill(!drill);
    else if (e.key === 't') { var b = document.getElementById('themebtn'); if (b) b.click(); }
  });

  /* ---------- routing ---------- */
  window.addEventListener('hashchange', function () { show((location.hash || '#00').slice(1)); });
  applyDrillLocal();
  var b0 = document.getElementById('drillbtn'); if (b0) b0.classList.toggle('on', drill);
  setTrack(track, false);
  show((location.hash || '#00').slice(1));
  paintProgress();
})();
