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

  var LS = {
    get: function (k, d) { try { var v = localStorage.getItem(k); return v === null ? d : v; }
                           catch (e) { return d; } },
    set: function (k, v) { try { localStorage.setItem(k, v); } catch (e) {} }
  };

  /* ---------------------------------------------------------------
     THEME — applied on every page, shell or part.
     --------------------------------------------------------------- */
  function applyTheme(t) { document.documentElement.setAttribute('data-theme', t); }
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

  function initPage(root) {
    initTabs(root);
    if (window.MLIPViz) window.MLIPViz.init(root);
  }

  /* ---------------------------------------------------------------
     Are we the shell, or a standalone part?
     --------------------------------------------------------------- */
  var nav = document.getElementById('nav');

  if (!nav) {
    /* ----- standalone part page ----- */
    document.body.classList.add('standalone');
    document.addEventListener('DOMContentLoaded', function () { initPage(document); });
    if (document.readyState !== 'loading') initPage(document);

    // let the shell drive theme + drill when we are inside its iframe
    window.addEventListener('message', function (e) {
      var d = e.data || {};
      if (d.mlip === 'theme') applyTheme(d.value);
      if (d.mlip === 'drill') document.body.classList.toggle('drill', !!d.value);
      if (d.mlip === 'expand') {
        var qs = document.querySelectorAll('details.q');
        for (var i = 0; i < qs.length; i++) qs[i].open = !!d.value;
      }
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

  /* ---------- progress ---------- */
  function doneSet() {
    try { return new Set(JSON.parse(LS.get('mlip.done', '[]'))); } catch (e) { return new Set(); }
  }
  function saveDone(s) { LS.set('mlip.done', JSON.stringify(Array.from(s))); }
  function paintProgress() {
    var s = doneSet();
    MODULES.forEach(function (m) { navLinks[m.id].classList.toggle('done', s.has(m.id)); });
    if (progEl) progEl.textContent = s.size + '/' + MODULES.length + ' done';
    var btn = document.getElementById('markbtn');
    if (btn) {
      var cur = (location.hash || '#00').slice(1);
      btn.classList.toggle('on', s.has(cur));
      btn.textContent = s.has(cur) ? '✓ done' : 'mark done';
    }
  }

  /* ---------- loading a module ---------- */
  var textCache = {};   // id -> plain text, for search
  var current = null;

  function byId(id) {
    for (var i = 0; i < MODULES.length; i++) if (MODULES[i].id === id) return MODULES[i];
    return null;
  }

  function show(id) {
    var m = byId(id);
    if (!m) { m = MODULES[0]; id = m.id; }
    current = id;
    for (var k in navLinks) navLinks[k].classList.toggle('on', k === id);
    if (titleEl) titleEl.textContent = m.g + '  •  ' + m.t;
    document.title = m.t + ' — ML Interview Prep';
    document.body.classList.remove('navopen');

    if (IFRAME_MODE) {
      wrap.hidden = true; frame.hidden = false;
      frame.src = 'parts/' + m.f;
      frame.onload = function () { pushToFrame(); };
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
    if (v) {  // collapse everything so nothing is pre-revealed
      var qs = (IFRAME_MODE ? [] : wrap.querySelectorAll('details.q'));
      for (var i = 0; i < qs.length; i++) qs[i].open = false;
    }
    pushToFrame();
  }

  /* ---------- search ---------- */
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

  function filterNav(term) {
    var t = term.trim().toLowerCase();
    var shownGroups = {};
    MODULES.forEach(function (m) {
      var a = navLinks[m.id];
      var hit = !t || a.dataset.k.indexOf(t) >= 0 ||
                (textCache[m.id] || '').toLowerCase().indexOf(t) >= 0;
      a.classList.toggle('hidden', !hit);
      if (hit) shownGroups[m.g] = 1;
    });
    var hs = nav.querySelectorAll('.grp');
    for (var i = 0; i < hs.length; i++)
      hs[i].classList.toggle('hidden', !!t && !shownGroups[hs[i].dataset.grp]);
  }

  if (qbox) {
    qbox.addEventListener('input', function () {
      var t = qbox.value;
      filterNav(t);
      // the box lives in the topbar but filters the sidebar, so on narrow screens
      // the results sit behind a closed drawer — open it while a term is active
      if (window.matchMedia('(max-width: 900px)').matches)
        document.body.classList.toggle('navopen', !!t.trim());
      if (t.length >= 3) fetchAllText().then(function () { filterNav(qbox.value); });
    });
  }

  /* ---------- buttons ---------- */
  function bind(id, fn) { var e = document.getElementById(id); if (e) e.addEventListener('click', fn); }

  bind('themebtn', function () {
    theme = (theme === 'dark') ? 'light' : 'dark';
    LS.set('mlip.theme', theme); applyTheme(theme); pushToFrame();
  });
  bind('drillbtn', function () { setDrill(!drill); });
  bind('markbtn', function () {
    var s = doneSet();
    if (s.has(current)) s.delete(current); else s.add(current);
    saveDone(s); paintProgress();
  });
  bind('expandbtn', function () {
    if (IFRAME_MODE) {
      try { frame.contentWindow.postMessage({ mlip: 'expand', value: true }, '*'); } catch (e) {}
      return;
    }
    var qs = wrap.querySelectorAll('details.q');
    var anyClosed = false;
    for (var i = 0; i < qs.length; i++) if (!qs[i].open) anyClosed = true;
    for (var j = 0; j < qs.length; j++) qs[j].open = anyClosed;
  });
  bind('menu', function () { document.body.classList.toggle('navopen'); });
  bind('prevbtn', function () { step(-1); });
  bind('nextbtn', function () { step(1); });

  function step(d) {
    var i = 0;
    for (var j = 0; j < MODULES.length; j++) if (MODULES[j].id === current) i = j;
    var n = Math.min(MODULES.length - 1, Math.max(0, i + d));
    location.hash = '#' + MODULES[n].id;
  }

  /* ---------- keyboard ---------- */
  document.addEventListener('keydown', function (e) {
    var tag = (e.target.tagName || '').toLowerCase();
    if (tag === 'input' || tag === 'textarea') {
      if (e.key === 'Escape') { e.target.value = ''; filterNav(''); e.target.blur(); }
      return;
    }
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
  show((location.hash || '#00').slice(1));
  paintProgress();
})();
