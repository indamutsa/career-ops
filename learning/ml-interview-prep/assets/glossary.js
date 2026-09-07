/* ===================================================================
   glossary.js — no term arrives naked.

   The failure this fixes: a sentence like "gradient clipping clips the
   L2 norm, weight decay penalises squared L2, and lasso penalises L1"
   is three undefined terms in a row. It is perfectly true and perfectly
   useless to anyone who has not already met them, and that pattern
   recurs across the course because each module was written by someone
   who had the whole map in their head.

   So: every term below carries a one-breath plain definition and the
   module where it is actually taught. MLIPGloss.init(root) then walks
   the rendered page, finds the FIRST bare occurrence of each term, and
   turns it into a chip you can click open in place. Nothing is
   hand-tagged, so this works on every page — including the generated
   ones — and a new page written tomorrow inherits it for free.

   A term is skipped on the page that defines it (it has its own term
   card there), and inside code, links and headings.
   =================================================================== */
window.MLIPGloss = (function () {
  'use strict';

  /* term : [ plain definition , where it is properly taught ] */
  var G = {
    /* ---- optimisation and training ---- */
    'gradient': ['The list of slopes — one per parameter — saying which way to nudge each weight to make the loss smaller, and how steeply. It is the direction of steepest increase, so training walks the other way.', '11 · Calculus'],
    'gradient clipping': ['If the gradient vector is longer than a set threshold (its L2 length, say 1.0), shrink the whole vector until it is exactly that long. Direction unchanged, size capped — so one freak batch cannot blow the weights up.', '14 · Optimisation'],
    'weight decay': ['Add the sum of the squared weights to the loss, so the optimiser pays a price for large weights and keeps them small unless the data really insists. Same thing as L2 regularisation.', '21 · Linear models'],
    'lasso': ['Linear regression with the sum of the absolute weights (L1) added to the loss. Because L1 has a corner at zero, the cheapest answer often sets a weight to exactly 0 — so it selects features as well as fitting them.', '21 · Linear models'],
    'ridge': ['Linear regression with the sum of squared weights (L2) added to the loss. It shrinks every weight towards zero without ever quite reaching it.', '21 · Linear models'],
    'regularisation': ['Anything you add to training that makes the model prefer a simpler answer, to stop it memorising the training set.', '20 · The learning problem'],
    'learning rate': ['How big a step to take in the direction the gradient points. Too small and training crawls; too large and it bounces out of the valley.', '14 · Optimisation'],
    'stochastic gradient descent': ['Gradient descent using a small random sample of the data for each step instead of all of it. "Stochastic" just means that random-sample part.', '14 · Optimisation'],
    'momentum': ['Keep a running average of recent gradients and step along that instead of the raw one, so the optimiser rolls through small bumps rather than rattling in them.', '14 · Optimisation'],
    'backpropagation': ['The chain rule applied backwards through the network: compute the loss, then hand each layer the slope of the loss with respect to its own outputs, layer by layer, until every weight knows its gradient.', '30 · Neural networks'],
    'loss': ['One number saying how wrong the model was on this data. Training is the search for parameters that make it small.', '14 · Optimisation'],
    'overfitting': ['The model has learned the training set rather than the pattern behind it: excellent on data it has seen, poor on data it has not.', '20 · The learning problem'],
    'epoch': ['One full pass through the training set.', '20 · The learning problem'],
    'mini-batch': ['The handful of examples processed together in one training step — typically 32 to 1,024.', '14 · Optimisation'],
    'convex': ['A loss surface shaped like a single bowl: one minimum, and any downhill walk reaches it. Neural networks are not convex, which is why initialisation and learning rate matter so much.', '14 · Optimisation'],
    'lagrange multiplier': ['A trick that turns "minimise this, subject to that constraint" into an ordinary unconstrained minimisation, by adding the constraint to the objective with an unknown price attached.', '14 · Optimisation'],

    /* ---- maths ---- */
    'norm': ['The length of a vector. L2 is straight-line length, L1 is the sum of absolute values, L∞ is the largest single component.', '10 · Linear algebra'],
    'dot product': ['Multiply two equal-length lists element by element and add the results. One number, measuring how much the two lists agree.', '10 · Linear algebra'],
    'cosine similarity': ['The dot product after dividing out both lengths, so only the angle between two vectors counts and not their size. Runs from −1 to 1.', '10 · Linear algebra'],
    'eigenvector': ['A direction a matrix does not turn — it only stretches or shrinks it. The stretch factor is the eigenvalue.', '10 · Linear algebra'],
    'rank': ['How many genuinely independent directions a matrix spans. Low rank means the columns repeat each other, so the matrix carries less information than its size suggests.', '10 · Linear algebra'],
    'jacobian': ['The table of every first derivative when a function has several inputs and several outputs.', '11 · Calculus'],
    'hessian': ['The table of second derivatives — the curvature of the loss surface, telling you whether a flat point is a valley, a peak or a saddle.', '11 · Calculus'],
    'entropy': ['The average surprise of a distribution — how many bits you need on average to say which outcome happened. Flat and unpredictable is high entropy; nearly certain is low.', '13 · Information theory'],
    'cross-entropy': ['The average surprise you suffer when you predict with one distribution and reality follows another. It is the standard classification loss for exactly that reason.', '13 · Information theory'],
    'kl divergence': ['How many extra bits your predicted distribution costs you compared with the true one. Zero when they match, and never negative.', '13 · Information theory'],
    'perplexity': ['The exponential of cross-entropy, read as "how many equally likely options the model was effectively choosing between". Perplexity 12 means it was about as unsure as a fair 12-sided die.', '13 · Information theory'],
    'likelihood': ['How probable the data you actually observed would be, under a given set of parameters. Training often means finding the parameters that make it largest.', '12 · Probability'],
    'bayes': ['The rule for updating a belief with evidence: posterior ∝ prior × likelihood.', '12 · Probability'],
    'expectation': ['The long-run average of a random quantity — each outcome weighted by how likely it is.', '12 · Probability'],
    'variance': ['How spread out a quantity is around its average.', '12 · Probability'],

    /* ---- classical ML ---- */
    'bias–variance': ['The trade-off between a model too rigid to fit the pattern (bias) and one so flexible it fits the noise (variance).', '20 · The learning problem'],
    'cross-validation': ['Split the training data k ways, train on k−1 parts and test on the held-out one, rotate, average. It buys a more trustworthy score from a small dataset.', '20 · The learning problem'],
    'leakage': ['Information from the future, or from the answer, sneaking into the features. It makes offline scores look wonderful and production scores collapse.', '20 · The learning problem'],
    'precision': ['Of the cases you flagged, what fraction were really positive. It is the cost of a false alarm.', '25 · Metrics'],
    'recall': ['Of the cases that really were positive, what fraction you caught. It is the cost of a miss.', '25 · Metrics'],
    'roc-auc': ['The chance that a randomly chosen positive scores above a randomly chosen negative. Reassuring but misleading when positives are rare.', '25 · Metrics'],
    'pr-auc': ['Precision plotted against recall, summarised as one number. The honest choice when positives are rare, because it ignores the huge easy negative class.', '25 · Metrics'],
    'calibration': ['Whether a predicted probability means what it says: of everything scored 0.7, about 70 % should actually be positive.', '25 · Metrics'],
    'class imbalance': ['One class hugely outnumbers the other — 1 fraud in 5,000 payments — so accuracy becomes meaningless and thresholds have to be chosen on cost.', '25 · Metrics'],
    'ensemble': ['Combine several models and use their consensus, because their mistakes are partly independent and averaging cancels some of them.', '22 · Trees and ensembles'],
    'bagging': ['Train many models on random resamples of the data and average them. It attacks variance.', '22 · Trees and ensembles'],
    'boosting': ['Train models in sequence, each one fitting the errors the previous ones left behind. It attacks bias.', '22 · Trees and ensembles'],
    'kernel trick': ['Compute what a dot product would be in a much bigger space, without ever building the vectors in that space.', '23 · SVM, kNN, Bayes'],
    'pca': ['Find the directions along which the data varies most and keep only the first few, so the data survives with fewer numbers.', '24 · Unsupervised'],
    'shap': ['Splits a single prediction into a per-feature contribution that adds up to the prediction, using a fair-share rule borrowed from game theory.', '26 · Interpretability'],

    /* ---- deep learning and transformers ---- */
    'activation function': ['The small non-linear function applied after each layer. Without one, stacking layers collapses back to a single matrix and the depth buys nothing.', '30 · Neural networks'],
    'dropout': ['During training, randomly switch off a fraction of the units each step, so the network cannot depend on any one of them.', '30 · Neural networks'],
    'batch normalisation': ['Rescale each feature across the batch to a steady mean and spread, which keeps the numbers flowing through a deep network in a usable range.', '30 · Neural networks'],
    'layer normalisation': ['Rescale each example across its own features rather than across the batch — which is what transformers use, because it does not depend on batch size.', '30 · Neural networks'],
    'residual connection': ['Add a layer’s input to its output, so the layer only has to learn the change. It is what lets networks be hundreds of layers deep.', '30 · Neural networks'],
    'vanishing gradient': ['Gradients shrink towards zero as they travel back through many layers, so the early layers stop learning.', '30 · Neural networks'],
    'receptive field': ['How much of the original input a single unit deep in the network can actually see.', '31 · CNNs'],
    'embedding': ['A learned vector standing for a discrete thing — a word, a user, a product — placed so that similar things sit near each other.', '33 · Classical NLP'],
    'tokenisation': ['Cutting text into the units the model actually consumes. Usually sub-word pieces, so rare words become several tokens rather than one unknown.', '33 · Classical NLP'],
    'logits': ['The raw, unnormalised scores a model emits before softmax turns them into probabilities.', '30 · Neural networks'],
    'softmax': ['Turns a list of arbitrary scores into positive numbers that add to 1, so they can be read as probabilities.', '30 · Neural networks'],
    'attention': ['Every position looks at every other position and takes a weighted average of what it finds, with the weights being dot products between a query and each key.', '40 · Attention'],
    'self-attention': ['Attention where the queries, keys and values all come from the same sequence — each token deciding which other tokens matter to it.', '40 · Attention'],
    'causal mask': ['Blocking every position from attending to positions after it, so a language model cannot read the answer it is meant to predict.', '40 · Attention'],
    'kv cache': ['The stored keys and values of every token generated so far, kept so that producing the next token costs one step rather than re-reading the whole sequence.', '45 · Inference'],
    'positional encoding': ['Extra information added to each token saying where it sits in the sequence, because attention on its own has no sense of order.', '41 · The transformer block'],
    'fine-tuning': ['Continuing to train an already-trained model on a smaller, more specific dataset.', '44 · Post-training'],
    'lora': ['Freeze the big model and train two small matrices whose product is added to each weight matrix, so a fine-tune costs a fraction of the memory.', '45 · Inference'],
    'quantisation': ['Storing weights in fewer bits — 8 or 4 instead of 16 — to fit a larger model into the same memory, at some cost in accuracy.', '45 · Inference'],
    'distillation': ['Train a small model to copy a large one’s outputs rather than the raw labels, so it inherits some of the larger model’s judgement.', '44 · Post-training'],
    'rlhf': ['Collect human preferences between model answers, fit a reward model to them, then use reinforcement learning to push the model towards what people preferred.', '44 · Post-training'],
    'rag': ['Retrieve relevant documents first, then hand them to the model as context, so answers rest on fetched text rather than memory alone.', '46 · LLM systems'],
    'hallucination': ['A fluent, confident, false answer. The model is producing plausible next tokens, not checking facts.', '46 · LLM systems'],

    /* ---- MLOps ---- */
    'drift': ['The live data has moved away from what the model was trained on, so yesterday’s accuracy stops being a promise about today.', '54 · Monitoring'],
    'psi': ['Population Stability Index: bucket a feature, compare the training and live proportions, and get one number for how far the distribution has moved. Roughly, below 0.1 is calm, above 0.25 needs attention.', '54 · Monitoring'],
    'canary': ['Send a small slice of live traffic — 1 %, then 5 % — to the new version, watch the metrics, and only then widen it.', '53 · Deployment'],
    'blue-green': ['Run two identical environments and switch all traffic from the old to the new in one move, so rollback is another switch rather than a redeploy.', '53 · Deployment'],
    'shadow': ['Send live traffic to the new model as well as the old one, but throw the new one’s answers away. You get production behaviour with zero production risk.', '53 · Deployment'],
    'feature store': ['A shared place where features are defined once and served both to training (historical values) and to production (current values), so the two cannot drift apart.', '51 · Feature store'],
    'point-in-time correctness': ['Building training rows using only the feature values that were actually knowable at that moment — never a value computed later.', '51 · Feature store'],
    'train/serve skew': ['The features at training time and at serving time are computed differently, so the model meets inputs in production it never saw in training.', '51 · Feature store'],
    'backpressure': ['A consumer that cannot keep up telling the producer to slow down, instead of quietly dropping messages or running out of memory.', '50 · Data lifecycle'],
    'idempotency': ['Doing the same operation twice leaves the same result as doing it once — the property that makes retries safe.', '50 · Data lifecycle'],
    'model registry': ['The catalogue of trained models with versions, metrics and lineage, so "which model is in production" has an answer.', '52 · Tracking'],
    'slo': ['A published target for a service — "99 % of scoring calls return within 80 ms" — that decides what counts as broken.', '54 · Monitoring'],

    /* ---- terms the maths modules lean on before anything defines them ---- */
    'curvature': ['How fast the slope itself is changing. Flat-bottomed valley: low curvature, a big step is safe. Sharp ravine: high curvature, the same step overshoots. It is the second derivative.', '11 · Calculus'],
    'condition number': ['The ratio of the steepest curvature to the shallowest. A large one means a long narrow valley, which is exactly the case plain gradient descent handles worst.', '11 · Calculus'],
    'second order': ['Any method that uses curvature (the second derivative), not just slope. More informed per step, and far more expensive — the curvature of an n-parameter model is an n × n table.', '11 · Calculus'],
    'second-order': ['Any method that uses curvature (the second derivative), not just slope. More informed per step, and far more expensive — the curvature of an n-parameter model is an n × n table.', '11 · Calculus'],
    'newton': ['Short for Newton\u2019s method: step straight to the bottom of the parabola that best matches the curve where you stand — slope divided by curvature. Very few steps, and unusable at scale, because it needs the full curvature table.', '14 · Optimisation'],
    "newton's method": ['Step straight to the bottom of the parabola that best matches the curve where you stand — slope divided by curvature. It converges in very few steps and is unusable at scale, because it needs the full curvature table.', '14 · Optimisation'],
    'trust region': ['A cap on how far one update may move you, on the grounds that your local picture of the landscape is only believable nearby. PPO is a trust region on a policy.', '14 · Optimisation'],
    'warmup': ['Start training at a tiny learning rate and raise it over the first few thousand steps. Early gradients are noisy and the weights are random; warmup stops the first steps from wrecking the run.', '14 · Optimisation'],
    'cosine schedule': ['Decay the learning rate along a cosine curve from its peak down to nearly zero over training — fast progress early, fine adjustment late. The default for transformer pretraining.', '14 · Optimisation'],
    'flat minima': ['Basins whose floor is wide, so nearby parameter values score almost as well. They generalise better than sharp ones: a small mismatch between training and live data barely moves the loss.', '14 · Optimisation'],
    'flat minimum': ['A basin whose floor is wide, so nearby parameter values score almost as well. These generalise better than sharp ones: a small mismatch between training and live data barely moves the loss.', '14 · Optimisation'],
    'ppo': ['Proximal Policy Optimisation — a reinforcement-learning method that improves a policy while penalising it for drifting far from the previous one. The optimiser behind classic RLHF.', '34 · Reinforcement learning'],
    'vae': ['Variational autoencoder — a model that compresses each input to a distribution rather than a point, then reconstructs from a sample of it. Its loss is reconstruction error plus a KL term.', '46 · LLM systems'],
    'elbo': ['Evidence lower bound — the quantity you actually maximise when the true likelihood is impossible to compute. Maximising the bound pushes the real thing up with it.', '13 · Information theory'],
    'variational': ['Replacing an intractable distribution with the closest member of a family you can handle, and optimising to make "closest" true. The K in KL is what measures closest.', '13 · Information theory'],
    'fisher information': ['How sharply the likelihood peaks around its best fit — high means the data pins the parameter down, low means many values explain it nearly as well. It is the curvature of the log-likelihood.', '14 · Optimisation'],
    'mutual information': ['How much knowing one variable reduces your uncertainty about another, in bits. Zero exactly when they are independent.', '13 · Information theory'],
    'jensen–shannon': ['A symmetric, bounded relative of KL divergence: compare both distributions against their average. Unlike KL it is finite even when one assigns zero probability where the other does not.', '13 · Information theory'],
    'bits-per-character': ['A language model\'s loss expressed as bits per character of text rather than nats per token, so models with different tokenisers can be compared.', '13 · Information theory'],
    'bits per character': ['A language model\'s loss expressed as bits per character of text rather than nats per token, so models with different tokenisers can be compared.', '13 · Information theory'],
    'temperature': ['A knob on the softmax that divides the logits before they become probabilities. Below 1 sharpens towards the top choice; above 1 flattens towards a coin flip.', '45 · Inference'],
    'bernoulli': ['The distribution of a single yes/no event with probability p — one coin flip, fair or not.', '12 · Probability'],
    'poisson': ['The distribution of how many independent events land in a fixed window when they arrive at a steady average rate — requests per second, defects per batch.', '12 · Probability'],
    'gaussian': ['The bell curve. It turns up everywhere because averages of many independent things tend towards it, whatever they individually looked like.', '12 · Probability'],
    'i.i.d': ['Independent and identically distributed: every example is drawn from the same distribution and none influences another. Nearly every guarantee in machine learning assumes it, and production breaks it first.', '12 · Probability'],
    'maximum a posteriori': ['Maximum likelihood with a prior belief folded in, so the answer is pulled towards what you expected before seeing data. Regularisation is this wearing different clothes.', '12 · Probability'],
    'mle': ['Maximum likelihood estimation — pick the parameters under which the data you actually observed was most probable.', '12 · Probability'],
    'posterior': ['What you believe about a quantity after seeing the data — your earlier belief updated by the evidence.', '12 · Probability']
  };

  var RX = null, KEYS = null;

  function esc(s) { return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&'); }

  function build() {
    KEYS = Object.keys(G).sort(function (a, b) { return b.length - a.length; });
    RX = new RegExp('\\b(' + KEYS.map(esc).join('|') + ')(s?)\\b', 'i');
  }

  var SKIP = { PRE: 1, CODE: 1, A: 1, SCRIPT: 1, STYLE: 1, SUMMARY: 1, BUTTON: 1,
               H1: 1, H2: 1, H3: 1, H4: 1, TEXTAREA: 1, INPUT: 1, SVG: 1 };

  function chip(term, word) {
    var e = G[term];
    var s = document.createElement('span');
    s.className = 'gl';
    s.setAttribute('role', 'button');
    s.setAttribute('tabindex', '0');
    s.setAttribute('aria-haspopup', 'dialog');
    s.setAttribute('aria-expanded', 'false');
    s.appendChild(document.createTextNode(word));
    var i = document.createElement('i');
    i.innerHTML = '<b>' + word + '</b> — ' + e[0] +
                  '<span class="at">taught properly in ' + e[1] + '</span>';
    s.appendChild(i);
    return s;
  }

  function walk(node, used, defined) {
    var kid = node.firstChild;
    while (kid) {
      var next = kid.nextSibling;
      if (kid.nodeType === 3) {
        var m = RX.exec(kid.nodeValue);
        if (m) {
          var term = m[1].toLowerCase();
          if (!used[term] && !defined[term]) {
            used[term] = 1;
            var after = kid.splitText(m.index);
            after.nodeValue = after.nodeValue.slice(m[1].length + m[2].length);
            node.insertBefore(chip(term, m[1] + m[2]), after);
            next = after;                       // rescan the tail for more terms
          }
        }
      } else if (kid.nodeType === 1) {
        var t = kid.tagName.toUpperCase();
        if (!SKIP[t] && !kid.classList.contains('gl') &&
            !kid.classList.contains('t') && !kid.classList.contains('shapestr') &&
            !kid.classList.contains('sym') && !kid.classList.contains('mth'))
          walk(kid, used, defined);
      }
      kid = next;
    }
  }

  function init(root) {
    root = root || document;
    if (!RX) build();
    /* a page that defines a term in its own term card does not need a chip */
    var defined = {}, cards = root.querySelectorAll('.term > .t'), i;
    for (i = 0; i < cards.length; i++) {
      var head = cards[i].textContent.toLowerCase();
      for (var k = 0; k < KEYS.length; k++)
        if (head.indexOf(KEYS[k]) >= 0) defined[KEYS[k]] = 1;
    }
    var scope = root.querySelector('#wrap') || root.body || root;
    walk(scope, {}, defined);
  }

  return { init: init, terms: G };
})();

/* ===================================================================
   The definition panel.

   A chip's definition lives in its own hidden <i> — that is what print
   and a scriptless page fall back to. Opening a chip copies that markup
   into ONE shared panel attached to <body> and anchors it to the chip.

   Body-level and fixed, deliberately: chips land inside table cells and
   list items, and expanding in place there pushes the row apart and
   reflows the paragraph around it. A fixed panel changes no layout at
   all, so the page underneath does not move.
   =================================================================== */
(function () {
  'use strict';

  var GAP = 10;          /* chip-to-panel breathing room */
  var EDGE = 12;         /* closest the panel may come to a viewport edge */
  var WIDE = 380;        /* panel width on a roomy screen */
  var NARROW = 560;      /* at or below this, dock it to the bottom instead */

  var pop = null, anchor = null, queued = false;

  function panel() {
    if (pop) return pop;
    pop = document.createElement('div');
    pop.id = 'glpop';
    pop.setAttribute('role', 'dialog');
    pop.setAttribute('aria-label', 'Definition');
    pop.hidden = true;
    pop.innerHTML = '<div class="glpop-body"></div>' +
                    '<button type="button" class="glpop-x" aria-label="Close">\u00d7</button>' +
                    '<span class="glpop-caret" aria-hidden="true"></span>';
    document.body.appendChild(pop);
    return pop;
  }

  /* Anchor the panel to the chip: below it by default, flipped above when
     that would run off the bottom, and always clamped inside the viewport
     so a chip near an edge cannot push it off-screen. */
  function place() {
    if (!anchor || !pop || pop.hidden) return;

    var r = anchor.getBoundingClientRect();
    var vw = document.documentElement.clientWidth;
    var vh = document.documentElement.clientHeight;

    /* the chip scrolled out of view — nothing left to point at */
    if (r.bottom < 0 || r.top > vh) { close(); return; }

    if (vw <= NARROW) {                     /* phone: a sheet, no arithmetic */
      pop.classList.add('dock');
      pop.classList.remove('up');
      pop.style.left = pop.style.top = pop.style.width = '';
      return;
    }
    pop.classList.remove('dock');

    var w = Math.min(WIDE, vw - 2 * EDGE);
    pop.style.width = w + 'px';             /* set width before reading height */

    var left = Math.round(r.left + r.width / 2 - w / 2);
    left = Math.max(EDGE, Math.min(left, vw - EDGE - w));

    var h = pop.offsetHeight;
    var below = r.bottom + GAP;
    var above = r.top - GAP - h;
    var up = (below + h > vh - EDGE) && above >= EDGE;

    pop.style.left = left + 'px';
    pop.style.top = (up ? above : below) + 'px';
    pop.classList.toggle('up', up);

    /* The caret tracks the chip but stays within the panel's own corners.
       It is offset from the panel's PADDING box, so the accent border on
       the left has to come out of the sum or the arrow sits beside the
       word rather than under it. */
    var cx = Math.max(left + 16, Math.min(r.left + r.width / 2, left + w - 16));
    pop.querySelector('.glpop-caret').style.left =
      (cx - left - pop.clientLeft) + 'px';
  }

  function reflow() {
    if (queued) return;
    queued = true;
    requestAnimationFrame(function () { queued = false; place(); });
  }

  function open(chip) {
    var src = chip.querySelector('i');
    if (!src) return;
    var p = panel();
    p.querySelector('.glpop-body').innerHTML = src.innerHTML;
    if (anchor && anchor !== chip) anchor.setAttribute('aria-expanded', 'false');
    anchor = chip;
    chip.setAttribute('aria-expanded', 'true');
    chip.classList.add('open');
    p.hidden = false;                       /* visible before measuring */
    place();
  }

  function close(refocus) {
    if (!pop || pop.hidden) return;
    pop.hidden = true;
    if (anchor) {
      anchor.setAttribute('aria-expanded', 'false');
      anchor.classList.remove('open');
      if (refocus) anchor.focus();
    }
    anchor = null;
  }

  function isOpen(chip) { return chip === anchor && pop && !pop.hidden; }

  document.addEventListener('click', function (e) {
    var t = e.target;
    if (!t || !t.closest) return;
    if (pop && !pop.hidden && t.closest('#glpop')) {
      if (t.closest('.glpop-x')) close(true);
      return;                               /* clicks inside the panel stay */
    }
    var g = t.closest('.gl');
    if (!g) { close(false); return; }       /* click anywhere else dismisses */
    if (isOpen(g)) close(false); else open(g);
  });

  document.addEventListener('keydown', function (e) {
    if (e.key === 'Escape') { close(true); return; }
    if (e.key !== 'Enter' && e.key !== ' ') return;
    var g = document.activeElement;
    if (g && g.classList && g.classList.contains('gl')) {
      e.preventDefault();
      if (isOpen(g)) close(false); else open(g);
    }
  });

  /* #main is the scroll container, not the window, so listen in the capture
     phase and catch scrolls from whichever element actually moved. */
  document.addEventListener('scroll', reflow, true);
  window.addEventListener('resize', reflow);
})();
