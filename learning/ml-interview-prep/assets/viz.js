/* ===================================================================
   MLIPViz — hand-authored SVG visualisations. No libraries.
   Usage in a part file:
       <div class="viz" data-viz="minima"></div>
   MLIPViz.init(root) finds every [data-viz] and renders it once.
   =================================================================== */
window.MLIPViz = (function () {
  'use strict';

  var NS = 'http://www.w3.org/2000/svg';

  function el(tag, attrs, parent) {
    var n = document.createElementNS(NS, tag);
    for (var k in attrs) n.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(n);
    return n;
  }
  function h(tag, cls, txt, parent) {
    var n = document.createElement(tag);
    if (cls) n.className = cls;
    if (txt !== undefined && txt !== null) n.textContent = txt;
    if (parent) parent.appendChild(n);
    return n;
  }
  function css(v) {
    return getComputedStyle(document.documentElement).getPropertyValue(v).trim() || '#888';
  }

  /* A small 2-D plotting frame: maps data coords -> svg pixels. */
  function Frame(svg, box, xr, yr) {
    this.svg = svg; this.box = box; this.xr = xr; this.yr = yr;
  }
  Frame.prototype.X = function (x) {
    return this.box.l + (x - this.xr[0]) / (this.xr[1] - this.xr[0]) * this.box.w;
  };
  Frame.prototype.Y = function (y) {
    return this.box.t + this.box.h - (y - this.yr[0]) / (this.yr[1] - this.yr[0]) * this.box.h;
  };
  Frame.prototype.path = function (fn, n, cls, extra) {
    var d = '', i, x, y;
    n = n || 240;
    for (i = 0; i <= n; i++) {
      x = this.xr[0] + (this.xr[1] - this.xr[0]) * i / n;
      y = fn(x);
      if (!isFinite(y)) { d = ''; continue; }
      y = Math.max(this.yr[0], Math.min(this.yr[1], y));
      d += (d ? ' L' : 'M') + this.X(x).toFixed(1) + ',' + this.Y(y).toFixed(1);
    }
    var a = { d: d, class: cls || 's-curve' };
    for (var k in (extra || {})) a[k] = extra[k];
    return el('path', a, this.svg);
  };
  Frame.prototype.axes = function (xl, yl) {
    var b = this.box;
    el('line', { x1: b.l, y1: b.t + b.h, x2: b.l + b.w, y2: b.t + b.h, class: 's-axis' }, this.svg);
    el('line', { x1: b.l, y1: b.t, x2: b.l, y2: b.t + b.h, class: 's-axis' }, this.svg);
    if (xl) el('text', { x: b.l + b.w, y: b.t + b.h + 20, class: 's-lbl', 'text-anchor': 'end' },
               this.svg).textContent = xl;
    if (yl) {
      var t = el('text', { x: b.l - 8, y: b.t + 4, class: 's-lbl', 'text-anchor': 'end' }, this.svg);
      t.textContent = yl;
      /* A long y label right-anchored inside a narrow left margin runs off
         the canvas and gets silently cut. Measure it, and if it does not
         fit, hang it above the axis instead. */
      var wdt = 0;
      try { wdt = t.getComputedTextLength(); } catch (e) { wdt = yl.length * 6; }
      if (b.l - 8 - wdt < 2) {
        t.setAttribute('text-anchor', 'start');
        t.setAttribute('x', Math.max(2, b.l - 10));
        t.setAttribute('y', Math.max(11, b.t - 7));
      }
    }
    return this;
  };

  function makeSVG(host, w, hgt) {
    var s = el('svg', { viewBox: '0 0 ' + w + ' ' + hgt, width: w, height: hgt,
                        preserveAspectRatio: 'xMidYMid meet' });
    host.appendChild(s);
    return s;
  }
  function cap(host, txt) { h('div', 'cap', txt, host); }
  function ctl(host) { return h('div', 'ctl', null, host); }

  /* Colour key. These figures use colour to carry meaning — which term subtracts,
     which direction turned, which cell is still wrong — and a reader who has to
     infer that from context is guessing. keys() is called from inside paint(), so
     the strip is rebuilt with the view and can never name a colour that is not
     currently on screen. */
  function keyBar(host) { return h('div', 'vizkey', null, host); }
  function keys(bar, rows) {
    if (!bar) return;
    bar.textContent = '';
    rows.forEach(function (r) {
      if (!r) return;
      var s = h('span', 'k', null, bar), sw = h('i', r[2] || null, null, s);
      if (r[0] === 'cell') {
        sw.style.background = css('--amber-bg');
        sw.style.borderColor = css('--gold');
      } else if (r[2] === 'line' || r[2] === 'dash' || r[2] === 'dot') {
        sw.style.borderTopColor = css(r[0]);
      } else {
        sw.style.background = css(r[0]);
        sw.style.borderColor = css(r[0]);
      }
      h('b', null, r[1], s);
    });
  }
  function slider(parent, label, min, max, step, val, oninput) {
    var lb = h('label', null, label + ' ', parent);
    var inp = document.createElement('input');
    inp.type = 'range'; inp.min = min; inp.max = max; inp.step = step; inp.value = val;
    lb.appendChild(inp);
    var out = h('span', 'val', '', lb);
    function fire() { out.textContent = oninput(parseFloat(inp.value)); }
    inp.addEventListener('input', fire);
    fire();
    return inp;
  }
  function button(parent, txt, fn) {
    var b = h('button', null, txt, parent);
    b.addEventListener('click', fn);
    return b;
  }

  /* =================================================================
     1. MINIMA — what a minimum actually is (local, global, saddle/plateau)
     ================================================================= */
  function vizMinima(host) {
    var W = 700, H = 300, box = { l: 46, t: 18, w: W - 80, h: H - 66 };
    var svg = makeSVG(host, W, H);
    // a deliberately bumpy landscape
    function f(x) { return 0.35 * Math.pow(x, 4) - 1.1 * Math.pow(x, 2) - 0.42 * x + 2.2; }
    function df(x) { return 1.4 * Math.pow(x, 3) - 2.2 * x - 0.42; }
    var F = new Frame(svg, box, [-2.2, 2.2], [0, 4]);
    F.axes('parameter  w', 'loss');
    F.path(f, 300);

    // mark the three interesting points
    var pts = [
      { x: -1.16, l: 'local minimum', c: 'amber' },
      { x:  0.19, l: 'local maximum', c: 'faint' },
      { x:  1.31, l: 'global minimum', c: 'green' }
    ];
    pts.forEach(function (p) {
      el('circle', { cx: F.X(p.x), cy: F.Y(f(p.x)), r: 4.5,
                     fill: css('--' + (p.c === 'faint' ? 'tx-faint' : p.c)) }, svg);
      var t = el('text', { x: F.X(p.x), y: F.Y(f(p.x)) + (p.c === 'faint' ? -14 : 22),
                           class: 's-lbl-s', 'text-anchor': 'middle' }, svg);
      t.textContent = p.l;
    });

    var ball = el('circle', { cx: 0, cy: 0, r: 7, fill: css('--accent') }, svg);
    var tang = el('line', { class: 's-axis', stroke: css('--pink'), 'stroke-width': 2 }, svg);
    var readout = el('text', { x: box.l + 6, y: box.t + 14, class: 's-lbl' }, svg);

    var x = -2.0, lr = 0.08, running = false, timer = null;
    function draw() {
      ball.setAttribute('cx', F.X(x)); ball.setAttribute('cy', F.Y(f(x)));
      var g = df(x), dx = 0.42;
      tang.setAttribute('x1', F.X(x - dx)); tang.setAttribute('y1', F.Y(f(x) - g * dx));
      tang.setAttribute('x2', F.X(x + dx)); tang.setAttribute('y2', F.Y(f(x) + g * dx));
      readout.textContent = 'w = ' + x.toFixed(3) + '   loss = ' + f(x).toFixed(3) +
                            '   slope = ' + g.toFixed(3);
    }
    function stepOne() {
      x = x - lr * df(x);
      x = Math.max(-2.2, Math.min(2.2, x));
      draw();
      if (Math.abs(df(x)) < 1e-4) stop();
    }
    function stop() { running = false; clearInterval(timer); }
    draw();

    var c = ctl(host);
    button(c, '▶ roll', function () {
      if (running) { stop(); return; }
      running = true; timer = setInterval(stepOne, 55);
    });
    button(c, 'step', function () { stop(); stepOne(); });
    slider(c, 'learning rate', 0.01, 0.55, 0.01, 0.08, function (v) { lr = v; return v.toFixed(2); });
    slider(c, 'start w', -2.1, 2.1, 0.05, -2.0, function (v) { stop(); x = v; draw(); return v.toFixed(2); });
    cap(host, 'A minimum is a point where the slope is zero AND the ground curves upward on both ' +
              'sides. Start on the left and gradient descent falls into the LOCAL minimum and stops ' +
              'there — the slope is zero, so it has no reason to move. Start on the right and it ' +
              'finds the global one. Push the learning rate past ~0.35 and it overshoots and bounces.');
  }

  /* =================================================================
     2. DERIVATIVE — a tangent line sliding along a curve
     ================================================================= */
  function vizDerivative(host) {
    var W = 640, H = 280, box = { l: 46, t: 18, w: W - 80, h: H - 62 };
    var svg = makeSVG(host, W, H);
    function f(x) { return 0.5 * x * x; }
    function df(x) { return x; }
    var F = new Frame(svg, box, [-3, 3], [-0.4, 4.6]);
    F.axes('x', 'f(x)');
    F.path(f, 200);

    var sec = el('line', { stroke: css('--tx-faint'), 'stroke-width': 1.6,
                           'stroke-dasharray': '5 4' }, svg);
    var tan = el('line', { stroke: css('--pink'), 'stroke-width': 2.4 }, svg);
    var p1 = el('circle', { r: 5, fill: css('--accent') }, svg);
    var p2 = el('circle', { r: 4, fill: css('--tx-faint') }, svg);
    var rd = el('text', { x: box.l + 6, y: box.t + 14, class: 's-lbl' }, svg);

    var x = 1.4, hstep = 1.2;
    function draw() {
      var y = f(x), x2 = x + hstep, y2 = f(x2);
      var slope = (y2 - y) / hstep;
      p1.setAttribute('cx', F.X(x)); p1.setAttribute('cy', F.Y(y));
      p2.setAttribute('cx', F.X(x2)); p2.setAttribute('cy', F.Y(y2));
      sec.setAttribute('x1', F.X(x)); sec.setAttribute('y1', F.Y(y));
      sec.setAttribute('x2', F.X(x2)); sec.setAttribute('y2', F.Y(y2));
      var d = 1.1, t = df(x);
      tan.setAttribute('x1', F.X(x - d)); tan.setAttribute('y1', F.Y(y - t * d));
      tan.setAttribute('x2', F.X(x + d)); tan.setAttribute('y2', F.Y(y + t * d));
      rd.textContent = 'gap h = ' + hstep.toFixed(2) +
        '   ·   average slope over the gap = ' + slope.toFixed(3) +
        '   ·   true slope at x = ' + t.toFixed(3);
    }
    draw();
    var c = ctl(host);
    slider(c, 'x', -2.6, 2.6, 0.05, 1.4, function (v) { x = v; draw(); return v.toFixed(2); });
    slider(c, 'gap h', 0.02, 1.6, 0.02, 1.2, function (v) { hstep = v; draw(); return v.toFixed(2); });
    cap(host, 'Drag "gap h" toward zero. The grey dashed line is the average slope between two ' +
              'points you can actually measure; the pink line is the derivative. The derivative is ' +
              'just what the average slope settles on as the gap shrinks to nothing.');
  }

  /* =================================================================
     3. LEARNING RATE — same bowl, three learning rates
     ================================================================= */
  function vizLR(host) {
    var W = 700, H = 300, box = { l: 46, t: 18, w: W - 80, h: H - 66 };
    var svg = makeSVG(host, W, H);
    function f(x) { return 0.5 * x * x; }
    function df(x) { return x; }
    var F = new Frame(svg, box, [-4.6, 4.6], [0, 10.6]);
    F.axes('parameter  w', 'loss');
    F.path(f, 200);

    var g = el('g', {}, svg);
    var rd = el('text', { x: box.l + 6, y: box.t + 14, class: 's-lbl' }, svg);
    var lr = 0.3;

    function run() {
      while (g.firstChild) g.removeChild(g.firstChild);
      var x = -4.2, pts = [[x, f(x)]], i;
      for (i = 0; i < 22; i++) { x = x - lr * df(x); pts.push([x, f(x)]); }
      var d = '';
      pts.forEach(function (p, k) {
        var px = Math.max(-4.6, Math.min(4.6, p[0]));
        d += (k ? ' L' : 'M') + F.X(px) + ',' + F.Y(Math.min(10.6, f(px)));
      });
      el('path', { d: d, fill: 'none', stroke: css('--amber'), 'stroke-width': 1.6,
                   'stroke-dasharray': '4 3' }, g);
      pts.forEach(function (p, k) {
        var px = Math.max(-4.6, Math.min(4.6, p[0]));
        el('circle', { cx: F.X(px), cy: F.Y(Math.min(10.6, f(px))), r: k === 0 ? 5 : 3,
                       fill: k === 0 ? css('--accent') : css('--amber'),
                       opacity: k === 0 ? 1 : (0.35 + 0.65 * (1 - k / 22)) }, g);
      });
      var last = pts[pts.length - 1][0];
      var verdict = Math.abs(last) < 0.05 ? 'converged'
                  : Math.abs(last) > 4.5  ? 'DIVERGED — loss is exploding'
                  : (lr > 1 ? 'oscillating outward' : 'still crawling');
      rd.textContent = 'lr = ' + lr.toFixed(2) + '   after 22 steps  w = ' +
                       last.toFixed(3) + '   → ' + verdict;
    }
    run();
    var c = ctl(host);
    slider(c, 'learning rate', 0.02, 2.2, 0.02, 0.3, function (v) { lr = v; run(); return v.toFixed(2); });
    cap(host, 'One bowl, one starting point, only the step size changes. Below ~0.1 it crawls and ' +
              'runs out of budget. Around 0.3–0.9 it converges. At exactly 2.0 it bounces forever ' +
              'between the same two points. Above 2.0 every step lands further out than the last ' +
              'and the loss goes to infinity — that is what "diverged" means in a training log.');
  }

  /* =================================================================
     4. ENTROPY — how surprised you are on average
     ================================================================= */
  function vizEntropy(host) {
    var W = 620, H = 250, box = { l: 46, t: 18, w: W - 80, h: H - 70 };
    var svg = makeSVG(host, W, H);
    var K = 8;
    var bars = [], labs = [];
    var F = new Frame(svg, box, [0, K], [0, 1]);
    F.axes('outcome', 'p');
    for (var i = 0; i < K; i++) {
      bars.push(el('rect', { x: 0, y: 0, width: 0, height: 0, fill: css('--accent'),
                             rx: 3 }, svg));
      labs.push(el('text', { class: 's-lbl-s', 'text-anchor': 'middle' }, svg));
    }
    var rd = el('text', { x: box.l + box.w, y: box.t + 14, class: 's-lbl',
                          'text-anchor': 'end' }, svg);

    function draw(temp) {
      // temp -> 0 : one-hot (certain).  temp large : uniform.
      var logits = [3.0, 1.4, 0.9, 0.5, 0.2, 0.0, -0.4, -0.9], p = [], s = 0, i;
      for (i = 0; i < K; i++) { p[i] = Math.exp(logits[i] / temp); s += p[i]; }
      var Hb = 0;
      for (i = 0; i < K; i++) { p[i] /= s; if (p[i] > 0) Hb -= p[i] * Math.log2(p[i]); }
      var bw = box.w / K * 0.7, gap = box.w / K;
      for (i = 0; i < K; i++) {
        var hpx = p[i] * box.h;
        bars[i].setAttribute('x', box.l + i * gap + (gap - bw) / 2);
        bars[i].setAttribute('width', bw);
        bars[i].setAttribute('y', box.t + box.h - hpx);
        bars[i].setAttribute('height', Math.max(0.5, hpx));
        labs[i].setAttribute('x', box.l + i * gap + gap / 2);
        labs[i].setAttribute('y', box.t + box.h - hpx - 5);
        labs[i].textContent = p[i] > 0.02 ? p[i].toFixed(2) : '';
      }
      return Hb;
    }
    var c = ctl(host);
    slider(c, 'flatness', 0.15, 6, 0.05, 1, function (v) {
      var Hb = draw(v);
      rd.textContent = 'entropy = ' + Hb.toFixed(3) + ' bits   (max = 3.000)';
      return v.toFixed(2);
    });
    cap(host, 'Entropy is the average number of yes/no questions you need to pin down the answer. ' +
              'Slide left: the distribution becomes a spike, you already know the answer, entropy → 0. ' +
              'Slide right: all eight outcomes equally likely, you need exactly log₂8 = 3 questions. ' +
              'This is the same quantity your training loss reports, in nats instead of bits.');
  }

  /* =================================================================
     5. KL DIVERGENCE — and why it is not symmetric
     ================================================================= */
  function vizKL(host) {
    var W = 640, H = 270, box = { l: 46, t: 20, w: W - 80, h: H - 84 };
    var svg = makeSVG(host, W, H);
    var K = 9;
    var F = new Frame(svg, box, [0, K], [0, 0.62]);
    F.axes('outcome', 'probability');
    var pb = [], qb = [];
    for (var i = 0; i < K; i++) {
      pb.push(el('rect', { fill: css('--accent'), rx: 2, opacity: .95 }, svg));
      qb.push(el('rect', { fill: css('--pink'), rx: 2, opacity: .85 }, svg));
    }
    var rd1 = el('text', { x: box.l + 4, y: box.t + 13, class: 's-lbl' }, svg);
    var rd2 = el('text', { x: box.l + 4, y: box.t + 30, class: 's-lbl' }, svg);
    var lg1 = el('text', { x: box.l + box.w, y: box.t + 13, class: 's-lbl',
                           'text-anchor': 'end', fill: css('--accent') }, svg);
    lg1.textContent = '■ P  (truth)';
    var lg2 = el('text', { x: box.l + box.w, y: box.t + 30, class: 's-lbl',
                           'text-anchor': 'end', fill: css('--pink') }, svg);
    lg2.textContent = '■ Q  (your model)';

    function gauss(mu, sd) {
      var a = [], s = 0, i;
      for (i = 0; i < K; i++) { a[i] = Math.exp(-0.5 * Math.pow((i - mu) / sd, 2)) + 1e-9; s += a[i]; }
      for (i = 0; i < K; i++) a[i] /= s;
      return a;
    }
    var shift = 1.6, spread = 1.2;
    function draw() {
      var P = gauss(3.0, 1.0), Q = gauss(3.0 + shift, spread);
      var kpq = 0, kqp = 0, i;
      for (i = 0; i < K; i++) {
        kpq += P[i] * Math.log(P[i] / Q[i]);
        kqp += Q[i] * Math.log(Q[i] / P[i]);
      }
      var gap = box.w / K, bw = gap * 0.36;
      for (i = 0; i < K; i++) {
        var hp = P[i] / 0.62 * box.h, hq = Q[i] / 0.62 * box.h;
        pb[i].setAttribute('x', box.l + i * gap + gap * 0.10);
        pb[i].setAttribute('width', bw);
        pb[i].setAttribute('y', box.t + box.h - hp);
        pb[i].setAttribute('height', Math.max(0.5, hp));
        qb[i].setAttribute('x', box.l + i * gap + gap * 0.52);
        qb[i].setAttribute('width', bw);
        qb[i].setAttribute('y', box.t + box.h - hq);
        qb[i].setAttribute('height', Math.max(0.5, hq));
      }
      rd1.textContent = 'KL(P ‖ Q) = ' + kpq.toFixed(3) + ' nats';
      rd2.textContent = 'KL(Q ‖ P) = ' + kqp.toFixed(3) + ' nats';
    }
    draw();
    var c = ctl(host);
    slider(c, 'Q shifted by', -2.5, 2.5, 0.1, 1.6, function (v) { shift = v; draw(); return v.toFixed(1); });
    slider(c, 'Q width', 0.45, 3.0, 0.05, 1.2, function (v) { spread = v; draw(); return v.toFixed(2); });
    cap(host, 'KL is "how many extra nats do I pay per sample because I used Q when the truth was P". ' +
              'Note the two numbers are different — KL is not a distance. Make Q narrow and offset: ' +
              'KL(P‖Q) explodes (P puts mass where Q says "impossible"), while KL(Q‖P) stays small. ' +
              'That asymmetry is exactly why forward-KL training is mode-covering and reverse-KL is ' +
              'mode-seeking.');
  }

  /* =================================================================
     6. BIAS–VARIANCE — the U curve
     ================================================================= */
  function vizBiasVar(host) {
    var W = 640, H = 290, box = { l: 50, t: 18, w: W - 90, h: H - 72 };
    var svg = makeSVG(host, W, H);
    var F = new Frame(svg, box, [0.4, 10], [0, 5]);
    F.axes('model complexity  →', 'error');
    function bias2(c) { return 4.2 / (c * c) + 0.05; }
    function vari(c) { return 0.045 * c * c; }
    function noise() { return 0.55; }
    function total(c) { return bias2(c) + vari(c) + noise(); }
    F.path(bias2, 200, 's-curve', { stroke: css('--green'), 'stroke-dasharray': '5 4',
                                    'stroke-width': 2 });
    F.path(vari, 200, 's-curve', { stroke: css('--pink'), 'stroke-dasharray': '5 4',
                                   'stroke-width': 2 });
    F.path(noise, 20, 's-curve', { stroke: css('--tx-faint'), 'stroke-dasharray': '2 4',
                                   'stroke-width': 1.6 });
    F.path(total, 240, 's-curve', { 'stroke-width': 3 });

    // minimum of total
    var best = 0.4, bv = 1e9;
    for (var c0 = 0.4; c0 <= 10; c0 += 0.01) if (total(c0) < bv) { bv = total(c0); best = c0; }
    el('line', { x1: F.X(best), y1: F.Y(0), x2: F.X(best), y2: F.Y(bv),
                 stroke: css('--amber'), 'stroke-width': 1.4, 'stroke-dasharray': '3 3' }, svg);
    el('circle', { cx: F.X(best), cy: F.Y(bv), r: 5, fill: css('--amber') }, svg);
    var lb = el('text', { x: F.X(best), y: F.Y(bv) - 12, class: 's-lbl-s',
                          'text-anchor': 'middle', fill: css('--amber') }, svg);
    lb.textContent = 'sweet spot';

    [['bias²  (underfit)', '--green', 0],
     ['variance  (overfit)', '--pink', 17],
     ['irreducible noise', '--tx-faint', 34],
     ['total test error', '--accent', 51]].forEach(function (r) {
      var t = el('text', { x: box.l + box.w + 4, y: box.t + 12 + r[2], class: 's-lbl-s',
                           fill: css(r[1]), 'text-anchor': 'end' }, svg);
      t.textContent = r[0];
    });
    cap(host, 'Left of the sweet spot the model is too simple — it is wrong in the same way every ' +
              'time (bias). Right of it the model is so flexible it memorises the noise, so a ' +
              'different training sample would give a very different model (variance). The flat ' +
              'grey line is noise in the labels: no model of any size ever removes it.');
  }

  /* =================================================================
     7. LAGRANGE — optimising with a constraint
     ================================================================= */
  function vizLagrange(host) {
    var W = 470, H = 400, box = { l: 46, t: 18, w: W - 76, h: H - 76 };
    var svg = makeSVG(host, W, H);
    var F = new Frame(svg, box, [-3.2, 3.2], [-3.2, 3.2]);
    F.axes('w₁', 'w₂');

    // objective contours: f = (w1-1.9)^2 + (w2-1.5)^2, centre = unconstrained optimum
    var cx = 1.9, cy = 1.5;
    [0.5, 1.0, 1.6, 2.2, 2.9, 3.6].forEach(function (r) {
      el('circle', { cx: F.X(cx), cy: F.Y(cy), r: r / 6.4 * box.w,
                     fill: 'none', stroke: css('--border-2'), 'stroke-width': 1 }, svg);
    });
    el('circle', { cx: F.X(cx), cy: F.Y(cy), r: 4, fill: css('--tx-faint') }, svg);
    var t0 = el('text', { x: F.X(cx) + 8, y: F.Y(cy) - 6, class: 's-lbl-s' }, svg);
    t0.textContent = 'unconstrained best';

    // constraint: ||w||^2 = t  (the ridge budget)
    var ring = el('circle', { cx: F.X(0), cy: F.Y(0), fill: 'none',
                              stroke: css('--pink'), 'stroke-width': 2.4 }, svg);
    var sol = el('circle', { r: 6, fill: css('--amber') }, svg);
    var rd = el('text', { x: box.l + 4, y: box.t + 14, class: 's-lbl' }, svg);
    var rd2 = el('text', { x: box.l + 4, y: box.t + 31, class: 's-lbl' }, svg);
    var grad = el('line', { stroke: css('--green'), 'stroke-width': 2 }, svg);
    var gradc = el('line', { stroke: css('--pink'), 'stroke-width': 2 }, svg);

    var t = 1.6;
    function draw() {
      var R = Math.sqrt(t);
      ring.setAttribute('r', R / 6.4 * box.w);
      var n = Math.sqrt(cx * cx + cy * cy);
      var sx, sy;
      if (n <= R) { sx = cx; sy = cy; } else { sx = cx / n * R; sy = cy / n * R; }
      sol.setAttribute('cx', F.X(sx)); sol.setAttribute('cy', F.Y(sy));
      // -grad f at the solution points toward the centre; grad g points outward. Parallel.
      var gx = (cx - sx), gy = (cy - sy), gl = Math.hypot(gx, gy) || 1;
      var A = 44;
      grad.setAttribute('x1', F.X(sx)); grad.setAttribute('y1', F.Y(sy));
      grad.setAttribute('x2', F.X(sx) + gx / gl * A); grad.setAttribute('y2', F.Y(sy) - gy / gl * A);
      var hl = Math.hypot(sx, sy) || 1;
      gradc.setAttribute('x1', F.X(sx)); gradc.setAttribute('y1', F.Y(sy));
      gradc.setAttribute('x2', F.X(sx) + sx / hl * A); gradc.setAttribute('y2', F.Y(sy) - sy / hl * A);
      var lam = (n > R) ? (n / R - 1) : 0;
      rd.textContent = 'budget ‖w‖² ≤ ' + t.toFixed(2) + '   solution = (' +
                       sx.toFixed(2) + ', ' + sy.toFixed(2) + ')';
      rd2.textContent = 'multiplier λ = ' + lam.toFixed(3) +
                        (lam === 0 ? '   (constraint inactive — KKT slackness)' : '   (constraint biting)');
    }
    draw();
    var c = ctl(host);
    slider(c, 'budget t', 0.15, 9, 0.05, 1.6, function (v) { t = v; draw(); return v.toFixed(2); });
    cap(host, 'Grey rings = the loss you want to minimise. Pink circle = "you may not spend more ' +
              'than t on your weights" — that is ridge regression. The best allowed point (amber) ' +
              'is always where a loss ring just kisses the pink circle, and at that kiss the two ' +
              'arrows are parallel. λ is how hard the constraint is pushing. Widen the budget past ' +
              'the unconstrained optimum and λ drops to 0 — the constraint stops mattering. That ' +
              'is complementary slackness, the KKT condition, on screen.');
  }

  /* =================================================================
     8. SIGMOID / SOFTMAX TEMPERATURE — activation shapes
     ================================================================= */
  function vizActivations(host) {
    var W = 640, H = 260, box = { l: 46, t: 18, w: W - 80, h: H - 62 };
    var svg = makeSVG(host, W, H);
    var F = new Frame(svg, box, [-5, 5], [-1.4, 3.2]);
    F.axes('input z', 'output');
    el('line', { x1: F.X(-5), y1: F.Y(0), x2: F.X(5), y2: F.Y(0), class: 's-grid' }, svg);
    var defs = [
      ['ReLU',    function (z) { return Math.max(0, z); },                    '--accent'],
      ['sigmoid', function (z) { return 1 / (1 + Math.exp(-z)); },            '--green'],
      ['tanh',    function (z) { return Math.tanh(z); },                      '--pink'],
      ['GELU',    function (z) { return 0.5 * z * (1 + Math.tanh(0.7978845608 * (z + 0.044715 * z * z * z))); }, '--purple'],
      ['SiLU',    function (z) { return z / (1 + Math.exp(-z)); },            '--amber']
    ];
    defs.forEach(function (d, i) {
      F.path(d[1], 220, 's-curve', { stroke: css(d[2]), 'stroke-width': 2.2 });
      var t = el('text', { x: box.l + box.w - 4, y: box.t + 14 + i * 16, class: 's-lbl-s',
                           fill: css(d[2]), 'text-anchor': 'end' }, svg);
      t.textContent = d[0];
    });
    cap(host, 'Every one of these is a way of saying "pass big numbers through, squash small ones". ' +
              'Sigmoid and tanh flatten at both ends — that flatness is where gradients die. ReLU ' +
              'never flattens on the right, which is why it unblocked deep networks; but it is ' +
              'exactly zero on the left, so a unit can get stuck dead. GELU and SiLU are the smooth ' +
              'fixes that transformers use.');
  }

  /* =================================================================
     9. COSINE — two draggable arrows, dot product vs cosine
     ================================================================= */
  function vizCosine(host) {
    var W = 440, H = 380, box = { l: 40, t: 20, w: W - 70, h: H - 90 };
    var svg = makeSVG(host, W, H);
    var F = new Frame(svg, box, [-3.2, 3.2], [-3.2, 3.2]);
    el('line', { x1: F.X(-3.2), y1: F.Y(0), x2: F.X(3.2), y2: F.Y(0), class: 's-grid' }, svg);
    el('line', { x1: F.X(0), y1: F.Y(-3.2), x2: F.X(0), y2: F.Y(3.2), class: 's-grid' }, svg);

    var arc = el('path', { fill: css('--amber'), opacity: .17 }, svg);
    var la = el('line', { stroke: css('--accent'), 'stroke-width': 3 }, svg);
    var lb = el('line', { stroke: css('--pink'), 'stroke-width': 3 }, svg);
    var ha = el('circle', { r: 8, fill: css('--accent'), cursor: 'grab' }, svg);
    var hb = el('circle', { r: 8, fill: css('--pink'), cursor: 'grab' }, svg);
    var r1 = el('text', { x: box.l, y: box.t + box.h + 26, class: 's-lbl' }, svg);
    var r2 = el('text', { x: box.l, y: box.t + box.h + 44, class: 's-lbl' }, svg);
    var r3 = el('text', { x: box.l, y: box.t + box.h + 62, class: 's-lbl' }, svg);

    var A = { x: 2.2, y: 1.1 }, B = { x: 0.7, y: 2.3 };
    function draw() {
      la.setAttribute('x1', F.X(0)); la.setAttribute('y1', F.Y(0));
      la.setAttribute('x2', F.X(A.x)); la.setAttribute('y2', F.Y(A.y));
      lb.setAttribute('x1', F.X(0)); lb.setAttribute('y1', F.Y(0));
      lb.setAttribute('x2', F.X(B.x)); lb.setAttribute('y2', F.Y(B.y));
      ha.setAttribute('cx', F.X(A.x)); ha.setAttribute('cy', F.Y(A.y));
      hb.setAttribute('cx', F.X(B.x)); hb.setAttribute('cy', F.Y(B.y));
      var dot = A.x * B.x + A.y * B.y;
      var na = Math.hypot(A.x, A.y), nb = Math.hypot(B.x, B.y);
      var cos = dot / (na * nb || 1);
      var ang = Math.acos(Math.max(-1, Math.min(1, cos))) * 180 / Math.PI;
      // angle wedge
      var rr = 46;
      var a0 = Math.atan2(A.y, A.x), a1 = Math.atan2(B.y, B.x);
      var lrg = Math.abs(a1 - a0) > Math.PI ? 1 : 0;
      var swp = ((a1 - a0 + 2 * Math.PI) % (2 * Math.PI)) > Math.PI ? 1 : 0;
      arc.setAttribute('d', 'M' + F.X(0) + ',' + F.Y(0) +
        ' L' + (F.X(0) + Math.cos(a0) * rr) + ',' + (F.Y(0) - Math.sin(a0) * rr) +
        ' A' + rr + ',' + rr + ' 0 ' + lrg + ' ' + (swp ? 1 : 0) + ' ' +
        (F.X(0) + Math.cos(a1) * rr) + ',' + (F.Y(0) - Math.sin(a1) * rr) + ' Z');
      r1.textContent = 'a·b = ' + dot.toFixed(2) +
                       '     ‖a‖ = ' + na.toFixed(2) + '   ‖b‖ = ' + nb.toFixed(2);
      r2.textContent = 'cosine = a·b / (‖a‖‖b‖) = ' + cos.toFixed(3) +
                       '     angle = ' + ang.toFixed(0) + '°';
      r3.textContent = cos > 0.9 ? 'nearly the same direction — "similar"'
                     : cos > 0.15 ? 'partly aligned'
                     : cos > -0.15 ? 'orthogonal — no shared information'
                     : 'pointing against each other';
    }
    draw();

    var drag = null;
    function pt(e) {
      var r = svg.getBoundingClientRect();
      var sx = (e.touches ? e.touches[0].clientX : e.clientX) - r.left;
      var sy = (e.touches ? e.touches[0].clientY : e.clientY) - r.top;
      sx *= W / r.width; sy *= H / r.height;
      return { x: (sx - box.l) / box.w * 6.4 - 3.2, y: 3.2 - (sy - box.t) / box.h * 6.4 };
    }
    function start(which) {
      return function (e) { drag = which; e.preventDefault(); };
    }
    ha.addEventListener('mousedown', start(A)); ha.addEventListener('touchstart', start(A));
    hb.addEventListener('mousedown', start(B)); hb.addEventListener('touchstart', start(B));
    function move(e) {
      if (!drag) return;
      var p = pt(e);
      drag.x = Math.max(-3, Math.min(3, p.x));
      drag.y = Math.max(-3, Math.min(3, p.y));
      draw(); e.preventDefault();
    }
    window.addEventListener('mousemove', move);
    window.addEventListener('touchmove', move, { passive: false });
    window.addEventListener('mouseup', function () { drag = null; });
    window.addEventListener('touchend', function () { drag = null; });

    cap(host, 'Drag either arrowhead. Watch what happens when you make the blue arrow LONGER without ' +
              'turning it: the dot product grows, the cosine does not move at all. That is the whole ' +
              'reason embedding search normalises — otherwise a long document beats a relevant one.');
  }

  /* =================================================================
     10. MATRIX2D — a 2x2 matrix as a machine, with its eigenvectors
     ================================================================= */
  function vizMatrix2D(host) {
    var W = 460, H = 430, box = { l: 40, t: 20, w: W - 70, h: W - 70 };
    var svg = makeSVG(host, W, H);
    var F = new Frame(svg, box, [-3, 3], [-3, 3]);
    var gGrid = el('g', {}, svg);
    var gShape = el('g', {}, svg);
    var gEig = el('g', {}, svg);
    var r1 = el('text', { x: box.l, y: box.t + box.h + 26, class: 's-lbl' }, svg);
    var r2 = el('text', { x: box.l, y: box.t + box.h + 44, class: 's-lbl' }, svg);

    var a = 1.4, b = 0.6, c = 0.4, d = 1.1, t = 1;

    function clearG(g) { while (g.firstChild) g.removeChild(g.firstChild); }

    function draw() {
      clearG(gGrid); clearG(gShape); clearG(gEig);
      // interpolate from identity to the matrix by t
      var A11 = 1 + t * (a - 1), A12 = t * b, A21 = t * c, A22 = 1 + t * (d - 1);
      function M(p) { return [A11 * p[0] + A12 * p[1], A21 * p[0] + A22 * p[1]]; }
      var i, j;
      for (i = -3; i <= 3; i++) {
        var pathH = '', pathV = '';
        for (j = -30; j <= 30; j++) {
          var ph = M([j / 10, i]), pv = M([i, j / 10]);
          pathH += (j === -30 ? 'M' : 'L') + F.X(ph[0]) + ',' + F.Y(ph[1]);
          pathV += (j === -30 ? 'M' : 'L') + F.X(pv[0]) + ',' + F.Y(pv[1]);
        }
        el('path', { d: pathH, fill: 'none', stroke: css('--border'),
                     'stroke-width': i === 0 ? 1.8 : 0.9 }, gGrid);
        el('path', { d: pathV, fill: 'none', stroke: css('--border'),
                     'stroke-width': i === 0 ? 1.8 : 0.9 }, gGrid);
      }
      // the unit square
      var sq = [[0, 0], [1, 0], [1, 1], [0, 1]].map(M);
      var dsq = sq.map(function (p, k) { return (k ? 'L' : 'M') + F.X(p[0]) + ',' + F.Y(p[1]); }).join('') + 'Z';
      el('path', { d: dsq, fill: css('--accent'), opacity: .22,
                   stroke: css('--accent'), 'stroke-width': 2 }, gShape);

      // eigen decomposition of the FULL matrix (not the interpolation)
      var tr = a + d, det = a * d - b * c, disc = tr * tr / 4 - det;
      if (disc >= 0) {
        var s = Math.sqrt(disc), l1 = tr / 2 + s, l2 = tr / 2 - s;
        [[l1, '--amber'], [l2, '--green']].forEach(function (L) {
          var lam = L[0], vx, vy;
          if (Math.abs(b) > 1e-9) { vx = b; vy = lam - a; }
          else if (Math.abs(c) > 1e-9) { vx = lam - d; vy = c; }
          else { vx = (Math.abs(lam - a) < 1e-9) ? 1 : 0; vy = 1 - vx; }
          var n = Math.hypot(vx, vy) || 1; vx /= n; vy /= n;
          var scale = 1 + t * (lam - 1);
          el('line', { x1: F.X(-vx * 2.8), y1: F.Y(-vy * 2.8),
                       x2: F.X(vx * 2.8), y2: F.Y(vy * 2.8),
                       stroke: css(L[1]), 'stroke-width': 1.3,
                       'stroke-dasharray': '5 4', opacity: .8 }, gEig);
          el('line', { x1: F.X(0), y1: F.Y(0),
                       x2: F.X(vx * scale), y2: F.Y(vy * scale),
                       stroke: css(L[1]), 'stroke-width': 3.4 }, gEig);
        });
        r2.textContent = 'eigenvalues λ = ' + l1.toFixed(2) + ' (amber),  ' +
                          l2.toFixed(2) + ' (green)   —   dashed lines never turn';
      } else {
        r2.textContent = 'no real eigenvalues — this matrix rotates every direction';
      }
      r1.textContent = 'A = [' + a.toFixed(2) + '  ' + b.toFixed(2) + ' ; ' +
                        c.toFixed(2) + '  ' + d.toFixed(2) + ']   det = ' +
                        (a * d - b * c).toFixed(2);
    }
    draw();
    var ct = ctl(host);
    slider(ct, 'apply', 0, 1, 0.02, 1, function (v) { t = v; draw(); return v.toFixed(2); });
    slider(ct, 'a', -2, 2.5, 0.05, 1.4, function (v) { a = v; draw(); return v.toFixed(2); });
    slider(ct, 'b', -2, 2, 0.05, 0.6, function (v) { b = v; draw(); return v.toFixed(2); });
    slider(ct, 'c', -2, 2, 0.05, 0.4, function (v) { c = v; draw(); return v.toFixed(2); });
    slider(ct, 'd', -2, 2.5, 0.05, 1.1, function (v) { d = v; draw(); return v.toFixed(2); });
    cap(host, 'Slide "apply" from 0 to 1 to watch the matrix bend space. The blue square is where ' +
              'the unit square lands — its area is the determinant. The two dashed lines are the ' +
              'eigenvectors: everything else in the picture turns, they never do. Set b and c to 0 ' +
              'and only stretching remains. Set a=0.2, b=-1.4, c=1.4, d=0.2 and the eigenvalues go ' +
              'complex — a pure rotation has no direction it leaves alone.');
  }

  /* =================================================================
     registry
     ================================================================= */
  /* =================================================================
     11. BAYES — the base-rate picture, 2000 dots
     ================================================================= */
  function vizBayes(host) {
    var COLS = 50, ROWS = 40, N = COLS * ROWS;
    var W = 660, H = 400, pad = 18, cell = (W - 2 * pad) / COLS;
    var svg = makeSVG(host, W, H);
    var g = el('g', {}, svg);
    var dots = [];
    for (var i = 0; i < N; i++) {
      dots.push(el('rect', {
        x: pad + (i % COLS) * cell + 0.6,
        y: pad + Math.floor(i / COLS) * cell + 0.6,
        width: cell - 1.2, height: cell - 1.2, rx: 1.4
      }, g));
    }
    var r1 = el('text', { x: pad, y: H - 52, class: 's-lbl' }, svg);
    var r2 = el('text', { x: pad, y: H - 34, class: 's-lbl' }, svg);
    var r3 = el('text', { x: pad, y: H - 12, class: 's-lbl', 'font-weight': 700 }, svg);

    var prev = 0.001, sens = 0.99, spec = 0.99;

    function draw() {
      var nSick = Math.round(N * prev);
      var nTP = Math.round(nSick * sens);
      var nFN = nSick - nTP;
      var nWell = N - nSick;
      var nFP = Math.round(nWell * (1 - spec));
      var cSick = css('--red'), cTP = css('--amber'), cFP = css('--purple'),
          cWell = css('--border');
      for (var i = 0; i < N; i++) {
        var f, o = 1;
        if (i < nTP)             { f = cTP;   }          // sick + positive
        else if (i < nSick)      { f = cSick; }          // sick + negative (missed)
        else if (i < nSick + nFP){ f = cFP;   }          // well + positive (false alarm)
        else                     { f = cWell; o = .5; }  // well + negative
        dots[i].setAttribute('fill', f);
        dots[i].setAttribute('opacity', o);
      }
      var pos = nTP + nFP;
      var ppv = pos ? nTP / pos : 0;
      r1.textContent = 'out of ' + N + ' people:  ' + nSick + ' actually have it  ·  ' +
                        nTP + ' true positives (amber)  ·  ' + nFN + ' missed (red)';
      r2.textContent = nFP + ' healthy people test positive anyway (purple)  →  ' +
                        pos + ' positive results in total';
      r3.textContent = 'P(disease | positive) = ' + nTP + ' / ' + pos + ' = ' +
                        (ppv * 100).toFixed(1) + '%';
      r3.setAttribute('fill', css(ppv < 0.3 ? '--red' : ppv < 0.7 ? '--amber' : '--green'));
    }
    draw();
    var c = ctl(host);
    slider(c, 'prevalence', 0.0005, 0.25, 0.0005, 0.001, function (v) {
      prev = v; draw(); return (v * 100).toFixed(2) + '%';
    });
    slider(c, 'sensitivity', 0.5, 1, 0.005, 0.99, function (v) {
      sens = v; draw(); return (v * 100).toFixed(1) + '%';
    });
    slider(c, 'specificity', 0.5, 1, 0.005, 0.99, function (v) {
      spec = v; draw(); return (v * 100).toFixed(1) + '%';
    });
    cap(host, 'A 99% accurate test on a 1-in-1000 disease. Both sliders are at 99% and the answer is ' +
              'still about 9%, because there are simply so many more healthy people that even a 1% ' +
              'false-alarm rate produces ten times more false positives than there are true cases. ' +
              'Now drag prevalence to 10% and watch the same test become trustworthy. The test did ' +
              'not change — the base rate did. Substitute "fraud" for "disease" and this is your ' +
              'review queue.');
  }

  /* =================================================================
     12. CONV — a 3x3 kernel sliding over an image, building a feature map
     ================================================================= */
  function vizConv(host) {
    var W = 720, H = 330;
    var svg = makeSVG(host, W, H);
    var N = 9, K = 3, cell = 26, gap = 2;

    /* input "image": a bright vertical bar and a bright horizontal bar */
    var img = [], r, c;
    for (r = 0; r < N; r++) {
      img[r] = [];
      for (c = 0; c < N; c++) {
        var v = 0.12;
        if (c >= 3 && c <= 4) v = 0.92;          /* vertical stroke  */
        if (r >= 5 && r <= 6) v = 0.85;          /* horizontal stroke */
        if (c >= 3 && c <= 4 && r >= 5 && r <= 6) v = 1.0;
        img[r][c] = v;
      }
    }

    var KERNELS = {
      'vertical edge': [[1, 0, -1], [2, 0, -2], [1, 0, -1]],
      'horizontal edge': [[1, 2, 1], [0, 0, 0], [-1, -2, -1]],
      'blur': [[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]],
      'sharpen': [[0, -1, 0], [-1, 5, -1], [0, -1, 0]]
    };
    var kname = 'vertical edge', stride = 1;

    function ker() { return KERNELS[kname]; }
    function outSize() { return Math.floor((N - K) / stride) + 1; }

    function convAt(or_, oc) {
      var s = 0, kr, kc, k = ker();
      for (kr = 0; kr < K; kr++)
        for (kc = 0; kc < K; kc++)
          s += img[or_ * stride + kr][oc * stride + kc] * k[kr][kc];
      return s;
    }

    var gIn = el('g', {}, svg), gK = el('g', {}, svg), gOut = el('g', {}, svg);
    var x0 = 20, y0 = 46, kx = x0 + N * (cell + gap) + 52, ky = y0 + 26;
    var ox = kx + K * (cell + gap) + 62, oy = y0;

    el('text', { x: x0, y: 30, class: 's-lbl' }, svg).textContent = 'input (9x9)';
    el('text', { x: kx, y: 30, class: 's-lbl' }, svg).textContent = 'kernel';
    var outLbl = el('text', { x: ox, y: 30, class: 's-lbl' }, svg);

    var pos = 0, timer = null, running = false;

    function grey(v) {
      var g = Math.round(255 * Math.max(0, Math.min(1, v)));
      return 'rgb(' + g + ',' + g + ',' + g + ')';
    }

    function draw() {
      while (gIn.firstChild) gIn.removeChild(gIn.firstChild);
      while (gK.firstChild) gK.removeChild(gK.firstChild);
      while (gOut.firstChild) gOut.removeChild(gOut.firstChild);

      var O = outSize();
      if (pos >= O * O) pos = 0;
      var or_ = Math.floor(pos / O), oc = pos % O;
      outLbl.textContent = 'feature map (' + O + 'x' + O + ')';

      for (r = 0; r < N; r++) for (c = 0; c < N; c++) {
        el('rect', {
          x: x0 + c * (cell + gap), y: y0 + r * (cell + gap),
          width: cell, height: cell, fill: grey(img[r][c]),
          stroke: css('--border'), 'stroke-width': 0.5
        }, gIn);
      }
      /* the sliding window */
      el('rect', {
        x: x0 + oc * stride * (cell + gap) - 2, y: y0 + or_ * stride * (cell + gap) - 2,
        width: K * (cell + gap) + 2, height: K * (cell + gap) + 2,
        fill: 'none', stroke: css('--amber'), 'stroke-width': 3, rx: 3
      }, gIn);

      var k = ker();
      for (r = 0; r < K; r++) for (c = 0; c < K; c++) {
        el('rect', {
          x: kx + c * (cell + gap), y: ky + r * (cell + gap),
          width: cell, height: cell, fill: css('--panel'),
          stroke: css('--amber'), 'stroke-width': 1, rx: 2
        }, gK);
        var t = el('text', {
          x: kx + c * (cell + gap) + cell / 2, y: ky + r * (cell + gap) + cell / 2 + 4,
          class: 's-lbl-s', 'text-anchor': 'middle'
        }, gK);
        t.textContent = Math.abs(k[r][c]) < 0.2 && k[r][c] !== 0 ? '⅑' : String(k[r][c]);
      }

      var vals = [], i, j, mx = 0.001;
      for (i = 0; i < O; i++) { vals[i] = []; for (j = 0; j < O; j++) {
        vals[i][j] = convAt(i, j); mx = Math.max(mx, Math.abs(vals[i][j])); } }

      for (i = 0; i < O; i++) for (j = 0; j < O; j++) {
        var done = (i * O + j) <= pos;
        el('rect', {
          x: ox + j * (cell + gap), y: oy + i * (cell + gap),
          width: cell, height: cell,
          fill: done ? grey(Math.abs(vals[i][j]) / mx) : css('--bg'),
          stroke: (i === or_ && j === oc) ? css('--amber') : css('--border'),
          'stroke-width': (i === or_ && j === oc) ? 3 : 0.5, rx: 2
        }, gOut);
      }

      var rd = el('text', { x: x0, y: H - 16, class: 's-lbl' }, gOut);
      rd.textContent = 'window at row ' + or_ + ', col ' + oc +
                       '  ->  sum of 9 products = ' + convAt(or_, oc).toFixed(2);
    }

    function stop() { running = false; clearInterval(timer); }
    draw();

    var ct = ctl(host);
    button(ct, '▶ slide', function () {
      if (running) { stop(); return; }
      running = true;
      timer = setInterval(function () {
        var O = outSize();
        pos++; if (pos >= O * O) { pos = 0; }
        draw();
      }, 220);
    });
    button(ct, 'step', function () { stop(); pos++; draw(); });
    button(ct, 'reset', function () { stop(); pos = 0; draw(); });
    var sel = h('label', null, 'kernel ', ct);
    var s = document.createElement('select');
    Object.keys(KERNELS).forEach(function (n) {
      var o = document.createElement('option'); o.value = n; o.textContent = n; s.appendChild(o);
    });
    s.addEventListener('change', function () { kname = s.value; pos = 0; draw(); });
    sel.appendChild(s);
    slider(ct, 'stride', 1, 3, 1, 1, function (v) { stride = v; pos = 0; draw(); return String(v); });

    cap(host, 'The same nine numbers are reused at every position — that is parameter sharing, and ' +
              'it is why a conv layer has 9 weights where a dense layer would have 81 per output. ' +
              'Switch to "horizontal edge" and watch which stroke lights up: the kernel is a learned ' +
              'feature detector, and its output is bright exactly where the pattern it looks for ' +
              'occurs. Raise the stride and the feature map shrinks — that is downsampling without ' +
              'pooling.');
  }

  /* =================================================================
     13. ATTENTION — a full self-attention matrix, with the sqrt(d)
         scaling and the causal mask as toggles
     ================================================================= */
  function vizAttention(host) {
    var W = 720, H = 420;
    var svg = makeSVG(host, W, H);

    var TOK = ['The', 'cat', 'sat', 'on', 'the', 'mat', 'and', 'it'];
    var n = TOK.length, cell = 40, x0 = 150, y0 = 66;

    /* hand-authored raw affinities: what a trained head might plausibly learn */
    var RULES = [
      [0, 1, 2.8], [1, 2, 1.2], [2, 1, 2.6], [2, 3, 1.6], [3, 5, 2.4],
      [4, 5, 2.8], [5, 3, 1.0], [6, 2, 1.0], [7, 1, 3.0], [7, 5, 2.4]
    ];
    var S = [], i, j;
    for (i = 0; i < n; i++) {
      S[i] = [];
      for (j = 0; j < n; j++) S[i][j] = 0.2 + (i === j ? 0.8 : 0);
    }
    RULES.forEach(function (r) { S[r[0]][r[1]] += r[2]; });

    var d = 64, scaled = true, causal = true;

    function weights(row) {
      var g = scaled ? 1 : Math.sqrt(d);
      var lo = [], m = -Infinity, k;
      for (k = 0; k < n; k++) {
        lo[k] = (causal && k > row) ? -Infinity : S[row][k] * g;
        if (lo[k] > m) m = lo[k];
      }
      var s = 0, e = [];
      for (k = 0; k < n; k++) { e[k] = lo[k] === -Infinity ? 0 : Math.exp(lo[k] - m); s += e[k]; }
      for (k = 0; k < n; k++) e[k] /= s;
      return e;
    }

    function txt(x, y, cls, anchor, s) {
      var t = el('text', { x: x, y: y, class: cls, 'text-anchor': anchor || 'start' }, svg);
      t.textContent = s;
      return t;
    }

    /* static labels */
    for (j = 0; j < n; j++) {
      txt(x0 + j * cell + (cell - 2) / 2, y0 - 10, 's-lbl-s', 'middle', TOK[j]);
    }
    txt(x0, y0 - 32, 's-lbl', 'start', 'keys — what this position reads from');
    txt(12, y0 - 32, 's-lbl', 'start', 'queries');

    var cells = [], texts = [];
    for (i = 0; i < n; i++) {
      txt(x0 - 10, y0 + i * cell + cell / 2 + 4, 's-lbl-s', 'end', TOK[i]);
      cells[i] = []; texts[i] = [];
      for (j = 0; j < n; j++) {
        cells[i][j] = el('rect', {
          x: x0 + j * cell, y: y0 + i * cell, width: cell - 2, height: cell - 2,
          rx: 3, fill: css('--accent'), stroke: css('--line')
        }, svg);
        var t = el('text', {
          x: x0 + j * cell + (cell - 2) / 2, y: y0 + i * cell + cell / 2 + 3,
          'text-anchor': 'middle', 'font-size': 9.5, fill: css('--tx')
        }, svg);
        texts[i][j] = t;
      }
    }

    var r1 = txt(12, y0 + n * cell + 26, 's-lbl', 'start', '');
    var r2 = txt(12, y0 + n * cell + 48, 's-lbl', 'start', '');

    function draw() {
      var maxLast = 0;
      for (i = 0; i < n; i++) {
        var w = weights(i);
        for (j = 0; j < n; j++) {
          var masked = causal && j > i;
          cells[i][j].setAttribute('fill', css(masked ? '--bg2' : '--accent'));
          cells[i][j].setAttribute('fill-opacity', masked ? 0.35 : Math.max(0.04, w[j]));
          texts[i][j].textContent = masked ? '' : (w[j] >= 0.01 ? w[j].toFixed(2) : '');
          texts[i][j].setAttribute('fill', css(w[j] > 0.55 ? '--bg' : '--tx-dim'));
        }
        if (i === n - 1) maxLast = Math.max.apply(null, w);
      }
      r1.textContent = 'head dim d = ' + d + '   ·   ' +
        (scaled ? 'logits divided by √d = ' + Math.sqrt(d).toFixed(1) : 'NO √d scaling') +
        '   ·   ' + (causal ? 'causal mask ON' : 'causal mask OFF (BERT-style)');
      r2.textContent = 'largest weight in the "it" row: ' + maxLast.toFixed(3) +
        (maxLast > 0.9 ? '   ← softmax has collapsed to one-hot; gradients here are ~0'
                       : '   ← a soft blend, which is what you want');
      r2.setAttribute('fill', css(maxLast > 0.9 ? '--red' : '--tx-dim'));
    }
    draw();

    var c = ctl(host);
    slider(c, 'head dim d', 2, 9, 1, 6, function (v) {
      d = Math.pow(2, v); draw(); return String(d);
    });
    var bScale = button(c, '÷ √d: on', function () {
      scaled = !scaled;
      bScale.textContent = scaled ? '÷ √d: on' : '÷ √d: OFF';
      draw();
    });
    var bMask = button(c, 'causal mask: on', function () {
      causal = !causal;
      bMask.textContent = 'causal mask: ' + (causal ? 'on' : 'off');
      draw();
    });

    cap(host, 'Each row is one query token asking "who should I read from?", and each row sums to 1. ' +
              'Turn off the √d scaling and raise d: the logits grow with dimension, the softmax ' +
              'saturates to a single 1.00, and every other gradient goes to zero — that is the ' +
              'entire reason for dividing by √d. Turn off the causal mask and the upper triangle ' +
              'lights up: the model can now see the future, which is fine for BERT and fatal for GPT.');
  }

  /* =================================================================
     15. KV CACHE — why memory, not FLOPs, caps your batch size
     ================================================================= */
  function vizKVCache(host) {
    var W = 720, H = 300;
    var svg = makeSVG(host, W, H);

    /* Llama-2-13B shape */
    var L = 40, HEAD = 128, QH = 40, PARAMS = 13e9;
    var CAP = 80;                       /* GB of HBM on the card */
    var Wg = PARAMS * 2 / 1e9;          /* fp16 weights, GB */

    var ctx = 4096, batch = 8, kvHeads = QH, kvBytes = 2;

    var bx = 60, by = 108, bw = W - 120, bh = 44;

    function txt(x, y, cls, anchor, s) {
      var t = el('text', { x: x, y: y, class: cls, 'text-anchor': anchor || 'start' }, svg);
      t.textContent = s;
      return t;
    }
    function fmt(x) { return x < 10 ? x.toFixed(2) : x.toFixed(1); }

    function draw() {
      while (svg.firstChild) svg.removeChild(svg.firstChild);

      var perTok = 2 * L * kvHeads * HEAD * kvBytes;       /* bytes per token per sequence */
      var perSeq = perTok * ctx / 1e9;                     /* GB */
      var kvTotal = perSeq * batch;
      var total = Wg + kvTotal;
      var fits = Math.max(0, Math.floor((CAP - Wg) / perSeq));

      txt(bx, 40, 's-lbl', 'start', 'One 80 GB GPU · Llama-2-13B · fp16 weights');
      txt(bx + bw, 40, 's-lbl', 'end', fmt(total) + ' GB / 80 GB');

      /* capacity track */
      el('rect', { x: bx, y: by, width: bw, height: bh, rx: 4,
                   fill: css('--bg-2'), stroke: css('--border'), 'stroke-width': 1 }, svg);

      var wpx = Math.min(bw, bw * Wg / CAP);
      el('rect', { x: bx, y: by, width: wpx, height: bh, rx: 4,
                   fill: css('--amber'), opacity: 0.55 }, svg);
      txt(bx + wpx / 2, by + bh / 2 + 4, 's-lbl-s', 'middle', 'weights ' + fmt(Wg));

      var over = total > CAP;
      var kvpx = Math.min(bw - wpx, bw * kvTotal / CAP);
      el('rect', { x: bx + wpx, y: by, width: Math.max(0, kvpx), height: bh,
                   fill: css(over ? '--red' : '--accent'), opacity: over ? 0.75 : 0.6 }, svg);
      if (kvpx > 62) {
        txt(bx + wpx + kvpx / 2, by + bh / 2 + 4, 's-lbl-s', 'middle',
            'KV cache ' + fmt(kvTotal));
      }

      /* ticks */
      for (var g = 0; g <= 80; g += 20) {
        var gx = bx + bw * g / CAP;
        el('line', { x1: gx, y1: by + bh, x2: gx, y2: by + bh + 6,
                     stroke: css('--border-2'), 'stroke-width': 1 }, svg);
        txt(gx, by + bh + 20, 's-lbl-s', 'middle', g + ' GB');
      }

      txt(bx, 196, 's-lbl', 'start',
          'per token: ' + (perTok / 1024).toFixed(0) + ' KB   ·   per sequence @ ' + ctx +
          ' ctx: ' + fmt(perSeq) + ' GB   ·   × ' + batch + ' concurrent = ' + fmt(kvTotal) + ' GB');
      txt(bx, 224, 's-lbl', 'start',
          over
            ? 'OUT OF MEMORY — this batch does not fit. Max concurrent sequences: ' + fits
            : 'fits, with room for ' + fits + ' concurrent sequences in total');
      txt(bx, 252, 's-lbl-s', 'start',
          'KV = 2 × L × n × (kv heads × head dim) × bytes   ·   L = 40, head dim = 128');
    }

    var c = ctl(host);
    slider(c, 'context', 9, 15, 1, 12, function (v) {
      ctx = Math.pow(2, v); draw(); return ctx + ' tokens';
    });
    slider(c, 'batch', 1, 64, 1, 8, function (v) {
      batch = v; draw(); return v + ' seqs';
    });
    var bH = button(c, 'KV heads: 40 (MHA)', function () {
      kvHeads = kvHeads === QH ? 8 : QH;
      bH.textContent = 'KV heads: ' + kvHeads + (kvHeads === QH ? ' (MHA)' : ' (GQA)');
      draw();
    });
    var bP = button(c, 'KV precision: fp16', function () {
      kvBytes = kvBytes === 2 ? 1 : 2;
      bP.textContent = 'KV precision: ' + (kvBytes === 2 ? 'fp16' : 'INT8');
      draw();
    });

    draw();

    cap(host, 'The weights are a fixed 26 GB and they were never the problem — the KV cache is. It ' +
              'grows linearly in both context length and batch size, so it, not the model, decides ' +
              'how many users you can serve at once. Drag context to 16K and the batch collapses. ' +
              'Then switch to GQA (8 KV heads instead of 40): the cache shrinks 5× and the batch you ' +
              'just lost comes straight back — which is exactly why every modern model ships with it. ' +
              'Quantising the cache to INT8 halves it again.');
  }

  /* =================================================================
     16. POINT-IN-TIME — the as-of join, and the leak you get without it
     ================================================================= */
  function vizPIT(host) {
    var W = 720, H = 320;
    var svg = makeSVG(host, W, H);
    var box = { l: 70, t: 46, w: W - 130, h: 108 };

    /* card_chargeback_count over time. The day-25 bump is CAUSED by the day-22 txn. */
    var STEPS = [
      { t: 0,  v: 0, why: '' },
      { t: 12, v: 1, why: 'a genuine earlier chargeback' },
      { t: 25, v: 2, why: 'raised BECAUSE of this transaction' },
      { t: 33, v: 3, why: 'later investigation' }
    ];
    var T = 40, VMAX = 3.6;
    var evt = 22, asof = true;

    function X(t) { return box.l + t / T * box.w; }
    function Y(v) { return box.t + box.h - v / VMAX * box.h; }
    function txt(x, y, cls, anchor, s) {
      var e = el('text', { x: x, y: y, class: cls, 'text-anchor': anchor || 'start' }, svg);
      e.textContent = s;
      return e;
    }
    function valueAt(t) {
      var v = 0, i;
      for (i = 0; i < STEPS.length; i++) if (STEPS[i].t <= t) v = STEPS[i].v;
      return v;
    }

    function draw() {
      while (svg.firstChild) svg.removeChild(svg.firstChild);

      var correct = valueAt(evt);
      var latest = STEPS[STEPS.length - 1].v;
      var used = asof ? correct : latest;

      txt(box.l, 26, 's-lbl', 'start', 'feature:  card_chargeback_count   (entity: card 42)');

      /* the unknowable future, shaded */
      el('rect', { x: X(evt), y: box.t - 6, width: box.l + box.w - X(evt), height: box.h + 6,
                   fill: css('--red'), opacity: asof ? 0.07 : 0.14 }, svg);
      txt(box.l + box.w, box.t + 8, 's-lbl-s', 'end', 'not knowable on day ' + evt);

      /* axes */
      el('line', { x1: box.l, y1: box.t + box.h, x2: box.l + box.w, y2: box.t + box.h,
                   class: 's-axis' }, svg);
      for (var g = 0; g <= T; g += 10) {
        el('line', { x1: X(g), y1: box.t + box.h, x2: X(g), y2: box.t + box.h + 5,
                     stroke: css('--border-2'), 'stroke-width': 1 }, svg);
        txt(X(g), box.t + box.h + 19, 's-lbl-s', 'middle', 'day ' + g);
      }

      /* staircase of feature values */
      var d = '', i;
      for (i = 0; i < STEPS.length; i++) {
        var x1 = X(STEPS[i].t), x2 = X(i + 1 < STEPS.length ? STEPS[i + 1].t : T);
        d += (i ? ' L' + x1 + ',' + Y(STEPS[i].v) : 'M' + x1 + ',' + Y(STEPS[i].v));
        d += ' L' + x2 + ',' + Y(STEPS[i].v);
      }
      el('path', { d: d, fill: 'none', stroke: css('--accent'), 'stroke-width': 2.4 }, svg);

      for (i = 0; i < STEPS.length; i++) {
        var s = STEPS[i], future = s.t > evt;
        el('circle', { cx: X(s.t), cy: Y(s.v), r: 4.5,
                       fill: css(future ? '--red' : '--accent') }, svg);
        txt(X(s.t), Y(s.v) - 11, 's-lbl-s', 'middle', '= ' + s.v);
        if (s.why) txt(X(s.t), Y(s.v) - 24, 's-lbl-s', 'middle', s.why);
      }

      /* the training row */
      el('line', { x1: X(evt), y1: box.t - 6, x2: X(evt), y2: box.t + box.h + 30,
                   stroke: css('--amber'), 'stroke-width': 2, 'stroke-dasharray': '5 4' }, svg);
      txt(X(evt), box.t - 14, 's-lbl', 'middle', 'training row · event time');

      /* arrow from the row to the value the join actually picks */
      var px = asof ? X(valueAtT(evt)) : X(STEPS[STEPS.length - 1].t);
      el('path', { d: 'M' + X(evt) + ',' + (box.t + box.h + 40) + ' L' + px + ',' +
                       (box.t + box.h + 40) + ' L' + px + ',' + (Y(used) + 8),
                   fill: 'none', stroke: css(asof ? '--green' : '--red'),
                   'stroke-width': 2 }, svg);
      el('circle', { cx: px, cy: Y(used) + 8, r: 3, fill: css(asof ? '--green' : '--red') }, svg);

      txt(box.l, 246, 's-lbl', 'start',
          asof ? 'AS-OF JOIN:  last value where feature_time ≤ event_time  →  feature = ' + correct
               : 'NAIVE JOIN:  join on card_id, take the current value  →  feature = ' + latest);
      var verdict = txt(box.l, 270, 's-lbl', 'start',
          asof ? 'correct — this is what the model could actually have known on day ' + evt
               : 'LEAK — ' + (latest - correct) + ' of those chargebacks happened AFTER day ' + evt +
                 ', one of them caused by this very transaction');
      verdict.setAttribute('fill', css(asof ? '--green' : '--red'));
      txt(box.l, 294, 's-lbl-s', 'start',
          'offline AUC rises, production AUC falls, and no error is raised anywhere in the pipeline.');
    }
    function valueAtT(t) {
      var best = 0, i;
      for (i = 0; i < STEPS.length; i++) if (STEPS[i].t <= t) best = STEPS[i].t;
      return best;
    }

    var c = ctl(host);
    slider(c, 'event day', 1, 39, 1, 22, function (v) { evt = v; draw(); return 'day ' + v; });
    var bJ = button(c, 'join: as-of (correct)', function () {
      asof = !asof;
      bJ.textContent = 'join: ' + (asof ? 'as-of (correct)' : 'NAIVE (leaks)');
      draw();
    });

    draw();

    cap(host, 'The blue staircase is the feature\'s real history. The gold line is one training row. ' +
              'The as-of join walks LEFT from the row and takes the last value that existed by then. ' +
              'The naive join — the one plain SQL gives you — takes the value that exists today, ' +
              'which on day 25 includes a chargeback raised because of this very transaction. Drag ' +
              'the event day: before day 12 the correct feature is 0, and the naive join still says 3.');
  }

  /* =================================================================
     17. CANARY — staged rollout, and why you alert per variant
     ================================================================= */
  function vizCanary(host) {
    var W = 720, H = 340;
    var svg = makeSVG(host, W, H);
    var box = { l: 58, t: 96, w: W - 100, h: 150 };

    var STAGES = [1, 5, 25, 100], HOLD = 30, TMAX = 4 * HOLD;
    var THR = 1.0, OLD = 0.40;

    var t = 0, hist = [], timer = null, rolled = null;
    var healthy = false, perVariant = true;

    function X(i) { return box.l + i / TMAX * box.w; }
    function Y(v) { return box.t + box.h - Math.min(v, 3) / 3 * box.h; }
    function txt(x, y, cls, anchor, s) {
      var e = el('text', { x: x, y: y, class: cls, 'text-anchor': anchor || 'start' }, svg);
      e.textContent = s;
      return e;
    }
    function stageAt(i) { return STAGES[Math.min(STAGES.length - 1, Math.floor(i / HOLD))]; }
    function noise(i) { return 0.06 * Math.sin(i * 1.7) + 0.04 * Math.sin(i * 0.53); }

    function tick() {
      if (rolled || t > TMAX) { stop(); return; }
      var pct = stageAt(t) / 100;
      var nv = (healthy ? 0.44 : 2.60) + noise(t);
      var bl = OLD * (1 - pct) + nv * pct + noise(t + 9) * 0.4;
      hist.push({ t: t, pct: pct, nv: nv, bl: bl });

      var watched = perVariant ? nv : bl;
      var over = 0, k;
      for (k = Math.max(0, hist.length - 4); k < hist.length; k++) {
        if ((perVariant ? hist[k].nv : hist[k].bl) > THR) over++;
      }
      if (over >= 4) rolled = { t: t, pct: stageAt(t), watched: watched };
      t++;
      draw();
    }
    function stop() { clearInterval(timer); timer = null; }
    function reset() { stop(); t = 0; hist = []; rolled = null; draw(); }

    function draw() {
      while (svg.firstChild) svg.removeChild(svg.firstChild);

      var pct = rolled ? 0 : (hist.length ? hist[hist.length - 1].pct * 100 : STAGES[0]);

      /* traffic split bar */
      var bx = box.l, bw = box.w, by = 40, bh = 26;
      el('rect', { x: bx, y: by, width: bw, height: bh, rx: 4, fill: css('--bg-2'),
                   stroke: css('--border') }, svg);
      el('rect', { x: bx, y: by, width: bw * (1 - pct / 100), height: bh, rx: 4,
                   fill: css('--tx-faint'), opacity: 0.35 }, svg);
      el('rect', { x: bx + bw * (1 - pct / 100), y: by, width: bw * pct / 100, height: bh,
                   fill: css(rolled ? '--red' : '--green'), opacity: 0.65 }, svg);
      txt(bx + 8, by + bh / 2 + 4, 's-lbl-s', 'start', 'current model');
      txt(bx + bw - 8, by + bh / 2 + 4, 's-lbl-s', 'end',
          'new model  ' + pct.toFixed(0) + '%');
      txt(bx, 26, 's-lbl', 'start', 'traffic split');

      /* stage boundaries */
      var s;
      for (s = 0; s < STAGES.length; s++) {
        var sx = X(s * HOLD);
        el('line', { x1: sx, y1: box.t, x2: sx, y2: box.t + box.h,
                     stroke: css('--border'), 'stroke-width': 1, 'stroke-dasharray': '2 5' }, svg);
        txt(sx + 4, box.t + 12, 's-lbl-s', 'start', STAGES[s] + '%');
      }

      /* axes + threshold */
      el('line', { x1: box.l, y1: box.t + box.h, x2: box.l + box.w, y2: box.t + box.h,
                   class: 's-axis' }, svg);
      el('line', { x1: box.l, y1: Y(THR), x2: box.l + box.w, y2: Y(THR),
                   stroke: css('--amber'), 'stroke-width': 1.5, 'stroke-dasharray': '6 4' }, svg);
      txt(box.l + box.w, Y(THR) - 6, 's-lbl-s', 'end', 'rollback threshold 1.0 %');
      txt(box.l - 8, box.t + 8, 's-lbl-s', 'end', '3 %');
      txt(box.l - 8, box.t + box.h, 's-lbl-s', 'end', '0');
      txt(box.l, box.t - 8, 's-lbl', 'start', 'error rate');

      /* the two series */
      function line(key, colour, width, dash) {
        if (hist.length < 2) return;
        var d = '', i;
        for (i = 0; i < hist.length; i++) {
          d += (i ? ' L' : 'M') + X(hist[i].t).toFixed(1) + ',' + Y(hist[i][key]).toFixed(1);
        }
        var a = { d: d, fill: 'none', stroke: css(colour), 'stroke-width': width };
        if (dash) a['stroke-dasharray'] = dash;
        el('path', a, svg);
      }
      line('bl', perVariant ? '--tx-faint' : '--accent', perVariant ? 1.6 : 2.6, '4 3');
      line('nv', perVariant ? '--accent' : '--tx-faint', perVariant ? 2.6 : 1.6, null);

      var last = hist.length ? hist[hist.length - 1] : null;
      if (last) {
        el('circle', { cx: X(last.t), cy: Y(last.nv), r: 3.5, fill: css('--accent') }, svg);
      }

      if (rolled) {
        el('line', { x1: X(rolled.t), y1: box.t, x2: X(rolled.t), y2: box.t + box.h,
                     stroke: css('--red'), 'stroke-width': 2 }, svg);
        txt(X(rolled.t) + 6, box.t + 30, 's-lbl-s', 'start', 'rollback');
      }

      txt(box.l, 276, 's-lbl', 'start',
          'watching: ' + (perVariant ? 'the NEW VARIANT\'s own error rate'
                                     : 'the OVERALL blended error rate') +
          (last ? '   ·   new ' + last.nv.toFixed(2) + ' %   ·   blended ' +
                  last.bl.toFixed(2) + ' %' : ''));
      var v = txt(box.l, 300, 's-lbl', 'start',
          rolled
            ? 'ROLLED BACK at ' + rolled.pct + ' % traffic — blast radius: ' + rolled.pct +
              ' % of requests saw the bad model'
            : (healthy ? 'healthy — promoting through the stages'
                       : (t >= TMAX ? 'reached 100 % with a broken model and never tripped the alert'
                                    : 'rolling out…')));
      v.setAttribute('fill', css(rolled ? '--red' : (healthy ? '--green' : '--tx-dim')));
      txt(box.l, 324, 's-lbl-s', 'start',
          'blended error = old × (1 − p) + new × p — at p = 1 % a broken model moves it by 0.02 points.');
    }

    var c = ctl(host);
    button(c, '▶ roll out', function () {
      if (timer) { stop(); return; }
      if (rolled || t > TMAX) reset();
      timer = setInterval(tick, 60);
    });
    button(c, 'reset', reset);
    var bH = button(c, 'new model: BROKEN', function () {
      healthy = !healthy;
      bH.textContent = 'new model: ' + (healthy ? 'healthy' : 'BROKEN');
      reset();
    });
    var bV = button(c, 'alert on: per-variant', function () {
      perVariant = !perVariant;
      bV.textContent = 'alert on: ' + (perVariant ? 'per-variant' : 'blended');
      reset();
    });

    draw();

    cap(host, 'Press roll out. With per-variant alerting the broken model trips the threshold during ' +
              'the 1 % stage and 1 % of requests were affected. Now switch to blended alerting and ' +
              'roll out again: the bad model is diluted by 99 % of healthy traffic, the overall error ' +
              'rate stays under the threshold through 1 %, 5 % and 25 %, and it only trips once ' +
              'everyone is on it. Same model, same threshold, hundredfold difference in blast radius — ' +
              'which is why canary metrics must be computed per variant, not over all traffic.');
  }

  /* =================================================================
     18. PSI — population stability, bin by bin
     ================================================================= */
  function vizPSI(host) {
    var W = 720, H = 350;
    var svg = makeSVG(host, W, H);
    var box = { l: 58, t: 46, w: W - 100, h: 150 };
    var NB = 10, EPS = 1e-4;

    var shift = 0, spread = 1, bug = false;

    function txt(x, y, cls, anchor, s) {
      var e = el('text', { x: x, y: y, class: cls, 'text-anchor': anchor || 'start' }, svg);
      e.textContent = s;
      return e;
    }
    function gauss(mu, sd) {
      var p = [], i, c, sum = 0;
      for (i = 0; i < NB; i++) {
        c = (i + 0.5) / NB;
        p[i] = Math.exp(-Math.pow(c - mu, 2) / (2 * sd * sd));
        sum += p[i];
      }
      for (i = 0; i < NB; i++) p[i] = Math.max(EPS, p[i] / sum);
      return p;
    }

    function draw() {
      while (svg.firstChild) svg.removeChild(svg.firstChild);

      var ref = gauss(0.45, 0.13), cur, i;
      if (bug) {
        cur = gauss(0.45, 0.13);
        var moved = 0;
        for (i = 1; i < NB; i++) { moved += cur[i] * 0.13; cur[i] *= 0.87; }
        cur[0] += moved;                       /* a default value appearing in bin 0 */
      } else {
        cur = gauss(0.45 + shift, 0.13 * spread);
      }

      var contrib = [], psi = 0;
      for (i = 0; i < NB; i++) {
        contrib[i] = (cur[i] - ref[i]) * Math.log(cur[i] / ref[i]);
        psi += contrib[i];
      }

      var maxP = 0, maxC = 0;
      for (i = 0; i < NB; i++) {
        maxP = Math.max(maxP, ref[i], cur[i]);
        maxC = Math.max(maxC, contrib[i]);
      }

      var bw = box.w / NB;
      txt(box.l, 26, 's-lbl', 'start',
          'feature distribution — outline = training reference, filled = this week');

      for (i = 0; i < NB; i++) {
        var x = box.l + i * bw + 3, w = bw - 6;
        var hr = ref[i] / maxP * box.h, hc = cur[i] / maxP * box.h;
        el('rect', { x: x, y: box.t + box.h - hr, width: w, height: hr,
                     fill: 'none', stroke: css('--tx-faint'), 'stroke-width': 1.4,
                     'stroke-dasharray': '3 3' }, svg);
        el('rect', { x: x, y: box.t + box.h - hc, width: w, height: hc,
                     fill: css('--accent'), opacity: 0.55 }, svg);
      }
      el('line', { x1: box.l, y1: box.t + box.h, x2: box.l + box.w, y2: box.t + box.h,
                   class: 's-axis' }, svg);
      txt(box.l, box.t + box.h + 18, 's-lbl-s', 'start', 'low');
      txt(box.l + box.w, box.t + box.h + 18, 's-lbl-s', 'end', 'high');

      /* per-bin contribution strip */
      var sy = box.t + box.h + 44, sh = 44;
      txt(box.l, sy - 8, 's-lbl-s', 'start', 'contribution to PSI, per bin');
      for (i = 0; i < NB; i++) {
        var cx = box.l + i * bw + 3;
        var ch = maxC > 0 ? contrib[i] / maxC * sh : 0;
        el('rect', { x: cx, y: sy + sh - ch, width: bw - 6, height: Math.max(1, ch),
                     fill: css(contrib[i] > 0.08 ? '--red' : '--amber'), opacity: 0.7 }, svg);
      }
      el('line', { x1: box.l, y1: sy + sh, x2: box.l + box.w, y2: sy + sh, class: 's-axis' }, svg);

      var verdict = psi < 0.1 ? 'stable — no action'
                  : psi < 0.25 ? 'moderate — investigate'
                               : 'significant — act';
      var col = psi < 0.1 ? '--green' : psi < 0.25 ? '--amber' : '--red';
      var v = txt(box.l, sy + sh + 32, 's-lbl', 'start',
                  'PSI = ' + psi.toFixed(3) + '   ·   ' + verdict);
      v.setAttribute('fill', css(col));
      txt(box.l, sy + sh + 54, 's-lbl-s', 'start',
          bug ? 'ONE bin carries almost all of it — that is the shape of a pipeline bug, not a ' +
                'population change'
              : 'the contribution is spread across bins — the shape of a genuine population shift');
    }

    var c = ctl(host);
    slider(c, 'mean shift', -0.3, 0.3, 0.01, 0, function (v) {
      shift = v; bug = false; if (bBug) bBug.textContent = 'inject a pipeline bug'; draw();
      return (v >= 0 ? '+' : '') + v.toFixed(2);
    });
    slider(c, 'spread ×', 0.6, 2.2, 0.05, 1, function (v) {
      spread = v; bug = false; if (bBug) bBug.textContent = 'inject a pipeline bug'; draw();
      return v.toFixed(2) + '×';
    });
    var bBug = button(c, 'inject a pipeline bug', function () {
      bug = !bug;
      bBug.textContent = bug ? 'bug: ON (default value in bin 1)' : 'inject a pipeline bug';
      draw();
    });

    draw();

    cap(host, 'PSI is a sum over bins, which is why the first thing you do with a PSI alert is open ' +
              'it up. Drag the mean: the contribution spreads across many bins — a real population ' +
              'shift. Now press the bug button: 13 % of the mass jumps into the lowest bin, as it ' +
              'does when an upstream field starts returning a default. Both can give you PSI above ' +
              '0.25, and they need opposite responses — retrain for the first, fix the pipeline and ' +
              'do NOT retrain for the second.');
  }

  /* =================================================================
     SHAPES — seeing a tensor.

     Every shape figure in the course goes through one renderer. A shape
     is a list of named axes, outermost first, and the rule that makes
     them readable is always the same: an axis is "how many of the thing
     to its right". [batch, sequence, d_model] is batch copies of a
     sequence-by-d_model grid. Say it that way and the shape errors stop.

     Axes are colour-coded by position (outermost amber, then blue,
     green, pink) and the shape string underneath uses the same colours,
     so "which axis is which" is answered by looking, not by counting.
     ================================================================= */

  var SH  = { cell: 17, gap: 2.5, dx: 9, dy: 9, gg: 34 };
  var CAPN = { g: 3, s: 4, r: 5, c: 7 };
  var AXC = ['amber', 'accent', 'green', 'pink'];

  /* "batch=8, sequence=128, d_model=4096" -> [{n,v}, ...] */
  function axes(raw) {
    var out = [];
    String(raw || '').split(',').forEach(function (p) {
      p = p.trim(); if (!p) return;
      var i = p.lastIndexOf('=');
      out.push(i < 0 ? { n: p, v: '' } : { n: p.slice(0, i).trim(), v: p.slice(i + 1).trim() });
    });
    return out;
  }

  /* how many cells to actually draw for an axis of stated size v */
  function shown(v, cap) {
    var k = parseInt(v, 10);
    if (!isFinite(k) || k < 1) return { k: cap, more: true };
    return { k: Math.min(k, cap), more: k > cap };
  }

  /* Draw one N-D block (0 to 4 axes) with its bounding box origin at
     (ox, oy). Returns the geometry plus every cell, so a caller can
     highlight slices, rows, columns or single cells afterwards. */
  function block(svg, dims, ox, oy) {
    var k = dims.length, step = SH.cell + SH.gap;
    var G = k >= 4 ? shown(dims[k - 4].v, CAPN.g) : { k: 1, more: false };
    var S = k >= 3 ? shown(dims[k - 3].v, CAPN.s) : { k: 1, more: false };
    var R = k >= 2 ? shown(dims[k - 2].v, CAPN.r) : { k: 1, more: false };
    var C = k >= 1 ? shown(dims[k - 1].v, CAPN.c) : { k: 1, more: false };

    var fw = C.k * step - SH.gap, fh = R.k * step - SH.gap;
    var bw = fw + (S.k - 1) * SH.dx, bh = fh + (S.k - 1) * SH.dy;
    var gw = bw + SH.gg;

    var cells = [], gi, si, ri, ci;
    for (gi = 0; gi < G.k; gi++) {
      var gx = ox + gi * gw;
      for (si = S.k - 1; si >= 0; si--) {          // back to front
        for (ri = 0; ri < R.k; ri++) {
          for (ci = 0; ci < C.k; ci++) {
            var rc = el('rect', {
              x: gx + si * SH.dx + ci * step,
              y: oy + (S.k - 1 - si) * SH.dy + ri * step,
              width: SH.cell, height: SH.cell, rx: 2.5,
              class: 'sh-cell' + (si === 0 ? ' sh-front' : '')
            }, svg);
            rc.setAttribute('data-g', gi); rc.setAttribute('data-s', si);
            rc.setAttribute('data-r', ri); rc.setAttribute('data-c', ci);
            cells.push(rc);
            rc.gi = gi; rc.si = si; rc.ri = ri; rc.ci = ci;
          }
        }
      }
    }
    return { x: ox, y: oy, w: G.k * bw + (G.k - 1) * SH.gg, h: bh,
             fw: fw, fh: fh, bw: bw, gw: gw, G: G, S: S, R: R, C: C,
             k: k, dims: dims, cells: cells,
             frontY: oy + (S.k - 1) * SH.dy };
  }

  /* an arrow with a label, used for every axis marker */
  function arrow(svg, x1, y1, x2, y2, colour) {
    var g = el('g', {}, svg);
    el('line', { x1: x1, y1: y1, x2: x2, y2: y2, stroke: css('--' + colour),
                 'stroke-width': 1.4, 'stroke-linecap': 'round' }, g);
    var a = Math.atan2(y2 - y1, x2 - x1), L = 5.5, sp = 0.42;
    el('path', { d: 'M' + x2 + ',' + y2 +
                    'L' + (x2 - L * Math.cos(a - sp)) + ',' + (y2 - L * Math.sin(a - sp)) +
                    'M' + x2 + ',' + y2 +
                    'L' + (x2 - L * Math.cos(a + sp)) + ',' + (y2 - L * Math.sin(a + sp)),
                 stroke: css('--' + colour), 'stroke-width': 1.4,
                 'stroke-linecap': 'round', fill: 'none' }, g);
    return g;
  }

  function txt(svg, x, y, s, colour, size, anchor, rot) {
    var t = el('text', { x: x, y: y, 'text-anchor': anchor || 'middle',
                         fill: colour ? css('--' + colour) : css('--tx-dim'),
                         'font-size': size || 11.5 }, svg);
    if (rot) t.setAttribute('transform', 'rotate(' + rot + ' ' + x + ' ' + y + ')');
    t.textContent = s;
    return t;
  }

  /* label every axis of a drawn block, colour-coded by position */
  function blockAxes(svg, B) {
    var k = B.k, d = B.dims;
    function lab(i) { return d[i].n + (d[i].v ? ' = ' + d[i].v : ''); }
    function col(i) { return AXC[Math.min(i, AXC.length - 1)]; }

    if (k >= 1) {                                   // columns: along the bottom
      var i = k - 1, y = B.frontY + B.fh + 11;
      arrow(svg, B.x, y, B.x + B.fw, y, col(i));
      txt(svg, B.x + B.fw / 2, y + 15, lab(i), col(i));
      if (B.C.more) txt(svg, B.x + B.fw + 9, B.frontY + B.fh / 2 + 4, '…', col(i), 14, 'start');
    }
    if (k >= 2) {                                   // rows: down the left edge
      var j = k - 2, x = B.x - 12;
      arrow(svg, x, B.frontY, x, B.frontY + B.fh, col(j));
      txt(svg, x - 8, B.frontY + B.fh / 2, lab(j), col(j), 11.5, 'middle', -90);
      if (B.R.more) txt(svg, B.x + B.fw / 2, B.frontY + B.fh + 3, '⋮', col(j), 13);
    }
    if (k >= 3) {                                   // slices: into the depth
      var m = k - 3, sx = B.x + B.fw + 7;
      arrow(svg, sx, B.frontY - 3, sx + (B.S.k - 1) * SH.dx, B.y - 3, col(m));
      txt(svg, sx + (B.S.k - 1) * SH.dx + 6, B.y - 7, lab(m) + (B.S.more ? ' …' : ''),
          col(m), 11.5, 'start');
    }
    if (k >= 4) {                                   // groups: the outer repeat
      var q = k - 4, gy = B.frontY + B.fh + 34;
      arrow(svg, B.x, gy, B.x + B.w, gy, col(q));
      txt(svg, B.x + B.w / 2, gy + 15, lab(q) + (B.G.more ? '  (…)' : ''), col(q));
    }
  }

  /* the shape string, coloured to match the arrows */
  function shapeStr(host, dims, note) {
    var d = h('div', 'shapestr', null, host);
    h('span', 'br', '[', d);
    dims.forEach(function (a, i) {
      if (i) h('span', 'sep', ', ', d);
      var s = h('span', 'ax ax-' + AXC[Math.min(i, AXC.length - 1)], a.n, d);
      if (a.v) h('span', 'axv', ' ' + a.v, s);
    });
    h('span', 'br', ']', d);
    if (note) h('span', 'note', '  ' + note, d);
    return d;
  }

  function hi(cells, fn, cls) {
    cells.forEach(function (c) {
      c.setAttribute('class', 'sh-cell' + (c.si === 0 ? ' sh-front' : '') +
                              (fn(c) ? ' ' + (cls || 'sh-hi') : ''));
    });
  }

  /* -----------------------------------------------------------------
     shape — the generic figure. Drop it anywhere:
       <div class="viz" data-viz="shape"
            data-dims="batch=32, channels=3, height=224, width=224"
            data-cap="one training batch of images"></div>
     ----------------------------------------------------------------- */
  function vizShape(host) {
    var dims = axes(host.dataset.dims);
    var probe = makeSVG(host, 10, 10);                      // measure first
    var B = block(probe, dims, 0, 0);
    probe.parentNode.removeChild(probe);

    var padL = 42, padR = 92, padT = 30, padB = dims.length >= 4 ? 58 : 34;
    var W = Math.max(300, B.w + padL + padR), H = B.h + padT + padB;
    var svg = makeSVG(host, W, H);
    B = block(svg, dims, padL, padT);
    blockAxes(svg, B);
    shapeStr(host, dims, host.dataset.note || '');
    if (host.dataset.cap) cap(host, host.dataset.cap);
  }

  /* -----------------------------------------------------------------
     tensor-ladder — the same idea, one axis at a time.
     ----------------------------------------------------------------- */
  var LADDER = [
    { name: 'scalar', dims: '',
      read: 'One number. No axes at all.',
      one:  'A loss value: 0.42.' },
    { name: 'vector', dims: 'd_model=6',
      read: 'A list of 6 numbers. One axis, so one index picks an element.',
      one:  'One token’s embedding — 6 numbers describing the word "bank".' },
    { name: 'matrix', dims: 'sequence=5, d_model=6',
      read: '5 rows, each a 6-number vector. Read it outwards: 5 copies of a 6-vector.',
      one:  'One sentence: 5 tokens, each with its own 6-number embedding.' },
    { name: '3-D', dims: 'batch=3, sequence=5, d_model=6',
      read: '3 copies of a 5×6 grid. The new axis is just "how many sentences".',
      one:  'A training batch: 3 sentences going through the model at once.' },
    { name: '4-D', dims: 'batch=3, heads=4, sequence=5, d_head=6',
      read: '3 sentences × 4 attention heads, each holding a 5×6 grid.',
      one:  'Exactly the shape inside multi-head attention. Nothing new — one more "how many".' }
  ];

  function vizTensorLadder(host) {
    var stage = h('div', 'sh-stage', null, host);
    var read  = h('div', 'shapenote', null, host);
    var bar   = ctl(host);
    var at = 0, btns = [];

    /* measure every rung first, then give them all one canvas, so the
       controls do not jump about as you step up the ladder. */
    var padL = 46, padR = 112, padT = 30, padB = 34;
    var probe = makeSVG(stage, 10, 10), geo = LADDER.map(function (r) {
      var d = axes(r.dims), B = block(probe, d, 0, 0);
      return { dims: d, w: B.w, h: B.h, deep: d.length >= 4 };
    });
    probe.parentNode.removeChild(probe);
    var CW = Math.max(430, Math.max.apply(null, geo.map(function (g) { return g.w; })) + padL + padR);
    var CH = Math.max.apply(null, geo.map(function (g) { return g.h + (g.deep ? 26 : 0); }))
             + padT + padB;

    function draw() {
      stage.innerHTML = '';
      var rung = LADDER[at], g = geo[at], dims = g.dims;
      var svg = makeSVG(stage, CW, CH);
      var oy = padT + (CH - padT - padB - g.h - (g.deep ? 26 : 0)) / 2;
      var B = block(svg, dims, padL, oy);
      blockAxes(svg, B);
      if (!dims.length) {                       // the scalar: put it in the box
        var c = B.cells[0];
        c.setAttribute('width', 46); c.setAttribute('height', 30);
        c.setAttribute('class', 'sh-cell sh-front sh-hi');
        txt(svg, padL + 23, oy + 20, '0.42', 'amber', 13);
      }
      shapeStr(stage, dims, dims.length ? '' : '— no axes at all');
      read.innerHTML = '';
      h('p', 'rd-a', rung.read, read);
      h('p', 'rd-b', rung.one, read);
      btns.forEach(function (b, i) { b.className = i === at ? 'on' : ''; });
    }

    LADDER.forEach(function (r, i) {
      btns.push(button(bar, r.name, function () { at = i; draw(); }));
    });
    draw();
    cap(host, 'Each step adds one axis, and every axis means the same thing: ' +
              'how many of the thing to its right. A tensor is not a new object — ' +
              'it is a matrix with more "how many" in front of it.');
  }

  /* -----------------------------------------------------------------
     tensor-axes — what an axis actually is, by slicing one.
     ----------------------------------------------------------------- */
  function vizTensorAxes(host) {
    var dims = axes('batch=4, sequence=6, d_model=8');
    var stage = h('div', 'sh-stage', null, host);
    var probe = makeSVG(stage, 10, 10);
    var B0 = block(probe, dims, 0, 0);
    probe.parentNode.removeChild(probe);

    var padL = 46, padR = 120, padT = 34, padB = 40;
    var svg = makeSVG(stage, Math.max(460, B0.w + padL + padR), B0.h + padT + padB);
    var B = block(svg, dims, padL, padT);
    blockAxes(svg, B);

    shapeStr(host, dims);
    var read = h('div', 'shapenote', null, host);
    var bar = ctl(host), btns = [];

    var VIEWS = [
      { l: 'x', f: function () { return false; }, s: '[4, 6, 8]',
        a: 'The whole thing: 4 sentences, each 6 tokens long, each token described by 8 numbers.',
        b: '192 numbers in total — 4 × 6 × 8.' },
      { l: 'x[0]', f: function (c) { return c.si === 0; }, s: '[6, 8]',
        a: 'Fix the first axis. You get one sentence out of the batch.',
        b: 'This is what people mean by "one example". The batch axis is gone.' },
      { l: 'x[0, 3]', f: function (c) { return c.si === 0 && c.ri === 3; }, s: '[8]',
        a: 'Fix the first two axes. One token, in one sentence.',
        b: 'An embedding: the 8 numbers that are this word, here, right now.' },
      { l: 'x[:, :, 5]', f: function (c) { return c.ci === 5; }, s: '[4, 6]',
        a: 'Fix the last axis instead. One feature, for every token of every sentence.',
        b: 'Slicing along d_model is what a probe does when it asks "which neuron fires for X?".' },
      { l: 'x[0, 3, 5]', f: function (c) { return c.si === 0 && c.ri === 3 && c.ci === 5; },
        s: 'scalar',
        a: 'Fix all three. One number.',
        b: 'Feature 5 of token 3 of sentence 0. The bottom of the ladder.' }
    ];

    function pick(i) {
      var v = VIEWS[i];
      hi(B.cells, v.f);
      read.innerHTML = '';
      h('p', 'rd-s', v.l + '   →   ' + v.s, read);
      h('p', 'rd-a', v.a, read);
      h('p', 'rd-b', v.b, read);
      btns.forEach(function (b, j) { b.className = j === i ? 'on' : ''; });
    }
    VIEWS.forEach(function (v, i) { btns.push(button(bar, v.l, function () { pick(i); })); });
    pick(0);
    cap(host, 'An index picks a position along one axis and that axis disappears from the shape. ' +
              'Every shape error you will ever debug is this picture disagreeing with itself — ' +
              'two tensors that both say "[4, 6, 8]" but mean different things by the 6.');
  }

  /* -----------------------------------------------------------------
     matmul — why the inner dimensions have to match, and where they go.
     ----------------------------------------------------------------- */
  function vizMatmul(host) {
    var m = 4, n = 3, p = 5;
    var W = 700, H = 250, svg = makeSVG(host, W, H);
    var A = block(svg, axes('m=' + m + ', n=' + n), 52, 60);
    var opx = A.x + A.w + 22;
    txt(svg, opx, A.y + A.fh / 2 + 5, '×', 'tx-dim', 17);
    var Bm = block(svg, axes('n=' + n + ', p=' + p), opx + 22, 60);
    var eqx = Bm.x + Bm.w + 22;
    txt(svg, eqx, A.y + A.fh / 2 + 5, '=', 'tx-dim', 17);
    var C = block(svg, axes('m=' + m + ', p=' + p), eqx + 22, 60);

    txt(svg, A.x + A.fw / 2,  46, 'A   (4×3)', 'tx-faint', 11.5);
    txt(svg, Bm.x + Bm.fw / 2, 46, 'B   (3×5)', 'tx-faint', 11.5);
    txt(svg, C.x + C.fw / 2,  46, 'C   (4×5)', 'amber', 11.5);

    var read = h('div', 'shapenote', null, host);
    var i = 0, j = 0, timer = null;

    function step() {
      hi(A.cells,  function (c) { return c.ri === i; });
      hi(Bm.cells, function (c) { return c.ci === j; }, 'sh-hi2');
      hi(C.cells,  function (c) { return c.ri === i && c.ci === j; });
      read.innerHTML = '';
      h('p', 'rd-s', 'C[' + i + ',' + j + ']  =  row ' + i + ' of A  ·  column ' +
                     j + ' of B', read);
      h('p', 'rd-a', 'Both have ' + n + ' numbers in them — that is the only reason ' +
                     'this dot product exists. The 3 is consumed and never appears in C.', read);
      j++; if (j >= p) { j = 0; i = (i + 1) % m; }
    }
    var bar = ctl(host);
    button(bar, 'step', step);
    button(bar, 'play / pause', function () {
      if (timer) { clearInterval(timer); timer = null; }
      else timer = setInterval(step, 620);
    });
    step();
    cap(host, '(m×n)(n×p) → (m×p). The two inner numbers must match and then ' +
              'they vanish; the outer two survive. Read a shape error as "the inner numbers ' +
              'disagree" and you will fix it in seconds instead of minutes.');
  }

  /* =================================================================
     NUMBERS — the analogy, then the same analogy with arithmetic in it.

     A formula is only frightening while it is abstract. Every figure
     below runs the formula on the numbers from the story that
     introduced it, one term at a time, so the symbols arrive already
     attached to something.
     ================================================================= */

  var FILMS = ['Heat', 'Amélie', 'Alien', 'Notting Hill', 'Blade Runner'];
  var RATER = {
    ana:  { l: 'Ana',  v: [5, 1, 4, 2, 5] },
    ben:  { l: 'Ben',  v: [4, 2, 5, 1, 4] },
    cara: { l: 'Cara', v: [1, 5, 2, 5, 1] }
  };

  function norm2(v) {
    var s = 0; v.forEach(function (x) { s += x * x; }); return Math.sqrt(s);
  }

  /* -----------------------------------------------------------------
     dot-ratings — two people rate five films out of five. Multiply
     pairwise, add up, and watch a single number appear that means
     "same taste".
     ----------------------------------------------------------------- */
  function vizDotRatings(host) {
    var W = 660, H = 268, svg = makeSVG(host, W, H);
    var other = 'ben', at = -1, timer = null;
    var colX = [], x0 = 176, colW = 88;
    for (var i = 0; i < 5; i++) colX.push(x0 + i * colW);

    var rowY = { film: 44, a: 84, b: 122, prod: 172, sum: 224 };
    txt(svg, 150, rowY.film, 'film', 'tx-faint', 11.5, 'end');
    var labA = txt(svg, 150, rowY.a, 'Ana', 'accent', 12.5, 'end');
    var labB = txt(svg, 150, rowY.b, 'Ben', 'pink', 12.5, 'end');
    txt(svg, 150, rowY.prod, 'aᵢ × bᵢ', 'amber', 12.5, 'end');
    txt(svg, 150, rowY.sum, 'running total', 'tx-dim', 12.5, 'end');

    el('line', { x1: 60, y1: rowY.prod - 26, x2: W - 40, y2: rowY.prod - 26,
                 class: 's-grid' }, svg);
    el('line', { x1: 60, y1: rowY.sum - 26, x2: W - 40, y2: rowY.sum - 26,
                 class: 's-grid' }, svg);

    var cellA = [], cellB = [], cellP = [], boxA = [], boxB = [], boxP = [];
    FILMS.forEach(function (f, i) {
      txt(svg, colX[i], rowY.film, f, 'tx-faint', 10.5);
      [['a', boxA, cellA, 'accent'], ['b', boxB, cellB, 'pink'],
       ['prod', boxP, cellP, 'amber']].forEach(function (spec) {
        var y = rowY[spec[0]];
        spec[1].push(el('rect', { x: colX[i] - 17, y: y - 15, width: 34, height: 24,
                                  rx: 4, class: 'sh-cell sh-front' }, svg));
        spec[2].push(txt(svg, colX[i], y + 3, '', spec[3], 13));
      });
    });
    var total = txt(svg, colX[4] + 62, rowY.sum + 4, '0', 'amber', 20, 'middle');
    var totBox = el('rect', { x: colX[4] + 34, y: rowY.sum - 17, width: 56, height: 28,
                              rx: 6, class: 'sh-cell sh-front' }, svg);
    totBox.setAttribute('stroke', css('--gold-dim'));
    svg.appendChild(total);

    var read = h('div', 'shapenote', null, host);

    function paint() {
      var A = RATER.ana.v, B = RATER[other].v, run = 0;
      labB.textContent = RATER[other].l;
      FILMS.forEach(function (f, i) {
        var live = i <= at;
        cellA[i].textContent = A[i];
        cellB[i].textContent = B[i];
        cellP[i].textContent = live ? A[i] * B[i] : '';
        [boxA[i], boxB[i], boxP[i]].forEach(function (b) {
          b.setAttribute('class', 'sh-cell sh-front' + (i === at ? ' sh-hi' : ''));
        });
        if (live) run += A[i] * B[i];
      });
      total.textContent = run;

      var full = at >= 4, dot = 0;
      A.forEach(function (x, i) { dot += x * B[i]; });
      read.innerHTML = '';
      if (!full) {
        h('p', 'rd-s', 'a · b  =  Σᵢ aᵢ bᵢ', read);
        h('p', 'rd-a', at < 0 ? 'Press step. Each film contributes one product to the total.'
                              : 'Film ' + (at + 1) + ': ' + A[at] + ' × ' + B[at] +
                                ' = ' + (A[at] * B[at]) + '. Both loved it, or both ignored it — ' +
                                'either way it pushes the total up only when both numbers are big.',
          read);
      } else {
        var na = norm2(A), nb = norm2(B), cs = dot / (na * nb);
        h('p', 'rd-s', 'a · b  =  ' + A.map(function (x, i) { return x + '×' + B[i]; }).join('  +  ') +
                       '  =  ' + dot, read);
        h('p', 'rd-a', 'Ana and ' + RATER[other].l + ' score ' + dot + '. On its own that number ' +
                       'means nothing — it grows if either person just rates everything highly. ' +
                       'Divide out both lengths and it becomes an angle instead.', read);
        h('p', 'rd-b', 'cos θ = ' + dot + ' / (' + na.toFixed(2) + ' × ' + nb.toFixed(2) + ')' +
                       ' = ' + cs.toFixed(3) + '  →  ' +
                       (cs > 0.9 ? 'nearly the same taste.'
                                 : 'related, but clearly different taste.'), read);
      }
    }

    var bar = ctl(host);
    button(bar, 'step', function () { at = Math.min(4, at + 1); paint(); });
    button(bar, 'play', function () {
      if (timer) { clearInterval(timer); timer = null; return; }
      at = -1; paint();
      timer = setInterval(function () {
        at++; paint();
        if (at >= 4) { clearInterval(timer); timer = null; }
      }, 700);
    });
    button(bar, 'reset', function () {
      if (timer) { clearInterval(timer); timer = null; }
      at = -1; paint();
    });
    var swap = button(bar, 'compare with Cara', function () {
      other = other === 'ben' ? 'cara' : 'ben';
      swap.textContent = other === 'ben' ? 'compare with Cara' : 'back to Ben';
      at = 4; paint();
    });
    paint();
    cap(host, 'Ana and Ben score 64; Ana and Cara score 33. The dot product is one number ' +
              'that says how much two lists agree — and that is exactly what an attention ' +
              'score is, with query and key vectors in place of film ratings.');
  }

  /* -----------------------------------------------------------------
     norms — one vector, three honest answers to "how long is it?".
     ----------------------------------------------------------------- */
  function vizNorms(host) {
    var vx = 3, vy = 4;
    var W = 560, H = 300, box = { l: 60, t: 24, w: 232, h: 232 };
    var svg = makeSVG(host, W, H);
    var F = new Frame(svg, box, [0, 5], [0, 5]);
    var i;
    for (i = 0; i <= 5; i++) {
      el('line', { x1: F.X(i), y1: F.Y(0), x2: F.X(i), y2: F.Y(5), class: 's-grid' }, svg);
      el('line', { x1: F.X(0), y1: F.Y(i), x2: F.X(5), y2: F.Y(i), class: 's-grid' }, svg);
      txt(svg, F.X(i), F.Y(0) + 15, i, 'tx-faint', 10);
      if (i) txt(svg, F.X(0) - 9, F.Y(i) + 4, i, 'tx-faint', 10, 'end');
    }
    F.axes();
    txt(svg, F.X(5) + 16, F.Y(0) + 4, 'x', 'tx-faint', 11, 'start');
    txt(svg, F.X(0) - 9, F.Y(5) - 8, 'y', 'tx-faint', 11, 'end');

    /* the three answers, side by side, so "three lengths" is one glance */
    var LX = box.l + box.w + 62, cards = [];
    [['L2  ‖v‖₂', '5', 'accent', 'crow’s flight'],
     ['L1  ‖v‖₁', '7', 'green',  'taxi on the grid'],
     ['L∞ ‖v‖∞', '4', 'pink',   'worst single street']].forEach(function (c, k) {
      var y = box.t + 14 + k * 74;
      var r = el('rect', { x: LX, y: y, width: 172, height: 60, rx: 8,
                           class: 'sh-cell sh-front' }, svg);
      txt(svg, LX + 14, y + 24, c[0], c[2], 13, 'start');
      txt(svg, LX + 158, y + 26, c[1], c[2], 21, 'end');
      txt(svg, LX + 14, y + 44, c[3], 'tx-faint', 11, 'start');
      cards.push(r);
    });
    var overlay = el('g', {}, svg);
    el('line', { x1: F.X(0), y1: F.Y(0), x2: F.X(vx), y2: F.Y(vy),
                 stroke: css('--tx-faint'), 'stroke-width': 1.5,
                 'stroke-dasharray': '3 3' }, svg);
    el('circle', { cx: F.X(vx), cy: F.Y(vy), r: 4.5, fill: css('--amber') }, svg);
    txt(svg, F.X(vx) + 10, F.Y(vy) - 6, 'v = (3, 4)', 'amber', 12, 'start');

    var read = h('div', 'shapenote', null, host);
    var MODES = [
      { l: 'L2  ‖v‖₂', c: 'accent',
        draw: function (g) {
          el('line', { x1: F.X(0), y1: F.Y(0), x2: F.X(vx), y2: F.Y(vy),
                       stroke: css('--accent'), 'stroke-width': 3,
                       'stroke-linecap': 'round' }, g);
        },
        f: '‖v‖₂ = √(v₁² + v₂²)',
        w: '= √(3² + 4²) = √(9 + 16) = √25 = 5',
        a: 'The crow’s flight. Straight from the origin to the point, ignoring the streets.',
        b: 'L2 norm clipping caps this length. Below, [3, 4] is shrunk to [1.2, 1.6], ' +
           'whose length is 2. An L2 penalty instead charges for the squared length: 25.' },
      { l: 'L1  ‖v‖₁', c: 'green',
        draw: function (g) {
          el('path', { d: 'M' + F.X(0) + ',' + F.Y(0) + 'H' + F.X(vx) + 'V' + F.Y(vy),
                       stroke: css('--green'), 'stroke-width': 3, fill: 'none',
                       'stroke-linejoin': 'round', 'stroke-linecap': 'round' }, g);
        },
        f: '‖v‖₁ = |v₁| + |v₂|',
        w: '= |3| + |4| = 7',
        a: 'The taxi. You cannot drive through the buildings, so you go 3 blocks east ' +
           'and then 4 blocks north: 7 blocks of road.',
        b: 'Longer than the crow’s 5 — and that gap is the whole reason L1 and L2 ' +
           'behave differently as penalties.' },
      { l: 'L∞  ‖v‖∞', c: 'pink',
        draw: function (g) {
          el('line', { x1: F.X(vx), y1: F.Y(0), x2: F.X(vx), y2: F.Y(vy),
                       stroke: css('--pink'), 'stroke-width': 3,
                       'stroke-linecap': 'round' }, g);
        },
        f: '‖v‖∞ = max(|v₁|, |v₂|)',
        w: '= max(3, 4) = 4',
        a: 'The worst street. Only the single longest leg counts; the other one is ignored.',
        b: 'Used when the thing you fear is one component blowing up, not the total — ' +
           'adversarial robustness is stated in L∞ for exactly that reason.' }
    ];
    var bar = ctl(host), btns = [];
    function pick(k) {
      var m = MODES[k];
      overlay.innerHTML = '';
      m.draw(overlay);
      cards.forEach(function (r, j) {
        r.setAttribute('class', 'sh-cell sh-front' + (j === k ? ' sh-hi' : ''));
      });
      read.innerHTML = '';
      var f = h('div', 'mth', null, read);
      h('div', 'f', m.f, f);
      h('div', 'w', m.w, f);
      h('p', 'rd-a', m.a, read);
      h('p', 'rd-b', m.b, read);
      btns.forEach(function (b, j) { b.className = j === k ? 'on' : ''; });
    }
    MODES.forEach(function (m, k) { btns.push(button(bar, m.l, function () { pick(k); })); });
    pick(0);
    cap(host, 'One vector, three lengths: 5, 7 and 4. Nothing is being approximated — ' +
              'they are three different honest questions. Which one a method uses tells ' +
              'you what that method is afraid of.');
  }

  /* -----------------------------------------------------------------
     transpose — watch the grid actually flip, then see why attention
     needs it. A[i,j] becomes Aᵀ[j,i]: the value never changes, only
     where it lives.
     ----------------------------------------------------------------- */
  function vizTranspose(host) {
    var R = 3, C = 4, cell = 40, gap = 5, step = cell + gap;
    var W = 620, H = 250, svg = makeSVG(host, W, H);
    var ox = 60, oy = 52, flipped = false, anim = null;

    var lbl = txt(svg, ox + (C * step - gap) / 2, 34, 'A   (3×4)', 'amber', 12.5);
    var rowLab = txt(svg, ox - 16, oy + (R * step - gap) / 2, 'rows = 3', 'accent', 11, 'middle', -90);
    var colLab = txt(svg, ox + (C * step - gap) / 2, oy + R * step + 16, 'columns = 4', 'green', 11);

    var cells = [];
    for (var r = 0; r < R; r++) for (var c = 0; c < C; c++) {
      var g = el('g', {}, svg);
      var rc = el('rect', { x: 0, y: 0, width: cell, height: cell, rx: 4,
                            class: 'sh-cell sh-front' }, g);
      var t = el('text', { x: cell / 2, y: cell / 2 + 5, 'text-anchor': 'middle',
                           fill: css('--chalk'), 'font-size': 12.5 }, g);
      t.textContent = 'a' + (r + 1) + (c + 1);
      cells.push({ g: g, rect: rc, r: r, c: c, x: 0, y: 0 });
    }
    function place(t) {                       // t = 0 original, 1 transposed
      cells.forEach(function (k) {
        var x0 = ox + k.c * step,       y0 = oy + k.r * step;
        var x1 = ox + k.r * step,       y1 = oy + k.c * step;
        k.x = x0 + (x1 - x0) * t; k.y = y0 + (y1 - y0) * t;
        k.g.setAttribute('transform', 'translate(' + k.x.toFixed(1) + ',' + k.y.toFixed(1) + ')');
      });
    }
    place(0);

    var read = h('div', 'shapenote', null, host);
    function say() {
      read.innerHTML = '';
      var f = h('div', 'mth', null, read);
      h('div', 'f', 'Aᵀ[j, i] = A[i, j]', f);
      h('div', 'w', flipped ? 'a₂₃ sat in row 2, column 3. In Aᵀ it sits in row 3, column 2 — '
                            + 'same number, mirrored across the diagonal.'
                            : 'a₂₃ sits in row 2, column 3.', f);
      h('p', 'rd-a', flipped
        ? 'Nothing was computed. The grid was re-labelled: what was a row is now a column.'
        : 'Press transpose. Watch a cell travel — the value goes with it.', read);
    }

    function run(to) {
      if (anim) cancelAnimationFrame(anim);
      var from = flipped ? 1 : 0, t0 = performance.now();
      (function frame(now) {
        var p = Math.min(1, (now - t0) / 520);
        var e = p < 0.5 ? 2 * p * p : 1 - Math.pow(-2 * p + 2, 2) / 2;   // ease in-out
        place(from + (to - from) * e);
        if (p < 1) anim = requestAnimationFrame(frame);
        else {
          flipped = to === 1;
          lbl.textContent = flipped ? 'Aᵀ   (4×3)' : 'A   (3×4)';
          rowLab.textContent = 'rows = ' + (flipped ? 4 : 3);
          colLab.textContent = 'columns = ' + (flipped ? 3 : 4);
          rowLab.setAttribute('x', ox - 16);
          rowLab.setAttribute('y', oy + ((flipped ? C : R) * step - gap) / 2);
          rowLab.setAttribute('transform', 'rotate(-90 ' + (ox - 16) + ' ' +
                              (oy + ((flipped ? C : R) * step - gap) / 2) + ')');
          colLab.setAttribute('x', ox + ((flipped ? R : C) * step - gap) / 2);
          colLab.setAttribute('y', oy + (flipped ? C : R) * step + 16);
          lbl.setAttribute('x', ox + ((flipped ? R : C) * step - gap) / 2);
          say();
        }
      })(performance.now());
    }

    var bar = ctl(host);
    button(bar, 'transpose', function () { run(flipped ? 0 : 1); });
    button(bar, 'highlight a₂₃', function () {
      cells.forEach(function (k) {
        k.rect.setAttribute('class', 'sh-cell sh-front' +
          (k.r === 1 && k.c === 2 ? ' sh-hi' : ''));
      });
    });
    say();
    cap(host, 'In attention you meet it as Q Kᵀ. Q is (tokens × d) and K is (tokens × d) — ' +
              'those will not multiply, because the inner numbers are tokens and d. ' +
              'Transposing K makes it (d × tokens), the inner d’s match and cancel, and the ' +
              'result is (tokens × tokens): every query scored against every key. ' +
              'The transpose is not a computation, it is the thing that makes the shapes agree.');
  }

  /* =================================================================
     CALCULUS — slopes you can watch, and a chain rule with numbers in it.
     ================================================================= */

  /* shared: a scalar field drawn as a heat map, used by the two
     figures that need two input axes at once */
  function field(svg, box, f, xr, yr, n) {
    n = n || 34;
    var F = new Frame(svg, box, xr, yr), lo = 1e9, hi = -1e9, v = [], i, j;
    for (i = 0; i < n; i++) { v.push([]); for (j = 0; j < n; j++) {
      var x = xr[0] + (xr[1] - xr[0]) * (i + 0.5) / n;
      var y = yr[0] + (yr[1] - yr[0]) * (j + 0.5) / n;
      var z = f(x, y); v[i].push(z);
      if (z < lo) lo = z; if (z > hi) hi = z;
    } }
    var cw = box.w / n, ch = box.h / n;
    for (i = 0; i < n; i++) for (j = 0; j < n; j++) {
      var t = (v[i][j] - lo) / (hi - lo || 1);
      /* Snap every tile to whole pixels and butt them edge to edge. Two
         semi-transparent rects that overlap by a fraction of a pixel
         composite twice and draw a grid that is not in the data. */
      var xa = Math.round(box.l + i * cw), xb = Math.round(box.l + (i + 1) * cw);
      var yb = Math.round(box.t + box.h - j * ch), ya = Math.round(box.t + box.h - (j + 1) * ch);
      el('rect', { x: xa, y: ya, width: xb - xa, height: yb - ya,
                   'shape-rendering': 'crispEdges',
                   fill: css('--accent'), opacity: (0.06 + 0.62 * t).toFixed(3) }, svg);
    }
    return F;
  }

  /* -----------------------------------------------------------------
     slope — the secant collapsing into the tangent. The derivative is
     not a new idea, it is this fraction after you stop shrinking.
     ----------------------------------------------------------------- */
  function vizSlope(host) {
    var W = 660, H = 330, box = { l: 56, t: 20, w: 380, h: 236 };
    var svg = makeSVG(host, W, H);
    function f(x) { return x * x; }
    var F = new Frame(svg, box, [0, 4], [0, 16]);
    F.axes('x', 'f(x) = x²');
    F.path(f, 260);

    var x0 = 2, sec = el('line', { stroke: css('--pink'), 'stroke-width': 2.2,
                                   'stroke-linecap': 'round' }, svg);
    var run = el('line', { stroke: css('--tx-faint'), 'stroke-width': 1.2,
                           'stroke-dasharray': '3 3' }, svg);
    var rise = el('line', { stroke: css('--tx-faint'), 'stroke-width': 1.2,
                            'stroke-dasharray': '3 3' }, svg);
    var pA = el('circle', { r: 5, fill: css('--amber') }, svg);
    var pB = el('circle', { r: 4.5, fill: css('--pink') }, svg);
    var lblRun = txt(svg, 0, 0, '', 'tx-dim', 11);
    var lblRise = txt(svg, 0, 0, '', 'tx-dim', 11, 'start');
    var read = h('div', 'shapenote', null, host);

    function draw(d) {
      var xa = x0, xb = x0 + d, ya = f(xa), yb = f(xb);
      var m = (yb - ya) / (xb - xa);
      var ext = 1.5;
      sec.setAttribute('x1', F.X(xa - ext)); sec.setAttribute('y1', F.Y(ya - m * ext));
      sec.setAttribute('x2', F.X(xb + ext)); sec.setAttribute('y2', F.Y(yb + m * ext));
      pA.setAttribute('cx', F.X(xa)); pA.setAttribute('cy', F.Y(ya));
      pB.setAttribute('cx', F.X(xb)); pB.setAttribute('cy', F.Y(yb));
      run.setAttribute('x1', F.X(xa)); run.setAttribute('y1', F.Y(ya));
      run.setAttribute('x2', F.X(xb)); run.setAttribute('y2', F.Y(ya));
      rise.setAttribute('x1', F.X(xb)); rise.setAttribute('y1', F.Y(ya));
      rise.setAttribute('x2', F.X(xb)); rise.setAttribute('y2', F.Y(yb));
      lblRun.setAttribute('x', F.X(xa + d / 2)); lblRun.setAttribute('y', F.Y(ya) + 14);
      lblRun.textContent = 'run ' + d.toFixed(2);
      lblRise.setAttribute('x', F.X(xb) + 7); lblRise.setAttribute('y', F.Y((ya + yb) / 2));
      lblRise.textContent = 'rise ' + (yb - ya).toFixed(2);

      read.innerHTML = '';
      var mm = h('div', 'mth', null, read);
      h('div', 'f', 'slope  =  rise / run  =  ( f(2+h) − f(2) ) / h', mm);
      h('div', 'w', '= (' + yb.toFixed(2) + ' − 4.00) / ' + d.toFixed(2) +
                    '  =  ' + m.toFixed(3), mm);
      h('p', 'rd-a', 'Rise over run, and nothing else: go across by h, see how far up the ' +
                     'curve went, divide. The straight line through the two points is the ' +
                     'slope you just measured.', read);
      h('p', 'rd-b', d > 0.2
        ? 'Slide h down and watch the number walk steadily towards 4.'
        : 'h = ' + d.toFixed(2) + ' and the slope reads ' + m.toFixed(3) + ', heading for 4. ' +
          'Keep shrinking h and you stop measuring a slope between two points and start ' +
          'measuring the slope at one — which is the derivative. Section 11.3 does that ' +
          'shrinking properly.',
        read);
    }
    var bar = ctl(host);
    var sl = slider(bar, 'h', 0.02, 1.6, 0.02, 1.6, function (v) { draw(v); return v.toFixed(2); });
    button(bar, 'shrink h', function () {
      var v = parseFloat(sl.value);
      var t = setInterval(function () {
        v = Math.max(0.02, v - 0.06); sl.value = v; draw(v);
        if (v <= 0.02) clearInterval(t);
      }, 40);
    });
    cap(host, 'A slope is a fraction you can measure with a ruler: how far up, divided by ' +
              'how far across. Everything later in this module is that fraction with the ' +
              '"across" shrunk to nothing.');
  }

  /* -----------------------------------------------------------------
     partial — two input axes, one slope at a time. This is where
     "gradient" stops being a word and becomes an arrow.
     ----------------------------------------------------------------- */
  function vizPartial(host) {
    var W = 660, H = 330, box = { l: 52, t: 18, w: 234, h: 234 };
    var svg = makeSVG(host, W, H);
    function f(x, y) { return x * x + 2 * y * y; }
    var F = field(svg, box, f, [-3, 3], [-3, 3]);
    F.axes('x', 'y');
    var px = 2, py = 1;
    var overlay = el('g', {}, svg);
    el('circle', { cx: F.X(px), cy: F.Y(py), r: 5, fill: css('--amber') }, svg);
    txt(svg, F.X(px) - 10, F.Y(py) - 9, '(2, 1)', 'amber', 11, 'end');

    var cutBox = { l: 386, t: 40, w: 214, h: 190 };
    var cut = el('g', {}, svg);
    var read = h('div', 'shapenote', null, host);

    var VIEWS = [
      { l: '∂f/∂x', c: 'accent',
        mark: function (g) {
          el('line', { x1: F.X(-3), y1: F.Y(py), x2: F.X(3), y2: F.Y(py),
                       stroke: css('--accent'), 'stroke-width': 2.2 }, g);
          arrow(svg, F.X(px), F.Y(py), F.X(px + 0.9), F.Y(py), 'accent');
        },
        slice: function () { return { g: function (t) { return f(px + t, py); },
                                      m: 4, yr: [0, 18], xl: 'nudge x', c: 'accent' }; },
        f: '∂f/∂x  =  2x', w: 'at (2, 1):  2 × 2  =  4',
        a: 'Freeze y at 1 and walk along x only. The surface collapses to an ordinary ' +
           'one-input curve — the right-hand panel is that curve — and you are back to ' +
           'rise over run.',
        b: '“Partial” means nothing more than: every other input held still.' },
      { l: '∂f/∂y', c: 'green',
        mark: function (g) {
          el('line', { x1: F.X(px), y1: F.Y(-3), x2: F.X(px), y2: F.Y(3),
                       stroke: css('--green'), 'stroke-width': 2.2 }, g);
          arrow(svg, F.X(px), F.Y(py), F.X(px), F.Y(py + 0.9), 'green');
        },
        slice: function () { return { g: function (t) { return f(px, py + t); },
                                      m: 4, yr: [0, 18], xl: 'nudge y', c: 'green' }; },
        f: '∂f/∂y  =  4y', w: 'at (2, 1):  4 × 1  =  4',
        a: 'Now freeze x at 2 and walk along y instead. A different axis, its own curve, ' +
           'its own slope.',
        b: 'Here both slopes happen to be 4. Move the point and they part company at once.' },
      { l: '∇f', c: 'amber',
        mark: function (g) {
          arrow(svg, F.X(px), F.Y(py), F.X(px + 0.85), F.Y(py + 0.85), 'amber');
        },
        vec: true,
        f: '∇f  =  [ ∂f/∂x , ∂f/∂y ]', w: 'at (2, 1):  [ 4 , 4 ]',
        a: 'Collect one slope per input into a list and that list is the gradient. It is a ' +
           'vector: 4 along x, 4 along y, and the diagonal arrow they add up to.',
        b: 'A model with 7 billion parameters has a gradient with 7 billion entries. Same ' +
           'object, same rule, longer list — and training steps the other way along it.' }
    ];

    function drawSlice(v) {
      cut.innerHTML = '';
      if (v.vec) {                       /* the two components, added */
        var G = new Frame(cut, cutBox, [-1, 6], [-1, 6]);
        G.axes('along x', 'along y');
        el('line', { x1: G.X(0), y1: G.Y(0), x2: G.X(4), y2: G.Y(0),
                     stroke: css('--accent'), 'stroke-width': 2.4 }, cut);
        el('line', { x1: G.X(4), y1: G.Y(0), x2: G.X(4), y2: G.Y(4),
                     stroke: css('--green'), 'stroke-width': 2.4 }, cut);
        arrow(cut, G.X(0), G.Y(0), G.X(4), G.Y(4), 'amber');
        txt(cut, G.X(2), G.Y(0) + 15, '4', 'accent', 11.5);
        txt(cut, G.X(4) + 9, G.Y(2), '4', 'green', 11.5, 'start');
        txt(cut, G.X(1.6), G.Y(3.1), 'length √32 ≈ 5.7', 'amber', 11);
        return;
      }
      var sl = v.slice(), G2 = new Frame(cut, cutBox, [-1.5, 1.5], sl.yr);
      G2.axes(sl.xl, 'f');
      G2.path(sl.g, 160, 's-curve', { stroke: css('--' + sl.c) });
      var y0 = sl.g(0);
      el('line', { x1: G2.X(-1.1), y1: G2.Y(y0 - sl.m * 1.1),
                   x2: G2.X(1.1),  y2: G2.Y(y0 + sl.m * 1.1),
                   stroke: css('--pink'), 'stroke-width': 2,
                   'stroke-dasharray': '5 3' }, cut);
      el('circle', { cx: G2.X(0), cy: G2.Y(y0), r: 4.5, fill: css('--amber') }, cut);
      txt(cut, G2.X(0.25), G2.Y(y0 + sl.m * 1.25), 'slope 4', 'pink', 11, 'start');
    }

    var bar = ctl(host), btns = [];
    function pick(k) {
      var v = VIEWS[k];
      overlay.innerHTML = ''; v.mark(overlay);
      drawSlice(v);
      read.innerHTML = '';
      var m = h('div', 'mth', null, read);
      h('div', 'f', v.f, m); h('div', 'w', v.w, m);
      h('p', 'rd-a', v.a, read); h('p', 'rd-b', v.b, read);
      btns.forEach(function (b, j) { b.className = j === k ? 'on' : ''; });
    }
    VIEWS.forEach(function (v, k) { btns.push(button(bar, v.l, function () { pick(k); })); });
    pick(0);
    cap(host, 'Left: f(x, y) = x² + 2y² seen from above — pale is low, dark is high, so the ' +
              'pale patch in the middle is the bottom of the bowl. Right: what is left of the ' +
              'surface once you hold one input still.');
  }

  /* -----------------------------------------------------------------
     chain-rule — slopes multiply. This figure IS backpropagation.
     ----------------------------------------------------------------- */
  function vizChain(host) {
    var W = 660, H = 210, svg = makeSVG(host, W, H);
    var bx = [70, 260, 450], by = 58, bw = 130, bh = 56;
    var LB = [
      { t: 'x', s: 'input', v: '2' },
      { t: 'u = 3x', s: 'du/dx = 3', v: '6' },
      { t: 'y = u²', s: 'dy/du = 2u = 12', v: '36' }
    ];
    var boxes = [], vals = [];
    LB.forEach(function (b, i) {
      var r = el('rect', { x: bx[i], y: by, width: bw, height: bh, rx: 8,
                           class: 'sh-cell sh-front' }, svg);
      boxes.push(r);
      txt(svg, bx[i] + bw / 2, by + 24, b.t, 'chalk', 14);
      txt(svg, bx[i] + bw / 2, by + 43, b.s, 'tx-faint', 11);
      vals.push(txt(svg, bx[i] + bw / 2, by + 82, '', 'amber', 13));
      if (i) arrow(svg, bx[i] - 58, by + bh / 2, bx[i] - 8, by + bh / 2, 'tx-dim');
    });
    var back = el('g', {}, svg);
    var read = h('div', 'shapenote', null, host);

    function show(stage) {
      back.innerHTML = '';
      vals.forEach(function (v, i) { v.textContent = stage >= i + 1 ? LB[i].v : ''; });
      boxes.forEach(function (b, i) {
        b.setAttribute('class', 'sh-cell sh-front' +
          (stage >= 1 && stage <= 3 && i === stage - 1 ? ' sh-hi' : ''));
      });
      read.innerHTML = '';
      if (stage < 4) {
        h('p', 'rd-s', 'forward', read);
        h('p', 'rd-a', stage === 0 ? 'Press step. Two functions, one after the other: ' +
                                     'triple the input, then square it.'
                     : 'x = 2 → u = 6 → y = 36. Nothing clever yet; just running the chain.', read);
      } else {
        arrow(svg, bx[2] + bw / 2, by + bh + 26, bx[1] + bw / 2, by + bh + 26, 'pink');
        arrow(svg, bx[1] + bw / 2, by + bh + 26, bx[0] + bw / 2, by + bh + 26, 'pink');
        txt(back, (bx[1] + bx[2]) / 2 + bw / 2, by + bh + 18, '× 12', 'pink', 12);
        txt(back, (bx[0] + bx[1]) / 2 + bw / 2, by + bh + 18, '× 3', 'pink', 12);
        h('p', 'rd-s', 'backward', read);
        var m = h('div', 'mth', null, read);
        h('div', 'f', 'dy/dx  =  dy/du  ×  du/dx', m);
        h('div', 'w', '=  12  ×  3  =  36', m);
        h('p', 'rd-a', 'Each stage knows only its own local slope. Multiply them along the ' +
                       'chain and you have the slope of the whole thing — without ever writing ' +
                       'out the combined function.', read);
        h('p', 'rd-b', 'That is backpropagation in full. A 96-layer transformer is this ' +
                       'picture with 96 boxes, and each box still only has to know its own ' +
                       'derivative.', read);
      }
    }
    var at = 0, bar = ctl(host);
    button(bar, 'step', function () { at = Math.min(4, at + 1); show(at); });
    button(bar, 'reset', function () { at = 0; show(0); });
    show(0);
    cap(host, 'Forward, the values travel left to right. Backward, the slopes travel right ' +
              'to left and multiply. Two passes over the same chain — and the second one is ' +
              'the reason a network can be trained at all.');
  }

  /* -----------------------------------------------------------------
     saddle — flat, and not a minimum. Where high-dimensional
     optimisation actually gets stuck.
     ----------------------------------------------------------------- */
  function vizSaddle(host) {
    var W = 660, H = 320, box = { l: 56, t: 18, w: 250, h: 240 };
    var svg = makeSVG(host, W, H);
    function f(x, y) { return x * x - y * y; }
    var F = field(svg, box, f, [-3, 3], [-3, 3]);
    F.axes('x', 'y');
    el('circle', { cx: F.X(0), cy: F.Y(0), r: 5, fill: css('--amber') }, svg);
    txt(svg, F.X(0) + 10, F.Y(0) - 9, '(0, 0)', 'amber', 11, 'start');

    var cutBox = { l: 380, t: 40, w: 230, h: 190 };
    var cut = el('g', {}, svg);
    var read = h('div', 'shapenote', null, host);
    var overlay = el('g', {}, svg);

    var VIEWS = [
      { l: 'walk along x', c: 'accent', g: function (t) { return t * t; },
        line: function (g) { el('line', { x1: F.X(-3), y1: F.Y(0), x2: F.X(3), y2: F.Y(0),
                                          stroke: css('--accent'), 'stroke-width': 2.4 }, g); },
        a: 'Along x the surface curves upward. Step either way and the value rises — ' +
           'from here it looks exactly like the bottom of a valley.' },
      { l: 'walk along y', c: 'pink', g: function (t) { return -t * t; },
        line: function (g) { el('line', { x1: F.X(0), y1: F.Y(-3), x2: F.X(0), y2: F.Y(3),
                                          stroke: css('--pink'), 'stroke-width': 2.4 }, g); },
        a: 'Along y it curves downward. Same point, and now it looks like the top of a hill.' }
    ];
    var bar = ctl(host), btns = [];
    function pick(k) {
      var v = VIEWS[k];
      overlay.innerHTML = ''; v.line(overlay);
      cut.innerHTML = '';
      var G = new Frame(cut, cutBox, [-3, 3], [-9, 9]);
      G.axes('step from the point', 'f');
      G.path(v.g, 160, 's-curve', { stroke: css('--' + v.c) });
      el('circle', { cx: G.X(0), cy: G.Y(0), r: 4.5, fill: css('--amber') }, cut);
      read.innerHTML = '';
      h('p', 'rd-s', 'the gradient here is [0, 0] — the surface is flat', read);
      h('p', 'rd-a', v.a, read);
      h('p', 'rd-b', 'Both are true at once, which is why it is a saddle and not a minimum. ' +
                     'Gradient descent stops here because the slope is zero, and it is not ' +
                     'the bottom of anything. In a model with millions of parameters, a flat ' +
                     'point going down in even one direction is far more common than a real ' +
                     'minimum — saddles, not local minima, are the thing that stalls training.',
        read);
      btns.forEach(function (b, j) { b.className = j === k ? 'on' : ''; });
    }
    VIEWS.forEach(function (v, k) { btns.push(button(bar, v.l, function () { pick(k); })); });
    pick(0);
    cap(host, 'f(x, y) = x² − y² seen from above; pale is low, dark is high. Flat is not the ' +
              'same as lowest, and a zero gradient only ever promises flat.');
  }

  /* =================================================================
     OPTIMISATION, PROBABILITY, INFORMATION — the arithmetic, run.
     ================================================================= */

  /* -----------------------------------------------------------------
     gd-step — one training step, with the actual numbers in it.
     ----------------------------------------------------------------- */
  function vizGDStep(host) {
    var W = 660, H = 300, box = { l: 72, t: 20, w: 316, h: 216 };
    var svg = makeSVG(host, W, H);
    function L(w) { return w * w; }
    var F = new Frame(svg, box, [-3.4, 3.4], [0, 11]);
    F.axes('weight  w', 'loss  L = w²');
    F.path(L, 240);

    var trail = el('g', {}, svg);
    var ball = el('circle', { r: 7, fill: css('--amber') }, svg);
    var tan = el('line', { stroke: css('--pink'), 'stroke-width': 2 }, svg);
    var rows = el('g', {}, svg);
    var w = 3, lr = 0.3, step = 0, read = h('div', 'shapenote', null, host);

    function paint(g, s, nw) {
      ball.setAttribute('cx', F.X(w)); ball.setAttribute('cy', F.Y(L(w)));
      var d = 0.9;
      tan.setAttribute('x1', F.X(w - d)); tan.setAttribute('y1', F.Y(L(w) - g * d));
      tan.setAttribute('x2', F.X(w + d)); tan.setAttribute('y2', F.Y(L(w) + g * d));
      rows.innerHTML = '';
      var lines = [
        ['w', w.toFixed(3)],
        ['dL/dw = 2w', g.toFixed(3)],
        ['step = lr × slope', s === null ? '—' : s.toFixed(3)],
        ['w − step', nw === null ? '—' : nw.toFixed(3)]
      ];
      lines.forEach(function (l, i) {
        txt(rows, 430, 60 + i * 34, l[0], 'tx-dim', 12, 'start');
        txt(rows, 620, 60 + i * 34, l[1], i === 3 ? 'amber' : 'chalk', 14, 'end');
      });
      txt(rows, 430, 30, 'step ' + step + '   ·   lr = ' + lr.toFixed(2), 'tx-faint', 11.5, 'start');
    }

    function advance() {
      var g = 2 * w, s = lr * g, nw = w - s;
      paint(g, s, nw);
      el('circle', { cx: F.X(w), cy: F.Y(L(w)), r: 3, fill: css('--gold-dim'),
                     opacity: 0.6 }, trail);
      read.innerHTML = '';
      var m = h('div', 'mth', null, read);
      h('div', 'f', 'w  ←  w  −  lr × dL/dw', m);
      h('div', 'w', '= ' + w.toFixed(3) + ' − ' + lr.toFixed(2) + ' × ' + g.toFixed(3) +
                    ' = ' + nw.toFixed(3), m);
      h('p', 'rd-a', 'The slope at w = ' + w.toFixed(2) + ' is ' + g.toFixed(2) +
                     ', which is positive — the loss goes up as w goes up. So move w down. ' +
                     'How far down is the learning rate’s job, and nothing else’s.', read);
      h('p', 'rd-b', 'That single line is the whole of training. A 7-billion-parameter model ' +
                     'runs it 7 billion times per step, and not one of those numbers knows ' +
                     'anything about the others.', read);
      w = nw; step++;
    }

    var bar = ctl(host);
    slider(bar, 'lr', 0.05, 0.95, 0.05, 0.3, function (v) { lr = v; return v.toFixed(2); });
    button(bar, 'take a step', advance);
    button(bar, 'reset', function () {
      w = 3; step = 0; trail.innerHTML = ''; paint(6, null, null);
      read.innerHTML = '';
      h('p', 'rd-a', 'The ball sits at w = 3, where the loss is 9. Press “take a step”.', read);
    });
    paint(6, null, null);
    h('p', 'rd-a', 'The ball sits at w = 3, where the loss is 9. Press “take a step”.', read);
    cap(host, 'Push lr up past about 0.9 and the steps overshoot the bottom and start growing — ' +
              'the same divergence you see as a loss curve that suddenly turns upward. ' +
              'Nothing has broken; the step size is simply bigger than the valley.');
  }

  /* -----------------------------------------------------------------
     clipping — the gradient is a vector, so it has a length, and the
     threshold is a limit on that length. Nothing more.
     ----------------------------------------------------------------- */
  function vizClip(host) {
    var W = 560, H = 300, box = { l: 70, t: 22, w: 226, h: 226 };
    var svg = makeSVG(host, W, H);
    var F = new Frame(svg, box, [-1, 9], [-1, 9]);
    var i;
    for (i = 0; i <= 8; i += 2) {
      el('line', { x1: F.X(i), y1: F.Y(-1), x2: F.X(i), y2: F.Y(9), class: 's-grid' }, svg);
      el('line', { x1: F.X(-1), y1: F.Y(i), x2: F.X(9), y2: F.Y(i), class: 's-grid' }, svg);
    }
    F.axes('g₁', 'g₂');
    var gx = 6, gy = 4.5, len = Math.sqrt(gx * gx + gy * gy);
    var ring = el('circle', { cx: F.X(0), cy: F.Y(0), fill: 'none',
                              stroke: css('--pink'), 'stroke-width': 1.6,
                              'stroke-dasharray': '5 4' }, svg);
    var raw = el('g', {}, svg), clipped = el('g', {}, svg);
    arrow(raw, F.X(0), F.Y(0), F.X(gx), F.Y(gy), 'tx-faint');
    txt(svg, F.X(gx) + 8, F.Y(gy) - 4, 'raw gradient', 'tx-faint', 11, 'start');

    var cards = [], read = h('div', 'shapenote', null, host);
    ['raw length', 'threshold', 'after clipping'].forEach(function (l, k) {
      var y = box.t + 12 + k * 72;
      cards.push({ box: el('rect', { x: 360, y: y, width: 172, height: 58, rx: 8,
                                     class: 'sh-cell sh-front' }, svg),
                   lab: txt(svg, 374, y + 24, l, 'tx-dim', 12, 'start'),
                   val: txt(svg, 518, y + 40, '', 'amber', 18, 'end') });
    });

    function draw(thr) {
      var scale = len > thr ? thr / len : 1;
      ring.setAttribute('r', Math.abs(F.X(thr) - F.X(0)));
      clipped.innerHTML = '';
      arrow(clipped, F.X(0), F.Y(0), F.X(gx * scale), F.Y(gy * scale), 'amber');
      cards[0].val.textContent = len.toFixed(2);
      cards[1].val.textContent = thr.toFixed(2);
      cards[2].val.textContent = (len * scale).toFixed(2);
      cards[2].box.setAttribute('class', 'sh-cell sh-front' + (scale < 1 ? ' sh-hi' : ''));
      read.innerHTML = '';
      var m = h('div', 'mth', null, read);
      h('div', 'f', 'if ‖g‖ > threshold:   g  ←  g × threshold / ‖g‖', m);
      h('div', 'w', scale < 1
        ? '‖g‖ = ' + len.toFixed(2) + ' > ' + thr.toFixed(2) + ', so g × ' +
          thr.toFixed(2) + '/' + len.toFixed(2) + ' = g × ' + scale.toFixed(3)
        : '‖g‖ = ' + len.toFixed(2) + ' ≤ ' + thr.toFixed(2) + ', so nothing happens', m);
      h('p', 'rd-a', scale < 1
        ? 'The arrow is shortened to sit exactly on the dashed circle. It still points the ' +
          'same way — the model is told the same thing about direction, just more quietly.'
        : 'Under the threshold, clipping is a no-op. It is a safety rail, not a normaliser: ' +
          'most steps never touch it.', read);
      h('p', 'rd-b', 'Why bother: one bad batch can produce a gradient hundreds of times the ' +
                     'usual size, and a single step along it lands the weights somewhere the ' +
                     'model never recovers from. Clipping caps the damage of the worst step ' +
                     'you will ever take.', read);
    }
    var bar = ctl(host);
    slider(bar, 'threshold', 0.5, 10, 0.25, 3, function (v) { draw(v); return v.toFixed(2); });
    cap(host, 'A gradient with two entries, drawn as an arrow: 6 along one weight, 4.5 along ' +
              'the other, so its L2 length is 7.5. Everything outside the dashed circle gets ' +
              'pulled back onto it.');
  }

  /* -----------------------------------------------------------------
     surprise — why a logarithm, in units you can count.
     ----------------------------------------------------------------- */
  function vizSurprise(host) {
    var W = 640, H = 290, box = { l: 60, t: 20, w: 330, h: 208 };
    var svg = makeSVG(host, W, H);
    var F = new Frame(svg, box, [0.01, 1], [0, 7]);
    F.axes('probability you gave it', 'surprise (bits)');
    F.path(function (p) { return -Math.log(p) / Math.LN2; }, 300);
    var dot = el('circle', { r: 6, fill: css('--amber') }, svg);
    var drop = el('line', { stroke: css('--gold-dim'), 'stroke-width': 1.2,
                            'stroke-dasharray': '3 3' }, svg);
    var read = h('div', 'shapenote', null, host);
    var pane = el('g', {}, svg);

    function draw(p) {
      var s = -Math.log(p) / Math.LN2;
      dot.setAttribute('cx', F.X(p)); dot.setAttribute('cy', F.Y(Math.min(7, s)));
      drop.setAttribute('x1', F.X(p)); drop.setAttribute('y1', F.Y(0));
      drop.setAttribute('x2', F.X(p)); drop.setAttribute('y2', F.Y(Math.min(7, s)));
      pane.innerHTML = '';
      txt(pane, 430, 60, 'you said', 'tx-dim', 12, 'start');
      txt(pane, 620, 60, (p * 100).toFixed(0) + ' %', 'chalk', 16, 'end');
      txt(pane, 430, 108, 'and it happened', 'tx-dim', 12, 'start');
      txt(pane, 620, 110, 'yes', 'chalk', 16, 'end');
      txt(pane, 430, 150, 'surprise', 'tx-dim', 12, 'start');
      txt(pane, 620, 152, s.toFixed(2) + ' bits', 'amber', 18, 'end');
      txt(pane, 620, 176, '= ' + (s * Math.LN2).toFixed(2) + ' nats', 'tx-faint', 11.5, 'end');
      read.innerHTML = '';
      var m = h('div', 'mth', null, read);
      h('div', 'f', 'surprise  =  −log₂ p', m);
      h('div', 'w', '= −log₂(' + p.toFixed(3) + ')  =  ' + s.toFixed(2) + ' bits', m);
      h('p', 'rd-a', 'One bit is one yes/no question. If you called it a coin flip (p = 0.5) ' +
                     'and it came up heads, you learned exactly one question’s worth: 1 bit.', read);
      h('p', 'rd-b', p > 0.9
        ? 'You were nearly certain and you were right, so almost nothing was learned — ' +
          s.toFixed(2) + ' bits.'
        : p < 0.06
          ? 'You called it almost impossible and it happened. ' + s.toFixed(1) + ' bits is a ' +
            'genuine shock, and a loss function built on this will punish you for it.'
          : 'Halve the probability and the surprise goes up by exactly one bit — that is the ' +
            'property the logarithm was chosen for, and the only reason it appears here.', read);
    }
    var bar = ctl(host);
    slider(bar, 'p', 0.01, 0.99, 0.01, 0.5, function (v) { draw(v); return v.toFixed(2); });
    cap(host, 'Surprise has to be zero when you were certain and large when you were not, and ' +
              'the surprise of two unrelated events has to add up. Only the logarithm does ' +
              'both — it is picked for that, not for tradition.');
  }

  /* -----------------------------------------------------------------
     crossentropy — the loss, computed on a real prediction.
     ----------------------------------------------------------------- */
  function vizCrossEntropy(host) {
    var CLS = ['cat', 'dog', 'fox', 'owl'];
    var W = 620, H = 230, svg = makeSVG(host, W, H);
    var base = 176, bw = 50, x0 = 70, gapx = 74;
    var bars = [], labs = [], names = [], read = h('div', 'shapenote', null, host);
    var truth = 0, pt = 0.6;
    el('line', { x1: 54, y1: base, x2: 396, y2: base, class: 's-axis' }, svg);
    txt(svg, 54, 26, 'what the model said', 'tx-faint', 11.5, 'start');
    CLS.forEach(function (c, i) {
      bars.push(el('rect', { x: x0 + i * gapx, y: base, width: bw, height: 0, rx: 4,
                             class: 'sh-cell sh-front' }, svg));
      names.push(txt(svg, x0 + i * gapx + bw / 2, base + 18, c, 'tx-dim', 12));
      labs.push(txt(svg, x0 + i * gapx + bw / 2, base - 6, '', 'chalk', 12));
    });
    var pane = el('g', {}, svg);

    function draw() {
      /* The model's prediction is fixed by the slider — it peaks on "cat"
         and knows nothing about which label turns out to be right. Cycling
         the truth therefore changes the loss without changing the bars,
         which is the point of the figure. */
      var rest = (1 - pt) / 3;
      var p = CLS.map(function (c, i) { return i === 0 ? pt : rest; });
      var pr = p[truth];
      p.forEach(function (v, i) {
        var hgt = v * 128;
        bars[i].setAttribute('y', base - hgt);
        bars[i].setAttribute('height', hgt);
        bars[i].setAttribute('class', 'sh-cell sh-front' + (i === truth ? ' sh-hi' : ''));
        labs[i].setAttribute('y', base - hgt - 7);
        labs[i].textContent = v.toFixed(2);
        names[i].setAttribute('fill', css(i === truth ? '--amber' : '--tx-dim'));
      });
      var loss = -Math.log(pr);
      pane.innerHTML = '';
      txt(pane, 440, 52, 'the answer that was right', 'tx-dim', 12, 'start');
      txt(pane, 596, 78, CLS[truth], 'amber', 17, 'end');
      txt(pane, 440, 120, 'probability you gave it', 'tx-dim', 12, 'start');
      txt(pane, 596, 146, pr.toFixed(2), 'chalk', 17, 'end');
      txt(pane, 440, 186, 'loss', 'tx-dim', 12, 'start');
      txt(pane, 596, 188, loss.toFixed(3), 'amber', 22, 'end');
      read.innerHTML = '';
      var m = h('div', 'mth', null, read);
      h('div', 'f', 'cross-entropy  =  − Σ true(i) × log p(i)', m);
      h('div', 'w', 'true(' + CLS[truth] + ') = 1 and every other true(i) = 0, so three of the ' +
                    'four terms are multiplied away:  −log(' + pr.toFixed(2) + ')  =  ' +
                    loss.toFixed(3), m);
      h('p', 'rd-a', 'The formula has four terms and three of them vanish. What is left is one ' +
                     'question: how much probability did you put on the answer that turned out ' +
                     'to be right? Nothing the model said about the other three classes enters ' +
                     'the loss at all.', read);
      h('p', 'rd-b', pr > 0.85
        ? 'Confident and correct: loss ' + loss.toFixed(3) + ', almost nothing left to learn here.'
        : pr < 0.15
          ? 'You put ' + (pr * 100).toFixed(0) + ' % on the right answer and the loss is ' +
            loss.toFixed(2) + '. Note that it is not capped — being confidently wrong costs ' +
            'without limit, and that is deliberate.'
          : 'This is the surprise figure above, one level up: surprise at a single outcome, ' +
            'averaged over a dataset. Same quantity, same logarithm.', read);
    }
    var bar = ctl(host);
    slider(bar, 'p(cat)', 0.02, 0.97, 0.01, 0.6, function (v) {
      pt = v; draw(); return v.toFixed(2);
    });
    button(bar, 'the truth was something else', function () {
      truth = (truth + 1) % CLS.length; draw();
    });
    draw();
    cap(host, 'Press the second button and the same four predictions get a different loss, ' +
              'because a different bar is now the one that counts. The model did not change; ' +
              'reality did.');
  }

  /* -----------------------------------------------------------------
     clt — averages go bell-shaped even when the thing being averaged
     does not. Watch it happen rather than take it on trust.
     ----------------------------------------------------------------- */
  function vizCLT(host) {
    var W = 640, H = 306, svg = makeSVG(host, W, H);
    var srcBox = { l: 60, t: 30, w: 230, h: 76 };
    var outBox = { l: 60, t: 176, w: 520, h: 96 };
    var HI = 3;                       /* both histograms are drawn over 0..3 */
    txt(svg, srcBox.l, 20, 'one draw — an ordinary skewed quantity, no bell in sight',
        'tx-faint', 11.5, 'start');
    txt(svg, outBox.l, 166, 'the average of n draws, one bar per average', 'tx-faint', 11.5, 'start');

    /* Source: exponential with mean 1. Continuous, so the averages land
       anywhere rather than on a lattice — a discrete source makes the
       lower histogram comb-shaped for arithmetic reasons that have
       nothing to do with the theorem being demonstrated. */
    function drawOne() { return -Math.log(1 - Math.random()); }

    var NS = 30, i, sw = srcBox.w / NS;
    for (i = 0; i < NS; i++) {
      var xm = (i + 0.5) / NS * HI, dens = Math.exp(-xm);
      var hgt = dens * srcBox.h;
      el('rect', { x: srcBox.l + i * sw, y: srcBox.t + srcBox.h - hgt,
                   width: Math.max(1, sw - 1), height: hgt,
                   class: 'sh-cell sh-front' }, svg);
    }
    el('line', { x1: srcBox.l, y1: srcBox.t + srcBox.h, x2: srcBox.l + srcBox.w,
                 y2: srcBox.t + srcBox.h, class: 's-axis' }, svg);

    var NB = 44, bins = new Array(NB), n = 1, drawn = 0;
    var barsG = el('g', {}, svg), bell = el('path', { fill: 'none', stroke: css('--pink'),
                                                      'stroke-width': 1.8, opacity: 0.85 }, svg);
    el('line', { x1: outBox.l, y1: outBox.t + outBox.h, x2: outBox.l + outBox.w,
                 y2: outBox.t + outBox.h, class: 's-axis' }, svg);
    for (i = 0; i <= 3; i++) {
      var tx = outBox.l + i / HI * outBox.w;
      el('line', { x1: tx, y1: outBox.t + outBox.h, x2: tx, y2: outBox.t + outBox.h + 5,
                   class: 's-axis' }, svg);
      txt(svg, tx, outBox.t + outBox.h + 18, String(i), 'tx-faint', 11);
    }
    var read = h('div', 'shapenote', null, host);

    function reset() {
      for (var k = 0; k < NB; k++) bins[k] = 0;
      drawn = 0; paint();
    }
    function run(k) {
      for (var t = 0; t < k; t++) {
        var acc = 0;
        for (var j = 0; j < n; j++) acc += drawOne();
        var m = acc / n;
        if (m < HI) bins[Math.floor(m / HI * NB)]++;
        drawn++;
      }
      paint();
    }
    function paint() {
      var max = Math.max.apply(null, bins) || 1, bwid = outBox.w / NB, k;
      barsG.innerHTML = '';
      for (k = 0; k < NB; k++) {
        var hgt = bins[k] / max * outBox.h;
        el('rect', { x: outBox.l + k * bwid, y: outBox.t + outBox.h - hgt,
                     width: Math.max(1, bwid - 1), height: hgt,
                     class: 'sh-cell sh-front' }, barsG);
      }
      /* the Gaussian the theorem predicts: mean 1, sd 1/√n */
      if (drawn > 300) {
        var sd = 1 / Math.sqrt(n), d = '';
        for (k = 0; k <= 120; k++) {
          var x = k / 120 * HI, z = (x - 1) / sd;
          var y = Math.exp(-0.5 * z * z);
          d += (k ? 'L' : 'M') + (outBox.l + x / HI * outBox.w).toFixed(1) + ',' +
               (outBox.t + outBox.h - y * outBox.h).toFixed(1);
        }
        bell.setAttribute('d', d);
      } else bell.setAttribute('d', '');

      read.innerHTML = '';
      h('p', 'rd-s', drawn + ' averages drawn   ·   n = ' + n + ' draw' + (n > 1 ? 's' : '') +
                     ' per average   ·   predicted spread 1/√n = ' +
                     (1 / Math.sqrt(n)).toFixed(3), read);
      h('p', 'rd-a', 'The top histogram is the thing being sampled: most draws are small, a few ' +
                     'are large, and it is lopsided. At n = 1 the bottom histogram is that same ' +
                     'shape, because an average of one number is the number.', read);
      h('p', 'rd-b', drawn < 300
        ? 'Draw a few hundred and the shape settles. Then raise n.'
        : 'Raise n and two things happen at once: the shape becomes symmetric, and the width ' +
          'shrinks by 1/√n. The pink curve is the bell the theorem predicts from n alone — ' +
          'nothing about the skewed source went into drawing it, and the bars land on it ' +
          'regardless. Quadrupling your sample halves the error; that is the same √n every ' +
          'confidence interval and every A/B test sample-size calculation is built on.', read);
    }
    var bar = ctl(host);
    slider(bar, 'n', 1, 40, 1, 1, function (v) {
      n = v; var d = drawn; reset(); if (d) run(Math.min(4000, Math.max(2000, d))); return v;
    });
    button(bar, 'draw 200', function () { run(200); });
    button(bar, 'draw 2000', function () { run(2000); });
    button(bar, 'reset', reset);
    reset();
    cap(host, 'The source never changes — only how many of its draws go into each average. ' +
              'At n = 1 the bottom picture is the top one. By n = 10 it is a bell, and by ' +
              'n = 40 it is a narrow one.');
  }

  /* -----------------------------------------------------------------
     expectation — E[X] and Var[X] on a die you can bend, worked out
     term by term rather than asserted.
     ----------------------------------------------------------------- */
  function vizExpectation(host) {
    var W = 640, H = 250, svg = makeSVG(host, W, H);
    var P = [1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6];
    var base = 190, bw = 46, x0 = 56, gapx = 62;
    var bars = [], labs = [], read = h('div', 'shapenote', null, host);
    el('line', { x1: 40, y1: base, x2: 420, y2: base, class: 's-axis' }, svg);
    for (var i = 0; i < 6; i++) {
      bars.push(el('rect', { x: x0 + i * gapx, y: base, width: bw, height: 0, rx: 4,
                             class: 'sh-cell sh-front' }, svg));
      txt(svg, x0 + i * gapx + bw / 2, base + 18, String(i + 1), 'tx-dim', 12);
      labs.push(txt(svg, x0 + i * gapx + bw / 2, base - 6, '', 'chalk', 11.5));
    }
    var mean = el('line', { stroke: css('--amber'), 'stroke-width': 2,
                            'stroke-dasharray': '5 4' }, svg);
    var mlab = txt(svg, 0, 0, '', 'amber', 12);
    var pane = el('g', {}, svg);

    function draw() {
      var s = P.reduce(function (a, b) { return a + b; }, 0);
      var p = P.map(function (v) { return v / s; });
      var ex = 0, i;
      for (i = 0; i < 6; i++) ex += (i + 1) * p[i];
      var vr = 0;
      for (i = 0; i < 6; i++) vr += p[i] * Math.pow(i + 1 - ex, 2);
      for (i = 0; i < 6; i++) {
        var hgt = p[i] / 0.5 * 130;
        bars[i].setAttribute('y', base - hgt);
        bars[i].setAttribute('height', hgt);
        labs[i].setAttribute('y', base - hgt - 6);
        labs[i].textContent = p[i].toFixed(2);
      }
      var mx = x0 + (ex - 1) * gapx + bw / 2;
      mean.setAttribute('x1', mx); mean.setAttribute('y1', base + 6);
      mean.setAttribute('x2', mx); mean.setAttribute('y2', 42);
      mlab.setAttribute('x', mx); mlab.setAttribute('y', 34);
      mlab.textContent = 'E[X] = ' + ex.toFixed(2);
      pane.innerHTML = '';
      txt(pane, 450, 70, 'E[X]', 'tx-dim', 12, 'start');
      txt(pane, 610, 70, ex.toFixed(3), 'amber', 18, 'end');
      txt(pane, 450, 118, 'Var[X]', 'tx-dim', 12, 'start');
      txt(pane, 610, 118, vr.toFixed(3), 'chalk', 16, 'end');
      txt(pane, 450, 158, 'σ', 'tx-dim', 12, 'start');
      txt(pane, 610, 158, Math.sqrt(vr).toFixed(3), 'chalk', 16, 'end');
      read.innerHTML = '';
      var m = h('div', 'mth', null, read);
      h('div', 'f', 'E[X]  =  Σ  value × its probability', m);
      h('div', 'w', p.map(function (v, k) {
        return (k + 1) + '×' + v.toFixed(2);
      }).join('  +  ') + '  =  ' + ex.toFixed(3), m);
      var m2 = h('div', 'mth', null, read);
      h('div', 'f', 'Var[X]  =  Σ  probability × (value − E[X])²', m2);
      h('div', 'w', 'each face’s distance from ' + ex.toFixed(2) + ', squared, weighted:  ' +
                    vr.toFixed(3) + '   →   σ = ' + Math.sqrt(vr).toFixed(3), m2);
      h('p', 'rd-a', 'The expectation is not “the most likely value” and need not be a value the ' +
                     'die can even show — a fair die averages 3.5, a face that does not exist. ' +
                     'It is the balance point of the bars.', read);
      h('p', 'rd-b', 'This is exactly what a training loss is: the expectation of the per-example ' +
                     'loss over the data distribution. You cannot compute that sum — you have a ' +
                     'batch, not the distribution — so you average the batch and accept the noise. ' +
                     'That noise is the “stochastic” in stochastic gradient descent.', read);
    }
    var bar = ctl(host);
    slider(bar, 'weight on 6', 1, 12, 0.5, 1, function (v) {
      P[5] = v / 6; draw(); return (v).toFixed(1) + '×';
    });
    button(bar, 'fair die', function () {
      for (var i = 0; i < 6; i++) P[i] = 1 / 6; draw();
    });
    draw();
    cap(host, 'Load the six and watch both numbers move. The balance point slides right ' +
              'immediately. The spread does something less obvious: it widens at first, because ' +
              'the die is now torn between the six and everything else, and only narrows again ' +
              'once the six really dominates. A certain outcome has variance zero.');
  }

  /* -----------------------------------------------------------------
     momentum — the ravine, and why an average of past gradients gets
     down it faster than the gradients themselves.
     ----------------------------------------------------------------- */
  function vizMomentum(host) {
    var W = 640, H = 300, box = { l: 66, t: 22, w: 540, h: 232 };
    var svg = makeSVG(host, W, H);
    var F = new Frame(svg, box, [-1.4, 10.5], [-2.6, 2.6]);
    /* f = A x² + B y². A is tiny and B is not: the valley is 900 times
       steeper across than along, which is the whole reason momentum has
       anything to do. Its contours are so stretched that on screen they
       are the two near-horizontal walls, not visible ovals. */
    var A = 0.002, B = 1.8, k;
    for (k = 1; k <= 4; k++) {
      var ry = k * 0.6, lv = B * ry * ry;
      el('ellipse', { cx: F.X(0), cy: F.Y(0),
                      rx: Math.abs(F.X(Math.sqrt(lv / A)) - F.X(0)),
                      ry: Math.abs(F.Y(ry) - F.Y(0)),
                      fill: 'none', stroke: css('--rule-line'), 'stroke-width': 1 }, svg);
    }
    txt(svg, box.l + box.w - 6, F.Y(2.3), 'valley wall', 'tx-faint', 10.5, 'end');
    txt(svg, box.l + box.w - 6, F.Y(-2.3) + 10, 'valley wall', 'tx-faint', 10.5, 'end');
    F.axes('weight along the valley floor', 'weight across it');
    el('circle', { cx: F.X(0), cy: F.Y(0), r: 4, fill: css('--green') }, svg);
    txt(svg, F.X(0) + 2, F.Y(0) - 14, 'minimum', 'green', 11, 'middle');

    var pa = el('polyline', { fill: 'none', stroke: css('--tx-faint'),
                              'stroke-width': 1.8 }, svg);
    var pb = el('polyline', { fill: 'none', stroke: css('--amber'),
                              'stroke-width': 2 }, svg);
    var da = el('circle', { r: 5, fill: css('--tx-faint') }, svg);
    var db = el('circle', { r: 5.5, fill: css('--amber') }, svg);
    txt(svg, box.l + 8, box.t + 14, 'plain gradient descent', 'tx-faint', 11.5, 'start');
    var leg = txt(svg, box.l + 8, box.t + 30, '', 'amber', 11.5, 'start');

    var read = h('div', 'shapenote', null, host);
    var lr = 0.5, beta = 0.85, timer = null;
    var s = {};

    function reset() {
      s = { a: [9, 0.8], b: [9, 0.8], v: [0, 0], A: [], B: [], n: 0 };
      step(0); paint();
    }
    function grad(p) { return [2 * A * p[0], 2 * B * p[1]]; }
    function step(n) {
      for (var t = 0; t < n; t++) {
        var ga = grad(s.a);
        s.a = [s.a[0] - lr * ga[0], s.a[1] - lr * ga[1]];
        var gb = grad(s.b);
        s.v = [beta * s.v[0] + gb[0], beta * s.v[1] + gb[1]];
        s.b = [s.b[0] - lr * s.v[0], s.b[1] - lr * s.v[1]];
        s.A.push(s.a.slice()); s.B.push(s.b.slice()); s.n++;
      }
    }
    function pts(arr) {
      return arr.map(function (p) {
        return F.X(p[0]).toFixed(1) + ',' + F.Y(Math.max(-2.5, Math.min(2.5, p[1]))).toFixed(1);
      }).join(' ');
    }
    function paint() {
      leg.textContent = 'with momentum (β = ' + beta.toFixed(2) + ')';
      pa.setAttribute('points', pts(s.A)); pb.setAttribute('points', pts(s.B));
      da.setAttribute('cx', F.X(s.a[0])); da.setAttribute('cy', F.Y(clamp(s.a[1])));
      db.setAttribute('cx', F.X(s.b[0])); db.setAttribute('cy', F.Y(clamp(s.b[1])));
      var la = A * s.a[0] * s.a[0] + B * s.a[1] * s.a[1];
      var lb = A * s.b[0] * s.b[0] + B * s.b[1] * s.b[1];
      read.innerHTML = '';
      h('p', 'rd-s', 'step ' + s.n + '   ·   plain loss ' + la.toFixed(3) +
                     '   ·   momentum loss ' + lb.toFixed(3), read);
      var m = h('div', 'mth', null, read);
      h('div', 'f', 'v  ←  β v  +  gradient        θ  ←  θ − lr × v', m);
      h('div', 'w', 'β = ' + beta.toFixed(2) + ': keep ' + (beta * 100).toFixed(0) +
                    ' % of the previous velocity, add the new gradient on top. A gradient that ' +
                    'keeps pointing the same way therefore builds up to about 1/(1−β) = ' +
                    (beta < 1 ? (1 / (1 - beta)).toFixed(1) : '∞') + '× its own size', m);
      h('p', 'rd-a', 'The valley is steep across and almost flat along. Plain descent spends its ' +
                     'step size on the steep direction and bounces from wall to wall, creeping ' +
                     'forward. Momentum keeps a running total: the across-the-valley gradients ' +
                     'flip sign every step and cancel each other out in that total, while the ' +
                     'along-the-valley ones all point the same way and add up.', read);
      h('p', 'rd-b', 'Watch what it does and does not fix. Momentum still wobbles across the ' +
                     'valley — it is a heavy ball, not a brake. What it buys is the other axis: ' +
                     'the grey walker is still up near where it started while the amber one has ' +
                     'crossed most of the floor, because along that axis every gradient it has ' +
                     'ever seen pointed the same way and they are all still in the running total.', read);
    }
    function clamp(v) { return Math.max(-2.5, Math.min(2.5, v)); }

    var bar = ctl(host);
    button(bar, 'run', function () {
      if (timer) { clearInterval(timer); timer = null; return; }
      timer = setInterval(function () {
        if (s.n >= 120) { clearInterval(timer); timer = null; return; }
        step(1); paint();
      }, 60);
    });
    button(bar, 'one step', function () { step(1); paint(); });
    slider(bar, 'β', 0, 0.95, 0.05, 0.85, function (v) {
      beta = v; if (timer) { clearInterval(timer); timer = null; } reset(); return v.toFixed(2);
    });
    button(bar, 'reset', function () {
      if (timer) { clearInterval(timer); timer = null; } reset();
    });
    reset();
    cap(host, 'Set β to 0 and the amber path lies exactly on top of the grey one — momentum with ' +
              'no memory is plain gradient descent. Raise it and the ball starts carrying speed ' +
              'from one step into the next; watch how much further left it gets by step 60.');
  }

  /* =================================================================
     Matrix panels — the workhorse behind "show me the thing, and show
     me what it became". Every card that names a matrix, a vector or a
     table gets one of these rather than a sentence describing it.

     panel() returns a laid-out part with a uniform { w, h, move(x, y) }
     interface; sym() returns an operator glyph with the same interface;
     lay() centres a row of parts on a point. A card figure is then
     three or four lines of declaration.
     ================================================================= */

  function panel(svg, M, opt) {
    opt = opt || {};
    var cs = opt.cell || 34, gp = 3, R = M.length, C = M[0].length;
    var fs = opt.fs || 12.5, br = opt.bracket === false ? 0 : 10;
    var lh = opt.label ? 20 : 0, sh = opt.sub ? 17 : 0;
    var g = el('g', {}, svg), cells = [];
    var gw = C * cs + (C - 1) * gp, gh = R * cs + (R - 1) * gp;
    var r, c;
    for (r = 0; r < R; r++) for (c = 0; c < C; c++) {
      var x = br + c * (cs + gp), y = lh + r * (cs + gp);
      var extra = opt.hi ? (opt.hi(r, c) || '') : '';
      var k = el('rect', { x: x, y: y, width: cs, height: cs, rx: 4,
                           class: 'sh-cell sh-front' + (extra ? ' ' + extra : '') }, g);
      var t = el('text', { x: x + cs / 2, y: y + cs / 2 + fs * 0.36, 'text-anchor': 'middle',
                           fill: css(opt.ink || '--chalk'), 'font-size': fs }, g);
      t.textContent = String(M[r][c]);
      cells.push({ rect: k, text: t, r: r, c: c });
    }
    if (br) {
      var y0 = lh - 3, y1 = lh + gh + 3, k1 = br + gw + br;
      el('path', { d: 'M' + (br - 3) + ' ' + y0 + 'h-6v' + (y1 - y0) + 'h6',
                   fill: 'none', stroke: css('--tx-faint'), 'stroke-width': 1.4 }, g);
      el('path', { d: 'M' + (k1 - br + 3) + ' ' + y0 + 'h6v' + (y1 - y0) + 'h-6',
                   fill: 'none', stroke: css('--tx-faint'), 'stroke-width': 1.4 }, g);
    }
    if (opt.label)
      txt(g, br + gw / 2, 13, opt.label, opt.labelInk || 'amber', opt.labelSize || 12.5);
    if (opt.sub)
      txt(g, br + gw / 2, lh + gh + 14, opt.sub, 'tx-faint', 11);
    return {
      g: g, cells: cells, w: gw + 2 * br, h: lh + gh + sh,
      at: function (r2, c2) {
        for (var i = 0; i < cells.length; i++)
          if (cells[i].r === r2 && cells[i].c === c2) return cells[i];
        return null;
      },
      move: function (x, y) { g.setAttribute('transform', 'translate(' + x + ',' + y + ')'); }
    };
  }

  /* an operator between two panels: =, ×, →, a word */
  function sym(svg, s, opt) {
    opt = opt || {};
    var g = el('g', {}, svg), size = opt.size || 17;
    txt(g, 0, size * 0.35, s, opt.colour || 'tx-dim', size);
    var w = opt.w || (s.length * size * 0.62 + 6);
    return { g: g, w: w, h: size,
             move: function (x, y) {
               g.setAttribute('transform', 'translate(' + (x + w / 2) + ',' + (y + size / 2) + ')');
             } };
  }

  /* centre a row of parts horizontally on cx, aligning their middles on cy */
  function lay(parts, cx, cy, gap) {
    gap = gap == null ? 14 : gap;
    var tw = 0, i;
    for (i = 0; i < parts.length; i++) tw += parts[i].w + (i ? gap : 0);
    var x = cx - tw / 2;
    for (i = 0; i < parts.length; i++) {
      parts[i].move(x, cy - parts[i].h / 2);
      x += parts[i].w + gap;
    }
    return tw;
  }

  /* a compact figure that is one row of panels plus a worked line */
  function eqfig(host, W, H, build, note) {
    var svg = makeSVG(host, W, H);
    var read = h('div', 'shapenote', null, host);
    build(svg, read);
    if (note) cap(host, note);
    return svg;
  }

  /* the "formula, then the same formula with numbers in it" pair */
  function worked(read, f, w) {
    var m = h('div', 'mth', null, read);
    h('div', 'f', f, m);
    if (w) h('div', 'w', w, m);
    return m;
  }

  /* ---------------- module 10 · one figure per term card ---------------- */

  function vizScalar(host) {
    eqfig(host, 560, 150, function (svg, read) {
      var a = panel(svg, [['0.42']], { label: 'a scalar', sub: 'shape [ ]', cell: 38 });
      var b = panel(svg, [['0.9', '−0.4', '0.2', '1.3']], { label: 'a vector', sub: 'shape [4]' });
      var c = panel(svg, [['1', '0', '2', '1'], ['0', '3', '1', '4'], ['5', '1', '0', '2']],
                    { label: 'a matrix', sub: 'shape [3, 4]', cell: 26, fs: 11 });
      lay([a, b, c], 280, 72, 34);
      h('p', 'rd-a', 'A scalar is one number and it has no axes at all — nothing to index, ' +
                     'nothing to count along. Its shape is an empty list, which is not a ' +
                     'notational quirk: it is the base case the whole ladder is built on.', read);
      h('p', 'rd-b', 'Everything above it is the same numbers with an axis added. A vector is ' +
                     'a scalar with one axis, a matrix is a vector with a second, and a tensor ' +
                     'keeps going.', read);
    }, 'Every loss you ever print is a scalar. That is the point of a loss function: it takes ' +
       'a model with billions of numbers in it and returns exactly one, because you can only ' +
       'sort by one number.');
  }

  function vizVector(host) {
    var W = 600, H = 230, svg = makeSVG(host, W, H);
    var v = ['0.9', '−0.4', '0.2', '1.3'];
    var p = panel(svg, v.map(function (x) { return [x]; }),
                  { label: 'v', sub: 'shape [4]', cell: 34 });
    p.move(56, 26);
    for (var i = 0; i < 4; i++)
      txt(svg, 44, 26 + 20 + i * 37 + 22, 'v[' + i + ']', 'tx-faint', 10.5, 'end');

    var box = { l: 250, t: 34, w: 190, h: 150 };
    var F = new Frame(svg, box, [-1, 1.6], [-1, 1.6]);
    el('line', { x1: F.X(-1), y1: F.Y(0), x2: F.X(1.6), y2: F.Y(0), class: 's-grid' }, svg);
    el('line', { x1: F.X(0), y1: F.Y(-1), x2: F.X(0), y2: F.Y(1.6), class: 's-grid' }, svg);
    arrow(svg, F.X(0), F.Y(0), F.X(0.9), F.Y(-0.4), 'amber');
    txt(svg, F.X(0.9) + 6, F.Y(-0.4) + 14, '(0.9, −0.4)', 'amber', 11, 'start');
    txt(svg, box.l + box.w / 2, box.t - 12, 'the first two entries, drawn', 'tx-faint', 11);

    var read = h('div', 'shapenote', null, host);
    h('p', 'rd-a', 'One object, two readings. It is a list of four numbers you can index, and ' +
                   'it is an arrow with a direction and a length. Both are always true at once, ' +
                   'and which one you reach for depends on the question.', read);
    h('p', 'rd-b', 'The drawing stops at two entries because paper does. Nothing else does: an ' +
                   'embedding is the same object with 768 or 4096 entries, and every operation ' +
                   'in this module works there unchanged. You stop picturing it and start ' +
                   'trusting the arithmetic.', read);
    cap(host, 'When a paper says “the token is a point in 4096-dimensional space”, this is the ' +
              'whole content of the sentence: a list of 4096 numbers.');
  }

  function vizCosMini(host) {
    var svg = makeSVG(host, 400, 295), drawing = el('g', {}, svg);
    svg.setAttribute('role', 'img');
    svg.setAttribute('aria-label', 'Cosine similarity calculated from the same five film ratings');
    var read = h('div', 'shapenote', null, host), other = 'ben';
    function draw() {
      drawing.innerHTML = ''; read.innerHTML = '';
      var a = RATER.ana.v, b = RATER[other].v;
      var dot = a.reduce(function (sum, v, i) { return sum + v * b[i]; }, 0);
      var na = norm2(a), nb = norm2(b), cosine = dot / (na * nb);
      var pa = panel(drawing, [a], { label: 'Ana · same five films as above', cell: 38, fs: 16 });
      var pb = panel(drawing, [b], { label: RATER[other].l, cell: 38, fs: 16 });
      pa.move((400 - pa.w) / 2, 5); pb.move((400 - pb.w) / 2, 82);
      txt(drawing, 200, 177, 'dot product = ' + dot, 'amber', 18);
      txt(drawing, 200, 206, 'cosine = ' + cosine.toFixed(3), 'green', 18);
      el('line', { x1: 42, y1: 252, x2: 358, y2: 252, class: 's-axis' }, drawing);
      [-1, 0, 1].forEach(function (v) { txt(drawing, 200 + v * 158, 280, String(v), 'tx-dim', 14); });
      el('circle', { cx: 200 + cosine * 158, cy: 252, r: 6, fill: css('--green') }, drawing);
      worked(read, 'cos(a, b) = (a · b) / (‖a‖₂ ‖b‖₂)',
        dot + ' / (√' + a.reduce(function (s, v) { return s + v * v; }, 0) +
        ' × √' + b.reduce(function (s, v) { return s + v * v; }, 0) + ') = ' + cosine.toFixed(3));
      h('p', 'rd-a', 'These are the actual five-dimensional ratings, not a made-up two-dimensional ' +
        'projection. Dividing by the two lengths removes the effect of multiplying a whole ' +
        'rating vector by a positive constant.', read);
      h('p', 'rd-b', 'A value near 1 means aligned directions. Here Ana and Ben align more than ' +
        'Ana and Cara. Adding a constant to every rating can still change cosine: it does not ' +
        'automatically correct every kind of generous or harsh rating habit.', read);
    }
    var toggle = button(ctl(host), 'compare Ana with Cara', function () {
      other = other === 'ben' ? 'cara' : 'ben'; draw();
      toggle.textContent = other === 'ben' ? 'compare Ana with Cara' : 'compare Ana with Ben';
    });
    draw();
  }

  function vizInverse(host) {
    var W = 620, H = 210, svg = makeSVG(host, W, H);
    var read = h('div', 'shapenote', null, host);
    var g = el('g', {}, svg), sing = false;

    function draw() {
      g.innerHTML = '';
      var A = sing ? [['2', '4'], ['1', '2']] : [['2', '1'], ['1', '1']];
      var B = sing ? [['?', '?'], ['?', '?']] : [['1', '−1'], ['−1', '2']];
      var I = sing ? [['—', '—'], ['—', '—']] : [['1', '0'], ['0', '1']];
      var pa = panel(g, A, { label: 'A', cell: 38 });
      var pb = panel(g, B, { label: sing ? 'A⁻¹ does not exist' : 'A⁻¹', cell: 38,
                             labelInk: sing ? 'pink' : 'amber' });
      var pi = panel(g, I, { label: sing ? 'nothing to multiply' : 'AA⁻¹  =  I', cell: 38,
                             labelInk: sing ? 'pink' : 'green',
                             hi: function (r, c) { return !sing && r === c ? 'sh-hi' : ''; } });
      lay([pa, sym(g, '×'), pb, sym(g, '='), pi], 310, 108, 16);
      read.innerHTML = '';
      if (sing) {
        worked(read, 'det(A)  =  2×2 − 4×1  =  0',
               'row 2 is row 1 halved — the second row says nothing the first did not');
        h('p', 'rd-a', 'This matrix throws information away: it maps the whole plane onto a ' +
                       'single line, and once two different inputs land on the same output ' +
                       'there is no undo. That is what singular means — not "hard to invert", ' +
                       'but "no inverse exists".', read);
        h('p', 'rd-b', 'Linear regression solves (XᵀX)⁻¹Xᵀy. Put a duplicated feature in X — ' +
                       'price in euros and price in cents, say — and XᵀX becomes exactly this: ' +
                       'singular, and the fit blows up. Ridge regression adds λI to the ' +
                       'diagonal, which nudges the matrix off singular and is the entire ' +
                       'mechanism behind "regularisation fixes collinearity".', read);
      } else {
        worked(read, 'AA⁻¹  =  I',
               'Row 1: 2×1 + 1×(−1) = 1; 2×(−1) + 1×2 = 0. ' +
               'Row 2: 1×1 + 1×(−1) = 0; 1×(−1) + 1×2 = 1.');
        h('p', 'rd-a', 'The identity is the do-nothing machine: multiply by it and every vector ' +
                       'comes back untouched. The inverse is whatever undoes A — apply A, then ' +
                       'A⁻¹, and you are exactly where you started.', read);
        h('p', 'rd-b', 'Press the button for a matrix that has no undo, and why that is the ' +
                       'thing that actually breaks in practice.', read);
      }
    }
    var bar = ctl(host);
    button(bar, 'show a singular matrix', function () { sing = !sing; draw(); });
    draw();
    cap(host, 'Nothing here needs solving. A and A⁻¹ are two grids of small numbers, and the ' +
              'claim is only that multiplying them gives the do-nothing grid — which you can ' +
              'check by hand in about fifteen seconds.');
  }

  function vizRank(host) {
    eqfig(host, 600, 230, function (svg, read) {
      var M = panel(svg, [['1', '0', '1'], ['0', '1', '1']],
                    { label: 'A', sub: 'shape [2, 3] — three columns', cell: 36,
                      hi: function (r, c) { return c === 2 ? 'sh-hi' : ''; } });
      M.move(40, 40);
      var box = { l: 300, t: 30, w: 170, h: 150 };
      var F = new Frame(svg, box, [-0.4, 1.8], [-0.4, 1.8]);
      el('line', { x1: F.X(-0.4), y1: F.Y(0), x2: F.X(1.8), y2: F.Y(0), class: 's-grid' }, svg);
      el('line', { x1: F.X(0), y1: F.Y(-0.4), x2: F.X(0), y2: F.Y(1.8), class: 's-grid' }, svg);
      arrow(svg, F.X(0), F.Y(0), F.X(1), F.Y(0), 'accent');
      arrow(svg, F.X(0), F.Y(0), F.X(0), F.Y(1), 'green');
      arrow(svg, F.X(0), F.Y(0), F.X(1), F.Y(1), 'amber');
      el('line', { x1: F.X(1), y1: F.Y(0), x2: F.X(1), y2: F.Y(1), stroke: css('--tx-faint'),
                   'stroke-width': 1, 'stroke-dasharray': '3 3' }, svg);
      el('line', { x1: F.X(0), y1: F.Y(1), x2: F.X(1), y2: F.Y(1), stroke: css('--tx-faint'),
                   'stroke-width': 1, 'stroke-dasharray': '3 3' }, svg);
      txt(svg, F.X(1) + 8, F.Y(0) + 4, 'c₁', 'accent', 11.5, 'start');
      txt(svg, F.X(0) - 8, F.Y(1), 'c₂', 'green', 11.5, 'end');
      txt(svg, F.X(1) + 8, F.Y(1), 'c₃', 'amber', 11.5, 'start');
      worked(read, 'c₃  =  c₁ + c₂', '[1, 1]  =  [1, 0] + [0, 1]   →   rank(A) = 2, not 3');
      h('p', 'rd-a', 'Three columns, but only two directions. The third arrow is the diagonal ' +
                     'of the other two — it adds no direction the first two could not already ' +
                     'reach, so it carries no new information.', read);
      h('p', 'rd-b', 'Rank is that count: how many genuinely independent directions are in ' +
                     'there. It can equal the number of columns. Compression becomes useful ' +
                     'when a matrix has low rank, or can be approximated well by one that does.', read);
    }, 'A model with rank-deficient features is not merely inefficient. Its normal equations ' +
       'are singular, which is the same failure the inverse card ends on.');
  }

  function vizEigen(host) {
    var W = 600, H = 250, svg = makeSVG(host, W, H);
    var box = { l: 70, t: 20, w: 190, h: 190 };
    var F = new Frame(svg, box, [-0.6, 3.4], [-0.6, 3.4]);
    var i;
    for (i = 0; i <= 3; i++) {
      el('line', { x1: F.X(i), y1: F.Y(-0.6), x2: F.X(i), y2: F.Y(3.4), class: 's-grid' }, svg);
      el('line', { x1: F.X(-0.6), y1: F.Y(i), x2: F.X(3.4), y2: F.Y(i), class: 's-grid' }, svg);
    }
    var V = [{ v: [1, 1], name: 'v = [1, 1]', eig: true, lam: 3 },
             { v: [1, -0.35], name: 'v = [1, −0.35]', eig: false, lam: 0 }];
    var k = 0, gA = el('g', {}, svg), pane = el('g', {}, svg);
    var read = h('div', 'shapenote', null, host);

    function draw() {
      var s = V[k], v = s.v, Av = [2 * v[0] + 1 * v[1], 1 * v[0] + 2 * v[1]];
      gA.innerHTML = ''; pane.innerHTML = '';
      arrow(gA, F.X(0), F.Y(0), F.X(Av[0]), F.Y(Av[1]), 'amber');
      arrow(gA, F.X(0), F.Y(0), F.X(v[0]), F.Y(v[1]), 'accent');
      txt(gA, F.X(v[0]) + 6, F.Y(v[1]) + 14, 'v', 'accent', 12, 'start');
      txt(gA, F.X(Av[0]) + 6, F.Y(Av[1]) - 4, 'Av', 'amber', 12, 'start');
      if (s.eig)
        el('line', { x1: F.X(0), y1: F.Y(0), x2: F.X(3.4), y2: F.Y(3.4),
                     stroke: css('--green'), 'stroke-width': 1,
                     'stroke-dasharray': '4 4' }, gA);
      var pa = panel(pane, [['2', '1'], ['1', '2']], { label: 'A', cell: 34 });
      var pv = panel(pane, [[String(v[0])], [String(v[1])]], { label: 'v', cell: 34 });
      var pr = panel(pane, [[Av[0].toFixed(2).replace(/\.00$/, '')],
                            [Av[1].toFixed(2).replace(/\.00$/, '')]],
                     { label: 'Av', cell: 34, labelInk: s.eig ? 'green' : 'pink' });
      lay([pa, sym(pane, '×'), pv, sym(pane, '='), pr], 430, 118, 12);
      read.innerHTML = '';
      if (s.eig) {
        worked(read, 'A v  =  λ v', '[2 1; 1 2] × [1, 1]  =  [3, 3]  =  3 × [1, 1]   →   λ = 3');
        h('p', 'rd-a', 'The output arrow lies exactly on top of the input arrow, three times ' +
                       'as long. A did not turn this vector at all — it only scaled it. That is ' +
                       'the entire definition of an eigenvector, and λ = 3 is its eigenvalue.', read);
        h('p', 'rd-b', 'This symmetric matrix has perpendicular eigenvector directions. PCA ' +
                       'uses that property of a covariance matrix — a table of how pairs of ' +
                       'features vary together. Not every real matrix has real eigenvectors: ' +
                       'a 90° rotation in a plane has none.', read);
      } else {
        worked(read, 'A v  =  ?', '[2 1; 1 2] × [1, −0.35]  =  [1.65, 0.30]   — a different direction');
        h('p', 'rd-a', 'This one gets turned. The output points somewhere the input did not, so ' +
                       'no single number λ can relate them: it is not an eigenvector.', read);
        h('p', 'rd-b', 'Check both components: 1.65/1 = 1.65, but 0.30/(−0.35) ≈ −0.86. ' +
                       'The multipliers disagree, so no single λ works for this vector.', read);
      }
    }
    var bar = ctl(host);
    button(bar, 'try another vector', function () { k = (k + 1) % V.length; draw(); });
    draw();
    cap(host, 'Press the button and watch only one thing: whether the amber arrow lands on the ' +
              'blue one’s line. That is the whole test.');
  }

  function vizDet(host) {
    var W = 600, H = 250, svg = makeSVG(host, W, H);
    var box = { l: 60, t: 20, w: 200, h: 200 };
    var F = new Frame(svg, box, [-0.5, 3.6], [-0.5, 3.6]);
    var i;
    for (i = 0; i <= 3; i++) {
      el('line', { x1: F.X(i), y1: F.Y(-0.5), x2: F.X(i), y2: F.Y(3.6), class: 's-grid' }, svg);
      el('line', { x1: F.X(-0.5), y1: F.Y(i), x2: F.X(3.6), y2: F.Y(i), class: 's-grid' }, svg);
    }
    var unit = el('polygon', { fill: css('--tx-faint'), opacity: 0.18,
                               stroke: css('--tx-faint'), 'stroke-width': 1 }, svg);
    unit.setAttribute('points', [[0, 0], [1, 0], [1, 1], [0, 1]]
      .map(function (p) { return F.X(p[0]) + ',' + F.Y(p[1]); }).join(' '));
    var shape = el('polygon', { fill: css('--amber'), opacity: 0.2,
                                stroke: css('--amber'), 'stroke-width': 1.8 }, svg);
    var pane = el('g', {}, svg), read = h('div', 'shapenote', null, host);
    var flat = false;

    function draw() {
      var A = flat ? [[2, 1], [2, 1]] : [[2, 1], [0, 3]];
      var d = A[0][0] * A[1][1] - A[0][1] * A[1][0];
      var corners = [[0, 0], [1, 0], [1, 1], [0, 1]].map(function (p) {
        return [A[0][0] * p[0] + A[0][1] * p[1], A[1][0] * p[0] + A[1][1] * p[1]];
      });
      shape.setAttribute('points', corners.map(function (p) {
        return F.X(p[0]) + ',' + F.Y(p[1]);
      }).join(' '));
      pane.innerHTML = '';
      var pa = panel(pane, A.map(function (r) { return r.map(String); }), { label: 'A', cell: 38 });
      pa.move(330, 40);
      txt(pane, 330, 150, 'unit square area', 'tx-dim', 12, 'start');
      txt(pane, 560, 150, '1', 'chalk', 15, 'end');
      txt(pane, 330, 182, 'after A', 'tx-dim', 12, 'start');
      txt(pane, 560, 184, String(d), d === 0 ? 'pink' : 'amber', 19, 'end');
      read.innerHTML = '';
      worked(read, 'det [a b; c d]  =  ad − bc',
             flat ? '2×1 − 1×2  =  0' : '2×3 − 1×0  =  6');
      h('p', 'rd-a', flat
        ? 'Area zero. The square has been flattened onto a line — two dimensions in, one ' +
          'dimension out — and area is exactly what was lost.'
        : 'The grey unit square becomes the amber parallelogram, and its area is 6. That is ' +
          'the determinant: the factor by which A scales area. Nothing else.', read);
      h('p', 'rd-b', flat
        ? 'A zero determinant and “no inverse” are the same fact said twice. You cannot undo a ' +
          'flattening, because everything on that line came from a whole family of inputs.'
        : 'The absolute determinant measures area scaling; its sign tells you whether ' +
          'orientation flips. A small determinant alone does not measure numerical stability: ' +
          'shrinking every direction equally can give a tiny determinant without making ' +
          'relative errors worse.', read);
    }
    var bar = ctl(host);
    button(bar, 'flatten it', function () { flat = !flat; draw(); });
    draw();
    cap(host, 'The sign matters too: a negative determinant means the shape was flipped over ' +
              'as well as scaled. Zero means it was crushed.');
  }

  function vizSVD(host) {
    var svg = makeSVG(host, 420, 375), stage = el('g', {}, svg);
    svg.setAttribute('role', 'img');
    svg.setAttribute('aria-label', 'Numeric singular value decomposition and its rank-one approximation');
    var read = h('div', 'shapenote', null, host), reduced = false;
    var U = [[0, -1], [1, 0]], VT = [[0.6, 0.8], [-0.8, 0.6]];
    function draw() {
      stage.innerHTML = ''; read.innerHTML = '';
      var s2 = reduced ? 0 : 1;
      var A = [[0.8 * s2, -0.6 * s2], [1.8, 2.4]];
      var pu = panel(stage, U, { label: 'U', cell: 32, fs: 14 });
      var ps = panel(stage, [[3, 0], [0, s2]], { label: reduced ? 'Σ₁' : 'Σ', cell: 32, fs: 14,
        hi: function (r, c) { return r === c ? 'sh-hi' : ''; } });
      var pv = panel(stage, VT, { label: 'Vᵀ', cell: 32, fs: 14 });
      lay([pu, sym(stage, '×'), ps, sym(stage, '×'), pv], 210, 68, 9);
      txt(stage, 210, 144, 'Multiply the factors →', 'tx-dim', 14);
      var pa = panel(stage, A, { label: reduced ? 'A₁ · approximation' : 'A · exact', cell: 36, fs: 15 });
      pa.move(158, 162);
      txt(stage, 210, 288, 'Follow x = [1, 2], right to left:', 'tx-dim', 14);
      txt(stage, 210, 317, 'Vᵀx = [2.2, 0.4]', 'accent', 16);
      txt(stage, 210, 342, 'ΣVᵀx = [6.6, ' + (0.4 * s2) + ']', 'amber', 16);
      txt(stage, 210, 367, 'UΣVᵀx = [' + (-0.4 * s2) + ', 6.6]', 'green', 16);
      worked(read, reduced ? 'A₁ = U diag(3, 0) Vᵀ ≈ A' : 'A = UΣVᵀ',
        reduced ? 'A₁ = [0, 0; 1.8, 2.4]. Squared entry error = 0.8² + (−0.6)² = 1.' :
        'UΣ = [0, −1; 3, 0]. Top-left entry of A: 0×0.6 + (−1)×(−0.8) = 0.8.');
      h('p', 'rd-a', reduced
        ? 'We replaced the smaller stretch, 1, with zero. The whole first row disappears, ' +
          'so one output direction is lost. The approximation is visibly different from A.'
        : 'Vᵀ turns the input coordinates; Σ stretches the first by 3 and the second by 1; ' +
          'U turns the result. The three little machines give exactly the same answer as A.', read);
      h('p', 'rd-b', 'The symbols σ₁ and σ₂ mean the two stretch amounts: here 3 and 1. ' +
        'Keeping only 3 gives a best rank-one approximation under squared entry error. ' +
        'Compression is accurate only when the discarded stretch amounts are small enough ' +
        'for your application; an SVD does not guarantee that they are.', read);
    }
    var toggle = button(ctl(host), 'keep only the largest stretch', function () {
      reduced = !reduced; draw();
      toggle.textContent = reduced ? 'restore both stretches' : 'keep only the largest stretch';
    });
    draw();
  }

  function vizNormActions(host) {
    var svg = makeSVG(host, 400, 230), drawing = el('g', {}, svg);
    svg.setAttribute('role', 'img');
    svg.setAttribute('aria-label', 'Compare gradient clipping, weight decay, and L1 versus L2 penalties');
    var read = h('div', 'shapenote', null, host), mode = 0;
    function draw() {
      drawing.innerHTML = ''; read.innerHTML = '';
      if (mode < 2) {
        var after = mode === 0 ? [1.2, 1.6] : [2.97, 3.96];
        var label = mode === 0 ? 'gradient g' : 'weights w';
        var a = panel(drawing, [[3], [4]], { label: label, cell: 38, fs: 16 });
        var b = panel(drawing, [[after[0]], [after[1]]], { label: 'after', cell: 38, fs: 16 });
        lay([a, sym(drawing, '→', { size: 24, w: 60 }), b], 200, 72, 22);
        txt(drawing, 200, 168, mode === 0 ? 'length 5 → length 2' : 'each weight keeps 99%', 'amber', 17);
        txt(drawing, 200, 202, mode === 0 ? 'Same direction. Shorter update signal.' : '3 × 0.99 = 2.97; 4 × 0.99 = 3.96', 'tx-dim', 14);
        worked(read, mode === 0 ? 'g′ = g × min(1, limit / ‖g‖₂)' : 'w′ = (1 − ηλ)w',
          mode === 0 ? '[3, 4] × (2/5) = [1.2, 1.6]; √(1.2² + 1.6²) = 2.' :
          'Illustrative ηλ = 0.01: [3, 4] × 0.99 = [2.97, 3.96].');
        h('p', 'rd-a', mode === 0
          ? 'A limit of 2 is smaller than this gradient’s length of 5. Scale the entire gradient ' +
            'by 0.4 before using it to update the weights. A gradient already shorter than 2 stays unchanged.'
          : 'η is the learning rate; λ sets the decay strength. This shows only the shrink part ' +
            'of an update. The optimizer also takes a step based on the data’s error.', read);
      } else {
        var rows = [['no penalty', 0.3, 'accent'], ['L2 penalty', 0.15, 'green'], ['L1 penalty', 0, 'pink']];
        rows.forEach(function (r, i) {
          var y = 40 + i * 64;
          txt(drawing, 20, y, r[0], 'tx-dim', 15, 'start');
          el('line', { x1: 150, x2: 340, y1: y - 5, y2: y - 5, class: 's-grid' }, drawing);
          if (r[1]) el('rect', { x: 150, y: y - 15, width: r[1] / 0.3 * 170, height: 20,
            fill: css('--' + r[2]), rx: 3 }, drawing);
          el('circle', { cx: 150 + r[1] / 0.3 * 170, cy: y - 5, r: 4, fill: css('--' + r[2]) }, drawing);
          txt(drawing, 380, y, String(r[1]), r[2], 17, 'end');
        });
        txt(drawing, 150, 220, '0', 'tx-dim', 14);
        txt(drawing, 320, 220, '0.3', 'tx-dim', 14);
        worked(read, 'Fit error = ½(w − 0.3)²; penalty strength = 0.5',
          'L1: max(0.3 − 0.5, 0) = 0. L2: 0.3/(1 + 2×0.5) = 0.15.');
        h('p', 'rd-a', 'The data alone prefers w = 0.3. Adding 0.5|w| makes the best weight ' +
          'exactly zero; adding 0.5w² leaves it at 0.15. These are the solved optima for this ' +
          'one-weight example, not a single gradient step.', read);
      }
    }
    var bar = ctl(host), choices = [];
    ['clip the gradient', 'shrink the weights', 'compare L1 and L2'].forEach(function (label, i) {
      choices.push(button(bar, label, function () {
        mode = i; draw();
        choices.forEach(function (b, j) { b.setAttribute('aria-pressed', String(i === j)); });
      }));
    });
    choices.forEach(function (b, i) { b.setAttribute('aria-pressed', String(i === mode)); });
    draw();
  }

  function vizOrtho(host) {
    eqfig(host, 600, 220, function (svg, read) {
      var box = { l: 70, t: 20, w: 175, h: 175 };
      var F = new Frame(svg, box, [-1.4, 1.4], [-1.4, 1.4]);
      el('line', { x1: F.X(-1.4), y1: F.Y(0), x2: F.X(1.4), y2: F.Y(0), class: 's-grid' }, svg);
      el('line', { x1: F.X(0), y1: F.Y(-1.4), x2: F.X(0), y2: F.Y(1.4), class: 's-grid' }, svg);
      arrow(svg, F.X(0), F.Y(0), F.X(0.6), F.Y(0.8), 'accent');
      arrow(svg, F.X(0), F.Y(0), F.X(-0.8), F.Y(0.6), 'green');
      var m = 16;
      el('path', { d: 'M' + F.X(0.6 * 0.22) + ' ' + F.Y(0.8 * 0.22) +
                      'L' + (F.X(0.6 * 0.22) - m * 0.8) + ' ' + (F.Y(0.8 * 0.22) - m * 0.6) +
                      'L' + (F.X(0) - m * 0.8) + ' ' + (F.Y(0) - m * 0.6),
                   fill: 'none', stroke: css('--tx-faint'), 'stroke-width': 1.2 }, svg);
      txt(svg, F.X(0.6) + 8, F.Y(0.8), 'u = [0.6, 0.8]', 'accent', 11, 'start');
      txt(svg, F.X(-0.8) - 8, F.Y(0.6), 'w = [−0.8, 0.6]', 'green', 11, 'end');
      var t = panel(svg, [['1', '0'], ['0', '1']],
                    { label: 'QᵀQ  =  I', labelInk: 'green', cell: 34,
                      hi: function (r, c) { return r === c ? 'sh-hi' : ''; } });
      t.move(430, 62);
      worked(read, 'u · w  =  0', '0.6×(−0.8) + 0.8×0.6  =  −0.48 + 0.48  =  0');
      h('p', 'rd-a', 'Perpendicular is not a picture here, it is an arithmetic fact: the dot ' +
                     'product comes out exactly zero. Two orthogonal directions share nothing — ' +
                     'moving along one changes nothing about your position along the other.', read);
      h('p', 'rd-b', 'Collect vectors that are all mutually orthogonal and all of length 1 and ' +
                     'you have an orthonormal basis, whose matrix Q satisfies QᵀQ = I. Its ' +
                     'inverse is its transpose — free, exact, and numerically perfect. This is ' +
                     'why PCA components are built orthogonal, and why rotation matrices are ' +
                     'the safest thing in numerical linear algebra.', read);
    }, 'Orthogonal directions are independent by construction, which is what makes a set of ' +
       'them a coordinate system worth using.');
  }

  /* ---------------- module 11 · one figure per term card ---------------- */

  function vizFnMachine(host) {
    eqfig(host, 620, 220, function (svg, read) {
      function f(x) { return x * x - 4 * x + 7; }
      var ins = [1, 2, 5];
      el('rect', { x: 150, y: 62, width: 190, height: 60, rx: 10,
                   fill: css('--panel-2'), stroke: css('--amber'), 'stroke-width': 1.4 }, svg);
      txt(svg, 245, 88, 'f(w)  =  w² − 4w + 7', 'amber', 14);
      txt(svg, 245, 108, 'the machine', 'tx-faint', 10.5);
      ins.forEach(function (v, i) {
        var y = 60 + i * 32;
        txt(svg, 40, y + 4, 'w = ' + v, 'accent', 12, 'start');
        arrow(svg, 100, y, 145, 92, 'accent');
        arrow(svg, 345, 92, 392, y, 'green');
        txt(svg, 400, y + 4, 'f = ' + f(v), 'green', 12, 'start');
      });
      var box = { l: 480, t: 30, w: 120, h: 130 };
      var F = new Frame(svg, box, [-0.5, 5.5], [0, 13]);
      F.axes('w', 'f');
      F.path(f, 120);
      ins.forEach(function (v) {
        el('circle', { cx: F.X(v), cy: F.Y(f(v)), r: 3.5, fill: css('--green') }, svg);
      });
      worked(read, 'f(w)  =  w² − 4w + 7', 'f(2)  =  4 − 8 + 7  =  3');
      h('p', 'rd-a', 'One number in, one number out, and the same input always gives the same ' +
                     'output. That is all a function is. The plot on the right is the same three ' +
                     'answers drawn instead of listed.', read);
      h('p', 'rd-b', 'A loss function is this machine with a few billion input dials instead of ' +
                     'one, and still exactly one number coming out. Everything in training is an ' +
                     'attempt to find the dial settings that make that one number small.', read);
    }, 'The machine is fixed; only what you feed it changes. When training changes the weights it ' +
       'is changing the input, not the machine.');
  }

  function vizDerivMini(host) {
    eqfig(host, 600, 240, function (svg, read) {
      var box = { l: 60, t: 20, w: 250, h: 180 };
      var F = new Frame(svg, box, [0.2, 2.4], [0, 5]);
      F.axes('x', 'f');
      F.path(function (x) { return x * x; }, 160);
      var HS = [1, 0.5, 0.1], cols = ['tx-faint', 'accent', 'green'];
      HS.forEach(function (hh, i) {
        var x2 = 1 + hh, s = (x2 * x2 - 1) / hh;
        el('line', { x1: F.X(1), y1: F.Y(1), x2: F.X(x2), y2: F.Y(x2 * x2),
                     stroke: css('--' + cols[i]), 'stroke-width': 1.3 }, svg);
        el('circle', { cx: F.X(x2), cy: F.Y(x2 * x2), r: 3, fill: css('--' + cols[i]) }, svg);
        txt(svg, 340, 52 + i * 30, 'h = ' + hh, 'tx-dim', 12, 'start');
        txt(svg, 452, 52 + i * 30, 'slope ' + s.toFixed(1), cols[i], 13, 'end');
      });
      el('line', { x1: F.X(0.45), y1: F.Y(1 - 2 * 0.55), x2: F.X(1.75), y2: F.Y(1 + 2 * 0.75),
                   stroke: css('--amber'), 'stroke-width': 2 }, svg);
      el('circle', { cx: F.X(1), cy: F.Y(1), r: 4, fill: css('--amber') }, svg);
      txt(svg, 340, 148, 'h → 0', 'amber', 12, 'start');
      txt(svg, 452, 148, 'slope 2.0', 'amber', 15, 'end');
      txt(svg, 396, 176, 'the derivative at x = 1', 'tx-faint', 11);
      worked(read, 'f′(x)  =  lim (f(x+h) − f(x)) / h    as h → 0',
             '((1+h)² − 1) / h  =  (2h + h²) / h  =  2 + h   →   2');
      h('p', 'rd-a', 'Three grey-to-green lines, each cutting the curve at two points, each with ' +
                     'an honest rise-over-run slope you could measure with a ruler: 3.0, then ' +
                     '2.5, then 2.1.', read);
      h('p', 'rd-b', 'Shrink the gap and the numbers walk towards 2 and stop there. The amber ' +
                     'line is the limit — it touches at one point only, and 2 is the derivative. ' +
                     'The algebra above says the same thing in one line: the answer is 2 + h, so ' +
                     'the answer at h = 0 is 2.', read);
    }, 'This is the only genuinely new idea in calculus. Everything after it is rules for ' +
       'computing this limit without having to take it.');
  }

  function vizPartialMini(host) {
    eqfig(host, 620, 235, function (svg, read) {
      function panelPlot(l, ttl, fn, at, slope, colour) {
        var box = { l: l, t: 40, w: 200, h: 140 };
        var F = new Frame(svg, box, [0, 3], [0, 12]);
        F.axes();
        F.path(fn, 120);
        var y0 = fn(at), d = 0.9;
        el('line', { x1: F.X(at - d), y1: F.Y(y0 - slope * d),
                     x2: F.X(at + d), y2: F.Y(y0 + slope * d),
                     stroke: css('--' + colour), 'stroke-width': 2 }, svg);
        el('circle', { cx: F.X(at), cy: F.Y(y0), r: 4, fill: css('--' + colour) }, svg);
        txt(svg, l + 100, 26, ttl, colour, 12.5);
        txt(svg, l + 100, 202, 'slope here = ' + slope, colour, 13);
      }
      panelPlot(50, 'freeze y = 1, move x', function (x) { return x * x + 3; }, 2, 4, 'accent');
      panelPlot(340, 'freeze x = 2, move y', function (y) { return 4 + 3 * y * y; }, 1, 6, 'green');
      worked(read, 'f(x, y)  =  x² + 3y²      ∂f/∂x  =  2x      ∂f/∂y  =  6y',
             'at (2, 1):   ∂f/∂x = 4,   ∂f/∂y = 6   →   ∇f = [4, 6]');
      h('p', 'rd-a', 'A partial derivative is an ordinary derivative with a promise attached: ' +
                     'every other input is nailed down. Freeze y and the two-input surface ' +
                     'becomes the single curve on the left, and the slope on it is 4.', read);
      h('p', 'rd-b', 'Freeze x instead and you get a different curve and a different slope, 6. ' +
                     'Neither number is "the" slope of f — each is the slope along one axis. ' +
                     'Collect all of them into a vector and that is the gradient, the next card.', read);
    }, 'Seven billion parameters means seven billion of these, each computed with all the others ' +
       'held still. Backpropagation is the trick that gets all of them in one pass.');
  }

  function vizStationary(host) {
    eqfig(host, 620, 235, function (svg, read) {
      var box = { l: 55, t: 22, w: 270, h: 170 };
      var F = new Frame(svg, box, [-2.3, 2.3], [-3.2, 3.2]);
      F.axes('x', 'f');
      el('line', { x1: box.l, y1: F.Y(0), x2: box.l + box.w, y2: F.Y(0), class: 's-grid' }, svg);
      F.path(function (x) { return x * x * x - 3 * x; }, 200);
      [[-1, 2, 'a maximum', 'pink'], [1, -2, 'a minimum', 'green']].forEach(function (p) {
        el('line', { x1: F.X(p[0] - 0.55), y1: F.Y(p[1]), x2: F.X(p[0] + 0.55), y2: F.Y(p[1]),
                     stroke: css('--amber'), 'stroke-width': 2 }, svg);
        el('circle', { cx: F.X(p[0]), cy: F.Y(p[1]), r: 4, fill: css('--' + p[3]) }, svg);
        txt(svg, F.X(p[0]), F.Y(p[1]) + (p[1] > 0 ? -14 : 22), p[2], p[3], 11.5);
      });
      txt(svg, box.l + box.w / 2, 212, 'f(x) = x³ − 3x', 'tx-faint', 11);

      var b2 = { l: 400, t: 22, w: 170, h: 170 };
      var G = new Frame(svg, b2, [-1.6, 1.6], [-3.2, 3.2]);
      G.axes('x', 'f');
      G.path(function (x) { return x * x * x; }, 160);
      el('line', { x1: G.X(-0.5), y1: G.Y(0), x2: G.X(0.5), y2: G.Y(0),
                   stroke: css('--amber'), 'stroke-width': 2 }, svg);
      el('circle', { cx: G.X(0), cy: G.Y(0), r: 4, fill: css('--tx-dim') }, svg);
      txt(svg, b2.l + b2.w / 2, G.Y(0) - 16, 'neither', 'tx-dim', 11.5);
      txt(svg, b2.l + b2.w / 2, 212, 'f(x) = x³', 'tx-faint', 11);
      worked(read, 'f′(x)  =  0', 'x³ − 3x:  f′ = 3x² − 3 = 0  at  x = −1 and x = +1.    ' +
                                  'x³:  f′ = 3x² = 0  at  x = 0.');
      h('p', 'rd-a', 'Three flat spots, three different things. On the left, the same equation ' +
                     'f′ = 0 produces a hilltop and a valley floor. On the right it produces a ' +
                     'moment of flatness in a curve that never stops climbing.', read);
      h('p', 'rd-b', 'So "the gradient reached zero" answers only one question — has it stopped ' +
                     'moving — and not the one you care about, which is whether it stopped ' +
                     'somewhere good. The second derivative, three cards down, is what ' +
                     'separates the three cases.', read);
    }, 'When a training run flatlines, this is the ambiguity you are looking at. A small gradient ' +
       'is equally consistent with a good minimum, a bad one, and a saddle you will eventually ' +
       'roll off.');
  }

  function vizMinimumCard(host) {
    eqfig(host, 600, 235, function (svg, read) {
      function f(x) { return 0.35 * Math.pow(x, 4) - 1.1 * x * x - 0.42 * x + 2.2; }
      var box = { l: 58, t: 20, w: 400, h: 165 };
      var F = new Frame(svg, box, [-2.1, 2.1], [0, 4]);
      F.axes('parameter', 'loss');
      F.path(f, 240);
      /* find the two local minima numerically rather than hard-coding
         them, so the labels can never drift away from the curve */
      var xs = [], x, i;
      for (i = 0; i <= 4200; i++) {
        x = -2.1 + i * 4.2 / 4200;
        if (i > 0 && i < 4200) {
          var a = f(x - 0.001), b = f(x), c = f(x + 0.001);
          if (b < a && b < c) xs.push([x, b]);
        }
      }
      xs.sort(function (p, q) { return p[1] - q[1]; });
      var best = xs[0], other = xs[1] || xs[0];
      [[other, 'a local minimum', 'pink'], [best, 'the global minimum', 'green']]
        .forEach(function (p) {
          el('circle', { cx: F.X(p[0][0]), cy: F.Y(p[0][1]), r: 5, fill: css('--' + p[2]) }, svg);
          txt(svg, F.X(p[0][0]), F.Y(p[0][1]) - 16, p[1], p[2], 11.5);
          txt(svg, F.X(p[0][0]), F.Y(p[0][1]) + 20, 'loss ' + p[0][1].toFixed(2), p[2], 12);
        });
      worked(read, 'f′ = 0  and  f″ > 0',
             'both dips satisfy it — the flat, upward-curving test cannot tell them apart');
      h('p', 'rd-a', 'Both marked points pass every local test: the ground is flat and it curves ' +
                     'up in every direction. A ball resting in either one has no reason to move, ' +
                     'because nothing nearby is lower.', read);
      h('p', 'rd-b', 'Only one of them is the lowest point on the whole curve, and no amount of ' +
                     'looking around locally reveals that. This is why deep learning has seeds, ' +
                     'restarts and warmups: gradient descent can only ever see the ground under ' +
                     'its own feet.', read);
    }, 'In practice the consolation is that in very high dimensions most local minima turn out to ' +
       'sit at almost the same loss — the disaster case is rarer than the picture suggests.');
  }

  function vizSecondDeriv(host) {
    eqfig(host, 620, 220, function (svg, read) {
      var C = [[function (x) { return x * x; }, 'f = x²', 'f″ = +2', 'curves up · bowl', 'green'],
               [function (x) { return -x * x; }, 'f = −x²', 'f″ = −2', 'curves down · dome', 'pink'],
               [function (x) { return x * x * x; }, 'f = x³', 'f″ = 0', 'no bend at 0', 'tx-dim']];
      C.forEach(function (c, i) {
        var box = { l: 46 + i * 195, t: 34, w: 150, h: 120 };
        var F = new Frame(svg, box, [-1.4, 1.4], [-1.6, 1.6]);
        el('line', { x1: box.l, y1: F.Y(0), x2: box.l + box.w, y2: F.Y(0), class: 's-grid' }, svg);
        el('line', { x1: F.X(0), y1: box.t, x2: F.X(0), y2: box.t + box.h, class: 's-grid' }, svg);
        F.path(c[0], 140);
        el('line', { x1: F.X(-0.6), y1: F.Y(0), x2: F.X(0.6), y2: F.Y(0),
                     stroke: css('--amber'), 'stroke-width': 1.6 }, svg);
        el('circle', { cx: F.X(0), cy: F.Y(0), r: 3.5, fill: css('--amber') }, svg);
        txt(svg, box.l + box.w / 2, 22, c[1], 'tx-faint', 11.5);
        txt(svg, box.l + box.w / 2, 176, c[2], c[4], 14);
        txt(svg, box.l + box.w / 2, 194, c[3], 'tx-faint', 10.5);
      });
      worked(read, 'f″  =  the derivative of f′',
             'f = x²  →  f′ = 2x  →  f″ = 2.    Differentiate twice, nothing more.');
      h('p', 'rd-a', 'All three curves are flat at the amber point — identical first derivative, ' +
                     'zero. What separates them is which way the curve bends away from that ' +
                     'flatness, and that is exactly what the second derivative measures.', read);
      h('p', 'rd-b', 'Positive means the ground rises on both sides, so you are in a bowl and the ' +
                     'flat point is a minimum. Negative means a dome. Zero leaves the question ' +
                     'open, which is the third picture.', read);
    }, 'Curvature is also what sets a safe step size: the sharper the bowl, the smaller the step ' +
       'you can take before overshooting. That is the link to learning rates in module 14.');
  }

  function vizHessian(host) {
    eqfig(host, 620, 230, function (svg, read) {
      var P = panel(svg, [['2', '0'], ['0', '18']],
                    { label: 'H', sub: 'shape [2, 2] — two inputs', cell: 40,
                      hi: function (r, c) { return r === c ? 'sh-hi' : ''; } });
      P.move(46, 44);
      var box = { l: 300, t: 34, w: 260, h: 150 };
      var F = new Frame(svg, box, [-3.2, 3.2], [-1.1, 1.1]);
      var k;
      for (k = 1; k <= 4; k++) {
        var lv = k * 0.5;
        el('ellipse', { cx: F.X(0), cy: F.Y(0),
                        rx: Math.abs(F.X(lv) - F.X(0)),
                        ry: Math.abs(F.Y(lv / 3) - F.Y(0)),
                        fill: 'none', stroke: css('--rule-line'), 'stroke-width': 1 }, svg);
      }
      el('circle', { cx: F.X(0), cy: F.Y(0), r: 3.5, fill: css('--amber') }, svg);
      arrow(svg, F.X(0), F.Y(0), F.X(2.4), F.Y(0), 'green');
      arrow(svg, F.X(0), F.Y(0), F.X(0), F.Y(0.8), 'pink');
      txt(svg, F.X(2.4) + 6, F.Y(0) + 4, 'λ = 2 · gentle', 'green', 11, 'start');
      txt(svg, F.X(0) + 8, F.Y(0.8) + 12, 'λ = 18 · steep', 'pink', 11, 'start');
      txt(svg, box.l + box.w / 2, 208, 'the same bowl, seen from above', 'tx-faint', 11);
      worked(read, 'condition number  =  λmax / λmin',
             '18 / 2  =  9 — this bowl is nine times steeper across than along');
      h('p', 'rd-a', 'The Hessian is a small square grid: one row and one column per input, ' +
                     'holding the curvature in every direction at once. Here the two diagonal ' +
                     'entries say the bowl bends gently along one axis and hard along the other.', read);
      h('p', 'rd-b', 'That ratio is the whole story. A learning rate small enough to be safe on ' +
                     'the steep axis is far too small on the gentle one, so progress crawls — ' +
                     'the ravine you see in module 14. Momentum and Adam are both attempts to ' +
                     'fix this without ever building H, which for a real model would have ' +
                     'billions of rows.', read);
    }, 'Its eigenvalues also classify the flat point: all positive is a minimum, all negative a ' +
       'maximum, mixed signs a saddle. That is the second-derivative test, in n dimensions.');
  }

  function vizJacobian(host) {
    eqfig(host, 620, 230, function (svg, read) {
      txt(svg, 300, 24, 'f(x, y)  =  ( x²y ,  x + y ,  3y )        at  (x, y) = (2, 1)',
          'tx-dim', 12.5);
      var s = panel(svg, [['2xy', 'x²'], ['1', '1'], ['0', '3']],
                    { label: 'J', sub: 'the rule', cell: 40, fs: 12 });
      var n = panel(svg, [['4', '4'], ['1', '1'], ['0', '3']],
                    { label: 'J at (2, 1)', sub: 'shape [3, 2]', cell: 40, labelInk: 'green' });
      lay([s, sym(svg, '→'), n], 300, 128, 34);
      txt(svg, 92, 128, '3 outputs', 'tx-faint', 10.5, 'end', -90);
      txt(svg, 508, 128, '2 inputs', 'tx-faint', 10.5, 'start');
      worked(read, 'J[i][j]  =  ∂(output i) / ∂(input j)',
             'row 1 is the gradient of x²y:  ∂/∂x = 2xy = 4,  ∂/∂y = x² = 4');
      h('p', 'rd-a', 'One row per output, one column per input. Each row on its own is just a ' +
                     'gradient — the Jacobian is what you get when the function returns several ' +
                     'numbers and each of them has its own gradient.', read);
      h('p', 'rd-b', 'Its shape is [outputs, inputs], which is why a loss — one output — has a ' +
                     'Jacobian with a single row, and we call that row the gradient. Every layer ' +
                     'in a network has one of these; backprop multiplies them together without ' +
                     'ever storing one.', read);
    }, 'A layer mapping 4096 numbers to 4096 numbers has a 4096×4096 Jacobian — 16 million ' +
       'entries for one layer. Backprop computes vector–Jacobian products instead, which is why ' +
       'a backward pass costs about the same as a forward one.');
  }

  function vizChainMini(host) {
    eqfig(host, 620, 200, function (svg, read) {
      var N = [['x', 'x = 2', 'accent'], ['u = 3x', 'u = 6', 'amber'], ['y = u²', 'y = 36', 'green']];
      N.forEach(function (nd, i) {
        var cx = 110 + i * 200;
        el('rect', { x: cx - 62, y: 52, width: 124, height: 54, rx: 10,
                     fill: css('--panel-2'), stroke: css('--' + nd[2]), 'stroke-width': 1.3 }, svg);
        txt(svg, cx, 74, nd[0], nd[2], 13.5);
        txt(svg, cx, 94, nd[1], 'tx-faint', 11.5);
        if (i) {
          arrow(svg, cx - 200 + 64, 79, cx - 64, 79, 'tx-dim');
          txt(svg, cx - 100, 42, i === 1 ? 'du/dx = 3' : 'dy/du = 2u = 12', 'tx-dim', 11.5);
        }
      });
      arrow(svg, 110, 122, 510, 122, 'amber');
      txt(svg, 310, 148, 'dy/dx  =  12 × 3  =  36', 'amber', 15);
      worked(read, 'dy/dx  =  (dy/du) × (du/dx)',
             '12 × 3 = 36 — nudge x by 0.01 and y moves by about 0.36');
      h('p', 'rd-a', 'Two links in a chain, each with its own exchange rate. Moving x by one unit ' +
                     'moves u by three; moving u by one unit moves y by twelve. So moving x by ' +
                     'one moves y by thirty-six. You multiply the rates — nothing subtler than ' +
                     'currency conversion.', read);
      h('p', 'rd-b', 'A network is this chain with fifty links instead of two. Backpropagation ' +
                     'walks it right to left, carrying the running product, which is why one ' +
                     'backward pass gets the gradient for every weight at once. It is also why ' +
                     'fifty numbers smaller than one multiply down to nothing — the vanishing ' +
                     'gradient.', read);
    }, 'Check it by hand if you like: y = (3x)² = 9x², so dy/dx = 18x, which at x = 2 is 36. The ' +
       'chain rule got there without expanding anything.');
  }

  function vizConvex(host) {
    eqfig(host, 620, 225, function (svg, read) {
      function draw(l, fn, xa, xb, ok, ttl) {
        var box = { l: l, t: 34, w: 230, h: 140 };
        var F = new Frame(svg, box, [-1.9, 1.9], [-0.4, 4.4]);
        F.axes();
        F.path(fn, 200);
        el('line', { x1: F.X(xa), y1: F.Y(fn(xa)), x2: F.X(xb), y2: F.Y(fn(xb)),
                     stroke: css(ok ? '--green' : '--pink'), 'stroke-width': 1.8,
                     'stroke-dasharray': '5 3' }, svg);
        [xa, xb].forEach(function (x) {
          el('circle', { cx: F.X(x), cy: F.Y(fn(x)), r: 3.5,
                         fill: css(ok ? '--green' : '--pink') }, svg);
        });
        txt(svg, box.l + box.w / 2, 22, ttl, 'tx-faint', 11.5);
        txt(svg, box.l + box.w / 2, 196, ok ? 'chord stays above — convex'
                                            : 'chord dips below — not convex',
            ok ? 'green' : 'pink', 12);
      }
      draw(50, function (x) { return 0.7 * x * x + 0.3; }, -1.5, 1.5, true, 'a bowl');
      draw(330, function (x) { return 0.55 * Math.pow(x, 4) - 1.6 * x * x + 2.2; },
           -1.2, 1.2, false, 'an egg carton');
      worked(read, 'f(½a + ½b)  ≤  ½f(a) + ½f(b)',
             'left:  f(0) = 0.30  ≤  the chord’s 1.88.    ' +
             'right:  f(0) = 2.20  >  the chord’s 1.04 — the test fails');
      h('p', 'rd-a', 'The test is mechanical and needs no calculus: pick any two points on the ' +
                     'curve, draw the straight line between them, and check whether the curve ' +
                     'ever pokes above it. On the left it never does.', read);
      h('p', 'rd-b', 'That single property buys you everything. A convex function has one ' +
                     'minimum, every local minimum is that minimum, and downhill always works. ' +
                     'Linear and logistic regression and SVMs live on the left. Neural networks ' +
                     'live on the right, which is why they need seeds, schedules and luck.', read);
    }, 'Note what convexity is not: it is not "smooth" and not "simple". Absolute value is convex ' +
       'and has a kink; a gentle wave is smooth and is not convex.');
  }

  function vizDiffable(host) {
    eqfig(host, 600, 225, function (svg, read) {
      function draw(l, fn, ttl, kink) {
        var box = { l: l, t: 34, w: 200, h: 130 };
        var F = new Frame(svg, box, [-1.6, 1.6], [-0.25, 1.7]);
        el('line', { x1: box.l, y1: F.Y(0), x2: box.l + box.w, y2: F.Y(0), class: 's-grid' }, svg);
        el('line', { x1: F.X(0), y1: box.t, x2: F.X(0), y2: box.t + box.h, class: 's-grid' }, svg);
        F.path(fn, 200);
        el('circle', { cx: F.X(0), cy: F.Y(0), r: 6, fill: 'none',
                       stroke: css('--pink'), 'stroke-width': 1.6 }, svg);
        txt(svg, box.l + box.w / 2, 22, ttl, 'tx-faint', 11.5);
        txt(svg, box.l + box.w / 2, 190, kink, 'pink', 11.5);
      }
      draw(50, function (x) { return Math.abs(x); }, 'f = |x|', 'slope −1 on the left, +1 on the right');
      draw(330, function (x) { return Math.max(0, x); }, 'ReLU', 'slope 0 on the left, +1 on the right');
      txt(svg, 150, 96, '−1', 'accent', 12, 'end');
      txt(svg, 250, 96, '+1', 'green', 12, 'start');
      worked(read, 'f′(0)  =  ?', 'from the left: −1.   from the right: +1.   two answers, so no ' +
                                  'derivative exists at that one point');
      h('p', 'rd-a', 'Everywhere except the circled point these functions have a perfectly good ' +
                     'slope. At the kink the answer depends on which side you approach from, and ' +
                     '"the slope" stops meaning anything.', read);
      h('p', 'rd-b', 'In practice nobody cares: the framework returns 0 for ReLU at exactly zero ' +
                     'and moves on, because a float landing exactly on the kink essentially ' +
                     'never happens. What does matter is the genuinely flat case — argmax, ' +
                     'sampling, rounding — where the slope is zero everywhere and carries no ' +
                     'information at all. That is why Gumbel-softmax and straight-through ' +
                     'estimators exist.', read);
    }, 'The question to ask of any new layer is not “is it smooth?” but “does its gradient tell ' +
       'the weights anything?”. A hard threshold is differentiable almost everywhere and still ' +
       'useless, because that derivative is zero.');
  }

  /* =================================================================
     THE HOUSE — the single worked example the whole of module 10 is
     built on. Four real houses, four features, one price model. No
     other numbers are ever introduced: every figure below reaches
     into this block, so a value the reader met in section 10.3 is
     literally the same value in 10.10.

     The live house is what the sliders at the top of the page drive.
     Figures subscribe with House.on(node, redraw) and are pruned
     automatically once their node leaves the document.
     ================================================================= */
  var H_NOW = 2026;
  var H_W = [3.0, 8, -0.4, -12];       /* k€ per m², bedroom, year of age, km */
  var H_F = ['area', 'bedrooms', 'age', 'km to centre'];
    var H_FS = ['area', 'beds', 'age', 'km'];   /* when the cell is too narrow for the full name */
  var H_U = ['m²', 'rooms', 'years', 'km'];
  var H_SET = [
    { id: 'A', area: 120, beds: 3, year: 1985, km: 4.2 },
    { id: 'B', area: 60,  beds: 2, year: 1975, km: 6.0 },
    { id: 'C', area: 200, beds: 5, year: 2010, km: 2.5 },
    { id: 'D', area: 90,  beds: 3, year: 1998, km: 5.1 }
  ];
  function hVec(h) { return [h.area, h.beds, H_NOW - h.year, h.km]; }
  function hPrice(h) {
    var v = hVec(h), s = 0, i;
    for (i = 0; i < 4; i++) s += H_W[i] * v[i];
    return s;
  }
  function hFmt(x, d) {
    var n = Number(x).toFixed(d === undefined ? 1 : d);
    return n.replace(/\.0$/, '').replace('-', '−');
  }

  var H_LIVE = { id: 'yours', area: 120, beds: 3, year: 1985, km: 4.2 };
  var H_SUBS = [];
  var House = {
    live: function () { return H_LIVE; },
    set: function (k, v) { H_LIVE[k] = v; House.emit(); },
    on: function (node, fn) { H_SUBS.push({ node: node, fn: fn }); fn(H_LIVE); },
    emit: function () {
      H_SUBS = H_SUBS.filter(function (s) { return s.node.isConnected; });
      H_SUBS.forEach(function (s) { try { s.fn(H_LIVE); } catch (e) { /* one bad figure must not stop the rest */ } });
    }
  };

  /* the compact side controller: drag the grip; click the house to edit */
  function vizHousePanel(host) {
    host.classList.add('house-panel');
    host.dataset.expanded = 'false';
    host.dataset.side = 'right';
    var wrap = h('div', 'hp', null, host);
    var head = h('div', 'hp-head', null, wrap);
    var drag = h('button', 'hp-drag', null, head);
    drag.type = 'button';
    drag.setAttribute('aria-label', 'Move house controller');
    drag.title = 'Drag to move';
    var grip = el('svg', { viewBox: '0 0 12 20', class: 'hp-grip', 'aria-hidden': 'true' }, drag);
    [4, 10, 16].forEach(function (cy) {
      el('circle', { cx: '3', cy: cy, r: '1.25', fill: 'currentColor' }, grip);
      el('circle', { cx: '9', cy: cy, r: '1.25', fill: 'currentColor' }, grip);
    });
    var toggle = h('button', 'hp-toggle', null, head);
    toggle.type = 'button';
    toggle.setAttribute('aria-expanded', 'false');
    toggle.setAttribute('aria-controls', 'house-editor');
    var icon = el('svg', { viewBox: '0 0 24 24', class: 'hp-icon', 'aria-hidden': 'true' }, toggle);
    el('path', { d: 'M3.5 10.5 12 3.8l8.5 6.7v9.2h-6v-5.6h-5v5.6h-6z', fill: 'none',
                 stroke: 'currentColor', 'stroke-width': '1.6', 'stroke-linejoin': 'round' }, icon);
    h('span', 'hp-title', 'house', toggle);
    var out = h('span', 'hp-price', '', toggle);
    var chevron = el('svg', { viewBox: '0 0 16 16', class: 'hp-chevron', 'aria-hidden': 'true' }, toggle);
    el('path', { d: 'm4.5 6 3.5 3.5L11.5 6', fill: 'none', stroke: 'currentColor',
                 'stroke-width': '1.6', 'stroke-linecap': 'round', 'stroke-linejoin': 'round' }, chevron);

    var editor = h('div', 'hp-editor', null, wrap);
    editor.id = 'house-editor';
    editor.setAttribute('aria-hidden', 'true');

    var ROWS = [['area', 'area', 30, 300, 5, 'm²'],
                ['beds', 'bedrooms', 1, 8, 1, 'rooms'],
                ['year', 'year built', 1900, 2026, 1, ''],
                ['km', 'distance to centre', 0.5, 20, 0.1, 'km']];
    var vals = {};
    ROWS.forEach(function (r) {
      var row = h('label', 'hp-row', null, editor);
      h('span', 'hp-k', r[1], row);
      var inp = document.createElement('input');
      inp.type = 'range'; inp.min = r[2]; inp.max = r[3]; inp.step = r[4];
      inp.value = H_LIVE[r[0]];
      inp.setAttribute('aria-label', r[1]);
      row.appendChild(inp);
      vals[r[0]] = h('span', 'hp-v', '', row);
      inp.addEventListener('input', function () { House.set(r[0], parseFloat(inp.value)); });
    });

    var vecline = h('div', 'hp-vec', null, editor);
    h('p', 'hp-hint', 'Drives 10.1 to 10.6 and the eigenvector figure in 10.9. ' +
                      'From 10.7 the page works on the four sold houses, which stay fixed.', editor);

    function expand(next, returnFocus) {
      host.dataset.expanded = next ? 'true' : 'false';
      toggle.setAttribute('aria-expanded', next ? 'true' : 'false');
      editor.setAttribute('aria-hidden', next ? 'false' : 'true');
      requestAnimationFrame(constrainPosition);
      if (!next && returnFocus) toggle.focus();
    }
    toggle.addEventListener('click', function () {
      expand(host.dataset.expanded !== 'true', false);
    });
    function closeOutside(event) {
      if (!host.isConnected) {
        document.removeEventListener('pointerdown', closeOutside);
        return;
      }
      if (host.dataset.expanded === 'true' && !host.contains(event.target)) expand(false, false);
    }
    function closeWithEscape(event) {
      if (!host.isConnected) {
        document.removeEventListener('keydown', closeWithEscape);
        return;
      }
      if (event.key === 'Escape' && host.dataset.expanded === 'true') expand(false, true);
    }
    document.addEventListener('pointerdown', closeOutside);
    document.addEventListener('keydown', closeWithEscape);

    var POSITION_KEY = 'mlip-house-controller-position';
    var moving = null;
    function limits() {
      var compact = window.innerWidth <= 640;
      var top = compact ? 12 : 72;
      var bottom = compact ? 88 : 16;
      return {
        edge: compact ? 12 : 16,
        top: top,
        maxTop: Math.max(top, window.innerHeight - host.offsetHeight - bottom)
      };
    }
    function placeAtSide(side, top) {
      var lim = limits();
      var y = Math.max(lim.top, Math.min(lim.maxTop, top));
      host.dataset.side = side;
      host.style.top = y + 'px';
      host.style.bottom = 'auto';
      if (side === 'left') {
        host.style.left = lim.edge + 'px';
        host.style.right = 'auto';
      } else {
        host.style.left = 'auto';
        host.style.right = lim.edge + 'px';
      }
      return y;
    }
    function savePosition(side, top) {
      var lim = limits();
      var span = Math.max(1, lim.maxTop - lim.top);
      try {
        sessionStorage.setItem(POSITION_KEY, JSON.stringify({
          side: side,
          y: Math.max(0, Math.min(1, (top - lim.top) / span))
        }));
      } catch (e) { /* storage is optional; dragging still works */ }
    }
    function restorePosition() {
      try {
        var saved = JSON.parse(sessionStorage.getItem(POSITION_KEY));
        if (!saved || (saved.side !== 'left' && saved.side !== 'right')) return;
        var lim = limits();
        placeAtSide(saved.side, lim.top + Number(saved.y || 0) * (lim.maxTop - lim.top));
      } catch (e) { /* ignore unavailable or malformed session storage */ }
    }
    function constrainPosition() {
      if (!host.style.top) return;
      placeAtSide(host.dataset.side || 'right', parseFloat(host.style.top));
    }
    drag.addEventListener('pointerdown', function (event) {
      if (event.button !== undefined && event.button !== 0) return;
      var box = host.getBoundingClientRect();
      moving = { id: event.pointerId, x: event.clientX, y: event.clientY,
                 left: box.left, top: box.top };
      host.dataset.dragging = 'true';
      host.style.left = box.left + 'px';
      host.style.right = 'auto';
      host.style.top = box.top + 'px';
      host.style.bottom = 'auto';
      drag.setPointerCapture(event.pointerId);
      event.preventDefault();
    });
    drag.addEventListener('pointermove', function (event) {
      if (!moving || moving.id !== event.pointerId) return;
      var lim = limits();
      var left = Math.max(lim.edge, Math.min(window.innerWidth - host.offsetWidth - lim.edge,
                                             moving.left + event.clientX - moving.x));
      var top = Math.max(lim.top, Math.min(lim.maxTop, moving.top + event.clientY - moving.y));
      host.style.left = left + 'px';
      host.style.top = top + 'px';
    });
    function finishMove(event) {
      if (!moving || moving.id !== event.pointerId) return;
      var box = host.getBoundingClientRect();
      var side = box.left + box.width / 2 < window.innerWidth / 2 ? 'left' : 'right';
      var top = placeAtSide(side, box.top);
      savePosition(side, top);
      host.dataset.dragging = 'false';
      moving = null;
    }
    drag.addEventListener('pointerup', finishMove);
    drag.addEventListener('pointercancel', finishMove);
    window.addEventListener('resize', constrainPosition);
    requestAnimationFrame(restorePosition);

    House.on(host, function (L) {
      ROWS.forEach(function (r) {
        vals[r[0]].textContent = (r[0] === 'year' ? String(L.year) : hFmt(L[r[0]])) +
                                 (r[5] ? ' ' + r[5] : '');
      });
      vecline.textContent = 'model-ready vector:  [' + hVec(L).map(function (x) { return hFmt(x); }).join(', ') +
                            ']  ·  age = ' + H_NOW + ' − ' + L.year;
      out.textContent = hFmt(hPrice(L)) + ' k€';
      toggle.setAttribute('aria-label', 'Edit house values. Current predicted price ' +
                          hFmt(hPrice(L)) + ' thousand euros.');
    });
    bindHouseText(host);
  }

  /* ---- prose that quotes the live house --------------------------------
     Not every number on the page lives in an SVG. The opening paragraph
     and the worked dot product in 10.3 spell the house out in ordinary
     HTML, and they sit next to live figures in the same visual style — so
     hard-coding them made them read as broken the moment a slider moved.
     Any element carrying data-house="{key}" is rewritten from the live
     house on every change; the markup keeps the default values so the page
     is still correct with JavaScript off.
     -------------------------------------------------------------------- */
  function hTerms(L) {
    return hVec(L).map(function (x, i) { return hFmt(x) + '(' + hFmt(H_W[i]) + ')'; }).join(' + ');
  }
  function hProducts(L) {
    return hVec(L).map(function (x, i) {
      var t = H_W[i] * x, s = hFmt(Math.abs(t));
      if (i === 0) return t < 0 ? '−' + s : s;
      return (t < 0 ? '− ' : '+ ') + s;
    }).join(' ');
  }
  var H_TEXT = {
    area:  function (L) { return hFmt(L.area); },
    beds:  function (L) { return hFmt(L.beds, 0); },
    year:  function (L) { return String(L.year); },
    age:   function (L) { return hFmt(H_NOW - L.year, 0); },
    km:    function (L) { return hFmt(L.km); },
    price: function (L) { return hFmt(hPrice(L)); },
    terms: hTerms,
    products: hProducts,
    'vec-raw': function (L) {
      return '[' + [L.area, L.beds, L.year, L.km].map(function (x) { return hFmt(x); }).join(', ') + ']';
    },
    'vec-model': function (L) {
      return '[' + hVec(L).map(function (x) { return hFmt(x); }).join(', ') + ']';
    },
    'age-calc': function (L) {
      return 'age = ' + H_NOW + ' − year built = ' + H_NOW + ' − ' + L.year +
             ' = ' + hFmt(H_NOW - L.year, 0);
    }
  };
  function bindHouseText(anchor) {
    var doc = anchor.ownerDocument;
    House.on(anchor, function (L) {
      var nodes = doc.querySelectorAll('[data-house]'), i, fn;
      for (i = 0; i < nodes.length; i++) {
        fn = H_TEXT[nodes[i].getAttribute('data-house')];
        if (fn) nodes[i].textContent = fn(L);
      }
    });
  }

  function houseSliderButton(parent) {
    return button(parent, 'open house sliders', function () {
      var panel = document.querySelector('[data-viz="house-panel"]');
      var toggle = panel && panel.querySelector('.hp-toggle');
      if (!panel || !toggle) return;
      if (panel.dataset.expanded !== 'true') toggle.click();
      requestAnimationFrame(function () {
        var first = panel.querySelector('input[type="range"]');
        if (first) first.focus();
      });
    });
  }

  /* ---- 10.2 · the ladder, as one card with tabs instead of four cards ---- */
  function vizLadder(host) {
    var W = 660, H = 262, svg = makeSVG(host, W, H);
    var g = el('g', {}, svg);
    var read = h('div', 'shapenote', null, host);
    var bar = ctl(host), TAB = ['scalar', 'vector', 'matrix', 'tensor'], k = 1, btns = [];
    TAB.forEach(function (t, i) {
      btns.push(button(bar, t, function () { k = i; paint(H_LIVE); }));
    });

    function paint(L) {
      var i, c;
      for (i = 0; i < btns.length; i++) btns[i].setAttribute('aria-pressed', i === k ? 'true' : 'false');
      g.innerHTML = ''; read.innerHTML = '';
      var v = hVec(L);

      if (k === 0) {
        var p0 = panel(g, [[hFmt(v[0])]], { label: 'the area of your house', cell: 54, fs: 17, bracket: false });
        p0.move(330 - p0.w / 2, 76);
        txt(g, 330, 176, 'shape [ ]  —  one loose card', 'tx-faint', 12);
        h('p', 'rd-a', 'One measurement card holding one number: the area of your house. It is ' +
                       'not packed inside another data container, so its shape is an empty list.', read);
        h('p', 'rd-b', 'The price is a scalar too, and so is a loss. That is the whole point of ' +
                       'a loss function: it takes a model with billions of numbers in it and ' +
                       'returns exactly one, because you can only sort by one number.', read);

      } else if (k === 1) {
        var X0 = 330 - (4 * 65 - 3 + 20) / 2, Y0 = 40, CS = 62;
        var pv = panel(g, [v.map(function (x) { return hFmt(x); })],
                       { label: 'your house', cell: CS, fs: 14 });
        pv.move(X0, Y0);
        for (i = 0; i < 4; i++) {
          var cx = X0 + 10 + i * (CS + 3) + CS / 2;
          txt(g, cx, Y0 + 20 + CS + 20, '[' + i + ']  ' + H_F[i], 'tx-dim', 11);
          txt(g, cx, Y0 + 20 + CS + 35, H_U[i], 'tx-faint', 10);
        }
        txt(g, 330, 206, 'shape [4]  —  four cards in one house folder', 'tx-faint', 12);
        h('p', 'rd-a', 'Put the four measurement cards in one house folder and you have a vector. ' +
                       'Their order is a promise: the first card is always area and the last is ' +
                       'always distance.', read);
        h('p', 'rd-b', 'This is the whole of “represent the house as a vector”. A real model may ' +
                       'use thousands of slots instead of four, but the rules on this page do not ' +
                       'change when the list becomes longer.', read);

      } else if (k === 2) {
        var M = H_SET.map(function (hh) { return hVec(hh).map(function (x) { return hFmt(x); }); });
        M.unshift(v.map(function (x) { return hFmt(x); }));
        var CS2 = 34, GP = 3, MX = 258, MY = 26;
        var pm = panel(g, M, { label: 'the table  ·  shape [5, 4]', cell: CS2, fs: 11.5,
                               hi: function (r) { return r === 0 ? 'sh-hi' : ''; } });
        pm.move(MX, MY);
        ['yours', 'A', 'B', 'C', 'D'].forEach(function (nm, r) {
          txt(g, MX - 8, MY + 20 + r * (CS2 + GP) + CS2 / 2 + 4, nm, r ? 'tx-faint' : 'amber', 11, 'end');
        });
        for (c = 0; c < 4; c++)
          txt(g, MX + 10 + c * (CS2 + GP) + CS2 / 2,
              MY + 20 + 5 * CS2 + 4 * GP + 16, H_FS[c], 'tx-faint', 10);
        h('p', 'rd-a', 'Put five house folders inside one city box and you have a matrix. Open the ' +
                       'box and you see five rows of folders, with the same four cards in each. ' +
                       'Your house did not change; it only gained neighbours.', read);
        h('p', 'rd-b', 'One row per example, one column per feature is the shape of every ' +
                       'dataset you will ever load. When a library asks for X of shape ' +
                       '[n_samples, n_features], this table is what it means.', read);

      } else {
        var CITY = ['Madrid', 'Lisbon', 'Porto'];
        var T = H_SET.map(function (hh) { return hVec(hh).map(function (x) { return hFmt(x); }); });
        T.unshift(v.map(function (x) { return hFmt(x); }));
        var BLANK = [['', '', '', ''], ['', '', '', ''], ['', '', '', ''],
                     ['', '', '', ''], ['', '', '', '']];
        for (i = 2; i >= 0; i--) {
          var pt = panel(g, i === 0 ? T : BLANK,
                         { label: CITY[i], cell: 26, fs: 9.5, bracket: false,
                           ink: '--tx-dim', labelInk: i === 0 ? 'amber' : 'tx-faint' });
          pt.move(160 + i * 116, 42 + (2 - i) * 14);
        }
        txt(g, 330, 250, '[ 3 cities,  5 houses,  4 features ]   —   shape [3, 5, 4]', 'amber', 13);
        h('p', 'rd-a', 'Put three city boxes inside one survey crate and you have a tensor. Open ' +
                       'any city box and the same five-house, four-card arrangement is inside.', read);
        h('p', 'rd-b', 'The label [3, 5, 4] is the packing list: 3 city boxes, each containing ' +
                       '5 house folders, each containing 4 measurement cards.', read);
      }
    }
    House.on(host, paint);
    cap(host, 'Four names for the same measurements at four packing levels: card, house folder, ' +
              'city box and survey crate.');
  }

  /* ---- 10.2 · slicing the same city / house / feature tensor ---- */
  function vizHouseAxes(host) {
    var dims = axes('cities=3, houses=5, features=4');
    var stage = h('div', 'sh-stage', null, host);
    var probe = makeSVG(stage, 10, 10);
    var B0 = block(probe, dims, 0, 0);
    probe.parentNode.removeChild(probe);

    var padL = 58, padR = 132, padT = 36, padB = 42;
    var svg = makeSVG(stage, Math.max(480, B0.w + padL + padR), B0.h + padT + padB);
    var B = block(svg, dims, padL, padT);
    blockAxes(svg, B);
    shapeStr(host, dims);

    var read = h('div', 'shapenote', null, host);
    var bar = ctl(host), btns = [];
    var VIEWS = [
      { l: 'all cities', f: function () { return false; }, s: '[3, 5, 4]',
        a: 'The closed survey crate holds three city boxes. Each city box holds five house folders. ' +
           'Each house folder holds four measurement cards.',
        b: 'The packing label is [3, 5, 4], and the crate contains 3 × 5 × 4 = 60 numbers.' },
      { l: 'one city', f: function (c) { return c.si === 0; }, s: '[5, 4]',
        a: 'Open the crate, take out the Madrid box and put its contents on the desk.',
        b: 'The outer city-box layer falls from view. You now hold five house folders with four ' +
           'cards each, so the label is [5, 4].' },
      { l: 'one house', f: function (c) { return c.si === 0 && c.ri === 0; }, s: '[4]',
        a: 'Open the Madrid box and take out your one house folder.',
        b: 'The house-folder layer falls from view. You now hold four cards: area, bedrooms, age ' +
           'and distance, so the label is [4].' },
      { l: 'all areas', f: function (c) { return c.ci === 0; }, s: '[3, 5]',
        a: 'Pull only the area card from every house folder while keeping all cities and houses.',
        b: 'Each folder contributes one number. What remains is three cities by five houses, so ' +
           'the label is [3, 5].' },
      { l: 'one area', f: function (c) { return c.si === 0 && c.ri === 0 && c.ci === 0; }, s: '[ ]',
        a: 'Open Madrid, open your house folder and pull out its area card.',
        b: 'You are holding one loose number. Every container is outside your current view, so ' +
           'the shape is [ ].' }
    ];

    function pick(i) {
      var v = VIEWS[i];
      hi(B.cells, v.f);
      read.innerHTML = '';
      h('p', 'rd-s', v.l + '   →   shape ' + v.s, read);
      h('p', 'rd-a', v.a, read);
      h('p', 'rd-b', v.b, read);
      btns.forEach(function (b, j) { b.setAttribute('aria-pressed', j === i ? 'true' : 'false'); });
    }
    VIEWS.forEach(function (v, i) { btns.push(button(bar, v.l, function () { pick(i); })); });
    pick(0);
    cap(host, 'Opening one box does not destroy it. It only removes that outer container from what ' +
              'you are currently holding and changes the inventory label.');
  }

  /* the fixed scale: each feature divided by its spread across houses A–D.
     Held fixed (the live house is excluded) so the yardstick does not move
     while you drag. */
  var H_RANGE = (function () {
    var mn = [Infinity, Infinity, Infinity, Infinity], mx = [-Infinity, -Infinity, -Infinity, -Infinity];
    H_SET.forEach(function (hh) {
      hVec(hh).forEach(function (x, i) { if (x < mn[i]) mn[i] = x; if (x > mx[i]) mx[i] = x; });
    });
    return mn.map(function (v, i) { return mx[i] - v; });
  })();
  function hScaled(h) { return hVec(h).map(function (x, i) { return x / H_RANGE[i]; }); }

  /* ---- 10.3 · the dot product is what prices the house ---- */
  function vizHousePrice(host) {
    var W = 660, H = 276, svg = makeSVG(host, W, H);
    var g = el('g', {}, svg), keyb = keyBar(host), read = h('div', 'shapenote', null, host);
    var bar = ctl(host), step = 4;
    button(bar, 'one term at a time', function () { step = step >= 4 ? 1 : step + 1; paint(H_LIVE); });
    button(bar, 'show all four', function () { step = 4; paint(H_LIVE); });
    houseSliderButton(bar);

    function paint(L) {
      g.innerHTML = ''; read.innerHTML = '';
      var v = hVec(L), i, run = 0;
      var pw = panel(g, [H_W.map(function (x) { return hFmt(x); })],
                     { label: 'the weights  w', sub: 'what the market pays per unit', cell: 62, fs: 12.5,
                       hi: function (r, c) { return c < step ? 'sh-hi' : ''; } });
      /* 76px between the two panels put the second panel's title on top of the
         first one's subtitle, and its own subtitle on top of the term row */
      var ph = panel(g, [v.map(function (x) { return hFmt(x); })],
                     { label: 'your house  h  ·  shape [4]', cell: 62, fs: 12.5,
                       hi: function (r, c) { return c < step ? 'sh-hi' : ''; } });
      pw.move(150, 16); ph.move(150, 120);
      for (i = 0; i < 4; i++) txt(g, 150 + 10 + i * 65 + 31, 216, H_F[i], 'tx-faint', 10);
      txt(g, 140, 16 + 34, 'w', 'amber', 13, 'end');
      txt(g, 140, 120 + 34, 'h', 'accent', 13, 'end');

      for (i = 0; i < step; i++) {
        var t = H_W[i] * v[i];
        run += t;
        txt(g, 150 + 10 + i * 65 + 31, 244,
            hFmt(H_W[i]) + '×' + hFmt(v[i]), 'tx-dim', 10.5);
        txt(g, 150 + 10 + i * 65 + 31, 261, hFmt(t), t < 0 ? 'pink' : 'green', 13);
      }
      txt(g, 500, 253, (step === 4 ? '= ' : 'so far  ') + hFmt(run) + ' k€', 'amber', 17, 'start');

      keys(keyb, [['cell', 'multiplied so far'],
                  ['--green', 'this term adds to the price'],
                  ['--pink', 'this term takes price off'],
                  ['--amber', 'the total so far']]);

      var W_WHY = ['each extra square metre adds 3 k€',
                   'each extra bedroom adds 8 k€',
                   'each year of age takes 0.4 k€ off',
                   'each km further out takes 12 k€ off'];
      worked(read, 'price  =  w · h  =  w₀h₀ + w₁h₁ + w₂h₂ + w₃h₃',
             hVec(L).map(function (x, j) { return hFmt(H_W[j]) + '×' + hFmt(x); }).join('  +  ')
               .replace(/\+  −/g, '−  ') + '  =  ' + hFmt(hPrice(L)) + ' k€');
      h('p', 'rd-a', 'A dot product is multiply-then-add, nothing more. Line the two lists up, ' +
                     'multiply each pair, sum the results. Here that arithmetic has a name you ' +
                     'already know: it is the price. ' + W_WHY[Math.min(step, 4) - 1] + '.', read);
      h('p', 'rd-b', 'Both minus signs are doing real work — age and distance push the price ' +
                     'down, so their terms subtract. A weight vector is just a price list, and ' +
                     'the sign of a weight says whether that measurement raises or lowers the ' +
                     'house price.', read);
    }
    House.on(host, paint);
    cap(host, 'Select “open house sliders”, then move area from 120 to 130 m². The area term rises ' +
              'by 3 × 10 = 30 k€, so the predicted price rises by exactly 30 k€.');
  }

  /* ---- 10.4 · a norm measures how different two houses are ---- */
  function vizHouseDist(host) {
    var W = 660, H = 392, svg = makeSVG(host, W, H);
    var g = el('g', {}, svg), keyb = keyBar(host), read = h('div', 'shapenote', null, host);
    var bar = ctl(host), scaled = false, D = H_SET[3];
    button(bar, 'put the features on one scale', function () { scaled = !scaled; paint(H_LIVE); });

    function paint(L) {
      g.innerHTML = ''; read.innerHTML = '';
      var a = scaled ? hScaled(L) : hVec(L), b = scaled ? hScaled(D) : hVec(D);
      var d = a.map(function (x, i) { return x - b[i]; });
      var dp = scaled ? 3 : 1;
      var sq = d.map(function (x) { return x * x; });
      var sum = sq.reduce(function (p, q) { return p + q; }, 0);
      var l2 = Math.sqrt(sum);
      var l1 = d.reduce(function (p, q) { return p + Math.abs(q); }, 0);
      var li = Math.max.apply(null, d.map(Math.abs));

      var pd = panel(g, [d.map(function (x) { return hFmt(x, dp); })],
                     { label: 'your house  −  house D', sub: 'the difference, feature by feature',
                       cell: 76, fs: 12.5,
                       hi: function (r, c) { return Math.abs(d[c]) === li && li > 0 ? 'sh-hi' : ''; } });
      pd.move(120, 14);
      var i, shortFeature = ['area', 'bedrooms', 'age', 'distance'];
      for (i = 0; i < 4; i++) {
        txt(g, 130 + i * 79 + 38, 150, shortFeature[i], 'tx-faint', 10);
        /* on two lines: standardised, "−0.257² = 0.0661" is wider than the
           79px column it has to sit in */
        txt(g, 130 + i * 79 + 38, 163, hFmt(d[i], dp) + '²', 'tx-dim', 10.5);
        txt(g, 130 + i * 79 + 38, 176, '=  ' + hFmt(sq[i], dp === 1 ? 2 : 4), 'tx-dim', 10.5);
      }
      /* one bar per feature, in the same four columns as the numbers above, so
         every difference is on the page and the reader picks the winner
         instead of being handed a shape built from two of them */
      var base = 292, s1 = 92 / Math.max(li, 1e-9);
      el('line', { x1: 122, y1: base, x2: 452, y2: base,
                   stroke: css('--rule-line'), 'stroke-width': 1 }, g);
      for (i = 0; i < 4; i++) {
        var bh = Math.max(Math.abs(d[i]) * s1, 2), bx = 130 + i * 79 + 9;
        var win = Math.abs(d[i]) === li && li > 0;
        el('rect', { x: bx, y: base - bh, width: 60, height: bh, rx: 2,
                     fill: css(win ? '--amber' : '--tx-faint'),
                     'fill-opacity': win ? 0.85 : 0.32,
                     stroke: css(win ? '--amber' : '--border-2'),
                     'stroke-width': 1 }, g);
        txt(g, bx + 30, base - bh - 6, hFmt(Math.abs(d[i]), dp),
            win ? 'amber' : 'tx-dim', 12);
      }
      el('line', { x1: 122, y1: base - 92, x2: 452, y2: base - 92,
                   stroke: css('--amber'), 'stroke-width': 1,
                   'stroke-dasharray': '3 4', 'stroke-opacity': 0.65 }, g);
      txt(g, 458, base - 88, 'L∞ stops here', 'tx-faint', 10.5, 'start');
      txt(g, 458, base - 76, 'nothing below it counts', 'tx-faint', 10.5, 'start');

      /* the three norms on one scale, each one built out of the same four bars
         so the number is never asserted — L1 is the four laid end to end, L∞
         is the winning one on its own, and both name where they came from */
      var winner = shortFeature[d.map(Math.abs).indexOf(li)];
      var NR = [['L1  add the four', l1,
                 d.map(function (x) { return hFmt(Math.abs(x), dp); }).join(' + ')],
                ['L2  square, add, root', l2, '√' + hFmt(sum, dp === 1 ? 2 : 4)],
                ['L∞  keep the biggest', li, 'the ' + winner + ' gap, on its own']];
      var s2 = 200 / Math.max(l1, 1e-9), X0 = 250;
      NR.forEach(function (r, j) {
        var y = 322 + j * 26, w = Math.max(r[1] * s2, 1), k, seg, sx = X0;
        txt(g, 240, y + 4, r[0], 'tx-dim', 12, 'end');
        if (j === 0) {
          /* four segments, in feature order, adding up to the bar */
          for (k = 0; k < 4; k++) {
            seg = Math.abs(d[k]) * s2;
            if (seg <= 0.5) continue;
            var top = Math.abs(d[k]) === li && li > 0;
            el('rect', { x: sx, y: y - 8, width: Math.max(seg - 1, 1), height: 13, rx: 2,
                         fill: css(top ? '--amber' : '--tx-faint'),
                         'fill-opacity': top ? 0.8 : 0.3 }, g);
            if (seg > 26) txt(g, sx + seg / 2, y + 4, hFmt(Math.abs(d[k]), dp),
                              top ? 'bg-2' : 'tx-dim', 10.5);
            sx += seg;
          }
        } else if (j === 2) {
          /* L∞ is the winning bar and nothing else, so it keeps its colour */
          el('rect', { x: X0, y: y - 8, width: w, height: 13, rx: 2,
                       fill: css('--amber'), 'fill-opacity': 0.8 }, g);
        } else {
          el('rect', { x: X0, y: y - 8, width: w, height: 13, rx: 2,
                       fill: css('--tx-faint'), 'fill-opacity': 0.3,
                       stroke: css('--amber'), 'stroke-width': 1,
                       'stroke-opacity': 0.55 }, g);
        }
        var vt = txt(g, X0 + w + 8, y + 4, hFmt(r[1], dp === 1 ? 2 : 3),
                     j === 1 ? 'amber' : 'chalk', 13.5, 'start');
        var vw = 0;
        try { vw = vt.getBBox().width; } catch (e) { vw = 0; }
        if (!vw) vw = String(hFmt(r[1], dp === 1 ? 2 : 3)).length * 7.4;
        txt(g, X0 + w + 20 + vw, y + 4, r[2], 'tx-faint', 10.5, 'start');
      });

      keys(keyb, [['--amber', 'the biggest of the four gaps — the one L∞ keeps'],
                  ['--tx-faint', 'the other three gaps, which L1 and L2 still count'],
                  ['--amber', 'the height L∞ stops at', 'dash']]);

      worked(read, '‖d‖₂  =  √( d₀² + d₁² + d₂² + d₃² )',
             '√( ' + sq.map(function (x) { return hFmt(x, dp === 1 ? 2 : 4); }).join(' + ') +
             ' )  =  √' + hFmt(sum, dp === 1 ? 2 : 4) + '  =  ' + hFmt(l2, dp === 1 ? 2 : 3));
      h('p', 'rd-a', 'One rule, however many features: square every difference, add them all, ' +
                     'take the root. The bars are those differences before squaring, and every ' +
                     'feature gets one — a feature that matches exactly draws a flat bar and ' +
                     'contributes 0 to the sum, but it is still there. Squaring is what turns a ' +
                     'modest lead into a decisive one, which is why the tallest bar ends up ' +
                     'deciding the answer almost on its own.', read);
      h('p', 'rd-b', scaled
        ? 'Now every feature has been divided by its own spread across houses A–D, so all four ' +
          'are measured on a comparable yardstick, and the answer changes character: the ' +
          'largest gap is no longer automatically the one measured in the biggest unit. This ' +
          'is what “standardise your features” means and why it is not optional.'
        : 'Look at which feature won. Area is measured in a unit that runs to hundreds and ' +
          'bedrooms in a unit that runs to single digits, so area drowns everything else and ' +
          '“distance between houses” has quietly become “difference in floor area”. Select ' +
          '“put the features on one scale” to divide each feature by its own spread and watch ' +
          'the answer change.', read);
    }
    House.on(host, paint);
    cap(host, 'Every norm bar is built from the four bars above it. L1 lays all four end to end. ' +
              'L∞ keeps the tallest and discards the rest. L2 squares all four, adds them and ' +
              'takes the root, which always lands between the other two. The more one feature ' +
              'dominates, the closer the three answers get to each other.');
  }

  /* ---- 10.5 · cosine strips out size and leaves the shape ---- */
  function vizHouseShape(host) {
    var W = 700, H = 272, svg = makeSVG(host, W, H);
    var g = el('g', {}, svg), keyb = keyBar(host), read = h('div', 'shapenote', null, host);
    var bar = ctl(host), scaled = true, other = 2;
    button(bar, 'compare with another house', function () {
      other = (other + 1) % H_SET.length; paint(H_LIVE);
    });
    button(bar, 'use the raw units instead', function () { scaled = !scaled; paint(H_LIVE); });

    function paint(L) {
      g.innerHTML = ''; read.innerHTML = '';
      var O = H_SET[other];
      var a = scaled ? hScaled(L) : hVec(L), b = scaled ? hScaled(O) : hVec(O);
      var dot = 0, na = 0, nb = 0, i;
      for (i = 0; i < 4; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
      na = Math.sqrt(na); nb = Math.sqrt(nb);
      var cos = dot / (na * nb), th = Math.acos(Math.max(-1, Math.min(1, cos)));

      /* two vectors always span a plane — this IS that plane, so the angle
         drawn is the true four-dimensional angle, not a projection */
      var cx = 150, cy = 178, R = 128;
      el('path', { d: 'M' + cx + ',' + cy + 'l' + R + ',0 A' + R + ',' + R + ' 0 0 0 ' +
                      (cx + R * Math.cos(th)) + ',' + (cy - R * Math.sin(th)) + 'Z',
                   fill: css('--amber'), 'fill-opacity': 0.1 }, g);
      arrow(g, cx, cy, cx + R, cy, 'accent');
      arrow(g, cx, cy, cx + R * Math.cos(th), cy - R * Math.sin(th), 'green');
      txt(g, cx + R + 6, cy + 4, 'yours', 'accent', 11, 'start');
      txt(g, cx + R * Math.cos(th) + 6, cy - R * Math.sin(th) - 4, 'house ' + O.id, 'green', 11, 'start');
      txt(g, cx + 46, cy - 12, (th * 180 / Math.PI).toFixed(1) + '°', 'amber', 13, 'start');

      /* every one of these four numbers is written out from the two houses'
         own components — none of them is asserted */
      var dq = scaled ? 3 : 1;
      function sq4(v) {
        return '√( ' + v.map(function (x) { return hFmt(x, dq) + '²'; }).join(' + ') + ' )';
      }
      var lines = [['yours · house ' + O.id, hFmt(dot, 2),
                    a.map(function (x, k) { return hFmt(x, dq) + '×' + hFmt(b[k], dq); }).join(' + ')],
                   ['‖yours‖', hFmt(na, 2), sq4(a)],
                   ['‖house ' + O.id + '‖', hFmt(nb, 2), sq4(b)],
                   ['cos θ', cos.toFixed(3),
                    hFmt(dot, 2) + '  /  ( ' + hFmt(na, 2) + ' × ' + hFmt(nb, 2) + ' )']];
      lines.forEach(function (r, j) {
        var y = 52 + j * 48;
        txt(g, 340, y, r[0], 'tx-dim', 12, 'start');
        txt(g, 690, y, r[1], j === 3 ? 'amber' : 'chalk', j === 3 ? 17 : 14, 'end');
        txt(g, 340, y + 16, r[2], 'tx-faint', 9.5, 'start');
      });
      txt(g, 510, 258, scaled ? 'features put on one scale' : 'raw units — area dominates',
          scaled ? 'green' : 'pink', 11);

      keys(keyb, [['--accent', 'your house, drawn as an arrow', 'line'],
                  ['--green', 'house ' + O.id + ', the one you are comparing against', 'line'],
                  ['--amber', 'the angle between them — what cosine measures']]);

      worked(read, 'cos θ  =  (a · b) / ( ‖a‖ ‖b‖ )',
             hFmt(dot, 2) + '  /  ( ' + hFmt(na, 2) + ' × ' + hFmt(nb, 2) + ' )  =  ' +
             cos.toFixed(3));
      h('p', 'rd-a', 'Nothing new is being computed. The top line is the dot product from 10.3 ' +
                     'run on these two houses, and the two lengths are the Pythagoras from ' +
                     '10.4. Every figure above is written out from the two houses’ own four ' +
                     'numbers.', read);
      h('p', 'rd-a', 'Two houses can be the same kind of house at different sizes. Double every ' +
                     'measurement and the arrow gets longer but points exactly where it did, so ' +
                     'the angle is zero and the cosine is 1. Dividing by both lengths is what ' +
                     'throws the size away and keeps only the direction.', read);
      h('p', 'rd-b', scaled
        ? 'Because the features are on one scale, this angle means something: it says how ' +
          'similar the two houses are in character — old-and-far-out versus big-and-central — ' +
          'independent of how large either one is.'
        : 'On raw units every house scores above 0.9 against every other house, because the ' +
          'area component is a hundred times bigger than the bedroom component and all the ' +
          'arrows end up pointing nearly along the area axis. The number looks like agreement ' +
          'and is really just a shared unit. Turn scaling back on and compare the two answers.', read);
    }
    House.on(host, paint);
    cap(host, 'Distance and cosine answer different questions. “How far apart are these two ' +
              'houses?” is L2. “Are they the same kind of house?” is cosine. Ranking by the ' +
              'wrong one is a real and common bug.');
  }

  /* --------- linear algebra on the four houses: 10.6 – 10.8 ---------
     Numeric helpers kept local. The four reference houses are the only
     ones with a known price, so these three figures work on A–D rather
     than on the live slider house — you cannot solve for the weights
     from a house whose price you do not know. ------------------------ */

  var H_X = H_SET.map(hVec);                       /* 4 houses × 4 features */
  var H_P = H_SET.map(hPrice);                     /* their prices          */

  function matInv(M) {
    var n = M.length, i, r, c, A = M.map(function (row, k) {
      return row.concat(row.map(function (_, j) { return k === j ? 1 : 0; }));
    });
    for (i = 0; i < n; i++) {
      var p = i;
      for (r = i + 1; r < n; r++) if (Math.abs(A[r][i]) > Math.abs(A[p][i])) p = r;
      var tmp = A[p]; A[p] = A[i]; A[i] = tmp;
      var pv = A[i][i];
      if (Math.abs(pv) < 1e-12) return null;
      for (c = 0; c < 2 * n; c++) A[i][c] /= pv;
      for (r = 0; r < n; r++) {
        if (r === i) continue;
        var f = A[r][i];
        for (c = 0; c < 2 * n; c++) A[r][c] -= f * A[i][c];
      }
    }
    return A.map(function (row) { return row.slice(n); });
  }

  function matVec(M, v) {
    return M.map(function (row) {
      return row.reduce(function (s, x, i) { return s + x * v[i]; }, 0);
    });
  }

  /* ---- 10.6 · every house priced at once — a matrix times a vector ---- */
  function vizHouseTable(host) {
    var W = 660, H = 268, svg = makeSVG(host, W, H);
    var g = el('g', {}, svg), keyb = keyBar(host), read = h('div', 'shapenote', null, host);
    var bar = ctl(host), row = 0, flip = false;
    button(bar, 'next house', function () { flip = false; row = (row + 1) % 5; paint(H_LIVE); });
    button(bar, 'transpose the table', function () { flip = !flip; paint(H_LIVE); });
    houseSliderButton(bar);

    function paint(L) {
      g.innerHTML = ''; read.innerHTML = '';
      var names = ['yours', 'A', 'B', 'C', 'D'];
      var rows = [hVec(L)].concat(H_X);
      var i, c;

      if (flip) {
        var pt = panel(g, H_F.map(function (_, f) {
          return rows.map(function (rv) { return hFmt(rv[f]); });
        }), { label: 'Xᵀ  —  shape [4, 5]', cell: 32, fs: 10.5 });
        pt.move(330 - pt.w / 2, 60);
        for (i = 0; i < 5; i++)
          txt(g, 330 - pt.w / 2 + 10 + i * 35 + 16, 46, names[i], i ? 'tx-faint' : 'amber', 10);
        for (c = 0; c < 4; c++)
          txt(g, 330 - pt.w / 2 - 8, 80 + c * 35 + 16, H_FS[c], 'tx-faint', 10, 'end');
        keys(keyb, [['--amber', 'your house — the only row the sliders move']]);
        worked(read, 'X is [5, 4]  →  Xᵀ is [4, 5]',
               'The entry in row 2, column 3 of X is the entry in row 3, column 2 of Xᵀ. ' +
               'Nothing was multiplied and nothing was added — every number kept its value ' +
               'and changed address.');
        h('p', 'rd-a', 'Transposing flips the table over its diagonal: houses become the ' +
                       'columns and features become the rows. It is pure bookkeeping, and it ' +
                       'exists because matrix shapes have to chain.', read);
        h('p', 'rd-b', 'The next section needs XᵀX to recover the price list: X is [5, 4], so ' +
                       'XᵀX is [4, 5]×[5, 4] = [4, 4] — one row and one column per house ' +
                       'measurement, regardless of how many houses are in the table.', read);
        return;
      }

      var px = panel(g, rows.map(function (rv) {
        return rv.map(function (x) { return hFmt(x); });
      }), { label: 'X  ·  the houses', cell: 32, fs: 10.5,
            hi: function (r) { return r === row ? 'sh-hi' : ''; } });
      var pw = panel(g, H_W.map(function (x) { return [hFmt(x)]; }),
                     { label: 'w', cell: 32, fs: 10.5, hi: function () { return 'sh-hi'; } });
      var pp = panel(g, rows.map(function (rv) {
        return [hFmt(rv.reduce(function (s, x, i) { return s + H_W[i] * x; }, 0))];
      }), { label: 'p  ·  prices', cell: 32, fs: 9.5,
            hi: function (r) { return r === row ? 'sh-hi' : ''; } });
      lay([px, sym(g, '×'), pw, sym(g, '='), pp], 342, 130, 22);

      var ox = px.g.getAttribute('transform').match(/translate\(([-\d.]+),([-\d.]+)\)/);
      var X0 = parseFloat(ox[1]), Y0 = parseFloat(ox[2]);
      for (i = 0; i < 5; i++)
        txt(g, X0 - 8, Y0 + 20 + i * 35 + 21, names[i], i === row ? 'amber' : 'tx-faint', 10.5, 'end');
      for (c = 0; c < 4; c++)
        txt(g, X0 + 10 + c * 35 + 16, Y0 + 20 + 5 * 32 + 4 * 3 + 16, H_FS[c], 'tx-faint', 10);

      keys(keyb, [['cell', 'the row being multiplied, and the price it produces'],
                  ['--amber', 'the house named in the line below']]);
      var rv = rows[row], terms = rv.map(function (x, j) {
        return hFmt(H_W[j]) + '×' + hFmt(x);
      }).join('  +  ').replace(/\+  −/g, '−  ');
      worked(read, 'p = X w  [5, 4] × [4] → [5]',
             'row ' + (row ? names[row] : 'yours') + ' of X  ·  w   =   ' + terms + '  =  ' +
             hFmt(rv.reduce(function (s, x, i) { return s + H_W[i] * x; }, 0)) + ' k€');
      h('p', 'rd-a', 'Nothing new is happening. Each row of the result is the dot product you ' +
                     'did by hand in the last section — the same price list w applied to a ' +
                     'different house. A matrix multiply is a stack of dot products and no more ' +
                     'than that.', read);
      h('p', 'rd-b', 'Read the shapes: [5, 4] × [4] → [5]. Each house row has four measurements ' +
                     'and the price list has four matching prices, so the inner 4s pair up. What ' +
                     'remains is one result for each of the five houses: shape [5].', read);
    }
    House.on(host, paint);
    cap(host, 'One price list, every house at once. Select “open house sliders” and change a value: ' +
              'only your top row and its price move; houses A–D remain fixed reference points.');
  }

  /* ---- 10.7 · running the model backwards — the inverse ---- */
  function vizHouseSolve(host) {
    var W = 660, H = 278, svg = makeSVG(host, W, H);
    var g = el('g', {}, svg), keyb = keyBar(host), read = h('div', 'shapenote', null, host);
    var bar = ctl(host), step = 0, XI = matInv(H_X), REC = matVec(XI, H_P);
    button(bar, 'next step', function () { step = (step + 1) % 3; paint(); });
    button(bar, 'start over', function () { step = 0; paint(); });

    function paint() {
      g.innerHTML = ''; read.innerHTML = '';
      var names = ['A', 'B', 'C', 'D'], i;

      if (step === 0) {
        var px = panel(g, H_X.map(function (rv) {
          return rv.map(function (x) { return hFmt(x); });
        }), { label: 'X  ·  four houses', cell: 34, fs: 10.5 });
        var pq = panel(g, ['?', '?', '?', '?'].map(function (x) { return [x]; }),
                       { label: 'w  ·  unknown', cell: 34, fs: 12, ink: '--pink' });
        var pp = panel(g, H_P.map(function (x) { return [hFmt(x)]; }),
                       { label: 'p  ·  known', cell: 34, fs: 9.5 });
        lay([px, sym(g, '×'), pq, sym(g, '='), pp], 330, 138, 22);
        worked(read, 'X w = p  four equations, four unknowns',
               '120w₀ + 3w₁ + 41w₂ + 4.2w₃ = 317.2 · and the same for B, C and D.');
        h('p', 'rd-a', 'Until now we knew the price list and computed the prices. Turn it round. ' +
                       'You have sold four houses, so you know X and you know p — what you want ' +
                       'is w, the price the market actually puts on a square metre.', read);
        h('p', 'rd-b', 'The house measurements and sale prices are known. The four entries in ' +
                       'the market price list are unknown. The inverse is the tool that recovers ' +
                       'those four missing entries in this square, independent table.', read);

      } else if (step === 1) {
        var pi = panel(g, XI.map(function (rv) {
          return rv.map(function (x) { return x.toFixed(2).replace('-', '−'); });
        }), { label: 'X⁻¹  ·  the undo machine', cell: 42, fs: 10.5 });
        pi.move(330 - pi.w / 2, 42);
        for (i = 0; i < 4; i++)
          txt(g, 330 - pi.w / 2 - 8, 62 + i * 45 + 25, H_FS[i], 'tx-faint', 10, 'end');
        txt(g, 200, 264, 'X⁻¹X = I', 'amber', 13);
        txt(g, 330, 264, 'X⁻¹(Xw) = X⁻¹p', 'amber', 13);
        txt(g, 470, 264, 'w = X⁻¹p', 'amber', 13);
        worked(read, 'A⁻¹A = I',
               'The identity I is the do-nothing machine, ones down the diagonal. ' +
               'Multiplying by X⁻¹ undoes multiplying by X, exactly as dividing by 3 undoes ' +
               'multiplying by 3. Apply it to both sides and the unknown w is left alone on ' +
               'the left.');
        h('p', 'rd-a', 'These sixteen numbers are not arbitrary: they are the only matrix that ' +
                       'cancels X. You never compute one by hand — a solver does it in ' +
                       'microseconds — but it is worth seeing that it exists and is unique.', read);
        h('p', 'rd-b', 'Look at the bedrooms row: ±17, an order of magnitude larger than the ' +
                       'rest. Big houses tend to have more bedrooms, so those two columns nearly ' +
                       'repeat each other, and the inverse has to take a large difference of ' +
                       'two large numbers to tell them apart. That fragility has a name — ' +
                       'ill-conditioning — and the next section is what happens when it goes ' +
                       'all the way.', read);

      } else {
        var pi2 = panel(g, XI.map(function (rv) {
          return rv.map(function (x) { return x.toFixed(2).replace('-', '−'); });
        }), { label: 'X⁻¹', cell: 38, fs: 9.5 });
        var pp2 = panel(g, H_P.map(function (x) { return [hFmt(x)]; }),
                        { label: 'p', cell: 38, fs: 9.5 });
        var pw2 = panel(g, REC.map(function (x) { return [hFmt(x, 1)]; }),
                        { label: 'w  ·  recovered', cell: 38, fs: 11,
                          hi: function () { return 'sh-hi'; } });
        lay([pi2, sym(g, '×'), pp2, sym(g, '='), pw2], 330, 138, 26);
        var ow = pw2.g.getAttribute('transform').match(/translate\(([-\d.]+),([-\d.]+)\)/);
        for (i = 0; i < 4; i++)
          txt(g, parseFloat(ow[1]) + pw2.w + 8,
              parseFloat(ow[2]) + 20 + i * 41 + 24, H_FS[i], 'tx-faint', 10, 'start');
        var r0 = XI[0].map(function (x, j) {
          return (x < 0 ? '− ' : (j ? '+ ' : '')) + Math.abs(x).toFixed(4) + '×' + hFmt(H_P[j], 1);
        }).join('  ');
        worked(read, 'w = X⁻¹p = [ 3, 8, −0.4, −12 ]',
               'row 0 of X⁻¹ against p:  ' + r0 + '  =  3.0 k€ per square metre. ' +
               '(The cells above are rounded to two places for room; the solver keeps every digit.)');
        h('p', 'rd-a', 'The four numbers you were handed at the start of this page — 3 k€ per ' +
                       'square metre, 8 per bedroom, −12 per kilometre — were never given to ' +
                       'you by an authority. They are what falls out of four sold houses and ' +
                       'their four prices. That is what fitting a model means.', read);
        h('p', 'rd-b', 'Real data never lands this cleanly: you have ten thousand houses, not ' +
                       'four, so X is tall and has no inverse. The standard repair is ' +
                       'w = (XᵀX)⁻¹Xᵀp — the normal equations — which is the same idea after ' +
                       'squaring X up first, and the transpose from the last section is exactly ' +
                       'what makes those shapes chain.', read);
      }
      keys(keyb, step === 0
        ? [['--pink', 'unknown — the four numbers you are solving for'],
           ['--chalk', 'known — measured houses and the prices they sold for']]
        : step === 1
          ? [['--amber', 'the three lines of algebra that isolate w']]
          : [['cell', 'the recovered price list — the answer']]);
      txt(g, 330, 20, ['the question', 'the tool', 'the answer'][step], 'gold-dim', 11);
    }
    paint();
    cap(host, 'Three steps from “what does the market pay?” to the four numbers this whole ' +
              'page has been using.');
  }

  /* ---- 10.8 · when the undo fails — rank and the determinant ---- */
  function vizHouseSingular(host) {
    var W = 660, H = 310, svg = makeSVG(host, W, H);
    var g = el('g', {}, svg), keyb = keyBar(host), read = h('div', 'shapenote', null, host);
    var FT = 10.7639, t = 0;
    var Y = H_X.map(function (rv) { return [rv[0], rv[0] * FT, rv[1], rv[2], rv[3]]; });
    var LBL = ['m²', 'ft²', 'beds', 'age', 'km'];
    var bar = ctl(host);
    slider(bar, 'blend the two answers', 0, 1, 0.02, 0, function (v) {
      t = v; paint();
      return t === 0 ? 'all on m²' : t === 1 ? 'all on ft²' : Math.round(t * 100) + '% on ft²';
    });

    function wOf() { return [3 * (1 - t), (3 / FT) * t, 8, -0.4, -12]; }

    function paint() {
      g.innerHTML = ''; read.innerHTML = '';
      var w = wOf(), pr = matVec(Y, w), i;
      var py = panel(g, Y.map(function (rv) {
        return rv.map(function (x, c) { return hFmt(x, c === 1 ? 0 : 1); });
      }), { label: 'Y  ·  four houses, five columns', cell: 34, fs: 9.5,
            hi: function (r, c) { return c < 2 ? 'sh-hi' : ''; } });
      var pw = panel(g, w.map(function (x, wi) { return [hFmt(x, wi === 1 ? 4 : 2)]; }),
                     { label: 'w', cell: 34, fs: 9.5 });
      var pp = panel(g, pr.map(function (x) { return [hFmt(x, 1)]; }),
                     { label: 'price', cell: 34, fs: 9.5 });
      lay([py, sym(g, '×'), pw, sym(g, '='), pp], 330, 128, 28);
      var oy = py.g.getAttribute('transform').match(/translate\(([-\d.]+),([-\d.]+)\)/);
      var X0 = parseFloat(oy[1]), Y0 = parseFloat(oy[2]);
      for (i = 0; i < 5; i++)
        txt(g, X0 + 10 + i * 37 + 17, Y0 + 20 + 4 * 34 + 3 * 3 + 15, LBL[i],
            i < 2 ? 'amber' : 'tx-faint', 10);
      for (i = 0; i < 4; i++)
        txt(g, X0 - 8, Y0 + 20 + i * 37 + 22, ['A', 'B', 'C', 'D'][i], 'tx-faint', 10, 'end');
      var ow = pw.g.getAttribute('transform').match(/translate\(([-\d.]+),([-\d.]+)\)/);
      for (i = 0; i < 5; i++)
        txt(g, parseFloat(ow[1]) + pw.w + 6, parseFloat(ow[2]) + 20 + i * 37 + 22,
            LBL[i], i < 2 ? 'amber' : 'tx-faint', 9.5, 'start');
      keys(keyb, [['cell', 'the two columns that say the same thing in different units'],
                  ['--amber', 'the m² and ft² labels — the duplicated pair, on the table and on w'],
                  ['--pink', 'the three ways of saying the undo has failed']]);
      txt(g, 185, 286, 'det(YᵀY) = 0', 'pink', 13);
      txt(g, 330, 286, 'rank 4, not 5', 'pink', 13);
      txt(g, 485, 286, '(YᵀY)⁻¹ does not exist', 'pink', 13);

      worked(read, 'column 1 = 10.7639 × column 0',
             'The 10.7639 is not measured from these houses — one metre is 3.28084 feet, so one ' +
             'square metre is 3.28084² = 10.7639 square feet, and column 1 is column 0 through ' +
             'that fixed conversion. ' +
             'The area contributes w₀ + w₁×10.7639 per square metre, whatever the split: ' +
             hFmt(w[0], 2) + ' + ' + hFmt(w[1], 4) + '×10.7639 = ' + hFmt(w[0], 2) + ' + ' +
             hFmt(w[1] * FT, 2) + ' = 3.00 k€. The second column carries no information the ' +
             'first did not already have, so the two weights can trade freely.');
      h('p', 'rd-a', 'Drag the slider. The two highlighted weights change all the way from ' +
                     '(3.00, 0.0000) to (0.00, 0.2787) — and not one price moves. There is no ' +
                     'single right answer any more: infinitely many price lists fit these four ' +
                     'houses exactly.', read);
      h('p', 'rd-b', 'That is what the three statements mean. Rank 4 — only four of the five ' +
                     'columns carry independent information. det(YᵀY) = 0 — the duplicate ' +
                     'direction is flattened completely. Singular — (YᵀY)⁻¹ in the normal ' +
                     'equations does not exist. In a real fit you do not see a clean error: you see ' +
                     'coefficients that swing wildly when you add one more row of data, the ' +
                     'textbook symptom of multicollinearity. Ridge regression repairs it by ' +
                     'adding λI to YᵀY, which lifts the determinant away from zero and picks one ' +
                     'answer out of the infinity.', read);
    }
    paint();
    cap(host, 'Five columns, but only four independent directions: m² and ft² say the same thing. ' +
              'The slider shows why the five price weights cannot have one unique answer.');
  }

  /* ---------------- 10.9 – 10.11 · eigen, SVD, and the two penalties ----
     Everything below is still the same four houses. The yardsticks (means,
     spreads, the fixed scale) are computed once from A–D so that dragging
     the live house moves the house and never the ruler. ----------------- */

  var H_MU = [0, 1, 2, 3].map(function (j) {
    return H_SET.reduce(function (s, hh) { return s + hVec(hh)[j]; }, 0) / H_SET.length;
  });
  var H_SD = [0, 1, 2, 3].map(function (j) {
    return Math.sqrt(H_SET.reduce(function (s, hh) {
      return s + Math.pow(hVec(hh)[j] - H_MU[j], 2);
    }, 0) / (H_SET.length - 1));
  });
  function hZ(hh, j) { return (hVec(hh)[j] - H_MU[j]) / H_SD[j]; }

  /* correlation between area and bedrooms across A–D */
  var H_R = (function () {
    var s = 0;
    H_SET.forEach(function (hh) { s += hZ(hh, 0) * hZ(hh, 1); });
    return s / (H_SET.length - 1);
  })();

  /* ---- 10.9 · the direction the market varies in ---- */
  function vizHouseEigen(host) {
    var W = 660, H = 306, svg = makeSVG(host, W, H);
    var g = el('g', {}, svg), keyb = keyBar(host), read = h('div', 'shapenote', null, host);
    var bar = ctl(host), mode = 0;
    var MODES = ['the cloud', 'the eigenvector', 'a direction that is not one',
                 'one number instead of two'];
    MODES.forEach(function (m, i) { button(bar, m, function () { mode = i; paint(H_LIVE); }); });
    houseSliderButton(bar);

    var L2 = Math.SQRT1_2, LAM1 = 1 + H_R, LAM2 = 1 - H_R;

    function paint(L) {
      g.innerHTML = ''; read.innerHTML = '';
      var box = { l: 66, t: 44, w: 224, h: 224 }, F = new Frame(g, box, [-2.4, 2.4], [-2.4, 2.4]);
      var i, hh, zx, zy;

      /* grid through the origin, because these are deviations from the average house */
      el('line', { x1: F.X(-2.4), y1: F.Y(0), x2: F.X(2.4), y2: F.Y(0), class: 's-axis' }, g);
      el('line', { x1: F.X(0), y1: F.Y(-2.4), x2: F.X(0), y2: F.Y(2.4), class: 's-axis' }, g);
      txt(g, F.X(2.4), F.Y(0) + 16, 'area →', 'tx-faint', 10, 'end');
      txt(g, F.X(0) + 6, F.Y(2.4) - 4, 'bedrooms ↑', 'tx-faint', 10, 'start');
      txt(g, 178, 26, 'the four houses, each feature in units of its own spread',
          'tx-faint', 10.5);

      if (mode === 1 || mode === 3) {
        el('line', { x1: F.X(-2.4), y1: F.Y(-2.4), x2: F.X(2.4), y2: F.Y(2.4),
                     stroke: css('--gold-dim'), 'stroke-width': 1, 'stroke-dasharray': '3 4' }, g);
      }

      H_SET.forEach(function (hs, k) {
        zx = hZ(hs, 0); zy = hZ(hs, 1);
        el('circle', { cx: F.X(zx), cy: F.Y(zy), r: 5, fill: css('--tx-dim') }, g);
        txt(g, F.X(zx) + 11, F.Y(zy) + 4, hs.id, 'tx-faint', 10, 'start');
        if (mode === 3) {
          var t = (zx + zy) / 2;
          el('line', { x1: F.X(zx), y1: F.Y(zy), x2: F.X(t), y2: F.Y(t),
                       stroke: css('--rule-line'), 'stroke-width': 1 }, g);
          el('circle', { cx: F.X(t), cy: F.Y(t), r: 3.5, fill: css('--gold') }, g);
        }
      });
      zx = hZ(L, 0); zy = hZ(L, 1);
      el('circle', { cx: F.X(Math.max(-2.4, Math.min(2.4, zx))),
                     cy: F.Y(Math.max(-2.4, Math.min(2.4, zy))), r: 6.5,
                     fill: 'none', stroke: css('--accent'), 'stroke-width': 2 }, g);
      /* the marker is clamped to the frame, so at the slider extremes the label
         would sit outside it — above the top edge, on the caption, or past the
         right edge. Flip it inwards instead of letting it leave the box. */
      var mx = F.X(Math.max(-2.4, Math.min(2.4, zx))), my = F.Y(Math.max(-2.4, Math.min(2.4, zy)));
      var far = mx > box.l + box.w - 46;
      txt(g, mx + (far ? -12 : 12), my < box.t + 18 ? my + 18 : my - 8, 'yours', 'accent', 10,
          far ? 'end' : 'start');

      if (mode === 1) {
        arrow(g, F.X(0), F.Y(0), F.X(L2), F.Y(L2), 'accent');
        arrow(g, F.X(0), F.Y(0), F.X(LAM1 * L2), F.Y(LAM1 * L2), 'amber');
        txt(g, F.X(0.55), F.Y(0.55) - 12, 'v', 'accent', 12);
        txt(g, F.X(1.45), F.Y(1.45) - 12, 'Cv', 'amber', 12);
      } else if (mode === 2) {
        arrow(g, F.X(0), F.Y(0), F.X(1.4), F.Y(0), 'accent');
        arrow(g, F.X(0), F.Y(0), F.X(1.4), F.Y(1.4 * H_R), 'pink');
        txt(g, F.X(1.5), F.Y(0) - 10, 'u', 'accent', 12, 'start');
        txt(g, F.X(1.5), F.Y(1.4 * H_R) + 20, 'Cu', 'pink', 12, 'start');
      }

      /* the right-hand column: the matrix, and what this mode is saying */
      var pc = panel(g, [['1.00', hFmt(H_R, 2)], [hFmt(H_R, 2), '1.00']],
                     { label: 'C  ·  how the two features move together', cell: 44, fs: 12 });
      pc.move(452 - pc.w / 2, 44);
      txt(g, 452, 176, ['area', 'beds'][0] + '   ' + ['area', 'beds'][1], 'tx-faint', 0);
      txt(g, 452 - 44, 172, 'area', 'tx-faint', 10);
      txt(g, 452 + 44, 172, 'beds', 'tx-faint', 10);

      keys(keyb, [['--tx-dim', 'the four sold houses'],
                  ['--accent', 'your house, and the direction going in', 'line'],
                  mode === 1 ? ['--amber', 'the same direction after C — longer, not turned', 'line'] : null,
                  mode === 2 ? ['--pink', 'after C — turned, so u is not an eigenvector', 'line'] : null,
                  mode === 3 ? ['--gold', 'each house reduced to one number on the bigness axis'] : null,
                  (mode === 1 || mode === 3) ? ['--gold-dim', 'the bigness axis', 'dash'] : null]);

      var LINES = [
        ['Almost a straight line.', 'Bigger houses have more bedrooms:',
         'the two features carry nearly the same news.'],
        ['C v  =  ' + hFmt(LAM1, 2) + ' v', 'v came out pointing exactly where it went in.',
         'That is what makes it an eigenvector; ' + hFmt(LAM1, 2) + ' is its eigenvalue.'],
        ['C u  =  [ 1.00, ' + hFmt(H_R, 2) + ' ]', 'u went in flat and came out tilted 44° up.',
         'A rotated arrow is not λu, so u is not an eigenvector.'],
        ['keep ' + hFmt(100 * LAM1 / 2, 1) + ' % of the spread', 'Each house replaced by one number:',
         'how far along the bigness axis it sits.']
      ];
      txt(g, 452, 210, LINES[mode][0], 'amber', 14);
      txt(g, 452, 234, LINES[mode][1], 'tx-dim', 11);
      txt(g, 452, 250, LINES[mode][2], 'tx-dim', 11);

      var WORK = [
        ['C is the correlation matrix of the four houses',
         'Every entry is an average of z-scores: the diagonal is each feature with itself (always 1), ' +
         'and the off-diagonal ' + hFmt(H_R, 2) + ' is area with bedrooms. Close to 1 means the ' +
         'cloud is nearly a line.'],
        ['C v = λ v      v = [ 0.71, 0.71 ],  λ = ' + hFmt(LAM1, 3),
         'Row 1 of C against v:  1.00×0.71 + ' + hFmt(H_R, 2) + '×0.71 = ' + hFmt(LAM1 * L2, 2) +
         '.  Row 2 gives the same. So Cv = [' + hFmt(LAM1 * L2, 2) + ', ' + hFmt(LAM1 * L2, 2) +
         '] = ' + hFmt(LAM1, 2) + ' × [0.71, 0.71]  —  same direction, ' + hFmt(LAM1, 2) +
         ' times as long.'],
        ['C u = [ 1.00, ' + hFmt(H_R, 2) + ' ]      u = [ 1, 0 ]',
         'Row 1 of C against u:  1.00×1 + ' + hFmt(H_R, 2) + '×0 = 1.00.  Row 2:  ' + hFmt(H_R, 2) +
         '×1 + 1.00×0 = ' + hFmt(H_R, 2) + '.  The result points ' +
         hFmt(Math.atan2(H_R, 1) * 180 / Math.PI, 0) + '° away from where u pointed.'],
        ['λ₁ = ' + hFmt(LAM1, 3) + ',   λ₂ = ' + hFmt(LAM2, 3) + ',   λ₁ + λ₂ = 2',
         'The two eigenvalues always add up to the total spread — here 2, one unit per ' +
         'standardised feature. The first axis holds ' + hFmt(100 * LAM1 / 2, 1) +
         ' % of it, so throwing the second one away costs you ' + hFmt(100 * LAM2 / 2, 1) + ' %.']
      ];
      worked(read, WORK[mode][0], WORK[mode][1]);

      var RD = [
        ['Plot the four houses with area across and bedrooms up, each measured in units of its ' +
         'own spread so neither can dominate. They fall almost on a single diagonal line. That ' +
         'line is the thing worth naming: call it bigness.',
         'C is not a transformation someone handed you. It is a summary of this cloud, computed ' +
         'from these four houses — the diagonal says each feature has spread 1, the corner says ' +
         'the two features agree ' + hFmt(100 * H_R, 0) + ' % of the time.'],
        ['Here is what “a matrix is a machine” actually means, with no metaphor. Feed the ' +
         'arrow v into C and an arrow comes out. For almost every input the output points ' +
         'somewhere else. For v it does not: it comes out on the same line, just longer.',
         'That is the entire definition — Av = λv. There is no deeper content to it. The ' +
         'eigenvector is the direction the machine leaves alone, and λ is how much it stretched ' +
         'that direction. Here λ = ' + hFmt(LAM1, 2) + ', so bigness is stretched ' +
         hFmt(LAM1, 2) + '-fold while the perpendicular direction is barely touched at ' +
         hFmt(LAM2, 2) + '.'],
        ['This is the contrast that makes the definition mean something. u points along the ' +
         'area axis alone. Push it through C and it comes back tilted ' +
         hFmt(Math.atan2(H_R, 1) * 180 / Math.PI, 0) + '° upward, because a house with more area ' +
         'also tends to have more bedrooms and C knows it.',
         'A rotated output cannot be written as λu for any number λ, so u fails the test. Only ' +
         'two directions pass it here: the diagonal, and the line at right angles to it.'],
        ['Drop each house onto the bigness axis and you have replaced two numbers with one. The ' +
         'gold dots are what survives; the short grey lines are what you threw away.',
         'This is PCA, and it is why eigenvectors are worth knowing. Find the eigenvectors of ' +
         'the covariance matrix, keep the ones with the largest λ, and you have compressed the ' +
         'data along the directions where it actually varies. For these houses, two features ' +
         'have become one measured bigness coordinate.']
      ];
      h('p', 'rd-a', RD[mode][0], read);
      h('p', 'rd-b', RD[mode][1], read);
    }
    House.on(host, paint);
    cap(host, 'Select “open house sliders” and change your house. Its point moves through the ' +
              'diagram, while the reference directions stay fixed because houses A–D define them.');
  }

  /* ---- 10.10 · SVD: every house as a mix of four recipes ---- */
  var H_SVD = (function () {
    var S = H_SET.map(function (hh) {
      return hVec(hh).map(function (x, i) { return (x - H_MU[i]) / H_SD[i]; });
    });
    var mu = [0, 1, 2, 3].map(function (j) {
      return S.reduce(function (s, r) { return s + r[j]; }, 0) / S.length;
    });
    var X = S.map(function (r) { return r.map(function (x, j) { return x - mu[j]; }); });

    var i, j, k, A = [0, 1, 2, 3].map(function (a) {
      return [0, 1, 2, 3].map(function (b) {
        return X.reduce(function (s, r) { return s + r[a] * r[b]; }, 0);
      });
    });
    var V = [0, 1, 2, 3].map(function (a) {
      return [0, 1, 2, 3].map(function (b) { return a === b ? 1 : 0; });
    });
    /* Jacobi rotations: XᵀX is symmetric, so this converges in a few sweeps
       and needs no library. */
    for (var s = 0; s < 200; s++) {
      var p = 0, q = 1, off = 0;
      for (i = 0; i < 4; i++) for (j = i + 1; j < 4; j++) {
        off += A[i][j] * A[i][j];
        if (Math.abs(A[i][j]) > Math.abs(A[p][q])) { p = i; q = j; }
      }
      if (off < 1e-24) break;
      var th = 0.5 * Math.atan2(2 * A[p][q], A[q][q] - A[p][p]);
      var c = Math.cos(th), sn = Math.sin(th), a1, b1;
      for (k = 0; k < 4; k++) { a1 = A[k][p]; b1 = A[k][q]; A[k][p] = c * a1 - sn * b1; A[k][q] = sn * a1 + c * b1; }
      for (k = 0; k < 4; k++) { a1 = A[p][k]; b1 = A[q][k]; A[p][k] = c * a1 - sn * b1; A[q][k] = sn * a1 + c * b1; }
      for (k = 0; k < 4; k++) { a1 = V[k][p]; b1 = V[k][q]; V[k][p] = c * a1 - sn * b1; V[k][q] = sn * a1 + c * b1; }
    }
    var ev = [0, 1, 2, 3].map(function (a) {
      return { l: A[a][a], v: V.map(function (row) { return row[a]; }) };
    }).sort(function (a, b) { return b.l - a.l; });

    var sig = ev.map(function (e) { return Math.sqrt(Math.max(e.l, 0)); });
    /* sign is arbitrary in an SVD; pin each recipe so it points the way price
       rises, otherwise "recipe 1" flips between page loads and means nothing */
    var wS = H_W.map(function (x, i2) { return x * H_SD[i2]; });
    var Vt = ev.map(function (e) {
      var d = e.v.reduce(function (s2, x, i2) { return s2 + x * wS[i2]; }, 0);
      return d < 0 ? e.v.map(function (x) { return -x; }) : e.v;
    });
    var U = X.map(function (row) {
      return Vt.map(function (v, kk) {
        return sig[kk] > 1e-9 ? row.reduce(function (s2, x, i2) { return s2 + x * v[i2]; }, 0) / sig[kk] : 0;
      });
    });
    var tot = sig.reduce(function (s2, x) { return s2 + x * x; }, 0);
    return {
      X: X, mu: mu, sig: sig, Vt: Vt, U: U,
      energy: sig.map(function (x) { return x * x / tot; }),
      /* rebuild the whole table from the first k recipes, back in real units */
      rank: function (kk) {
        return X.map(function (_, i2) {
          return [0, 1, 2, 3].map(function (j2) {
            var s2 = 0, t;
            for (t = 0; t < kk; t++) s2 += U[i2][t] * sig[t] * Vt[t][j2];
            return (s2 + mu[j2]) * H_SD[j2] + H_MU[j2];
          });
        });
      }
    };
  })();

  function vizHouseSVD(host) {
    var W = 660, H = 300, svg = makeSVG(host, W, H);
    var g = el('g', {}, svg), keyb = keyBar(host), read = h('div', 'shapenote', null, host);
    var bar = ctl(host), view = 0, k = 1;
    button(bar, 'the three factors', function () { view = 0; paint(); });
    button(bar, 'what the recipes say', function () { view = 1; paint(); });
    button(bar, 'rebuild the houses', function () { view = 2; paint(); });
    slider(bar, 'recipes kept', 1, 3, 1, 1, function (v) {
      k = v; view = 2; paint();
      return k + ' of 4  ·  ' + hFmt(100 * H_SVD.energy.slice(0, k)
        .reduce(function (s, x) { return s + x; }, 0), 1) + ' % of the variation';
    });

    function paint() {
      g.innerHTML = ''; read.innerHTML = '';
      var i, j;

      if (view === 0) {
        var pu = panel(g, H_SVD.U.map(function (r) {
          return r.map(function (x) { return hFmt(x, 2); });
        }), { label: 'U · which houses', sub: '4 × 4', cell: 40, fs: 9.5 });
        var ps = panel(g, [0, 1, 2, 3].map(function (r) {
          return [0, 1, 2, 3].map(function (c) { return r === c ? hFmt(H_SVD.sig[r], 2) : '0'; });
        }), { label: 'Σ · how strong', sub: '4 × 4', cell: 40, fs: 9.5,
              hi: function (r, c) { return r === c ? 'sh-hi' : ''; } });
        var pv = panel(g, H_SVD.Vt.map(function (r) {
          return r.map(function (x) { return hFmt(x, 2); });
        }), { label: 'Vᵀ · the recipes', sub: '4 × 4', cell: 40, fs: 9.5 });
        lay([pu, sym(g, '×'), ps, sym(g, '×'), pv], 330, 140, 18);
        txt(g, 330, 30, 'X   =   U  Σ  Vᵀ', 'amber', 15);
        txt(g, 330, 272, 'one row of Vᵀ = one recipe over  area · bedrooms · age · km',
            'tx-faint', 11);
        worked(read, 'X = U Σ Vᵀ',
               'Read it right to left. Vᵀ says what the four recipes are, in the language of ' +
               'your own features. Σ says how much of the table each recipe accounts for. U says ' +
               'how much of each recipe each house takes.');
        h('p', 'rd-a', 'Vᵀ is not arbitrary and it is not in some other space. Each of its four ' +
                       'rows is a list of four numbers over area, bedrooms, age and km — the ' +
                       'same four slots your house has lived in since 10.2. A row is a recipe: ' +
                       '“so much area, so much age, so much distance”.', read);
        h('p', 'rd-b', 'Σ is diagonal, and the numbers on that diagonal are sorted biggest ' +
                       'first: ' + H_SVD.sig.map(function (x) { return hFmt(x, 2); }).join(', ') +
                       '. The last one is exactly zero, and 10.8 already told you why — four ' +
                       'houses measured against their own average can only span three ' +
                       'directions, so the fourth recipe has nothing left to describe.', read);

      } else if (view === 1) {
        var NAME = ['big, roomy, new and central', 'old but large and close in',
                    'the third direction', 'nothing left to explain'];
        /* the bars had to come left and shorten: at BX 250 with a 62px scale the
           names ran off the right edge and the last column's bar reached them */
        var BX = 205, BW = 260;
        for (i = 0; i < 4; i++) {
          var y0 = 48 + i * 62;
          txt(g, 187, y0 + 4, 'recipe ' + (i + 1), i < 2 ? 'amber' : 'tx-faint', 12, 'end');
          txt(g, 187, y0 + 19, hFmt(100 * H_SVD.energy[i], 1) + ' %', 'tx-faint', 10, 'end');
          for (j = 0; j < 4; j++) {
            var x = BX + j * (BW / 4), val = H_SVD.Vt[i][j], bw = Math.abs(val) * 50;
            var mid = x + BW / 8;
            el('line', { x1: mid, y1: y0 - 10, x2: mid, y2: y0 + 18,
                         stroke: css('--rule-line-soft'), 'stroke-width': 1 }, g);
            el('rect', { x: val >= 0 ? mid : mid - bw, y: y0 - 6, width: bw, height: 13, rx: 2,
                         fill: css(val >= 0 ? '--green' : '--pink'), opacity: i < 2 ? 0.85 : 0.4 }, g);
            if (i === 3) txt(g, mid, y0 + 32, H_FS[j], 'tx-faint', 10);
          }
          txt(g, 652, y0 + 4, NAME[i], i < 2 ? 'tx-dim' : 'tx-faint', 10.5, 'end');
        }
        txt(g, 330, 24, 'each bar is one weight of one recipe   ·   right = adds, left = subtracts',
            'tx-faint', 10.5);
        worked(read, 'recipe 1  =  ' + H_SVD.Vt[0].map(function (x, j2) {
                 return hFmt(x, 2) + '·' + H_FS[j2];
               }).join('  ').replace(/ −/g, ' −'),
               'Positive on area and bedrooms, negative on age and distance — which is the ' +
               'definition of a desirable house, discovered from the table rather than assumed. ' +
               'It carries ' + hFmt(100 * H_SVD.energy[0], 1) + ' % of everything that varies ' +
               'between these four houses.');
        h('p', 'rd-a', 'This is the answer to “what is Vᵀ?”. Recipe 1 says: go up in area and ' +
                       'bedrooms while going down in age and distance. Nobody wrote that rule — ' +
                       'it fell out of four rows of a table, and it is the axis along which ' +
                       'these houses actually differ.', read);
        h('p', 'rd-b', 'Recipe 2 is what is left after recipe 1 has had its say: up in area and ' +
                       'up in age, down in distance — old, large, close in. Recipe 3 is a ' +
                       'sliver, and recipe 4 is zero. The recipes are sorted from the direction ' +
                       'that reconstructs the most house variation to the direction that ' +
                       'reconstructs the least.', read);

      } else {
        var REC = H_SVD.rank(k);
        var pt = panel(g, H_SET.map(function (hh) {
          return hVec(hh).map(function (x) { return hFmt(x, 1); });
        }), { label: 'the real houses', cell: 40, fs: 10 });
        var pr = panel(g, REC.map(function (r) {
          return r.map(function (x) { return hFmt(x, 1); });
        }), { label: 'rebuilt from ' + k + ' recipe' + (k > 1 ? 's' : ''), cell: 40, fs: 10,
              hi: function (r, c) {
                var e = Math.abs(REC[r][c] - hVec(H_SET[r])[c]) / (H_SD[c] || 1);
                return e > 0.06 ? 'sh-hi' : '';
              } });
        lay([pt, sym(g, '≈', { size: 20 }), pr], 330, 140, 40);
        [pt, pr].forEach(function (p2) {
          var o = p2.g.getAttribute('transform').match(/translate\(([-\d.]+),([-\d.]+)\)/);
          for (j = 0; j < 4; j++)
            txt(g, parseFloat(o[1]) + 10 + j * 43 + 20,
                parseFloat(o[2]) + 20 + 4 * 40 + 3 * 3 + 15, H_FS[j], 'tx-faint', 9.5);
        });
        var err = 0, nrm = 0;
        H_SET.forEach(function (hh, i2) {
          hVec(hh).forEach(function (x, j2) {
            err += Math.pow((x - REC[i2][j2]) / H_SD[j2], 2);
            nrm += Math.pow((x - H_MU[j2]) / H_SD[j2], 2);
          });
        });
        txt(g, 330, 34, k + ' of 4 recipes kept  ·  ' +
            hFmt(100 * H_SVD.energy.slice(0, k).reduce(function (s, x) { return s + x; }, 0), 1) +
            ' % of the variation  ·  ' + hFmt(100 * Math.sqrt(err / nrm), 1) + ' % error left',
            'amber', 12.5);
        txt(g, 330, 274, 'highlighted cells are the ones still visibly wrong', 'tx-faint', 10.5);
        worked(read, 'house  ≈  score × strength × recipe,  summed over the ' + k +
               ' recipe' + (k > 1 ? 's' : '') + ' you kept',
               'House A from one recipe alone: ' + H_SVD.rank(1)[0].map(function (x, j2) {
                 return hFmt(x, 1) + ' ' + H_U[j2];
               }).join(',  ') + '.  The truth is ' + hVec(H_SET[0]).map(function (x, j2) {
                 return hFmt(x, 1) + ' ' + H_U[j2];
               }).join(',  ') + '.');
        var kept = 100 * H_SVD.energy.slice(0, k).reduce(function (s2, x) { return s2 + x; }, 0);
        var sqr = function (v) {
          return v.map(function (x) { return hFmt(x * x, 2); }).join(' + ');
        };
        var tot2 = H_SVD.sig.reduce(function (s2, x) { return s2 + x * x; }, 0);
        h('p', 'rd-a', 'Neither percentage is a figure you have to take on trust. The strengths ' +
                       'on the diagonal of Σ are ' +
                       H_SVD.sig.map(function (x) { return hFmt(x, 2); }).join(', ') +
                       ' — the same list the first view prints. Square them and they add up to ' +
                       hFmt(tot2, 2) + '; the ' + k + ' you kept account' + (k > 1 ? '' : 's') +
                       ' for ' +
                       sqr(H_SVD.sig.slice(0, k)) + ' of that, which is the ' + hFmt(kept, 1) +
                       ' %. The error left is the rebuilt table minus the real one, every ' +
                       'feature counted in units of its own spread so metres and rooms can be ' +
                       'added: √( ' + hFmt(err, 2) + ' / ' + hFmt(nrm, 2) + ' ) = ' +
                       hFmt(100 * Math.sqrt(err / nrm), 1) + ' %, where ' + hFmt(nrm, 2) +
                       ' is how far the real houses sit from their own average to begin with.', read);
        h('p', 'rd-b', 'For this house table, the tradeoff is now visible: keep fewer recipes ' +
                       'to store fewer numbers, and accept the highlighted measurement errors ' +
                       'that remain in the rebuilt houses.', read);
      }
      keys(keyb, view === 0
        ? [['--amber', 'the three factors and what each one carries']]
        : view === 1
          ? [['--green', 'the recipe adds this feature'],
             ['--pink', 'the recipe subtracts it'],
             ['--amber', 'the two recipes worth keeping'],
             ['--tx-faint', 'the two that carry almost nothing']]
          : [['cell', 'rebuilt cells still visibly wrong'],
             ['--amber', 'the variation kept and the error left']]);
    }
    paint();
    cap(host, 'The four houses are fixed here — a decomposition describes a table, so it needs ' +
              'a table that holds still.');
  }

  /* ---- 10.11 · where 0.3 comes from, and what each penalty does to it ---- */
  var H_JUNK = { day: [10, 2, 15, 6], resid: [5, -6, 6, -3] };
  H_JUNK.num = H_JUNK.day.reduce(function (s, x, i) { return s + x * H_JUNK.resid[i]; }, 0);
  H_JUNK.den = H_JUNK.day.reduce(function (s, x) { return s + x * x; }, 0);
  H_JUNK.a = H_JUNK.num / H_JUNK.den;

  function vizHouseL1L2(host) {
    var W = 660, H = 320, svg = makeSVG(host, W, H);
    var g = el('g', {}, svg), keyb = keyBar(host), read = h('div', 'shapenote', null, host);
    var bar = ctl(host), view = 0, al = 0, timer = null, A = H_JUNK.a;
    button(bar, 'where 0.3014 comes from', function () { view = 0; stop(); paint(); });
    button(bar, 'what the penalties do', function () { view = 1; paint(); });
    var sl = slider(bar, 'penalty strength α', 0, 0.6, 0.01, 0, function (v) {
      al = v; view = 1; paint();
      return 'α = ' + hFmt(al, 2);
    });
    /* slider() invokes its callback while constructing the control. The page
       must still open on the derivation that explains where 0.3014 came from. */
    view = 0;
    al = 0;
    button(bar, 'sweep α', function () {
      view = 1;
      if (timer) return stop();
      timer = setInterval(function () {
        if (!host.isConnected) return stop();
        al = al >= 0.6 ? 0 : Math.round((al + 0.01) * 100) / 100;
        sl.value = al; paint();
      }, 55);
    });
    function stop() { if (timer) clearInterval(timer); timer = null; }

    function lasso(x) { return Math.max(A - x, 0); }
    function ridge(x) { return A / (1 + x); }

    function paint() {
      g.innerHTML = ''; read.innerHTML = '';
      var i;

      if (view === 0) {
        var rows = H_SET.map(function (hh, i2) {
          return [hh.id, String(H_JUNK.day[i2]), hFmt(hPrice(hh), 1),
                  hFmt(hPrice(hh) + H_JUNK.resid[i2], 1),
                  (H_JUNK.resid[i2] > 0 ? '+' : '−') + Math.abs(H_JUNK.resid[i2]),
                  hFmt(H_JUNK.day[i2] * H_JUNK.resid[i2], 0)];
        });
        var HEAD = ['house', 'listed on day', 'model says', 'actually sold for',
                    'residual r', 'day × r'];
        var CX = [92, 190, 292, 412, 512, 596];
        for (i = 0; i < 6; i++) txt(g, CX[i], 40, HEAD[i], 'tx-faint', 10);
        el('line', { x1: 60, y1: 50, x2: 630, y2: 50, stroke: css('--rule-line-soft') }, g);
        rows.forEach(function (r, ri) {
          for (i = 0; i < 6; i++)
            txt(g, CX[i], 74 + ri * 28, r[i],
                i === 5 ? 'amber' : (i === 4 ? 'tx-soft' : 'tx-dim'), 12);
        });
        el('line', { x1: 60, y1: 74 + 4 * 28 - 16, x2: 630, y2: 74 + 4 * 28 - 16,
                     stroke: css('--rule-line-soft') }, g);
        txt(g, 512, 74 + 4 * 28 + 6, 'sum', 'tx-faint', 10);
        txt(g, 596, 74 + 4 * 28 + 6, hFmt(H_JUNK.num, 0), 'amber', 13);
        txt(g, 330, 248,
            'a  =  Σ day×r  /  Σ day²   =   ' + hFmt(H_JUNK.num, 0) + ' / ' +
            hFmt(H_JUNK.den, 0) + '   =   ' + hFmt(A, 4), 'gold', 15);
        txt(g, 330, 274, 'that is the 0.3014 — nobody chose it', 'tx-faint', 11);
        keys(keyb, [['--amber', 'the day × residual column, its sum, and the weight ' +
                                'that comes out of the division']]);
        worked(read, 'a  =  ( j · r ) / ( j · j )  =  ' + hFmt(H_JUNK.num, 0) + ' / ' +
               hFmt(H_JUNK.den, 0) + '  =  ' + hFmt(A, 4),
               'j is the junk column — the day of the month each house was listed. r is what the ' +
               'four-feature model got wrong on each house. Fitting one more feature to those ' +
               'leftovers is a single division, and it hands back 0.3014 k€ per day.');
        h('p', 'rd-a', 'Someone adds “day of the month it was listed” to your table. It cannot ' +
                       'possibly move a house price. But your model is not perfect — it is off ' +
                       'by +5, −6, +6 and −3 k€ on the four houses — and the listing days happen ' +
                       'to line up with those errors well enough that least squares gives the ' +
                       'column a weight of 0.3014 rather than 0.', read);
        h('p', 'rd-b', 'Nothing has gone wrong mechanically. Least squares did exactly what it ' +
                       'is for: it found the value that best explains the leftovers. With four ' +
                       'houses and enough columns you can explain anything, and the model will ' +
                       'then predict listing-day effects on houses it has never seen. That is ' +
                       'overfitting, and the next view is the standard repair.', read);
        return;
      }

      var box = { l: 78, t: 70, w: 218, h: 168 };
      var box2 = { l: 400, t: 70, w: 218, h: 168 };
      var XR = [-0.12, 0.46], YR = [0, 0.22];
      var f1 = new Frame(g, box, XR, YR), f2 = new Frame(g, box2, XR, YR);
      var data = function (x) { return 0.5 * (x - A) * (x - A); };

      [[f1, box, 'L1  ·  lasso', function (x) { return al * Math.abs(x); }, lasso(al), '--gold'],
       [f2, box2, 'L2  ·  ridge', function (x) { return 0.5 * al * x * x; }, ridge(al), '--accent']]
        .forEach(function (P) {
          var F = P[0], b = P[1], pen = P[3], wmin = P[4];
          F.axes('w', null);
          txt(g, b.l + b.w / 2, b.t - 20, P[2], 'amber', 13);
          el('line', { x1: F.X(0), y1: b.t, x2: F.X(0), y2: b.t + b.h,
                       stroke: css('--rule-line'), 'stroke-width': 1 }, g);
          F.path(data, 200, null, { stroke: css('--tx-faint'), fill: 'none',
                                    'stroke-width': 1.2, 'stroke-dasharray': '3 4' });
          F.path(pen, 200, null, { stroke: css('--rule-line'), fill: 'none',
                                   'stroke-width': 1.2, 'stroke-dasharray': '1 3' });
          F.path(function (x) { return data(x) + pen(x); }, 400, null,
                 { stroke: css(P[5]), fill: 'none', 'stroke-width': 2.2 });
          var yv = data(wmin) + pen(wmin);
          el('circle', { cx: F.X(wmin), cy: F.Y(Math.min(yv, YR[1])), r: 5,
                         fill: css(P[5]) }, g);
          el('line', { x1: F.X(wmin), y1: F.Y(Math.min(yv, YR[1])), x2: F.X(wmin),
                       y2: b.t + b.h, stroke: css(P[5]), 'stroke-width': 1,
                       'stroke-dasharray': '2 3' }, g);
          txt(g, F.X(wmin), b.t + b.h + 34,
              'w* = ' + (wmin < 0.0005 ? '0  (gone)' : hFmt(wmin, 3)),
              wmin < 0.0005 ? 'green' : 'tx-soft', 12.5);
          txt(g, F.X(A), b.t + b.h + 15, hFmt(A, 4), 'tx-faint', 9.5);
          el('line', { x1: F.X(A), y1: b.t + b.h, x2: F.X(A), y2: b.t + b.h + 4,
                       stroke: css('--tx-faint') }, g);
        });
      keys(keyb, [['--tx-faint', 'dashed: the data cost alone, pulling w towards 0.3014', 'dash'],
                  ['--rule-line', 'dotted: the penalty alone, pulling w towards 0', 'dot'],
                  ['--gold', 'solid: the two added together under L1, and where it bottoms out', 'line'],
                  ['--accent', 'solid: the same under L2', 'line']]);

      worked(read, 'minimise  ½(w − 0.3014)²  +  α·penalty(w)',
             'L1 penalty α|w| → w* = max(0.3014 − α, 0), so the weight hits exactly zero once ' +
             'α reaches 0.3014 and stays there. L2 penalty ½αw² → w* = 0.3014 / (1 + α), which ' +
             'shrinks towards zero and never arrives: at α = 0.6 it is still ' +
             hFmt(ridge(0.6), 3) + '.');
      h('p', 'rd-a', 'The dashed parabola is the data pulling the weight towards 0.3014 — that is ' +
                     'the number you just derived. The penalty pulls it towards zero, and how ' +
                     'hard it pulls is α, the one quantity on this figure that is not computed ' +
                     'from anything: it is the slider, and you set it. Where the two pulls ' +
                     'balance is where the solid curve bottoms out, and that is the weight you ' +
                     'actually get.', read);
      h('p', 'rd-b', 'The whole difference is the shape at w = 0. |w| has a corner there: its ' +
                     'slope jumps from −α to +α no matter how small α is, so as soon as α beats ' +
                     'the data’s pull of 0.3014, zero becomes the minimum outright and the ' +
                     'feature is deleted. w² is smooth there, with slope exactly 0, so it can ' +
                     'never out-pull anything — it only ever squeezes. That single fact is why ' +
                     'lasso selects features and ridge merely shrinks them.', read);
    }
    paint();
    cap(host, 'Drag α, or press sweep. Once α passes 0.3014, watch the left-hand dot reach zero and stop; the ' +
              'right-hand one never quite gets there.');
  }

  var REG = {
    psi: vizPSI,
    canary: vizCanary,
    pit: vizPIT,
    kvcache: vizKVCache,
    attention: vizAttention,
    conv: vizConv,
    cosine: vizCosine,
    matrix2d: vizMatrix2D,
    bayes: vizBayes,
    minima: vizMinima,
    derivative: vizDerivative,
    lr: vizLR,
    entropy: vizEntropy,
    kl: vizKL,
    biasvar: vizBiasVar,
    lagrange: vizLagrange,
    activations: vizActivations,
    shape: vizShape,
    'tensor-ladder': vizTensorLadder,
    'tensor-axes': vizTensorAxes,
    matmul: vizMatmul,
    'dot-ratings': vizDotRatings,
    norms: vizNorms,
    'norm-actions': vizNormActions,
    transpose: vizTranspose,
    slope: vizSlope,
    partial: vizPartial,
    'chain-rule': vizChain,
    saddle: vizSaddle,
    'gd-step': vizGDStep,
    clipping: vizClip,
    surprise: vizSurprise,
    crossentropy: vizCrossEntropy,
    clt: vizCLT,
    expectation: vizExpectation,
    momentum: vizMomentum,
    scalar: vizScalar,
    vector: vizVector,
    'cos-mini': vizCosMini,
    inverse: vizInverse,
    rank: vizRank,
    eigen: vizEigen,
    determinant: vizDet,
    svd: vizSVD,
    orthogonal: vizOrtho,
    'fn-machine': vizFnMachine,
    'deriv-mini': vizDerivMini,
    'partial-mini': vizPartialMini,
    stationary: vizStationary,
    minimum: vizMinimumCard,
    'second-deriv': vizSecondDeriv,
    hessian: vizHessian,
    jacobian: vizJacobian,
    'chain-mini': vizChainMini,
    convex: vizConvex,
    diffable: vizDiffable,
    'house-panel': vizHousePanel,
    ladder: vizLadder,
    'house-axes': vizHouseAxes,
    'house-price': vizHousePrice,
    'house-dist': vizHouseDist,
    'house-shape': vizHouseShape,
    'house-table': vizHouseTable,
    'house-solve': vizHouseSolve,
    'house-singular': vizHouseSingular,
    'house-eigen': vizHouseEigen,
    'house-svd': vizHouseSVD,
    'house-l1l2': vizHouseL1L2
  };

  /* Module 12: each small figure keeps the story, numbers and interpretation together. */
  function p12Figure(host, height, title) {
    var svg = makeSVG(host, 400, height);
    svg.setAttribute('role', 'img');
    svg.setAttribute('aria-label', title);
    el('title', {}, svg).textContent = title;
    return { svg: svg, read: h('div', 'shapenote', null, host) };
  }
  function p12Bars(svg, values, labels, opt) {
    opt = opt || {};
    var top = opt.top || 32, height = opt.height || 105;
    var max = opt.max || 1, left = 42, width = 316, step = width / values.length;
    el('line', { x1: left, y1: top + height, x2: left + width, y2: top + height, class: 's-axis' }, svg);
    return values.map(function (v, i) {
      var x = left + i * step + step / 2, bh = v / max * height;
      var rect = el('rect', { x: x - step * 0.3, y: top + height - bh,
        width: step * 0.6, height: bh, rx: 3, fill: css('--' + (opt.colour || 'amber')) }, svg);
      txt(svg, x, top + height - bh - 7, opt.format ? opt.format(v) : v.toFixed(2), 'tx-dim', 13);
      txt(svg, x, top + height + 20, labels[i], 'tx-dim', 13);
      return rect;
    });
  }
  function p12Table(svg, headings, rows, top) {
    var left = 12, width = 376, cw = width / headings.length, rh = 34;
    headings.forEach(function (v, i) { txt(svg, left + cw * (i + 0.5), top + 17, v, 'amber', 13); });
    rows.forEach(function (row, r) {
      row.forEach(function (v, c) {
        el('rect', { x: left + c * cw + 1, y: top + (r + 1) * rh,
          width: cw - 2, height: rh - 2, rx: 3, class: 'sh-cell sh-front' }, svg);
        txt(svg, left + cw * (c + 0.5), top + (r + 1) * rh + 22, v, 'tx-dim', 13);
      });
    });
  }
  function p12Random(host) {
    var f = p12Figure(host, 180, 'A die outcome is mapped to a numerical value X.');
    txt(f.svg, 200, 22, 'Six possible faces → six possible values of X', 'tx-dim', 13);
    var pips = [[[0, 0]], [[-1, -1], [1, 1]], [[-1, -1], [0, 0], [1, 1]],
      [[-1, -1], [1, -1], [-1, 1], [1, 1]],
      [[-1, -1], [1, -1], [0, 0], [-1, 1], [1, 1]],
      [[-1, -1], [1, -1], [-1, 0], [1, 0], [-1, 1], [1, 1]]];
    pips.forEach(function (dots, i) {
      var x = 63 + i * 46;
      el('rect', { x: x, y: 38, width: 43, height: 43, rx: 5, class: 'sh-cell sh-front' }, f.svg);
      dots.forEach(function (p) { el('circle', { cx: x + 21.5 + p[0]*10, cy: 59.5 + p[1]*10,
        r: 3, fill: css('--amber') }, f.svg); });
    });
    var numbers = panel(f.svg, [[1, 2, 3, 4, 5, 6]], { cell: 43, bracket: false });
    numbers.move(63, 105);
    txt(f.svg, 200, 97, 'X records the face as a number', 'amber', 13);
    worked(f.read, 'P(X = x) = favourable faces / all faces', 'P(X = 4) = 1 / 6 ≈ 0.167');
    h('p', 'rd-a', 'Before the roll, X can be any of these six numbers. After a four appears, its observed value is x = 4. P means probability: a fair die gives that outcome one chance in six.', f.read);
  }
  function p12Distribution(host) {
    var f = p12Figure(host, 180, 'Probabilities for three travel choices: walk 0.2, bus 0.5, bike 0.3.');
    txt(f.svg, 200, 20, 'Tomorrow’s travel choice: probability per option', 'tx-dim', 13);
    p12Bars(f.svg, [0.2, 0.5, 0.3], ['walk', 'bus', 'bike'], { max: 0.6 });
    worked(f.read, 'Σ P(X = x) = 1   (Σ means add all options)', '0.20 + 0.50 + 0.30 = 1.00');
    h('p', 'rd-a', 'Half the probability belongs to the bus. The three bars are the full distribution; choosing only the tallest bar loses how uncertain the choice is.', f.read);
    h('p', 'rd-b', 'For a continuous quantity such as journey time, probability is an area over an interval instead of a bar at one exact value. A density of 0.1 per minute over 2 minutes gives area 0.1 × 2 = 0.2.', f.read);
  }
  function p12Variance(host) {
    var f = p12Figure(host, 190, 'Values 1, 3, 5 have mean 3, deviations minus 2, 0, 2, and squared deviations 4, 0, 4.');
    p12Table(f.svg, ['value x', 'x − mean', 'squared gap'], [['1', '−2', '4'], ['3', '0', '0'], ['5', '+2', '4']], 9);
    txt(f.svg, 200, 173, 'All three values are equally likely; mean = 3', 'amber', 13);
    worked(f.read, 'Var(X) = E[(X − μ)²];   σ = √Var(X)', 'μ = (1 + 3 + 5)/3 = 3\nVar = (4 + 0 + 4)/3 = 8/3 ≈ 2.67\nσ = √(8/3) ≈ 1.63');
    h('p', 'rd-a', 'μ (mu) is the mean; σ (sigma) is the standard deviation. Squaring stops the −2 and +2 gaps cancelling. If values are minutes, variance is in minutes squared and standard deviation is in minutes.', f.read);
    h('p', 'rd-b', 'This is the variance of the stated three-outcome distribution. Estimating an unknown population variance from a sample uses a different correction: divide by n − 1.', f.read);
  }
  function p12Covariance(host) {
    var f = p12Figure(host, 205, 'Three paired points on an increasing line show positive covariance and correlation one.');
    var F = new Frame(f.svg, { l: 56, t: 33, w: 275, h: 122 }, [0, 4], [0, 8]);
    F.axes('study hours x', 'score y');
    [[1, 2], [2, 4], [3, 6]].forEach(function (p) {
      el('circle', { cx: F.X(p[0]), cy: F.Y(p[1]), r: 5, fill: css('--amber') }, f.svg);
      txt(f.svg, F.X(p[0]) + 10, F.Y(p[1]) - 7, '(' + p.join(', ') + ')', 'tx-dim', 13, 'start');
    });
    txt(f.svg, 200, 201, 'Higher x always comes with higher y here', 'amber', 13);
    worked(f.read, 'Cov(X,Y) = E[(X − μx)(Y − μy)]', 'μx = 2; μy = 4\nCov = ((−1)(−2) + 0×0 + 1×2)/3 = 4/3');
    worked(f.read, 'Correlation ρ = Cov(X,Y)/(σx σy)', 'ρ = (4/3)/(√(2/3) × √(8/3)) = 1');
    h('p', 'rd-a', 'Below-average hours pair with below-average scores; above pairs with above. Both products are positive. Correlation 1 means these points sit exactly on a rising straight line. It does not prove that more study caused the scores.', f.read);
  }
  function p12Joint(host) {
    var f = p12Figure(host, 195, 'Twenty journeys cross-tabulated by rain and bus use. Six rainy journeys use the bus.');
    p12Table(f.svg, ['20 journeys', 'bus', 'no bus', 'total'], [['rain', '6', '2', '8'], ['dry', '4', '8', '12'], ['total', '10', '10', '20']], 10);
    worked(f.read, 'Joint: P(bus AND rain) = matching / all', '6 / 20 = 0.30');
    worked(f.read, 'Marginal: P(bus) = bus total / all', '(6 + 4) / 20 = 0.50');
    worked(f.read, 'Conditional: P(bus | rain) = both / rain', '6 / 8 = 0.75');
    h('p', 'rd-a', 'Joint picks a cell. Marginal adds down the bus column. Conditional keeps only the rain row, so the denominator shrinks from 20 to 8. The vertical bar | means “given”.', f.read);
  }
  function p12Independence(host) {
    var f = p12Figure(host, 188, 'Four equally likely results of two independent fair coin flips.');
    p12Table(f.svg, ['first coin', 'second H', 'second T'], [['H', 'HH · 1/4', 'HT · 1/4'], ['T', 'TH · 1/4', 'TT · 1/4']], 18);
    txt(f.svg, 200, 158, 'Each row still has one H and one T for coin 2', 'amber', 13);
    worked(f.read, 'P(second H | first H) = P(second H)', '(1/4)/(1/2) = 1/2');
    worked(f.read, 'P(first H AND second H) = P(H) × P(H)', '1/2 × 1/2 = 1/4');
    h('p', 'rd-a', 'Learning the first coin is heads removes the bottom row. Within the remaining row, the second coin is still half heads. Knowledge of one coin did not change the other coin’s chances.', f.read);
  }
  function p12Bayes(host) {
    var f = p12Figure(host, 210, 'Of 100 messages, 10 are spam. The word prize appears in 8 spam and 9 normal messages.');
    p12Table(f.svg, ['100 messages', 'says “prize”', 'no “prize”'], [['spam: 10', '8', '2'], ['normal: 90', '9', '81']], 12);
    txt(f.svg, 200, 154, 'Keep the “prize” column: 8 spam + 9 normal', 'amber', 13);
    txt(f.svg, 200, 184, '8 / 17 spam → 47.1%', 'tx-dim', 16);
    worked(f.read, 'P(spam | prize) = P(prize | spam)P(spam)/P(prize)', '(0.80 × 0.10) / 0.17 = 8/17 ≈ 0.471');
    h('p', 'rd-a', 'Before reading the word, 10% of messages were spam (the prior). Spam says “prize” 80% of the time (the likelihood). Among the 17 messages saying “prize”, only 8 are spam: the updated probability, or posterior, is 47.1%, not 80%.', f.read);
  }
  function p12Beliefs(host) {
    var f = p12Figure(host, 195, 'Update belief between a fair coin and a heads-biased coin after observing two heads.');
    p12Table(f.svg, ['coin model', 'prior', 'P(HH)', 'product'], [['p = 0.5', '0.50', '0.25', '0.125'], ['p = 0.8', '0.50', '0.64', '0.320']], 9);
    txt(f.svg, 200, 155, 'Normalise the products: divide by their total', 'amber', 13);
    txt(f.svg, 200, 181, 'Fair: 28.1%        Heads-biased: 71.9%', 'tx-dim', 15);
    worked(f.read, 'P(model | HH) = prior × likelihood / total', 'P(p = 0.8 | HH) = 0.320/(0.125 + 0.320) ≈ 0.719');
    h('p', 'rd-a', 'p is a coin’s chance of heads. We consider only these two possible coins, initially equally plausible. Seeing HH is more likely with p = 0.8, so its share of our belief grows. A likelihood scores the observed data under a model; it is not itself a probability over models.', f.read);
  }
  function p12MLE(host) {
    var f = p12Figure(host, 190, 'Likelihood of the observed coin sequence heads heads heads tails as the heads probability varies.');
    var F = new Frame(f.svg, { l: 45, t: 30, w: 304, h: 112 }, [0, 1], [0, 0.12]);
    F.axes('heads probability p', 'P(HHHT)');
    F.path(function (p) { return p * p * p * (1 - p); });
    [0, 0.5, 1].forEach(function (p) { txt(f.svg, F.X(p), 158, String(p), 'tx-faint', 12); });
    var dot = el('circle', { r: 5, fill: css('--amber') }, f.svg);
    var value = txt(f.svg, 200, 20, '', 'amber', 13);
    var math = worked(f.read, 'L(p) = p³(1 − p)', '');
    var result = h('div', 'w', '', math);
    function draw(p) {
      var likelihood = p * p * p * (1 - p);
      dot.setAttribute('cx', F.X(p)); dot.setAttribute('cy', F.Y(likelihood));
      value.textContent = 'Observed data: H H H T · L = ' + likelihood.toFixed(4);
      result.textContent = p.toFixed(2) + '³ × (1 − ' + p.toFixed(2) + ') = ' + likelihood.toFixed(4);
      return p.toFixed(2);
    }
    slider(ctl(host), 'Try heads probability p', 0, 1, 0.01, 0.75, draw);
    h('p', 'rd-a', 'The curve peaks at p = 3/4 = 0.75: three heads out of four flips. This is the maximum likelihood estimate. It picks the coin probability that gives this observed sequence the largest likelihood, 0.1055. It does not claim the sequence is certain.', f.read);
  }
  function p12MAP(host) {
    var f = p12Figure(host, 202, 'For heads heads heads tails, likelihood peaks at 0.75 while the posterior with a Beta(2,2) prior peaks at two thirds.');
    var F = new Frame(f.svg, { l: 42, t: 40, w: 310, h: 111 }, [0, 1], [0, 1.05]);
    F.axes('heads probability p', 'relative height');
    F.path(function (p) { return Math.pow(p, 3) * (1 - p) / (27 / 256); }, 150, null, { style: 'stroke:' + css('--amber') });
    F.path(function (p) { return Math.pow(p, 4) * Math.pow(1 - p, 2) / (16 / 729); }, 150, null, { style: 'stroke:' + css('--green') });
    txt(f.svg, 200, 18, 'Amber: likelihood · Green: posterior', 'tx-dim', 13);
    txt(f.svg, 200, 194, 'Each curve scaled to its own peak = 1', 'tx-faint', 12);
    worked(f.read, 'Posterior ∝ likelihood × prior', 'p³(1 − p) × p(1 − p) = p⁴(1 − p)²');
    worked(f.read, 'MAP = peak of the posterior', 'pMAP = 4/(4 + 2) = 2/3 ≈ 0.667\npMLE = 3/4 = 0.75');
    h('p', 'rd-a', 'Use the same HHHT data as MLE. A Beta(2,2) prior has shape p(1 − p): it favours probabilities near 0.5 over extreme coins. Multiplying it in shifts the preferred p from 0.75 to 0.667. ∝ means “has the same shape up to a constant multiplier”.', f.read);
  }
  function p12IID(host) {
    var f = p12Figure(host, 180, 'Drawing balls with replacement from a bag containing one red and one blue ball gives independent draws with the same probabilities.');
    p12Table(f.svg, ['draw', 'P(red)', 'what happens after'], [['1', '1/2', 'put ball back'], ['2', '1/2', 'put ball back'], ['3', '1/2', 'put ball back']], 7);
    worked(f.read, 'P(R₁,R₂,R₃) = P(R₁)P(R₂)P(R₃)', '1/2 × 1/2 × 1/2 = 1/8');
    h('p', 'rd-a', 'A bag has one red and one blue ball. Replace and mix after each draw: every draw uses the same 50/50 distribution, and the previous colour gives no information about the next. Those are the two separate promises in i.i.d.', f.read);
    h('p', 'rd-b', 'Without replacement, red first forces blue second: independence fails. Changing to a bag with three red balls and one blue changes P(red) to 3/4: identical distribution fails.', f.read);
  }
  function p12LLN(host) {
    var f = p12Figure(host, 200, 'A reproducible illustrative sequence of fair coin flips has a running heads fraction that fluctuates near one half.');
    var F = new Frame(f.svg, { l: 44, t: 30, w: 310, h: 120 }, [1, 120], [0, 1]);
    F.axes('number of flips n', 'heads fraction');
    el('line', { x1: F.X(1), y1: F.Y(0.5), x2: F.X(120), y2: F.Y(0.5), stroke: css('--green'), 'stroke-dasharray': '4 4' }, f.svg);
    var seed = 137, heads = 0, path = '';
    for (var n = 1; n <= 120; n++) {
      seed = (Math.imul(1664525, seed) + 1013904223) >>> 0;
      heads += seed / 4294967296 < 0.5 ? 1 : 0;
      path += (n === 1 ? 'M' : 'L') + F.X(n).toFixed(2) + ',' + F.Y(heads / n).toFixed(2);
    }
    el('path', { d: path, fill: 'none', stroke: css('--amber'), 'stroke-width': 2 }, f.svg);
    txt(f.svg, 200, 18, 'Dashed line: true heads probability = 0.5', 'tx-dim', 13);
    txt(f.svg, 200, 194, 'One fixed illustrative run; wiggles can move away', 'tx-faint', 12);
    worked(f.read, 'Running average = heads so far / flips so far', 'At n = 120: ' + heads + '/120 = ' + (heads / 120).toFixed(3));
    h('p', 'rd-a', 'A heads flip contributes 1; tails contributes 0. Averaging these numbers estimates the coin’s heads probability. With independent draws from the same distribution and a finite mean, the average settles toward that mean over the long run. It need not improve on every flip, or land exactly on 0.5.', f.read);
  }
  function p12Stochastic(host) {
    var f = p12Figure(host, 192, 'Four examples have gradients minus two, zero, two and four. Two different mini-batches estimate gradients minus one and three instead of the full average one.');
    p12Table(f.svg, ['examples', 'their gradients', 'average'], [['all four', '−2, 0, 2, 4', '1'], ['batch A', '−2, 0', '−1'], ['batch B', '2, 4', '3']], 9);
    worked(f.read, 'Full gradient = sum of individual gradients / N', '(−2 + 0 + 2 + 4)/4 = 1');
    worked(f.read, 'Batch gradient = sum within batch / batch size', 'A: (−2 + 0)/2 = −1\nB: (2 + 4)/2 = 3');
    h('p', 'rd-a', 'A gradient tells a weight which direction increases the loss, and how strongly. Training subtracts a small multiple of it. Randomly selecting fewer examples makes this signal noisy: batch A even points in the opposite direction to the full average.', f.read);
    h('p', 'rd-b', 'Uniformly sampled mini-batches give the full gradient on average, although any one batch can disagree. This saves computation. Noise can help exploration, but it does not guarantee a better solution.', f.read);
  }
  function p12Common(host) {
    var f = p12Figure(host, 240, 'Select a distribution to see its probabilities or density with a worked numerical example.');
    var plots = [], readings = [], selected = 0;
    var cases = [
      { name: 'Bernoulli', values: [0.7, 0.3], labels: ['0: tails', '1: heads'], max: 1,
        formula: 'P(X = 1) = p; P(X = 0) = 1 − p', numbers: 'p = 0.3 → P(heads) = 0.3; P(tails) = 0.7',
        text: 'One yes/no event. X = 1 means heads, X = 0 means tails. There are exactly two possible outcomes.' },
      { name: 'Binomial', values: [0.25, 0.5, 0.25], labels: ['0 heads', '1 head', '2 heads'], max: 0.6,
        formula: 'P(k heads) = choose(n,k) pᵏ(1 − p)ⁿ⁻ᵏ', numbers: 'n = 2, p = 0.5 → P(1 head) = 2 × 0.5 × 0.5 = 0.5',
        text: 'Count heads in two independent fair flips. HH, HT, TH, TT are equally likely. Two of the four contain one head; choose(2,1) = 2 counts those arrangements.' },
      { name: 'Categorical', values: [0.2, 0.5, 0.3], labels: ['walk', 'bus', 'bike'], max: 0.6,
        formula: 'P(X = category i) = pᵢ; Σpᵢ = 1', numbers: 'P(bus) = 0.5; 0.2 + 0.5 + 0.3 = 1',
        text: 'One choice from several named options. The category labels need not be numbers. These are the same travel probabilities as the Distribution card.' },
      { name: 'Gaussian', gaussian: true, formula: 'f(x) = exp(−x²/2) / √(2π), for μ = 0, σ = 1',
        numbers: 'f(0) = 1/√(2π) ≈ 0.399; P(−1 ≤ X ≤ 1) ≈ 0.683',
        text: 'A continuous bell-shaped density. The centre is μ = 0 and standard deviation σ = 1. Height 0.399 is a density, not P(X = 0). The shaded area between −1 and 1 gives about 68.3% probability.' },
      { name: 'Poisson', values: [1, 1, 0.5, 1 / 6, 1 / 24].map(function (v) { return v / Math.E; }), labels: ['0', '1', '2', '3', '4'], max: 0.5,
        formula: 'P(X = k) = exp(−λ) λᵏ / k!', numbers: 'λ = 1 → P(2 events) = exp(−1) × 1²/2 ≈ 0.184',
        text: 'Count events in a fixed window under a constant-rate, independent-arrival model. λ = 1 means one event per window on average; k! multiplies integers 1 through k, so 2! = 2. Counts above 4 remain possible and are omitted here.' }
    ];
    cases.forEach(function (item, i) {
      var g = el('g', {}, f.svg); plots.push(g);
      txt(g, 200, 23, item.name + (item.gaussian ? ': probability is area' : ': probability per outcome'), 'amber', 14);
      if (item.gaussian) {
        var F = new Frame(g, { l: 43, t: 49, w: 310, h: 120 }, [-3, 3], [0, 0.44]);
        F.axes('value x', 'density');
        var d = 'M' + F.X(-1) + ',' + F.Y(0);
        for (var x = -1; x <= 1.001; x += 0.025) d += 'L' + F.X(x) + ',' + F.Y(Math.exp(-x*x/2)/Math.sqrt(2*Math.PI));
        d += 'L' + F.X(1) + ',' + F.Y(0) + 'Z';
        el('path', { d: d, fill: css('--amber'), opacity: 0.3 }, g);
        F.path(function (v) { return Math.exp(-v*v/2)/Math.sqrt(2*Math.PI); });
        [-1, 0, 1].forEach(function (v) { txt(g, F.X(v), 193, String(v), 'tx-dim', 13); });
        txt(g, 200, 223, 'Shaded area ≈ 0.683', 'amber', 13);
      } else p12Bars(g, item.values, item.labels, { top: 56, height: 117, max: item.max });
      var read = h('div', null, null, f.read); readings.push(read);
      worked(read, item.formula, item.numbers); h('p', 'rd-a', item.text, read);
      if (i > 0) { g.style.display = 'none'; read.hidden = true; }
    });
    var buttons = [], controls = ctl(host);
    cases.forEach(function (item, i) {
      var b = button(controls, item.name, function () {
        plots[selected].style.display = 'none'; readings[selected].hidden = true;
        selected = i; plots[i].style.display = ''; readings[i].hidden = false;
        buttons.forEach(function (btn, j) { btn.setAttribute('aria-pressed', String(i === j)); btn.className = i === j ? 'on' : ''; });
        f.svg.setAttribute('aria-label', item.name + '. ' + item.numbers + '. ' + item.text);
      });
      b.setAttribute('aria-pressed', String(i === 0)); b.className = i === 0 ? 'on' : ''; buttons.push(b);
    });
  }

  /* REG assignments: insert after the existing registry object is declared. */
  REG['p12-random'] = p12Random;
  REG['p12-distribution'] = p12Distribution;
  REG['p12-variance'] = p12Variance;
  REG['p12-covariance'] = p12Covariance;
  REG['p12-joint'] = p12Joint;
  REG['p12-independence'] = p12Independence;
  REG['p12-bayes'] = p12Bayes;
  REG['p12-beliefs'] = p12Beliefs;
  REG['p12-mle'] = p12MLE;
  REG['p12-map'] = p12MAP;
  REG['p12-iid'] = p12IID;
  REG['p12-lln'] = p12LLN;
  REG['p12-stochastic'] = p12Stochastic;
  REG['p12-common'] = p12Common;

  /* Module 13: compact worked information-theory cards. All logs here use bits. */
  function p13Scene(host, height, title) {
    var svg = makeSVG(host, 400, height);
    svg.setAttribute('role', 'img');
    svg.setAttribute('aria-label', title);
    el('title', {}, svg).textContent = title;
    var read = h('div', 'shapenote', null, host);
    var math = worked(read, '', ' ');
    var explanation = h('p', 'rd-a', '', read);
    return { svg: svg, read: read, controls: ctl(host),
      update: function (formula, numbers, meaning) {
        math.querySelector('.f').textContent = formula;
        math.querySelector('.w').textContent = numbers;
        explanation.textContent = meaning;
      } };
  }
  function p13Choices(scene, labels, render) {
    var buttons = labels.map(function (label, i) {
      var b = button(scene.controls, label, function () { choose(i); });
      b.type = 'button';
      return b;
    });
    function choose(i) {
      buttons.forEach(function (b, j) { b.setAttribute('aria-pressed', String(i === j)); });
      render(i);
    }
    choose(0);
  }
  /* Two probability bars, with text values so colour is never the only key. */
  function p13Bars(svg, y, heading, labels) {
    txt(svg, 20, y, heading, 'chalk', 15, 'start');
    var bars = labels.map(function (label, i) {
      var yy = y + 20 + i * 34;
      txt(svg, 20, yy + 16, label, 'tx-dim', 14, 'start');
      el('rect', { x: 110, y: yy, width: 220, height: 22, rx: 3,
        fill: css('--tx-faint'), opacity: 0.12 }, svg);
      var rect = el('rect', { x: 110, y: yy, width: 0, height: 22, rx: 3,
        fill: css(i ? '--accent' : '--amber') }, svg);
      var value = txt(svg, 380, yy + 16, '', 'chalk', 14, 'end');
      return { rect: rect, value: value };
    });
    return function (p) {
      bars.forEach(function (bar, i) {
        bar.rect.setAttribute('width', 220 * p[i]);
        bar.value.textContent = (100 * p[i]).toFixed(0) + '%';
      });
    };
  }
  function p13Entropy(p) {
    return p.reduce(function (sum, x) { return sum + (x ? -x * Math.log2(x) : 0); }, 0);
  }
  function p13Bits(host) {
    var scene = p13Scene(host, 182, 'Two or four equally likely answers: the same surprise in bits and nats.');
    txt(scene.svg, 200, 25, 'Find one equally likely answer', 'chalk', 16);
    var tiles = [];
    for (var i = 0; i < 4; i++) {
      var g = el('g', {}, scene.svg);
      el('rect', { x: 24 + i * 94, y: 43, width: 70, height: 56, rx: 5,
        class: 'sh-cell sh-front' }, g);
      txt(g, 59 + i * 94, 66, String.fromCharCode(65 + i), 'amber', 18);
      tiles.push({ g: g, code: txt(g, 59 + i * 94, 88, '', 'chalk', 14) });
    }
    var conversion = txt(scene.svg, 200, 137, '', 'chalk', 17);
    var questions = txt(scene.svg, 200, 164, '', 'tx-dim', 14);
    p13Choices(scene, ['Four answers', 'Two answers'], function (mode) {
      var n = mode ? 2 : 4, bits = Math.log2(n), nats = Math.log(n);
      tiles.forEach(function (tile, j) {
        tile.g.setAttribute('visibility', j < n ? 'visible' : 'hidden');
        tile.code.textContent = mode ? String(j) : ('0' + j.toString(2)).slice(-2);
      });
      conversion.textContent = bits + ' bits = ' + nats.toFixed(3) + ' nats';
      questions.textContent = bits + ' yes/no ' + (bits === 1 ? 'answer identifies' : 'answers identify') + ' the letter';
      scene.update('Ibits = −log₂(p); Inats = −ln(p); bits = nats / ln(2)',
        'p = 1/' + n + ': −log₂(1/' + n + ') = ' + bits + '; −ln(1/' + n + ') = ' + nats.toFixed(3),
        'The digits under each letter are its yes/no code. A logarithm asks “what power gives this number?” Here 2^' + bits + ' = ' + n + '. Bits and nats measure the same surprise, just as metres and feet measure the same length.');
    });
  }
  function p13EntropyCard(host) {
    var scene = p13Scene(host, 170, 'Coin outcome probabilities and their average surprise, called entropy.');
    var bars = p13Bars(scene.svg, 25, 'Coin probabilities', ['Heads', 'Tails']);
    var answer = txt(scene.svg, 200, 153, '', 'chalk', 17);
    p13Choices(scene, ['Fair coin', '75% heads', 'Always heads'], function (mode) {
      var p = [[0.5, 0.5], [0.75, 0.25], [1, 0]][mode], entropy = p13Entropy(p);
      bars(p); answer.textContent = 'Average surprise = ' + entropy.toFixed(3) + ' bits';
      scene.update('H(P) = −Σ p log₂(p)',
        mode === 2 ? 'H = −1 × log₂(1) + 0 = 0 bits' :
          'H = −' + p[0] + ' × log₂(' + p[0] + ') − ' + p[1] + ' × log₂(' + p[1] + ') = ' + entropy.toFixed(3) + ' bits',
        mode === 2 ? 'Heads is guaranteed: the result tells you nothing new. An impossible outcome contributes zero; we do not evaluate log(0).' :
        mode === 1 ? 'Heads gives 0.415 bits of surprise, tails gives 2 bits. Weight those by how often they occur: 0.75 × 0.415 + 0.25 × 2 = 0.811 bits on average.' :
        'Either outcome gives 1 bit of surprise. Averaging gives 0.5 × 1 + 0.5 × 1 = 1 bit. Biasing this coin makes its result easier to predict.');
    });
  }
  function p13KL(host) {
    var scene = p13Scene(host, 300, 'True weather and forecast probabilities, followed by unavoidable and extra prediction cost.');
    var truth = p13Bars(scene.svg, 24, 'Reality P', ['Sun', 'Rain']);
    var forecast = p13Bars(scene.svg, 137, 'Forecast Q', ['Sun', 'Rain']);
    truth([0.5, 0.5]);
    var result = txt(scene.svg, 200, 275, '', 'chalk', 16);
    p13Choices(scene, ['Biased forecast', 'Correct forecast'], function (mode) {
      var q = mode ? [0.5, 0.5] : [0.75, 0.25];
      var ce = -0.5 * Math.log2(q[0]) - 0.5 * Math.log2(q[1]);
      forecast(q);
      result.textContent = ce.toFixed(3) + ' total = 1 unavoidable + ' + (ce - 1).toFixed(3) + ' extra';
      scene.update('D(P‖Q) = Σ p log₂(p/q) = H(P,Q) − H(P)',
        'D = 0.5 × log₂(0.5/' + q[0] + ') + 0.5 × log₂(0.5/' + q[1] + ') = ' + (ce - 1).toFixed(3) + ' bits',
        mode ? 'The forecast now matches the true 50/50 weather. Its extra cost is zero, although weather still has 1 bit of uncertainty.' :
        'Rain happens half the time, but your forecast gives it only 25%. That mismatch costs 0.208 extra bits per day on average. One term in KL can be negative; the complete weighted sum cannot.');
    });
  }
  function p13Perplexity(host) {
    var scene = p13Scene(host, 206, 'Three actual words with predicted probabilities, surprise values, and the equivalent number of equally likely choices.');
    txt(scene.svg, 200, 24, 'Actual next words in three predictions', 'chalk', 15);
    var tiles = [];
    ['the', 'cat', 'sat'].forEach(function (word, i) {
      var x = 18 + i * 128;
      el('rect', { x: x, y: 42, width: 108, height: 109, rx: 5, class: 'sh-cell sh-front' }, scene.svg);
      txt(scene.svg, x + 54, 68, word, 'amber', 18);
      tiles.push({ probability: txt(scene.svg, x + 54, 98, '', 'chalk', 15),
        surprise: txt(scene.svg, x + 54, 129, '', 'tx-dim', 14) });
    });
    var result = txt(scene.svg, 200, 184, '', 'chalk', 17);
    p13Choices(scene, ['Mixed confidence', '50% on each word'], function (mode) {
      var p = mode ? [0.5, 0.5, 0.5] : [0.5, 0.25, 0.125];
      var losses = p.map(function (x) { return -Math.log2(x); });
      var average = losses.reduce(function (a, b) { return a + b; }, 0) / 3;
      tiles.forEach(function (tile, i) {
        tile.probability.textContent = 'p = ' + p[i];
        tile.surprise.textContent = losses[i] + (losses[i] === 1 ? ' bit' : ' bits');
      });
      result.textContent = 'Average ' + average + ' bits → perplexity ' + Math.pow(2, average);
      scene.update('H = average[−log₂(pcorrect)]; perplexity = 2^H',
        'H = (' + losses.join(' + ') + ')/3 = ' + average + '; perplexity = 2^' + average + ' = ' + Math.pow(2, average),
        'Each p is the probability the model assigned to the word that actually appeared. Perplexity ' + Math.pow(2, average) + ' means the same average prediction cost as assigning each correct word probability 1/' + Math.pow(2, average) + '. It does not mean the model literally considered only that many words. With nats, use exp(H) instead.');
    });
  }
  function p13CardClue(host, mutual) {
    var scene = p13Scene(host, 274, 'Four equally likely cards: a table of colour and shape counts showing uncertainty before and after seeing colour.');
    txt(scene.svg, 200, 24, 'Draw one of four equally likely cards', 'chalk', 15);
    var table = panel(scene.svg, [[2, 0], [0, 2]], { cell: 54, fs: 18, bracket: false });
    table.move(166, 74);
    txt(scene.svg, 193, 60, '○ Circle', 'chalk', 14);
    txt(scene.svg, 250, 60, '△ Triangle', 'chalk', 14, 'start');
    txt(scene.svg, 150, 108, 'Red', 'chalk', 15, 'end');
    txt(scene.svg, 150, 165, 'Blue', 'chalk', 15, 'end');
    txt(scene.svg, 200, 208, 'Cells count cards, not probabilities', 'tx-dim', 14);
    var before = txt(scene.svg, 200, 239, 'Before seeing colour: H(shape) = 1 bit', 'chalk', 15);
    var after = txt(scene.svg, 200, 263, '', 'amber', 15);
    p13Choices(scene, ['Colour reveals shape', 'Colour gives no clue'], function (mode) {
      table.cells.forEach(function (cell) { cell.text.textContent = mode ? '1' : (cell.r === cell.c ? '2' : '0'); });
      after.textContent = 'After seeing colour: ' + mode + ' bits remain';
      scene.update(mutual ? 'I(colour; shape) = H(shape) − H(shape | colour)' :
        'H(shape | colour) = Σ p(colour) H(shape for that colour)',
        mutual ? 'I = 1 − ' + mode + ' = ' + (1 - mode) + ' bit' :
        'H(shape | colour) = (2/4) × ' + mode + ' + (2/4) × ' + mode + ' = ' + mode + ' bits',
        mode ? 'There is one circle and one triangle in each colour. Even after seeing red or blue, shape is still 50/50: −0.5 log₂(0.5) − 0.5 log₂(0.5) = 1 bit. Colour removed zero uncertainty, so mutual information is zero.' :
        'There are two red circles and two blue triangles. Shape starts 50/50 (1 bit). Seeing red guarantees circle; seeing blue guarantees triangle. In either colour, shape entropy is −1 log₂(1) = 0. The clue removed 1 bit of uncertainty.');
    });
  }
  function p13JS(host) {
    var scene = p13Scene(host, 380, 'Two coin distributions and their midpoint: Jensen-Shannon divergence stays finite even when the two distributions disagree completely.');
    var pBars = p13Bars(scene.svg, 24, 'P: first coin', ['Heads', 'Tails']);
    var qBars = p13Bars(scene.svg, 137, 'Q: second coin', ['Heads', 'Tails']);
    var mBars = p13Bars(scene.svg, 250, 'M: average probabilities', ['Heads', 'Tails']);
    pBars([1, 0]);
    p13Choices(scene, ['Opposite coins', 'Identical coins'], function (mode) {
      qBars(mode ? [1, 0] : [0, 1]); mBars(mode ? [1, 0] : [0.5, 0.5]);
      scene.update('M = (P+Q)/2; JS(P,Q) = ½ D(P‖M) + ½ D(Q‖M)',
        mode ? 'M = ((1,0)+(1,0))/2 = (1,0); JS = ½ × log₂(1/1) + ½ × log₂(1/1) = 0 bits' :
        'M = ((1,0)+(0,1))/2 = (0.5,0.5); JS = ½ × log₂(1/0.5) + ½ × log₂(1/0.5) = 1 bit',
        mode ? 'Both coins guarantee heads. The midpoint is identical to both, so each KL-to-midpoint is zero.' :
        'One coin always gives heads; the other always tails. Their midpoint allows both, so each coin pays only 1 bit against it. JS averages those two costs: 1 bit, the maximum in base 2. Direct KL between the opposite coins is infinite because each calls the other’s outcome impossible. Zero-probability terms contribute zero.');
    });
  }
  REG['p13-bits'] = p13Bits;
  REG['p13-entropy'] = p13EntropyCard;
  REG['p13-kl'] = p13KL;
  REG['p13-perplexity'] = p13Perplexity;
  REG['p13-conditional'] = function (host) { p13CardClue(host, false); };
  REG['p13-mutual'] = function (host) { p13CardClue(host, true); };
  REG['p13-js'] = p13JS;

  /* Module 14: every term gets numbers, a picture, and a reading. */
  function p14Figure(host, height, label) {
    var svg = makeSVG(host, 420, height);
    svg.setAttribute('role', 'img');
    svg.setAttribute('aria-label', label);
    return { svg: svg, read: h('div', 'shapenote', null, host) };
  }
  function p14Plot(svg, xr, yr, xLabel, yLabel) {
    var F = new Frame(svg, { l: 48, t: 30, w: 332, h: 155 }, xr, yr);
    F.axes(xLabel, yLabel);
    return F;
  }
  function p14Dot(svg, F, x, y, colour) {
    el('circle', { cx: F.X(x), cy: F.Y(y), r: 5, fill: css('--' + colour) }, svg);
  }
  function p14Read(read, formula, numbers, meaning) {
    worked(read, formula, numbers);
    h('p', 'rd-a', meaning, read);
  }
  function p14Loss(host) {
    var fig = p14Figure(host, 205, 'Two prediction errors become squared losses, then their average and a size penalty.');
    var svg = fig.svg;
    txt(svg, 20, 25, 'Predict film ratings: true scores 3 and 5', 'tx-dim', 15, 'start');
    [['Prediction 2', 'error −1', 'loss 1'], ['Prediction 3', 'error −2', 'loss 4']].forEach(function (row, i) {
      var y = 68 + i * 55;
      txt(svg, 20, y, row[0], 'accent', 16, 'start');
      txt(svg, 167, y, row[1], 'tx-dim', 16, 'start');
      el('rect', { x: 282, y: y - 23, width: (i ? 100 : 25), height: 30, rx: 4,
        fill: css('--amber'), opacity: 0.3 }, svg);
      txt(svg, 290, y, row[2], 'amber', 16, 'start');
    });
    txt(svg, 210, 182, 'Average loss 2.5  +  size penalty 0.4  =  2.9', 'green', 15);
    p14Read(fig.read, 'loss = (prediction − truth)²', '(2−3)² = 1; (3−5)² = 4; cost = (1+4)/2 = 2.5.',
      'One bad prediction contributes more here because we square the error. Averaging makes one score for both examples.');
    p14Read(fig.read, 'objective = cost + λw²', 'For weight w=2 and penalty strength λ=0.1: 2.5 + 0.1×2² = 2.9.',
      'A weight is an adjustable model number. Regularisation means charging a penalty for a property such as large weights. The objective is the complete bill. Names vary across libraries; read the definition.');
  }
  function p14Batch(host) {
    var fig = p14Figure(host, 210, 'Six example gradients; highlighted examples determine the estimated update.'), n = 3;
    var values = [-2, 0, 2, 4, 6, 8], cells = [], label;
    values.forEach(function (g, i) {
      cells.push(el('rect', { x: 25 + i * 64, y: 55, width: 52, height: 49, rx: 5 }, fig.svg));
      txt(fig.svg, 51 + i * 64, 85, String(g), 'chalk', 18);
      txt(fig.svg, 51 + i * 64, 127, 'item ' + (i + 1), 'tx-faint', 13);
    });
    txt(fig.svg, 210, 25, 'Each item suggests a gradient g', 'tx-dim', 16);
    label = txt(fig.svg, 210, 178, '', 'amber', 16);
    function draw() {
      var sum = values.slice(0, n).reduce(function (a, b) { return a + b; }, 0), mean = sum / n;
      cells.forEach(function (cell, i) { cell.setAttribute('fill', css(i < n ? '--accent' : '--panel-2')); cell.setAttribute('opacity', i < n ? '0.45' : '1'); });
      label.textContent = n + ' of 6 items → average gradient ' + mean;
      fig.read.innerHTML = '';
      p14Read(fig.read, 'ḡ = sum of selected gradients / count', values.slice(0, n).join(' + ') + ' = ' + sum + '; ḡ = ' + sum + '/' + n + ' = ' + mean + '.',
        'At the same weight, each example suggests a direction and size of change. Average those suggestions, then subtract learning rate × average. Here one item points one way (−2), while the full batch points the other (+3).');
      h('p', 'rd-b', 'These first items are fixed so you can check the arithmetic. Training normally shuffles or samples examples; a small batch is cheaper, but its gradient can differ from the whole dataset.', fig.read);
    }
    var bar = ctl(host);
    [[1, 'one item'], [3, 'mini-batch: 3'], [6, 'whole batch: 6']].forEach(function (choice) {
      button(bar, choice[1], function () { n = choice[0]; draw(); });
    });
    draw();
  }
  function p14Epoch(host) {
    var fig = p14Figure(host, 215, 'Twelve examples grouped in four batches of three; advance one update at a time.'), step = 1, cells = [];
    for (var i = 0; i < 12; i++) {
      var col = i % 3, row = Math.floor(i / 3);
      cells.push(el('rect', { x: 118 + col * 54, y: 18 + row * 42, width: 44, height: 32, rx: 4 }, fig.svg));
      txt(fig.svg, 140 + col * 54, 40 + row * 42, String(i + 1), 'chalk', 16);
      if (col === 0) txt(fig.svg, 102, 40 + row * 42, 'batch ' + (row + 1), 'tx-dim', 14, 'end');
    }
    var label = txt(fig.svg, 210, 202, '', 'amber', 16);
    function draw() {
      var position = (step - 1) % 4;
      cells.forEach(function (cell, i) { cell.setAttribute('fill', css(Math.floor(i / 3) === position ? '--accent' : '--panel-2')); cell.setAttribute('opacity', Math.floor(i / 3) === position ? '0.55' : '1'); });
      label.textContent = 'Step ' + step + ' · epoch ' + Math.ceil(step / 4) + ' · batch ' + (position + 1);
      fig.read.innerHTML = '';
      p14Read(fig.read, 'steps per epoch = examples / batch size', '12/3 = 4 steps. Step ' + step + ' has completed ' + Math.floor(step / 4) + ' full epoch(s).',
        'Each highlighted group produces one weight update. After all four groups, every example has been seen once: one epoch. Step 5 revisits the data for epoch 2. Real runs usually reshuffle between epochs.');
      h('p', 'rd-b', 'This example uses every item once, no incomplete batch and no gradient accumulation. Accumulation means combining several batches before updating; then batches and update steps are no longer one-to-one.', fig.read);
    }
    button(ctl(host), 'next update (1–8)', function () { step = step % 8 + 1; draw(); });
    draw();
  }
  function p14Rate(host) {
    var fig = p14Figure(host, 235, 'Four gradient descent steps on L(w)=w squared, with a selectable learning rate.'), rate = 0.1;
    function draw() {
      fig.svg.innerHTML = ''; fig.read.innerHTML = '';
      var F = p14Plot(fig.svg, [0, 4], [0, 18], 'update number', 'loss w²'), w = 2, points = [], ws = [w];
      for (var i = 0; i <= 4; i++) {
        points.push(F.X(i) + ',' + F.Y(w * w)); p14Dot(fig.svg, F, i, w * w, 'amber');
        txt(fig.svg, F.X(i), 204, String(i), 'tx-faint', 13);
        if (i < 4) { w *= 1 - 2 * rate; ws.push(w); }
      }
      el('polyline', { points: points.join(' '), fill: 'none', stroke: css('--amber'), 'stroke-width': 2 }, fig.svg);
      txt(fig.svg, 215, 20, 'η = ' + rate + ' · start w=2, loss=4', 'tx-dim', 16);
      p14Read(fig.read, 'w next = w − η×2w', 'First step: 2 − ' + rate + '×4 = ' + ws[1].toFixed(2) + '. After four: w=' + w.toFixed(3) + ', loss=' + (w * w).toFixed(3) + '.',
        rate > 1 ? 'The sign flips each time, but the distance from zero grows: every overshoot is worse. A big learning rate can climb the very hill you meant to descend.' : rate === 0.5 ? 'For this particular quadratic, this rate lands exactly at the minimum in one step. Other loss curves have different curvature and need different rates.' : 'The gradient says which direction is uphill. The learning rate η decides how much of that slope to subtract: here four smaller steps steadily reduce the error.');
    }
    var bar = ctl(host);
    [0.1, 0.5, 1.1].forEach(function (v) { button(bar, 'η = ' + v, function () { rate = v; draw(); }); });
    draw();
  }
  function p14Schedule(host) {
    var fig = p14Figure(host, 230, 'Learning rate rises to 0.1 over two updates, then falls along a cosine curve to 0.01.'), at = 2;
    function value(t) { return t <= 2 ? 0.1 * t / 2 : 0.01 + 0.09 * (1 + Math.cos(Math.PI * (t - 2) / 8)) / 2; }
    var F = p14Plot(fig.svg, [0, 10], [0, 0.11], 'training step t', 'learning rate η');
    F.path(value, 140); [0, 2, 6, 10].forEach(function (t) { txt(fig.svg, F.X(t), 204, String(t), 'tx-faint', 13); });
    txt(fig.svg, 80, 25, 'warmup', 'green', 14); txt(fig.svg, 265, 25, 'cosine decay', 'amber', 14);
    var dot = el('circle', { r: 6, fill: css('--amber') }, fig.svg);
    slider(ctl(host), 'step', 0, 10, 1, at, function (t) {
      at = t; dot.setAttribute('cx', F.X(t)); dot.setAttribute('cy', F.Y(value(t))); fig.read.innerHTML = '';
      if (t <= 2) p14Read(fig.read, 'η(t) = peak × t / warmup steps', 'η(' + t + ') = 0.1×' + t + '/2 = ' + value(t).toFixed(3) + '.',
        'Warmup means starting with small updates and increasing them gradually. Here it lasts two steps; this is a teaching schedule, not a training prescription.');
      else p14Read(fig.read, 'u = (t−2)/8; η = 0.01 + 0.045(1+cos(πu))', 'At t=' + t + ', u=' + ((t - 2) / 8).toFixed(3) + '; η=' + value(t).toFixed(4) + '.',
        'Cosine is the smooth wave that gives this curve its rounded shape. Halfway through decay (step 6), the rate is 0.055. At step 10, it is 0.01, allowing finer adjustments.');
      return t;
    });
  }
  function p14Adaptive(host) {
    var fig = p14Figure(host, 225, 'RMSProp, Adam and AdamW weights after one or two updates from the same gradients.'), count = 1;
    var eta = 0.1, beta = 0.9, epsilon = 1e-8, decay = 0.1, gradients = [2, -1];
    function state(n) {
      var rms = 2, adam = 2, adamw = 2, s = 0, m = 0, mh, sh, delta;
      for (var t = 1; t <= n; t++) {
        var g = gradients[t - 1]; s = beta * s + (1 - beta) * g * g; m = beta * m + (1 - beta) * g;
        mh = m / (1 - Math.pow(beta, t)); sh = s / (1 - Math.pow(beta, t));
        rms -= eta * g / (Math.sqrt(s) + epsilon);
        delta = eta * mh / (Math.sqrt(sh) + epsilon);
        adam -= delta; adamw = adamw * (1 - eta * decay) - delta;
      }
      return { rms: rms, adam: adam, adamw: adamw, s: s, m: m, mh: mh, sh: sh, delta: delta };
    }
    function draw() {
      fig.svg.innerHTML = ''; fig.read.innerHTML = '';
      var s = state(count);
      txt(fig.svg, 210, 24, 'Start w=2 · gradient history ' + (count === 1 ? '[2]' : '[2, −1]'), 'tx-dim', 16);
      [['RMSProp', s.rms, 'accent'], ['Adam', s.adam, 'amber'], ['AdamW', s.adamw, 'green']].forEach(function (row, i) {
        var y = 70 + i * 53;
        txt(fig.svg, 18, y, row[0], row[2], 16, 'start');
        el('rect', { x: 120, y: y - 22, width: row[1] * 100, height: 28, rx: 4, fill: css('--' + row[2]), opacity: 0.35 }, fig.svg);
        txt(fig.svg, 395, y, row[1].toFixed(4), row[2], 16, 'end');
      });
      h('p', 'rd-a', 'One parameter, same supplied gradients. We choose η=0.1 and both memory factors β=0.9 for easy arithmetic. A memory factor of 0.9 keeps 90% of the old average and mixes in 10% of the new value. These are illustrative settings.', fig.read);
      p14Read(fig.read, 'RMSProp: s = 0.9s old + 0.1g²; Δw = −ηg/(√s+ε)',
        count === 1 ? 's=0.9×0+0.1×2²=0.4; w=2−0.1×2/√0.4≈1.6838.' : 's=0.9×0.4+0.1×(−1)²=0.46; w≈1.6838+0.1/√0.46=1.8312.',
        's remembers squared gradients. Dividing by its square root rescales the step. ε=0.00000001 keeps division safe when s is zero; displayed arithmetic rounds away its tiny effect.');
      p14Read(fig.read, 'Adam: m=0.9m old+0.1g; m̂=m/(1−0.9ᵗ); ŝ=s/(1−0.9ᵗ)',
        'At t=' + count + ': m=' + s.m.toFixed(2) + ', s=' + s.s.toFixed(2) + ', m̂=' + s.mh.toFixed(4) + ', ŝ=' + s.sh.toFixed(4) + '.',
        'm remembers signed gradients, so opposing suggestions can cancel. The hats mean bias correction: compensate for averages that started at zero.');
      p14Read(fig.read, 'Adam step = ηm̂/(√ŝ+ε); w next = w old − step',
        'step≈' + s.delta.toFixed(4) + '; resulting w≈' + s.adam.toFixed(4) + '.',
        count === 2 ? 'The latest gradient is negative, but remembered positive gradients still outweigh it. Adam therefore keeps moving downward in weight, while RMSProp reverses.' : 'On this first step, correction turns m=0.2 and s=0.4 into m̂=2 and ŝ=4. The update is 0.1×2/2=0.1.');
      p14Read(fig.read, 'AdamW: w next = (1−ηλ)w old − Adam step',
        count === 1 ? '(1−0.1×0.1)×2−0.1 = 1.88.' : '(1−0.1×0.1)×1.88−0.0271 ≈ 1.8341.',
        'Weight decay means shrinking the weight directly: λ=0.1 gives 1% shrinkage each step here. AdamW performs this separately from gradient rescaling. Adam above has no decay, so you can see the additional change.');
      h('p', 'rd-b', 'Both moving averages start at zero. We use plain, uncentred RMSProp without momentum. The gradients are prescribed for comparison; an actual run recomputes them at each new weight.', fig.read);
    }
    var bar = ctl(host); [1, 2].forEach(function (n) { button(bar, 'after step ' + n, function () { count = n; draw(); }); }); draw();
  }
  function p14Wall(svg, limit) {
    var F = p14Plot(svg, [0, 4], [0, 9], 'choice x', 'cost f(x)');
    el('rect', { x: F.X(0), y: 30, width: F.X(limit) - F.X(0), height: 155, fill: css('--green'), opacity: 0.1 }, svg);
    F.path(function (x) { return (x - 3) * (x - 3); });
    el('line', { x1: F.X(limit), y1: 30, x2: F.X(limit), y2: 185, stroke: css('--pink'), 'stroke-width': 2, 'stroke-dasharray': '5 4' }, svg);
    txt(svg, F.X(limit), 23, 'limit ' + limit, 'pink', 15);
    [0, 1, 2, 3, 4].forEach(function (x) { txt(svg, F.X(x), 205, String(x), 'tx-faint', 13); });
    return F;
  }
  function p14Constrained(host) {
    var fig = p14Figure(host, 230, 'A cost curve prefers x=3, but the rule x≤2 restricts the answer to x=2.'), limit = 2;
    function draw() {
      fig.svg.innerHTML = ''; fig.read.innerHTML = ''; var F = p14Wall(fig.svg, limit), best = Math.min(3, limit);
      fig.svg.setAttribute('aria-label', 'Cost (x−3) squared with rule x≤' + limit + ': the best allowed choice is x=' + best + '.');
      p14Dot(fig.svg, F, 3, 0, 'tx-faint'); p14Dot(fig.svg, F, best, (best - 3) * (best - 3), 'amber');
      p14Read(fig.read, 'minimise f(x)=(x−3)², subject to x≤b', 'For b=' + limit + ': x*=' + best + ', f(x*)=(' + best + '−3)²=' + ((best - 3) * (best - 3)) + '.',
        limit === 2 ? 'Think of x as hours spent polishing a film. Three hours would give the lowest cost, but your budget allows only two. Green shading marks the displayed allowed choices. The best allowed point is at the wall.' : 'With a four-hour budget, the preferred three-hour choice is already allowed. Extra budget has no value here: you do not have to use it all.');
    }
    var bar = ctl(host); button(bar, 'budget: 2 hours', function () { limit = 2; draw(); }); button(bar, 'budget: 4 hours', function () { limit = 4; draw(); }); draw();
  }
  function p14Multiplier(host) {
    var fig = p14Figure(host, 230, 'Increasing the multiplier shifts the minimum of cost plus constraint toward the two-hour budget.');
    slider(ctl(host), 'price λ', 0, 4, 0.5, 2, function (lambda) {
      fig.svg.innerHTML = ''; fig.read.innerHTML = '';
      var F = p14Plot(fig.svg, [0, 4], [-4, 9], 'choice x', 'Lagrangian ℒ'), best = 3 - lambda / 2;
      F.path(function (x) { return (x - 3) * (x - 3) + lambda * (x - 2); });
      el('line', { x1: F.X(2), y1: 30, x2: F.X(2), y2: 185, stroke: css('--pink'), 'stroke-dasharray': '5 4' }, fig.svg);
      p14Dot(fig.svg, F, best, lambda - lambda * lambda / 4, 'amber');
      txt(fig.svg, 210, 220, 'ℒ minimum at x=' + best.toFixed(2) + ' · budget wall x=2', 'amber', 15);
      p14Read(fig.read, 'ℒ(x,λ) = (x−3)² + λ(x−2)', '∂ℒ/∂x = 2(x−3)+λ = 0; x = 3−λ/2 = ' + best.toFixed(2) + '.',
        'The multiplier λ is a price attached to exceeding the two-hour budget. Choose λ=2 and the preferred x moves from 3 to 2: the budget is met exactly. A random price does not automatically solve the constrained problem.');
      h('p', 'rd-b', 'At the solution, λ=2 measures a local tradeoff: slightly more budget reduces the best cost at about 2 cost units per extra hour. It is a marginal price, not a claim that every full hour saves exactly 2.', fig.read);
      return lambda.toFixed(1);
    });
  }
  function p14Dual(host) {
    var fig = p14Figure(host, 230, 'The dual lower bound q(λ)=λ−λ²/4 reaches the constrained minimum cost 1 at λ=2.');
    var F = p14Plot(fig.svg, [0, 4], [0, 1.2], 'multiplier λ ≥ 0', 'lower bound q(λ)');
    F.path(function (lambda) { return lambda - lambda * lambda / 4; });
    el('line', { x1: F.X(0), y1: F.Y(1), x2: F.X(4), y2: F.Y(1), stroke: css('--green'), 'stroke-dasharray': '4 4' }, fig.svg);
    txt(fig.svg, 210, 20, 'best allowed cost = 1', 'green', 15);
    [0, 1, 2, 3, 4].forEach(function (x) { txt(fig.svg, F.X(x), 205, String(x), 'tx-faint', 13); });
    var dot = el('circle', { r: 6, fill: css('--amber') }, fig.svg);
    slider(ctl(host), 'price λ', 0, 4, 0.5, 2, function (lambda) {
      var q = lambda - lambda * lambda / 4; dot.setAttribute('cx', F.X(lambda)); dot.setAttribute('cy', F.Y(q)); fig.read.innerHTML = '';
      p14Read(fig.read, 'q(λ) = min over x of ℒ(x,λ) = λ−λ²/4', 'q(' + lambda + ') = ' + lambda + '−' + lambda + '²/4 = ' + q.toFixed(4) + '.',
        'For any allowed x≤2 and λ≥0, the added term λ(x−2) is non-positive. Minimising ℒ therefore gives a lower bound on the original best cost. The dual asks for the highest such bound.');
      p14Read(fig.read, 'max over λ≥0 of q(λ)', 'q′(λ)=1−λ/2=0 → λ*=2; q(2)=1=f(2).',
        'The lower bound meets the actual cost, so this example has strong duality: no gap. For other problems the best lower bound can remain below the original optimum.');
      return lambda.toFixed(1);
    });
  }
  function p14KKT(host) {
    var fig = p14Figure(host, 290, 'Four checks certify the optimum for this convex cost with an upper budget bound.'), selection = 0;
    var cases = [{ x: 2, lambda: 2, b: 2 }, { x: 3, lambda: 0, b: 2 }, { x: 2, lambda: 0, b: 2 }, { x: 3, lambda: 0, b: 4 }];
    function draw() {
      var s = cases[selection], gap = s.x - s.b, station = 2 * (s.x - 3) + s.lambda;
      fig.svg.innerHTML = ''; fig.read.innerHTML = '';
      txt(fig.svg, 210, 24, 'Candidate x=' + s.x + ', λ=' + s.lambda + '; rule x≤' + s.b, 'amber', 17);
      var rows = [ ['Allowed choice', s.x + '−' + s.b + '=' + gap + ' ≤ 0', gap <= 0],
        ['Non-negative price', 'λ=' + s.lambda + ' ≥ 0', s.lambda >= 0],
        ['Balanced slope', '2(' + s.x + '−3)+' + s.lambda + '=' + station, station === 0],
        ['Slackness', s.lambda + '×(' + s.x + '−' + s.b + ')=' + (s.lambda * gap), s.lambda * gap === 0] ];
      rows.forEach(function (row, i) {
        var y = 61 + i * 55;
        el('rect', { x: 18, y: y - 20, width: 384, height: 48, rx: 5, fill: css(row[2] ? '--green' : '--pink'), opacity: 0.12 }, fig.svg);
        txt(fig.svg, 30, y, row[0], 'tx-dim', 15, 'start');
        txt(fig.svg, 30, y + 20, row[1], row[2] ? 'green' : 'pink', 15, 'start');
        txt(fig.svg, 385, y + 9, row[2] ? 'PASS' : 'FAIL', row[2] ? 'green' : 'pink', 15, 'end');
      });
      p14Read(fig.read, 'x−b≤0; λ≥0; 2(x−3)+λ=0; λ(x−b)=0',
        'x=' + s.x + ', b=' + s.b + ', λ=' + s.lambda + ': ' + rows.filter(function (r) { return r[2]; }).length + ' of 4 checks pass.',
        'KKT names four conditions, including the zero product called complementary slackness. When budget is spare (x<b), its price must be zero. A positive price can occur only at the wall (x=b).');
      h('p', 'rd-b', 'All four conditions together certify the global optimum for this convex quadratic and linear constraint. For general non-convex problems, passing KKT need not mean a global minimum. A tight constraint may also have a zero multiplier.', fig.read);
    }
    var bar = ctl(host); button(bar, 'next candidate (4 cases)', function () { selection = (selection + 1) % cases.length; draw(); }); draw();
  }
  function p14Laplace(host) {
    var fig = p14Figure(host, 235, 'A Gaussian approximation and a non-Gaussian peak share their centre and local curvature.');
    slider(ctl(host), 'curvature H', 1, 9, 1, 4, function (curvature) {
      fig.svg.innerHTML = ''; fig.read.innerHTML = '';
      var F = p14Plot(fig.svg, [-2, 4], [0, 1.1], 'parameter x', 'relative height'), sigma = 1 / Math.sqrt(curvature);
      F.path(function (x) { var d = x - 1; return Math.exp(-0.5 * curvature * d * d - 0.2 * Math.pow(d, 4)); }, 180, 's-curve', { style: 'stroke:' + css('--accent') });
      F.path(function (x) { return Math.exp(-0.5 * curvature * (x - 1) * (x - 1)); }, 180, 's-curve', { style: 'stroke:' + css('--amber'), 'stroke-dasharray': '5 4' });
      txt(fig.svg, 105, 22, 'true shape', 'accent', 15); txt(fig.svg, 290, 22, 'Gaussian (dashed)', 'amber', 15);
      [-1, 0, 1, 2, 3].forEach(function (x) { txt(fig.svg, F.X(x), 204, String(x), 'tx-faint', 13); });
      p14Read(fig.read, 'p(x) ∝ exp(−φ(x)); φ = H(x−1)²/2 + 0.2(x−1)⁴', 'The peak is at x=1; φ″(1)=H=' + curvature + '.',
        'The mode is the position of the highest point. Curvature measures how fast the negative log-density bends upward there. The fourth-power term changes the tails but contributes no curvature at the peak.');
      p14Read(fig.read, 'Laplace: p(x) ≈ Normal(1, variance 1/H)', 'variance=1/' + curvature + '=' + (1 / curvature).toFixed(3) + '; standard deviation σ=1/√' + curvature + '=' + sigma.toFixed(3) + '.',
        'Match the peak and local curvature, then use a bell curve with that width. Larger curvature means a narrower bell. Both plotted heights are divided by their own peak value; these lines show shape, not normalised density.');
      h('p', 'rd-b', 'In several dimensions, the Hessian is the matrix of second derivatives and its inverse gives the approximate covariance, provided it is positive definite. The approximation is local; skewed tails or multiple peaks can make it poor.', fig.read);
      return curvature;
    });
  }
  REG['p14-loss'] = p14Loss;
  REG['p14-batch'] = p14Batch;
  REG['p14-epoch'] = p14Epoch;
  REG['p14-rate'] = p14Rate;
  REG['p14-schedule'] = p14Schedule;
  REG['p14-adaptive'] = p14Adaptive;
  REG['p14-constrained'] = p14Constrained;
  REG['p14-multiplier'] = p14Multiplier;
  REG['p14-dual'] = p14Dual;
  REG['p14-kkt'] = p14KKT;
  REG['p14-laplace'] = p14Laplace;

  /* Keep matrix entries legible on a phone; pan the figure rather than shrinking its text. */
  function readableMathDiagram(host) {
    if (!host.closest('.math-terms')) return;
    var svgs = host.querySelectorAll('svg');
    for (var i = 0; i < svgs.length; i++) {
      var svg = svgs[i];
      if (svg.closest('.math-drawing')) continue;
      var region = h('div', 'math-drawing');
      region.tabIndex = 0;
      region.setAttribute('role', 'region');
      region.setAttribute('aria-label', (svg.getAttribute('aria-label') || 'Worked maths diagram') +
        '. Scroll horizontally if the diagram is wider than the screen.');
      svg.parentNode.insertBefore(region, svg);
      region.appendChild(svg);
      var hint = h('p', 'math-pan-hint', 'Scroll the diagram sideways to see every column.');
      region.insertAdjacentElement('afterend', hint);
      (function (box, note) {
        function resize() { note.hidden = box.scrollWidth <= box.clientWidth + 1; }
        if (typeof ResizeObserver !== 'undefined') new ResizeObserver(resize).observe(box);
        resize();
      })(region, hint);
    }
  }

  function init(root) {
    root = root || document;
    var nodes = root.querySelectorAll('[data-viz]');
    for (var i = 0; i < nodes.length; i++) {
      var n = nodes[i];
      if (n.dataset.vizReady) continue;
      n.dataset.vizReady = '1';
      var fn = REG[n.dataset.viz];
      if (fn) { try { fn(n); readableMathDiagram(n); } catch (e) { n.innerHTML = '<p class="faint small">viz error: ' + e.message + '</p>'; } }
      else n.innerHTML = '<p class="faint small">no visualisation registered for "' + n.dataset.viz + '"</p>';
    }
  }

  return { init: init, names: function () { return Object.keys(REG); },
           register: function (k, f) { REG[k] = f; } };
})();
