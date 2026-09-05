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
    activations: vizActivations
  };

  function init(root) {
    root = root || document;
    var nodes = root.querySelectorAll('[data-viz]');
    for (var i = 0; i < nodes.length; i++) {
      var n = nodes[i];
      if (n.dataset.vizReady) continue;
      n.dataset.vizReady = '1';
      var fn = REG[n.dataset.viz];
      if (fn) { try { fn(n); } catch (e) { n.innerHTML = '<p class="faint small">viz error: ' + e.message + '</p>'; } }
      else n.innerHTML = '<p class="faint small">no visualisation registered for "' + n.dataset.viz + '"</p>';
    }
  }

  return { init: init, register: function (k, f) { REG[k] = f; } };
})();
