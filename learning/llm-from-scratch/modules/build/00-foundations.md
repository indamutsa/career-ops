# B0 — Foundations: probability, loss, gradients, backprop

Nothing is assumed here. By the end of this lesson you will have derived the loss function every
language model is trained with, and written an automatic differentiation engine from scratch that
agrees with PyTorch to machine precision.

If you already know this material, do Lab 3 anyway. Writing `backward()` yourself is the difference
between believing backpropagation and knowing it.

---

## Terms

| Term | Meaning |
|------|---------|
| **Token** | The atomic unit a model reads and writes. Roughly a word-piece. B2 builds these. |
| **Vocabulary** | The finite set of tokens. Ours will be 4096. |
| **Language model** | A function assigning a probability to every possible next token. |
| **Autoregressive** | Generates one token at a time, each conditioned on all previous. |
| **Chain rule of probability** | `P(a,b,c) = P(a)·P(b|a)·P(c|a,b)`. |
| **Logits** | Raw unnormalised scores, one per vocabulary entry. Range: all reals. |
| **Softmax** | Turns logits into a probability distribution. |
| **Likelihood** | The probability the model assigns to the data actually observed. |
| **Maximum likelihood estimation (MLE)** | Choose parameters making the observed data most probable. |
| **Cross-entropy** | The loss. Negative log-likelihood, averaged. |
| **Perplexity** | `exp(loss)`. "Effective number of choices" the model is deciding between. |
| **Parameter** | A number the model learns. Ours will have ~3.4 million. |
| **Gradient** | The vector of partial derivatives of the loss w.r.t. every parameter. |
| **Gradient descent** | Step parameters in the direction that reduces loss. |
| **Learning rate** | How big that step is. |
| **Backpropagation** | Computing all gradients efficiently by applying the chain rule backwards. |
| **Computational graph** | The DAG of operations, recorded so it can be differentiated. |
| **Autograd** | Machinery that builds that graph and traverses it. |

---

## Concepts

### 1. A language model is a probability distribution

Not "an AI that talks". A language model assigns a probability to every possible sequence of
tokens. That is the whole definition.

The trick that makes it tractable is the chain rule of probability. The probability of a sequence
factorises into a product of next-token probabilities:

```
P(t₁, t₂, …, tₙ) = P(t₁) · P(t₂|t₁) · P(t₃|t₁,t₂) · … · P(tₙ|t₁…tₙ₋₁)
                 = ∏ᵢ P(tᵢ | t₍<ᵢ₎)
```

So modelling *all of language* reduces to one repeated question: **given everything so far, what is
the distribution over the next token?**

That is what your model will compute. Input: a sequence of token IDs. Output: for each position, a
vector of 4096 numbers — a probability for every token that could come next.

### 2. From scores to probabilities: softmax

A neural network naturally produces unbounded real numbers — **logits**. Probabilities must be
non-negative and sum to 1. Softmax bridges them:

```
softmax(z)ᵢ = exp(zᵢ) / Σⱼ exp(zⱼ)
```

`exp` makes everything positive; dividing by the sum makes it total 1.

Two properties to internalise:

**Shift invariance.** `softmax(z) = softmax(z + c)` for any constant `c`. Adding a constant to
every logit changes nothing. This is not a curiosity — it is the reason every real implementation
subtracts the max logit before exponentiating:

```python
def softmax(z):
    z = z - z.max()          # numerically safe: exp(large) overflows to inf
    e = z.exp()
    return e / e.sum()
```

Skip that line and you will get `nan` the first time a logit reaches ~89 in float32. This bug will
find you eventually; recognise it when it does.

**Temperature.** Divide logits by `T` before softmax: `softmax(z/T)`. Small `T` sharpens toward the
argmax, large `T` flattens toward uniform. That is the entire mechanism behind the temperature knob
you have used in every API. B7 makes it concrete.

### 3. Deriving the loss

We want parameters `θ` that make the *observed* data as probable as possible — maximum likelihood:

```
θ* = argmax_θ  ∏ᵢ P_θ(tᵢ | t₍<ᵢ₎)
```

Products of thousands of numbers each < 1 underflow to zero immediately. Take the log — monotonic,
so the argmax is unchanged, and it turns the product into a sum:

```
θ* = argmax_θ  Σᵢ log P_θ(tᵢ | t₍<ᵢ₎)
```

Optimisers minimise by convention, so negate, and average over tokens so the value doesn't depend
on sequence length:

```
L(θ) = − (1/N) Σᵢ log P_θ(tᵢ | t₍<ᵢ₎)
```

**That is cross-entropy loss, and it is the only loss used to pretrain a language model.** It is
not an arbitrary choice someone found to work well — it falls directly out of "make the observed
data as likely as possible."

The per-token form is worth staring at:

```
loss for one token = − log( probability the model assigned to the correct token )
```

| Model's probability for the true token | Loss |
|----------------------------------------|------|
| 1.0 (certain, correct) | 0.00 |
| 0.5 | 0.69 |
| 0.1 | 2.30 |
| 0.01 | 4.61 |
| → 0 (certain, wrong) | → ∞ |

**Nothing about the probabilities assigned to wrong tokens appears in the formula.** The loss only
looks at the true token. The wrong tokens are punished indirectly: softmax forces the total to 1,
so raising the truth necessarily lowers everything else.

### 4. Perplexity, and the number to expect

Perplexity is just the exponentiated loss:

```
perplexity = exp(loss)
```

It has a clean reading: **the effective number of tokens the model is choosing between.** A model
with perplexity 12 is behaving as though it were picking uniformly among 12 options at each step.

This gives you the single most useful sanity number in the whole track:

> **An untrained model must have loss ≈ `ln(vocab_size)`.**

With a 4096-token vocabulary, a randomly initialised model assigns roughly `1/4096` to everything,
so:

```
loss ≈ −ln(1/4096) = ln(4096) ≈ 8.32     perplexity ≈ 4096
```

**Your first training step must print ≈ 8.3.** If it prints 2, your labels are leaking (the model
can see the answer). If it prints 40, your initialisation is broken. This one check will save you
hours in B6, and it is the first thing an experienced person looks at.

### 5. Gradients

A derivative answers: *if I nudge this input a little, how much does the output change, and in
which direction?*

```
df/dx = limit as h→0 of  (f(x+h) − f(x)) / h
```

For a function of millions of inputs, the **gradient** `∇L` is the vector of all partial
derivatives — one per parameter. It points in the direction of steepest *increase*, so to reduce
the loss we step against it:

```
θ ← θ − lr · ∇L(θ)
```

That is gradient descent, entire. Every optimiser you will meet — SGD, Adam, AdamW — is this line
with a smarter estimate of the step size and direction.

### 6. Backpropagation is the chain rule, applied efficiently

Computing 3.4 million derivatives numerically would need 3.4 million forward passes. Backprop
computes all of them in **one** backward pass, and the whole idea is the chain rule:

```
if  y = f(u)  and  u = g(x)   then   dy/dx = (dy/du) · (du/dx)
```

Every operation in a neural network is a simple function whose local derivative we know. Compose
them into a graph, then walk it backwards multiplying local derivatives, accumulating into each
node.

Three rules cover almost everything:

| Operation | Local gradient rule |
|-----------|---------------------|
| `c = a + b` | Gradient flows through unchanged: `a.grad += c.grad`, `b.grad += c.grad` |
| `c = a * b` | Each gets the *other's* value: `a.grad += b.data * c.grad`, `b.grad += a.data * c.grad` |
| `c = f(a)` | `a.grad += f'(a.data) * c.grad` |

Two details that trip people up, and that you will feel in Lab 3:

- **`+=`, never `=`.** If a node feeds two places, its gradients must *accumulate*. Using `=` gives
  you silently wrong gradients — the model still trains, just badly, which is the worst kind of bug.
  This is also why real training loops call `optimizer.zero_grad()` every step: gradients accumulate
  by design, so you must clear them.
- **Order matters.** A node can only be processed once everything downstream of it is done. That
  ordering is a topological sort of the graph.

---

## Where it's used

- Every training run you will ever launch, at every scale.
- `loss ≈ ln(vocab)` is the first-step sanity check in B6 and in any real pretraining job.
- The `+=` accumulation rule is why gradient accumulation for large batches works at all (Module 17).
- Softmax shift-invariance is why `logsumexp` exists and why attention implementations subtract row
  maxima.
- Cross-entropy with `ignore_index=-100` is the masked-loss mechanism of SFT (Module 11) — the same
  formula, applied to fewer positions.

---

## Labs

### Lab 1 — Loss by hand, then by machine

Do this on paper first. A 4-token vocabulary `[a, b, c, d]`, the model outputs logits
`[2.0, 1.0, 0.1, -1.0]`, and the correct token is `b` (index 1).

1. Subtract the max (2.0) from every logit.
2. Exponentiate.
3. Divide by the sum.
4. Take `−log` of the entry for `b`.

Then check yourself:

```python
import torch, torch.nn.functional as F

logits = torch.tensor([2.0, 1.0, 0.1, -1.0])
target = torch.tensor(1)

probs = F.softmax(logits, dim=-1)
manual = -torch.log(probs[target])
builtin = F.cross_entropy(logits.unsqueeze(0), target.unsqueeze(0))

print(probs, manual.item(), builtin.item())
assert torch.allclose(manual, builtin)
```

**They must match exactly.** `F.cross_entropy` takes *logits*, not probabilities — it applies
softmax internally. Passing it softmaxed values is one of the most common bugs in the field, and
it does not error; it just trains a worse model. Prove it to yourself:

```python
print("wrong:", F.cross_entropy(probs.unsqueeze(0), target.unsqueeze(0)).item())
```

Then confirm shift invariance and see the overflow it prevents:

```python
print(F.softmax(logits + 1000, dim=-1))          # fine — torch subtracts the max
print(torch.exp(logits + 1000))                  # inf — this is what naive code does
```

### Lab 2 — The number you must see at step 0

```python
import math
for V in (4, 256, 4096, 50257):
    print(f"vocab {V:>6}  expected initial loss {math.log(V):.3f}")
```

Now verify it empirically with an untrained model:

```python
V, B, T = 4096, 8, 16
logits = torch.randn(B, T, V) * 0.02              # small random init
targets = torch.randint(0, V, (B, T))
loss = F.cross_entropy(logits.view(-1, V), targets.view(-1))
print(loss.item(), math.log(V))                   # ≈ 8.32 either way
```

**Write `8.32` somewhere you will see it in B6.** Then break it on purpose — set the init scale to
`5.0` instead of `0.02` and watch the loss jump well above `ln(V)`, because confident *wrong*
predictions are worse than uniform ones. That is why initialisation scale matters.

### Lab 3 — Write autograd yourself

The centrepiece. Roughly 60 lines, and it removes the last piece of magic.

```python
import math

class Value:
    """A scalar that remembers how it was computed."""

    def __init__(self, data, _children=(), _op=""):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None     # how to push gradient to my inputs
        self._prev = set(_children)
        self._op = _op

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward():
            self.grad  += out.grad        # += not =, always
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward():
            self.grad  += other.data * out.grad
            other.grad += self.data  * out.grad
        out._backward = _backward
        return out

    def __pow__(self, k):
        out = Value(self.data ** k, (self,), f"**{k}")

        def _backward():
            self.grad += k * (self.data ** (k - 1)) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out

    def exp(self):
        e = math.exp(self.data)
        out = Value(e, (self,), "exp")

        def _backward():
            self.grad += e * out.grad          # d/dx eˣ = eˣ
        out._backward = _backward
        return out

    def log(self):
        out = Value(math.log(self.data), (self,), "log")

        def _backward():
            self.grad += (1.0 / self.data) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        topo, visited = [], set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build(child)
                topo.append(v)
        build(self)

        self.grad = 1.0                        # dL/dL = 1
        for v in reversed(topo):
            v._backward()

    # conveniences so expressions read normally
    def __neg__(self):        return self * -1
    def __sub__(self, o):     return self + (-o)
    def __radd__(self, o):    return self + o
    def __rmul__(self, o):    return self * o
    def __truediv__(self, o): return self * (o ** -1 if isinstance(o, Value) else 1 / o)
    def __repr__(self):       return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"
```

Verify against PyTorch — this is the proof, not the code:

```python
a = Value(2.0); b = Value(-3.0); c = Value(10.0)
d = (a * b + c).tanh()
d.backward()

import torch
ta = torch.tensor(2.0,  requires_grad=True)
tb = torch.tensor(-3.0, requires_grad=True)
tc = torch.tensor(10.0, requires_grad=True)
(ta * tb + tc).tanh().backward()

print(a.grad, ta.grad.item())
print(b.grad, tb.grad.item())
assert abs(a.grad - ta.grad.item()) < 1e-6
```

**When that assert passes you have written backpropagation.** Everything PyTorch does is this, over
tensors instead of scalars, with fused kernels.

### Lab 4 — Softmax and cross-entropy in your own autograd

Now build the actual loss out of your `Value` class:

```python
def softmax(logits):
    m = max(v.data for v in logits)                 # shift for stability
    exps = [(v - m).exp() for v in logits]
    total = sum(exps[1:], exps[0])
    return [e / total for e in exps]

def cross_entropy(logits, target_idx):
    return -softmax(logits)[target_idx].log()

logits = [Value(2.0), Value(1.0), Value(0.1), Value(-1.0)]
loss = cross_entropy(logits, 1)
loss.backward()
print(loss.data, [f"{v.grad:+.4f}" for v in logits])
```

Compare the gradients to `softmax(z) − onehot(target)`, which is the closed form:

```python
p = torch.softmax(torch.tensor([2.0, 1.0, 0.1, -1.0]), dim=-1)
p[1] -= 1
print(p)                                            # must match your grads
```

**Look at the signs.** The true token's gradient is negative (push its logit *up*); every other
token's is positive (push them *down*). The magnitude of each wrong token's gradient is exactly the
probability the model gave it — so the confidently-wrong tokens get corrected hardest. That is the
learning signal, visible.

### Lab 5 — Descend a loss surface by hand

```python
w = Value(5.0)
for step in range(50):
    loss = (w - 3.0) ** 2          # minimum obviously at w = 3
    w.grad = 0.0                   # ← zero_grad, by hand
    loss.backward()
    w.data -= 0.1 * w.grad
    if step % 10 == 0:
        print(f"step {step:2d}  w={w.data:.4f}  loss={loss.data:.6f}")
```

Then run the experiment that teaches learning rate in thirty seconds — set `lr` to `0.01`, `0.1`,
`0.9`, `1.0`, and `1.1`:

| lr | Behaviour |
|----|-----------|
| 0.01 | Converges, slowly |
| 0.1 | Converges cleanly |
| 0.9 | Oscillates, still converges |
| 1.0 | Bounces forever, never converges |
| 1.1 | **Diverges to infinity** |

**You have just seen every learning-rate failure you will meet at 3.4M parameters, on one
parameter.** Also delete the `w.grad = 0.0` line and watch the run fall apart — that is what a
missing `zero_grad()` does.

---

## Failure modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| `nan` immediately | `exp` of a large logit | Subtract the max before exponentiating |
| Initial loss far below `ln(V)` | Labels leaking into inputs | Check your shift: inputs `[:-1]`, targets `[1:]` |
| Initial loss far above `ln(V)` | Init scale too large | Small init (≈0.02); confident-wrong is worse than uniform |
| Loss decreases then explodes | Learning rate above the stability threshold | Lower it; add warmup (B6) |
| Gradients subtly wrong, model trains badly | `=` instead of `+=` in backward | Accumulate, always |
| Loss stuck exactly flat | Forgot to zero gradients, or no `requires_grad` | Zero each step; check the graph is connected |
| Passed probabilities to `cross_entropy` | It expects logits | Pass raw logits |

---

## Interview

**"Why cross-entropy?"**
It isn't a design choice, it falls out of maximum likelihood. You want parameters that make the
observed text as probable as possible; that's a product of per-token probabilities, which underflows,
so you take the log and get a sum; optimisers minimise, so you negate; you average over tokens so
the number doesn't depend on sequence length. What you're left with is cross-entropy. The per-token
form is just negative log probability of the correct token — nothing about the wrong tokens appears
in the formula, they're punished only through softmax's normalisation.

**"What loss should an untrained model have?"**
`ln(vocab_size)` — about 8.3 for a 4k vocabulary, 10.8 for GPT-2's 50k. It's the first thing I check
on any new training run. Materially below it at step zero means labels are leaking into the inputs,
usually an off-by-one in the shift; materially above it means initialisation is too large, because
confidently wrong is worse than uniform. It costs nothing to check and it catches the two bugs that
waste the most time.

**"Explain backpropagation."**
It's the chain rule applied to a computational graph, arranged so that all the derivatives come out
of one backward pass instead of one pass per parameter. You record every operation as you go
forward, then walk the graph in reverse topological order, and at each node multiply the incoming
gradient by that operation's local derivative and accumulate it into the inputs. The two details
that matter in practice are that gradients must accumulate rather than overwrite, since a node can
feed several downstream consumers — which is also why you have to zero gradients each step — and
that the reverse ordering has to be a genuine topological sort, or you'll compute a node's gradient
before everything downstream has contributed.

**"What's perplexity?"**
`exp` of the cross-entropy loss, and it reads as the effective number of tokens the model is
choosing between. It's more interpretable than raw loss for that reason. The caveat is that it's
only comparable across models sharing a tokenizer — different vocabularies mean different numbers of
tokens for the same text, so the per-token average isn't measuring the same thing. For cross-
tokenizer comparison you need bits-per-byte.

---

## Checkpoint

1. Write the chain-rule factorisation of `P(sequence)` from memory.
2. Derive cross-entropy from maximum likelihood in four steps.
3. State the expected initial loss for `vocab=4096` and both bugs a deviation would indicate.
4. Explain why softmax subtracts the max, and show the overflow it prevents.
5. Your `Value` class matches PyTorch gradients to 1e-6.
6. Your hand-built cross-entropy gradient equals `softmax(z) − onehot(target)`.
7. Reproduce the five-row learning-rate table and name the divergence threshold.

---

**Next:** [B1 — Tensors, and your first language model](01-first-model.md) ·
**Back:** [Build Track](README.md) · [Syllabus](../../SYLLABUS.md)
