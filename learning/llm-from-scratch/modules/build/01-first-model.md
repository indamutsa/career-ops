# B1 — Tensors, and your first language model

B0 gave you scalars. Real models move millions of numbers at once, and the difference between code
that trains in a minute and code that trains in a day is entirely about doing that in batches
rather than loops.

By the end of this lesson you will have trained a working language model on real text and generated
samples from it. The model will be bad — that is deliberate. It establishes the baseline every
later model must beat, and it builds the training loop skeleton you will reuse unchanged for the
next eight lessons.

**You write** `code/b1_bigram.py` · **you record in** `notes/b1-first-model.md` ·
**reference companion** [M02 — Embeddings and the residual stream](../02-embeddings.md)

```bash
cp notes/_template.md notes/b1-first-model.md
touch code/b1_bigram.py
```

---

## Terms

| Term | Meaning |
|------|---------|
| **Tensor** | An n-dimensional array. Scalars, vectors, matrices, and beyond, uniformly. |
| **Shape** | The size along each dimension. `(8, 16, 4096)` is batch 8, sequence 16, vocab 4096. |
| **Rank / ndim** | The number of dimensions. |
| **dtype** | Element type: `float32`, `bfloat16`, `int64`. |
| **Device** | Where the tensor lives: `cpu`, `mps`, `cuda`. |
| **Broadcasting** | Automatic expansion of size-1 dimensions so shapes align. |
| **Contiguous** | Whether elements are laid out in memory in row-major order. |
| **`view` vs `reshape`** | `view` requires contiguity and never copies; `reshape` may copy. |
| **Batch** | Several independent sequences processed together. |
| **`B, T, C`** | The universal naming convention: Batch, Time (sequence), Channels (features). |
| **Embedding** | A lookup table mapping token ID → vector. Just a matrix indexed by row. |
| **`nn.Module`** | PyTorch's base class; tracks parameters and submodules. |
| **Parameter** | A tensor registered for training. |
| **Optimizer** | Applies the update rule using the gradients. |
| **AdamW** | Adam with decoupled weight decay. The default for language models. |
| **Bigram model** | Predicts the next token from the previous token alone. |
| **Baseline** | The score a new approach must beat to have earned its complexity. |

---

## Concepts

### Tensors, and why shapes are the whole game

Most bugs you will hit in this track are shape bugs, and most of them are silent — the code runs,
the loss looks plausible, the model is wrong. Learning to read shapes is the single highest-return
skill in this lesson.

Adopt the convention now and never deviate:

```
B = batch size        how many independent sequences
T = time / block      how many tokens per sequence
C = channels          the feature dimension (d_model)
V = vocab size
```

A model's forward pass is then legible as a sequence of shape transformations:

```
tokens        (B, T)          int64 IDs
embeddings    (B, T, C)       each ID replaced by its vector
… layers …    (B, T, C)       shape is preserved all the way through
logits        (B, T, V)       one score per vocabulary entry, per position
```

**The shape is constant through the entire stack.** Every transformer layer is `(B,T,C) → (B,T,C)`.
That invariant is what lets you stack six of them, or sixty, without changing anything else — and
it is the residual stream of Module 02, seen as a shape.

### Broadcasting, and the bug it causes

Broadcasting aligns shapes from the right, expanding size-1 dimensions:

```python
a = torch.randn(4, 3)
b = torch.randn(3)          # treated as (1, 3), expanded to (4, 3)
(a + b).shape               # (4, 3)
```

Convenient, and the source of a specific silent failure: an operation you *meant* to be elementwise
becoming an outer product.

```python
x = torch.randn(4, 1)
y = torch.randn(4)          # NOT (4, 1)
(x + y).shape               # (4, 4) — not what you wanted, no error raised
```

**Defence: assert your shapes.** In a training loop this costs nothing and catches everything:

```python
assert logits.shape == (B, T, V), f"got {logits.shape}"
```

### Embeddings are a lookup table, not a transformation

An embedding layer looks sophisticated and is not. `nn.Embedding(V, C)` holds a `(V, C)` matrix.
Indexing it with token ID `k` returns row `k`. That's all.

```python
emb = nn.Embedding(4096, 192)
emb.weight.shape                      # (4096, 192)
ids = torch.tensor([[5, 100, 2000]])  # (1, 3)
emb(ids).shape                        # (1, 3, 192)
torch.equal(emb(ids)[0, 0], emb.weight[5])    # True — it is row 5
```

It is equivalent to multiplying a one-hot vector by the matrix, just without the wasted arithmetic.
Those rows are learned like any other parameter, and Module 02 is about what they end up meaning.

### The bigram model

Your first language model predicts the next token from the current token and nothing else. It
throws away all context beyond one step — a deliberately terrible assumption, and it makes the
model trivially simple:

**A `(V, V)` table where entry `[i, j]` is the logit for token `j` following token `i`.**

That is one `nn.Embedding(V, V)`. No hidden layers, no nonlinearity. Given token `i`, look up row
`i`, and those are your logits over what comes next.

This is worth building for three reasons, and the third is the real one:

1. It exercises the full pipeline — data, forward, loss, backward, step, generate — with a model too
   simple to hide a bug.
2. It gives you the **baseline loss**. Every later architecture must beat it or it has not earned
   its parameter count.
3. **The optimal bigram model is computable in closed form by counting.** So you can verify that
   gradient descent actually converges to the right answer — which almost nobody does, and which
   turns "training works" from an article of faith into something you have checked.

### The training loop, once and for all

Every training run in this curriculum, up to and including GRPO, is this shape:

```python
for step in range(max_steps):
    xb, yb = get_batch("train")        # 1. data
    logits, loss = model(xb, yb)       # 2. forward
    optimizer.zero_grad(set_to_none=True)   # 3. clear old gradients  ← B0's rule
    loss.backward()                    # 4. backward: fill .grad everywhere
    optimizer.step()                   # 5. update parameters
```

Five lines. Learn them in this order, because from here on only step 2 changes.

`set_to_none=True` sets gradients to `None` rather than zero-filling — slightly faster, and it makes
a forgotten `backward()` fail loudly instead of silently reusing stale gradients.

### Why AdamW rather than plain SGD

B0's update was `θ ← θ − lr · ∇L`. That works, but converges slowly on the loss surfaces language
models have, where different parameters need very different step sizes.

Adam keeps two running averages per parameter — the mean of recent gradients (momentum) and the mean
of their squares — and divides the step by the square root of the second. Parameters with
consistently large gradients get smaller steps; consistently small gradients get larger ones. It is
**per-parameter adaptive learning rates**.

The practical consequence, which returns in Module 17: Adam stores two extra float32 numbers per
parameter. That is where "16 bytes per parameter" comes from — 4 for the parameter, 4 for its
gradient, 8 for Adam's two states. Your 3.4M model therefore needs ~55 MB to train, not ~14 MB.

The "W" is decoupled weight decay: shrink weights toward zero as a separate step rather than by
adding a penalty to the loss. It regularises correctly under adaptive step sizes, where the naive
version does not.

---

## Where it's used

- The `B, T, C` convention appears in every model file you will ever read.
- The five-line loop is unchanged through B9 and through Modules 11–16.
- Shape assertions are the cheapest debugging tool in deep learning.
- Bigram-as-baseline is the same discipline as Module 15's "beat rejection-sampling SFT before
  claiming GRPO worked" — always know the number your complexity has to beat.

---

## Labs

### Lab 0 — Environment and corpus

```bash
python -m venv .venv && source .venv/bin/activate
pip install torch numpy matplotlib
curl -o input.txt https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
wc -c input.txt        # ≈ 1,115,394 bytes
```

```python
import torch
print(torch.__version__, torch.backends.mps.is_available())
torch.manual_seed(1337)
```

**Use CPU for this lesson.** At this size MPS is slower — kernel launch overhead dominates — and
CPU error messages are far more useful. B6 switches over.

### Lab 1 — Shapes, deliberately broken

```python
x = torch.randn(2, 3, 4)
print(x.shape, x.ndim, x.dtype, x.device)
print(x.view(6, 4).shape)
print(x.transpose(0, 1).shape)                  # (3, 2, 4)

y = x.transpose(0, 1)
try:
    y.view(24)
except RuntimeError as e:
    print("expected:", e)                       # not contiguous
print(y.reshape(24).shape)                      # works — copies
```

Now cause the silent bug on purpose and prove the fix:

```python
a = torch.randn(4, 1)
b = torch.randn(4)
print((a + b).shape)                            # (4, 4) — wrong, and silent
print((a + b.unsqueeze(1)).shape)               # (4, 1) — what you meant
```

**Write down which one surprised you.** That is the bug you will make at 3 a.m. in B5.

### Lab 2 — Character tokenizer

A real BPE tokenizer is B2. For now, characters — enough to get text flowing.

```python
text = open("input.txt").read()
chars = sorted(set(text))
V = len(chars)                                   # 65 for tinyshakespeare
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join(itos[i] for i in l)

assert decode(encode("hello world")) == "hello world"
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data, val_data = data[:n], data[n:]
print(V, data.shape, data.dtype)
```

**The roundtrip assert is not optional.** A tokenizer that cannot reproduce its input is the single
most expensive bug in this track, because everything downstream trains happily on corrupted data.
You will write this same assert again in B2.

Note `V = 65` here versus `4096` later — expected initial loss changes accordingly:
`ln(65) ≈ 4.17`.

### Lab 3 — Batching, and the shift

```python
BLOCK, BATCH = 8, 4

def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = torch.randint(len(d) - BLOCK, (BATCH,))
    x = torch.stack([d[i     : i + BLOCK]     for i in ix])
    y = torch.stack([d[i + 1 : i + BLOCK + 1] for i in ix])
    return x, y

xb, yb = get_batch("train")
print(xb.shape, yb.shape)                        # (4, 8) both
```

Now make the shift concrete — this is the "shifted right" of Module 05a, and the label-masking of
Module 11, in its simplest form:

```python
for t in range(BLOCK):
    print(f"context {xb[0, :t+1].tolist()}  →  target {yb[0, t].item()}")
```

**One batch of shape `(4, 8)` contains 32 independent training examples**, because every prefix
position is a prediction. That density is why transformers train efficiently and why the causal
mask (B4) is worth the trouble.

### Lab 4 — The bigram model

```python
import torch.nn as nn
import torch.nn.functional as F

class BigramLM(nn.Module):
    def __init__(self, V):
        super().__init__()
        self.table = nn.Embedding(V, V)          # row i = logits for what follows token i

    def forward(self, idx, targets=None):
        logits = self.table(idx)                 # (B, T, V)
        if targets is None:
            return logits, None
        B, T, V_ = logits.shape
        loss = F.cross_entropy(logits.view(B * T, V_), targets.view(B * T))
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            logits = logits[:, -1, :]            # last position only
            probs = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, nxt], dim=1)
        return idx

m = BigramLM(V)
_, loss = m(xb, yb)
print(loss.item(), math.log(V))                  # ≈ 4.17 for V=65
```

**Check that number before training anything.** It is B0's `ln(V)` test, and this is the first
place it earns its keep.

Note the `.view(B*T, V)` — `F.cross_entropy` wants `(N, classes)`, so the batch and time dimensions
get flattened together. Every position is just another example.

Then look at what an untrained model produces:

```python
start = torch.zeros((1, 1), dtype=torch.long)
print(decode(m.generate(start, 300)[0].tolist()))
```

Pure noise. Keep it — the before/after is the point.

### Lab 5 — Train it

```python
opt = torch.optim.AdamW(m.parameters(), lr=1e-2)

@torch.no_grad()
def estimate_loss(model, iters=200):
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(iters)
        for k in range(iters):
            X, Y = get_batch(split)
            _, losses[k] = model(X, Y)
        out[split] = losses.mean().item()
    model.train()
    return out

BATCH = 32
for step in range(10000):
    xb, yb = get_batch("train")
    _, loss = m(xb, yb)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()
    if step % 1000 == 0:
        print(step, estimate_loss(m))
```

**Expected: ~4.17 → ~2.45.** Then generate again:

```python
print(decode(m.generate(start, 400)[0].tolist()))
```

You will get pronounceable nonsense — plausible letter pairs, no words, no structure. **That is
exactly what a model with one token of context should produce**, and being able to predict that
from the architecture is the lesson.

Record your final train and val loss. **This is the baseline for the entire track.**

### Lab 6 — Verify gradient descent found the right answer

The lab that makes this more than a tutorial. The optimal bigram model can be computed by counting,
so you can check SGD against the exact solution.

```python
counts = torch.zeros((V, V))
for a, b in zip(data[:-1], data[1:]):
    counts[a, b] += 1

probs_exact = (counts + 1e-8) / (counts + 1e-8).sum(dim=1, keepdim=True)
probs_learned = F.softmax(m.table.weight, dim=-1)

print("max abs diff:", (probs_exact - probs_learned).abs().max().item())
print("mean abs diff:", (probs_exact - probs_learned).abs().mean().item())

nll = -torch.log(probs_exact[data[:-1], data[1:]]).mean()
print("theoretical optimum:", nll.item(), " your model:", estimate_loss(m)["train"])
```

**Your trained model should land within ~0.02 of the theoretical optimum.** That is empirical proof
that the optimiser converged to the true minimum — not an assumption, a measurement. Nothing later
in this track is verifiable this cleanly, which is exactly why it is worth doing once.

### Lab 7 — The ceiling, and why context matters

Extend the counting to trigrams and compute the optimal loss without training anything:

```python
tri = torch.zeros((V, V, V))
for a, b, c in zip(data[:-2], data[1:-1], data[2:]):
    tri[a, b, c] += 1
p3 = (tri + 1e-8) / (tri + 1e-8).sum(dim=-1, keepdim=True)
print("trigram optimum:", -torch.log(p3[data[:-2], data[1:-1], data[2:]]).mean().item())
```

| Context | Optimal loss | Parameters |
|---------|--------------|------------|
| Unigram (0 tokens) | | `V` |
| Bigram (1 token) | ~2.45 | `V²` = 4,225 |
| Trigram (2 tokens) | | `V³` = 274,625 |
| 4-gram (3 tokens) | (try it) | `V⁴` = 17.8 M |

**Two things fall out of that table, and both motivate everything that follows.** Loss drops
steadily as context grows — context is genuinely informative. And parameters grow as `V^n`, so
n-gram counting hits a wall almost immediately: a 256-token context would need `65^256` entries.

**The transformer's claim is that it can use 256 tokens of context with 3.4 million parameters
instead of `65^256`.** That is the problem it solves, and now you have measured both ends of it.

---

## Failure modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| Silent `(4, 4)` where you wanted `(4, 1)` | Broadcasting an outer product | `unsqueeze`; assert shapes |
| `view() ... not compatible` | Tensor not contiguous after transpose | `.contiguous().view()` or `.reshape()` |
| Loss starts near 0 | Targets leaking into inputs | Check the `+1` in `get_batch` |
| Loss starts far above `ln(V)` | Init scale too large | Default `nn.Embedding` init is fine; check you didn't scale it |
| Loss flat from step 0 | Missing `backward()`, or optimizer built before the model moved device | Build the optimizer *after* `.to(device)` |
| Loss decreases then `nan` | Learning rate too high | Lower it; B6 adds warmup and clipping |
| Generation returns one repeated token | Taking argmax, or sampling from logits instead of probs | `multinomial` on softmaxed values |
| `decode(encode(s)) != s` | Tokenizer bug | Fix before anything else — everything downstream is corrupt |

---

## Interview

**"Walk me through a training step."**
Fetch a batch of inputs and targets, where targets are the inputs shifted one position. Forward pass
gives logits of shape batch × time × vocab, flattened to N × vocab against flattened targets for
cross-entropy. Zero the gradients — necessary because PyTorch accumulates by design, which is what
makes gradient accumulation possible in the first place. Backward fills `.grad` on every parameter
in one pass. Then the optimizer applies the update. That's it, and it doesn't change from a bigram
model to GRPO; only the forward pass and the loss get more interesting.

**"Why AdamW instead of SGD?"**
Per-parameter adaptive step sizes. Adam keeps running averages of the gradient and its square, and
divides the step by the root of the second, so parameters with consistently large gradients take
smaller steps and vice versa. On language-model loss surfaces that converges much faster than a
single global learning rate. The cost is memory: two extra float32 states per parameter, which is
where the sixteen-bytes-per-parameter rule for training comes from — four for the weight, four for
the gradient, eight for Adam. The W is decoupled weight decay, applied as a separate shrink rather
than a loss penalty, which is the correct form under adaptive step sizes.

**"Why start with a bigram model?"**
Because it gives me a baseline I can verify. It exercises the whole pipeline with a model too
simple to hide a bug, and unusually the optimum is computable in closed form by counting — so I can
confirm gradient descent converged to the true minimum rather than assuming it. Then I can extend
the counting to trigrams and 4-grams without training anything, and watch loss fall as context grows
while the parameter count grows as vocabulary to the power of context length. That table is the
argument for the transformer: same or better use of context, at a parameter count that doesn't
explode.

**"What's the most common bug in this kind of code?"**
Shape bugs that don't raise. Broadcasting will happily turn an intended elementwise operation into
an outer product and the code runs, the loss looks plausible, and the model is wrong. Close second
is an off-by-one in the input/target shift, which shows up as an implausibly low loss at step zero
— which is why I always check the first loss against `ln(vocab)` before letting anything run.

---

## Checkpoint

1. State what `B`, `T`, `C` and `V` mean and give the shape at each stage of a forward pass.
2. Produce the broadcasting bug and its fix.
3. Show `decode(encode(s)) == s`.
4. Explain why one `(4, 8)` batch is 32 training examples.
5. Report bigram initial loss vs `ln(V)`, and final train/val loss. **This is your baseline.**
6. Show your trained bigram is within ~0.02 of the counted optimum.
7. Produce the context-length table and state what it implies about n-grams.
8. Recite the five-line training loop from memory.

---

**Next:** [B2 — The tokenizer, from scratch](02-tokenizer.md) ·
**Back:** [B0 — Foundations](00-foundations.md) · [Build Track](README.md)
