# B3 — Data: corpus to batches

Everything so far ran on a 1 MB file held in RAM. Real pretraining does not, and the difference is
not just size — it is document boundaries, packing, memory-mapped storage, and a validation split
that isn't lying to you.

This is the least glamorous lesson in the track and the one where a silent bug costs you the most.
A model trained on subtly corrupted data trains *fine*. The loss goes down. You find out at
evaluation, days later.

By the end you will have a tokenized TinyStories corpus on disk and a batch loader you trust.

**You write** `code/b3_data.py` · **you record in** `notes/b3-data.md` · **reference companion**
[M13 — Fine-tuning dataset construction](../13-datasets.md) ·
[M06 — Pretraining](../06-pretraining.md)

```bash
cp notes/_template.md notes/b3-data.md
touch code/b3_data.py
```

---

## Terms

| Term | Meaning |
|------|---------|
| **Corpus** | The raw text you train on. |
| **TinyStories** | Synthetic children's stories with a deliberately small vocabulary. |
| **Memory-mapped file (memmap)** | A file on disk addressed as if it were an array. |
| **uint16** | 2-byte unsigned integer. Holds vocabularies up to 65,535. |
| **Packing** | Concatenating documents into one stream and cutting fixed-length blocks. |
| **Document boundary** | Where one document ends and the next begins. |
| **`<eos>`** | The token marking that boundary. |
| **Document masking** | Preventing attention from crossing a boundary inside a packed block. |
| **Train/val split** | Held-out data used only for measuring, never for training. |
| **Leakage** | Validation content appearing in training. Makes val loss meaningless. |
| **Epoch** | One full pass over the corpus. |
| **Tokens seen** | The metric that actually matters — `steps × batch × block`. |
| **Shard** | One file of a corpus split across many. |
| **Deterministic sampling** | Seeded batch selection, so a run is reproducible. |

---

## Concepts

### Why TinyStories

A 3.4M-parameter model trained on Wikipedia produces syntactic mush. The same model trained on
TinyStories writes grammatical, coherent English with a plot.

The reason is vocabulary and structure, not model capacity. TinyStories was generated with a
constrained word list — roughly what a four-year-old knows — and simple sentence structure. It
removes the long tail that small models cannot afford to learn, leaving exactly the grammar and
narrative structure they *can*.

**This is the finding that makes the whole Build Track viable.** Data quality and data–model fit
matter more than parameter count at this scale, which is the same lesson Module 06 teaches about
deduplication and Module 13 teaches about mixtures — you just get to feel it directly.

### Packing, and the `<eos>` that must not be forgotten

Documents vary in length; training needs fixed-size blocks. The universal solution is to concatenate
everything into one long token stream and slice fixed-length windows from it.

```
doc1 <eos> doc2 <eos> doc3 <eos> doc4 <eos> …
└──── block 0 ────┘
        └──── block 1 ────┘
```

Blocks cut across document boundaries. That is fine and it is deliberate — no padding, no wasted
compute, every position trains.

**But `<eos>` between documents is mandatory, for two reasons.** Without it, the model learns that
one story flows into an unrelated one, which teaches incoherence. And it never learns to stop — at
generation time it runs to your token limit every single time, because it has never seen a sequence
end. A model that will not terminate is the single most common symptom of a forgotten `<eos>`, and
you cannot fix it with sampling parameters.

### Document masking, and why we skip it here

Strictly, a packed block lets position 200 attend to the tail of the previous document at position
50. Correct implementations pass a block-diagonal attention mask so attention cannot cross an
`<eos>`.

**We deliberately do not do this in B5.** The effect is small when documents are short relative to
the block (TinyStories averages ~200 tokens against a 256 block), the implementation complicates
attention meaningfully, and skipping it lets you *measure* the cost later rather than assume it. It
is flagged in B6's failure table and in Module 13, and adding it is a genuinely good exercise once
the base model trains.

Knowing that you made this trade-off deliberately, and being able to say what it costs, is worth
more than silently doing it right.

### uint16 and memmap

Two decisions that seem like plumbing and are not.

**uint16.** Token IDs fit in 2 bytes for any vocabulary below 65,536. Storing them as int64 — the
default if you are careless — is **4× the disk and 4× the read bandwidth** for identical data. With
a 4096 vocabulary, uint16 is correct and free.

```python
assert vocab_size < 2**16, "uint16 cannot hold this vocabulary"
```

**Memmap.** A 100M-token corpus is 200 MB as uint16 — that fits in RAM, but the habit matters
because at 10B tokens it does not. `np.memmap` addresses the file as an array and the OS pages in
only what you touch. Random access to a random offset reads one page, not the file.

The one trap: **re-open the memmap each batch, or you leak memory.** The mapping accumulates pages
in the page cache attributed to your process, and a long training run grows without bound. This is a
real bug that people hit and misdiagnose as a memory leak in their model.

### The validation split

Split by **document**, never by token offset, and never randomly at the character level.

Splitting mid-document leaks: the model trains on the first half of a story and is evaluated on the
second half, so val loss measures memorisation of that specific story rather than generalisation.
Your val loss looks great and means nothing.

Hold out whole documents, from the end of the file, and never touch them:

```python
split_at = int(0.995 * len(documents))
train_docs, val_docs = documents[:split_at], documents[split_at:]
```

0.5% of TinyStories is still thousands of stories — plenty to measure with, and a 200× cheaper
evaluation than 10%.

### Tokens seen is the metric, not steps

"I trained for 10,000 steps" says nothing without batch size and block size. The comparable number
is:

```
tokens_seen = steps × batch_size × block_size
```

Your `base` model at `batch=32, block=256` sees 8,192 tokens per step. Chinchilla-optimal for 3.4M
parameters is roughly `20 × 3.4M ≈ 68M tokens`, so **~8,300 steps.** In practice you will train
longer than that — over-training a small model is the right call when inference matters (Module 06),
and you have the tokens.

Log `tokens_seen` on your x-axis from the beginning. B8's scaling law needs it, and any comparison
across model sizes is meaningless without it.

---

## Where it's used

- Every pretraining run. The data pipeline is written once and reused for years.
- Packing and `<eos>` behaviour directly determine whether your model can stop (B7).
- The train/val discipline is what makes B6's loss curves interpretable.
- `tokens_seen` is the x-axis of B8's scaling law.
- The same masking question returns as loss masking in SFT (Module 11, B9).

---

## Labs

### Lab 1 — Get the corpus

```bash
pip install datasets numpy tqdm
```

```python
from datasets import load_dataset
ds = load_dataset("roneneldan/TinyStories", split="train")
print(len(ds), ds[0]["text"][:300])
```

Take a subset you can tokenize in a few minutes:

```python
docs = [ds[i]["text"] for i in range(200_000)]
chars = sum(len(d) for d in docs)
print(f"{len(docs):,} stories, {chars:,} chars, ~{chars/4:,.0f} tokens")
```

**Report the average document length in tokens.** You need it to reason about the 256-token block:
if the average story is ~200 tokens, most blocks contain one or two documents, which is exactly the
regime where skipping document masking is defensible.

### Lab 2 — Retrain the tokenizer on this corpus

Your B2 tokenizer was trained on Shakespeare. B2 Lab 6 showed what happens when a tokenizer meets
out-of-domain text — do not now make that mistake yourself.

```python
sample = "\n".join(docs[:20_000])              # enough to learn merges; all of it is wasteful
chunks = re.findall(PAT, sample)
merges, vocab = train_bpe(chunks, 4096 - len(SPECIALS))
tok = Tokenizer(merges, vocab)

print("bytes/token:", len(sample.encode()) / len(tok.encode(sample)))
```

**Compare against the Shakespeare-trained tokenizer on this same text.** The gap is the cost of a
mismatched tokenizer, measured on your own corpus — and it is the argument for why tokenizer
training data is a decision you cannot revise later.

Save it, because everything downstream depends on this exact artefact:

```python
import json, pathlib
pathlib.Path("tokenizer.json").write_text(json.dumps({
    "merges": {f"{a},{b}": i for (a, b), i in merges.items()},
    "specials": special_ids,
}))
```

### Lab 3 — Tokenize to a memmap

```python
import numpy as np
from tqdm import tqdm

EOS = special_ids["<eos>"]

def write_shard(docs, path):
    ids = []
    for d in tqdm(docs):
        ids.extend(tok.encode(d))
        ids.append(EOS)                        # ← the line that must not be forgotten
    arr = np.array(ids, dtype=np.uint16)
    arr.tofile(path)
    return len(arr)

split_at = int(0.995 * len(docs))
n_train = write_shard(docs[:split_at], "train.bin")
n_val   = write_shard(docs[split_at:], "val.bin")
print(f"train {n_train:,} tokens ({n_train*2/1e6:.0f} MB)   val {n_val:,}")
```

Then verify — do not skip this, it is the whole point of the lesson:

```python
m = np.memmap("train.bin", dtype=np.uint16, mode="r")
print("eos count:", (m[:1_000_000] == EOS).sum())          # ≈ 1M / avg_doc_len
print("max id:", m.max(), "vocab:", 4096)
assert m.max() < 4096
print(tok.decode([int(x) for x in m[:200]]))               # must read as English
```

**Three checks, three different bugs.** `<eos>` count near zero means you forgot the separator. A
max ID at or above vocab means an encoding bug and training will crash on an embedding lookup.
Decoded text that isn't English means your merge order is wrong.

### Lab 4 — The batch loader

```python
import torch

def get_batch(split, batch_size=32, block_size=256, device="cpu"):
    # re-open every call — do NOT hoist this out of the function
    data = np.memmap(f"{split}.bin", dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i     : i + block_size    ].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + block_size + 1].astype(np.int64)) for i in ix])
    if device != "cpu":
        x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x, y

xb, yb = get_batch("train")
assert xb.shape == (32, 256) and yb.shape == (32, 256)
assert torch.equal(xb[0, 1:], yb[0, :-1])          # the shift, proven
print(tok.decode(xb[0, :60].tolist()))
```

**That `torch.equal` assert is the off-by-one check from B0**, and it belongs in your code
permanently. Then measure throughput:

```python
import time
t0 = time.time()
for _ in range(100):
    get_batch("train")
print(f"{100 * 32 * 256 / (time.time() - t0):,.0f} tokens/sec loading")
```

If loading is slower than your model's forward+backward, you are data-bound and the GPU is idle.
**Know this number before you blame the model.**

### Lab 5 — Prove the leakage bug

The most instructive lab here, because it produces a *good-looking wrong answer*.

```python
# WRONG: split by token offset, mid-document
all_ids = np.memmap("train.bin", dtype=np.uint16, mode="r")
cut = int(0.9 * len(all_ids))
bad_train, bad_val = all_ids[:cut], all_ids[cut:]

# WORSE: overlapping — 10% of val also appears in train
overlap_val = all_ids[cut - 100_000:]
```

Train the B1 bigram on each split arrangement for 2000 steps and compare val loss:

| Split | Val loss | Honest? |
|-------|----------|---------|
| By document (correct) | | ✅ |
| By token offset | | Mid-document leakage |
| Overlapping windows | | Directly memorised |

**The dishonest splits give lower val loss.** That is the trap: the bug improves the number you are
using to judge quality, so nothing alerts you. Every subsequent decision — when to stop, which
architecture is better — is then made on a corrupted signal.

### Lab 6 — Document boundaries inside a block

```python
data = np.memmap("train.bin", dtype=np.uint16, mode="r")
block = data[:256]
positions = np.where(block == EOS)[0]
print("eos at positions:", positions)

frac = sum((get_batch("train")[0] == EOS).any(dim=1).float().mean().item() for _ in range(50)) / 50
print(f"{frac:.1%} of sequences cross at least one document boundary")
```

**Report that percentage.** It is the size of the approximation you are accepting by skipping
document masking, and it turns a hand-wave into a number you can defend. If it is above ~60%,
implementing block-diagonal masking in B5 becomes worth it — and you will know because you measured.

### Lab 7 — Plan the run

```python
def plan(n_params, batch=32, block=256, tokens_per_sec=None):
    chinchilla = 20 * n_params
    tokens_per_step = batch * block
    steps = chinchilla / tokens_per_step
    print(f"params {n_params:,}")
    print(f"chinchilla-optimal tokens: {chinchilla:,.0f}")
    print(f"tokens/step: {tokens_per_step:,}   → {steps:,.0f} steps")
    if tokens_per_sec:
        print(f"≈ {chinchilla / tokens_per_sec / 60:.0f} min")

for n in (1.1e6, 3.4e6, 12e6):
    plan(n)
```

Also check you have enough data: `chinchilla_tokens ≤ n_train` or you are doing multiple epochs.
**Note how many epochs each ladder rung implies** — B8 needs this, and repeated epochs on a small
corpus is itself a Module 06 topic (memorisation rises with duplication, which is also Module 25's
privacy lab).

---

## Failure modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| Model never stops generating | No `<eos>` between documents | Append `<eos>` per document; retokenize |
| Stories run into each other incoherently | Same | Same |
| Val loss suspiciously low | Split mid-document, or overlapping | Split by whole documents |
| Val loss ≈ train loss forever | Val is drawn from train | Check the split code, not the model |
| Memory grows over a long run | Memmap hoisted out of `get_batch` | Re-open each call |
| `IndexError` in embedding lookup | Token ID ≥ vocab_size | `assert data.max() < vocab_size` after tokenizing |
| Corpus 4× larger than expected | Stored as int64 | uint16 for vocab < 65,536 |
| Loss is fine, samples are garbage | Merge order wrong; data decodes to noise | Decode the first 200 tokens and read them |
| GPU underutilised | Data-bound | Measure loader tokens/sec against step time |
| Runs not reproducible | Unseeded batch sampling | Seed; log the seed |

---

## Interview

**"How do you prepare a pretraining corpus?"**
Tokenize with a tokenizer trained on representative text, append an end-of-sequence token after
every document, concatenate into one stream, and store as uint16 in a memory-mapped file. Then
sample fixed-length windows at random offsets. Packing means no padding and every position trains.
The two details that bite are the `<eos>` — without it the model never learns to stop, and that's
unfixable at sampling time — and splitting validation by whole documents rather than by offset,
because a mid-document split lets the model train on the first half of a story and be evaluated on
the second, so val loss measures memorisation and looks *better* than the honest number.

**"What's the cost of packing documents together?"**
Attention can cross a document boundary inside a block, so a position can attend to the tail of an
unrelated document. The correct fix is a block-diagonal mask so attention never crosses an `<eos>`.
On my own corpus I measured what fraction of sequences actually cross a boundary — with 200-token
stories in a 256-token block it's substantial, but the model still learns fine, so I treated it as a
known, measured approximation rather than pretending it wasn't there. At long context with short
documents it matters much more and I'd implement the mask.

**"How do you decide how long to train?"**
In tokens seen, not steps — steps are meaningless without batch and block size. Chinchilla says
roughly 20 tokens per parameter for a training-compute-optimal model, so about 68M tokens for a
3.4M-parameter model. But training-optimal isn't deployment-optimal: if the model will be served
many times, you over-train well past Chinchilla to get a smaller model at the same quality, which is
what Llama 3 did at around 2,000 tokens per parameter. I log tokens seen on the x-axis from step
one, because otherwise nothing is comparable across model sizes.

**"What data bug have you found the hardest to catch?"**
Anything that makes validation loss *better*. A leaky split, overlapping windows, contamination — the
metric you'd use to catch the problem is the metric the problem improves, so nothing alerts you, and
every downstream decision gets made on a corrupted signal. I deliberately reproduced the leaky split
on my own corpus and confirmed it gives a lower val loss than the correct one. After that I check
split construction before I check anything else.

---

## Checkpoint

1. Report corpus size, document count, and mean document length in tokens.
2. Show the compression gap between a matched and a mismatched tokenizer on this corpus.
3. Pass all three memmap checks: `<eos>` count, max ID, and decoded text that reads as English.
4. Have `torch.equal(xb[0,1:], yb[0,:-1])` permanently in your loader.
5. Report loader throughput in tokens/sec.
6. Show that the leaky split produces a *lower* val loss than the correct one.
7. Report what fraction of sequences cross a document boundary, and defend your masking decision.
8. State your Chinchilla token budget and the implied step count for all three ladder rungs.

---

**Next:** [B4 — Attention, from one head to many](04-attention.md) ·
**Back:** [B2 — The tokenizer](02-tokenizer.md) · [Build Track](README.md)
