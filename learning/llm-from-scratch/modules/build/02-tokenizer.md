# B2 — The tokenizer, from scratch

The character tokenizer in B1 worked, and it is the reason your bigram model needed 8 characters of
context to see two words. Real models tokenize into sub-word pieces, and that decision shapes
everything downstream: sequence length, effective context, embedding parameter count, arithmetic
ability, and how the model handles a language it barely saw in training.

By the end of this lesson you will have implemented Byte-Pair Encoding from nothing, trained a
4096-token vocabulary on your own corpus, and measured the trade-off you just made.

The reference companion is [Module 01](../01-tokenization.md) — read it after this, not before.

---

## Terms

| Term | Meaning |
|------|---------|
| **Tokenization** | Splitting text into the units the model reads. |
| **Byte-level** | The base alphabet is the 256 possible bytes, not characters. |
| **BPE** | Byte-Pair Encoding: repeatedly merge the most frequent adjacent pair. |
| **Merge rule** | An ordered pair `(a, b) → new_id`. The training output. |
| **Merge order** | The sequence of merges. Encoding must replay it in the same order. |
| **Vocabulary** | The mapping from token ID to byte string. |
| **Special token** | A reserved ID with no text spelling: `<bos>`, `<eos>`, `<pad>`. |
| **Fertility** | Tokens per word. Lower is better for the same text. |
| **Compression ratio** | Bytes per token. Your headline number. |
| **Pre-tokenization** | Splitting on a regex *before* BPE, so merges never cross word boundaries. |
| **Unknown token / UNK** | A token for unrepresentable input. Byte-level BPE **never needs one**. |
| **Roundtrip** | `decode(encode(s)) == s`. Non-negotiable. |
| **Token-free byte model** | Skipping tokenization entirely. Longer sequences, no vocabulary. |

---

## Concepts

### Why not characters, and why not words

| Approach | Vocabulary | Sequence length | Problem |
|----------|------------|-----------------|---------|
| Characters | ~100 | Very long | Attention is O(T²); 256 tokens is 256 characters — 40 words |
| Words | 500k+ | Short | Enormous embedding table; every typo and rare word is UNK |
| **Sub-word (BPE)** | **4k–200k** | **Middle** | **The compromise everyone uses** |

The pressure is two-sided and you can feel it in your own model. Attention cost grows as `T²`, so
shorter sequences are worth a lot. But the embedding matrix is `V × d_model`, so a large vocabulary
costs parameters — for you, `4096 × 192 = 786,432`, already **23% of the entire model**. Push the
vocabulary to 32k and embeddings alone would be 6.1M, nearly twice the size of everything else
combined.

**At small scale, vocabulary size is a first-order architecture decision, not a detail.** That is a
thing you can only really learn by having a parameter budget you care about.

### Byte-level: why there is no UNK

Start from bytes, not characters. Every possible input — any language, emoji, corrupted encoding,
binary garbage — is a sequence of bytes, and all 256 of them are in the base vocabulary.

**Therefore any input is encodable, and there is no unknown token.** This is why byte-level BPE won.
The failure mode it eliminates is not rare: a word-level tokenizer meeting a language it has never
seen produces a sequence of UNKs and the model is blind.

The cost is that unusual input tokenizes badly — a language absent from your training corpus may
land near one byte per token, so the same sentence costs several times more tokens. That is not a
crash, it is a tax, and it is why token pricing is effectively higher for some languages.

### The algorithm

Training:

```
1. Start with the 256 byte values as tokens 0–255.
2. Count every adjacent pair in the corpus.
3. Find the most frequent pair.
4. Mint a new token for it; record the merge rule.
5. Replace all occurrences of that pair with the new token.
6. Repeat until you reach the target vocabulary size.
```

Encoding replays the merges **in training order** — this is the part people get wrong. The merges
are not a set, they are an ordered list, and applying them out of order gives a different (still
decodable, but inconsistent) tokenization.

Decoding is a lookup: each ID maps to a byte string, concatenate, decode UTF-8.

### Pre-tokenization: the step that isn't in the algorithm

Run BPE directly on raw text and it will happily merge across word boundaries — `" the"` and
`" of"` become single tokens, but so does `"dog. The"`. That wastes vocabulary on artefacts of your
particular corpus and generalises badly.

Every real tokenizer splits on a regex first, then runs BPE **inside** each piece:

```python
import regex as re
PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
```

Read the alternatives: contractions, a run of letters with an optional leading space, a run of
digits with an optional leading space, punctuation, then whitespace.

Two consequences worth knowing before you meet them as bugs:

- **The leading space belongs to the word.** `" the"` and `"the"` are different tokens. This is why
  a prompt ending in a trailing space tokenizes differently and often generates worse — you have
  handed the model an orphaned space and asked it to continue.
- **Digits split individually in modern tokenizers.** If `"1234"` is one token, the model must learn
  arithmetic per-number rather than per-digit. Splitting digits measurably improves arithmetic, and
  it is a deliberate choice, not a side effect.

### What you are choosing when you choose 4096

| Vocab | Embedding params (`d=192`) | Share of a 3.4M model | Bytes/token |
|-------|----------------------------|-----------------------|-------------|
| 256 (bytes only) | 49,152 | 1.4% | 1.0 |
| 1024 | 196,608 | 6% | ~2.8 |
| **4096** | **786,432** | **23%** | **~3.5** |
| 16384 | 3,145,728 | 48% | ~4.2 |

Diminishing returns on compression, linear growth in parameters. **4096 is where the curve bends
for a corpus this size** — and you will produce this exact table yourself in Lab 5 rather than
taking my word for it.

### Special tokens

Reserve IDs that no byte sequence can produce:

| Token | Purpose |
|-------|---------|
| `<bos>` | Beginning of sequence. The seed for generation. |
| `<eos>` | End. **The model must learn to emit this or it never stops.** |
| `<pad>` | Padding, masked out of the loss. |
| `<user>` `<assistant>` | Chat roles — B9 needs these, so reserve them now. |

Reserve them **now**, at the end of the vocabulary. Adding tokens later means resizing the embedding
matrix and invalidating every checkpoint you have trained.

---

## Where it's used

- Every model. The tokenizer is loaded before the weights and outlives them.
- Sequence length, and therefore attention cost and KV cache size (Modules 03, 19).
- Embedding parameter count — 23% of your model.
- Chat templates are just special tokens in an agreed order (Module 09, B9).
- Loss masking works on token boundaries, so mask bugs are usually tokenizer bugs (Module 11).
- Perplexity is per-token, so it is **not comparable across tokenizers** (Module 06).

---

## Labs

### Lab 1 — BPE training, from nothing

```python
from collections import Counter

def get_stats(ids_list):
    counts = Counter()
    for ids in ids_list:
        counts.update(zip(ids, ids[1:]))
    return counts

def merge(ids, pair, new_id):
    out, i = [], 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            out.append(new_id)
            i += 2
        else:
            out.append(ids[i])
            i += 1
    return out

def train_bpe(chunks, vocab_size, verbose=True):
    """chunks: list of pre-tokenized strings."""
    ids_list = [list(c.encode("utf-8")) for c in chunks]
    merges, vocab = {}, {i: bytes([i]) for i in range(256)}

    for new_id in range(256, vocab_size):
        stats = get_stats(ids_list)
        if not stats:
            break
        pair = max(stats, key=stats.get)
        ids_list = [merge(ids, pair, new_id) for ids in ids_list]
        merges[pair] = new_id
        vocab[new_id] = vocab[pair[0]] + vocab[pair[1]]
        if verbose and new_id % 256 == 0:
            print(f"{new_id}: {vocab[new_id]!r}  (count {stats[pair]})")
    return merges, vocab
```

Run it with pre-tokenization on tinyshakespeare, `vocab_size=4096`, and **read the merge log**.

```python
text = open("input.txt").read()
chunks = re.findall(PAT, text)
merges, vocab = train_bpe(chunks, 4096)
```

**The first merges will be `th`, `he`, `in`, `an`, `ou` — English bigram frequency, discovered from
the data with no linguistics anywhere in the algorithm.** By token 1000 you will see whole common
words. By 3000, character names from the corpus. Watching that progression is the point of the lab.

### Lab 2 — Encode, decode, and the assert that matters

```python
class Tokenizer:
    def __init__(self, merges, vocab):
        self.merges, self.vocab = merges, vocab

    def _encode_chunk(self, text):
        ids = list(text.encode("utf-8"))
        while len(ids) >= 2:
            stats = get_stats([ids])
            # the merge that was learned EARLIEST wins — order is the algorithm
            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            ids = merge(ids, pair, self.merges[pair])
        return ids

    def encode(self, text):
        return [i for chunk in re.findall(PAT, text) for i in self._encode_chunk(chunk)]

    def decode(self, ids):
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")
```

```python
tok = Tokenizer(merges, vocab)
for s in ["hello world", "To be, or not to be", "café 🎉 日本語", "", "   ", "a" * 500]:
    assert tok.decode(tok.encode(s)) == s, f"roundtrip failed: {s!r}"
print("roundtrip OK")
```

**The emoji and Japanese cases must pass even though neither appears in Shakespeare.** That is
byte-level BPE's guarantee, and confirming it on characters your tokenizer has never seen is the
whole argument for the design.

Note the `min` over merge index in `_encode_chunk`. Using `max` by frequency instead — the obvious-
looking mistake — produces a tokenizer that still roundtrips but disagrees with its own training,
so a token can be split differently at train and inference time. Try it and watch the compression
ratio get worse.

### Lab 3 — Measure what you built

```python
enc = tok.encode(text)
print(f"bytes:  {len(text.encode('utf-8')):>9,}")
print(f"tokens: {len(enc):>9,}")
print(f"compression: {len(text.encode('utf-8')) / len(enc):.2f} bytes/token")

words = len(text.split())
print(f"fertility: {len(enc) / words:.2f} tokens/word")
```

Then compare against B1's character tokenizer on the same 256-token context:

| Tokenizer | Vocab | Bytes/token | **Characters in a 256-token context** |
|-----------|-------|-------------|---------------------------------------|
| Character (B1) | 65 | 1.0 | 256 |
| **BPE 4096** | 4096 | ~3.5 | **~900** |

**Same attention cost, 3.5× more text.** That is what you bought, quantified — and it is exactly
why your B5 model with a 256-token window will see a whole TinyStories story where B1's saw a
sentence fragment.

### Lab 4 — Add special tokens

```python
SPECIALS = ["<pad>", "<bos>", "<eos>", "<user>", "<assistant>"]
base = 4096 - len(SPECIALS)
special_ids = {s: base + i for i, s in enumerate(SPECIALS)}
```

Retrain BPE with `vocab_size=base` so the specials occupy the top of the range, then verify the
property that makes them special:

```python
assert all(sid not in tok.encode("<bos> hello") for sid in special_ids.values()), \
    "special tokens must never be produced by encoding ordinary text"
```

**This assert is a security control, not housekeeping.** If a user's text can encode to `<assistant>`,
they can forge turn boundaries in your chat template — the injection surface of Module 25, appearing
here in its simplest form. Real tokenizers handle specials by splitting them out *before* BPE and
mapping them directly.

### Lab 5 — The vocabulary-size decision, measured

Train at 512, 1024, 2048, 4096, 8192, 16384 and fill in the trade:

```python
for V in (512, 1024, 2048, 4096, 8192, 16384):
    m, vb = train_bpe(chunks, V, verbose=False)
    t = Tokenizer(m, vb)
    n_tok = len(t.encode(text))
    emb = V * 192
    print(f"V={V:>6}  bytes/tok {len(text.encode()) / n_tok:.2f}  "
          f"emb params {emb:>9,}  ctx256 covers {256 * len(text.encode()) / n_tok:>6.0f} bytes")
```

Plot bytes-per-token against vocabulary size. **You are looking for the knee — compression gains
flatten while parameters keep growing linearly.** Then state, in one sentence, why 4096 is the right
choice for a 3.4M-parameter model and would be the wrong choice for a 7B one.

### Lab 6 — Break it deliberately

Four experiments, each one a real bug you will otherwise meet later:

```python
# 1. No pre-tokenization — merges cross word boundaries
m1, v1 = train_bpe([text], 4096, verbose=False)
print([v1[i] for i in range(4000, 4020)])     # look for tokens spanning punctuation + space

# 2. Trailing space
print(tok.encode("The cat sat on the"), tok.encode("The cat sat on the "))

# 3. Digits
for s in ["1234", "12 34", "3.14159"]:
    print(s, [tok.decode([i]) for i in tok.encode(s)])

# 4. Out-of-domain text
shakespeare_ratio = len(text.encode()) / len(tok.encode(text))
code = "def f(x):\n    return x ** 2\n" * 100
print("shakespeare", round(shakespeare_ratio, 2),
      "code", round(len(code.encode()) / len(tok.encode(code)), 2))
```

**Experiment 4 is the one to remember: your tokenizer's compression collapses on text unlike its
training corpus.** A tokenizer trained on Shakespeare handles Python badly, and the model pays for
it in context length forever. This is why tokenizer training data should match deployment data, and
why it is a decision you cannot revise after pretraining.

### Lab 7 — Retokenize the corpus and rerun the bigram

Close the loop. Re-run B1's bigram model on BPE tokens instead of characters:

| Model | Vocab | Initial loss | Final loss | Bytes of context |
|-------|-------|--------------|------------|------------------|
| Bigram, chars | 65 | 4.17 | ~2.45 | 1 |
| Bigram, BPE 4096 | 4096 | **8.32** | ? | ~3.5 |

**Two things will surprise you and both are instructive.** The BPE loss is much higher in absolute
terms — but it is not comparable, because it is per *token* and each token now carries 3.5× more
text. And the generated text is noticeably more word-like even though the model is architecturally
identical, because the tokenizer is doing work the model used to have to do.

Convert both to **bits per byte** to compare them honestly:

```python
bpb = loss / math.log(2) / bytes_per_token
```

That is the only fair cross-tokenizer comparison, and now you know why Module 06 insists on it.

---

## Failure modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| `decode(encode(s)) != s` | Merge order not replayed correctly | `min` by merge index, not `max` by frequency |
| Unicode crashes | Decoding partial multi-byte sequences | `errors="replace"` on decode; operate on bytes throughout |
| Tokens spanning `". "` | No pre-tokenization | Split on the regex before BPE |
| Prompt with trailing space generates poorly | Orphaned space token | Strip trailing whitespace before encoding |
| Bad arithmetic | Multi-digit numbers as single tokens | Split digits in pre-tokenization |
| Compression collapses on real data | Tokenizer corpus ≠ deployment corpus | Train the tokenizer on representative text |
| User text forges a role token | Specials reachable from ordinary encoding | Split specials out before BPE; assert unreachable |
| Perplexity beats a published number | Comparing across tokenizers | Use bits-per-byte |
| Need a new token after pretraining | Vocabulary is fixed at pretraining | Reserve specials up front |

---

## Interview

**"How does BPE work?"**
You start from the 256 byte values, count every adjacent pair in the corpus, merge the most frequent
one into a new token, and repeat to your target vocabulary size. The output is an *ordered* list of
merge rules, and encoding replays them in that order — that ordering is the part people miss, and
getting it wrong gives you a tokenizer that disagrees with its own training data. Byte-level is what
matters in practice: because every input is bytes and all 256 are in the base vocabulary, there's no
unknown token and nothing is unencodable. The cost is that text unlike the training corpus tokenizes
close to one byte per token, which is a tax rather than a failure.

**"How do you choose vocabulary size?"**
It's a trade between sequence length and parameters, and the balance depends entirely on model size.
Attention is quadratic in sequence length, so better compression is worth a lot; but the embedding
matrix is vocab times d_model, so a large vocabulary is expensive. On the 3.4M model I built, a 4096
vocabulary is already 23% of all parameters — going to 16k would make embeddings half the model. At
7B those same embeddings are noise, so you take the compression. I measured the curve on my own
corpus: bytes-per-token flattens out while parameter count keeps growing linearly, and the knee for a
small model sits around 4k.

**"What's the most under-appreciated tokenizer bug?"**
Trailing whitespace. The leading space belongs to the word — `" the"` and `"the"` are different
tokens — so a prompt ending in a space hands the model an orphaned space token and generation
degrades, with no error anywhere. Related: multi-digit numbers as single tokens hurt arithmetic,
which is why modern tokenizers split digits deliberately. And the security-relevant one is special
tokens being reachable from ordinary text — if a user's input can encode to your assistant-role
token they can forge turn boundaries in the chat template, so specials have to be split out before
BPE ever runs and I'd assert that property in a test.

**"Why can't you compare perplexity across models?"**
Because perplexity is per token and different tokenizers cut the same text into different numbers of
tokens. A model with better compression is averaging over fewer, harder predictions, so the number
isn't measuring the same thing. Bits-per-byte normalises to the underlying text and is the only
honest cross-tokenizer comparison. I hit this directly when I moved my own bigram model from
characters to BPE — absolute loss went from 2.45 to something much higher while the generated text
got visibly better.

---

## Checkpoint

1. Implement `train_bpe`, and quote the first five merges it discovers.
2. Roundtrip on emoji and Japanese, neither of which is in your training corpus.
3. Report bytes/token and fertility, and how many bytes fit in a 256-token context.
4. Reserve special tokens and assert ordinary text cannot produce them.
5. Produce the vocab-size curve and defend 4096 for a 3.4M model.
6. Show the four deliberate breakages, especially compression on out-of-domain text.
7. Rerun the bigram on BPE tokens and compare to characters **in bits per byte**.

---

**Next:** [B3 — Data: corpus to batches](03-data.md) ·
**Back:** [B1 — Your first language model](01-first-model.md) · [Build Track](README.md)
