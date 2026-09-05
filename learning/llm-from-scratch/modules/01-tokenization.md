# Module 01 — Tokenization

The layer everyone skips. Then they cannot explain why their model fails at arithmetic, why their
non-English users burn 3× the tokens, or why a fine-tune produced fluent garbage.

**Why it's first:** a language model never sees text. It sees integers. Every downstream
behaviour — context limits, cost, arithmetic, multilingual quality, fine-tuning corruption — is
shaped by how that mapping was built.

---

## Terms

| Term | Meaning |
|------|---------|
| **Token** | The atomic unit a model processes. An integer index into a vocabulary. Usually a word fragment, not a word. |
| **Vocabulary** | The fixed set of tokens a model knows. Qwen3 ≈ 151k, Llama 3 ≈ 128k, GPT-2 ≈ 50k. |
| **BPE** (Byte-Pair Encoding) | Build a vocab by repeatedly merging the most frequent adjacent pair of symbols in the corpus. The dominant algorithm. |
| **Byte-level BPE** | BPE over raw UTF-8 *bytes* rather than characters. Guarantees no input is ever unrepresentable. |
| **Merge rules** | The ordered list of pair-merges learned during training. Encoding replays them in order — which is why order matters and the list must ship with the model. |
| **WordPiece** | BERT-era variant; picks merges by likelihood gain rather than raw frequency. |
| **SentencePiece** | A *library*, not an algorithm — implements BPE and Unigram, treats input as a raw stream, encodes spaces as `▁`. Common in Llama/Mistral lineage. |
| **Unigram** | Alternative: start with a large vocab, prune tokens that cost least likelihood. Probabilistic, supports sampling different segmentations. |
| **Pre-tokenizer** | Regex split applied *before* BPE — separates whitespace, punctuation, digits. Determines what merges are even possible. |
| **Special token** | A token with a reserved role, not learned from corpus text: `<|im_start|>`, `<|endoftext|>`. |
| **BOS / EOS / PAD / UNK** | Beginning-of-sequence, end-of-sequence, padding, unknown. Byte-level BPE has no UNK by construction. |
| **Token healing** | Fixing the boundary artefact when a prompt ends mid-token, so continuation isn't forced into a bad split. |
| **Fertility** | Average tokens per word. English ≈ 1.3; many other languages 2–4× worse on the same tokenizer. |
| **Tokenizer mismatch** | Using tokenizer A with model B. Produces fluent, confident nonsense — no error is raised. |
| **Vocabulary overlap** | How much two tokenizers share. Governs whether cross-model distillation on logits is even meaningful. |

---

## Concepts

### Why not characters or words?

**Characters:** vocabulary of ~100, but sequences become enormous. Attention is O(n²) in sequence
length — see Module 03 — so a 4× longer sequence is 16× the attention cost. Also forces the model
to spend capacity relearning spelling.

**Words:** sequences are short, but the vocabulary is unbounded. Every typo, every proper noun,
every compound is out-of-vocabulary. And the embedding matrix is `vocab_size × d_model` — a 1M-word
vocab at d_model=2048 is 2B parameters in embeddings alone.

**Subwords are the compromise:** common words stay whole, rare words decompose into pieces. The
vocabulary is bounded, and nothing is unrepresentable.

### How BPE training actually works

Start with the alphabet (for byte-level: all 256 byte values). Then repeat until you hit the target
vocab size:

1. Count every adjacent symbol pair across the corpus.
2. Take the most frequent pair.
3. Merge it into a new symbol; record the merge rule.

So `("l","o")` merging into `"lo"`, then `("lo","w")` into `"low"`. The output is the vocabulary
**plus the ordered merge list**. Encoding new text replays those merges in learned order — this is
why the merge list ships with the model and why you cannot reorder it.

### Byte-level: the trick that eliminates UNK

Operate on UTF-8 bytes, not characters. Any string is a byte sequence, every byte is in the base
vocab, so **every possible input is encodable**. No `UNK` token, ever. Emoji, Chinese, control
characters, corrupted data — all encode, sometimes inefficiently, never impossibly.

The cost: one non-Latin character can consume 3–4 byte-tokens. This is the mechanical origin of
the multilingual token-cost gap, and you will measure it in Lab 2.

### Where the pre-tokenizer decides everything

Before BPE runs, a regex splits the text. GPT-2's pattern keeps a leading space attached to a word,
which is why `"hello"` and `" hello"` are **different tokens**. Modern tokenizers also split digits
into individual tokens — that decision alone is most of why arithmetic works at all (Lab 3).

Consequence you will hit in production: a prompt ending with a trailing space tokenizes differently
from one without, and can measurably change output. It is not superstition; it is a different
integer sequence.

---

## Where it's used

- **Cost and context.** APIs bill per token. "8k context" is 8k tokens, not words — and not the
  same amount of text in Spanish as in English.
- **Fine-tuning.** Your training data must be tokenized with the model's own tokenizer. Adding
  domain tokens means resizing the embedding matrix and training the new rows from scratch.
- **Serving.** KV-cache memory is per token. Fertility directly multiplies your serving cost.
- **Evaluation.** Perplexity is per token, so **perplexity is not comparable across tokenizers**.
  A model with worse fertility can show lower perplexity while being worse. This is a real trap in
  model-comparison write-ups.
- **Distillation.** Logit-level distillation between models requires shared vocabulary. Different
  tokenizers force sequence-level distillation instead (Module 16).
- **Structured output.** Grammar-constrained decoding operates on token boundaries, so a JSON
  grammar has to be compiled against a specific tokenizer (Module 10).

---

## Labs

### Lab 1 — Look at what the model actually sees

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")

print("vocab size:", tok.vocab_size)
print("specials:", tok.all_special_tokens)

def show(text: str) -> None:
    ids = tok.encode(text, add_special_tokens=False)
    pieces = [tok.decode([i]) for i in ids]
    print(f"\n{text!r}")
    print(f"  {len(ids)} tokens: {ids}")
    print(f"  pieces: {pieces}")

show("Hello world")
show(" Hello world")          # leading space -> different ids
show("hello world")           # case -> different ids
show("indamutsa")             # rare string -> fragments
show("The quick brown fox jumps over the lazy dog")
```

**Observe:** `"Hello"` vs `" Hello"` are different tokens. Rare proper nouns fragment. Note the
fragment count for your own surname — that fragmentation is why models are bad at rare entities.

### Lab 2 — Fertility across languages

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")

samples = {
    "English":    "The company announced a new product yesterday in the capital city.",
    "Spanish":    "La empresa anunció ayer un nuevo producto en la ciudad capital.",
    "French":     "L'entreprise a annoncé hier un nouveau produit dans la capitale.",
    "German":     "Das Unternehmen kündigte gestern ein neues Produkt in der Hauptstadt an.",
    "Kinyarwanda":"Ikigo cyatangaje ejo umusaruro mushya mu murwa mukuru.",
    "Chinese":    "该公司昨天在首都发布了一款新产品。",
    "Arabic":     "أعلنت الشركة أمس عن منتج جديد في العاصمة.",
}

print(f"{'lang':<14}{'chars':>7}{'words':>7}{'tokens':>8}{'tok/word':>10}{'chars/tok':>11}")
for lang, text in samples.items():
    ids = tok.encode(text, add_special_tokens=False)
    words = len(text.split())
    print(f"{lang:<14}{len(text):>7}{words:>7}{len(ids):>8}"
          f"{len(ids)/words:>10.2f}{len(text)/len(ids):>11.2f}")
```

**Expected:** English ~1.2–1.5 tokens/word. Kinyarwanda and Arabic substantially worse. Chinese has
few "words" by whitespace but high tokens-per-character.

**The point:** identical meaning, different cost, different effective context. A "128k context"
model gives a Kinyarwanda user materially less room than an English one, and they pay more per
sentence for it. This is a fairness and cost issue you can speak about credibly.

### Lab 3 — Where tokenization breaks arithmetic

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")

for n in ["7", "42", "127", "1024", "31415", "1000000", "3.14159"]:
    ids = tok.encode(n, add_special_tokens=False)
    print(f"{n:>10}  {len(ids)} tokens  {[tok.decode([i]) for i in ids]}")
```

Then look at what the model does with it:

```python
from common.load import load
import torch

tok, model = load("Qwen/Qwen3-1.7B")

def complete(prompt: str, n: int = 12) -> str:
    ids = tok(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=n, do_sample=False,
                             pad_token_id=tok.eos_token_id)
    return tok.decode(out[0][ids["input_ids"].shape[1]:], skip_special_tokens=True)

for p in ["2 + 2 = ", "47 + 58 = ", "1234 + 5678 = ", "98765 + 43210 = "]:
    print(f"{p!r:>22} -> {complete(p)!r}")
```

**What to see:** accuracy falls as digit count rises. A model that splits numbers into
inconsistent chunks has to learn addition separately for every chunking pattern it encounters.
Consistent single-digit splitting is a deliberate tokenizer design choice made precisely to fix
this — check which one Qwen3 does, and reason about what you observed.

### Lab 4 — Break a model with a tokenizer mismatch

The failure mode that raises no error.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B",
                                             torch_dtype=torch.bfloat16).to("mps")
right = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
wrong = AutoTokenizer.from_pretrained("gpt2")          # different vocabulary entirely

prompt = "The capital of France is"

for name, t in [("correct", right), ("MISMATCHED", wrong)]:
    ids = t(prompt, return_tensors="pt").to("mps")
    ids["input_ids"] = ids["input_ids"].clamp(max=model.config.vocab_size - 1)
    with torch.no_grad():
        out = model.generate(**ids, max_new_tokens=20, do_sample=False,
                             pad_token_id=model.config.eos_token_id)
    print(f"\n[{name}] {right.decode(out[0], skip_special_tokens=True)!r}")
```

**Observe:** no exception, no warning — just confident nonsense. This is what a wrong
`tokenizer_config.json` in a serving deployment looks like, and why "the model got worse after the
deploy" is sometimes a tokenizer problem, not a weights problem.

### Lab 5 — Train BPE from scratch

Nothing makes the algorithm concrete like implementing it.

```python
"""Minimal BPE. Not efficient — legible."""
from collections import Counter


def get_pairs(seqs):
    pairs = Counter()
    for seq, freq in seqs.items():
        for a, b in zip(seq, seq[1:]):
            pairs[(a, b)] += freq
    return pairs


def merge(seqs, pair):
    a, b = pair
    out = {}
    for seq, freq in seqs.items():
        new, i = [], 0
        while i < len(seq):
            if i < len(seq) - 1 and seq[i] == a and seq[i + 1] == b:
                new.append(a + b)
                i += 2
            else:
                new.append(seq[i])
                i += 1
        out[tuple(new)] = freq
    return out


def train_bpe(corpus: str, n_merges: int = 50):
    words = Counter(corpus.lower().split())
    seqs = {tuple(w) + ("</w>",): f for w, f in words.items()}
    merges = []
    for _ in range(n_merges):
        pairs = get_pairs(seqs)
        if not pairs:
            break
        best = max(pairs, key=pairs.get)
        merges.append((best, pairs[best]))
        seqs = merge(seqs, best)
    return merges, seqs


corpus = (
    "the quick brown fox jumps over the lazy dog " * 20
    + "the quicker browner foxes jumped over the lazier dogs " * 10
    + "lower lowest slower slowest newer newest " * 15
)

merges, final = train_bpe(corpus, n_merges=40)
for i, (pair, count) in enumerate(merges):
    print(f"{i:>3}  {pair[0]!r:>12} + {pair[1]!r:<12} (seen {count})  ->  {pair[0]+pair[1]!r}")

print("\nfinal segmentations:")
for seq, freq in sorted(final.items(), key=lambda kv: -kv[1])[:12]:
    print(f"  {freq:>4}  {' '.join(seq)}")
```

**Observe:** early merges are the most frequent character pairs (`th`, `er`, `ow`). Later merges
assemble whole frequent words. `</w>` marks word ends so `"er"` inside `lower` and `"er</w>"` at a
word end can be distinguished. Watch `"low"` and `"est"` emerge as reusable units — that
compositionality is the entire value of BPE.

Then increase `n_merges` to 200 and watch full words absorb the vocabulary. That trade — more
merges means shorter sequences but a bigger embedding matrix — is the vocab-size decision every
model architect makes.

---

## Failure modes

| Symptom | Likely cause | How to confirm |
|---------|--------------|----------------|
| Fluent, confident nonsense after a deploy | Tokenizer mismatch | Encode a fixed string with both tokenizers, diff the ids |
| Non-English users hit context limits early | Fertility | Lab 2 on their actual text |
| Arithmetic fails above N digits | Digit chunking | Lab 3 |
| Fine-tune degrades a previously fine capability | Training data tokenized differently from inference (e.g. chat template applied at train but not at eval) | Diff the exact token ids of one training example against one inference prompt |
| Added domain tokens produce gibberish | New embedding rows are randomly initialised and undertrained | Check gradient norms on the new rows; initialise from the mean of existing embeddings |
| Perplexity comparison flatters the wrong model | Different tokenizers | Compare bits-per-byte instead |
| Generation degrades when prompt ends mid-word | Token boundary artefact | Apply token healing |

---

## Interview

**"Why do LLMs struggle with arithmetic?"**
Partly tokenization. If numbers split inconsistently, the model must learn addition separately per
chunking pattern rather than as one algorithm. Consistent digit-level splitting helps measurably.
The rest is that next-token prediction has no carry mechanism — the model computes in a fixed
number of forward-pass layers, so multi-digit carrying has to be either memorised or externalised
into chain-of-thought tokens. That is why "think step by step" improves arithmetic: it converts a
depth-limited computation into a sequential one.

**"What breaks if I swap the tokenizer and keep the weights?"**
Everything, silently. Token id *k* indexes row *k* of the embedding matrix; different tokenizers
assign different meanings to *k*. You get fluent output because the language-modelling head is
intact, and wrong output because every input embedding is the wrong vector. It raises no
exception, which is what makes it dangerous.

**"How would you add domain vocabulary to a model?"**
Weigh it first — usually you should not. Adding tokens means resizing the embedding and LM head,
and the new rows start random while everything else is converged, so they undertrain and produce
degraded output on exactly the domain terms you added them for. Mitigations: initialise new rows
from the mean of their existing subword pieces, and use a higher LR on those rows. Do it only when
fertility on your domain is genuinely pathological — measure first, and remember it forecloses
logit-level distillation from any model with the original vocab.

**"Why is perplexity not comparable across models?"**
Perplexity is per token, and tokens differ. A coarser tokenizer packs more text per token, so each
prediction is harder and perplexity rises even at equal quality. Compare bits-per-byte, which
normalises to the underlying text.

---

## Checkpoint

You can move on when you can:

1. State your own name's token count and explain the fragmentation.
2. State the fertility ratio between English and Kinyarwanda on Qwen3, from your own measurement.
3. Explain byte-level BPE's guarantee that there is no UNK token.
4. Describe the tokenizer-mismatch failure signature without looking it up.
5. Explain why perplexity is not comparable across tokenizers.

---

**Next:** [02 — Embeddings and the residual stream](02-embeddings.md) ·
**Back:** [00 — Setup](00-setup.md) · [Syllabus](../SYLLABUS.md)
