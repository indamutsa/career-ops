# Module 05a — The encoder–decoder transformer, and reading Figure 1

Module 05 built a decoder-only model, because that is what you will fine-tune, serve, and be paid
to work on. But the diagram everyone learns from — Figure 1 of *Attention Is All You Need* — is not
that model. It has two stacks, three kinds of attention, and a normalisation placement no current
model uses.

You will be shown that figure in an interview. This module is about being able to walk it box by
box, say which parts survived, and name the two places where the picture actively misleads.

---

## Terms

| Term | Meaning |
|------|---------|
| **Encoder stack** | Bidirectional layers producing one contextual vector per input token. |
| **Decoder stack** | Causal layers producing the output autoregressively. |
| **Self-attention** | Q, K, V all from the same sequence. |
| **Masked self-attention** | Self-attention with a causal mask — position *i* sees only ≤ *i*. |
| **Cross-attention** | **Q from the decoder, K and V from the encoder output.** The bridge between stacks. |
| **Memory / encoder output** | The encoder's final hidden states, consumed by every decoder layer. |
| **Shifted right** | The decoder input is the target sequence offset by one, so position *i* predicts token *i*. |
| **Teacher forcing** | Feeding ground-truth prefixes during training rather than the model's own outputs. |
| **Post-norm** | `x = LayerNorm(x + Sublayer(x))` — the original. Norm *after* the residual add. |
| **Pre-norm** | `x = x + Sublayer(LayerNorm(x))` — every modern model. Norm *inside* the branch. |
| **Warmup** | Ramping the learning rate from ~0 over the first few thousand steps. |
| **Add & Norm** | The residual add plus normalisation — the yellow boxes in the figure. |
| **Nx** | The stack is *N* identical layers deep. Encoder and decoder have separate counts. |
| **Encoder-only** | BERT family — bidirectional, no generation. |

---

## Concepts

### The figure, box by box

Read it as two columns. Left is the encoder, right is the decoder.

**Left column — encoder, repeated N×:**

```
Inputs → Input Embedding → + Positional Encoding
  ├─ Multi-Head Attention (self, bidirectional — no mask)
  ├─ Add & Norm
  ├─ Feed Forward
  └─ Add & Norm
→ memory
```

**Right column — decoder, repeated N×:**

```
Outputs (shifted right) → Output Embedding → + Positional Encoding
  ├─ Masked Multi-Head Attention (self, causal)
  ├─ Add & Norm
  ├─ Multi-Head Attention  ← the arrows arriving from the left are K and V from memory
  ├─ Add & Norm
  ├─ Feed Forward
  └─ Add & Norm
→ Linear → Softmax → Output Probabilities
```

**The single most important detail in the whole diagram is those two arrows crossing from the
encoder into the decoder's middle attention block.** That is cross-attention, it is the only place
the two stacks communicate, and it is the box that does not exist in GPT, Llama, Qwen, or anything
else you will fine-tune.

Note also that the arrows leave the encoder *once*, at the top, and feed *every* decoder layer.
There is no layer-to-layer pairing between the stacks.

### Cross-attention, precisely

Self-attention: `Q, K, V = xW_q, xW_k, xW_v` — all from the same `x`.

Cross-attention: **`Q` from the decoder's current hidden state, `K` and `V` from the encoder's
output.**

```python
class CrossAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.q = nn.Linear(d_model, d_model, bias=False)   # from decoder
        self.k = nn.Linear(d_model, d_model, bias=False)   # from encoder memory
        self.v = nn.Linear(d_model, d_model, bias=False)   # from encoder memory
        self.o = nn.Linear(d_model, d_model, bias=False)
        self.n_heads, self.d_head = n_heads, d_model // n_heads

    def forward(self, x, memory, memory_mask=None):
        B, T, _ = x.shape
        S = memory.shape[1]                       # source length ≠ target length
        shape = lambda t, L: t.view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        q = shape(self.q(x), T)
        k = shape(self.k(memory), S)
        v = shape(self.v(memory), S)
        # NO causal mask here — every target position may see the entire source.
        out = F.scaled_dot_product_attention(q, k, v, attn_mask=memory_mask)
        return self.o(out.transpose(1, 2).reshape(B, T, -1))
```

Three consequences that make it different from self-attention:

- **The attention matrix is rectangular** — `T × S`, target length by source length. Self-attention
  is always square.
- **No causal mask.** Hiding the future is about the *output* being generated; the input was given
  in full. Masking cross-attention is a classic bug that produces a model that trains and
  translates badly.
- **K and V are computed once for the whole generation.** The encoder runs once; every decoding
  step reuses the same memory. This is a genuine efficiency advantage over stuffing the source into
  a decoder-only context, and it is why encoder-decoders remain competitive for translation.

The interpretation is worth holding onto: **cross-attention is retrieval over the source.** For
each output position the decoder asks "which input tokens matter for what I'm producing now?" In
translation those attention maps are close to a soft word alignment, which is exactly what the
architecture was designed to learn.

### "Outputs (shifted right)" is teacher forcing drawn as a wire

The bottom-right label confuses nearly everyone. The decoder's *input* at training time is the
target sequence shifted one position, so that position *i* sees tokens `< i` and predicts token
*i*. A `<bos>` fills the first slot.

```
target :  Le   chat  est  noir  <eos>
input  : <bos>  Le   chat est   noir
```

This is exactly the label-shift you implemented in Module 11, and exactly the exposure bias
Modules 11 and 16 keep returning to: at training the prefix is always correct, at inference it is
whatever the model produced. Same idea, drawn as a box.

### The trap: the figure is post-norm, your code is pre-norm

The original places normalisation *after* the residual addition:

```python
x = LayerNorm(x + Attention(x))     # post-norm — the figure's "Add & Norm"
```

Every model you will actually work on does the opposite:

```python
x = x + Attention(RMSNorm(x))       # pre-norm — Module 05, and every modern config
```

| | Post-norm | Pre-norm |
|---|---|---|
| Residual path | Normalised at every layer | **Clean, unnormalised all the way through** |
| Deep training | Needs warmup; diverges without it | Trains stably at depth |
| Final quality at equal depth | Slightly better when it converges | Slightly worse, but it converges |
| Used by | The 2017 paper, BERT | GPT-2 onward, Llama, Qwen, everything |

The mechanism is Module 02's residual stream. Pre-norm keeps a clean additive highway from
embeddings to logits — gradients reach layer 1 without passing through *N* normalisations.
Post-norm interrupts that path at every layer, which is why the original needed a warmup schedule
and why deep post-norm transformers were notoriously fragile.

**So the diagram is a historical document, not a specification.** Being able to say that — and
explain *why* the change was made — is a stronger answer than reciting the boxes.

### Why the encoder disappeared, and where it didn't

Decoder-only won for reasons that are mostly practical rather than theoretical (Module 07):

- One stack, one objective, any text is training data.
- Prefix-as-input works: put the source in the context and let causal attention read it.
- One KV cache, one serving path, one set of kernels to optimise.
- It scales without the architectural decisions an encoder-decoder forces (how deep is each stack?
  which layers cross-attend?).

But the encoder did not die — it moved:

| Still encoder | Where |
|---------------|-------|
| Encoder-decoder | T5/FLAN-T5, Whisper (audio → text), most translation systems |
| **Encoder-only** | **Embedding and reranking models — the whole of Module 21's retrieval stack** |
| Vision encoder | The tower in a VLM, projected into the decoder's residual stream (Module 07) |

That last row is the connection worth making out loud: **when you build RAG, your retriever is a
bidirectional encoder and your generator is a decoder — you are running both halves of Figure 1,
just in separate processes and trained separately.** Bidirectional attention is genuinely better
for producing one vector that represents a whole passage, because every token can see every other,
and that has never stopped being true.

---

## Where it's used

- Reading the original paper and any diagram derived from it — which is most of them.
- Translation, summarisation, and speech (Whisper) systems still shipping encoder-decoders.
- Understanding what a VLM's vision tower is doing.
- Interviews, where Figure 1 is the most-shown image in the field.
- Knowing which boxes are *absent* from the model you fine-tune, which is the real test of whether
  you understand it.

---

## Labs

### Lab 1 — Annotate the figure against your own code

Take your Module 05 `TinyLM`. For every box in Figure 1, write the line of your code that
implements it, or `ABSENT`.

| Figure box | Your line | Note |
|------------|-----------|------|
| Input Embedding | `self.embed` | |
| Positional Encoding | RoPE inside attention | *Not* an additive input encoding — Module 04 |
| Multi-Head Attention (encoder) | `ABSENT` | No encoder |
| Masked Multi-Head Attention | `Attention(..., is_causal=True)` | |
| **Multi-Head Attention (cross)** | **`ABSENT`** | **The defining difference** |
| Add & Norm | `x = x + attn(norm(x))` | **Pre-norm — order differs from the figure** |
| Feed Forward | `SwiGLU` | Figure shows ReLU MLP |
| Linear → Softmax | `lm_head` | Weight-tied (Module 02) |

**Four rows differ from the picture.** Being able to produce that table from memory is the actual
deliverable of this module.

### Lab 2 — Implement cross-attention and prove the shape

```python
d_model, n_heads, B, T, S = 128, 4, 2, 7, 11     # target len 7, source len 11
xa = CrossAttention(d_model, n_heads)
dec = torch.randn(B, T, d_model)
mem = torch.randn(B, S, d_model)
out = xa(dec, mem)
assert out.shape == (B, T, d_model)              # output follows the TARGET length
```

Then expose the attention weights and confirm the matrix is `B × heads × T × S` — rectangular.
**Print `T`, `S`, and the weight shape.** Everyone says "Q from the decoder, K and V from the
encoder"; far fewer can state that the attention matrix is not square, and the shape is the proof
you actually built it.

### Lab 3 — A working encoder-decoder on a toy task

Build `Encoder` (bidirectional self-attn + FFN), `DecoderLayer` (masked self-attn → cross-attn →
FFN), and train on reversal: input `a b c d`, target `d c b a`.

```python
src = torch.randint(4, V, (B, S))
tgt = torch.randint(4, V, (B, T))
tgt_in, tgt_out = tgt[:, :-1], tgt[:, 1:]        # the "shifted right" of the figure
logits = model(src, tgt_in)
loss = F.cross_entropy(logits.reshape(-1, V), tgt_out.reshape(-1))
```

Then run the ablation that makes the point:

| Run | Expected |
|-----|----------|
| Full model | Solves it |
| **Cross-attention removed** | **Fails completely — the decoder cannot see the input at all** |
| Causal mask added to cross-attention | Degrades badly on later positions |
| Decoder self-attention mask removed | Trains beautifully, generates garbage — it saw the answer |

**Row 2 is the lesson: without cross-attention the two stacks are unconnected and the decoder is an
unconditional language model.** Row 4 is label leakage and it is the same bug as forgetting `-100`
masking in Module 11 — a suspiciously low training loss is the tell in both cases.

### Lab 4 — Post-norm versus pre-norm, at depth

```python
for norm_style in ("pre", "post"):
    for depth in (4, 8, 16, 24):
        for warmup in (0, 500):
            m = TinyLM(n_layers=depth, norm=norm_style)
            print(norm_style, depth, warmup, train(m, steps=2000))   # final loss or "diverged"
```

**Expect post-norm at depth 16–24 with zero warmup to diverge or plateau, and pre-norm to train in
every cell.** That table is the empirical reason the field abandoned the figure's arrangement, and
producing it yourself means you never have to take the claim on faith.

### Lab 5 — Attention maps as word alignment

Train the toy model on a small copy-with-substitution task, then plot the cross-attention weights
as a `T × S` heatmap.

**You should see a near-diagonal for copying, and off-diagonal mass exactly where the substitution
rule reorders tokens.** That picture is what "cross-attention is retrieval over the source" means,
and it is the most direct visual intuition for attention in the entire curriculum.

### Lab 6 — Both halves of Figure 1, running

Load a real encoder-decoder and a real decoder-only model and compare them on the same
summarisation input:

```python
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM
t5   = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
qwen = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-0.6B")
print([n for n, _ in t5.named_modules() if "EncDecAttention" in n][:3])   # cross-attn, named
```

Report parameter counts, and count how many times each model must process the source when
generating 100 tokens. **T5 encodes once; the decoder-only model re-attends over the source at
every step — which is what the KV cache exists to make cheap.**

---

## Failure modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| Decoder ignores the input entirely | Cross-attention missing or memory not wired through | Check `memory` reaches every decoder layer |
| Poor quality on long sources | Causal mask wrongly applied to cross-attention | Cross-attention is never causal |
| Training loss implausibly low, generation garbage | No causal mask on decoder *self*-attention | Mask the decoder's self-attention only |
| Off-by-one; model predicts the token it was given | Forgot the right-shift | `tgt_in = tgt[:, :-1]`, `tgt_out = tgt[:, 1:]` |
| Deep model diverges early | Post-norm without warmup | Pre-norm, or add warmup |
| Padded source tokens attended to | No `memory_mask` | Pass the source padding mask into cross-attention |
| Reading the figure into modern code and getting confused | The figure is post-norm and additive-positional | It is a 2017 document, not a spec |

---

## Interview

**"Walk me through the transformer architecture."**
Two stacks in the original. The encoder embeds the input, adds positional information, and runs N
layers of bidirectional self-attention plus a feed-forward network, each wrapped in a residual add
and a normalisation — that produces one contextual vector per input token, the memory. The decoder
takes the target shifted right, and each of its layers does three things rather than two: masked
self-attention so a position only sees what came before it, then cross-attention where the queries
come from the decoder and the keys and values come from the encoder's memory, then the
feed-forward. Linear and softmax to vocabulary at the top. The part I'd emphasise is that
cross-attention is the only channel between the stacks, and it's the block that doesn't exist in
any decoder-only model — so if someone shows me that figure and asks what a GPT-style model looks
like, the answer is the right column with the middle attention block deleted.

**"What's different between that figure and a model you'd actually work on?"**
Four things. There's no encoder and no cross-attention — one causal stack. Normalisation moved from
after the residual add to inside the branch: pre-norm instead of post-norm, which keeps a clean
additive residual path from embeddings to logits and is why deep models train without the warmup
schedule the original needed. Positional information moved from an additive sinusoidal encoding at
the input to RoPE applied to queries and keys inside every attention layer, which gives relative
positions and extends better. And the feed-forward is SwiGLU with three matrices rather than a ReLU
MLP with two. So the figure is a historical document — it's still the right thing to reason from,
but nobody ships it.

**"Why did decoder-only win?"**
Mostly practical rather than theoretical. One stack means one objective and one training path, and
any text becomes training data — no need for paired source/target. Prefix-as-input works fine:
putting the source in the context and letting causal attention read it turns out to be close enough
to cross-attention for most tasks. And it collapses the serving story to one KV cache and one set
of kernels. The encoder didn't die though, it specialised — it's the retriever in every RAG system,
because bidirectional attention is genuinely better at compressing a passage into one vector, and
it's the vision tower in every VLM. When I build RAG I'm running both halves of that figure, just
as separate models.

**"What breaks if you mask cross-attention causally?"**
The decoder loses access to the parts of the source it hasn't "reached" yet, so quality collapses
on anything involving reordering, and it degrades progressively worse toward the end of the output.
It's a bug I'd expect to see in a hand-rolled implementation because the causal mask is right there
from the self-attention block and it's easy to pass through. The principle is that masking is about
not seeing your own future output — the source was given in full and there's nothing to hide.

---

## Checkpoint

1. Draw Figure 1 from memory and label the three attention blocks by type.
2. State what Q, K and V are for each of the three, and which one is not square.
3. Produce the box-by-box table mapping the figure to your decoder-only code, with the ABSENT rows.
4. Implement cross-attention; show the `T × S` weight shape.
5. Show the four-row ablation, including "cross-attention removed → decoder ignores the input".
6. Produce the post-norm/pre-norm × depth × warmup table.
7. Say where the encoder still lives, and why it is the right tool there.

---

**Next:** [06 — Pretraining](06-pretraining.md) ·
**Back:** [05 — The transformer block, assembled](05-transformer-block.md) · [Syllabus](../SYLLABUS.md)
