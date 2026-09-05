# Module 10 — Structured output and tool calling

An agent is a model whose output is parsed by a program. The moment that is true, "usually valid
JSON" stops being acceptable — a 2% parse-failure rate on an 8-step trajectory means roughly 15% of
episodes die on syntax rather than on reasoning. This module is how you take that number to zero
and why doing so is a *decoding* problem, not a prompting one.

---

## Terms

| Term | Meaning |
|------|---------|
| **Structured output** | Generation constrained to a machine-parseable format. |
| **Constrained decoding** | Masking invalid tokens' logits to `-inf` before sampling. |
| **Grammar-guided generation** | Constraining to a formal grammar (CFG, EBNF, GBNF). |
| **JSON Schema** | Declarative description of a valid JSON document. |
| **FSM / automaton** | The state machine compiled from a schema; drives the mask. |
| **Outlines** | Library compiling regex/schema into an FSM over the vocabulary. |
| **XGrammar** | Fast grammar backend used by vLLM/SGLang. |
| **llguidance** | Constraint engine used by several serving stacks. |
| **Token mask** | Per-step boolean over the vocabulary: which tokens are legal now. |
| **Tool / function calling** | Model emits a structured call; the harness executes it. |
| **Tool schema** | Name, description, and parameter schema given to the model. |
| **Parallel tool calls** | Multiple calls emitted in one turn. |
| **Tool role** | The message role carrying a tool's result back. |
| **ReAct** | Interleaved Reason → Act → Observe loop. |
| **Function-calling fine-tune** | Post-training specifically for tool-call format compliance. |
| **Schema drift** | Serving-time schema differs from the trained one. |
| **Parse-failure rate** | Fraction of generations a parser rejects. A first-class metric. |

---

## Concepts

### Why prompting alone cannot get you to zero

Sampling picks a token from a distribution. If the distribution places any mass on `"` when the
grammar requires `}`, that token can be sampled. A better prompt lowers the probability; it never
makes it zero.

So there are exactly three levers:

| Lever | Effect | Guarantee |
|-------|--------|-----------|
| Better prompt / few-shot | Reduces the rate | None |
| Fine-tune on correct format | Reduces the rate a lot | None |
| **Constrained decoding** | **Invalid tokens are unreachable** | **Total** |

**Use all three, and understand what each buys.** Fine-tuning is what makes the constrained output
*good* rather than merely valid; constraining is what makes it *always* parse. Neither substitutes
for the other — a model that can only produce valid JSON can still produce valid JSON containing
the wrong answer.

### How constrained decoding works

1. Compile the schema or regex into a finite automaton over the *token* vocabulary (not
   characters — the compilation must account for the tokenizer's merges).
2. Track the automaton's state as generation proceeds.
3. At each step, build a boolean mask of tokens that keep the automaton in a valid state.
4. Set every masked-out logit to `-inf`, then sample normally.

Cost: the compilation is the expensive part and is **cached per schema**, so a fixed schema is
nearly free at steady state. Per-token mask application is cheap. This is why serving stacks want
your schema up front rather than per request.

### The quality caveat nobody mentions

Over-constraining can hurt content quality. If the schema forces the model to emit `"answer"`
before it has "thought", you have removed its ability to reason before committing.

The fix is to put the reasoning **inside** the schema:

```json
{
  "reasoning": "string",
  "tool": "run_sql",
  "args": {"q": "string"}
}
```

Field order in the schema is generation order. `reasoning` first means the model produces its
chain of thought and then conditions the tool call on it. Put `tool` first and you get a
post-hoc rationalisation of a decision already made. **This is a real, measurable effect and a good
thing to raise unprompted** — it shows you have actually used constrained decoding rather than read
about it.

### Tool calling, mechanically

Three parts:

1. **Schema in context.** Usually rendered into the system prompt by the chat template's `tools`
   argument. It is *tokens*, so it costs prompt budget on every call.
2. **Model emits a call**, either in the model's native tool-call format or in JSON you defined.
3. **Harness executes and returns the result** as a `tool`-role message.

```python
tok.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True)
```

Whether to use the model's native format or your own JSON protocol:

| | Native tool format | Your own JSON protocol |
|---|---|---|
| Compliance out of the box | Better — it was trained on it | Worse until you fine-tune |
| Portability across models | Poor | Good |
| Control over the schema | Limited | Total |
| Fine-tuning target | Awkward | Clean |

For a post-training project, **your own strict JSON protocol is usually the right call** — you are
going to fine-tune the format anyway, and you want it identical across base models so ablations are
comparable. That is exactly the choice Lab 01 makes.

### Tool design is agent design

The tools you expose determine the difficulty of the task far more than the model does.

- **Too few / too coarse** → the model must do multi-step work inside one call, and failures are
  unattributable.
- **Too many / overlapping** → selection becomes the hard problem, and you learn about tool choice
  rather than about the task.
- **Deliberate overlap** (Lab 01 has `list_tables`, `describe_table`, `search_columns`,
  `sample_rows`) → creates a *discovery strategy* to learn. That is the point when the research
  question is agent post-training.

Every tool needs: a name the model can guess, a one-line description, a typed parameter schema,
**errors returned as observations rather than raised**, and a bounded output size. That last one
matters — an unbounded `run_sql` result can blow the context window in a single step and end the
episode.

### Errors are observations

If a tool raises, the episode ends and you learn nothing about recovery. If it returns

```json
{"error": "column 'ordr_date' does not exist. Did you mean 'order_date'?"}
```

then error recovery becomes a **learnable behaviour**, and your trajectories contain examples of
it. Post-training an agent that has never seen an error recover from one is not going to work.

### Parse-failure rate is a first-class metric

Track it separately from task accuracy, always. They have different fixes:

| Metric | Cause | Fix |
|--------|-------|-----|
| Parse failure | Format | Constrained decoding + format SFT |
| Wrong tool chosen | Selection | Better descriptions, or RL |
| Right tool, wrong args | Reasoning | Data, or RL |
| Right call, wrong answer | Task ability | Data, or RL |

Reporting one number that mixes them tells you nothing about what to do next. This four-way split
is Module 24's whole thesis, arriving early.

---

## Where it's used

- **Every agent.** Unparseable output is a dead episode.
- **RL rollouts.** A parse failure has to be a distinct reward case (Module 15) — it is a different
  failure from a wrong answer and rewarding them identically teaches the wrong lesson.
- **Data extraction pipelines**, where the output feeds a database.
- **Evaluation harnesses** that need to compare structured predictions.
- **Enterprise integration** — the JD's "agentic systems at enterprise scale" is mostly this.

---

## Labs

### Lab 1 — Measure your unconstrained parse-failure rate

```python
import json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-1.7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B",
                                             torch_dtype=torch.bfloat16).to("mps")

SYS = ('Respond with ONLY a JSON object: '
       '{"reasoning": "...", "tool": "run_sql", "args": {"q": "..."}}')

qs = ["How many orders in March?", "Top 5 customers by revenue?",
      "Average order value?", "Which products never sold?",
      "Revenue by month for 2025?"]

fails, total = 0, 0
for q in qs:
    prompt = tok.apply_chat_template(
        [{"role": "system", "content": SYS}, {"role": "user", "content": q}],
        tokenize=False, add_generation_prompt=True)
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).to("mps")
    outs = model.generate(**ids, max_new_tokens=160, do_sample=True, temperature=1.0,
                          num_return_sequences=8, pad_token_id=tok.eos_token_id)
    for o in outs:
        total += 1
        text = tok.decode(o[ids["input_ids"].shape[1]:], skip_special_tokens=True)
        try:
            obj = json.loads(text.strip())
            assert "tool" in obj and "args" in obj
        except Exception:
            fails += 1

print(f"parse-failure rate: {fails}/{total} = {fails/total:.1%}")
```

**Record this number at `T=1.0`** — the RL rollout temperature, where it is worst and where it
matters. Then compute the episode-level cost: with per-step failure `f` over `n` steps, the
probability an episode survives is `(1-f)^n`. At `f=0.02, n=8` that is 85%.

### Lab 2 — Constrain it to zero

```python
from outlines import models, generate
import outlines

class Action(outlines.Model):  # or use a pydantic BaseModel
    pass

from pydantic import BaseModel
from typing import Literal

class Args(BaseModel):
    q: str

class Action(BaseModel):
    reasoning: str
    tool: Literal["list_tables", "describe_table", "search_columns",
                  "sample_rows", "run_sql", "submit_answer", "give_up"]
    args: dict

m = models.transformers("Qwen/Qwen3-1.7B")
gen = generate.json(m, Action)

for q in qs:
    for _ in range(8):
        result = gen(f"{SYS}\n\nUser: {q}\nAssistant:")
        assert isinstance(result, Action)      # cannot fail by construction
print("parse-failure rate: 0/40 = 0.0%")
```

**Then measure the cost:** time both loops and report tokens/second with and without constraints.
Expect single-digit percent overhead once the schema is compiled. Report the *first* call
separately — that one pays compilation.

### Lab 3 — Field order changes answer quality

The most interesting lab in the module.

```python
class ReasonFirst(BaseModel):
    reasoning: str
    tool: str
    args: dict

class ToolFirst(BaseModel):
    tool: str
    args: dict
    reasoning: str
```

Run both over your Lab 01 task set, 8 samples each, and compare **task accuracy** (not parse rate —
both are 100%).

**Expected:** reasoning-first wins, sometimes substantially. Because generation is left-to-right,
`reasoning` first means the tool call is conditioned on the reasoning; `reasoning` last means it is
a justification produced after the decision. Write down the two accuracy numbers. This is a genuine
experimental result you can describe in an interview.

### Lab 4 — Tool-schema token cost

```python
tools = json.load(open("env/tool_schemas.json"))

rendered = tok.apply_chat_template(
    [{"role": "user", "content": "hi"}], tools=tools,
    tokenize=False, add_generation_prompt=True)
n_with = len(tok(rendered, add_special_tokens=False)["input_ids"])

plain = tok.apply_chat_template(
    [{"role": "user", "content": "hi"}], tokenize=False, add_generation_prompt=True)
n_without = len(tok(plain, add_special_tokens=False)["input_ids"])

print(f"tool schemas cost {n_with - n_without} tokens per call")
for n in (1_000, 100_000, 1_000_000):
    print(f"  at {n:>9,} calls/day: {(n_with-n_without)*n/1e6:.1f}M tokens/day")
```

Then trim the descriptions and re-measure. Tool descriptions are prompt tokens on **every single
call, every step of every episode** — an 8-step episode pays the tool schema 8 times. Verbose
descriptions are a recurring bill.

### Lab 5 — Errors as observations

Take Lab 01's `tools.py` and run two variants over the same tasks:

| Variant | On a bad query |
|---------|----------------|
| A | raises → episode terminates |
| B | returns `{"error": "...", "hint": "..."}` → episode continues |

Measure: task success rate, mean steps, and **recovery rate** (fraction of episodes that hit an
error and still succeeded).

Variant B's recovery rate is the number that matters. It is also the behaviour you cannot train
without, since variant A generates zero examples of recovery.

### Lab 6 — Tool-selection confusion matrix

```python
from collections import Counter

confusion = Counter()
for r in rollouts:                      # from data/rollouts.jsonl
    for step in r["steps"]:
        confusion[(r["task_tier"], step["tool"])] += 1

tools_seen = sorted({t for _, t in confusion})
print(f"{'tier':<8}" + "".join(f"{t[:11]:>13}" for t in tools_seen))
for tier in ["easy", "medium", "hard"]:
    row = "".join(f"{confusion[(tier, t)]:>13}" for t in tools_seen)
    print(f"{tier:<8}{row}")
```

**Look for:** does the model ever use `search_columns`? Does it call `describe_table` before
`run_sql`, or guess? Unused tools are either badly described or genuinely redundant — either way
that is a finding. Compare this matrix before and after post-training; the change in strategy is
one of the strongest artefacts you can bring to an interview.

---

## Failure modes

| Symptom | Cause | Fix |
|---------|-------|-----|
| Occasional invalid JSON | Sampling can pick invalid tokens | Constrained decoding |
| Valid JSON, wrong schema | Constrained to "JSON", not to *your* schema | Constrain to the schema |
| Valid output, poor answers | Reasoning field last, or absent | Reasoning first in field order |
| Model invents a tool name | `tool` typed as free string | `Literal[...]` enum |
| Args wrong type | Loose `dict` args | Per-tool typed arg models |
| Episode dies on first error | Tool raised instead of returning | Errors as observations |
| Context blown mid-episode | Unbounded tool output | Cap rows and characters |
| Works locally, fails in prod | Schema drift between training and serving | Version the schema as an artefact |
| First call slow, rest fast | Grammar compilation | Expected; cache and warm up |
| Model never uses a tool | Poor description, or genuinely redundant | Rewrite or remove it |
| Parse failures rewarded like wrong answers | Reward function conflates them | Distinct reward case (Module 15) |

---

## Interview

**"Your agent produces invalid JSON 2% of the time. How do you get to zero?"**
Constrained decoding — compile the schema to an automaton over the vocabulary and mask every token
that would leave a valid state to `-inf` before sampling. Invalid output becomes unreachable rather
than unlikely, which is the only thing that gets you to actually zero; prompting and fine-tuning
both just lower the rate. And 2% per step is worse than it sounds: over an 8-step trajectory that's
about 15% of episodes dying on syntax. I'd still fine-tune the format alongside it, because
constraining makes output *valid*, not *good*.

**"Any downside to constrained decoding?"**
Two. There's a small throughput cost, mostly one-off schema compilation which caches. The real one
is that over-constraining hurts quality — if the schema forces the answer field before any
reasoning field, you've removed the model's ability to think before committing, and you get
post-hoc rationalisation. Field order is generation order. I put a `reasoning` field first in the
schema, and it's measurable: same 100% parse rate either way, different task accuracy.

**"How do you design the tool set for an agent?"**
Tools define the difficulty more than the model does. Too coarse and the model does multi-step work
inside one call, so failures are unattributable. Too many overlapping ones and you're studying tool
selection instead of the task. For a research setup I deliberately include overlapping discovery
tools — list tables, describe, search columns, sample rows — because that creates a discovery
strategy that's actually worth learning. Every tool returns errors as observations rather than
raising, so recovery is learnable, and every tool has a bounded output size so one call can't blow
the context window.

**"You report 60% task success. What else do I need to know?"**
The breakdown, because the 40% has four different causes with four different fixes: parse failures,
wrong tool selected, right tool with wrong arguments, and correct execution with a wrong answer.
Parse failures are a decoding fix. Tool selection is usually a description problem. Wrong args and
wrong answers are data or RL. A single aggregate number tells you nothing about which lever to
pull, so I track them separately from the start.

**"Native tool-calling format or your own JSON protocol?"**
For production on one model, native — it was post-trained on that format so compliance is better
for free. For a post-training research project, my own strict protocol, because I'm going to
fine-tune the format regardless and I want it identical across base models so ablations are
actually comparable. Native formats differ per family, which makes cross-model comparison a
confound.

---

## Checkpoint

1. Report your unconstrained parse-failure rate at `T=1.0` and the episode-level survival maths.
2. Demonstrate zero failures under constraint, with the throughput cost measured.
3. Report the reasoning-first vs tool-first accuracy difference.
4. State the per-call token cost of your tool schemas and the daily total.
5. Report recovery rate with errors-as-observations versus raising.
6. Produce the tool-selection confusion matrix and name one finding from it.

---

**Next:** [11 — Supervised fine-tuning](11-sft.md) ·
**Back:** [09 — Chat templates and prompting](09-prompting.md) · [Syllabus](../SYLLABUS.md)
