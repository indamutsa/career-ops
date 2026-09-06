# Lab 01 — Build the agent environment

**Goal:** a verifiable multi-step SQL agent environment, plus a measured baseline pass rate for a
small model on it.

**Why this first:** the environment silently determines what a trajectory looks like, what a reward
can measure, and whether GRPO produces a gradient at all. Training code written before the
environment gets thrown away.

**Done when:** you can state the base model's pass rate per difficulty tier, and you have found a
tier where it lands between 20% and 60%. That tier is your GRPO training set.

---

## 0. Prerequisites

```bash
docker info --format '{{.ServerVersion}}'          # daemon is up

uv venv .venv && source .venv/bin/activate
uv pip install "psycopg[binary]" torch transformers accelerate
```

**Machine note:** Docker on macOS has no GPU passthrough. Postgres runs in Docker; the model runs
natively on MPS. Never try to train inside the container.

Layout for this lab:

```
agent-posttrain/
  docker/{docker-compose.yml,seed.sql}
  env/{__init__.py,db.py,tools.py,rollout.py,verifier.py,tasks.py,model.py}
  scripts/baseline.py
  data/
```

`env/__init__.py` can be empty, but it must exist — `scripts/baseline.py` imports `env.*` as a package.

---

## 1. The database

`docker/docker-compose.yml`:

```yaml
services:
  db:
    image: postgres:16-alpine
    container_name: agentpt-db
    environment:
      POSTGRES_USER: agent
      POSTGRES_PASSWORD: agent
      POSTGRES_DB: shop
    ports:
      - "55432:5432"
    volumes:
      - ./seed.sql:/docker-entrypoint-initdb.d/10-seed.sql:ro
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U agent -d shop"]
      interval: 2s
      timeout: 3s
      retries: 30
```

Port 55432 avoids collision with any local Postgres.

`docker/seed.sql`:

```sql
-- Near-duplicate column names and one denormalised table, so schema discovery
-- is a real step and not a formality.

CREATE TABLE customers (
    customer_id   SERIAL PRIMARY KEY,
    full_name     TEXT NOT NULL,
    country_code  CHAR(2) NOT NULL,
    signup_date   DATE NOT NULL,
    is_active     BOOLEAN NOT NULL DEFAULT TRUE
);

CREATE TABLE products (
    product_id    SERIAL PRIMARY KEY,
    title         TEXT NOT NULL,
    category      TEXT NOT NULL,
    unit_price    NUMERIC(10,2) NOT NULL,
    discontinued  BOOLEAN NOT NULL DEFAULT FALSE
);

CREATE TABLE orders (
    order_id      SERIAL PRIMARY KEY,
    customer_id   INT NOT NULL REFERENCES customers(customer_id),
    order_date    DATE NOT NULL,
    status        TEXT NOT NULL CHECK (status IN ('placed','shipped','delivered','cancelled','refunded'))
);

CREATE TABLE order_items (
    order_item_id SERIAL PRIMARY KEY,
    order_id      INT NOT NULL REFERENCES orders(order_id),
    product_id    INT NOT NULL REFERENCES products(product_id),
    quantity      INT NOT NULL,
    unit_price    NUMERIC(10,2) NOT NULL   -- price AT TIME OF ORDER, not products.unit_price
);

-- Denormalised: overlaps orders, tempts the model into the wrong join.
CREATE TABLE support_tickets (
    ticket_id     SERIAL PRIMARY KEY,
    order_id      INT REFERENCES orders(order_id),
    customer_name TEXT NOT NULL,           -- duplicated from customers.full_name, can be stale
    opened_at     TIMESTAMP NOT NULL,
    resolved_at   TIMESTAMP,
    severity      TEXT NOT NULL CHECK (severity IN ('low','medium','high'))
);

INSERT INTO customers (full_name, country_code, signup_date, is_active)
SELECT 'Customer ' || i,
       (ARRAY['ES','FR','DE','IT','GB','US'])[1 + (i % 6)],
       DATE '2024-01-01' + (i % 500),
       (i % 9) <> 0
FROM generate_series(1, 400) AS i;

INSERT INTO products (title, category, unit_price, discontinued)
SELECT 'Product ' || i,
       (ARRAY['audio','video','storage','network','power'])[1 + (i % 5)],
       ROUND((5 + (i * 7 % 300))::numeric, 2),
       (i % 17) = 0
FROM generate_series(1, 120) AS i;

INSERT INTO orders (customer_id, order_date, status)
SELECT 1 + (i * 13 % 400),
       DATE '2025-01-01' + (i % 400),
       (ARRAY['placed','shipped','delivered','delivered','cancelled','refunded'])[1 + (i % 6)]
FROM generate_series(1, 3000) AS i;

INSERT INTO order_items (order_id, product_id, quantity, unit_price)
SELECT 1 + (i % 3000),
       1 + (i * 29 % 120),
       1 + (i % 5),
       ROUND((5 + (i * 11 % 280))::numeric, 2)
FROM generate_series(1, 9000) AS i;

INSERT INTO support_tickets (order_id, customer_name, opened_at, severity, resolved_at)
SELECT 1 + (i * 7 % 3000),
       'Customer ' || (1 + (i * 3 % 400)),
       TIMESTAMP '2025-02-01 00:00:00' + (i % 300) * INTERVAL '1 day',
       (ARRAY['low','medium','high'])[1 + (i % 3)],
       CASE WHEN i % 4 = 0 THEN NULL
            ELSE TIMESTAMP '2025-02-01 00:00:00' + (i % 300) * INTERVAL '1 day' + INTERVAL '6 hours' END
FROM generate_series(1, 900) AS i;

-- Read-only role the agent connects as. A hallucinated DELETE fails loudly
-- instead of corrupting the fixture.
CREATE USER agent_ro WITH PASSWORD 'agent_ro';
GRANT CONNECT ON DATABASE shop TO agent_ro;
GRANT USAGE ON SCHEMA public TO agent_ro;
GRANT SELECT ON ALL TABLES IN SCHEMA public TO agent_ro;
```

Bring it up and verify:

```bash
cd docker && docker compose up -d && cd ..
docker exec agentpt-db psql -U agent -d shop -c "\dt"
docker exec agentpt-db psql -U agent -d shop -c "SELECT count(*) FROM order_items;"   # expect 9000
```

> **Design note.** `order_items.unit_price` and `products.unit_price` share a name and mean
> different things — historical price versus current list price. Revenue questions that join to
> `products` are *wrong but plausible*, and they execute without error. This is deliberate: you
> need failures your verifier catches and a naive eyeball does not.

---

## 2. Read-only DB access

`env/db.py`:

```python
"""Read-only Postgres access for the agent environment."""
from __future__ import annotations

import os
from contextlib import contextmanager

import psycopg

DSN = os.environ.get("AGENTPT_DSN", "postgresql://agent_ro:agent_ro@localhost:55432/shop")

# Hard caps so a runaway cross join cannot hang a rollout.
STATEMENT_TIMEOUT_MS = int(os.environ.get("AGENTPT_SQL_TIMEOUT_MS", "4000"))
MAX_ROWS = int(os.environ.get("AGENTPT_MAX_ROWS", "200"))


@contextmanager
def cursor():
    with psycopg.connect(DSN, autocommit=True) as conn:
        with conn.cursor() as cur:
            cur.execute(f"SET statement_timeout = {STATEMENT_TIMEOUT_MS}")
            yield cur


def run_readonly(sql: str) -> tuple[list[str], list[tuple]]:
    """Execute SQL and return (column_names, rows). Raises on any failure."""
    with cursor() as cur:
        cur.execute(sql)
        if cur.description is None:
            raise ValueError("statement returned no result set")
        cols = [d.name for d in cur.description]
        rows = cur.fetchmany(MAX_ROWS)
    return cols, rows
```

The statement timeout is not a nicety. Without it, one bad join in one of 16 sampled trajectories
stalls the whole group — and GRPO needs the whole group before it can compute an advantage.

---

## 3. The tool surface

Seven tools. Three of them (`list_tables`, `describe_table`, `search_columns`) overlap on purpose —
that overlap is what makes tool *selection* a learnable skill rather than a formality.

`env/tools.py`:

```python
"""Tool surface for the SQL analytics agent."""
from __future__ import annotations

import json
import re
from typing import Any

from .db import run_readonly

_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|truncate|grant|revoke|copy)\b", re.IGNORECASE
)
_IDENT = re.compile(r"[a-z_][a-z0-9_]*")


class ToolError(Exception):
    """Malformed or rejected tool call. The message is shown to the policy."""


def list_tables() -> str:
    _, rows = run_readonly(
        "SELECT table_name FROM information_schema.tables "
        "WHERE table_schema = 'public' ORDER BY table_name"
    )
    return "\n".join(r[0] for r in rows)


def describe_table(table: str) -> str:
    if not _IDENT.fullmatch(table or ""):
        raise ToolError(f"invalid table name: {table!r}")
    _, rows = run_readonly(
        "SELECT column_name, data_type, is_nullable FROM information_schema.columns "
        f"WHERE table_schema = 'public' AND table_name = '{table}' ORDER BY ordinal_position"
    )
    if not rows:
        raise ToolError(f"no such table: {table}")
    return "\n".join(f"{c} {t} {'NULL' if n == 'YES' else 'NOT NULL'}" for c, t, n in rows)


def search_columns(keyword: str) -> str:
    kw = re.sub(r"[^a-z0-9_]", "", (keyword or "").lower())
    if not kw:
        raise ToolError("keyword must be alphanumeric")
    _, rows = run_readonly(
        "SELECT table_name, column_name FROM information_schema.columns "
        f"WHERE table_schema = 'public' AND column_name LIKE '%{kw}%' "
        "ORDER BY table_name, column_name"
    )
    return "\n".join(f"{t}.{c}" for t, c in rows) if rows else "(no matching columns)"


def sample_rows(table: str, n: int = 5) -> str:
    if not _IDENT.fullmatch(table or ""):
        raise ToolError(f"invalid table name: {table!r}")
    n = max(1, min(int(n), 20))
    cols, rows = run_readonly(f"SELECT * FROM {table} LIMIT {n}")
    return _render(cols, rows)


def run_sql(query: str) -> str:
    q = (query or "").strip().rstrip(";")
    if not q:
        raise ToolError("empty query")
    if ";" in q:
        raise ToolError("only a single statement is allowed")
    if _FORBIDDEN.search(q):
        raise ToolError("only read-only SELECT queries are allowed")
    if not q.lower().lstrip("( ").startswith(("select", "with")):
        raise ToolError("query must start with SELECT or WITH")
    cols, rows = run_readonly(q)
    return _render(cols, rows)


def _render(cols: list[str], rows: list[tuple]) -> str:
    if not rows:
        return "(0 rows)"
    head = " | ".join(cols)
    body = "\n".join(" | ".join("NULL" if v is None else str(v) for v in r) for r in rows)
    return f"{head}\n{body}\n({len(rows)} rows)"


# --- terminal actions -------------------------------------------------------

def submit_answer(sql: str) -> str:
    """Terminal. The submitted SQL is what the verifier executes."""
    return sql


def give_up(reason: str) -> str:
    """Terminal. Correct give-up on an impossible task is rewarded, not punished."""
    return reason


TOOLS: dict[str, Any] = {
    "list_tables": list_tables,
    "describe_table": describe_table,
    "search_columns": search_columns,
    "sample_rows": sample_rows,
    "run_sql": run_sql,
    "submit_answer": submit_answer,
    "give_up": give_up,
}

TERMINAL = {"submit_answer", "give_up"}

SCHEMAS = [
    {"name": "list_tables", "description": "List every table in the database.",
     "parameters": {"type": "object", "properties": {}, "required": []}},
    {"name": "describe_table", "description": "Column names, types and nullability for one table.",
     "parameters": {"type": "object", "properties": {"table": {"type": "string"}}, "required": ["table"]}},
    {"name": "search_columns", "description": "Find columns anywhere in the schema whose name contains a keyword.",
     "parameters": {"type": "object", "properties": {"keyword": {"type": "string"}}, "required": ["keyword"]}},
    {"name": "sample_rows", "description": "Return up to 20 example rows from one table.",
     "parameters": {"type": "object", "properties": {"table": {"type": "string"}, "n": {"type": "integer"}}, "required": ["table"]}},
    {"name": "run_sql", "description": "Execute one read-only SELECT and see its result. Use this to check your work before submitting.",
     "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "submit_answer", "description": "Submit the final SQL that answers the question. Ends the episode.",
     "parameters": {"type": "object", "properties": {"sql": {"type": "string"}}, "required": ["sql"]}},
    {"name": "give_up", "description": "Declare the task unanswerable with the available schema. Ends the episode.",
     "parameters": {"type": "object", "properties": {"reason": {"type": "string"}}, "required": ["reason"]}},
]
```

Smoke test before going further:

```bash
python -c "
from env.tools import list_tables, describe_table, run_sql
print(list_tables())
print(describe_table('order_items'))
print(run_sql('SELECT status, count(*) FROM orders GROUP BY status ORDER BY status'))
"
```

---

## 4. The rollout loop

This is the state machine. Four decisions live here, and each one bites later — read the notes.

`env/rollout.py`:

```python
"""Multi-step rollout loop: prompt -> tool calls -> terminal action."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable

from .tools import SCHEMAS, TERMINAL, TOOLS, ToolError

MAX_STEPS = 8

SYSTEM = """You are a SQL analyst agent working against a PostgreSQL database.

You act by emitting exactly one JSON object per turn, and nothing else:
{"tool": "<name>", "args": {...}}

Available tools:
""" + json.dumps(SCHEMAS, indent=2) + """

Rules:
- Explore the schema before writing SQL. Column names are not always what you expect.
- Verify your query with run_sql before submit_answer.
- If the question cannot be answered from this schema, call give_up with a reason.
- Emit only the JSON object. No prose, no markdown fences.
"""

_JSON = re.compile(r"\{.*\}", re.DOTALL)


@dataclass
class Step:
    raw: str
    tool: str | None
    args: dict
    observation: str
    ok: bool


@dataclass
class Trajectory:
    task_id: str
    question: str
    steps: list[Step] = field(default_factory=list)
    terminal: str | None = None       # "submit_answer" | "give_up" | None (step limit)
    final_sql: str | None = None
    give_up_reason: str | None = None

    @property
    def n_steps(self) -> int:
        return len(self.steps)


def parse_action(raw: str) -> tuple[str | None, dict, str]:
    """Extract a {"tool","args"} object. Returns (tool, args, error_message)."""
    m = _JSON.search(raw or "")
    if not m:
        return None, {}, "could not find a JSON object in your reply"
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        return None, {}, f"invalid JSON: {e}"
    tool = obj.get("tool")
    if tool not in TOOLS:
        return None, {}, f"unknown tool: {tool!r}"
    args = obj.get("args") or {}
    if not isinstance(args, dict):
        return None, {}, "args must be an object"
    return tool, args, ""


def rollout(task_id: str, question: str, generate: Callable[[list[dict]], str]) -> Trajectory:
    """`generate` takes a message list and returns the model's next reply."""
    traj = Trajectory(task_id=task_id, question=question)
    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": question},
    ]

    for _ in range(MAX_STEPS):
        raw = generate(messages)
        tool, args, err = parse_action(raw)

        if tool is None:
            obs, ok = f"ERROR: {err}", False
        elif tool in TERMINAL:
            traj.steps.append(Step(raw, tool, args, "(terminal)", True))
            traj.terminal = tool
            if tool == "submit_answer":
                traj.final_sql = args.get("sql")
            else:
                traj.give_up_reason = args.get("reason")
            return traj
        else:
            try:
                obs, ok = TOOLS[tool](**args), True
            except ToolError as e:
                obs, ok = f"ERROR: {e}", False
            except TypeError as e:
                obs, ok = f"ERROR: bad arguments: {e}", False
            except Exception as e:
                obs, ok = f"ERROR: {type(e).__name__}: {e}", False

        traj.steps.append(Step(raw, tool, args, obs, ok))
        messages.append({"role": "assistant", "content": raw})
        messages.append({"role": "user", "content": f"Observation:\n{obs}"})

    return traj  # step limit hit, terminal stays None
```

Four decisions, and why each is what it is:

1. **Strict JSON protocol instead of the model's native tool-calling template.** Native templates
   differ per model family and you will swap models. Owning the action space also means you own the
   parse-failure rate — itself a metric worth watching, because a policy that emits unparseable
   actions 30% of the time looks like a reasoning failure when it is a formatting one.
2. **Full history replay, not summarisation.** Summarising introduces a second model whose failures
   contaminate your reward signal. Full replay caps how long a task can run — accept that ceiling
   for v1, and know that you accepted it.
3. **Errors go back to the policy as observations.** A rejected tool call is a *learning signal*,
   not a crash. Recovery-after-error is exactly the behaviour you want GRPO to reinforce.
4. **`terminal is None` means the step limit was hit.** Keep that distinct from `give_up`. One is a
   model that never decided; the other is a model that decided it could not. They deserve different
   rewards, and conflating them is the most common environment bug.

---

## 5. The verifier

Execution-based, not judged. Compare result *sets*.

`env/verifier.py`:

```python
"""Execution-based verification: run gold and candidate, compare result sets."""
from __future__ import annotations

from collections import Counter
from decimal import Decimal

from .db import run_readonly


def _norm(v):
    if isinstance(v, Decimal):
        return round(float(v), 4)
    if isinstance(v, float):
        return round(v, 4)
    return v


def _fingerprint(rows, ordered: bool):
    body = [tuple(_norm(v) for v in r) for r in rows]
    return ("ordered", body) if ordered else ("unordered", Counter(body))


def verify(candidate_sql: str | None, gold_sql: str, ordered: bool = False) -> tuple[bool, str]:
    """Returns (passed, reason). Never raises."""
    if not candidate_sql:
        return False, "no sql submitted"
    try:
        g_cols, g_rows = run_readonly(gold_sql)
    except Exception as e:
        return False, f"GOLD QUERY BROKEN: {e}"      # your bug, not the model's
    try:
        c_cols, c_rows = run_readonly(candidate_sql)
    except Exception as e:
        return False, f"candidate failed to execute: {type(e).__name__}: {e}"

    if len(c_cols) != len(g_cols):
        return False, f"column count {len(c_cols)} != expected {len(g_cols)}"
    if _fingerprint(c_rows, ordered) != _fingerprint(g_rows, ordered):
        return False, f"result mismatch: got {len(c_rows)} rows, expected {len(g_rows)}"
    return True, "ok"
```

Two things to notice:

- **Column *names* are not compared, only the count and the values.** `AS revenue` versus
  `AS total_revenue` is not a failure. Being strict here punishes cosmetics and teaches the policy
  to guess your naming, which is not the skill you want.
- **`GOLD QUERY BROKEN` is a distinct return.** When a whole tier scores 0%, this is the first thing
  to check. A broken gold query looks exactly like a hard task.

---

## 6. The task set

Three tiers. Tier 3 includes genuinely impossible tasks — that is what makes `give_up` trainable.

`env/tasks.py`:

```python
"""Task definitions. Tier 1 single-table, tier 2 joins, tier 3 hard + impossible."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class Task:
    id: str
    tier: int
    question: str
    gold_sql: str | None       # None => impossible, correct answer is give_up
    ordered: bool = False


TASKS: list[Task] = [
    # --- tier 1: single table, direct ---------------------------------------
    Task("t1-01", 1, "How many customers are there in total?",
         "SELECT count(*) FROM customers"),
    Task("t1-02", 1, "How many orders have status 'cancelled'?",
         "SELECT count(*) FROM orders WHERE status = 'cancelled'"),
    Task("t1-03", 1, "List the distinct product categories, alphabetically.",
         "SELECT DISTINCT category FROM products ORDER BY category", ordered=True),
    Task("t1-04", 1, "How many customers are inactive?",
         "SELECT count(*) FROM customers WHERE is_active = FALSE"),

    # --- tier 2: joins and aggregation --------------------------------------
    Task("t2-01", 2, "What is the total revenue from delivered orders? "
                     "Use the price recorded on the order line.",
         "SELECT sum(oi.quantity * oi.unit_price) FROM order_items oi "
         "JOIN orders o ON o.order_id = oi.order_id WHERE o.status = 'delivered'"),
    Task("t2-02", 2, "Which product category has the highest total quantity sold?",
         "SELECT p.category FROM order_items oi JOIN products p ON p.product_id = oi.product_id "
         "GROUP BY p.category ORDER BY sum(oi.quantity) DESC LIMIT 1"),
    Task("t2-03", 2, "How many orders were placed by customers in Spain?",
         "SELECT count(*) FROM orders o JOIN customers c ON c.customer_id = o.customer_id "
         "WHERE c.country_code = 'ES'"),
    Task("t2-04", 2, "How many support tickets are still unresolved?",
         "SELECT count(*) FROM support_tickets WHERE resolved_at IS NULL"),

    # --- tier 3: multi-hop, and impossible ----------------------------------
    Task("t3-01", 3, "For each country, the average order value of delivered orders, "
                     "highest first. Order value uses the price on the order line.",
         "SELECT c.country_code, avg(t.order_value) FROM ("
         "  SELECT o.order_id, o.customer_id, sum(oi.quantity * oi.unit_price) AS order_value"
         "  FROM orders o JOIN order_items oi ON oi.order_id = o.order_id"
         "  WHERE o.status = 'delivered' GROUP BY o.order_id, o.customer_id) t "
         "JOIN customers c ON c.customer_id = t.customer_id "
         "GROUP BY c.country_code ORDER BY avg(t.order_value) DESC", ordered=True),
    Task("t3-02", 3, "How many customers placed an order but never opened a support ticket?",
         "SELECT count(DISTINCT o.customer_id) FROM orders o WHERE o.customer_id NOT IN ("
         "  SELECT DISTINCT o2.customer_id FROM orders o2 "
         "  JOIN support_tickets s ON s.order_id = o2.order_id)"),

    # impossible: no such data exists in the schema
    Task("t3-90", 3, "What is the average customer satisfaction rating per product?", None),
    Task("t3-91", 3, "Which marketing channel drove the most signups?", None),
]

BY_TIER = {t: [x for x in TASKS if x.tier == t] for t in (1, 2, 3)}
```

> Twelve tasks is enough to measure a baseline. Expand to ~200 *after* the pass rates tell you which
> tier is worth investing in. Writing 200 gold queries for a tier you end up discarding is the
> classic week-one waste.

---

## 7. The model

`env/model.py`:

```python
"""Small local model on MPS. Prove the pipeline here, scale the model last."""
from __future__ import annotations

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "Qwen/Qwen3-1.7B"


def load(model_id: str = MODEL_ID):
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.bfloat16, device_map="mps"
    )
    model.eval()
    return tok, model


def make_generator(tok, model, temperature: float = 1.0, max_new_tokens: int = 256):
    """Returns generate(messages) -> str. Temperature 1.0 on purpose — see note."""
    def generate(messages: list[dict]) -> str:
        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        inputs = tok(text, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temperature,
                top_p=0.95,
                pad_token_id=tok.eos_token_id,
            )
        return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return generate
```

**Temperature 1.0, not 0.** Greedy decoding gives you 16 identical trajectories, every advantage is
zero, and GRPO produces no gradient. Sampling diversity *is* the training signal. Measure your
baseline under the same sampling regime you will train under, or the number means nothing.

**`enable_thinking=False`** on Qwen3 — reasoning traces blow the context budget over 8 steps, and
you are not training on them yet.

---

## 8. The baseline

`scripts/baseline.py`:

```python
"""Sample G trajectories per task, report pass rate per tier."""
from __future__ import annotations

import collections
import json
import sys

from env.model import load, make_generator
from env.rollout import rollout
from env.tasks import TASKS
from env.verifier import verify

G = int(sys.argv[1]) if len(sys.argv) > 1 else 8


def score(task, traj) -> bool:
    if task.gold_sql is None:                       # impossible task
        return traj.terminal == "give_up"
    if traj.terminal != "submit_answer":
        return False
    ok, _ = verify(traj.final_sql, task.gold_sql, ordered=task.ordered)
    return ok


def main():
    tok, model = load()
    generate = make_generator(tok, model)

    per_tier = collections.defaultdict(lambda: [0, 0])
    parse_fail = steps_total = episodes = 0
    records = []

    for task in TASKS:
        wins = 0
        for k in range(G):
            traj = rollout(task.id, task.question, generate)
            ok = score(task, traj)
            wins += ok
            episodes += 1
            steps_total += traj.n_steps
            parse_fail += sum(1 for s in traj.steps if s.tool is None)
            records.append({
                "task": task.id, "tier": task.tier, "sample": k, "passed": ok,
                "terminal": traj.terminal, "steps": traj.n_steps, "sql": traj.final_sql,
            })
        per_tier[task.tier][0] += wins
        per_tier[task.tier][1] += G
        print(f"{task.id}  {wins}/{G}")

    print("\n--- pass rate by tier ---")
    for tier in sorted(per_tier):
        w, n = per_tier[tier]
        flag = "  <-- TRAIN HERE" if 0.20 <= w / n <= 0.60 else ""
        print(f"tier {tier}: {w}/{n} = {w/n:.0%}{flag}")

    print(f"\nparse failure rate: {parse_fail}/{steps_total} steps")
    print(f"mean steps/episode: {steps_total/episodes:.1f}")

    with open("data/baseline.jsonl", "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    main()
```

Run it:

```bash
mkdir -p data
python -m scripts.baseline 8
```

---

## Checkpoint — what to bring back

Four numbers:

1. **Pass rate per tier.** You want one tier in the 20–60% band. If tier 1 is 90% and tier 2 is 5%,
   your difficulty gap is too wide — add an intermediate tier rather than training on either.
2. **Parse failure rate.** Above ~15% and your problem is the action protocol, not the reasoning.
   Fix that first; it is cheap, and every point of parse failure is a wasted rollout.
3. **Mean steps per episode.** If it is pinned at 8 (the limit), the model never terminates — check
   whether it understands `submit_answer` at all.
4. **Give-up behaviour on `t3-90` / `t3-91`.** Most small models never give up and grind to the step
   limit. That gap is one of the more interesting things you will be able to show *fixed* by the end
   of Lab 03.

### If everything is 0%

In this order: a broken gold query (`verify` returns `GOLD QUERY BROKEN`), then parse failures, then
the step limit. It is almost never the model.

### If you want to go faster

`Qwen/Qwen3-0.6B` runs quicker and will score worse — fine for debugging the plumbing. Never debug
plumbing on a big model; you spend the time waiting instead of thinking.

---

**Next:** Lab 02 — trajectory collection and dataset construction. The 20–60% band you just found is
where the training data comes from, and I will show you why those trajectories beat any dataset you
could download.
