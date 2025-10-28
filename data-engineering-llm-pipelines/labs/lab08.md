# Lab 08 — Why Schema Validation in ML/LLM Pipelines

**Focus Area:** Why schema validation — catching upstream drift; protecting training/eval

> This lab is the *why* and *show‑me* for validation. You’ll simulate upstream changes (types, ranges, unexpected categories) and see how a light schema gate prevents bad data from reaching LLM‑adjacent stages.

---

## Outcomes

By the end of this lab, you will be able to:

1. Explain the difference between **structural drift** (columns/types) and **semantic drift** (values/ranges/categories), and why each harms LLM workflows.
2. Add a **pre‑flight validation gate** that fails fast with actionable messages.
3. Use a **minimal Pandera schema** (or Pydantic model per row) to enforce types, ranges, and categorical sets.
4. Capture **human‑readable failure reports** for debugging, CI, and incident triage.

---

## Prerequisites & Setup

- Python 3.13 with `pandas`, `numpy`, `pandera`, `pydantic`, `pyarrow` installed.  
- JupyterLab or VS Code with Jupyter extension.
- Artifacts from previous labs (optional but recommended): `artifacts/clean/per_customer_enriched.parquet` or `users2_clean.parquet`  

**Start a notebook:** `week02_lab08.ipynb`

If you don’t have prior artifacts, synthesize a small frame now:

```python
import numpy as np, pandas as pd
rng = np.random.default_rng(42)
users2 = pd.DataFrame({
    'CustomerID': [f'C{i:05d}' for i in range(300)],
    'country_norm': rng.choice(['USA','DE','SG','BR'], size=300, p=[.55,.2,.15,.1]),
    'age': rng.integers(16, 80, size=300).astype('Int64'),
    'ltv_usd': np.round(np.clip(rng.lognormal(3.0, 0.7, size=300), 0, 5e4), 2),
    'is_adult': (rng.integers(16, 80, size=300) >= 18),
    'is_high_value': rng.random(300) > 0.85,
})
users2.head()
```

---

## Part A — What can go wrong, concretely?

In LLM/ML pipelines, silent data drift can:

- **Break transforms** (e.g., `to_datetime` fails after a type flip from string→int).
- **Bias metrics** (e.g., new country labels split a cohort: `U.S.A.` appears again).
- **Explode tokens/costs** (e.g., unexpectedly long text fields; numeric → string inflation).
- **Poison eval/train** (e.g., negative prices; out‑of‑range ages; missing required keys).

**Exercise:** Create 3 synthetic drifts.

```python
broken = users2.copy()
# 1) Structural drift: age becomes string for some rows
broken.loc[broken.index[:20], 'age'] = broken.loc[broken.index[:20], 'age'].astype(str)
# 2) Semantic drift: country label out of policy
broken.loc[10:15, 'country_norm'] = ['U.S.A.','United States','usa','US','USA','USA']
# 3) Range drift: negative ltv sneaks in
broken.loc[50:55, 'ltv_usd'] = [-10, -5, -1, 0, 1, 2]
broken.head()
```

---

## Part B — Minimal Pandera schema as a gate

We’ll define a small DataFrame schema to catch the above.

```python
import pandera as pa
from pandera import Column, Check

Schema = pa.DataFrameSchema({
    'CustomerID': Column(object, nullable=False),
    'country_norm': Column(object, Check.isin(['USA','DE','SG','BR']), nullable=False),
    'age': Column(pa.Int64, Check.in_range(0, 120), nullable=False),
    'ltv_usd': Column(float, Check.ge(0), nullable=False),
    'is_adult': Column(bool, nullable=False),
    'is_high_value': Column(bool, nullable=False),
})
```

### B1. Validate clean vs broken

```python
# Clean should pass
ok = Schema.validate(users2, lazy=True)
print('clean rows:', len(ok))

# Broken should fail with a report
try:
    Schema.validate(broken, lazy=True)
except pa.errors.SchemaErrors as err:
    report = err.failure_cases
    report.head(10)
```

**Checkpoint:** Inspect `report` to see: wrong dtype (`age`), out‑of‑set categories (`country_norm`), and negative values (`ltv_usd`).

### B2. Actionable messages for CI / logs

```python
# Summarize by column + failure type
summary = (report
           .groupby(['column', 'check'])
           .size()
           .reset_index(name='failures')
           .sort_values('failures', ascending=False))
summary
```

> **Interpretation:** This summary is what you’d attach to a CI artifact or Slack alert.

---

## Part C — Row‑level validation with Pydantic (optional)

Use Pydantic models when you’re validating **per‑row payloads** (e.g., API messages) or writing contracts across services.

```python
from pydantic import BaseModel, Field, ValidationError
from typing import Literal

class CustomerRow(BaseModel):
    CustomerID: str
    country_norm: Literal['USA','DE','SG','BR']
    age: int = Field(ge=0, le=120)
    ltv_usd: float = Field(ge=0)
    is_adult: bool
    is_high_value: bool

row = users2.iloc[0].to_dict()
CustomerRow(**row)

try:
    CustomerRow(**broken.iloc[12].to_dict())
except ValidationError as e:
    print(e)
```

**When to prefer Pydantic:** API boundaries, message queues, microservices. **When to prefer Pandera:** bulk DataFrame validation in ETL/ELT.

---

## Part D — Pre‑flight gate function + fail‑fast

Wrap the schema check in a reusable function that raises a concise, friendly error and writes a CSV report for triage.

```python
from pathlib import Path

def validate_or_raise(df: pd.DataFrame, schema: pa.DataFrameSchema, name: str, out_dir: str = 'artifacts/validation') -> pd.DataFrame:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    try:
        return schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as err:
        rep = err.failure_cases
        dest = Path(out_dir) / f'{name}_schema_failures.csv'
        rep.to_csv(dest, index=False)
        # compact message for logs/CI
        top = (rep.groupby(['column','check']).size().reset_index(name='n')
                 .sort_values('n', ascending=False).head(5).to_dict(orient='records'))
        raise RuntimeError(f"Validation failed for {name}. Top issues: {top}. See {dest}")

# Example usage
_ = validate_or_raise(users2, Schema, name='users2_clean')
try:
    _ = validate_or_raise(broken, Schema, name='users2_broken')
except RuntimeError as e:
    print('\nGATE BLOCKED ->', e)
```

---

## Part E — Wrap‑Up

Add a markdown cell and answer:

1. Name one structural and one semantic drift you simulated. How would each impact an LLM component downstream?  
2. Paste the top 3 failure types from your summary and propose a remediation (fix in source vs transform rule).  
3. Where would you place this validation gate in your Day‑1/Day‑2 pipeline, and why?

---

- **Common pitfalls:** Using `object` dtypes everywhere; not distinguishing `structural` vs `semantic` drift; over‑fitting schemas (too strict for expected evolution).  

---

## Solution Snippets (reference)

**Quick failure roll‑up:**

```python
summary = (report.groupby(['column','check'])
           .size().reset_index(name='failures')
           .sort_values('failures', ascending=False))
summary.head()
```

**CI‑style assert:**

```python
assert Schema.validate(users2, lazy=True) is not None
```

**Lightweight allow‑list for categories:**

```python
allowed = {'USA','DE','SG','BR'}
viol = set(broken['country_norm']) - allowed
viol
```
