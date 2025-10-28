# Lab 09 — Pandera & Pydantic Walkthrough

**Focus Area:** Column types, constraints (`Check.in_range`, regex), row/DF checks, error handling, CI hooks

> This lab is the *how*: you’ll author practical schemas with **Pandera** (DataFrame‑level) and **Pydantic** (row/message‑level), wire clear error messages, and add lightweight CI hooks so bad data never reaches downstream LLM steps.

---

## Outcomes

By the end of this lab, you will be able to:

1. Define **column types** and **constraints** in Pandera (range checks, regex/category allow‑lists, cross‑column rules).
2. Write **row‑level contracts** with Pydantic and use validators for custom logic.
3. Handle validation errors gracefully: produce compact roll‑ups for logs/CI and detailed CSVs for triage.
4. Add **CI hooks** (pytest + pre‑commit) that fail fast on drift.

---

## Prerequisites & Setup

- Python 3.13 with `pandas`, `numpy`, `pandera>=0.20`, `pydantic>=2.0`, `pyarrow`, `pytest` (for the CI section).  
- JupyterLab or VS Code with Jupyter extension.
- Artifacts: Prefer `artifacts/clean/per_customer_enriched.parquet` and `users2_clean.parquet`. If not present, use the synthetic block below.

**Start a notebook:** `week02_lab09.ipynb`

Synthetic fallback (run only if needed):

```python
import numpy as np, pandas as pd
rng = np.random.default_rng(1)
users2 = pd.DataFrame({
    'CustomerID': [f'C{i:05d}' for i in range(500)],
    'country_norm': rng.choice(['USA','DE','SG','BR'], 500, p=[.55,.2,.15,.1]),
    'age': rng.integers(16, 80, 500).astype('Int64'),
    'ltv_usd': np.round(np.clip(rng.lognormal(3.0, 0.7, 500), 0, 5e5), 2),
    'email': [f'user{i}@example.com' for i in range(500)],
    'is_adult': True,
    'is_high_value': rng.random(500) > 0.9,
})
users2.head(3)
```

---

## Part A — Pandera DataFrame Schema

### A1. Define column types & simple constraints

```python
import pandera as pa
from pandera import Column, Check

AllowedCountries = ['USA','DE','SG','BR']

UsersSchema = pa.DataFrameSchema({
    'CustomerID': Column(object, nullable=False, checks=Check.str_matches(r'^C\d{5}$')),
    'country_norm': Column(object, nullable=False, checks=Check.isin(AllowedCountries)),
    'age': Column(pa.Int64, nullable=False, checks=Check.in_range(0, 120)),
    'ltv_usd': Column(float, nullable=False, checks=Check.ge(0)),
    'email': Column(object, nullable=False, checks=Check.str_matches(r'^.+@.+\..+$')),
    'is_adult': Column(bool, nullable=False),
    'is_high_value': Column(bool, nullable=False),
})

clean = UsersSchema.validate(users2, lazy=True)
len(clean)
```

### A2. Cross‑column & DataFrame‑wide checks

```python
# Row-level: if age >= 18 then is_adult must be True
UsersSchema = UsersSchema.update_checks({
    'is_adult': [
        Check(lambda s, df: (~(df['age'] >= 18)) | s, error="is_adult must be True when age>=18")
    ]
})

# DF-level: median ltv_usd must be within a sane band; and all CustomerIDs unique
UsersSchema = UsersSchema.add_checks([
    pa.Check(lambda df: df['ltv_usd'].median() <= 1e5, element_wise=False, error="Median LTV too large"),
    pa.Check(lambda df: df['CustomerID'].is_unique, element_wise=False, error="Duplicate CustomerID")
])

ok = UsersSchema.validate(clean, lazy=True)
ok.head(2)
```

### A3. Friendly error reporting

```python
try:
    # Inject a couple of bad rows to see errors
    broken = clean.copy()
    broken.loc[0, 'age'] = 200
    broken.loc[1, 'email'] = 'not-an-email'
    UsersSchema.validate(broken, lazy=True)
except pa.errors.SchemaErrors as err:
    fc = err.failure_cases
    rollup = (fc.groupby(['column','check']).size().reset_index(name='n')
                .sort_values('n', ascending=False))
    display(rollup.head(10))
    fc.head()
```

**Checkpoint:** How would you surface `rollup` in CI vs provide `failure_cases` as a CSV for analysts?

---

## Part B — Pydantic Row Contracts

Use Pydantic for **message boundaries** (e.g., API payloads) or row‑wise validation when building microservices.

### B1. Define a model with field constraints & validators

```python
from pydantic import BaseModel, Field, EmailStr, ValidationError, field_validator
from typing import Literal

class CustomerRow(BaseModel):
    CustomerID: str = Field(pattern=r'^C\d{5}$')
    country_norm: Literal['USA','DE','SG','BR']
    age: int = Field(ge=0, le=120)
    ltv_usd: float = Field(ge=0)
    email: EmailStr
    is_adult: bool
    is_high_value: bool

    @field_validator('is_adult')
    @classmethod
    def adult_flag_consistent(cls, v, info):
        age = info.data.get('age', None)
        if age is not None and age >= 18 and v is False:
            raise ValueError('is_adult must be true when age>=18')
        return v

row = users2.iloc[0].to_dict()
CustomerRow(**row)

try:
    bad = users2.iloc[1].to_dict() | {'email': 'nope', 'age': 200}
    CustomerRow(**bad)
except ValidationError as e:
    print(e)
```

### B2. Apply to a batch (sample)

```python
def validate_batch(df):
    errors = []
    for i, rec in df.head(50).iterrows():  # sample for speed in demo
        try:
            CustomerRow(**rec.to_dict())
        except ValidationError as e:
            errors.append({'idx': i, 'error': str(e).split('\n')[0]})
    return pd.DataFrame(errors)

validate_batch(users2).head(5)
```

**Guidance:** Pydantic is great for boundaries; Pandera stays the workhorse for DataFrame ETL.

---

## Part C — Error Handling & CI Hooks

### C1. Utility: validate or raise with artifacts

```python
from pathlib import Path

def validate_or_artifact(df, schema, name, out_dir='artifacts/validation'):
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    try:
        return schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as err:
        fc = err.failure_cases
        dest = Path(out_dir)/f'{name}_failures.csv'
        fc.to_csv(dest, index=False)
        # compact summary for console/CI
        top = (fc.groupby(['column','check']).size().reset_index(name='n')
                .sort_values('n', ascending=False).head(5).to_dict(orient='records'))
        raise SystemExit(f"Validation failed for {name}. Top: {top}. See {dest}")

_ = validate_or_artifact(clean, UsersSchema, 'users2_clean')
```

### C2. Pytest smoke test

Create `tests/test_schema.py` (in repo) with:

```python
import pandas as pd, pandera as pa
from lab3b_context import UsersSchema, load_users  # write a tiny loader util in your repo

def test_users_schema():
    df = load_users()  # returns a pandas DataFrame
    try:
        UsersSchema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as err:
        # Fail with a compact roll-up
        fc = err.failure_cases
        top = (fc.groupby(['column','check']).size().reset_index(name='n')
                .sort_values('n', ascending=False).head(5))
        assert False, f"Schema violations:\n{top.to_string(index=False)}"
```

### C3. Pre-commit hook (concept)

Add a pre-commit step that runs `pytest -q` on changed data modules or a tiny CLI that loads the latest Parquet and validates. (Instructor repo will provide a ready YAML in the capstone.)

---

## Part D — Wrap‑Up

Add a markdown cell and answer:

1. One Pandera **column** check and one **DF-level** check you authored. Why both?  
2. Where would you place Pydantic vs Pandera in your pipeline? Give a concrete boundary.  
3. Paste a compact violation roll-up you’d show in CI.

Export the notebook to HTML. If you created a `tests/` file, run `pytest -q` and screenshot the passing test.

---
  
- **Common pitfalls:** Overly strict schemas; forgetting nullable dtypes; using `object` everywhere; not separating row contracts (Pydantic) from DF contracts (Pandera).

---

## Solution Snippets (reference)

**Named regex + range checks:**

```python
Column(object, checks=[Check.str_matches(r'^C\d{5}$', error='bad id')])
Column(pa.Int64, checks=Check.in_range(0,120))
```

**DF-level uniqueness:**

```python
pa.Check(lambda df: df['CustomerID'].is_unique, element_wise=False)
```

**Pydantic field validator:**

```python
@field_validator('is_adult')
def adult_v(cls, v, info):
    return v if info.data.get('age',0) < 18 or v else (_ for _ in ()).throw(ValueError('age>=18 requires is_adult'))
```
