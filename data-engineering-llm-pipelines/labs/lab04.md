# Lab 04 — Pandas Selection & Filtering

**Focus Area:** Turning messy raw data into consistent, joined datasets using Pandas selection & filtering (`loc`, boolean masks, `query`, chained masks)

---

## Outcomes

By the end of this lab, you will be able to:

1. Construct correct boolean masks using `&`, `|`, and `~` with parentheses.
2. Select rows/columns using `df.loc[row_mask, col_list]` vs. positional `iloc`.
3. Apply chained masks safely without the chained‑assignment trap.
4. Use `DataFrame.query` for readable filters (and when to prefer it vs. masks).
5. Build reusable filter functions and compose them for clarity.

---

## Prerequisites & Setup

- Python 3.13 with `pandas`, `numpy`, `pyarrow`, `matplotlib` installed.
- JupyterLab or VS Code with Jupyter extension.
- Reuse artifacts from Day 1:
  - `data/mini_eda_users.parquet` (from Lab 01 export) **or** run the synthetic cell below.
  - `artifacts/parquet/orders/` (optional from Lab 03; used in a bonus section).

**Start a notebook:** `week02_lab04.ipynb`

If you don’t have the Day 1 files handy, create a small synthetic dataset now:

```python
import numpy as np, pandas as pd
rng = np.random.default_rng(42)
n = 1000
users = pd.DataFrame({
    'user_id': np.arange(n),
    'age': rng.integers(16, 80, size=n),
    'country': rng.choice(['US','U.S.A.','USA','SG','DE','BR','IN'], size=n, p=[.25,.05,.1,.15,.15,.15,.15]),
    'sessions': rng.poisson(3, size=n),
    'avg_session_sec': rng.normal(300, 60, size=n).clip(30, 1500),
    'spend_usd': np.round(rng.lognormal(mean=3.0, sigma=0.7, size=n), 2)
})
users.head()
```

---

## Part A — Boolean Masks & Parentheses

### A1. Build masks correctly

```python
adults = users['age'] >= 18
heavy_users = users['sessions'] >= 5
us_like = users['country'].isin(['US','U.S.A.','USA'])

# Combine with parentheses!
mask = adults & heavy_users & us_like

# Select rows and a subset of columns
view = users.loc[mask, ['user_id','age','country','sessions','avg_session_sec']]
view.head(), view.shape
```

**Checkpoint:** Why are parentheses required? What happens if you write `adults & heavy_users == True`?

### A2. Negation and OR

```python
low_engagement = users['sessions'] <= 1
non_us = ~us_like
subset = users.loc[low_engagement | non_us, ['user_id','country','sessions']]
subset.sample(5, random_state=1)
```

**Quick note:** `~` negates a boolean Series; ensure the operand has boolean dtype.

---

## Part B — `loc` vs `iloc` & Avoiding Chained Assignment

### B1. Label vs positional selection

```python
# loc uses labels; iloc uses integer positions
first_10_by_label = users.loc[0:9, ['user_id','age']]
first_10_by_pos = users.iloc[0:10, [0,1]]  # same rows; columns by index

# Correct single-column assign with loc (avoids chained assignment)
users.loc[adults, 'is_adult'] = True
users.loc[~adults, 'is_adult'] = False
users[['age','is_adult']].head()
```

### B2. Chained assignment trap demo

#### What the "chained assignment trap" is

In pandas, expressions like `df[cond][col] = value` may operate on a temporary view or a temporary copy of your data. If it’s a copy, the assignment modifies only the temporary object — not the original `df`. Because pandas has to balance speed and memory, whether you get a view or a copy is not guaranteed across versions/operations. That uncertainty is the "trap."

- **Chaining = two indexing steps** (e.g., `df[cond][col]`) instead of a single `.loc` step.
- **Symptom:** You see a `SettingWithCopyWarning` (often), or worse, *no warning* but the change silently doesn’t stick.

```python
# Bad (may or may not modify users)
tmp = users[users['sessions'] > 5]
tmp['flag'] = 1
users['flag'].isna().sum()  # Often shows many NaNs → users wasn't updated

# Good (always modifies users)
users.loc[users['sessions'] > 5, 'flag'] = 1
users['flag'].isna().sum()  # Now the count of NaNs should drop appropriately
```

- `tmp = users[users['sessions'] > 5]` creates a temporary object. It might be a view or a copy.
- `tmp['flag'] = 1` tries to write into that temporary object.
  - If `tmp` is a copy, `users` is unchanged (bug!).
  - If `tmp` is a view, it works *this time* — but you can’t rely on it.
- The **correct** version uses one indexing operation: `users.loc[mask, 'flag'] = 1`. This targets users directly and is reliable.

#### How to avoid it (rules of thumb)

- Use `.loc` for assignment:
  - `df.loc[row_mask, 'col'] = value`
- If you *intend* to work on a subset, make it explicit:
  - `subset = df[cond].copy()` → mutate `subset`, then merge it back or use it downstream.
- Don’t chain when assigning (reading is fine):
  - Prefer `df.loc[cond, 'col']` over `df[cond]['col']`.
- Normalize complex conditions into variables (clear & safer):

```python
m = (df.a > 0) & (df.b == 'x')
df.loc[m, 'flag'] = 1
```

- Alternative patterns that avoid the trap:
  - `df.assign(flag=np.where(m, 1, df['flag']))`
  - `df.loc[:, 'flag'] = np.where(m, 1, df['flag'])`
  - `df['flag'] = df['flag'].where(~m, 1)`

#### Quick mental model

- **One-step selection + assignment (`.loc`)** → modifies the original reliably.
- **Two-step chained selection + assignment** → *maybe* modifies the original; pandas may warn, but not always.

**Checkpoint:** Explain why `tmp['flag'] = 1` may not modify `users`.

---

## Part C — Chained Masks & Reusable Filters

### C1. Compose filters

```python
def f_is_adult(df):
    return df['age'] >= 18

def f_high_value(df, p=90):
    thr = df['spend_usd'].quantile(p/100)
    return df['spend_usd'] >= thr

def f_core_markets(df):
    return df['country'].isin(['US','SG','DE'])

mask = f_is_adult(users) & f_high_value(users, 85) & f_core_markets(users)
hv_core = users.loc[mask, ['user_id','age','country','spend_usd']]
len(hv_core), hv_core.head()
```

### C2. Multi-condition with between/isin/str.contains

```python
mask2 = users['age'].between(25, 40) & users['country'].isin(['US','SG'])
mask3 = users['country'].str.contains('US', regex=False)  # matches 'US' and 'U.S.A.'? (no)

# Safer normalization before contains
norm = users['country'].str.replace('.', '', regex=False).str.upper()
mask3b = norm.str.contains('USA')
filtered = users.loc[mask2 & mask3b]
filtered.head()
```

**Checkpoint:** Why normalize before `contains`? Show counts difference.

---

## Part D — `DataFrame.query` & `eval` (≈15 min)

### D1. Using `query`

```python
# Query understands column names directly; use @ for Python vars
min_sess = 3
q1 = users.query('(age >= 18) & (sessions >= @min_sess) & country in ["US", "U.S.A.", "USA"]')
q1[['user_id','age','country','sessions']].head()
```

### D2. When to choose query vs masks

- Prefer **masks** when you need IDE/type support, refactors, or complex Python expressions.
- Prefer **`query`** for **readability** in complex boolean logic across many columns (esp. in notebooks/presentations).

### D3. Bonus: `eval` for computed columns in‑place

```python
# Using eval can be faster for large frames and avoids temporary objects
users = users.eval('engagement = sessions * avg_session_sec')
crit = users.query('engagement >= @users.engagement.quantile(0.9)')
len(crit)
```

---

## Part E — Bonus (Optional) — Filter orders Parquet artifacts

From Day 1 Lab 03, you should have artifacts/parquet/orders/shipcountry=*.parquet files. Use them here to practice filtering on a larger, partitioned dataset.

### E1. Load partitioned Parquet

```python
from pathlib import Path
import pandas as pd
p = Path('artifacts/parquet/orders')
files = sorted(p.glob('shipcountry=*.parquet'))
orders = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
orders.head(), len(orders)
```

### E2. Boolean masks vs `query`

```python
# Masks
m_country = orders['ShipCountry'].isin(['USA','DE'])
m_date = pd.to_datetime(orders['OrderDate'], errors='coerce').between('1997-01-01','1998-12-31')
m_freight = orders['Freight'] >= orders['Freight'].quantile(0.9)
subset_mask = orders.loc[m_country & m_date & m_freight, ['OrderID','CustomerID','ShipCountry','OrderDate','Freight']]

# Query (note: ensure OrderDate is string or convert to dtype=datetime64[ns] and use @vars)
orders2 = orders.assign(OrderDate=pd.to_datetime(orders['OrderDate'], errors='coerce'))
q = orders2.query('ShipCountry in ["USA","DE"] and (OrderDate >= @pd.Timestamp("1997-01-01")) and (OrderDate <= @pd.Timestamp("1998-12-31")) and Freight >= @orders2.Freight.quantile(0.9)')
subset_query = q[['OrderID','CustomerID','ShipCountry','OrderDate','Freight']]

len(subset_mask), len(subset_query)
```

### E3. Group & sanity‑check

```python
subset_mask.groupby('ShipCountry').size().sort_values(ascending=False)
```

**Checkpoint:** Confirm mask and query produce the same number of rows. If not, inspect dtype differences and how dates were handled.

---

## Part F — Wrap‑Up

- Add a markdown cell and answer:
  1. Show two equivalent filters: one with masks and one with `query`.
  2. Why is `df.loc[mask, 'col'] = ...` preferred over chained assignment?
  3. When might `eval` or `query` *not* be appropriate (hint: dynamic column names, complex Python functions)?

---

- **Common pitfalls:**
  - Missing parentheses in `&`/`|` combinations.
  - Confusing `and`/`or` (Python scalars) with `&`/`|` (elementwise).
  - SettingWithCopy warnings from chained assignment.

---

## Solution Snippets (reference)

**Mask vs query equivalence:**

```python
mask = (users['age'] >= 18) & users['sessions'].between(3, 10) & users['country'].isin(['US','U.S.A.','USA'])
sol_mask = users.loc[mask, ['user_id','age','sessions','country']]
sol_query = users.query('(age >= 18) and (sessions >= 3) and (sessions <= 10) and country in ["US","U.S.A.","USA"]').loc[:, ['user_id','age','sessions','country']]
```

**Normalized contains:**

```python
norm = users['country'].str.replace('.', '', regex=False).str.upper()
sol = users.loc[norm.isin(['USA','SG','DE'])]
```

**Avoid chained assignment:**

```python
users.loc[users['sessions'] > 5, 'flag'] = 1
```
