# Lab 05 — Missing Data, Duplicates, & Type Normalization

**Focus Area:** Turning messy raw data into consistent, joined datasets by managing missing data, duplicates, and type normalization — `dropna`, `fillna`, `astype`, `to_datetime`, `drop_duplicates`, `Series.apply`

---

## Outcomes

By the end of this lab, you will be able to:

1. Inspect and quantify missingness with `isna`, `info`, and `value_counts(dropna=False)`.
2. Impute or remove nulls using `fillna`, `ffill/bfill`, group‑wise imputations, and `dropna` with `subset`.
3. Normalize types: parse currency/percent strings to numerics (`str.replace` + `pd.to_numeric`/`astype`), and normalize dates with `pd.to_datetime(..., errors='coerce')` and (optionally) timezone to UTC.
4. Identify and remove duplicates with `duplicated`/`drop_duplicates`, including composite keys and "keep" strategies.
5. Apply `Series.apply` for targeted cleanups and know when to prefer vectorized ops.

---

## Prerequisites & Setup

- Python 3.13 with `pandas`, `numpy`, `pyarrow`, `matplotlib` installed.
- JupyterLab or VS Code with Jupyter extension.
- Reuse data from earlier labs when available:
  - `data/mini_eda_users.parquet` (from **Lab 01**), or create synthetic `users` below.
  - Optional: `artifacts/parquet/orders/` or `orders_joined.parquet` (from **Lab 03**) for bonus checks.

**Start a notebook:** `week02_lab05.ipynb`

If you don’t have the Day 1 files handy, run this to synthesize a messy dataset:

```python
import numpy as np, pandas as pd
rng = np.random.default_rng(123)

n = 1500
users = pd.DataFrame({
    'user_id': np.arange(n),
    'email': [f'user{i}@example.com' if rng.random() > 0.02 else None for i in range(n)],
    'age': rng.integers(16, 80, size=n).astype('float'),  # will inject NaNs
    'country': rng.choice(['US','U.S.A.','usa','SG','DE','BR','IN', None], size=n, p=[.25,.05,.05,.15,.15,.2,.1,.05]),
    'signup_date': rng.choice(['2025-01-05','01/06/2025','06-01-2025','2025/01/07', None], size=n, p=[.25,.25,.25,.2,.05]),
    'spend': rng.choice(['$12,345.60','$0.00','$99','1,234.50','€45,00','', None], size=n, p=[.15,.15,.25,.2,.15,.05,.05]),
    'is_marketing_opt_in': rng.choice([True, False, None], size=n, p=[.45,.5,.05])
})
# Inject duplicates (same user_id appearing twice with slight diffs)
dup_ids = rng.choice(users['user_id'], size=60, replace=False)
users = pd.concat([users, users.loc[users['user_id'].isin(dup_ids)].assign(spend='$0.00')], ignore_index=True)
users.sample(5, random_state=7)
```

---

## Part A — Inspect & Plan Missing‑Data Strategy

### A1. Quick profile of missingness

```python
users.info()
users.isna().mean().sort_values(ascending=False).to_frame('null_frac')
users['country'].value_counts(dropna=False).head(10)
```

**Checkpoint:** Identify which columns can tolerate `NaN` via imputation vs must be present (e.g., `user_id`, `email`).

See **Appendix** below for some discussion points.

### A2. Drop rows only when necessary

```python
# Require a user_id and email; tolerate other nulls for now
clean1 = users.dropna(subset=['user_id','email'])
len(users), len(clean1)
```

**Note:** Use `subset` to avoid dropping rows due to nullable, non‑key fields.

---

## Part B — Imputation (`fillna`, `ffill/bfill`, groupwise) & Type Fixes

### B1. Normalize country category with simple mapping + `fillna`

```python
country_map = {'U.S.A.':'USA','usa':'USA','US':'USA'}
clean1['country_norm'] = clean1['country'].map(country_map).fillna(clean1['country']).fillna('UNKNOWN')
clean1['country_norm'].value_counts().head()
```

### B2. Currency parsing → numeric (`spend`)

We’ll prefer *vectorized* string ops + `pd.to_numeric` and fall back to `apply` for stubborn cases.

```python
# Replace common currency symbols/commas; handle European comma decimals by a heuristic
s = clean1['spend'].astype('string')
s1 = s.str.replace('[ $,]', '', regex=True)
s1 = s1.str.replace('€', '', regex=False)
# Convert comma-decimal like '45,00' -> '45.00' when there is one comma and no dot
s1 = s1.where(~(s1.str.contains('^\d+,\d{1,2}$', regex=True)), s1.str.replace(',', '.', regex=False))
clean1['spend_usd'] = pd.to_numeric(s1, errors='coerce')

# Impute missing spend with group median by country
med = clean1.groupby('country_norm')['spend_usd'].transform('median')
clean1['spend_usd'] = clean1['spend_usd'].fillna(med).fillna(0.0)
clean1['spend_usd'].describe()
```

### B3. Dates to `datetime64[ns]` with parsing variations

```python
clean1['signup_dt'] = pd.to_datetime(clean1['signup_date'], errors='coerce', infer_datetime_format=True, dayfirst=False)
# Optional: normalize to UTC if tz-aware sources are mixed
# clean1['signup_dt'] = clean1['signup_dt'].dt.tz_localize('UTC')  # only if naive and known to be UTC
clean1[['signup_date','signup_dt']].head(8)
```

### B4. Coerce numeric/boolean types with `astype`

```python
# Age may have NaN -> use pandas nullable types if you need to keep nulls
clean1['age'] = clean1['age'].astype('Float64')
# Normalize boolean with fillna then astype
clean1['is_marketing_opt_in'] = clean1['is_marketing_opt_in'].fillna(False).astype('bool')
clean1.dtypes
```

---

## Part C — Duplicates & De‑duplication Strategies

### C1. Detect duplicates by key

```python
# A user should be unique by user_id (business key)
dup_mask = clean1.duplicated(subset=['user_id'], keep=False)
clean_dups = clean1.loc[dup_mask].sort_values(['user_id','signup_dt'])
clean_dups.head(8)
clean_dups['user_id'].nunique(), len(clean_dups)
```

### C2. Resolve duplicates: pick the "best" record per key

Common policies: newest by `signup_dt`, max `spend_usd`, non‑null precedence.

```python
# 1) Prefer newest signup_dt, then higher spend
resolved = (clean1
            .sort_values(['user_id','signup_dt','spend_usd'], ascending=[True, False, False])
            .drop_duplicates(subset=['user_id'], keep='first'))
len(clean1), len(resolved)
```

### C3. Alternative: custom reducer via `groupby().agg`

```python
def coalesce(series):
    # first non-null
    return series.dropna().iloc[0] if series.notna().any() else pd.NA

best = (clean1
        .sort_values(['signup_dt'], ascending=False)
        .groupby('user_id')
        .agg(email=('email','first'),
             country_norm=('country_norm', 'first'),
             signup_dt=('signup_dt','first'),
             spend_usd=('spend_usd','max'),
             age=('age', coalesce),
             is_marketing_opt_in=('is_marketing_opt_in','max'))
        .reset_index())
len(best), best.head()
```

---

## Part D — `Series.apply` vs Vectorized Ops

### D1. When to use `apply`

Use `apply` for moderately complex transforms not easily expressed vectorially (e.g., conditional parsing rules). Compare performance with `%timeit`.

```python
def parse_spend(x: object) -> float | None:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return None
    s = str(x).strip()
    s = s.replace('€','').replace(' ','')
    if s.count(',') == 1 and s.count('.') == 0:
        s = s.replace(',','.')
    s = s.replace('$','').replace(',','')
    try:
        return float(s)
    except ValueError:
        return None

%timeit clean1['spend'].map(parse_spend)
%timeit pd.to_numeric(clean1['spend'].astype('string').str.replace('[ $,]','',regex=True).str.replace('€','',regex=False), errors='coerce')
```

**Guideline:** Prefer vectorized operations for large frames; reserve `apply` for edge cases, then cache results.

---

## Part E — Bonus (Optional) — Apply to orders artifacts

From Day 1 Lab 03, you should have artifacts/parquet/orders/ or orders_joined.parquet. Use them here to practice missing‑data handling, type normalization, and deduplication on a more realistic dataset.

### E1. Load data

```python
from pathlib import Path
import pandas as pd
p_orders = Path('artifacts/parquet/orders')
joined_path = Path('artifacts/parquet/orders_joined.parquet')

if joined_path.exists():
orders = pd.read_parquet(joined_path)
else:
files = sorted(p_orders.glob('shipcountry=*.parquet'))
orders = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)

orders.head(), orders.dtypes
```

### E2. Missingness & types

```python
orders.isna().mean().sort_values(ascending=False).head(10)
orders['OrderDate'] = pd.to_datetime(orders['OrderDate'], errors='coerce')
orders['Freight'] = pd.to_numeric(orders['Freight'], errors='coerce')
```

### E3. Deduplicate line items (composite key)

```python
# If you have line items with duplicate OrderID/ProductID rows, keep the one with max UnitPrice*Quantity
if {'ProductID','Quantity','UnitPrice'}.issubset(orders.columns):
orders['ext_price'] = orders['Quantity'] * orders['UnitPrice']
dedup = (orders
.sort_values(['OrderID','ProductID','ext_price'], ascending=[True,True,False])
.drop_duplicates(subset=['OrderID','ProductID'], keep='first'))
else:
# Fallback: dedupe by OrderID only (newest date wins)
dedup = (orders
.sort_values(['OrderID','OrderDate'], ascending=[True,False])
.drop_duplicates(subset=['OrderID'], keep='first'))

len(orders), len(dedup)
```

### E4. Export cleaned orders

```python
import pyarrow as pa, pyarrow.parquet as pq
out = Path('artifacts/clean')
out.mkdir(parents=True, exist_ok=True)
pq.write_table(pa.Table.from_pandas(dedup, preserve_index=False), out / 'orders_clean.parquet')
```

---

## Part F — Wrap‑Up

- Add a markdown cell and answer:
  1. Which columns did you drop vs impute, and why?
  2. What deduplication policy did you choose? How does it handle ties or null dates?
  3. Show a before/after `dtypes` table and explain at least two `astype`/`to_datetime` choices you made.

Export the cleaned, deduped dataset for downstream labs:

```python
import pyarrow.parquet as pq, pyarrow as pa
from pathlib import Path
out = Path('artifacts/clean')
out.mkdir(parents=True, exist_ok=True)
final = resolved  # or `best` depending on policy
pq.write_table(pa.Table.from_pandas(final, preserve_index=False), out / 'users_clean.parquet')
len(final), (out / 'users_clean.parquet').stat().st_size
```

---

- **Common pitfalls:**
  - Over‑dropping with `dropna()` (forgetting `subset=`) and losing too many rows.
  - `astype(float)` on columns with non‑numeric strings → `ValueError` (use `to_numeric(..., errors='coerce')`).
  - Duplicates resolved inconsistently when sort order isn’t deterministic; always sort before `drop_duplicates`.
  - Mixing `object`, `string`, and nullable dtypes; be explicit.

---

## Solution Snippets (reference)

**Drop only when key is missing:**

```python
clean1 = users.dropna(subset=['user_id','email'])
```

**Group‑wise imputation:**

```python
clean1['spend_usd'] = (pd.to_numeric(clean1['spend'].astype('string').str.replace('[ $,]','',regex=True), errors='coerce')
                       .groupby(clean1['country'].fillna('UNKNOWN')).transform(lambda s: s.fillna(s.median())))
```

**Type normalization:**

```python
clean1['age'] = clean1['age'].astype('Float64')
clean1['signup_dt'] = pd.to_datetime(clean1['signup_date'], errors='coerce')
```

**Duplicates by composite key:**

```python
dedup = (df
         .sort_values(['business_key','updated_at'], ascending=[True, False])
         .drop_duplicates(subset=['business_key'], keep='first'))
```

**`apply` for targeted cleanup:**

```python
def normalize_country(x):
    if x is None: return 'UNKNOWN'
    s = str(x).replace('.','').upper()
    return {'USA':'USA','US':'USA','SG':'SG','DE':'DE','BR':'BR','IN':'IN'}.get(s, s)
users['country_norm'] = users['country'].apply(normalize_country)
```

---

## **Appendix - Managing `NaN`**

### Quick decision rules

#### Must be present (no `NaN`; drop or backfill from a trusted source)

- **Business/Join keys:** `user_id`, `order_id`, composite keys (`user_id`,`event_ts`) — needed for dedupe, joins, and referential integrity.
- **Timestamps used for sequencing/watermarks:** `signup_dt`, `created_at`, `updated_at` — required for incremental pipelines and "latest-wins" dedupe.
- **Hard requirements for downstream logic:** e.g., `email` if you must contact the user, or `country` if geography drives compliance rules.
- **Columns that define uniqueness or partitioning:** anything used in `drop_duplicates(subset=...)` or partitioned Parquet paths.

**Action:** `dropna(subset=[...])` early; optionally log counts so you know the cost.

```python
required = ['user_id', 'email']  # adjust per use case
clean = df.dropna(subset=required)
lost = len(df) - len(clean)
print(f"dropped {lost} rows missing required fields: {required}")
```

#### Can tolerate NaN (impute or leave null with nullable dtypes)

- Measures that can be estimated: `spend_usd`, `age`, `sessions`, `discount`. Use median/mean, or group-wise medians (by `country_norm`, segment).
- **Booleans that default safely:** `is_marketing_opt_in` → default `False` if that’s the safer policy.
- **Categoricals used for descriptive slice, not keys:** `device_type`, `marketing_channel`, `segment`.
- **Free-text/optional attributes:** `notes`, `middle_name` — often leave as null.

**Action:** choose imputation that won’t distort analysis; prefer group-wise and document it.

```python
df['spend_usd'] = df['spend_usd'].groupby(df['country_norm']).transform(
    lambda s: s.fillna(s.median())
).fillna(0.0)

df['is_marketing_opt_in'] = df['is_marketing_opt_in'].fillna(False).astype('bool')
```

### Heuristics & trade-offs

- **If a field is used to *join*, *dedupe*, *partition*, or *sequence*** → treat as required.
- **If a field influences metrics/models materially** (e.g., label or key feature) → prefer informed imputation (group-wise, domain rules) or exclude from model features if missingness is not random.
- **High missingness (>30–40%)** → consider leaving nulls with nullable dtypes, or engineer a "missing" indicator rather than aggressive imputation.
- **Dates:** Always coerce with `errors='coerce'`, then decide: drop missing *if* needed for ordering/watermarks; otherwise imputing with a sentinel is risky (can break time logic).

```python
df['signup_dt'] = pd.to_datetime(df['signup_date'], errors='coerce')
needs_time = df.dropna(subset=['signup_dt'])        # for sequencing/dedupe
```

### LLM-pipeline nuance

- **Provenance & incrementals:** Watermarks rely on valid timestamps → missing here should be dropped or backfilled from a trusted canonical date (not a made-up default).
- **Text fields destined for prompts:** Missing optional text (`notes`) is fine; keep nulls and handle downstream (e.g., template "N/A" at render time rather than polluting source data).
- Avoid leakage: Don’t impute labels or outcomes (if you later use this dataset for modeling).

### Make it explicit with a tiny contract

```python
import pandera as pa
from pandera import Column, Check

Schema = pa.DataFrameSchema({
    "user_id": Column(int, nullable=False),
    "email": Column(object, nullable=False),
    "signup_dt": Column(object, nullable=False),  # after to_datetime
    "country_norm": Column(object, nullable=True),
    "spend_usd": Column(float, Check.ge(0), nullable=True),
    "is_marketing_opt_in": Column(bool, nullable=False),
})
Schema.validate(clean, lazy=True)
```

### TL;DR mapping for the lab’s dataset

- **Required:** `user_id`, `email`, `signup_dt` (post-parse) → `dropna(subset=...)`.
- **Impute:** `spend_usd` (group median by `country_norm`, then 0.0), `age` (nullable Float64 or group median), `is_marketing_opt_in` (default `False`).
- **Normalize/optional:** `country_norm` (map + `fillna('UNKNOWN')`) — acceptable to keep `UNKNOWN` for analysis.
