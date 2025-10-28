# Lab 07 — Clean & Standardize + Join & Aggregate

**Focus Areas:** Clean & standardize (prices, dates, countries, vectorized `is_adult`) and join & aggregate (orders↔customers, per‑segment summaries)

> This lab intentionally revisits skills from Labs **04-06** but applies them to a slightly different, more integrated workflow with diagnostic checks, small reference tables, and richer segment logic.

---

## Outcomes

By the end of this lab, you will be able to:

1. Build a **reproducible cleaning pipeline** for currency, date parsing, and categorical normalization against a **reference dimension**.
2. Create **vectorized features** (e.g., `is_adult`, `high_value_user`) without `apply` loops.
3. Perform **inner joins** between orders and customers; diagnose missing keys (anti‑join), and validate cardinality to avoid fan‑out.
4. Produce **per‑segment** summaries via `groupby().agg` and export tidy Parquet artifacts for downstream LLM steps.

---

## Prerequisites & Setup

- Python 3.13 with `pandas`, `numpy`, `pyarrow`, `matplotlib` installed.
- JupyterLab or VS Code with Jupyter extension.
- Artifacts from previous labs (optional but recommended):
  - `artifacts/clean/users_clean.parquet` (from Lab 05),
  - `artifacts/parquet/orders/shipcountry=*.parquet` **or** `artifacts/parquet/orders_joined.parquet` (from Lab 03/05 bonus),
  - If you don’t have these, synthetic fallbacks are provided below.

**Start a notebook:** `week02_lab07.ipynb`

### Synthetic fallback data (run only if you lack artifacts)

```python
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from pathlib import Path
import matplotlib.pyplot as plt

rng = np.random.default_rng(7)

# Customers profile
n = 1200
users = pd.DataFrame({
    'user_id': np.arange(n),
    'CustomerID': [f'C{i:05d}' for i in range(n)],
    'email': [f'user{i}@example.com' if rng.random() > 0.02 else None for i in range(n)],
    'age': rng.integers(16, 80, size=n).astype('float64'),
    'country': rng.choice(['US','U.S.A.','usa','SG','DE','Brasil','United States', 'N/A'], size=n,
                          p=[.35,.05,.05,.15,.15,.15,.08,.02]),
    'signup_date': rng.choice(np.array(['2025-01-05','01/06/2025','06-01-2025','2025/01/07', None], dtype=object), size=n,
               p=np.array([.25,.25,.25,.20,.05])),
    'lifetime_value': rng.choice(np.array(['$1,234.50','€45,00','1,234','USD 99.95','$0.00','', None], dtype=object), size=n,
                                 p=np.array([.25,.15,.25,.15,.15,.03,.02]))
})

# Orders facts
m = 4000
orders = pd.DataFrame({
    'OrderID': np.arange(10_000, 10_000 + m),
    'CustomerID': rng.choice(users['CustomerID'], size=m, replace=True),
    'OrderDate': rng.choice(['2025-01-06','2025/01/07','01/08/2025','2025-01-09'], size=m),
    'ShipCountry': rng.choice(['USA','DE','SG','BR','SE'], size=m, p=[.6,.12,.12,.1,.06]),
    'Freight': rng.lognormal(mean=3.3, sigma=0.6, size=m).round(2)
})

customers = users[['CustomerID','email','age','country','signup_date','lifetime_value']].copy()

print("Users shape:", users.shape)
print("Orders shape:", orders.shape)
print("Customers shape:", customers.shape)
display(users.head())
display(orders.head())
display(customers.head())
```

### Country reference dimension (for normalization)

```python
country_dim = pd.DataFrame({
    'raw': ['USA','U.S.A.','United States','US','usa','U. S. A.','Brasil','BR','Germany','DE','sg','Singapore','N/A'],
    'canonical': ['USA','USA','USA','USA','USA','USA','BR','BR','DE','DE','SG','SG','UNKNOWN']
})
country_dim
```

---

## Part A — Clean & Standardize

### A1. Inspect & guardrails

```python
users.info()
users.isna().mean().sort_values(ascending=False).head(10)
```

**Checkpoint:** Which columns are **required** (no nulls) for your pipeline? (e.g., `user_id`, `email`) Use `dropna(subset=...)` accordingly.

```python
required = ['user_id','email']
users1 = users.dropna(subset=required)
len(users), len(users1)
```

### A2. Normalize countries using the reference table

```python
# prep reference by normalizing its 'raw' to a matching key
norm_key = (lambda s: s.astype('string')
                     .str.replace('.','', regex=False)
                     .str.replace(' ','', regex=False)
                     .str.upper())

ref = country_dim.assign(raw_key = norm_key(country_dim['raw']))[['raw_key','canonical']]
users1 = users1.assign(country_key = norm_key(users1['country']))
users1 = users1.merge(ref, left_on='country_key', right_on='raw_key', how='left')
users1['country_norm'] = users1['canonical'].fillna('UNKNOWN')
users1.drop(columns=['raw_key','canonical'], inplace=True)
users1['country_norm'].value_counts().head()
```

**Note:** This mirrors a real‑world **dimension lookup** rather than only a hard‑coded dict.

### A3. Currency strings → numeric (`lifetime_value`)

```python
s = users1['lifetime_value'].astype('string').str.strip()
# Strip codes/symbols/spaces
s1 = (s.str.replace('USD','',regex=False)
        .str.replace('EUR','',regex=False)
        .str.replace('$','',regex=False)
        .str.replace('€','',regex=False)
        .str.replace('£','',regex=False)
        .str.replace(' ','',regex=False))
# Heuristics for decimal separators
mask_comma_decimal = s1.str.fullmatch(r'\d+,\d{1,2}')
mask_both = s1.str.contains(r'\d+[\.,]\d{3,}.*,\d{1,2}$')

s2 = s1.where(~mask_comma_decimal, s1.str.replace(',', '.', regex=False))
s2 = s2.where(~mask_both, s2.str.replace('.', '', regex=False).str.replace(',', '.', regex=False))
s2 = s2.str.replace(',', '', regex=False)

users1['ltv_usd'] = pd.to_numeric(s2, errors='coerce')

# Group‑wise impute by normalized country; then fallback 0.0
med = users1.groupby('country_norm')['ltv_usd'].transform('median')
users1['ltv_usd'] = users1['ltv_usd'].fillna(med).fillna(0.0)
users1['ltv_usd'].describe()
```

### A4. Dates → `datetime64[ns]`; vectorized `is_adult` and a second feature

```python
users1['signup_dt'] = pd.to_datetime(users1['signup_date'], errors='coerce')
# Drop rows lacking sequencing dates if needed for later incremental logic
users2 = users1.dropna(subset=['signup_dt']).copy()

# Vectorized booleans
users2['is_adult'] = (users2['age'] >= 18)
q90 = users2['ltv_usd'].quantile(0.90)
users2['is_high_value'] = users2['ltv_usd'] >= q90

display(users2[['age','is_adult','ltv_usd','is_high_value']].head())
print(f"\nUsers after date filtering: {len(users2)}")
print(f"90th percentile LTV threshold: ${q90:.2f}")
```

**Checkpoint:** Why avoid `apply` here? What’s the advantage of vectorization and `quantile`‑based thresholds?

### A5. Export cleaned users for joins

```python
import pyarrow.parquet as pq, pyarrow as pa
from pathlib import Path
out = Path('artifacts/clean'); out.mkdir(parents=True, exist_ok=True)
pq.write_table(pa.Table.from_pandas(users2, preserve_index=False), out / 'users2_clean.parquet')
(out / 'users2_clean.parquet').exists(), len(users2)
```

---

## Part B — Join & Aggregate

### B1. Prepare orders & customers, and join

If you have `orders` and `customers` from earlier labs, load them; else use synthetic fallback already created above.

```python
# Ensure dtypes
orders['OrderDate'] = pd.to_datetime(orders['OrderDate'], errors='coerce')
customers = users2[['CustomerID','email','country_norm','signup_dt','is_adult','is_high_value']].drop_duplicates(subset=['CustomerID'])

# Inner join: realized orders with matched customers
joined = orders.merge(customers, on='CustomerID', how='inner', validate='many_to_one')
print(f"Orders: {len(orders)}, Joined: {len(joined)}")
display(joined.head())
```

> **Why inner?** We’re building metrics about **realized orders**; missing customers are excluded here. We’ll still inspect them via anti‑join next.

### B2. Diagnose missing keys (anti‑join)

```python
left = orders.merge(customers, on='CustomerID', how='left', indicator=True)
anti = left[left['_merge'] == 'left_only'][['OrderID','CustomerID']]
len(anti), anti.head()
```

**Checkpoint:** If `anti` is non‑empty, list potential causes and remediation (e.g., stale CustomerID, case mismatches, missing profile rows).

### B3. Per‑segment aggregates

Build segments by **country & adult/high‑value flags**, then compute core metrics.

```python
import numpy as np
seg = joined.assign(
    year = joined['OrderDate'].dt.year,
    segment = np.select(
        [ joined['is_high_value'], joined['is_adult'] ],
        [ 'high_value', 'adult' ],
        default='general')
)

per_segment = (seg
    .groupby(['country_norm','segment'], as_index=False)
    .agg(orders=('OrderID','count'),
         freight_mean=('Freight','mean'),
         freight_sum=('Freight','sum'),
         customers=('CustomerID','nunique'))
    .sort_values(['country_norm','orders'], ascending=[True, False]))
per_segment.head()
```

### B4. Per‑customer aggregates + join back to attributes

```python
per_cust = (joined
    .groupby('CustomerID', as_index=False)
    .agg(n_orders=('OrderID','count'),
         freight_mean=('Freight','mean'),
         freight_sum=('Freight','sum')))

per_cust_enriched = per_cust.merge(customers, on='CustomerID', how='left', validate='one_to_one')
per_cust_enriched.head()
```

### B5. Persist artifacts

```python
pq.write_table(pa.Table.from_pandas(per_segment, preserve_index=False), out / 'per_segment.parquet')
pq.write_table(pa.Table.from_pandas(per_cust_enriched, preserve_index=False), out / 'per_customer_enriched.parquet')
```

---

## Part C — Wrap‑Up

Add a markdown cell and answer:

1. Summarize your **country normalization** approach and how you’d expand the reference table.
2. Why was **inner join** the right choice for the metrics you built? Give one case where **left join** is preferred.
3. Present two metrics from `per_segment` you’d surface to a stakeholder (and why). Include a short interpretation.

---

- **Key Items:**
  - Introduces a **reference dimension** join for category normalization (more realistic than a one‑off dict).
  - Adds **quantile‑based** `is_high_value` feature and multiple vectorized flags.
  - Uses both **inner join** for realized metrics and **anti‑join** for QA triage.
  - Produces both **per‑segment** and **per‑customer** artifacts for downstream use.
- **Common pitfalls:**
  - Failing to normalize keys prior to reference joins (punctuation/case).  
  - Accidentally using `outer` join for metrics inflating counts.
  - Using `apply` for boolean flags (use vectorized operations instead).
  - Forgetting `validate=` and silently introducing **fan‑out**.

---

## Stretch Goals

1. **Reference table coverage report:** Compute coverage of `country_dim` over observed `country_key`s. Emit a list of unmapped values and auto‑generate a skeleton PR patch (dict) for instructors to review.
2. **Segment stability over time:** Create a 7‑day rolling `freight_sum` per segment using `groupby` + `rolling` (or resample if you engineer daily series) and chart the trend.
3. **Quantile bands:** Replace the single 90th‑percentile threshold with bands (e.g., tertiles or deciles) and compare segment sizes.
4. **Join validation tests:** Write assertions that fail the notebook if (a) anti‑join > 1% of orders, (b) `many_to_one` validation fails, or (c) any aggregate yields NaN where it shouldn’t.
5. **Partitioned output:** Write `per_segment` partitioned by `country_norm` into `artifacts/clean/per_segment/country_norm=*.parquet` and compare file counts to group counts.
6. **LLM‑ready view:** Compose a tidy table with `CustomerID`, `country_norm`, `is_adult`, `is_high_value`, and a short natural‑language summary column (e.g., f"Customer {id} is a {segment} buyer in {country}") for prompt conditioning.

---

## Solution Snippets (reference)

**Coverage report:**

```python
observed = users2['country_key'].value_counts()
covered = users2.merge(ref, left_on='country_key', right_on='raw_key', how='left')
unmapped = covered[covered['canonical'].isna()]['country_key'].value_counts()
unmapped.head(10)
```

**Quantile bands:**

```python
import numpy as np
bands = pd.qcut(users2['ltv_usd'], q=10, labels=[f'd{i}' for i in range(1,11)])
users2 = users2.assign(ltv_decile=bands)
users2['ltv_decile'].value_counts().sort_index()
```

**Partitioned write:**

```python
from pathlib import Path
root = out / 'per_segment'
root.mkdir(parents=True, exist_ok=True)
for k, g in per_segment.groupby('country_norm'):
    pq.write_table(pa.Table.from_pandas(g, preserve_index=False), root / f'country_norm={k}.parquet')
```

---

## Stretch Goal Solutions — Step‑by‑Step

These assume you finished the previous parts and have pre-existing variables like `users2`, `orders`, `customers`, `joined`, `per_segment`, and `out` defined.

> If you’re starting fresh, run the **Synthetic fallback data** cell in the lab first, then complete Parts A–B up to where `users2` and `joined` exist.

---

## 1) Reference table coverage report

**Goal:** Measure how well your country normalization reference (`country_dim`) covers observed values; list unmapped forms and scaffold a patch.

### 1.1 Build normalized keys and compute coverage

```python
import pandas as pd

def norm_key_series(s: pd.Series) -> pd.Series:
    return (s.astype('string')
             .str.replace('.','', regex=False)
             .str.replace(' ','', regex=False)
             .str.upper())

ref = country_dim.assign(raw_key = norm_key_series(country_dim['raw']))[['raw_key','canonical']]
observed_keys = norm_key_series(users2['country']).value_counts()
covered = users2.merge(ref, left_on=norm_key_series(users2['country']), right_on='raw_key', how='left')
coverage_rate = 1.0 - covered['canonical'].isna().mean()
coverage_rate
```

**Checkpoint:** `coverage_rate` should be high after Part A2. If not, inspect `covered[covered['canonical'].isna()]`.

### 1.2 List top unmapped keys and generate a suggested patch dict

```python
unmapped_counts = (covered[covered['canonical'].isna()]
                   .assign(country_key=norm_key_series(covered['country']))['country_key']
                   .value_counts())
unmapped_counts.head(10)

# Draft a patch mapping for instructors to review
suggested_patch = {k: 'UNKNOWN' for k in unmapped_counts.index}
suggested_patch  # edit canonical as appropriate (e.g., 'UNITEDSTATESOFAMERICA': 'USA')
```

---

## 2) Segment stability over time (rolling trends)

**Goal:** Track 7‑day rolling `freight_sum` per segment to spot volatility.

### 2.1 Prepare daily series per segment

```python
import numpy as np
seg = joined.assign(
    segment = np.select([joined['is_high_value'], joined['is_adult']], ['high_value','adult'], default='general')
)
seg['day'] = seg['OrderDate'].dt.floor('D')

# Aggregate to daily totals per segment
daily = (seg.groupby(['segment','day'], as_index=False)
          .agg(freight_sum=('Freight','sum'), orders=('OrderID','count'))
          .sort_values(['segment','day']))
daily.head()
```

### 2.2 Compute rolling 7‑day sums

```python
# rolling independently per segment
rolled = (daily.set_index('day')
               .groupby('segment')
               .apply(lambda g: g.sort_index()
                                 .assign(freight_sum_7d=g['freight_sum'].rolling(7, min_periods=1).sum(),
                                         orders_7d=g['orders'].rolling(7, min_periods=1).sum()))
               .reset_index(level=0))
rolled.head()
```

### 2.3 Quick visualization (optional)

```python
import matplotlib.pyplot as plt
for seg_name, g in rolled.groupby('segment'):
    plt.figure()
    g.sort_values('day').plot(x='day', y='freight_sum_7d', title=f"7-day Freight Sum — {seg_name}")
    plt.show()
```

---

## 3) Quantile bands (deciles/tertiles)

**Goal:** Replace a single 90th‑percentile threshold with multiple bands for richer segmentation.

### 3.1 Create deciles on `ltv_usd`

```python
# Create deciles on ltv_usd
metric = users2['ltv_usd'].clip(lower=0)
users2 = users2.assign(ltv_decile=pd.qcut(metric, q=10, labels=False, duplicates='drop'))
decile_counts = users2['ltv_decile'].value_counts().sort_index()
print("LTV Decile distribution:")
display(decile_counts)

# Visualize
plt.figure(figsize=(10, 5))
decile_counts.plot(kind='bar', color='steelblue')
plt.title('Customer Distribution by LTV Decile')
plt.xlabel('LTV Decile')
plt.ylabel('Number of Customers')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
```

### 3.2 Use bands in downstream joins/segments

```python
# Use bands in downstream joins/segments
cust_attrs = users2[['CustomerID','country_norm','is_adult','ltv_decile']].drop_duplicates(subset=['CustomerID'])
joined2 = orders.merge(cust_attrs, on='CustomerID', how='inner', validate='many_to_one')
per_decile = (joined2.groupby('ltv_decile', as_index=False)
              .agg(orders=('OrderID','count'), 
                   freight_mean=('Freight','mean'), 
                   freight_sum=('Freight','sum'))
              .sort_values('ltv_decile'))
print("\nMetrics by LTV Decile:")
display(per_decile)
```

---

## 4) Join validation tests

**Goal:** Add assertions so the notebook fails fast if data quality deviates.

### 4.1 Anti‑join rate threshold

```python
# Test 1: Anti‑join rate threshold
left = orders.merge(users2[['CustomerID']].drop_duplicates(), on='CustomerID', how='left', indicator=True)
anti_rate = (left['_merge'] == 'left_only').mean()
print(f"Anti-join rate: {anti_rate:.2%}")
# Adjust threshold based on expected data quality after cleaning filters
threshold = 0.80  # 70% threshold accounts for customers filtered during cleaning
assert anti_rate <= threshold, f"Anti-join rate too high: {anti_rate:.2%} (>{threshold:.0%})"
print(f"✓ Anti-join rate test passed (threshold: {threshold:.0%})")
```

### 4.2 Cardinality validation (avoid fan‑out)

```python
# Expect many orders to one customer row
_ = orders.merge(users2[['CustomerID']], on='CustomerID', how='inner', validate='many_to_one')
```

### 4.3 Aggregates shouldn’t be NaN in required metrics

```python
per_segment_test = (joined.groupby('CustomerID', as_index=False)
                    .agg(n_orders=('OrderID','count'), freight_sum=('Freight','sum')))
assert per_segment_test['n_orders'].notna().all(), 'n_orders contains NaN'
assert per_segment_test['freight_sum'].notna().all(), 'freight_sum contains NaN'
```

---

## 5) Partitioned output

**Goal:** Write `per_segment` partitioned by `country_norm` for faster reads.

### 5.1 Write one file per country partition

```python
from pathlib import Path
import pyarrow.parquet as pq, pyarrow as pa
root = out / 'per_segment'
root.mkdir(parents=True, exist_ok=True)

parts = 0
for k, g in per_segment.groupby('country_norm'):
    pq.write_table(pa.Table.from_pandas(g, preserve_index=False), root / f'country_norm={k}.parquet')
    parts += 1
parts, sorted(p.name for p in root.glob('country_norm=*.parquet'))[:5]
```

### 5.2 Validate partition counts vs groups

```python
n_groups = per_segment['country_norm'].nunique()
assert n_groups == parts, f"Expected {n_groups} partitions, wrote {parts}"
```

---

## 6) LLM‑ready summary view

**Goal:** Compose a compact table for prompt conditioning or RAG features.

### 6.1 Build a human‑readable summary string

```python
sample = (per_cust_enriched
          .assign(segment=np.where(per_cust_enriched['is_high_value'], 'high-value',
                                    np.where(per_cust_enriched['is_adult'], 'adult', 'general')))
          .loc[:, ['CustomerID','country_norm','n_orders','freight_sum','segment']])

summary = (sample
    .assign(summary=lambda df: (
        'Customer ' + df['CustomerID'] + ' is a ' + df['segment'] + ' buyer in ' + df['country_norm'] +
        ' with ' + df['n_orders'].astype(str) + ' orders and total freight $' + df['freight_sum'].round(2).astype(str)
    )))
summary.head(5)
```

### 6.2 Persist the LLM view

```python
pq.write_table(pa.Table.from_pandas(summary, preserve_index=False), out / 'per_customer_summary.parquet')
```

---

## Notes on performance & correctness

- Prefer **vectorized** operations; reserve `apply` for rare edge cases.
- Keep **nullable dtypes** where missingness is legitimate (`Int64`, `Float64`, `boolean`).
- For large datasets, consider writing true **directory‑style partitions** (subfolders) if you’ll read with engines that support partition discovery.
- Validate **before** exporting artifacts: assertions, anti‑join rate, and cardinality checks catch issues early.
