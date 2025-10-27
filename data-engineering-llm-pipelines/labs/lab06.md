# Lab 06 — GroupBy & Joins

**Focus Area:** Turning messy raw data into consistent, joined datasets by using `groupby` aggregations and join patterns (inner/left/outer)

---

## Outcomes

By the end of this lab, you will be able to:

1. Use `groupby().agg(...)` to compute per‑key metrics (mean, sum, count, nunique) with **named aggregations**.
2. Choose the correct **join** (inner/left/outer) for a question, and verify cardinality with `validate=`.
3. Diagnose join pitfalls: duplicated keys (fan‑out), missing keys (anti‑join), and column suffix collisions.
4. Build tidy aggregates for downstream LLM prompts/features and persist results to Parquet.

---

## Prerequisites & Setup

- Python 3.13 with `pandas`, `numpy`, `pyarrow` installed.
- JupyterLab or VS Code with Jupyter extension.
- Reuse artifacts from Day 1/2 when available:
  - `artifacts/parquet/orders/shipcountry=*.parquet` and/or `orders_joined.parquet` (from Lab 03 / Lab 05 bonus)
  - `users_clean.parquet` (from Lab 05) – optional

**Start a notebook:** `week02_lab06.ipynb`

If you lack artifacts, synthesize mini tables now:

```python
import pandas as pd
orders = pd.DataFrame({
    'OrderID':[1,2,3,4,5,6],
    'CustomerID':['ALFKI','ANATR','ANTON','ALFKI','BERGS','CHOPS'],
    'ShipCountry':['USA','DE','USA','USA','SE','SG'],
    'Freight':[32.1, 12.0, 5.0, 50.0, 80.0, 22.0]
})
customers = pd.DataFrame({
    'CustomerID':['ALFKI','ANATR','ANTON','BONAP','BERGS'],
    'CompanyName':['Alfreds','Ana Trujillo','Antonio Moreno','Bon app','Berglunds'],
    'Country':['Germany','Mexico','Mexico','France','Sweden']
})
orders.head(), customers.head()
```

---

## Part A — GroupBy Fundamentals & Named Aggregations

### A1. Basic aggregates

```python
# Per ShipCountry: orders, mean freight, total freight
agg_country = (orders
    .groupby('ShipCountry', as_index=False)
    .agg(orders=('OrderID','count'),
         freight_mean=('Freight','mean'),
         freight_sum=('Freight','sum'))
    .sort_values('orders', ascending=False))
agg_country
```

### A2. Multiple keys & nunique

```python
# Per (ShipCountry, CustomerID): count & distinct orders
agg_cc = (orders
    .groupby(['ShipCountry','CustomerID'], as_index=False)
    .agg(n_orders=('OrderID','count'), n_cust=('CustomerID','nunique')))
agg_cc.head()
```

### A3. `size` vs `count` and missing values

```python
# size counts rows; count ignores NaN in the column
orders.assign(Maybe=None).groupby('ShipCountry').size()
orders.assign(Maybe=None).groupby('ShipCountry')['Maybe'].count()
```

**Checkpoint:** When would you choose `size` vs `count`?

---

## Part B — Join Patterns & Cardinality Checks

### B1. Inner vs Left vs Outer (visual & code)

```python
inner = orders.merge(customers, on='CustomerID', how='inner', validate='many_to_one')
left  = orders.merge(customers, on='CustomerID', how='left',  validate='many_to_one')
outer = orders.merge(customers, on='CustomerID', how='outer')
len(orders), len(inner), len(left), len(outer)
```

- **Inner:** keep only matching `CustomerID` on both sides (typical for **orders↔customers** when analyzing realized orders).
- **Left:** keep all orders even if customer row is missing (good for **data quality** checks / anti‑join).
- **Outer:** keep all keys from both sides (useful for audits, rare in production metrics).

### B2. Anti‑join & keys that didn’t match

```python
# orders without a matching customer (left rows with NaN on right)
anti = left[left['CompanyName'].isna()][['OrderID','CustomerID']]
anti
```

**Checkpoint:** What business question would left join answer that inner wouldn’t?

### B3. Guard against fan‑out (duplicated keys)

```python
# Introduce a duplicate key to illustrate
cust_dupe = pd.concat([customers, customers.iloc[[0]]], ignore_index=True)
try:
    orders.merge(cust_dupe, on='CustomerID', how='inner', validate='many_to_one')
except Exception as e:
    print('Validation caught:', e)
```

Use `validate='one_to_one' | 'one_to_many' | 'many_to_one'` to catch unexpected cardinality.

### B4. Column name collisions & suffixes

```python
# When both sides share column names (e.g., Country), set suffixes
joined = orders.merge(customers.rename(columns={'Country':'CustCountry'}), on='CustomerID', how='inner')
joined.filter(items=['CustomerID','ShipCountry','CustCountry']).head()
```

---

## Part C — End‑to‑End Mini Task: Customer Segments & Country Rollups

> Goal: Build per‑customer metrics and join to customer attributes for LLM‑ready features.

### C1. Per‑customer aggregates

```python
per_cust = (orders
    .groupby('CustomerID', as_index=False)
    .agg(n_orders=('OrderID','count'),
         freight_mean=('Freight','mean'),
         freight_sum=('Freight','sum')))
per_cust.head()
```

### C2. Join with customers (inner vs left) and create segments

```python
per_cust_inner = per_cust.merge(customers, on='CustomerID', how='inner', validate='one_to_one')
per_cust_left  = per_cust.merge(customers, on='CustomerID', how='left',  validate='one_to_one')

import numpy as np
bins = [0, 20, 50, np.inf]
labels = ['low','mid','high']
seg = pd.cut(per_cust_inner['freight_sum'], bins=bins, labels=labels, right=False)
per_cust_inner = per_cust_inner.assign(spend_segment=seg)
per_cust_inner.head()
```

### C3. Country rollup for reporting

```python
country_rollup = (per_cust_inner
    .groupby('Country', as_index=False)
    .agg(customers=('CustomerID','count'),
         orders=('n_orders','sum'),
         freight_sum=('freight_sum','sum'))
    .sort_values('orders', ascending=False))
country_rollup
```

**Checkpoint:** Which join choice (inner/left) makes more sense for this segment report and why?

---

## Part D — Bonus (Optional) — Use partitioned `orders` Parquet

If you have `artifacts/parquet/orders/shipcountry=*.parquet`, load them and repeat C1–C3 on a larger dataset.

```python
from pathlib import Path
p = Path('artifacts/parquet/orders')
if p.exists():
    import pandas as pd
    files = sorted(p.glob('shipcountry=*.parquet'))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    per_cust = (df
        .groupby('CustomerID', as_index=False)
        .agg(n_orders=('OrderID','count'),
             freight_mean=('Freight','mean'),
             freight_sum=('Freight','sum')))
    per_cust.head(3)
```

---

## Part E — Wrap‑Up

Add a markdown cell and answer:

1. Provide one example where **inner** is preferred and one where **left** is required.
2. Show a `merge(..., validate=...)` that catches a fan‑out **before** it pollutes your metrics.
3. Export your final `per_cust_inner` and `country_rollup` to Parquet for downstream labs.

```python
import pyarrow.parquet as pq, pyarrow as pa
from pathlib import Path
out = Path('artifacts/clean'); out.mkdir(parents=True, exist_ok=True)
pq.write_table(pa.Table.from_pandas(per_cust_inner, preserve_index=False), out / 'per_customer.parquet')
pq.write_table(pa.Table.from_pandas(country_rollup, preserve_index=False), out / 'country_rollup.parquet')
```

---

- **Common pitfalls:**
  - Using `count` vs `size` incorrectly; `count` skips NaN in the counted column.
  - Forgetting `validate=` and accidentally creating a **fan‑out join**.
  - Relying on **outer** joins for metrics (often inflates counts); prefer **inner** for realized facts and **left** for QA.
  - Column name collisions; use `suffixes=` or rename in advance.

---

## Solution Snippets (reference)

**Named aggregations:**

```python
orders.groupby('CustomerID', as_index=False).agg(n_orders=('OrderID','count'), freight_mean=('Freight','mean'))
```

**Cardinality checks:**

```python
orders.merge(customers, on='CustomerID', how='inner', validate='many_to_one')
```

**Anti‑join:**

```python
orders.merge(customers, on='CustomerID', how='left', indicator=True).query("_merge == 'left_only'")
```
