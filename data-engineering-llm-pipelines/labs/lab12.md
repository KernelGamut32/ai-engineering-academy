# Lab 12 — Author a Schema + Profiling Report

**Focus Areas:** Author a schema (Pandera) + Profiling report (ydata‑profiling)

> This capstone‑style lab combines a **production‑ish Pandera schema** for your cleaned & joined data with a **focused profiling report**. You’ll author constraints (types/ranges/enums), add cross‑column/DF checks, generate an HTML profile, and produce a prioritized **risk list** with mitigations—all wired for CI.

---

## Outcomes

By the end of this lab, you will be able to:

1. Author a **typed Pandera schema** with column checks (regex, enums) and cross‑column/DF‑level checks; validate **clean** vs **broken** frames with actionable messages.  
2. Implement **schema versioning** and light **evolution** (e.g., allow a new category via a controlled update).  
3. Generate a **ydata‑profiling** HTML report for a column subset at realistic scale; interpret key sections and extract **machine‑readable metrics**.  
4. Produce a **Top‑5 risks** table (with severity & mitigation) and persist artifacts (HTML + JSON + CSV) for review/CI.

---

## Prerequisites & Setup

- Python 3.13 with `pandas`, `numpy`, `pandera>=0.20`, `pydantic>=2.0` (optional), `ydata-profiling`, `pyarrow`.  
- JupyterLab or VS Code with Jupyter extension.
- Preferred artifacts from previous labs:  
  - `artifacts/clean/per_customer_enriched.parquet`  
  - `artifacts/clean/per_segment.parquet`  
  - If missing, run the synthetic fallback below.

**Start a notebook:** `week02_lab12.ipynb`

**Directory prep:**

```python
from pathlib import Path
Path('artifacts/reports').mkdir(parents=True, exist_ok=True)
Path('artifacts/metrics').mkdir(parents=True, exist_ok=True)
```

**Load or synthesize data:**

```python
import numpy as np, pandas as pd
try:
    per_cust_enriched = pd.read_parquet('artifacts/clean/per_customer_enriched.parquet')
except Exception:
    # Synthetic fallback
    rng = np.random.default_rng(3)
    N = 60_000
    per_cust_enriched = pd.DataFrame({
        'CustomerID': [f'C{i:05d}' for i in range(N)],
        'country_norm': rng.choice(['USA','DE','SG','BR'], size=N, p=[.58,.18,.16,.08]),
        'n_orders': rng.poisson(3, size=N),
        'freight_sum': np.round(np.clip(rng.lognormal(3.1, 0.8, N), 0, 2e5), 2),
        'freight_mean': np.round(np.clip(rng.lognormal(2.5, 0.6, N), 0, 1e4), 2),
        'signup_dt': pd.Timestamp('2025-01-01') + pd.to_timedelta(rng.integers(0, 40, N), unit='D'),
        'email': [f'user{i}@example.com' for i in range(N)],
        'is_adult': rng.random(N) > 0.1,
        'is_high_value': rng.random(N) > 0.9,
    })
per_cust_enriched.head(3)
```

---

## Part A — Author a Production‑ish Pandera Schema

### A1. Column types & checks (regex, enums, ranges)

```python
import pandera as pa
from pandera import Column, Check

SCHEMA_VERSION = '1.0.0'
AllowedCountries = ['USA','DE','SG','BR']

PerCustomerSchema = pa.DataFrameSchema({
    'CustomerID': Column(object, nullable=False, checks=Check.str_matches(r'^C\d{5}$', error='bad_id')),
    'country_norm': Column(object, nullable=False, checks=Check.isin(AllowedCountries)),
    'n_orders': Column(pa.Int64, nullable=False, checks=Check.ge(0)),
    'freight_sum': Column(float, nullable=False, checks=Check.ge(0)),
    'freight_mean': Column(float, nullable=False, checks=Check.ge(0)),
    'signup_dt': Column(object, nullable=False),
    'email': Column(object, nullable=False, checks=Check.str_matches(r'^.+@.+\..+$')),
    'is_adult': Column(bool, nullable=False),
    'is_high_value': Column(bool, nullable=False),
},
    name=f'PerCustomerSchema_v{SCHEMA_VERSION}', strict=True
)
```

### A2. Cross‑column rules & DF‑level checks

```python
PerCustomerSchema = PerCustomerSchema.add_checks([
    # DF-level: mean of freight_mean should be <= mean of freight_sum (approx; demo check)
    pa.Check(lambda df: df['freight_mean'].mean() <= max(df['freight_sum'].mean(), 1.0), element_wise=False,
             error='freight_mean too large vs freight_sum'),
    # DF-level: IDs must be unique
    pa.Check(lambda df: df['CustomerID'].is_unique, element_wise=False, error='duplicate CustomerID')
])

# Row-level cross-field: freight_sum >= freight_mean when n_orders >= 1
PerCustomerSchema = PerCustomerSchema.update_checks({
    'freight_sum': [Check(lambda s, df: s >= df['freight_mean'], element_wise=True,
                          error='sum must be >= mean')]
})
```

### A3. Validate clean → then exercise failures with a “broken” sample

```python
# Clean should pass
ok = PerCustomerSchema.validate(per_cust_enriched, lazy=True)
print('rows:', len(ok))

# Make a small broken copy to see errors
broken = per_cust_enriched.copy().iloc[:500].copy()
broken.loc[0, 'CustomerID'] = 'BADID'
broken.loc[1, 'country_norm'] = 'U.S.A.'
broken.loc[2, 'freight_sum'] = -10
broken.loc[3, 'email'] = 'nope'

try:
    PerCustomerSchema.validate(broken, lazy=True)
except pa.errors.SchemaErrors as err:
    fc = err.failure_cases
    rollup = (fc.groupby(['column','check']).size().reset_index(name='n').sort_values('n', ascending=False))
    display(rollup.head(10))
    fc.head()
```

**Checkpoint:** Paste two violations from `rollup` and explain how each protects downstream LLM steps.

### A4. Schema evolution: allow a new country (controlled)

```python
# Simulate a business-accepted new category
AllowedCountries_v2 = AllowedCountries + ['SE']
PerCustomerSchema_v2 = PerCustomerSchema.update_column_checks('country_norm', checks=Check.isin(AllowedCountries_v2))
SCHEMA_VERSION = '1.1.0'
PerCustomerSchema_v2.name = f'PerCustomerSchema_v{SCHEMA_VERSION}'
```

**Note:** Commit schema files with a version tag; add a CHANGELOG entry for policy changes.

---

## Part B — Focused Profiling Report & Metric Extraction

### B1. Create a tuned profile (subset + minimal heavy bits)

```python
from ydata_profiling import ProfileReport
cols = ['country_norm','n_orders','freight_sum','freight_mean','is_high_value']
subset = per_cust_enriched[cols].sample(40_000, random_state=7) if len(per_cust_enriched) > 40_000 else per_cust_enriched[cols]

profile = ProfileReport(
    subset,
    title='Per-Customer Enriched — Focused Profile',
    minimal=False, explorative=True,
    correlations={'pearson': {'calculate': True}, 'spearman': {'calculate': True}},
    progress_bar=True
)
profile_path = 'artifacts/reports/per_customer_profile.html'
profile.to_file(profile_path)
profile_path
```

### B2. Interpret: variables, alerts, correlations

- **Variables:** note **skew/outliers** in `freight_sum`; check **distinct counts** for `country_norm`.  
- **Alerts:** capture high cardinality or extreme zeros distribution.  
- **Correlations:** expect positive `n_orders` ↔ `freight_sum`; sanity‑check magnitude.

**Checkpoint:** Record one expected correlation and one surprising alert.

### B3. Extract metrics JSON for drift tracking

```python
summary = profile.to_dict()
metrics = {
    'n_rows': summary['table']['n'],
    'freight_sum_mean': summary['variables']['freight_sum']['mean'],
    'freight_sum_std': summary['variables']['freight_sum']['std'],
    'n_orders_mean': summary['variables']['n_orders']['mean'],
    'n_orders_distinct': summary['variables']['n_orders']['distinct_count'],
    'country_cardinality': summary['variables']['country_norm']['distinct_count'],
}
import json
with open('artifacts/metrics/per_customer_profile_metrics.json','w') as f:
    json.dump(metrics, f, indent=2)
metrics
```

---

## Part C — Top‑5 Data Risks & Mitigations

### C1. Pull alerts table from profile dict

```python
alerts = summary.get('alerts', [])  # ydata-profiling structures alerts differently by version
# Build a generic “risks” list from variable summaries
risks = []
for col in ['freight_sum','freight_mean','n_orders','country_norm']:
    v = summary['variables'][col]
    if v.get('p_zeros', 0) > 0.5:
        risks.append((col, 'high zeros fraction', v['p_zeros']))
    if v.get('skewness', 0) > 2:
        risks.append((col, 'high skewness', v['skewness']))
    if v.get('distinct_count', 0) > 0.8 * metrics['n_rows']:
        risks.append((col, 'high cardinality', v['distinct_count']))

# Prioritize and pick top 5
risks_sorted = sorted(risks, key=lambda x: float(x[2]), reverse=True)[:5]
risks_sorted
```

### C2. Document mitigations (template)

```python
import pandas as pd
mitigations = pd.DataFrame([
    {'column': c, 'risk': r, 'value': float(val),
     'mitigation': 'Log-transform; winsorize 99th pct; monitor via metric drift'},
    for c, r, val in risks_sorted
])
mitigations
mitigations.to_csv('artifacts/reports/top5_risks.csv', index=False)
'Wrote artifacts/reports/top5_risks.csv'
```

**Examples of mitigations:**

- **High skew/outliers:** log‑transform features; cap at high quantiles; monitor tails.  
- **High cardinality IDs:** avoid as categorical features; use as join keys only.  
- **Zeros inflation:** split populations (zero vs non‑zero) or engineer indicator features.  
- **Category drift:** expand schema allow‑list **via versioned change** + DQ alert.

---

## Part D — Integrate: CI Checks & Artifacts

### D1. Gate with schema + persist artifacts

```python
try:
    _ = PerCustomerSchema.validate(per_cust_enriched, lazy=True)
    status = 'OK'
except pa.errors.SchemaErrors as err:
    status = 'FAIL'
    err.failure_cases.to_csv('artifacts/reports/schema_failures.csv', index=False)
status
```

### D2. Minimal CI rule (concept)

- Always attach `per_customer_profile.html`, `per_customer_profile_metrics.json`, and `top5_risks.csv` to PRs.  
- Fail PR if schema **FAIL** or if key metrics change > **30%** from baseline without a waiver.

---

## Part E — Wrap‑Up

Add a markdown cell and answer:

1. Paste a **schema version** and one change you’d record in a CHANGELOG.  
2. List your **Top‑5 risks** and the mitigation you selected for each.  
3. Show the two metrics you’ll track in CI and the thresholds you chose.

---
  
- **Common pitfalls:** over‑strict schemas (block expected evolution), not sampling profiles, forgetting `strict=True` or uniqueness checks.

---

## Solution Snippets (reference)

**Update a single column’s checks (schema evolution):**

```python
PerCustomerSchema_v2 = PerCustomerSchema.update_column_checks('country_norm', checks=Check.isin(['USA','DE','SG','BR','SE']))
```

**Compact roll‑up of schema failures:**

```python
try:
    PerCustomerSchema.validate(df, lazy=True)
except pa.errors.SchemaErrors as err:
    roll = (err.failure_cases.groupby(['column','check']).size().reset_index(name='n')
              .sort_values('n', ascending=False))
    print(roll.head(10))
```

**Pick top risks with a simple heuristic:**

```python
summary = profile.to_dict()
risks = [(c, 'skew', summary['variables'][c]['skewness']) for c in ['freight_sum','freight_mean'] if summary['variables'][c]['skewness']>2]
```
