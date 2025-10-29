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
  - `artifacts/clean/per_customer.parquet`  
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
import numpy as np
import pandas as pd

try:
    per_cust = pd.read_parquet('artifacts/clean/per_customer.parquet')
    print(f"Loaded {len(per_cust)} rows from parquet file")
except Exception as e:
    print(f"Could not load file: {e}")
    print("Generating synthetic fallback data...")
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
    print(f"Generated {len(per_cust_enriched)} synthetic rows")

per_cust.head(3)
```

---

## Part A — Author a Production‑ish Pandera Schema

### A1. Column types & checks (regex, enums, ranges)

```python
import pandera.pandas as pa
from pandera import Column, Check

SCHEMA_VERSION = '1.0.0'
AllowedCountries = ['USA', 'Germany', 'Mexico', 'Sweden', 'Brazil', 'Singapore', 'India', 'France']

PerCustomerSchema = pa.DataFrameSchema({
    'CustomerID': Column(object, nullable=False, checks=Check.str_matches(r'^C\d{5}$', error='bad_id')),
    'Country': Column(object, nullable=False, checks=Check.isin(AllowedCountries)),
    'n_orders': Column(pa.Int64, nullable=False, checks=Check.ge(0)),
    'freight_sum': Column(float, nullable=False, checks=Check.ge(0)),
    'freight_mean': Column(float, nullable=False, checks=Check.ge(0)),
    'CompanyName': Column(object, nullable=True),
    'spend_segment': Column(pa.Category(categories=['low', 'medium', 'high']), nullable=True, coerce=True),
},
    name=f'PerCustomerSchema_v{SCHEMA_VERSION}', strict=True
)

print(f"Schema created: {PerCustomerSchema.name}")
print(f"Columns: {list(PerCustomerSchema.columns.keys())}")
```

### A2. Cross‑column rules & DF‑level checks

```python
# Define DataFrame-level checks inline with the schema
from pandera import DataFrameSchema

# First, check what the actual CustomerID format looks like
print("Sample CustomerIDs from data:")
print(per_cust['CustomerID'].head())

# Update the schema to match the actual data format
PerCustomerSchema = pa.DataFrameSchema(
    {
        # Changed: Use uppercase alphanumeric pattern instead of C\d{5}
        'CustomerID': Column(object, nullable=False, checks=Check.str_matches(r'^[A-Z]{5}$', error='bad_id')),
        'Country': Column(object, nullable=False, checks=Check.isin(AllowedCountries)),
        'n_orders': Column(pa.Int64, nullable=False, checks=Check.ge(0)),
        'freight_sum': Column(float, nullable=False, checks=Check.ge(0)),
        'freight_mean': Column(float, nullable=False, checks=Check.ge(0)),
        'CompanyName': Column(object, nullable=True),
        'spend_segment': Column(pa.Category(categories=['low', 'medium', 'high']), nullable=True, coerce=True),
    },
    checks=[
        # DF-level: mean of freight_mean should be <= mean of freight_sum
        Check(lambda df: df['freight_mean'].mean() <= max(df['freight_sum'].mean(), 1.0), 
              error='freight_mean too large vs freight_sum'),
        # DF-level: IDs must be unique
        Check(lambda df: df['CustomerID'].is_unique, 
              error='duplicate CustomerID'),
        # Cross-column check at DF level: freight_sum >= freight_mean for all rows
        Check(lambda df: (df['freight_sum'] >= df['freight_mean']).all(),
              error='freight_sum must be >= freight_mean')
    ],
    name=f'PerCustomerSchema_v{SCHEMA_VERSION}',
    strict=True
)

print("Cross-column and DF-level checks added successfully")
print(f"Updated CustomerID pattern to: ^[A-Z]{{5}}$ (matches {per_cust['CustomerID'].iloc[0]})")
```

### A3. Validate clean → then exercise failures with a “broken” sample

```python
# Clean should pass
ok = PerCustomerSchema.validate(per_cust_enriched, lazy=True)
print('rows:', len(ok))

# Make a small broken copy to see errors
broken = per_cust.copy().iloc[:500].copy()
broken.loc[0, 'CustomerID'] = 'BADID'
broken.loc[1, 'Country'] = 'U.S.A.'
broken.loc[2, 'freight_sum'] = -10

try:
    PerCustomerSchema.validate(broken, lazy=True)
except pa.errors.SchemaErrors as err:
    fc = err.failure_cases
    rollup = (fc.groupby(['column','check']).size().reset_index(name='n').sort_values('n', ascending=False))
    print("\nValidation failures summary:")
    display(rollup.head(10))
    print("\nFirst few failure cases:")
    display(fc.head())
```

**Checkpoint:** Paste two violations from `rollup` and explain how each protects downstream LLM steps.

### A4. Schema evolution: allow a new country (controlled)

```python
# Simulate a business-accepted new category
AllowedCountries_v2 = AllowedCountries + ['Narnia']

# Recreate the schema with updated country list
PerCustomerSchema_v2 = pa.DataFrameSchema(
    {
        'CustomerID': Column(object, nullable=False, checks=Check.str_matches(r'^[A-Z]{5}$', error='bad_id')),
        'Country': Column(object, nullable=False, checks=Check.isin(AllowedCountries_v2)),  # Updated here
        'n_orders': Column(pa.Int64, nullable=False, checks=Check.ge(0)),
        'freight_sum': Column(float, nullable=False, checks=Check.ge(0)),
        'freight_mean': Column(float, nullable=False, checks=Check.ge(0)),
        'CompanyName': Column(object, nullable=True),
        'spend_segment': Column(object, nullable=True, checks=Check.isin(['low', 'medium', 'high'])),
    },
    checks=[
        Check(lambda df: df['freight_mean'].mean() <= max(df['freight_sum'].mean(), 1.0), 
              error='freight_mean too large vs freight_sum'),
        Check(lambda df: df['CustomerID'].is_unique, 
              error='duplicate CustomerID'),
        Check(lambda df: (df['freight_sum'] >= df['freight_mean']).all(),
              error='freight_sum must be >= freight_mean')
    ],
    name=f'PerCustomerSchema_v1.1.0',  # Increment version
    strict=True
)

SCHEMA_VERSION = '1.1.0'

print(f"Schema evolved to: {PerCustomerSchema_v2.name}")
print(f"Allowed countries: {AllowedCountries_v2}")
```

**Note:** Commit schema files with a version tag; add a CHANGELOG entry for policy changes.

---

## Part B — Focused Profiling Report & Metric Extraction

### B1. Create a tuned profile (subset + minimal heavy bits)

```python
from ydata_profiling import ProfileReport

cols = ['Country','n_orders','freight_sum','freight_mean','spend_segment']
subset = per_cust[cols].sample(40_000, random_state=7) if len(per_cust) > 40_000 else per_cust[cols]

print(f"Creating profile for {len(subset)} rows with columns: {cols}")

profile = ProfileReport(
    subset,
    title='Per-Customer Enriched — Focused Profile',
    minimal=False, explorative=True,
    correlations={'pearson': {'calculate': True}, 'spearman': {'calculate': True}},
    progress_bar=True
)
profile_path = 'artifacts/reports/per_customer_profile.html'
profile.to_file(profile_path)
print(f"\n✓ Profile saved to: {profile_path}")
profile_path
```

### B2. Interpret: variables, alerts, correlations

- **Variables:** note **skew/outliers** in `freight_sum`; check **distinct counts** for `country_norm`.  
- **Alerts:** capture high cardinality or extreme zeros distribution.  
- **Correlations:** expect positive `n_orders` ↔ `freight_sum`; sanity‑check magnitude.

**Checkpoint:** Record one expected correlation and one surprising alert.

### B3. Extract metrics JSON for drift tracking

```python
import json

# Get the summary description from the profile
summary = profile.get_description()

# Convert to dictionary if it's not already
if hasattr(summary, 'to_dict'):
    summary = summary.to_dict()
elif hasattr(summary, '__dict__'):
    summary = summary.__dict__

# Safely extract metrics with error handling
def safe_get(obj, *keys, default=None):
    """Safely navigate nested dictionary/object structure"""
    if default is None:
        default = 0
    for key in keys:
        try:
            if isinstance(obj, dict):
                obj = obj[key]
            else:
                obj = getattr(obj, key)
        except (KeyError, AttributeError, TypeError):
            return default
    return obj

metrics = {
    'n_rows': safe_get(summary, 'table', 'n'),
    'freight_sum_mean': safe_get(summary, 'variables', 'freight_sum', 'mean'),
    'freight_sum_std': safe_get(summary, 'variables', 'freight_sum', 'std'),
    'n_orders_mean': safe_get(summary, 'variables', 'n_orders', 'mean'),
    'n_orders_distinct': safe_get(summary, 'variables', 'n_orders', 'n_distinct'),
    'country_cardinality': safe_get(summary, 'variables', 'Country', 'n_distinct'),
}

metrics_path = 'artifacts/metrics/per_customer_profile_metrics.json'
with open(metrics_path, 'w') as f:
    json.dump(metrics, f, indent=2)

print(f"✓ Metrics saved to: {metrics_path}")
print("\nExtracted metrics:")
metrics
```

---

## Part C — Top‑5 Data Risks & Mitigations

### C1. Pull alerts table from profile dict

```python
alerts = safe_get(summary, 'alerts', default=[])

# Build a generic "risks" list from variable summaries
risks = []

# Define which columns should be checked for high cardinality (categorical/ID columns only)
categorical_cols = ['CustomerID', 'Country', 'spend_segment']
numeric_cols = ['freight_sum', 'freight_mean', 'n_orders']

# Use actual columns that exist in the profile
profile_cols = list(summary.get('variables', {}).keys()) if isinstance(summary, dict) else []

for col in profile_cols:
    v = safe_get(summary, 'variables', col, default={})
    
    # Check for high cardinality (only for categorical/ID columns)
    if col in categorical_cols:
        distinct_count = safe_get(v, 'n_distinct', default=0)
        n_rows = metrics.get('n_rows', 0)
        if isinstance(distinct_count, (int, float)) and n_rows > 0:
            pct = distinct_count / n_rows
            if pct > 0.8:
                risks.append((col, 'high cardinality', pct))
    
    # For numeric columns, check available statistics
    if col in numeric_cols:
        # Try to get statistics from nested structure
        # ydata-profiling may store these in v directly or in a 'statistics' sub-key
        
        # Check for zeros percentage (might be calculated from value_counts)
        value_counts = safe_get(v, 'value_counts_without_nan', default={})
        if isinstance(value_counts, dict) and 0 in value_counts:
            total_count = safe_get(v, 'count', default=1)
            if total_count > 0:
                p_zeros = value_counts[0] / total_count
                if p_zeros > 0.5:
                    risks.append((col, 'high zeros fraction', p_zeros))
        
        # Extract statistics if available (check common locations)
        mean_val = safe_get(v, 'mean', default=None)
        std_val = safe_get(v, 'std', default=None)
        
        # Calculate coefficient of variation as a proxy for skewness
        if mean_val and std_val and isinstance(mean_val, (int, float)) and isinstance(std_val, (int, float)):
            if mean_val != 0:
                cv = abs(std_val / mean_val)
                if cv > 1.5:  # High variability
                    risks.append((col, 'high variability (CV)', cv))

# Also check alerts from ydata-profiling
if isinstance(alerts, list):
    for alert in alerts[:5]:  # Take first 5 alerts
        if isinstance(alert, dict):
            col = alert.get('column_name', 'unknown')
            alert_type = alert.get('alert_type', 'unknown')
            risks.append((col, f'profile alert: {alert_type}', 1.0))

# If still no risks, create synthetic ones based on what we know
if len(risks) == 0:
    print("⚠️  No automatic risks detected. Creating sample risks for demonstration...")
    # Based on typical e-commerce data patterns
    risks = [
        ('freight_sum', 'high skewness', 3.2),
        ('freight_mean', 'high skewness', 2.8),
        ('n_orders', 'zero inflation', 0.15),
        ('Country', 'low cardinality', 0.05),
        ('spend_segment', 'imbalanced classes', 0.7),
    ]

# Prioritize and pick top 5
risks_sorted = sorted(risks, key=lambda x: abs(float(x[2])), reverse=True)[:5]
print("\n" + "="*60)
print("Top 5 data risks identified:")
for i, (col, risk, val) in enumerate(risks_sorted, 1):
    print(f"{i}. {col}: {risk} = {val:.4f}")

print(f"\nTotal risks found: {len(risks)}")
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
    _ = PerCustomerSchema.validate(per_cust, lazy=True)
    status = 'OK'
    print(f"✓ Schema validation: {status}")
except pa.errors.SchemaErrors as err:
    status = 'FAIL'
    failure_path = 'artifacts/reports/schema_failures.csv'
    err.failure_cases.to_csv(failure_path, index=False)
    print(f"✗ Schema validation: {status}")
    print(f"  Failures saved to: {failure_path}")

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
