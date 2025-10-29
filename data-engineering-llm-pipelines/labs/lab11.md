# Lab 11 — Data Quality Dimensions & Thresholded Alerts

**Focus Area:** Quality dimensions — **Completeness, Validity, Consistency, Timeliness** with thresholds & alerts

> In this lab you’ll operationalize data‑quality checks across four core dimensions and wire lightweight **alerts** that block or warn before downstream LLM processing. You’ll reuse artifacts from earlier labs and produce a small, reusable **DQ report**.

---

## Outcomes

By the end of this lab, you will be able to:

1. Define and compute metrics for **completeness**, **validity**, **consistency**, and **timeliness**.
2. Set **thresholds** (warn vs fail) and emit a compact machine‑readable **DQ report**.
3. Detect **consistency** issues (e.g., country naming) using a reference dimension.
4. Integrate DQ checks with earlier **Pandera** schema validation and **profiling** outputs.

---

## Prerequisites & Setup

- Python 3.13 with `pandas`, `numpy`, `pandera` (optional), `pyarrow`  
- Artifacts (preferred): from previous lab — `users_clean.parquet`, `per_customer.parquet`
- JupyterLab or VS Code with Jupyter extension.

**Start a notebook:** `week02_lab11.ipynb`

### Loading Artifacts (Preferred Method)

If you have completed earlier labs, load the existing artifacts:

```python
import pandas as pd
from pathlib import Path

# Load users data from clean artifacts
users_path = Path('artifacts/clean/users_clean.parquet')
if users_path.exists():
    users2 = pd.read_parquet(users_path)
    print(f"✓ Loaded {len(users2)} users from {users_path}")
else:
    print(f"✗ {users_path} not found. Use synthetic data fallback below.")

# Load per-customer aggregated data
per_customer_path = Path('artifacts/clean/per_customer.parquet')
if per_customer_path.exists():
    per_customer = pd.read_parquet(per_customer_path)
    print(f"✓ Loaded {len(per_customer)} customer records from {per_customer_path}")
else:
    print(f"✗ {per_customer_path} not found. Use synthetic data fallback below.")

# Examine the structure
print("\nUsers columns:", users2.columns.tolist() if 'users2' in locals() else 'N/A')
print("Per-customer columns:", per_customer.columns.tolist() if 'per_customer' in locals() else 'N/A')

# country reference (same idea as 2E)
country_dim = pd.DataFrame({
    'raw': ['USA','U.S.A.','United States','US','usa','U. S. A.','BR','Brasil','DE','Germany','SG','Singapore','N/A'],
    'canonical': ['USA','USA','USA','USA','USA','USA','BR','BR','DE','DE','SG','SG','UNKNOWN']
})
Path('artifacts/reports').mkdir(parents=True, exist_ok=True)
```

### Synthetic Data Fallback

If artifacts are missing, synthesize data:

```python
import numpy as np, pandas as pd
from pathlib import Path
rng = np.random.default_rng(11)
N = 5000
users2 = pd.DataFrame({
    'CustomerID': [f'C{i:05d}' for i in range(N)],
    'email': [f'user{i}@example.com' if rng.random()>.01 else None for i in range(N)],
    'age': rng.integers(16, 80, size=N).astype('Int64'),
    'signup_dt': pd.to_datetime('2025-01-01') + pd.to_timedelta(rng.integers(0, 40, size=N), unit='D'),
    'country': rng.choice(['US','usa','United States','DE','SG','BR','N/A'], size=N, p=[.35,.05,.08,.2,.2,.1,.02]),
    'ltv_usd': np.round(np.clip(rng.lognormal(3.1, .7, size=N), 0, 1e5), 2),
})
orders = pd.DataFrame({
    'OrderID': np.arange(10_000, 10_000+N*2),
    'CustomerID': rng.choice(users2['CustomerID'], size=N*2),
    'OrderDate': pd.to_datetime('2025-02-01') + pd.to_timedelta(rng.integers(0, 10, size=N*2), unit='D'),
    'Freight': np.round(np.clip(rng.lognormal(3.0, 0.7, size=N*2), 0, 2e4), 2)
})

# country reference (same idea as 2E)
country_dim = pd.DataFrame({
    'raw': ['USA','U.S.A.','United States','US','usa','U. S. A.','BR','Brasil','DE','Germany','SG','Singapore','N/A'],
    'canonical': ['USA','USA','USA','USA','USA','USA','BR','BR','DE','DE','SG','SG','UNKNOWN']
})
Path('artifacts/reports').mkdir(parents=True, exist_ok=True)
```

---

## Part A — Define Quality Dimensions & Base Metrics

**Note:** If using real artifacts, adjust column names as needed. The examples below assume the synthetic dataset structure, but you can adapt them to match your `users_clean.parquet` schema (e.g., `Email` vs `email`, `Age` vs `age`, etc.).

### A1. Completeness (null rate on required columns)

```python
print(users2.columns.tolist())
```

```python
# Adjust column names to match your actual dataset
# For synthetic data: ['CustomerID','email','signup_dt']
# For real artifacts, check: users2.columns.tolist()
required_cols = ['user_id','email','signup_date', 'spend_usd']  # Update as needed
null_rates = users2[required_cols].isna().mean().to_dict()
null_rates
```

### A2. Validity (type/range rules)

```python
valid_age = users2['age'].between(0, 120) | users2['age'].isna()
valid_ltv = users2['spend_usd'].ge(0) | users2['spend_usd'].isna()
validity = {
    'age_in_range_rate': float(valid_age.mean()),
    'ltv_nonnegative_rate': float(valid_ltv.mean())
}
validity
```

### A3. Consistency (country naming via reference)

```python
# Normalize, then left join to reference mapping
norm = (users2['country'].astype('string')
         .str.replace('.','', regex=False)
         .str.replace(' ','', regex=False)
         .str.upper())
ref = country_dim.assign(raw_key = country_dim['raw'].str.replace('.','', regex=False).str.replace(' ','', regex=False).str.upper())
map_df = pd.DataFrame({'country_key': norm})
map_df = map_df.merge(ref[['raw_key','canonical']], left_on='country_key', right_on='raw_key', how='left')
consistency_rate = float(map_df['canonical'].notna().mean())
consistency_rate
```

### A4. Timeliness (freshness lag)

```python
import pandas as pd
now = pd.Timestamp('2025-02-15')  # fixed for reproducibility; replace with pd.Timestamp.utcnow()

# Convert signup_date to datetime if it's stored as string
users2['signup_date'] = pd.to_datetime(users2['signup_date'], format='mixed')

lag_days = (now - users2['signup_date']).dt.days
fresh_rate = float((lag_days <= 30).mean())  # % rows updated/arrived within SLA window
fresh_stats = {'lag_p50': int(lag_days.median()), 'lag_p95': int(lag_days.quantile(0.95))}
{'fresh_rate': fresh_rate, **fresh_stats}
```

**Checkpoint:** In your words, distinguish validity vs consistency for `country`.

---

## Part B — Thresholds: Warn vs Fail & Compact Alert Object (≈20 min)

### B1. Define thresholds

```python
thresholds = {
    'completeness': { 'email_null_rate_max': 0.02, 'signup_dt_null_rate_max': 0.00 },
    'validity':     { 'age_in_range_min': 0.995,  'ltv_nonnegative_min': 1.00 },
    'consistency':  { 'country_mapped_min': 0.98 },
    'timeliness':   { 'fresh_rate_min': 0.90, 'lag_p95_max': 40 }
}
```

### B2. Evaluate metrics against thresholds

```python
def evaluate_dq(null_rates, validity, consistency_rate, fresh_rate, fresh_stats, thresholds):
    alerts = []
    def add(level, dim, metric, value, target, msg):
        alerts.append({'level': level, 'dimension': dim, 'metric': metric, 'value': float(value), 'target': float(target), 'message': msg})

    # Completeness
    if null_rates['email'] > thresholds['completeness']['email_null_rate_max']:
        add('FAIL','completeness','email_null_rate', null_rates['email'], thresholds['completeness']['email_null_rate_max'], 'Email null rate too high')
    elif null_rates['email'] > thresholds['completeness']['email_null_rate_max'] * 0.8:
        add('WARN','completeness','email_null_rate', null_rates['email'], thresholds['completeness']['email_null_rate_max'], 'Email null rate nearing limit')

    # Validity
    if validity['age_in_range_rate'] < thresholds['validity']['age_in_range_min']:
        add('FAIL','validity','age_in_range_rate', validity['age_in_range_rate'], thresholds['validity']['age_in_range_min'], 'Age out of range')
    if validity['ltv_nonnegative_rate'] < thresholds['validity']['ltv_nonnegative_min']:
        add('FAIL','validity','ltv_nonnegative_rate', validity['ltv_nonnegative_rate'], thresholds['validity']['ltv_nonnegative_min'], 'Negative LTV detected')

    # Consistency
    if consistency_rate < thresholds['consistency']['country_mapped_min']:
        add('WARN','consistency','country_mapped_rate', consistency_rate, thresholds['consistency']['country_mapped_min'], 'New/unmapped country variants observed')

    # Timeliness
    if fresh_rate < thresholds['timeliness']['fresh_rate_min']:
        add('WARN','timeliness','fresh_rate', fresh_rate, thresholds['timeliness']['fresh_rate_min'], 'Records stale beyond SLA')
    if fresh_stats['lag_p95'] > thresholds['timeliness']['lag_p95_max']:
        add('WARN','timeliness','lag_p95', fresh_stats['lag_p95'], thresholds['timeliness']['lag_p95_max'], 'Tail latency too high')

    return alerts

alerts = evaluate_dq(null_rates, validity, consistency_rate, fresh_rate, fresh_stats, thresholds)
alerts[:5]
```

### B3. Persist a machine‑readable DQ report

```python
import json
from pathlib import Path
Path('artifacts/reports').mkdir(parents=True, exist_ok=True)
report = {
    'timestamp': pd.Timestamp.utcnow().isoformat(),
    'metrics': {
        'completeness': null_rates,
        'validity': validity,
        'consistency': {'country_mapped_rate': consistency_rate},
        'timeliness': {'fresh_rate': fresh_rate, **fresh_stats}
    },
    'thresholds': thresholds,
    'alerts': alerts
}
with open('artifacts/reports/dq_report.json','w') as f:
    json.dump(report, f, indent=2)
'Wrote artifacts/reports/dq_report.json'
```

**Checkpoint:** Which alerts would be **FAIL** (block) vs **WARN** (notify) in your org? Justify.

---

## Part C — Hook into Validation & Profiling

### C1. Combine with Pandera (optional gate)

```python
import pandera.pandas as pa
from pandera import Column, Check
UsersSchema = pa.DataFrameSchema({
    'user_id': Column(pa.Int64, nullable=False),
    'email': Column(object, nullable=False, checks=Check.str_matches(r'^.+@.+\..+$')),
    'age': Column(pa.Float64, nullable=False, checks=Check.in_range(0,120)),
    'signup_date': Column(pa.DateTime, nullable=False),
    'spend_usd': Column(pa.Float64, nullable=False, checks=Check.ge(0))
})
try:
    _ = UsersSchema.validate(users2.dropna(subset=['user_id','email','signup_date']), lazy=True)
except pa.errors.SchemaErrors as err:
    print('Schema gate failed; see failure cases below:')
    display(err.failure_cases.head())
```

### C2. Single boolean to drive CI

```python
#print(alerts)
fail = any(a['level']=='FAIL' for a in alerts)
warn = any(a['level']=='WARN' for a in alerts)
print('DQ status =>', 'FAIL' if fail else 'WARN' if warn else 'OK')
# In CI: sys.exit(1) if fail
```

---

## Part D — Wrap‑Up

Add a markdown cell and answer:

1. Give one concrete metric per dimension that you computed and the threshold you chose.
2. Which alerts would block the pipeline vs notify only? Why?
3. Where in your pipeline would you store and review the DQ report?

Export the DQ JSON and (optionally) a short CSV of `alerts`.

---

- **Common pitfalls:** Too strict thresholds causing constant red; ambiguous units (days vs hours) for timeliness; mixing validity (range) with consistency (naming).

---

## Solution Snippets (reference)

**Null‑rate dict for any set of columns:**

```python
lambda df, cols: df[cols].isna().mean().to_dict()
```

**Country mapping coverage:**

```python
coverage = (map_df['canonical'].notna().mean())
```

**SLA freshness check for arbitrary datetime col:**

```python
lambda s, now, days: float(((now - s).dt.days <= days).mean())
```

**CI fail/warn toggle:**

```python
fail = any(a['level']=='FAIL' for a in alerts)
warn = any(a['level']=='WARN' for a in alerts)
```
