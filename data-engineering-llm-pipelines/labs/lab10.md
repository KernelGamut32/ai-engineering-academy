# Lab 10 — Data Profiling at Scale with ydata‑profiling

**Focus Area:** Data profiling — summary stats, cardinality, distributions, outlier flags, and integrating reports into review

> This lab shows how to generate actionable **profiling reports** for medium–large datasets using **ydata‑profiling**, interpret the outputs (summary stats, high‑cardinality, distributions, correlations, outliers), and integrate those artifacts into your review/CI workflow alongside Pandera/Pydantic gates.

---

## Outcomes

By the end of this lab, you will be able to:

1. Produce a **ProfileReport** (full and minimal) and export it to HTML for team review.
2. Interpret key sections: **overview**, **variables**, **interactions**, **correlations**, **missingness**, **alerts** (outliers, skew, high cardinality).
3. Extract **machine‑readable metrics** from the report to track drift over time.
4. Profile **at scale** using sampling, column subsets, and configuration tuning.

---

## Prerequisites & Setup

- Python 3.13 with `pandas`, `numpy`, `ydata-profiling`, `pyarrow`.
- JupyterLab or VS Code with Jupyter extension.
- Artifacts (preferred): `artifacts/clean/per_customer_enriched.parquet` and `per_segment.parquet`.

**Start a notebook:** `week02_lab10.ipynb`

If you don’t have artifacts, synthesize a dataset:

```python
import numpy as np, pandas as pd
rng = np.random.default_rng(0)
N = 50_000
per_cust_enriched = pd.DataFrame({
    'CustomerID': [f'C{i:05d}' for i in range(N)],
    'country_norm': rng.choice(['USA','DE','SG','BR'], size=N, p=[.58,.18,.16,.08]),
    'n_orders': rng.poisson(3, size=N),
    'freight_sum': np.round(np.clip(rng.lognormal(3.0, 0.8, size=N), 0, 2e5), 2),
    'freight_mean': np.round(np.clip(rng.lognormal(2.5, 0.6, size=N), 0, 1e4), 2),
    'is_adult': rng.random(size=N) > 0.1,
    'is_high_value': rng.random(size=N) > 0.9,
})
per_cust_enriched.head()
```

---

## Part A — Generate a Minimal Profile

### A1. Basic report (minimal config)

```python
from ydata_profiling import ProfileReport

sample = per_cust_enriched.sample(15_000, random_state=42) if len(per_cust_enriched) > 15_000 else per_cust_enriched
profile_min = ProfileReport(
    sample,
    title="Per-Customer Enriched — Minimal Profile",
    minimal=True,  # disables heavy calculations (e.g., interactions)
    explorative=True,
    progress_bar=True
)
profile_min.to_file("artifacts/reports/per_customer_minimal.html")
"artifacts/reports/per_customer_minimal.html"
```

### A2. Read the overview

- **Warnings/Alerts:** high cardinality (e.g., `CustomerID`), skewed distributions (`freight_sum`), zeros inflation.
- **Missingness:** ensure expected null rates (should be near 0 post‑cleaning).

**Checkpoint:** List 3 alerts the report shows and classify them: quality issue vs expected property.

---

## Part B — Focused Full Profile (column subset + tuned)

### B1. Choose columns and tune config

```python
cols = ["country_norm","n_orders","freight_sum","freight_mean","is_high_value"]
subset = per_cust_enriched[cols].copy()

profile_cfg = {
    "title": "Per-Customer Enriched — Focused Profile",
    "dataset": {"description": "Subset profile for review & CI"},
    "variables": {"descriptions": {
        "freight_sum": "Total freight per customer (currency units)",
        "freight_mean": "Average freight per order",
        "n_orders": "Order count per customer"
    }},
    "correlations": {"pearson": {"calculate": True}, "spearman": {"calculate": True}},
    "missing_diagrams": {"heatmap": True, "dendrogram": False},
}

profile_full = ProfileReport(
    subset,
    title=profile_cfg["title"],
    explorative=True,
    minimal=False,
    correlations=profile_cfg["correlations"],
    progress_bar=True
)
profile_full.to_file("artifacts/reports/per_customer_focused.html")
"artifacts/reports/per_customer_focused.html"
```

### B2. Interpret variables & correlations

- **Variables tab:** check **distributions**, **zeros**, **distinct counts** (cardinality), **outlier flags** for `freight_sum`.
- **Correlations:** look for strong positive or negative relationships (e.g., `n_orders` vs `freight_sum`), and verify they are **business‑plausible**.

**Checkpoint:** Name one correlation you’d expect and whether the profile confirms it.

---

## Part C — Extract Metrics Programmatically

### C1. Get summary dict

```python
summary = profile_full.to_dict()
# Example: pull high-level stats
n_rows = summary['table']['n']
var_summaries = {k: v for k, v in summary['variables'].items() if k in cols}

n_rows, list(var_summaries)[:3]
```

### C2. Build a compact drift tracker

```python
import json
from pathlib import Path

metrics = {
    "n_rows": n_rows,
    "freight_sum_mean": var_summaries['freight_sum']['mean'],
    "freight_sum_std": var_summaries['freight_sum']['std'],
    "n_orders_mean": var_summaries['n_orders']['mean'],
    "n_orders_distinct": var_summaries['n_orders']['distinct_count'],
    "country_cardinality": var_summaries['country_norm']['distinct_count'],
}
Path("artifacts/metrics").mkdir(parents=True, exist_ok=True)
with open("artifacts/metrics/per_customer_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)
metrics
```

### C3. Compare to a baseline (simulated)

```python
# Create a fake baseline and compare for illustration
baseline = {k: (v * 0.9 if isinstance(v, (int, float)) else v) for k, v in metrics.items()}

def pct_diff(a, b):
    return None if b == 0 else (a - b) / b

delta = {k: pct_diff(metrics[k], baseline[k]) if isinstance(metrics[k], (int, float)) else None for k in metrics}
{k: round(v, 3) for k, v in delta.items() if v is not None}
```

**Checkpoint:** Which metric movements would trigger investigation (>20% by default)?

---

## Part D — Operate at Scale & Integrate in Review

### D1. Tips for biggish data

- **Sampling:** `.sample(50_000)` for profiles; keep full data for Pandera validation.
- **Disable heavy bits:** `minimal=True` or turn off interactions/correlations you don’t need.
- **Column subsets:** profile only **review‑critical** columns per PR.
- **Persist artifacts:** write to `artifacts/reports/` and link in your PR checklist.

### D2. Review checklist snippet (add to PR template)

- [ ] Profile HTML attached (`per_customer_focused.html`).
- [ ] Key metrics JSON updated (`per_customer_metrics.json`).
- [ ] Any new high‑cardinality or outlier alerts acknowledged.
- [ ] Pandera schema still passes (link to Lab 3B test).

### D3. Wire to CI (concept)

- Save `profile.to_file()` output as a CI artifact.
- Parse `profile.to_dict()` and **fail** if critical thresholds are exceeded (e.g., null rate, cardinality spike, extreme mean/STD drift).

---

## Wrap‑Up

Add a markdown cell and answer:

1. List two alerts flagged by the profile and how you’d mitigate them.
2. Paste two metrics from your JSON that you will watch in CI and why.
3. Where in the pipeline will you generate and store the profiling report?

Export HTML report(s) and commit the JSON metrics.

---

- **Common pitfalls:** Running full profiles on multi‑million rows; forgetting to sample; not persisting artifacts; treating expected skew as an error instead of documenting it.

---

## Solution Snippets (reference)

**Minimal profile one‑liner:**

```python
ProfileReport(df.sample(20_000), minimal=True, explorative=True).to_file("artifacts/reports/df_min.html")
```

**Turn off heavy interactions:**

```python
ProfileReport(df, minimal=False, correlations={"pearson": {"calculate": True}}, interactions=None)
```

**Extract a null‑rate table from dict:**

```python
summary = profile_full.to_dict()
null_rates = {k: v.get('p_missing', None) for k, v in summary['variables'].items()}
{k: round(v, 4) for k, v in null_rates.items() if v is not None}
```
