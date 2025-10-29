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
- Artifacts: Prefer `artifacts/clean/per_customer.parquet`.

**Start a notebook:** `week02_lab09.ipynb`

Synthetic fallback (run only if needed):

```python
import numpy as np
import pandas as pd

rng = np.random.default_rng(1)
ages = rng.integers(16, 80, 500).astype('int64')
users2 = pd.DataFrame({
    'CustomerID': [f'C{i:05d}' for i in range(500)],
    'country_norm': rng.choice(['USA','DE','SG','BR'], 500, p=[.55,.2,.15,.1]),
    'age': ages,
    'ltv_usd': np.round(np.clip(rng.lognormal(3.0, 0.7, 500), 0, 5e5), 2),
    'email': [f'user{i}@example.com' for i in range(500)],
    'is_adult': ages >= 18,  # Changed: derive from age
    'is_high_value': rng.random(500) > 0.9,
})
users2.head(3)
```

---

## Part A — Pandera DataFrame Schema

### A1. Define column types & simple constraints

```python
import pandera.pandas as pa
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
# DF-level: median ltv_usd must be within a sane band; and all CustomerIDs unique
UsersSchema = pa.DataFrameSchema(
    columns={
        'CustomerID': Column(object, nullable=False, checks=Check.str_matches(r'^C\d{5}$')),
        'country_norm': Column(object, nullable=False, checks=Check.isin(AllowedCountries)),
        'age': Column(pa.Int64, nullable=False, checks=Check.in_range(0, 120)),
        'ltv_usd': Column(float, nullable=False, checks=Check.ge(0)),
        'email': Column(object, nullable=False, checks=Check.str_matches(r'^.+@.+\..+$')),
        'is_adult': Column(bool, nullable=False),
        'is_high_value': Column(bool, nullable=False),
    },
    checks=[
        # Cross-column check: if age >= 18 then is_adult must be True
        pa.Check(lambda df: (df['age'] < 18) | df['is_adult'], 
                 error="is_adult must be True when age>=18"),
        # DF-level checks
        pa.Check(lambda df: df['ltv_usd'].median() <= 1e5, 
                 error="Median LTV too large"),
        pa.Check(lambda df: df['CustomerID'].is_unique, 
                 error="Duplicate CustomerID")
    ]
)

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
%pip install email-validator pydantic

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

Utility: validate or raise with artifacts

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

---

## Bonus Exercise — Complete Walkthrough with per_customer.parquet

This section demonstrates **all lab concepts** using the real `artifacts/clean/per_customer.parquet` file.

### Step 1: Load the per_customer dataset

```python
import pandas as pd
from pathlib import Path

# Load the per-customer aggregated data
per_cust_path = Path('artifacts/clean/per_customer.parquet')
if per_cust_path.exists():
    per_cust = pd.read_parquet(per_cust_path)
    print(f"Loaded {len(per_cust)} customer records")
    display(per_cust.head())
    print("\nData types:")
    print(per_cust.dtypes)
else:
    print(f"File not found: {per_cust_path}")
    print("Please run Lab 06 to generate this artifact first.")
```

**Expected columns:**

- `CustomerID` (string): Customer identifier (e.g., "ALFKI")
- `n_orders` (int64): Count of orders per customer
- `freight_mean` (float64): Average freight cost per customer
- `freight_sum` (float64): Total freight cost per customer
- `CompanyName` (string): Customer company name
- `Country` (string): Customer country
- `spend_segment` (string): Spending category ("low", "mid", "high")

---

### Step 2: Define Pandera DataFrame Schema with all constraint types

```python
import pandera.pandas as pa
from pandera import Column, Check

# Define allowed values
AllowedCountries = ['Germany', 'Mexico', 'France', 'Sweden', 'USA', 'Brazil', 
                    'Switzerland', 'Austria', 'UK', 'Canada', 'Denmark', 
                    'Finland', 'Norway', 'Spain', 'Italy', 'Belgium', 
                    'Portugal', 'Ireland', 'Poland', 'Argentina', 'Venezuela']
AllowedSegments = ['low', 'mid', 'high']

# Comprehensive schema with multiple check types
PerCustomerSchema = pa.DataFrameSchema(
    columns={
        # String pattern check with regex
        'CustomerID': Column(
            object, 
            nullable=False, 
            checks=Check.str_matches(r'^[A-Z]{5}$', error='CustomerID must be 5 uppercase letters')
        ),
        # Range checks on integers
        'n_orders': Column(
            pa.Int64, 
            nullable=False, 
            checks=[
                Check.ge(1, error='Must have at least 1 order'),
                Check.le(100, error='Unexpectedly high order count')
            ]
        ),
        # Range checks on floats
        'freight_mean': Column(
            float, 
            nullable=False, 
            checks=[
                Check.ge(0, error='Freight cannot be negative'),
                Check.le(1000, error='Average freight seems too high')
            ]
        ),
        'freight_sum': Column(
            float, 
            nullable=False, 
            checks=[
                Check.ge(0, error='Total freight cannot be negative'),
                Check.le(10000, error='Total freight seems unrealistic')
            ]
        ),
        # Category/allowlist checks
        'CompanyName': Column(
            object, 
            nullable=False, 
            checks=Check.str_length(min_value=2, max_value=100)
        ),
        'Country': Column(
            object, 
            nullable=False, 
            checks=Check.isin(AllowedCountries, error='Country not in allowed list')
        ),
        'spend_segment': Column(
            object, 
            nullable=False, 
            checks=Check.isin(AllowedSegments, error='Invalid spend segment')
        ),
    },
    checks=[
        # Cross-column check: freight_mean should be freight_sum / n_orders (with tolerance)
        pa.Check(
            lambda df: (
                (df['freight_mean'] - (df['freight_sum'] / df['n_orders'])).abs() < 0.01
            ).all(),
            error="freight_mean must equal freight_sum / n_orders"
        ),
        # DataFrame-level check: all CustomerIDs must be unique
        pa.Check(
            lambda df: df['CustomerID'].is_unique,
            error="Duplicate CustomerID found"
        ),
        # DataFrame-level: median freight_sum should be reasonable
        pa.Check(
            lambda df: 10 <= df['freight_sum'].median() <= 500,
            error="Median total freight outside expected range"
        ),
        # Check: high segment customers should have freight_sum >= 50
        pa.Check(
            lambda df: (df[df['spend_segment'] == 'high']['freight_sum'] >= 50).all(),
            error="High segment customers must have freight_sum >= 50"
        ),
    ],
    strict=False,  # Allow extra columns if they exist
    coerce=False   # Don't auto-convert types
)

# Validate the data
try:
    validated = PerCustomerSchema.validate(per_cust, lazy=True)
    print(f"✓ Validation passed for {len(validated)} records")
    display(validated.head(3))
except pa.errors.SchemaErrors as err:
    print(f"✗ Validation failed with {len(err.failure_cases)} errors")
    display(err.failure_cases.head(10))
```

---

### Step 3: Test error handling with intentional violations

```python
# Create a copy with intentional errors to test our schema
corrupted = per_cust.copy()

# Error 1: Invalid CustomerID pattern
corrupted.loc[0, 'CustomerID'] = 'ABC123'  # Should be 5 letters

# Error 2: Negative freight
corrupted.loc[1, 'freight_mean'] = -10.0

# Error 3: Invalid country
corrupted.loc[2, 'Country'] = 'Atlantis'

# Error 4: Unrealistic order count
corrupted.loc[3, 'n_orders'] = 150

# Error 5: Invalid spend segment
corrupted.loc[4, 'spend_segment'] = 'ultra'

# Error 6: Break cross-column consistency
corrupted.loc[5, 'freight_mean'] = 999.0  # Doesn't match sum/count

print("Testing schema with corrupted data...")
try:
    PerCustomerSchema.validate(corrupted, lazy=True)
except pa.errors.SchemaErrors as err:
    fc = err.failure_cases
    
    # Create a rollup summary for CI/logs
    rollup = (
        fc.groupby(['column', 'check'])
        .size()
        .reset_index(name='violation_count')
        .sort_values('violation_count', ascending=False)
    )
    
    print(f"\n✗ Found {len(fc)} total violations across {len(rollup)} check types\n")
    print("VIOLATION SUMMARY (for CI logs):")
    display(rollup)
    
    # Save detailed failure cases for analyst triage
    failure_path = Path('artifacts/validation/per_customer_failures.csv')
    failure_path.parent.mkdir(parents=True, exist_ok=True)
    fc.to_csv(failure_path, index=False)
    print(f"\n✓ Detailed failures saved to: {failure_path}")
    
    # Show first few detailed failures
    print("\nDETAILED FAILURES (sample):")
    display(fc[['schema_context', 'column', 'check', 'failure_case']].head(10))
```

---

### Step 4: Pydantic row-level contract

```python
from pydantic import BaseModel, Field, field_validator, ValidationError
from typing import Literal

class PerCustomerRow(BaseModel):
    """Row-level contract for per-customer data"""
    
    CustomerID: str = Field(pattern=r'^[A-Z]{5}$', description="5 uppercase letters")
    n_orders: int = Field(ge=1, le=100, description="Order count between 1-100")
    freight_mean: float = Field(ge=0, le=1000, description="Average freight 0-1000")
    freight_sum: float = Field(ge=0, le=10000, description="Total freight 0-10000")
    CompanyName: str = Field(min_length=2, max_length=100)
    Country: Literal['Germany', 'Mexico', 'France', 'Sweden', 'USA', 'Brazil',
                     'Switzerland', 'Austria', 'UK', 'Canada', 'Denmark',
                     'Finland', 'Norway', 'Spain', 'Italy', 'Belgium',
                     'Portugal', 'Ireland', 'Poland', 'Argentina', 'Venezuela']
    spend_segment: Literal['low', 'mid', 'high']
    
    @field_validator('freight_mean')
    @classmethod
    def freight_mean_matches_calculation(cls, v, info):
        """Ensure freight_mean equals freight_sum / n_orders"""
        freight_sum = info.data.get('freight_sum')
        n_orders = info.data.get('n_orders')
        
        if freight_sum is not None and n_orders is not None and n_orders > 0:
            expected = freight_sum / n_orders
            if abs(v - expected) > 0.01:  # Allow small floating point tolerance
                raise ValueError(
                    f'freight_mean ({v}) does not match freight_sum/n_orders ({expected:.2f})'
                )
        return v
    
    @field_validator('spend_segment')
    @classmethod
    def segment_matches_freight(cls, v, info):
        """Ensure spend_segment aligns with freight_sum thresholds"""
        freight_sum = info.data.get('freight_sum', 0)
        
        # Based on Lab 06: bins=[0, 20, 50, inf], labels=['low', 'mid', 'high']
        if v == 'low' and freight_sum >= 20:
            raise ValueError(f'spend_segment "low" but freight_sum={freight_sum} >= 20')
        if v == 'mid' and (freight_sum < 20 or freight_sum >= 50):
            raise ValueError(f'spend_segment "mid" but freight_sum={freight_sum} not in [20, 50)')
        if v == 'high' and freight_sum < 50:
            raise ValueError(f'spend_segment "high" but freight_sum={freight_sum} < 50')
        
        return v

# Test with a valid row
sample_row = per_cust.iloc[0].to_dict()
print("Testing valid row:")
valid_customer = PerCustomerRow(**sample_row)
print(f"✓ Valid: {valid_customer.CustomerID} - {valid_customer.CompanyName}")

# Test with an invalid row
print("\nTesting invalid row:")
try:
    invalid_row = sample_row.copy()
    invalid_row['CustomerID'] = '12345'  # Numbers instead of letters
    invalid_row['freight_mean'] = -5.0   # Negative
    PerCustomerRow(**invalid_row)
except ValidationError as e:
    print(f"✗ Validation errors:")
    for error in e.errors():
        print(f"  - {error['loc'][0]}: {error['msg']}")
```

---

### Step 5: Batch validation with Pydantic

```python
def validate_batch_pydantic(df, model_class, sample_size=None):
    """
    Validate a DataFrame using a Pydantic model row-by-row.
    Returns both valid records and error details.
    """
    errors = []
    valid_indices = []
    
    sample_df = df.head(sample_size) if sample_size else df
    
    for idx, row in sample_df.iterrows():
        try:
            model_class(**row.to_dict())
            valid_indices.append(idx)
        except ValidationError as e:
            for error in e.errors():
                errors.append({
                    'index': idx,
                    'CustomerID': row.get('CustomerID', 'UNKNOWN'),
                    'field': error['loc'][0],
                    'error_type': error['type'],
                    'message': error['msg']
                })
    
    return valid_indices, pd.DataFrame(errors)

# Validate the entire dataset
print("Validating all per-customer records with Pydantic...")
valid_idx, error_df = validate_batch_pydantic(per_cust)

print(f"\n✓ Valid records: {len(valid_idx)}/{len(per_cust)}")
if len(error_df) > 0:
    print(f"✗ Invalid records: {len(error_df.index.unique())}")
    print("\nError summary by field:")
    display(error_df.groupby(['field', 'error_type']).size().reset_index(name='count'))
    
    print("\nSample errors:")
    display(error_df.head(10))
else:
    print("✓ All records passed Pydantic validation!")
```

---

### Step 6: CI Hook Example with validate_or_artifact

```python
def validate_or_artifact(df, schema, name, out_dir='artifacts/validation'):
    """
    Validate a DataFrame or save failure artifacts for CI.
    Raises SystemExit on failure (fails CI pipeline).
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    try:
        return schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as err:
        fc = err.failure_cases
        dest = Path(out_dir) / f'{name}_failures.csv'
        fc.to_csv(dest, index=False)
        
        # Compact summary for console/CI logs
        top = (
            fc.groupby(['column', 'check'])
            .size()
            .reset_index(name='n')
            .sort_values('n', ascending=False)
            .head(5)
            .to_dict(orient='records')
        )
        
        # This would fail a CI build
        raise SystemExit(
            f"❌ Validation failed for {name}.\n"
            f"   Found {len(fc)} violations.\n"
            f"   Top issues: {top}\n"
            f"   Details: {dest}"
        )

# Example: Use in CI pipeline
print("Simulating CI validation check...")
try:
    validated_ci = validate_or_artifact(per_cust, PerCustomerSchema, 'per_customer_ci')
    print(f"✓ CI Check PASSED: {len(validated_ci)} records validated successfully")
except SystemExit as e:
    print(f"✗ CI Check FAILED:\n{e}")
```

---

### Step 7: Write a pytest test

Create a file `tests/test_per_customer_validation.py`:

```python
# lab09/test_per_customer_validation.py
import pytest
import pandas as pd
import pandera.pandas as pa
from pandera import Column, Check, DataFrameSchema
from pathlib import Path

@pytest.fixture
def per_customer_data():
    """Load the per_customer dataset"""
    # Update path to match the actual location from the lab instructions
    path = Path(__file__).parent.parent.parent / 'solutions' / 'artifacts' / 'clean' / 'per_customer.parquet'
    if not path.exists():
        # Try alternative path
        path = Path('solutions/artifacts/clean/per_customer.parquet')
    if not path.exists():
        pytest.skip(f"Test data not found: {path}")
    return pd.read_parquet(path)

@pytest.fixture
def per_customer_schema():
    """Define the Pandera schema"""
    AllowedSegments = ['low', 'mid', 'high']
    
    return DataFrameSchema(
        columns={
            'CustomerID': Column(str, nullable=False, 
                                checks=Check.str_matches(r'^[A-Z]{5}$')),
            'n_orders': Column(int, nullable=False, checks=Check.ge(1)),
            'freight_mean': Column(float, nullable=False, checks=Check.ge(0)),
            'freight_sum': Column(float, nullable=False, checks=Check.ge(0)),
            'CompanyName': Column(str, nullable=False),
            'Country': Column(str, nullable=False),
            'spend_segment': Column(str, nullable=False, 
                                   checks=Check.isin(AllowedSegments)),
        },
        checks=[
            Check(lambda df: df['CustomerID'].is_unique,
                  error="Duplicate CustomerID")
        ],
        strict=False,
        coerce=True  # Allow type coercion
    )

def test_schema_validates(per_customer_data, per_customer_schema):
    """Test that per_customer data passes schema validation"""
    try:
        validated = per_customer_schema.validate(per_customer_data, lazy=True)
        assert len(validated) > 0, "No records after validation"
        assert len(validated) == len(per_customer_data), "Some records were filtered out"
    except pa.errors.SchemaErrors as e:
        # Print detailed error information for debugging
        print(f"\nSchema validation failed with {len(e.failure_cases)} errors:")
        print(e.failure_cases)
        raise

def test_no_null_customer_ids(per_customer_data):
    """Test that CustomerID has no nulls"""
    assert per_customer_data['CustomerID'].notna().all()

def test_freight_consistency(per_customer_data):
    """Test that freight_mean matches freight_sum / n_orders"""
    calculated_mean = per_customer_data['freight_sum'] / per_customer_data['n_orders']
    diff = (per_customer_data['freight_mean'] - calculated_mean).abs()
    max_diff = diff.max()
    assert (diff < 0.01).all(), f"freight_mean inconsistent with freight_sum/n_orders. Max diff: {max_diff}"

def test_spend_segments_valid(per_customer_data):
    """Test that spend segments align with freight_sum thresholds"""
    low = per_customer_data[per_customer_data['spend_segment'] == 'low']
    if len(low) > 0:
        assert (low['freight_sum'] < 20).all(), f"Low segment has freight_sum >= 20. Max: {low['freight_sum'].max()}"
    
    mid = per_customer_data[per_customer_data['spend_segment'] == 'mid']
    if len(mid) > 0:
        assert ((mid['freight_sum'] >= 20) & (mid['freight_sum'] < 50)).all(), \
            f"Mid segment out of range [20, 50). Range: [{mid['freight_sum'].min()}, {mid['freight_sum'].max()})"
    
    high = per_customer_data[per_customer_data['spend_segment'] == 'high']
    if len(high) > 0:
        assert (high['freight_sum'] >= 50).all(), f"High segment has freight_sum < 50. Min: {high['freight_sum'].min()}"

def test_customer_id_format(per_customer_data):
    """Test that all CustomerIDs match the expected 5-letter format"""
    invalid = per_customer_data[~per_customer_data['CustomerID'].str.match(r'^[A-Z]{5}$', na=False)]
    assert len(invalid) == 0, f"Found {len(invalid)} invalid CustomerIDs: {invalid['CustomerID'].tolist()[:5]}"

def test_positive_orders(per_customer_data):
    """Test that all customers have at least 1 order"""
    assert (per_customer_data['n_orders'] >= 1).all(), \
        f"Found orders < 1. Min: {per_customer_data['n_orders'].min()}"

def test_data_types(per_customer_data):
    """Test that columns have expected data types"""
    assert pd.api.types.is_string_dtype(per_customer_data['CustomerID']) or \
           pd.api.types.is_object_dtype(per_customer_data['CustomerID'])
    assert pd.api.types.is_integer_dtype(per_customer_data['n_orders'])
    assert pd.api.types.is_float_dtype(per_customer_data['freight_mean'])
    assert pd.api.types.is_float_dtype(per_customer_data['freight_sum'])
```

Run the tests:

```bash
pip3.13 install pytest pandas pandera pyarrow fastparquet
pytest lab09/test_per_customer_validation.py -v
```

---

### Step 8: Reflection Questions

Add a markdown cell and answer:

1. **Which constraint types did you use?**
   - Pattern matching (regex for CustomerID)
   - Range checks (ge, le for numeric fields)
   - Category allowlists (isin for Country and spend_segment)
   - Cross-column checks (freight_mean vs calculated mean)
   - DataFrame-level checks (uniqueness, median thresholds)

2. **Pandera vs Pydantic boundaries:**
   - **Pandera**: Use for ETL pipelines, bulk DataFrame validation, aggregated data checks
   - **Pydantic**: Use for API boundaries, row-by-row ingestion, microservice contracts
   - For `per_customer.parquet`: Pandera is ideal since it's batch ETL output

3. **CI Violation Summary Example:**

   ```text
   ❌ Validation failed for per_customer_ci
   Found 6 violations:
   - column: Country, check: isin, count: 1
   - column: CustomerID, check: str_matches, count: 1
   - column: freight_mean, check: ge, count: 1
   See: artifacts/validation/per_customer_ci_failures.csv
   ```

4. **How would you surface errors?**

   - **Console/CI**: Show rollup summary (column + check + count)
   - **Analysts**: Export detailed `failure_cases.csv` with all violation metadata
   - **Monitoring**: Track validation metrics over time (violation rate, top issues)

---

### Step 9: Export and document

```python
# Generate a summary report
summary = {
    'dataset': 'per_customer.parquet',
    'total_records': len(per_cust),
    'validation_method': 'Pandera + Pydantic',
    'schema_checks': {
        'column_checks': 7,
        'dataframe_checks': 4,
        'pydantic_validators': 2
    },
    'validation_result': 'PASSED',
    'issues_found': 0
}

print("VALIDATION SUMMARY")
print("=" * 50)
for key, value in summary.items():
    print(f"{key:20s}: {value}")

# Export notebook to HTML
print("\n✓ Export notebook to HTML with: Jupyter > File > Save and Export > HTML")
print("✓ Include validation summary, rollup table, and test results")
```

---

## Key Takeaways

This complete walkthrough demonstrated:

✅ **Column constraints**: Regex patterns, range checks, category allowlists  
✅ **DataFrame checks**: Uniqueness, cross-column validation, aggregate thresholds  
✅ **Error handling**: Rollup summaries for CI, detailed CSVs for triage  
✅ **Pydantic contracts**: Field validators, custom business logic  
✅ **CI integration**: pytest tests, validate_or_artifact pattern  
✅ **Real data**: Applied all concepts to actual pipeline artifact

**Common pitfalls avoided:**

- Used lazy validation to collect all errors
- Provided both summary (CI) and detailed (triage) error outputs
- Separated DataFrame validation (Pandera) from row validation (Pydantic)
- Included floating-point tolerance for calculated fields
- Made schemas readable with clear error messages
