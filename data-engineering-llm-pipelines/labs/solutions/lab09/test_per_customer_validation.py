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