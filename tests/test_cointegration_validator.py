import numpy as np
import pandas as pd
import pytest
from src.feature_engineering.cointegration_validator import (
    CointegrationValidator,
    CointegrationStatus
)

def test_johansen_test_execution():
    """Johansen test should run without error on cointegrated data."""
    # Generate cointegrated series
    np.random.seed(42)
    n = 200
    e1 = np.random.randn(n).cumsum()  # Random walk
    e2 = e1 + np.random.randn(n) * 0.5  # Cointegrated with e1
    
    df = pd.DataFrame({'series1': e1, 'series2': e2})
    
    # Needs lower min_observations for this test
    validator = CointegrationValidator(min_observations=50)
    result = validator.test_pair(df, 'series1', 'series2', 'test_pair', 'test')
    
    assert result.status == CointegrationStatus.VALIDATED
    assert result.coint_rank >= 1
    assert result.include_in_model == True
    assert result.ect_stationarity_pvalue < 0.05

def test_rejection_of_independent_series():
    """Should reject cointegration for independent random walks."""
    np.random.seed(42)
    n = 200
    e1 = np.random.randn(n).cumsum()
    e2 = np.random.randn(n).cumsum()  # Independent
    
    df = pd.DataFrame({'series1': e1, 'series2': e2})
    
    validator = CointegrationValidator(min_observations=50, allow_theory_override=False)
    result = validator.test_pair(df, 'series1', 'series2', 'test_pair', 'test')
    
    assert result.status == CointegrationStatus.REJECTED
    assert result.include_in_model == False

def test_theory_override():
    """Theory override should include rejected pairs when enabled."""
    np.random.seed(42)
    n = 200
    e1 = np.random.randn(n).cumsum()
    e2 = np.random.randn(n).cumsum()  # Independent
    
    df = pd.DataFrame({'series1': e1, 'series2': e2})
    
    validator = CointegrationValidator(min_observations=50, allow_theory_override=True)
    # Mock override pairs for test
    validator.THEORY_OVERRIDE_PAIRS = ['special_pair']
    
    result = validator.test_pair(df, 'series1', 'series2', 'special_pair', 'theory')
    
    assert result.status == CointegrationStatus.THEORY_OVERRIDE
    assert result.include_in_model == True

def test_insufficient_data_handling():
    """Should handle insufficient data gracefully."""
    df = pd.DataFrame({
        'series1': [1, 2, 3],
        'series2': [4, 5, 6]
    })
    
    validator = CointegrationValidator(min_observations=120)
    result = validator.test_pair(df, 'series1', 'series2', 'test', 'test')
    
    assert result.status == CointegrationStatus.INSUFFICIENT_DATA
    assert result.include_in_model == False

if __name__ == "__main__":
    print("Running CointegrationValidator unit tests...")
    test_johansen_test_execution()
    print("✓ test_johansen_test_execution passed")
    test_rejection_of_independent_series()
    print("✓ test_rejection_of_independent_series passed")
    test_theory_override()
    print("✓ test_theory_override passed")
    test_insufficient_data_handling()
    print("✓ test_insufficient_data_handling passed")
    print("\nAll tests passed!")
