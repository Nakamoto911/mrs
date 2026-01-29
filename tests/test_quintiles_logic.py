import pandas as pd
import numpy as np
import pytest
from src.feature_engineering.quintiles import QuintileFeatureGenerator

def test_new_high_case():
    """
    Test Case: 'The New High'
    Input Sequence: [1, 2, 3, 4, 100]
    Behavior: 100 is higher than all 4 previous values. Rank = 100%. Result: Q5.
    """
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 100.0])
    # Set min_observations to 2 to test with small sample
    generator = QuintileFeatureGenerator(min_observations=1)
    
    quintiles = generator._compute_quintile_rank(series)
    
    # Values at index 0 will be Q1 because score = (1-1)/(1-1) = NaN in my formula?
    # Wait, if count=1, (rank-1)/(count-1) = 0/0 = NaN.
    # The user said min_observations = 60 normally.
    
    # For index 4 (value 100):
    # rank = 5, count = 5. score = (5-1)/(5-1) = 1.0.
    # q = floor(1.0*5) + 1 = 6 -> clip to 5.
    assert quintiles.iloc[4] == 5

def test_new_low_case():
    """
    Test Case: 'The New Low'
    Input Sequence: [10, 9, 8, 7, 1]
    Behavior: 1 is lower than all 4 previous values. Rank = 0%. Result: Q1.
    """
    series = pd.Series([10.0, 9.0, 8.0, 7.0, 1.0])
    generator = QuintileFeatureGenerator(min_observations=1)
    
    quintiles = generator._compute_quintile_rank(series)
    
    # For index 4 (value 1):
    # rank = 1, count = 5. score = (1-1)/(5-1) = 0.0.
    # q = floor(0.0*5) + 1 = 1.
    assert quintiles.iloc[4] == 1

def test_1980_reversal():
    """
    Test Case: 'The 1980 Reversal'
    Input: [...trend up to 15%..., 14%]
    If using expanding sample (where max seen so far is 15%), 14% is still very high (Q5).
    """
    # Create a trend from 1 to 15, then 14.
    data = list(range(1, 16)) + [14.5]
    series = pd.Series(data)
    generator = QuintileFeatureGenerator(min_observations=1)
    
    quintiles = generator._compute_quintile_rank(series)
    
    # For index 15 (value 14.5):
    # History: 1...15. count = 16.
    # rank of 14.5 in [1...15, 14.5] is 15.
    # score = (15-1)/(16-1) = 14/15 = 0.933
    # q = floor(0.933 * 5) + 1 = floor(4.66) + 1 = 5.
    assert quintiles.iloc[15] == 5

def test_burn_in():
    """Verify that quintiles are NaN before min_observations."""
    series = pd.Series(np.random.randn(100))
    min_obs = 60
    generator = QuintileFeatureGenerator(min_observations=min_obs)
    
    quintiles = generator._compute_quintile_rank(series)
    
    # First 59 values should be NaN
    assert quintiles.iloc[:min_obs-1].isna().all()
    # 60th value (index 59) should have a value
    assert not np.isnan(quintiles.iloc[min_obs-1])

if __name__ == "__main__":
    pytest.main([__file__])
