import pytest
import numpy as np
import pandas as pd
from src.preprocessing.scaling import TimeSeriesScaler
from src.preprocessing.imputation import PointInTimeImputer

def test_time_series_scaler_expanding():
    """Test that expanding window scaling matches manual calculation."""
    scaler = TimeSeriesScaler(window_type='expanding', min_periods=2)
    
    data = pd.DataFrame({'a': [1.0, 2.0, 3.0, 4.0, 5.0]})
    # Expected stats for t=2 (using 1.0, 2.0): mean=1.5, std=0.707
    # t=2 value is 3.0 -> scaled: (3.0 - 1.5) / 0.707 = 2.12
    
    # fit_transform_rolling uses shifted stats (t-1 stats for t scaling)
    scaled = scaler.fit_transform_rolling(data)
    
    assert pd.isna(scaled.iloc[0, 0])
    assert pd.isna(scaled.iloc[1, 0]) # min_periods=2
    
    # Check index 2
    # mean of [1,2] = 1.5
    # std of [1,2] = 0.7071
    # value is 3.0
    expected_2 = (3.0 - 1.5) / 0.707106781
    assert np.isclose(scaled.iloc[2, 0], expected_2)

def test_imputer_expanding():
    """Test that expanding imputation uses past data only."""
    imputer = PointInTimeImputer(strategy='median')
    
    data = pd.DataFrame({'a': [1.0, 10.0, np.nan, 2.0, np.nan]})
    # t=2 (np.nan): median of [1, 10] = 5.5. Should be filled with 5.5.
    
    imputed = imputer.transform_expanding(data)
    
    assert imputed.iloc[0, 0] == 1.0
    assert imputed.iloc[2, 0] == 5.5 # median of 1, 10
    
    # t=4 (np.nan): median of [1, 10, nan, 2] -> [1, 2, 10] median = 2.0
    assert imputed.iloc[4, 0] == 2.0

def test_scaler_oos():
    """Test that OOS scaling uses the final training state."""
    scaler = TimeSeriesScaler()
    train_data = pd.DataFrame({'a': [1.0, 2.0, 3.0]}) # mean=2, std=1
    
    scaler.fit(train_data)
    
    test_data = pd.DataFrame({'a': [10.0]})
    scaled_test = scaler.transform(test_data)
    
    assert scaled_test.iloc[0, 0] == (10.0 - 2.0) / 0.81649658 # std of 1,2,3 is 0.816
