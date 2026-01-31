import pytest
import pandas as pd
import numpy as np
from src.evaluation.holdout import HoldoutManager, validate_holdout_never_touched

def test_holdout_split_percentage():
    """Test percentage-based holdout split."""
    manager = HoldoutManager(method="percentage", holdout_pct=0.2, min_holdout_months=10)
    dates = pd.date_range('2000-01-01', periods=100, freq='ME')
    X = pd.DataFrame({'a': range(100)}, index=dates)
    y = pd.Series(range(100), index=dates)
    
    X_dev, y_dev, X_hold, y_hold = manager.split_data(X, y)
    
    assert len(X_dev) == 80  # 80%
    assert len(X_hold) == 20   # 20%
    assert X_dev.index.max() < X_hold.index.min()  # No overlap


def test_holdout_split_date():
    """Test date-based holdout split."""
    manager = HoldoutManager(method="date", holdout_start="2018-01-01", min_holdout_months=1)
    dates = pd.date_range('2017-01-01', periods=24, freq='ME')
    X = pd.DataFrame({'a': range(24)}, index=dates)
    y = pd.Series(range(24), index=dates)
    
    X_dev, y_dev, X_hold, y_hold = manager.split_data(X, y)
    
    assert X_hold.index.min() >= pd.Timestamp('2018-01-01')
    assert len(X_hold) == 12


def test_leakage_detection():
    """Test that leakage is detected."""
    dates = pd.date_range('2000-01-01', periods=240, freq='ME')
    
    # Simulate training dates that include holdout period
    training_dates = pd.DatetimeIndex(dates[100:220])  # Includes 2018+
    
    with pytest.raises(ValueError, match="DATA LEAKAGE"):
        validate_holdout_never_touched(training_dates, pd.Timestamp('2018-01-01'))


def test_min_holdout_enforcement():
    """Test that minimum holdout requirements are enforced."""
    manager = HoldoutManager(
        method="percentage", 
        holdout_pct=0.05,  # Too small
        min_holdout_months=48
    )
    dates = pd.date_range('2000-01-01', periods=240, freq='ME')
    
    with pytest.raises(ValueError, match="less than minimum"):
        manager.compute_split(dates)
