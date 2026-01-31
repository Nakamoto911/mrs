import pandas as pd
import numpy as np
from src.evaluation.holdout import HoldoutManager, validate_holdout_never_touched

def test_holdout_split_percentage():
    print("Testing percentage-based holdout split...")
    manager = HoldoutManager(method="percentage", holdout_pct=0.2, min_holdout_months=10)
    dates = pd.date_range('2000-01-01', periods=100, freq='ME')
    X = pd.DataFrame({'a': range(100)}, index=dates)
    y = pd.Series(range(100), index=dates)
    
    X_dev, y_dev, X_hold, y_hold = manager.split_data(X, y)
    
    assert len(X_dev) == 80
    assert len(X_hold) == 20
    assert X_dev.index.max() < X_hold.index.min()
    print("✓ Success")


def test_holdout_split_date():
    print("Testing date-based holdout split...")
    manager = HoldoutManager(method="date", holdout_start="2018-01-01", min_holdout_months=1)
    dates = pd.date_range('2017-01-01', periods=24, freq='ME')
    X = pd.DataFrame({'a': range(24)}, index=dates)
    y = pd.Series(range(24), index=dates)
    
    X_dev, y_dev, X_hold, y_hold = manager.split_data(X, y)
    
    assert X_hold.index.min() >= pd.Timestamp('2018-01-01')
    assert len(X_hold) == 12
    print("✓ Success")

def test_leakage_detection():
    print("Testing leakage detection...")
    dates = pd.date_range('2000-01-01', periods=240, freq='ME')
    training_dates = pd.DatetimeIndex(dates[100:220])
    
    try:
        validate_holdout_never_touched(training_dates, pd.Timestamp('2018-01-01'))
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "DATA LEAKAGE DETECTED" in str(e)
        print("✓ Success")

if __name__ == "__main__":
    try:
        test_holdout_split_percentage()
        test_holdout_split_date()
        test_leakage_detection()
        print("\nAll holdout tests passed!")
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
