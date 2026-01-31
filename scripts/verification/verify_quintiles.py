import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from feature_engineering.quintiles import QuintileFeatureGenerator

def run_verification():
    print("Running Quintile Logic Verification...")
    
    # Test 1: New High Case
    print("\nTest 1: New High Case")
    series = pd.Series([1.0, 2.0, 3.0, 4.0, 100.0])
    generator = QuintileFeatureGenerator(min_observations=1)
    quintiles = generator._compute_quintile_rank(series)
    print(f"Input: {series.tolist()}")
    print(f"Output: {quintiles.tolist()}")
    assert quintiles.iloc[4] == 5, f"Expected 5, got {quintiles.iloc[4]}"
    print("SUCCESS: 100 correctly mapped to Q5")

    # Test 2: New Low Case
    print("\nTest 2: New Low Case")
    series = pd.Series([10.0, 9.0, 8.0, 7.0, 1.0])
    generator = QuintileFeatureGenerator(min_observations=1)
    quintiles = generator._compute_quintile_rank(series)
    print(f"Input: {series.tolist()}")
    print(f"Output: {quintiles.tolist()}")
    assert quintiles.iloc[4] == 1, f"Expected 1, got {quintiles.iloc[4]}"
    print("SUCCESS: 1 correctly mapped to Q1")

    # Test 3: 1980 Reversal
    print("\nTest 3: 1980 Reversal")
    data = list(range(1, 16)) + [14.5]
    series = pd.Series(data)
    generator = QuintileFeatureGenerator(min_observations=1)
    quintiles = generator._compute_quintile_rank(series)
    print(f"Last value: {series.iloc[-1]}")
    print(f"Last quintile: {quintiles.iloc[-1]}")
    assert quintiles.iloc[15] == 5, f"Expected 5, got {quintiles.iloc[15]}"
    print("SUCCESS: 14.5 correctly mapped to Q5 (High relative to history)")

    # Test 4: Burn-in
    print("\nTest 4: Burn-in Check")
    series = pd.Series(np.random.randn(100))
    min_obs = 60
    generator = QuintileFeatureGenerator(min_observations=min_obs)
    quintiles = generator._compute_quintile_rank(series)
    nan_count = quintiles.iloc[:min_obs-1].isna().sum()
    print(f"NaN count in first {min_obs-1} observations: {nan_count}")
    assert nan_count == min_obs - 1
    assert not np.isnan(quintiles.iloc[min_obs-1])
    print("SUCCESS: Burn-in period correctly respected")

    print("\nALL VERIFICATION TESTS PASSED!")

if __name__ == "__main__":
    try:
        run_verification()
    except Exception as e:
        print(f"VERIFICATION FAILED: {e}")
        sys.exit(1)
