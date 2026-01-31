import numpy as np
import pandas as pd
from src.evaluation.effective_sample_size import EffectiveSampleSizeEstimator

def test_simple_ess():
    print("Testing simple ESS...")
    estimator = EffectiveSampleSizeEstimator(method="simple", horizon=24)
    result = estimator.estimate(n_obs=240)
    assert result.n_effective == 10
    print("✓ Success")

def test_newey_west_with_iid_data():
    print("Testing NW ESS with IID data...")
    np.random.seed(42)
    residuals = np.random.randn(200)
    estimator = EffectiveSampleSizeEstimator(method="newey_west", horizon=24)
    result = estimator.estimate(residuals)
    assert result.n_effective > 150
    print(f"✓ Success (n_eff={result.n_effective:.1f})")

def test_newey_west_with_autocorrelated_data():
    print("Testing NW ESS with autocorrelated data...")
    np.random.seed(42)
    n = 200
    rho = 0.8
    residuals = np.zeros(n)
    residuals[0] = np.random.randn()
    for t in range(1, n):
        residuals[t] = rho * residuals[t-1] + np.random.randn()
    estimator = EffectiveSampleSizeEstimator(method="newey_west", horizon=24)
    result = estimator.estimate(residuals)
    assert result.n_effective < 100
    print(f"✓ Success (n_eff={result.n_effective:.1f})")

def test_autocorr_based():
    print("Testing autocorrelation-based ESS...")
    np.random.seed(42)
    n = 200
    # Highly persistent
    residuals = np.cumsum(np.random.randn(n))
    estimator = EffectiveSampleSizeEstimator(method="autocorr", horizon=24)
    result = estimator.estimate(residuals)
    assert result.n_effective < 25
    print(f"✓ Success (n_eff={result.n_effective:.1f}, dep={result.details['dependence_length']})")

if __name__ == "__main__":
    try:
        test_simple_ess()
        test_newey_west_with_iid_data()
        test_newey_west_with_autocorrelated_data()
        test_autocorr_based()
        print("\nAll Effective Sample Size tests passed!")
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
