import pytest
import numpy as np
from src.evaluation.effective_sample_size import EffectiveSampleSizeEstimator


def test_simple_ess():
    """Test simple horizon-based ESS."""
    estimator = EffectiveSampleSizeEstimator(method="simple", horizon=24)
    result = estimator.estimate(n_obs=240)
    assert result.n_effective == 10


def test_newey_west_with_iid_data():
    """With IID data, NW ESS should be close to n."""
    np.random.seed(42)
    residuals = np.random.randn(200)
    
    estimator = EffectiveSampleSizeEstimator(method="newey_west", horizon=24)
    result = estimator.estimate(residuals)
    
    # IID data: n_eff should be close to n
    assert result.n_effective > 150


def test_newey_west_with_autocorrelated_data():
    """With autocorrelated data, NW ESS should be reduced."""
    np.random.seed(42)
    
    # Generate AR(1) process with high persistence
    n = 200
    rho = 0.8
    residuals = np.zeros(n)
    residuals[0] = np.random.randn()
    for t in range(1, n):
        residuals[t] = rho * residuals[t-1] + np.random.randn()
    
    estimator = EffectiveSampleSizeEstimator(method="newey_west", horizon=24)
    result = estimator.estimate(residuals)
    
    # Should be much less than n due to autocorrelation
    assert result.n_effective < 100
    assert result.adjustment_ratio < 0.5


def test_hansen_hodrick():
    """Test Hansen-Hodrick ESS."""
    np.random.seed(42)
    n = 200
    residuals = np.random.randn(n)
    
    estimator = EffectiveSampleSizeEstimator(method="hansen_hodrick", horizon=24)
    result = estimator.estimate(residuals)
    
    # For IID data, HH should also be close to n
    assert result.n_effective > 150


def test_autocorr_based():
    """Test autocorrelation-based ESS."""
    np.random.seed(42)
    n = 200
    # Highly persistent
    residuals = np.cumsum(np.random.randn(n))
    
    estimator = EffectiveSampleSizeEstimator(method="autocorr", horizon=24)
    result = estimator.estimate(residuals)
    
    # Persistent data should have very low ESS
    assert result.n_effective < 20
    assert result.details['dependence_length'] > 10
