import pytest
import numpy as np
import pandas as pd
from src.evaluation.inference import (
    compute_newey_west_se,
    compute_hansen_hodrick_se,
    compute_effective_sample_size,
    compute_ic_with_inference
)

def test_newey_west_increases_se():
    """NW SE should be larger than naive SE for overlapping data."""
    # Generate overlapping returns
    np.random.seed(42)
    n = 240
    horizon = 24
    
    # Create persistent returns to induce autocorrelation in residuals
    returns = np.random.randn(n).cumsum()
    forward_returns = pd.Series([returns[i+horizon] - returns[i] for i in range(n - horizon)])
    
    # Lagged signal (highly correlated but overlapping)
    signal = forward_returns.shift(1).dropna()
    y_true = forward_returns[1:]
    
    result = compute_ic_with_inference(y_true, signal, horizon=horizon)
    
    assert result.se_nw > result.se_raw, f"NW SE ({result.se_nw:.4f}) should exceed naive SE ({result.se_raw:.4f})"
    assert result.se_nw > 1.5 * result.se_raw, f"NW SE should be substantially larger (factor of {result.se_nw/result.se_raw:.2f})"


def test_effective_sample_size():
    """Effective N should be approximately n/horizon."""
    eff_n = compute_effective_sample_size(n_obs=480, horizon=24)
    assert eff_n == 20
    
    eff_n_small = compute_effective_sample_size(n_obs=10, horizon=24)
    assert eff_n_small == 1


def test_hansen_hodrick_vs_newey_west():
    """Compare HH and NW SEs."""
    np.random.seed(42)
    n = 100
    horizon = 12
    y_true = np.random.randn(n)
    y_pred = y_true + np.random.randn(n) * 0.5
    
    se_nw, _ = compute_newey_west_se(y_true, y_pred, horizon)
    se_hh, _ = compute_hansen_hodrick_se(y_true, y_pred, horizon)
    
    assert se_nw > 0
    assert se_hh > 0
    # They should be in the same ballpark but NW is typically smoother
    assert 0.5 < se_nw / se_hh < 2.0


def test_kernels():
    """Test NW with different kernels."""
    np.random.seed(42)
    n = 100
    horizon = 12
    y_true = np.random.randn(n)
    y_pred = y_true + np.random.randn(n) * 0.5
    
    se_bartlett, _ = compute_newey_west_se(y_true, y_pred, horizon, kernel='bartlett')
    se_parzen, _ = compute_newey_west_se(y_true, y_pred, horizon, kernel='parzen')
    
    assert se_bartlett != se_parzen
    assert se_bartlett > 0
    assert se_parzen > 0

def test_inference_result_attributes():
    """Verify InferenceResult contains expected metrics."""
    np.random.seed(42)
    n = 100
    y_true = pd.Series(np.random.randn(n))
    y_pred = pd.Series(np.random.randn(n))
    
    result = compute_ic_with_inference(y_true, y_pred, horizon=12)
    
    assert hasattr(result, 'estimate')
    assert hasattr(result, 't_stat_nw')
    assert hasattr(result, 'p_value_nw')
    assert hasattr(result, 'effective_n')
    assert result.n_obs == n
    assert result.horizon == 12
