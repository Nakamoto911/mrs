import numpy as np
import pandas as pd
from src.evaluation.benchmarking import (
    AssetClass,
    ASSET_THRESHOLDS,
    rate_ic,
    benchmark_model,
    compute_implied_sharpe,
    compute_naive_benchmark_sharpe,
    get_thresholds_for_asset
)

def test_threshold_ordering():
    """Thresholds should be monotonically ordered."""
    for asset_class, thresholds in ASSET_THRESHOLDS.items():
        assert thresholds.ic_excellent > thresholds.ic_good
        assert thresholds.ic_good > thresholds.ic_acceptable
        assert thresholds.ic_acceptable > thresholds.ic_minimum
        assert thresholds.ic_suspicious > thresholds.ic_excellent


def test_rating_boundaries():
    """Rating function should respect boundaries."""
    thresh = ASSET_THRESHOLDS[AssetClass.EQUITIES]
    
    assert rate_ic(thresh.ic_excellent, thresh) == 'excellent'
    assert rate_ic(thresh.ic_excellent - 0.001, thresh) == 'good'
    assert rate_ic(thresh.ic_good, thresh) == 'good'
    assert rate_ic(thresh.ic_good - 0.001, thresh) == 'acceptable'
    assert rate_ic(thresh.ic_acceptable, thresh) == 'acceptable'
    assert rate_ic(thresh.ic_acceptable - 0.001, thresh) == 'minimum'
    assert rate_ic(thresh.ic_minimum, thresh) == 'minimum'
    assert rate_ic(thresh.ic_minimum - 0.001, thresh) == 'poor'


def test_suspicious_detection():
    """Should flag suspiciously high ICs."""
    result = benchmark_model(
        ic=0.35,  # Way too high for equities
        ic_t_stat=5.0,
        ic_p_value=0.0001,
        asset='SPX'
    )
    
    assert result.is_suspicious
    assert any("suspicious" in w.lower() for w in result.warnings)


def test_implied_sharpe_calculation():
    """Implied Sharpe should scale with IC."""
    sr_low = compute_implied_sharpe(0.05)
    sr_high = compute_implied_sharpe(0.15)
    
    assert sr_high > sr_low
    assert sr_high < 3.0  # Sanity check


def test_naive_benchmark_sharpe():
    """Test naive benchmark Sharpe calculation."""
    # Create simple returns
    returns = pd.Series([0.01, 0.02, -0.01, 0.03, 0.01] * 12) # 60 months
    sr = compute_naive_benchmark_sharpe(returns)
    assert sr > 0
    
    # Test empty
    assert compute_naive_benchmark_sharpe(pd.Series([])) == 0.0


def test_get_thresholds_for_asset():
    """Test asset mapping."""
    assert get_thresholds_for_asset('SPX') == ASSET_THRESHOLDS[AssetClass.EQUITIES]
    assert get_thresholds_for_asset('BOND') == ASSET_THRESHOLDS[AssetClass.BONDS]
    assert get_thresholds_for_asset('GOLD') == ASSET_THRESHOLDS[AssetClass.COMMODITIES]
if __name__ == "__main__":
    # Workaround for running without pytest
    print("Running tests...")
    test_threshold_ordering()
    print("test_threshold_ordering passed")
    test_rating_boundaries()
    print("test_rating_boundaries passed")
    test_suspicious_detection()
    print("test_suspicious_detection passed")
    test_implied_sharpe_calculation()
    print("test_implied_sharpe_calculation passed")
    test_naive_benchmark_sharpe()
    print("test_naive_benchmark_sharpe passed")
    test_get_thresholds_for_asset()
    print("test_get_thresholds_for_asset passed")
    print("All tests passed!")
