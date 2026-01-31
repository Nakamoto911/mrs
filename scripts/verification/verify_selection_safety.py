import pandas as pd
import numpy as np
from src.feature_engineering.hierarchical_clustering import (
    HierarchicalClusterSelector,
    SelectionConfig,
    SelectionMethod
)

def test_centroid_selection_ignores_target():
    print("Testing centroid selection ignores target...")
    config = SelectionConfig(method=SelectionMethod.CENTROID)
    selector = HierarchicalClusterSelector(selection_config=config)
    X = pd.DataFrame({'a1': np.random.randn(100), 'a2': np.random.randn(100), 'b1': np.random.randn(100)})
    selector.fit(X, y=None)
    assert selector.fitted_
    assert len(selector.selected_features_) > 0
    print("✓ Success")

def test_ic_selection_uses_safe_target():
    print("Testing IC selection uses safe target...")
    config = SelectionConfig(method=SelectionMethod.UNIVARIATE_IC, ic_lag_buffer_months=24, ic_min_observations=30)
    selector = HierarchicalClusterSelector(selection_config=config, min_observations=30)
    dates = pd.date_range('2000-01-01', periods=300, freq='ME')
    X = pd.DataFrame({'a': np.random.randn(300), 'b': np.random.randn(300)}, index=dates)
    y = pd.Series(np.random.randn(300), index=dates)
    fold_val_start = pd.Timestamp('2020-01-01')
    safe_target = selector._validate_target_for_ic_selection(X, y, fold_val_start)
    assert safe_target.index.max() < pd.Timestamp('2018-01-01')
    print("✓ Success")

def test_variance_selection_deterministic():
    print("Testing variance selection is deterministic...")
    config = SelectionConfig(method=SelectionMethod.VARIANCE)
    selector = HierarchicalClusterSelector(selection_config=config)
    np.random.seed(42)
    X = pd.DataFrame({'low_var': np.random.randn(100) * 0.1, 'high_var': np.random.randn(100) * 10})
    selector.fit(X)
    assert 'high_var' in selector.selected_features_
    print("✓ Success")

if __name__ == "__main__":
    try:
        test_centroid_selection_ignores_target()
        test_ic_selection_uses_safe_target()
        test_variance_selection_deterministic()
        print("\nAll feature selection safety tests passed!")
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
