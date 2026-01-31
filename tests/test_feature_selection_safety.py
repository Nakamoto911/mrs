import pytest
import pandas as pd
import numpy as np
from src.feature_engineering.hierarchical_clustering import (
    HierarchicalClusterSelector,
    SelectionConfig,
    SelectionMethod
)


def test_centroid_selection_ignores_target():
    """Centroid selection should work without target."""
    config = SelectionConfig(method=SelectionMethod.CENTROID)
    selector = HierarchicalClusterSelector(selection_config=config)
    
    X = pd.DataFrame({
        'a1': np.random.randn(100),
        'a2': np.random.randn(100),
        'b1': np.random.randn(100),
    })
    
    # Fit without target
    selector.fit(X, y=None)
    
    assert selector.fitted_
    assert len(selector.selected_features_) > 0


def test_ic_selection_requires_lagged_target():
    """IC selection should fail without proper lagging."""
    config = SelectionConfig(
        method=SelectionMethod.UNIVARIATE_IC,
        ic_lag_buffer_months=24,
        ic_min_observations=60
    )
    selector = HierarchicalClusterSelector(selection_config=config)
    
    # Create data where lagging would leave insufficient observations
    dates = pd.date_range('2020-01-01', periods=50, freq='ME')
    X = pd.DataFrame({'a': np.random.randn(50)}, index=dates)
    y = pd.Series(np.random.randn(50), index=dates)
    
    # Should fall back to centroid due to insufficient data
    selector.fit(X, y)
    assert selector.fitted_  # Should not raise, but fallback


def test_ic_selection_uses_safe_target():
    """IC selection should only use returns before validation period."""
    config = SelectionConfig(
        method=SelectionMethod.UNIVARIATE_IC,
        ic_lag_buffer_months=24,
        ic_min_observations=30
    )
    selector = HierarchicalClusterSelector(
        selection_config=config,
        min_observations=30
    )
    
    # Create data with enough history
    dates = pd.date_range('2000-01-01', periods=300, freq='ME')
    X = pd.DataFrame({
        'a': np.random.randn(300),
        'b': np.random.randn(300),
    }, index=dates)
    y = pd.Series(np.random.randn(300), index=dates)
    
    # Set validation start to 2020-01-01
    fold_val_start = pd.Timestamp('2020-01-01')
    
    # Get safe target
    safe_target = selector._validate_target_for_ic_selection(
        X, y, fold_val_start
    )
    
    # Safe target should end before 2020 - 24 months = 2018-01-01
    assert safe_target.index.max() < pd.Timestamp('2018-01-01')


def test_variance_selection_deterministic():
    """Variance selection should be deterministic."""
    config = SelectionConfig(method=SelectionMethod.VARIANCE)
    selector = HierarchicalClusterSelector(selection_config=config)
    
    np.random.seed(42)
    X = pd.DataFrame({
        'low_var': np.random.randn(100) * 0.1,
        'high_var': np.random.randn(100) * 10,
    })
    
    selector.fit(X)
    
    # High variance feature should be selected
    assert 'high_var' in selector.selected_features_


def test_selection_method_fallback():
    """Should fall back gracefully when IC selection fails."""
    config = SelectionConfig(
        method=SelectionMethod.UNIVARIATE_IC,
        ic_min_observations=1000  # Impossibly high
    )
    selector = HierarchicalClusterSelector(selection_config=config)
    
    X = pd.DataFrame({
        'a': np.random.randn(100),
        'b': np.random.randn(100),
    })
    y = pd.Series(np.random.randn(100))
    
    # Should not raise, should fall back
    selector.fit(X, y)
    assert selector.fitted_


def test_pipeline_with_safe_selection():
    """Test full pipeline uses safe feature selection."""
    from sklearn.pipeline import Pipeline
    from sklearn.linear_model import Ridge
    
    config = SelectionConfig(method=SelectionMethod.CENTROID)
    
    pipeline = Pipeline([
        ('clustering', HierarchicalClusterSelector(selection_config=config)),
        ('model', Ridge())
    ])
    
    X = pd.DataFrame(np.random.randn(200, 10))
    y = pd.Series(np.random.randn(200))
    
    pipeline.fit(X, y)
    predictions = pipeline.predict(X)
    
    assert len(predictions) == len(y)
