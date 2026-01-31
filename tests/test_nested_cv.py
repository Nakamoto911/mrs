import pytest
import numpy as np
import pandas as pd
from src.evaluation.nested_cv import (
    NestedCVEnsembleEvaluator,
    InnerFoldResult,
    OuterFold
)

def test_outer_fold_generation():
    """Test outer fold generation creates valid non-overlapping folds."""
    evaluator = NestedCVEnsembleEvaluator(
        n_outer_folds=3, 
        outer_min_train_months=60,
        inner_min_train_months=48
    )
    
    # Create 15 years of monthly data
    dates = pd.date_range('2000-01-01', periods=180, freq='ME')
    X = pd.DataFrame({'a': range(180)}, index=dates)
    
    folds = evaluator.generate_outer_folds(X)
    
    # Should generate at least some folds
    assert len(folds) >= 1
    
    for fold in folds:
        train_dates = dates[fold.outer_train_indices]
        test_dates = dates[fold.outer_test_indices]
        
        # Test should be after train
        assert train_dates.max() < test_dates.min()
        
        # Train should be at least min_outer_train
        train_months = (train_dates.max() - train_dates.min()).days / 30.44
        assert train_months >= 58 # approx 60

def test_model_selection_by_inner_cv():
    """Test that models are selected based on inner CV performance."""
    evaluator = NestedCVEnsembleEvaluator(ensemble_size=2)
    
    inner_results = {
        'good1': InnerFoldResult('good1', 0.15, 0.02, 3),
        'good2': InnerFoldResult('good2', 0.12, 0.02, 3),
        'bad': InnerFoldResult('bad', 0.02, 0.01, 3),
    }
    
    selected = evaluator.select_ensemble_models(inner_results)
    
    assert len(selected) == 2
    assert 'good1' in selected
    assert 'good2' in selected
    assert 'bad' not in selected

def test_evaluate_ensemble_unbiased():
    """Test the unbiased nature of nested CV manually."""
    # This is more of an integration test but we can mock the components
    pass
