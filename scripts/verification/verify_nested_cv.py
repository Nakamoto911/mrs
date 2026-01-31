import pandas as pd
import numpy as np
from src.evaluation.nested_cv import NestedCVEnsembleEvaluator, InnerFoldResult

def test_fold_generation():
    print("Testing fold generation...")
    evaluator = NestedCVEnsembleEvaluator(
        n_outer_folds=3, 
        outer_min_train_months=60
    )
    dates = pd.date_range('2000-01-01', periods=180, freq='ME')
    X = pd.DataFrame({'a': range(180)}, index=dates)
    folds = evaluator.generate_outer_folds(X)
    assert len(folds) > 0
    print(f"✓ Success ({len(folds)} folds generated)")

def test_selection_logic():
    print("Testing selection logic...")
    evaluator = NestedCVEnsembleEvaluator(ensemble_size=2)
    inner_results = {
        'm1': InnerFoldResult('m1', 0.2, 0.05, 5),
        'm2': InnerFoldResult('m2', 0.1, 0.05, 5),
        'm3': InnerFoldResult('m3', 0.01, 0.05, 5),
    }
    selected = evaluator.select_ensemble_models(inner_results)
    assert selected == ['m1', 'm2']
    print("✓ Success")

if __name__ == "__main__":
    try:
        test_fold_generation()
        test_selection_logic()
        print("\nAll Nested CV tests passed!")
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
