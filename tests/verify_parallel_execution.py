import logging
# Configure logging before importing run_tournament to see output
logging.basicConfig(level=logging.INFO)

from run_tournament import ModelTournament
import pandas as pd
from pathlib import Path

# Mock config or use a real one?
# Let's try to run with a minimal setup.

def test_parallel_execution():
    print("Initializing Tournament...")
    tournament = ModelTournament()
    
    # Override config for speed and to pass checks
    tournament.config['models'] = {
        'linear': {'ridge': {'type': 'linear', 'params': {'alpha': 1.0}, 'enabled': True}},
        'tree': {'xgboost': {'type': 'tree', 'params': {'n_estimators': 10}, 'enabled': True}}
    }
    tournament.model_configs = {
        'ridge': {'type': 'linear', 'params': {'alpha': 1.0}},
        'xgboost': {'type': 'tree', 'params': {'n_estimators': 10}}
    }
    # Disable holdout for test
    tournament.config['validation'] = {'holdout': {'enabled': False}, 'cv': {'min_train_months': 12, 'validation_months': 12, 'step_months': 12}}
    tournament.holdout_enabled = False # Force update instance variable if already set
    
    # Mock data to avoid loading 1GB files (Need more for CV)
    dates = pd.date_range(start='2000-01-01', periods=300, freq='M')
    tournament.features = pd.DataFrame({
        'feat1': pd.Series(range(300), index=dates),
        'feat2': pd.Series(range(300), index=dates)
    })
    tournament.targets = {
        'SPX_return': pd.Series(range(300), index=dates)
    }
    tournament.ASSETS = ['SPX']
    tournament.experiments_dir = Path('tests/temp_experiments')
    tournament.experiments_dir.mkdir(parents=True, exist_ok=True)
    (tournament.experiments_dir / 'models').mkdir(exist_ok=True)
    (tournament.experiments_dir / 'predictions').mkdir(exist_ok=True)
    (tournament.experiments_dir / 'cv_results').mkdir(exist_ok=True)
    
    print("Running Tournament...")
    try:
        results = tournament.run_tournament(assets=['SPX'], models=['ridge', 'xgboost'])
        print("Tournament Finished!")
        print(results)
    except Exception as e:
        print(f"FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_parallel_execution()
