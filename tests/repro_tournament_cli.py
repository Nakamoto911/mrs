
import sys
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
import pandas as pd
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))
sys.path.insert(0, str(Path(__file__).parent.parent))

# Mock modules that might not be installed or need bypassing
sys.modules['preprocessing'] = MagicMock()
sys.modules['feature_engineering'] = MagicMock()
sys.modules['models'] = MagicMock()
sys.modules['evaluation'] = MagicMock()
sys.modules['yaml'] = MagicMock()

# Now import the class to test
# We need to mock logging config in run_tournament to avoid it running on import
with patch('logging.basicConfig'):
    from run_tournament import ModelTournament

class TestTournamentFlow(unittest.TestCase):
    def setUp(self):
        self.tournament = ModelTournament()
        self.tournament.experiments_dir = Path('experiments')
        self.tournament.ASSETS = ['SPX', 'BOND', 'GOLD']
        self.tournament.MODEL_CONFIGS = {'m1': {}, 'm2': {}}
        self.tournament.targets = {'SPX_return': pd.Series([1, 2]), 'BOND_return': pd.Series([1, 2]), 'GOLD_return': pd.Series([1, 2])}
        self.tournament.features = pd.DataFrame({'f1': [1, 2]})
        
    def test_assets_all_expansion(self):
        """Test that --assets all expands to full asset list."""
        
        # Mock dependencies to avoid actual execution
        self.tournament.train_model = MagicMock()
        
        # Mock Validator
        validator_mock = MagicMock()
        validator_mock.evaluate.return_value = MagicMock(metrics={'IC_mean': 0.1}, n_folds=5)
        
        with patch('run_tournament.CrossValidator', return_value=validator_mock), \
             patch('run_tournament.TimeSeriesCV'), \
             patch('run_tournament.pd.DataFrame.to_csv'): 
            
            # Case 1: passing ['all']
            self.tournament.run_tournament(assets=['all'], models=['m1'])
            
            # Check if it iterated over all assets
            # We can check by seeing if train_model was called for each asset
            calls = [c[0][0] for c in validator_mock.evaluate.call_args_list]
            # Since evaluate takes (model, X, y...), the model object is first arg.
            # But wait, run_tournament calls evaluate(self.train_model(...), ...)
            # So let's check the asset argument in evaluate call kwargs if available or context
            
            # Better: check logging calls? Or check that result list has all assets
            results = self.tournament.results
            unique_assets = results['asset'].unique()
            self.assertEqual(set(unique_assets), set(['SPX', 'BOND', 'GOLD']))

    @patch('run_tournament.ModelTournament')
    @patch('run_tournament.argparse.ArgumentParser.parse_args')
    @patch('run_tournament.pd.read_parquet')
    @patch('run_tournament.Path.exists')
    def test_main_feature_skip(self, mock_exists, mock_read_parquet, mock_args, mock_tournament_cls):
        """Test that main skips feature pipeline if file exists."""
        from run_tournament import main
        
        # Setup mocks
        mock_args.return_value = MagicMock(
            assets=['all'], models=['all'], 
            features_only=False, eval_only=False, 
            fred_api_key=None
        )
        
        tournament_instance = mock_tournament_cls.return_value
        tournament_instance.experiments_dir = Path('experiments')
        
        # Case 1: File exists -> Skip pipeline
        mock_exists.return_value = True 
        
        main()
        
        # Verify run_feature_pipeline was NOT called
        tournament_instance.run_feature_pipeline.assert_not_called()
        # Verify read_parquet WAS called
        mock_read_parquet.assert_called()
        
        # Case 2: File does not exist -> Run pipeline
        mock_exists.return_value = False
        tournament_instance.reset_mock()
        mock_read_parquet.reset_mock()
        
        main()
        
        tournament_instance.run_feature_pipeline.assert_called()

if __name__ == '__main__':
    unittest.main()
