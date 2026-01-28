import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
import sys
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from run_tournament import ModelTournament
from evaluation import SHAPAnalyzer

logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

def main():
    experiments_dir = Path('experiments')
    results_path = experiments_dir / 'cv_results' / 'tournament_results.csv'
    
    if not results_path.exists():
        logger.error(f"Results not found at {results_path}")
        return

    # 1. Load results and find best models
    results_df = pd.read_csv(results_path)
    # Filter out rows with errors
    results_df = results_df[results_df['IC_mean'].notna()]
    best_models = results_df.loc[results_df.groupby('asset')['IC_mean'].idxmax()]
    
    logger.info("Best models identified:")
    print(best_models[['asset', 'model', 'IC_mean']])

    # 2. Initialize tournament to use its data loading logic
    tournament = ModelTournament()
    
    # Load latest features
    feature_files = sorted(list(experiments_dir.glob('features/features_*.parquet')))
    if not feature_files:
        logger.error("No feature files found.")
        return
    
    latest_feature_file = feature_files[-1]
    logger.info(f"Loading features from {latest_feature_file}")
    tournament.features = pd.read_parquet(latest_feature_file)
    
    # Prepare targets
    tournament.prepare_targets()
    
    # 3. Retrain and generate SHAP/Monitoring for each best model
    for _, row in best_models.iterrows():
        asset = row['asset']
        model_name = row['model']
        
        logger.info(f"Processing {asset} with {model_name}...")
        
        target_key = f'{asset}_return'
        if target_key not in tournament.targets:
            logger.warning(f"Target {target_key} not found, skipping.")
            continue
            
        y = tournament.targets[target_key]
        common_idx = tournament.features.index.intersection(y.dropna().index)
        X = tournament.features.loc[common_idx]
        y = y.loc[common_idx]
        
        # Train model
        model = tournament.train_model(model_name, X, y)
        
        # Save model
        model_path = experiments_dir / 'models' / f'{asset}_{model_name}.pkl'
        joblib.dump(model, model_path)
        logger.info(f"Saved model to {model_path}")
        
        # Compute SHAP
        analyzer = SHAPAnalyzer(n_top_features=10)
        # Determine model type for SHAP
        model_type = 'tree' if any(t in model_name for t in ['forest', 'boost', 'gbm']) else 'linear'
        if 'mlp' in model_name or 'lstm' in model_name: model_type = 'kernel'
        
        logger.info(f"Computing SHAP values (type={model_type})...")
        analyzer.compute_shap_values(model, X, model_type=model_type)
        
        # Save SHAP values
        shap_path = experiments_dir / 'shap' / f'{asset}_{model_name}_shap.npy'
        np.save(shap_path, analyzer.shap_values)
        
        # Save feature names
        names_path = experiments_dir / 'shap' / f'{asset}_{model_name}_features.json'
        with open(names_path, 'w') as f:
            import json
            json.dump(analyzer.feature_names, f)
            
        # Generate monitoring sheet
        logger.info("Generating monitoring sheet...")
        monitoring_sheet = analyzer.generate_monitoring_sheet(X)
        monitor_path = experiments_dir / 'reports' / f'{asset}_monitoring.csv'
        monitoring_sheet.to_csv(monitor_path, index=False)
        logger.info(f"Saved monitoring sheet to {monitor_path}")

    logger.info("Missing results generation complete.")

if __name__ == "__main__":
    main()
