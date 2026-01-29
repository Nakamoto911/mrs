#!/usr/bin/env python3
"""
Model Tournament Runner
=======================
Main entry point for running the model tournament.

Usage:
    python run_tournament.py --assets all --models all
    python run_tournament.py --asset SPX --model xgboost
    python run_tournament.py --features-only
    python run_tournament.py --eval-only
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional
import yaml
import pandas as pd
import numpy as np
import json
import joblib

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from preprocessing import FREDMDLoader, AssetPriceLoader, FREDMDTransformer
from preprocessing import identify_level_stationary_features
from feature_engineering import (
    generate_all_ratio_features,
    generate_all_quintile_features,
    generate_cointegration_features,
    generate_all_momentum_features,
    reduce_features_by_clustering
)
from models import create_linear_model, create_tree_model, create_neural_model
from evaluation import TimeSeriesCV, CrossValidator, compute_all_metrics, SHAPAnalyzer
from evaluation.ensemble import EnsembleEvaluator


# Configure logging
Path('experiments/tournament_logs').mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    handlers=[
        logging.StreamHandler(),
        # logging.FileHandler(f'experiments/tournament_logs/tournament_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    ]
)
logger = logging.getLogger(__name__)


class ModelTournament:
    """
    Orchestrates the complete model tournament.
    """
    
    ASSETS = ['SPX', 'BOND', 'GOLD']
    
    MODEL_CONFIGS = {
        'ridge': {'type': 'linear', 'params': {'alpha': 1.0}},
        'lasso': {'type': 'linear', 'params': {'alpha': 0.0001, 'max_iter': 100000}},
        'elastic_net': {'type': 'linear', 'params': {'alpha': 0.0001, 'l1_ratio': 0.5, 'max_iter': 100000}},
        'random_forest': {'type': 'tree', 'params': {'n_estimators': 20, 'max_depth': 4, 'n_jobs': 1}},
        'xgboost': {'type': 'tree', 'params': {'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.1, 'n_jobs': 1}},
        'lightgbm': {'type': 'tree', 'params': {'n_estimators': 100, 'max_depth': 4, 'learning_rate': 0.1, 'n_jobs': 1}},
        'mlp': {'type': 'neural', 'params': {'hidden_layers': [64, 32], 'epochs': 50}},
    }
    
    def __init__(self, config_path: str = 'configs/experiment_config.yaml'):
        """
        Initialize tournament.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Initialize paths
        self.data_dir = Path('data')
        self.experiments_dir = Path('experiments')
        self._setup_directories()
        
        # Data storage
        self.features: Optional[pd.DataFrame] = None
        self.targets: Dict[str, pd.Series] = {}
        self.results: Dict = {}
    
    def _load_config(self) -> dict:
        """Load configuration file."""
        if self.config_path.exists():
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        return {}
    
    def _setup_directories(self):
        """Create necessary directories."""
        dirs = [
            self.experiments_dir / 'features',
            self.experiments_dir / 'models',
            self.experiments_dir / 'cv_results',
            self.experiments_dir / 'shap',
            self.experiments_dir / 'regimes',
            self.experiments_dir / 'reports',
            self.experiments_dir / 'tournament_logs',
            self.experiments_dir / 'predictions',
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)
    
    def run_feature_pipeline(self, fred_api_key: Optional[str] = None) -> pd.DataFrame:
        """
        Run the complete feature engineering pipeline.
        
        Args:
            fred_api_key: Optional FRED API key
            
        Returns:
            Feature DataFrame
        """
        logger.info("=" * 60)
        logger.info("STEP 1: DATA ACQUISITION & FEATURE ENGINEERING")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # Step 0: Download FRED-MD
        logger.info("Step 0: Loading FRED-MD data...")
        fred_loader = FREDMDLoader(data_dir='data/raw')
        raw_data = fred_loader.download_current_vintage()
        
        # Step 1: Identify stationary levels
        logger.info("Step 1: Identifying stationary level series...")
        stationary_levels = fred_loader.get_stationary_levels()
        
        # Step 2: Apply transformations
        logger.info("Step 2: Applying FRED-MD transformations...")
        transformer = FREDMDTransformer(fred_loader.get_transform_codes())
        transformed = transformer.transform_dataframe(raw_data)
        
        # Step 3: Generate macro ratios
        logger.info("Step 3: Generating macro ratios...")
        ratios = generate_all_ratio_features(raw_data)
        
        # Step 3.5: Generate quintile features
        logger.info("Step 3.5: Generating quintile features...")
        # Combine raw data and ratios for quintile analysis
        quintile_data = pd.concat([raw_data, ratios], axis=1)
        # Remove duplicates
        quintile_data = quintile_data.loc[:, ~quintile_data.columns.duplicated()]
        quintiles = generate_all_quintile_features(quintile_data)
        
        # Step 4: Cointegration analysis
        logger.info("Step 4: Running cointegration analysis...")
        ect_features, coint_summary = generate_cointegration_features(raw_data)
        
        # Step 5: Momentum features
        logger.info("Step 5: Generating momentum features...")
        momentum = generate_all_momentum_features(transformed)
        
        # Combine all features
        logger.info("Step 6: Combining features...")
        all_features = pd.concat([
            transformed,
            ratios,
            quintiles,
            ect_features,
            momentum
        ], axis=1)
        
        # Remove duplicates and clean
        all_features = all_features.loc[:, ~all_features.columns.duplicated()]
        
        logger.info(f"Total features before clustering: {len(all_features.columns)}")
        
        # Step 7: Hierarchical clustering
        logger.info("Step 7: Applying hierarchical clustering...")
        reduced_features, cluster_info = reduce_features_by_clustering(
            all_features,
            similarity_threshold=0.80
        )
        
        logger.info(f"Features after clustering: {len(reduced_features.columns)}")
        
        # Save features
        timestamp = datetime.now().strftime('%Y%m%d')
        reduced_features.to_parquet(self.experiments_dir / 'features' / f'features_{timestamp}.parquet')
        cluster_info.to_csv(self.experiments_dir / 'features' / f'clusters_{timestamp}.csv')
        
        elapsed = time.time() - start_time
        logger.info(f"Feature pipeline completed in {elapsed/60:.1f} minutes")
        
        self.features = reduced_features
        return reduced_features
    
    def prepare_targets(self, horizon_months: int = 24) -> Dict[str, pd.Series]:
        """
        Prepare target variables for each asset.
        
        Args:
            horizon_months: Forecast horizon in months
            
        Returns:
            Dictionary of target Series
        """
        logger.info(f"Preparing targets with {horizon_months}M horizon...")
        
        asset_loader = AssetPriceLoader(data_dir='data/raw')
        
        targets = {}
        
        for asset in self.ASSETS:
            try:
                # Load prices
                prices = asset_loader.load_extended_prices(asset)
                
                # Align dates to Start-of-Month to match FRED-MD features
                prices.index = prices.index.to_period('M').to_timestamp()
                
                # Compute forward returns
                returns = np.log(prices.shift(-horizon_months) / prices)
                targets[f'{asset}_return'] = returns
                
                # Compute forward volatility
                monthly_returns = np.log(prices / prices.shift(1))
                fwd_vol = monthly_returns.shift(-horizon_months).rolling(6).std() * np.sqrt(12)
                targets[f'{asset}_volatility'] = fwd_vol
                
            except Exception as e:
                logger.warning(f"Could not load target for {asset}: {e}")
        
        self.targets = targets
        return targets
    
    def train_model(self, model_name: str, X: pd.DataFrame, y: pd.Series) -> any:
        """
        Train a single model.
        
        Args:
            model_name: Model name
            X: Features
            y: Target
            
        Returns:
            Trained model
        """
        config = self.MODEL_CONFIGS.get(model_name)
        if config is None:
            raise ValueError(f"Unknown model: {model_name}")
        
        model_type = config['type']
        params = config['params']
        
        if model_type == 'linear':
            model = create_linear_model(model_name, **params)
        elif model_type == 'tree':
            model = create_tree_model(model_name, **params)
        elif model_type == 'neural':
            model = create_neural_model(model_name.replace('_', ''), **params)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model.fit(X, y)
        return model
    
    def run_tournament(self, assets: List[str] = None, models: List[str] = None):
        """
        Run the complete model tournament.
        
        Args:
            assets: List of assets to run (all if None)
            models: List of models to run (all if None)
        """
        logger.info("=" * 60)
        logger.info("STEP 2: MODEL TOURNAMENT")
        logger.info("=" * 60)
        
        if assets is None or (len(assets) == 1 and assets[0].lower() == 'all'):
            assets = self.ASSETS
        
        if models is None or (len(models) == 1 and models[0].lower() == 'all'):
            models = list(self.MODEL_CONFIGS.keys())
        
        if self.features is None:
            raise ValueError("Features not loaded. Run feature pipeline first.")
        
        if not self.targets:
            self.prepare_targets()
        
        # Setup cross-validation
        cv = TimeSeriesCV(
            min_train_months=120,
            validation_months=12,
            step_months=6
        )
        validator = CrossValidator(cv)
        
        results = []
        
        for asset in assets:
            logger.info(f"\n{'='*40}")
            logger.info(f"Processing {asset}")
            logger.info(f"{'='*40}")
            
            target_key = f'{asset}_return'
            if target_key not in self.targets:
                logger.warning(f"No target for {asset}, skipping")
                continue
            
            y = self.targets[target_key]
            
            # Aligned features and target (LinearModelWrapper now handles PIT pruning)
            common_idx = self.features.index.intersection(y.dropna().index)
            X = self.features.loc[common_idx]
            y = y.loc[common_idx]
            
            logger.info(f"    Forwarding {len(X.columns)} features and {len(X)} samples to model layer")
            
            for model_name in models:
                logger.info(f"  Training {model_name}...")
                
                try:
                    start = time.time()
                    
                    # Train model
                    model = self.train_model(model_name, X, y)
                    
                    # Cross-validation
                    result = validator.evaluate(
                        model,
                        X, y,
                        model_name=model_name,
                        asset=asset,
                        target='return'
                    )
                    elapsed = time.time() - start
                    logger.info(f"    IC: {result.metrics.get('IC_mean', np.nan):.3f} ± {result.metrics.get('IC_std', np.nan):.3f}")
                    
                    # Store results
                    results.append({
                        'asset': asset,
                        'model': model_name,
                        'IC_mean': result.metrics.get('IC_mean', np.nan),
                        'IC_std': result.metrics.get('IC_std', np.nan),
                        'RMSE_mean': result.metrics.get('RMSE_mean', np.nan),
                        'hit_rate_mean': result.metrics.get('hit_rate_mean', np.nan),
                        'n_folds': result.n_folds,
                        'time_seconds': elapsed
                    })

                    # Logic: Save OOS Predictions
                    if result.predictions is not None:
                        # File naming convention: {Asset}_{ModelName}_preds.csv
                        pred_path = self.experiments_dir / 'predictions' / f'{asset}_{model_name}_preds.csv'
                        preds_to_save = result.predictions.copy()
                        # Ensure no NaN indices (can cause Join issues later)
                        if preds_to_save.index.isna().any():
                            preds_to_save = preds_to_save.loc[preds_to_save.index.notna()]
                        preds_to_save.index.name = 'date'
                        preds_to_save.to_csv(pred_path)
                    
                    # Save model
                    model_path = self.experiments_dir / 'models' / f'{asset}_{model_name}.pkl'
                    joblib.dump(model, model_path)
                    
                    # Store results ONLY after successful save if we want to be strict,
                    # but here we want the IC even if save fails, so we just wrap the dump.
                except Exception as e:
                    logger.error(f"    Error training {model_name}: {e}")
                    # Only append failure if success wasn't already appended
                    already_appended = any(r['asset'] == asset and r['model'] == model_name and not np.isnan(r.get('IC_mean', np.nan)) for r in results)
                    if not already_appended:
                        results.append({
                            'asset': asset,
                            'model': model_name,
                            'IC_mean': np.nan,
                            'error': str(e)
                        })

            # --- Ensemble Strategy Execution (Post-Asset Loop) ---
            try:
                # 1. Filter: Retrieve all successful results for the current asset
                asset_results = [r for r in results if r['asset'] == asset and not np.isnan(r.get('IC_mean', np.nan))]
                
                # 2. Rank: Sort models by IC_mean (descending)
                ranked_results = sorted(asset_results, key=lambda x: x['IC_mean'], reverse=True)
                
                # 3. Select: Identify the top models
                ensemble_size = self.config.get('ensemble', {}).get('size', 5)
                top_models = ranked_results[:ensemble_size]
                
                if len(top_models) >= 2:
                    logger.info(f"  Executing Ensemble (Top {len(top_models)}) for {asset}...")
                    
                    # 4. Filter Prediction Paths
                    pred_paths = []
                    constituent_names = []
                    for r in top_models:
                        m_name = r['model']
                        p_path = self.experiments_dir / 'predictions' / f'{asset}_{m_name}_preds.csv'
                        if p_path.exists():
                            pred_paths.append(p_path)
                            constituent_names.append(m_name)
                    
                    if pred_paths:
                        # 4. Execute Ensemble
                        evaluator = EnsembleEvaluator()
                        ensemble_df = evaluator.load_and_average(pred_paths)
                        metrics = evaluator.compute_ensemble_metrics(ensemble_df)
                        
                        # 6. Save Ensemble Predictions
                        ensemble_name = f"Ensemble_Top{len(top_models)}"
                        ensemble_pred_path = self.experiments_dir / 'predictions' / f'{asset}_{ensemble_name}_preds.csv'
                        ensemble_df.to_csv(ensemble_pred_path)
                        
                        # 7. Register Result
                        results.append({
                            'asset': asset,
                            'model': ensemble_name,
                            'IC_mean': metrics.get('IC_mean', np.nan),
                            'RMSE_mean': metrics.get('RMSE_mean', np.nan),
                            'hit_rate_mean': metrics.get('hit_rate_mean', np.nan),
                            'n_folds': top_models[0]['n_folds'],
                            'time_seconds': 0.0, # Ensemble is fast
                        })
                        
                        # 8. Generate Manifest
                        manifest_path = self.experiments_dir / 'models' / f'{asset}_{ensemble_name}.json'
                        manifest = {
                            "models": constituent_names,
                            "weights": "equal",
                            "timestamp": datetime.now().isoformat()
                        }
                        with open(manifest_path, 'w') as f:
                            json.dump(manifest, f, indent=4)
                        
                        logger.info(f"    Ensemble IC: {metrics.get('IC_mean', np.nan):.3f}")
                
            except Exception as e:
                logger.error(f"  Error creating ensemble for {asset}: {e}")

        # Save results - carefully merging with existing to avoid overwriting other assets
        results_path = self.experiments_dir / 'cv_results' / 'tournament_results.csv'
        results_df = pd.DataFrame(results)
        
        if results_path.exists():
            try:
                existing_df = pd.read_csv(results_path)
                # Filter out old results for the assets we just ran
                # "keep rows where asset is NOT in the current run's assets"
                assets_ran = set(results_df['asset'].unique())
                preserved_df = existing_df[~existing_df['asset'].isin(assets_ran)]
                
                if not preserved_df.empty:
                    results_df = pd.concat([preserved_df, results_df], ignore_index=True)
                    
            except Exception as e:
                logger.error(f"Error merging with existing results: {e}")
                # Fallback: Just save what we have if merge fails (better than crashing)
        
        results_df.to_csv(results_path, index=False)
        
        self.results = results_df
        return results_df
    
    def generate_reports(self):
        """Generate summary reports."""
        logger.info("=" * 60)
        logger.info("GENERATING REPORTS")
        logger.info("=" * 60)
        
        if self.results is None or len(self.results) == 0:
            logger.warning("No results to report")
            return
        
        # Best models per asset
        if self.results['IC_mean'].dropna().empty:
            logger.warning("No valid IC metrics found in results. Skipping best models report.")
            return
            
        best_models = self.results.loc[self.results.groupby('asset')['IC_mean'].idxmax()]
        
        print("\n" + "=" * 60)
        print("TOURNAMENT RESULTS SUMMARY")
        print("=" * 60)
        print("\nBest Model per Asset:")
        print(best_models[['asset', 'model', 'IC_mean', 'hit_rate_mean']].to_string(index=False))
        
        # Save to report
        report_path = self.experiments_dir / 'reports' / f'tournament_summary_{datetime.now().strftime("%Y%m%d")}.txt'
        with open(report_path, 'w') as f:
            f.write("ASSET-SPECIFIC MACRO REGIME DETECTION SYSTEM\n")
            f.write("Tournament Results Summary\n")
            f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n")
            f.write("=" * 60 + "\n\n")
            f.write("Best Model per Asset:\n")
            f.write(best_models.to_string())
        
        logger.info(f"Report saved to {report_path}")
        
        # Generate SHAP monitoring for winners
        logger.info("Generating SHAP monitoring for winners...")
        for _, row in best_models.iterrows():
            asset = row['asset']
            model_name = row['model']
            
            model_path = self.experiments_dir / 'models' / f'{asset}_{model_name}.pkl'
            if model_path.exists():
                try:
                    model = joblib.load(model_path)
                    
                    # Get data (need to prepare it if not available)
                    if self.features is not None:
                        target_key = f'{asset}_return'
                        y = self.targets.get(target_key)
                        if y is not None:
                            common_idx = self.features.index.intersection(y.dropna().index)
                            X = self.features.loc[common_idx]
                            
                            # Compute SHAP
                            analyzer = SHAPAnalyzer(n_top_features=10)
                            model_type = 'tree' if any(t in model_name for t in ['forest', 'boost', 'gbm']) else 'linear'
                            analyzer.compute_shap_values(model, X, model_type=model_type)
                            
                            # Save SHAP
                            np.save(self.experiments_dir / 'shap' / f'{asset}_{model_name}_shap.npy', analyzer.shap_values)
                            
                            # Save Monitoring
                            monitoring_sheet = analyzer.generate_monitoring_sheet(X)
                            monitoring_sheet.to_csv(self.experiments_dir / 'reports' / f'{asset}_monitoring.csv', index=False)
                            logger.info(f"Generated monitoring for {asset}")
                except Exception as e:
                    logger.error(f"Error generating SHAP for {asset}: {e}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Run Model Tournament')
    parser.add_argument('--assets', nargs='+', default=None, help='Assets to process (SPX, BOND, GOLD)')
    parser.add_argument('--models', nargs='+', default=None, help='Models to train')
    parser.add_argument('--features-only', action='store_true', help='Only run feature pipeline')
    parser.add_argument('--eval-only', action='store_true', help='Only run evaluation')
    parser.add_argument('--fred-api-key', type=str, default=None, help='FRED API key')
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("ASSET-SPECIFIC MACRO REGIME DETECTION SYSTEM")
    logger.info("Model Tournament")
    logger.info("=" * 60)
    
    tournament = ModelTournament()
    
    # Run feature pipeline
    feature_file = tournament.experiments_dir / 'features' / f'features_{datetime.now().strftime("%Y%m%d")}.parquet'
    
    if not args.eval_only:
        if feature_file.exists():
            logger.info(f"Features found at {feature_file}. Skipping generation.")
            tournament.features = pd.read_parquet(feature_file)
        else:
            tournament.run_feature_pipeline(args.fred_api_key)
    
    if args.features_only:
        logger.info("Feature pipeline complete. Exiting.")
        return
    
    # Run tournament
    tournament.run_tournament(assets=args.assets, models=args.models)
    
    # Generate reports
    tournament.generate_reports()
    
    logger.info("\n" + "=" * 60)
    logger.info("TOURNAMENT COMPLETE")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
