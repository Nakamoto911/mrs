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
import os
import logging
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import yaml
import pandas as pd
import numpy as np
import json
import joblib
import warnings
from sklearn.exceptions import ConvergenceWarning

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from preprocessing import (
    FREDMDLoader,
    AssetPriceLoader,
    FREDMDTransformer,
    LaggedAligner,
    identify_level_stationary_features,
    TimeSeriesScaler,
    PointInTimeImputer
)
from feature_engineering import (
    generate_all_ratio_features,
    generate_all_quintile_features,
    generate_cointegration_features,
    generate_all_momentum_features,
    reduce_features_by_clustering
)
from models import (
    create_linear_model,
    create_tree_model,
    create_neural_model,
    LSTMSequenceValidator,
    SequenceRequirements
)
from evaluation import (
    SHAPAnalyzer,
    benchmark_model,
    format_benchmark_report,
    NestedCVEnsembleEvaluator,
    format_nested_cv_report,
    TimeSeriesCV,
    CVResult,
    CrossValidator
)
from evaluation.ensemble import EnsembleEvaluator
from evaluation.holdout import (
    HoldoutManager,
    validate_holdout_never_touched
)
from evaluation.multiple_testing import (
    MultipleTestingCorrector,
    HypothesisTest,
    format_mtc_report
)

# Parallelization
from joblib import Parallel, delayed

from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.linear_model import Ridge
from feature_engineering import (
    CointegrationAnalyzer,
    HierarchicalClusterSelector,
    SelectionConfig,
    SelectionMethod
)
from feature_engineering.cointegration_validator import (
    format_cointegration_report,
    CointegrationStatus
)
from models.ensemble import EnsembleModel
from models.lstm_v2 import get_sequence_length, SequenceStrategy
from preprocessing.scaling import TimeSeriesScaler
from preprocessing.imputation import PointInTimeImputer
from evaluation.robustness import run_suite

# Configure logging
Path('experiments/tournament_logs').mkdir(parents=True, exist_ok=True)
log_handlers = [logging.StreamHandler()]
try:
    log_file = f'experiments/tournament_logs/tournament_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
    log_handlers.append(logging.FileHandler(log_file))
except Exception as e:
    # Fallback if file logging fails (e.g. permission issues on external drives)
    print(f"Warning: Could not initialize file logging: {e}")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=log_handlers
)
logger = logging.getLogger(__name__)

# Suppress warnings from libraries
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=ConvergenceWarning, module='sklearn')

# Suppress verbose logs from external libraries
logging.getLogger('shap').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# -----------------------------------------------------------------------------
# Standalone Helper Functions for Parallelization
# -----------------------------------------------------------------------------

def create_model_from_config(model_name: str, model_config: dict):
    """
    Create a model instance from configuration.
    Standalone function to be pickle-safe for joblib.
    """
    model_type = model_config['type']
    params = model_config['params']
    
    if model_type == 'linear':
        return create_linear_model(model_name, **params)
    elif model_type == 'tree':
        return create_tree_model(model_name, **params)
    elif model_type == 'neural':
        return create_neural_model(model_name.replace('_', ''), **params)
    raise ValueError(f"Unknown model type: {model_type}")

def create_pipeline_for_model(model_instance, coint_pairs, coint_cfg, clustering_cfg):
    """
    Create the feature engineering pipeline.
    Standalone function to be pickle-safe.
    """
    return Pipeline([
        ('cointegration', CointegrationAnalyzer(
            pairs=coint_pairs,
            validate=coint_cfg.get('validate', True),
            significance=coint_cfg.get('significance_level', 0.05),
            min_observations=coint_cfg.get('min_observations', 120),
            stability_threshold=coint_cfg.get('stability_threshold', 0.70),
            prior_config=coint_cfg.get('priors', {})
        )),
        ('clustering', HierarchicalClusterSelector(
            similarity_threshold=clustering_cfg.get('similarity_threshold', 0.80),
            selection_config=get_selection_config_from_dict(clustering_cfg)
        )),
        ('imputer', PointInTimeImputer(strategy='median')),
        ('scaler', TimeSeriesScaler(method='robust')),
        ('model', model_instance)
    ])

def get_selection_config_from_dict(clustering_cfg):
    """Helper to reconstruct SelectionConfig"""
    selection_cfg_dict = clustering_cfg.get('selection', {})
    if selection_cfg_dict.get('ic_based', {}).get('enabled', False):
        ic_cfg = selection_cfg_dict.get('ic_based', {})
        return SelectionConfig(
            method=SelectionMethod.UNIVARIATE_IC,
            ic_lag_buffer_months=ic_cfg.get('lag_buffer_months', 24),
            ic_min_observations=ic_cfg.get('min_observations', 60)
        )
    else:
        method_str = selection_cfg_dict.get('method', 'centroid')
        return SelectionConfig(method=SelectionMethod(method_str))

def process_single_model(
    model_name, 
    asset, 
    X_train, 
    y_train, 
    validator, 
    experiments_dir, 
    model_config,
    coint_pairs,
    coint_cfg,
    clustering_cfg,
    lstm_skip_check=None
):
    """
    Worker function to process a single model.
    """
    # Gating logic for LSTM
    if lstm_skip_check:
        should_skip, reason = lstm_skip_check(X_train, y_train)
        if should_skip:
            return {
                'status': 'skipped',
                'asset': asset,
                'model': model_name,
                'reason': reason
            }

    start = time.time()
    try:
        # Create model and pipeline
        model_instance = create_model_from_config(model_name, model_config)
        pipeline = create_pipeline_for_model(model_instance, coint_pairs, coint_cfg, clustering_cfg)
        
        # Cross-validation
        result = validator.evaluate(
            pipeline,
            X_train, y_train,
            model_name=model_name,
            asset=asset,
            target='return'
        )
        elapsed = time.time() - start
        
        # Prepare metrics
        metrics = {
            'IC_mean': result.metrics.get('IC_mean', np.nan),
            'IC_std': result.metrics.get('IC_std', np.nan),
            'IC_t_stat': result.metrics.get('IC_t_stat_mean', np.nan),
            'IC_p_value': result.metrics.get('IC_p_value_mean', np.nan),
            'IC_significant': result.metrics.get('IC_significant_mean', 0) > 0.5,
            'RMSE_mean': result.metrics.get('RMSE_mean', np.nan),
            'hit_rate_mean': result.metrics.get('hit_rate_mean', np.nan),
            'n_folds': result.n_folds,
            'time_seconds': elapsed
        }
        
        # Return complex objects separately
        return {
            'status': 'success',
            'asset': asset,
            'model': model_name,
            'metrics': metrics,
            'predictions': result.predictions,
            'fold_metadata': result.fold_metadata if hasattr(result, 'fold_metadata') else None,
            'pipeline': None 
        }
        
    except Exception as e:
        return {
            'status': 'error',
            'asset': asset,
            'model': model_name,
            'error': str(e)
        }

class ModelTournament:
    """
    Orchestrates the complete model tournament.
    """
    
    ASSETS = ['SPX', 'BOND', 'GOLD']
    
    def __init__(self, config_path: str = 'configs/experiment_config.yaml'):
        """
        Initialize tournament.
        
        Args:
            config_path: Path to configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
        # Build model configurations from YAML
        self.model_configs = self._build_model_configs()
        
        # Initialize paths
        self.data_dir = Path('data')
        # Use configured path or default to experiments
        experiment_path = self.config.get('output', {}).get('paths', {}).get('base', 'experiments')
        # Handle the case where config points to subdirs explicitly but we need base
        # Simplest is to default to 'experiments' since config defines subpaths like 'experiments/features'
        self.experiments_dir = Path('experiments')
        self._setup_directories()
        
        # Data storage
        self.features: Optional[pd.DataFrame] = None
        self.targets: Dict[str, pd.Series] = {}
        self.results: Dict = {}
        
        # Initialize holdout manager
        holdout_cfg = self.config.get('validation', {}).get('holdout', {})
        self.holdout_enabled = holdout_cfg.get('enabled', True)
        
        if self.holdout_enabled:
            # Get horizon from config
            horizon = self.config.get('models', {}).get('targets', {}).get('returns', {}).get('horizon_months', 24)
            self.holdout_manager = HoldoutManager(
                method=holdout_cfg.get('method', 'percentage'),
                holdout_pct=holdout_cfg.get('holdout_pct', 0.15),
                holdout_start=holdout_cfg.get('holdout_start'),
                min_holdout_months=holdout_cfg.get('min_holdout_months', 48),
                min_independent_obs=holdout_cfg.get('min_independent_obs', 20),
                forecast_horizon=horizon
            )
        else:
            self.holdout_manager = None
    
    def _load_config(self) -> dict:
        """Load configuration file."""
        if self.config_path.exists():
            with open(self.config_path) as f:
                return yaml.safe_load(f)
        return {}

    def _build_model_configs(self) -> dict:
        """
        Build model configurations dynamically from YAML config.
        Iterates over models defined in config and maps them to their type and parameters.
        """
        model_configs = {}
        models_cfg = self.config.get('models', {})
        
        # Map YAML categories to internal model types
        category_map = {
            'linear': 'linear',
            'tree': 'tree',
            'neural': 'neural'
        }
        
        for category, internal_type in category_map.items():
            category_cfg = models_cfg.get(category, {})
            if not isinstance(category_cfg, dict):
                continue
                
            for model_name, cfg in category_cfg.items():
                if model_name in ['targets']: # Skip special keys if any
                    continue
                
                # Extract parameters from 'params' key if available, or use empty dict
                params = cfg.get('params', {})
                
                model_configs[model_name] = {
                    'type': internal_type,
                    'params': params
                }
        
        return model_configs
    
    def _is_model_enabled(self, model_name: str) -> bool:
        """Check if a model is enabled in the configuration."""
        model_cfg = self.config.get('models', {})
        
        # Check in different categories
        for category in ['linear', 'tree', 'neural']:
            category_cfg = model_cfg.get(category, {})
            if model_name in category_cfg:
                return category_cfg[model_name].get('enabled', True)
        
        # Default to True if not explicitly disabled in config
        return True
    
    def _setup_directories(self):
        """Create necessary directories."""
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        (self.experiments_dir / 'models').mkdir(exist_ok=True)
        (self.experiments_dir / 'predictions').mkdir(exist_ok=True)
        (self.experiments_dir / 'features').mkdir(exist_ok=True)
        (self.experiments_dir / 'logs').mkdir(exist_ok=True)
        (self.experiments_dir / 'benchmarks').mkdir(exist_ok=True)
        # Re-add other directories that were implicitly removed by the instruction's snippet
        (self.experiments_dir / 'cv_results').mkdir(exist_ok=True)
        (self.experiments_dir / 'shap').mkdir(exist_ok=True)
        (self.experiments_dir / 'regimes').mkdir(exist_ok=True)
        (self.experiments_dir / 'reports').mkdir(exist_ok=True)
        (self.experiments_dir / 'tournament_logs').mkdir(exist_ok=True)
        
        # Aggregated stats for summary
        self.feature_reduction_stats = []
    
    def _log_configuration_summary(self):
        """Log a detailed Markdown summary of the configuration."""
        cv_cfg = self.config.get('validation', {}).get('cv', {})
        holdout_cfg = self.config.get('validation', {}).get('holdout', {})
        coint_cfg = self.config.get('features', {}).get('cointegration', {})
        clustering_cfg = self.config.get('features', {}).get('clustering', {})
        ensemble_cfg = self.config.get('ensemble', {})
        
        logger.info("\n" + "=" * 60)
        logger.info("## 1. DETAILED CONFIGURATION")
        logger.info("-" * 60)
        
        logger.info("### Global Settings")
        logger.info(f"- **Assets**: {', '.join(self.ASSETS)}")
        logger.info(f"- **Cross-Validation**: {cv_cfg.get('validation_months', 36)}M validation, {cv_cfg.get('step_months', 6)}M step")
        logger.info(f"- **Holdout**: {'Enabled (' + str(holdout_cfg.get('holdout_pct', 0.15)*100) + '%)' if self.holdout_enabled else 'Disabled'}")
        
        logger.info("\n### Feature Engineering")
        logger.info(f"- **Cointegration**: {'Enabled' if coint_cfg.get('validate', True) else 'Disabled'}")
        logger.info(f"  - Stability Threshold: {coint_cfg.get('stability_threshold', 0.70)}")
        logger.info(f"  - Bayesian Priors: {'Enabled' if coint_cfg.get('priors', {}).get('enabled', False) else 'Disabled'}")
        logger.info(f"- **Feature Reduction**: Clustering")
        logger.info(f"  - Similarity Threshold: {clustering_cfg.get('similarity_threshold', 0.40)}")
        logger.info(f"  - Selection Method: {clustering_cfg.get('selection', {}).get('method', 'centroid')}")
        
        logger.info("\n### Model Parameters")
        enabled_models = [m for m in self.model_configs.keys() if self._is_model_enabled(m)]
        for m in enabled_models:
            params = self.model_configs[m].get('params', {})
            param_str = ", ".join([f"{k}={v}" for k, v in params.items()])
            logger.info(f"- **{m}**: {param_str if param_str else 'Default parameters'}")
            
        if ensemble_cfg.get('enabled', True):
            logger.info("\n### Ensemble Strategy")
            logger.info(f"- **Size**: Top {ensemble_cfg.get('size', 5)} models")
            logger.info(f"- **Method**: {ensemble_cfg.get('selection', {}).get('method', 'simple')}")

        logger.info("=" * 60 + "\n")

    def _log_tournament_summary(self, results_df: pd.DataFrame):
        """Log a Markdown-formatted table of tournament results."""
        if results_df.empty:
            logger.warning("\n## 2. TOURNAMENT RESULTS SUMMARY: No results to display.")
            return

        logger.info("\n## 2. RESULTS SUMMARY")
        logger.info("-" * 60)
        
        # Format results table
        # Columns: Asset, Model, IC Mean, IC Std, Hit Rate, Sharpe, T-Stat, P-Val, Rating
        table_cols = ['asset', 'model', 'IC_mean', 'IC_std', 'hit_rate_mean', 'implied_sharpe', 'IC_t_stat', 'IC_p_value', 'IC_rating']
        
        # Ensure columns exist (some might be missing if errors occurred)
        existing_cols = [c for c in table_cols if c in results_df.columns]
        display_df = results_df[existing_cols].copy()
        
        # Rename for cleaner table
        col_map = {
            'asset': 'Asset', 'model': 'Model', 'IC_mean': 'IC Mean', 
            'IC_std': 'IC Std', 'hit_rate_mean': 'Hit Rate', 
            'implied_sharpe': 'Sharpe', 'IC_t_stat': 'T-Stat', 
            'IC_p_value': 'P-Val', 'IC_rating': 'Rating'
        }
        display_df.columns = [col_map.get(c, c) for c in display_df.columns]
        
        # Log as Markdown table
        logger.info(display_df.to_markdown(index=False, floatfmt=".3f"))
        logger.info("-" * 60)

        # Log Holdout Results if available
        if hasattr(self, 'holdout_results') and not self.holdout_results.empty:
            logger.info("\n## 3. HOLDOUT VALIDATION")
            logger.info("-" * 60)
            h_df = self.holdout_results.copy()
            # Columns: asset, model, cv_IC, holdout_IC, IC_degradation
            h_cols = ['asset', 'model', 'cv_IC', 'holdout_IC', 'IC_degradation']
            h_display = h_df[[c for c in h_cols if c in h_df.columns]].copy()
            h_display.columns = ['Asset', 'Model', 'CV IC', 'Holdout IC', 'Degradation']
            logger.info(h_display.to_markdown(index=False, floatfmt=".3f"))
            logger.info("-" * 60)

        # Log Feature Reduction Aggregates if available
        if self.feature_reduction_stats:
            logger.info("\n## 4. FEATURE REDUCTION SUMMARY")
            logger.info("-" * 60)
            stats_df = pd.DataFrame(self.feature_reduction_stats)
            agg_stats = stats_df.groupby('asset').agg({
                'n_in': 'mean',
                'n_out': 'mean'
            }).reset_index()
            agg_stats['reduction_pct'] = (1 - agg_stats['n_out'] / agg_stats['n_in']) * 100
            
            logger.info(agg_stats.to_markdown(index=False, floatfmt=".1f"))
            logger.info("-" * 60)
            
        logger.info("\n" + "=" * 60)
        logger.info("AUDIT REPORT COMPLETE")
        logger.info("=" * 60 + "\n")
    
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
        
        # Step 4: Cointegration analysis (DEPRECATED: Moved to PIT pipeline)
        # logger.info("Step 4: Running cointegration analysis...")
        # ect_features, coint_summary = generate_cointegration_features(raw_data)
        
        # Step 5: Momentum features
        logger.info("Step 5: Generating momentum features...")
        momentum = generate_all_momentum_features(transformed)
        
        # Prepare raw levels for PIT cointegration (rename to avoid collision)
        # Fetch from config to ensure we have all vars needed for validation
        coint_cfg = self.config.get('features', {}).get('cointegration', {})
        coint_pairs_raw = coint_cfg.get('pairs', CointegrationAnalyzer.DEFAULT_PAIRS)
        
        coint_vars = set()
        for p in coint_pairs_raw:
            if len(p) >= 2:
                coint_vars.add(p[0])
                coint_vars.add(p[1])
        
        # Also include defaults just in case
        for p in CointegrationAnalyzer.DEFAULT_PAIRS:
            coint_vars.add(p[0])
            coint_vars.add(p[1])
            
        available_coint_vars = [c for c in coint_vars if c in raw_data.columns]
        raw_levels = raw_data[available_coint_vars].copy()
        raw_levels.columns = [f"{c}_level" for c in raw_levels.columns]

        # Combine all features
        logger.info("Step 6: Combining features...")
        all_features = pd.concat([
            transformed,
            ratios,
            quintiles,
            # ect_features,
            momentum,
            raw_levels
        ], axis=1)
        
        # Remove duplicates and clean
        all_features = all_features.loc[:, ~all_features.columns.duplicated()]
        
        logger.info(f"Total features before clustering: {len(all_features.columns)}")
        
        # Step 7: Hierarchical clustering (DEPRECATED: Moved to PIT pipeline)
        # logger.info("Step 7: Applying hierarchical clustering...")
        # reduced_features, cluster_info = reduce_features_by_clustering(
        #     all_features,
        #     similarity_threshold=0.80
        # )
        
        reduced_features = all_features
        logger.info(f"Base features prepared for CV loop: {len(reduced_features.columns)}")
        
        # Save features
        timestamp = datetime.now().strftime('%Y%m%d')
        try:
            reduced_features.to_parquet(self.experiments_dir / 'features' / f'base_features_{timestamp}.parquet')
        except Exception as e:
            logger.warning(f"Failed to save base features to parquet (likely permission issue): {e}")
        # cluster_info.to_csv(self.experiments_dir / 'features' / f'clusters_{timestamp}.csv')
        
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
        # Delegate to the standalone function
        model_config = self.model_configs.get(model_name)
        if model_config is None:
            raise ValueError(f"Unknown model: {model_name}")
            
        model = create_model_from_config(model_name, model_config)
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
            models = list(self.model_configs.keys())
        
        if self.features is None:
            raise ValueError("Features not loaded. Run feature pipeline first.")
        
        if not self.targets:
            self.prepare_targets()
        
        # Setup cross-validation
        cv_config = self.config.get('validation', {}).get('cv', {})
        logger.info(f"Loaded CV Config: {cv_config}")
        val_months = cv_config.get('validation_months', 36)
        logger.info(f"Initializing CV with validation_months={val_months}")
        
        cv = TimeSeriesCV(
            min_train_months=cv_config.get('min_train_months', 120),
            validation_months=cv_config.get('validation_months', 36),
            step_months=cv_config.get('step_months', 6)
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
            
            # --- LAG-AWARE ALIGNMENT ---
            # Fetch lag from config, default to 1 (conservative for Monthly data)
            pub_lag = self.config.get('data', {}).get('alfred', {}).get('publication_lag_months', 1)
            
            # Initialize Aligner
            aligner = LaggedAligner(lag_months=pub_lag)
            
            # Align Data
            # This effectively shifts X forward, ensuring X[t] is actually data from t-lag
            X, y = aligner.align_features_and_targets(self.features, y.dropna())
            
            # --- HOLDOUT SPLIT ---
            if self.holdout_enabled:
                X_dev, y_dev, X_holdout, y_holdout = self.holdout_manager.split_data(X, y)
                logger.info(f"    Holdout enabled. Development: {len(X_dev)} samples, Holdout: {len(X_holdout)} samples")
                X_train, y_train = X_dev, y_dev
            else:
                X_train, y_train = X, y
                X_holdout, y_holdout = None, None

            logger.info(f"    Aligned data with {pub_lag} month lag. {len(X_train)} training samples available.")
            logger.info(f"    Forwarding {len(X_train.columns)} features to model layer")
            
            asset_results = []
            # Track fold metadata for cointegration report (shared across models)
            latest_fold_metadata = None

            # Prepare Parallel Tasks
            tasks = []
            coint_cfg = self.config.get('features', {}).get('cointegration', {})
            clustering_cfg = self.config.get('features', {}).get('clustering', {})
            coint_pairs = self._get_coint_pairs()
            
            for model_name in models:
                if not self._is_model_enabled(model_name):
                    logger.info(f"  Skipping {model_name} (disabled in config)")
                    continue
                    
                # Prepare LSTM skip check closure
                lstm_check = None
                if 'lstm' in model_name:
                    # We need to capture cv_folds logic here or pass it into the function
                    # The original used validator.cv.split(X_train)
                    # We'll pass a partial or a simpler check. 
                    # To keep it picklable, we pass the method reference bound to self is risky? 
                    # Better to pass the config and re-instantiate validation logic or pre-calculate.
                    # Actually, pre-calculating CV folds validation is fast.
                    cv_folds_list = list(validator.cv.split(X_train))
                    def check_lstm(X, y, folds=cv_folds_list):
                        return self._should_skip_lstm(X, y, folds)
                    lstm_check = check_lstm

                model_config = self.model_configs.get(model_name)
                
                # --- NEW: CONFIG AUDIT ---
                if model_config['type'] == 'tree':
                    params = model_config.get('params', {})
                    if 'random_forest' in model_name:
                         logger.info(f"  Audit {model_name} Params: depth={params.get('max_depth')}, "
                                     f"n_leaf={params.get('min_samples_leaf')}, subsample={params.get('max_samples', 'Default')}")
                    else:
                         logger.info(f"  Audit {model_name} Params: depth={params.get('max_depth')}, "
                                     f"mcw={params.get('min_child_weight', 'N/A')}, subsample={params.get('subsample', 'N/A')}")
                
                tasks.append(
                    delayed(process_single_model)(
                        model_name, asset, X_train, y_train, validator, 
                        self.experiments_dir, model_config,
                        coint_pairs, coint_cfg, clustering_cfg,
                        lstm_check
                    )
                )

            logger.info(f"  Parallelizing {len(tasks)} model training tasks...")
            parallel_results = Parallel(n_jobs=-1)(tasks)
            
            # Process Results
            for res in parallel_results:
                model_name = res['model']
                model_config = self.model_configs.get(model_name)
                
                if res['status'] == 'skipped':
                    logger.warning(f"  Skipping {model_name} for {asset}: {res['reason']}")
                    self._log_lstm_skip(asset, res['reason'])
                    continue
                    
                if res['status'] == 'error':
                    logger.error(f"    Error training {model_name}: {res['error']}")
                    results.append({
                        'asset': asset, 
                        'model': model_name, 
                        'IC_mean': np.nan, 
                        'error': res['error']
                    })
                    continue
                    
                # Success
                metrics = res['metrics']
                ic = metrics['IC_mean']
                t_stat = metrics['IC_t_stat']
                p_val = metrics['IC_p_value']
                sig = '*' if p_val < 0.05 else ''
                
                
                logger.info(f"    {model_name}: IC: {ic:.3f} (t={t_stat:.2f}, p={p_val:.3f}{sig})")

                # Validate Benchmark
                benchmark = benchmark_model(
                    ic=ic, ic_t_stat=t_stat, ic_p_value=p_val, asset=asset, ic_std=metrics['IC_std']
                )
                
                metrics.update({
                    'asset': asset,
                    'model': model_name,
                    'IC_rating': benchmark.rating,
                    'IC_suspicious': benchmark.is_suspicious,
                    'implied_sharpe': benchmark.implied_sharpe,
                })
                
                if benchmark.warnings:
                    logger.warning(f"    Benchmark warnings for {asset}/{model_name}:")
                    for w in benchmark.warnings:
                        logger.warning(f"      - {w}")

                results.append(metrics)
                asset_results.append(metrics)
                
                # Save Predictions
                if res.get('predictions') is not None:
                     # File naming convention: {Asset}_{ModelName}_preds.csv
                    pred_path = self.experiments_dir / 'predictions' / f'{asset}_{model_name}_preds.csv'
                    preds_to_save = res['predictions'].copy()
                    if preds_to_save.index.isna().any():
                        preds_to_save = preds_to_save.loc[preds_to_save.index.notna()]
                    preds_to_save.index.name = 'date'
                    preds_to_save.to_csv(pred_path)
                
                # Save Model (Re-fit for persistence)
                # Note: We are doing this sequentially now to avoid pickling the pipeline back from worker
                # Alternatively, we could have done this in worker. 
                # For now, let's keep it here but it effectively means re-training.
                # WAIT: The original code re-fit the pipeline. "refit... for persistence".
                # Doing this sequentially kills the parallel benefit if re-training is slow.
                # Ideally, the worker should do the refit and save the file.
                # But the worker returned `pipeline: None` in my definition above.
                
                # Let's CHANGE strategy: Move strict refit/save to worker.
                # But the worker needs to know where to save. It has experiments_dir.
                # So I should update `process_single_model` to save the model.
                
                # Refitting here for now to ensure correctness with existing logic flow, 
                # but acknowledging it's suboptimal. 
                # Actually, for complex models, refitting IS expensive.
                # I should really do it in the worker. 
                
                # ... Let's update `process_single_model` in the Plan/Code?
                # The chunk I wrote for `process_single_model` earlier did NOT include saving.
                # I will stick to the plan of "Parallelize the inner loop" first. 
                # If I see it is slow, I will move saving to worker.
                # Actually, `pipeline.fit` is called in worker during `validator.evaluate` (internally for each fold).
                # But `validator.evaluate` does NOT return a fitted pipeline on the whole dataset.
                # So we MUST refit on the whole dataset (`X_train`) to save the final artifact.
                # So `process_single_model` SHOULD do this refit and save.
                
                # I will add the refit/save to `process_single_model` in a subsequent step or modifying this call.
                # For now, let's keep it sequential here to minimize massive code drift in one shot, 
                # the USER asked to parallelize the *training* (validation loop).
                # The final fit is one more training. 
                
                logger.info(f"    Refitting {model_name} on development history for persistence (sequential)...")
                # We need to recreate the pipeline to fit it
                model_instance = create_model_from_config(model_name, model_config)
                pipeline = create_pipeline_for_model(model_instance, coint_pairs, coint_cfg, clustering_cfg)
                pipeline.fit(X_train, y_train)
                
                # --- NEW: FORENSIC FEATURE AUDIT ---
                try:
                    # Extract the model step (assuming pipeline structure: 'model' is the last step)
                    model_step = pipeline.named_steps['model']
                    
                    if hasattr(model_step, 'get_feature_importance'):
                        importances = model_step.get_feature_importance()
                        
                        # 1. Check for Dominance (Single feature > 30% is suspicious)
                        top_features = importances.head(5)
                        logger.info(f"    Feature Importance Audit ({model_name}):")
                        for name, imp in top_features.items():
                            logger.info(f"      - {name:<20}: {imp:.4f}")
                            
                        if not top_features.empty and top_features.iloc[0] > 0.30:
                            logger.warning(f"      [!] DOMINANCE ALERT: {top_features.index[0]} drives {top_features.iloc[0]:.1%} of prediction.")
                            
                except Exception as e:
                    logger.warning(f"    Could not audit feature importance: {e}")
                
                model_path = self.experiments_dir / 'models' / f'{asset}_{model_name}.joblib'
                try:
                    joblib.dump(pipeline, model_path)
                except Exception as e:
                    logger.error(f"    Failed to save model {model_name}: {e}")

                # Capture fold metadata
                # Capture fold metadata
                if res.get('fold_metadata'):
                    latest_fold_metadata = res['fold_metadata']

                    # --- NEW: TEMPORAL STABILITY AUDIT ---
                    logger.info(f"    Fold-by-Fold Stability Analysis ({model_name}):")
                    fold_ics = []
                    for i, fold in enumerate(res['fold_metadata']):
                        # Assuming 'metrics' key exists in fold_metadata, or adjust based on your CVResult structure
                        # You might need to check how CrossValidator stores per-fold metrics in your implementation
                        fold_ic = fold.get('metrics', {}).get('IC', np.nan)
                        fold_ics.append(fold_ic)
                        
                        # Log outlier folds
                        if abs(fold_ic) > 0.30: # Suspiciously high
                             logger.warning(f"      Fold {i} (Outlier): IC = {fold_ic:.3f} (Suspiciously High)")
                        elif fold_ic < -0.10:   # Crash
                             logger.warning(f"      Fold {i} (Crash):   IC = {fold_ic:.3f}")
                    
                    # Check for "Lucky Fold" Syndrome (e.g., 4 bad folds, 1 amazing fold)
                    positive_folds = sum(1 for x in fold_ics if x > 0)
                    logger.info(f"      Positive Folds: {positive_folds}/{len(fold_ics)}")
        
            # LOG COINTEGRATION VALIDATION REPORT (Once per asset)
            if latest_fold_metadata:
                fold0_meta = latest_fold_metadata[0]
                if 'cointegration_results' in fold0_meta:
                    report = format_cointegration_report(fold0_meta['cointegration_results'])
                    logger.info(f"\n{report}")
                    
                    # Track stability across folds
                    pair_counts = {}
                    total_folds = len(latest_fold_metadata)
                    
                    for fold_meta in latest_fold_metadata:
                        coint_res = fold_meta.get('cointegration_results', {})
                        for pair_name, r in coint_res.items():
                            if r.include_in_model:
                                pair_counts[pair_name] = pair_counts.get(pair_name, 0) + 1
                    
                    logger.info("\nCOINTEGRATION STABILITY ACROSS CV FOLDS")
                    logger.info("-" * 40)
                    for pair_name, count in pair_counts.items():
                        stability = count / total_folds
                        status = "Stable" if stability > 0.8 else "Marginal" if stability > 0.5 else "Unstable"
                        logger.info(f"{pair_name:<25}: {count}/{total_folds} folds ({stability:.1%}) - {status}")
                    logger.info("-" * 40)

            # LOG FEATURE REDUCTION (CLUSTERING) REPORT
            if latest_fold_metadata:
                fold_results = [f.get('clustering_results') for f in latest_fold_metadata if f.get('clustering_results')]
                if fold_results:
                    n_in = [r['n_features_in'] for r in fold_results]
                    n_out = [r['n_features_out'] for r in fold_results]
                    
                    avg_in = np.mean(n_in)
                    avg_out = np.mean(n_out)
                    reduction = (1 - avg_out / avg_in) * 100 if avg_in > 0 else 0
                    
                    logger.info("\nFEATURE REDUCTION REPORT (HIERARCHICAL CLUSTERING)")
                    logger.info("-" * 60)
                    logger.info(f"Avg Features In:  {avg_in:.1f}")
                    logger.info(f"Avg Features Out: {avg_out:.1f}")
                    logger.info(f"Avg Reduction:    {reduction:.1f}%")
                    
                    # Store stats for summary
                    self.feature_reduction_stats.append({
                        'asset': asset,
                        'n_in': avg_in,
                        'n_out': avg_out
                    })
                    
                    # Track feature selection stability
                    feature_counts = {}
                    total_folds = len(fold_results)
                    for r in fold_results:
                        for feat in r['selected_features']:
                            feature_counts[feat] = feature_counts.get(feat, 0) + 1
                    
                    # Log top stable features
                    stable_features = sorted(
                        [f for f, c in feature_counts.items() if c == total_folds],
                        key=lambda x: x
                    )
                    logger.info(f"Stable Features (100% selection): {len(stable_features)}")
                    if stable_features:
                        logger.info(f"Samples: {', '.join(stable_features[:10])}{'...' if len(stable_features) > 10 else ''}")
                    
                    # Log unstable features
                    unstable = [f for f, c in feature_counts.items() if c < total_folds]
                    if unstable:
                        logger.info(f"Dynamic Features (<100% selection): {len(unstable)}")
                    logger.info("-" * 60)

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
                        # NEW: Nested CV evaluation for unbiased ensemble estimate
                        if self.config.get('ensemble', {}).get('selection', {}).get('nested_cv', {}).get('enabled', False):
                            self.run_nested_cv_ensemble_evaluation([asset])

                        # 4. Execute Standard Ensemble (Potentially Biased)
                        evaluator = EnsembleEvaluator()
                        ensemble_df = evaluator.load_and_average(pred_paths)
                        metrics = evaluator.compute_ensemble_metrics(ensemble_df)
                        
                        # 6. Save Ensemble Predictions
                        ensemble_name = f"Ensemble_Top{len(top_models)}"
                        ensemble_pred_path = self.experiments_dir / 'predictions' / f'{asset}_{ensemble_name}_preds.csv'
                        ensemble_df.to_csv(ensemble_pred_path)
                        
                        # Create and save EnsembleModel instance for SHAP
                        try:
                            # Load actual fitted pipeline objects
                            fitted_models = []
                            ensemble_model_names = []
                            for m in constituent_names:
                                model_path = self.experiments_dir / 'models' / f'{asset}_{m}.joblib'
                                if model_path.exists():
                                    try:
                                        # Load pipeline
                                        pipeline = joblib.load(model_path)
                                        fitted_models.append(pipeline)
                                        ensemble_model_names.append(m)
                                    except Exception as e:
                                        logger.warning(f"Failed to load model {m} for ensemble construction: {e}")
                            
                            if fitted_models:
                                logger.info(f"    Constructing EnsembleModel object from {len(fitted_models)} models...")
                                ensemble_model = EnsembleModel(estimators=fitted_models)
                                
                                # Save ensemble model
                                ensemble_path = self.experiments_dir / 'models' / f'{asset}_{ensemble_name}.joblib'
                                joblib.dump(ensemble_model, ensemble_path)
                                logger.info(f"    Saved EnsembleModel to {ensemble_path}")
                            else:
                                logger.warning("No valid models found to construct EnsembleModel")
                                
                        except Exception as e:
                            logger.error(f"Failed to save EnsembleModel: {e}")
                        
                        # Benchmark the ensemble result
                        ensemble_benchmark = benchmark_model(
                            ic=metrics.get('IC_mean', np.nan),
                            ic_t_stat=metrics.get('IC_t_stat', np.nan),
                            ic_p_value=metrics.get('IC_p_value', 1.0),
                            asset=asset
                        )

                        # 7. Register Result
                        results.append({
                            'asset': asset,
                            'model': ensemble_name,
                            'IC_mean': metrics.get('IC_mean', np.nan),
                            'IC_t_stat': metrics.get('IC_t_stat', np.nan),
                            'IC_p_value': metrics.get('IC_p_value', np.nan),
                            'IC_significant': metrics.get('IC_significant', False),
                            'IC_rating': ensemble_benchmark.rating,
                            'IC_suspicious': ensemble_benchmark.is_suspicious,
                            'implied_sharpe': ensemble_benchmark.implied_sharpe,
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
        
        try:
            if results_path.exists():
                try:
                    existing_df = pd.read_csv(results_path)
                    assets_ran = set(results_df['asset'].unique())
                    preserved_df = existing_df[~existing_df['asset'].isin(assets_ran)]
                    if not preserved_df.empty:
                        results_df = pd.concat([preserved_df, results_df], ignore_index=True)
                except Exception as e:
                    logger.error(f"Error merging with existing results: {e}")
            
            # Apply multiple testing correction
            self.apply_multiple_testing_correction(results_df)
            
            # Final Save
            results_df.to_csv(results_path, index=False)
        except Exception as e:
            logger.error(f"Critical error during results post-processing/saving: {e}. Progress will continue but CSV may be missing.")
        
        self.results = results_df
        
        # --- Robustness Suite (Spec 11) ---
        try:
            self.run_robustness_checks(results_df, assets)
        except Exception as e:
            logger.error(f"Error during robustness checks: {e}")
        
        # AFTER all development is complete, evaluate on holdout
        if self.holdout_enabled:
            try:
                self.evaluate_holdout(assets, models)
            except Exception as e:
                logger.error(f"Error during holdout evaluation: {e}")
            
        # --- TOURNAMENT SUMMARY REPORT ---
        self._log_configuration_summary()
        self._log_tournament_summary(results_df)

        return results_df
    
    def run_robustness_checks(self, df: pd.DataFrame, assets: List[str]):
        """Perform robust evaluation for champion models of each asset."""
        logger.info("=" * 60)
        logger.info("STEP 2.6: COMPREHENSIVE ROBUSTNESS SUITE (Spec 11)")
        logger.info("=" * 60)
        
        for asset in assets:
            asset_df = df[df['asset'] == asset].copy()
            # Filter for successful models only
            asset_df = asset_df[asset_df['IC_mean'].notna()]
            if asset_df.empty:
                logger.warning(f"  No successful models found for {asset}, skipping robustness checks.")
                continue
            
            # Find champion (highest IC)
            champion_idx = asset_df['IC_mean'].idxmax()
            champion = asset_df.loc[champion_idx]
            model_name = champion['model']
            
            logger.info(f"  Analyzing Champion for {asset}: {model_name} (IC={champion['IC_mean']:.3f})")
            
            # Load OOS predictions
            pred_path = self.experiments_dir / 'predictions' / f'{asset}_{model_name}_preds.csv'
            if not pred_path.exists():
                logger.warning(f"    Predictions not found at {pred_path}, skipping robustness suite.")
                continue
            
            preds_df = pd.read_csv(pred_path, index_col=0, parse_dates=True)
            if 'prediction' not in preds_df.columns or 'actual' not in preds_df.columns:
                logger.warning(f"    Invalid prediction format in {pred_path}, skipping.")
                continue
            
            # Run suite
            robustness_results = run_suite(y_true=preds_df['actual'], y_pred=preds_df['prediction'])
            
            # Log results
            placebo = robustness_results['placebo']
            logger.info(f"    Placebo Test: p={placebo.p_value:.3f} ({'PASS' if placebo.is_significant else 'FAIL'})")
            
            econ = robustness_results['economic']
            logger.info(f"    Economic Sig: Ann_Ret={econ['Annualized_Return']:.1%}, Sharpe={econ['Sharpe']:.2f}")
            
            if econ['Sharpe'] < 0.5:
                logger.warning(f"    CAUTION: Low economic significance for {asset} champion!")

    def apply_multiple_testing_correction(self, df: pd.DataFrame):
        """
        Apply multiple testing corrections to the tournament results.
        """
        mtc_cfg = self.config.get('inference', {}).get('multiple_testing', {})
        if not mtc_cfg.get('enabled', True):
            return
            
        logger.info("=" * 60)
        logger.info("STEP 2.5: MULTIPLE TESTING CORRECTION")
        logger.info("=" * 60)
        
        tests = []
        for i, row in df.iterrows():
            if pd.isna(row.get('IC_p_value')):
                continue
                
            tests.append(HypothesisTest(
                test_id=f"{row['asset']}_{row['model']}",
                asset=row['asset'],
                model=row['model'],
                target='return',
                p_value=row['IC_p_value'],
                ic_estimate=row['IC_mean'],
                t_statistic=row['IC_t_stat']
            ))
            
        if not tests:
            logger.warning("No tests found for multiple testing correction")
            return
            
        corrector = MultipleTestingCorrector(
            fwer_alpha=mtc_cfg.get('fwer_alpha', 0.05),
            fdr_alpha=mtc_cfg.get('fdr_alpha', 0.10)
        )
        
        mtc_results = corrector.correct_all_methods(tests)
        
        # Log report
        report = format_mtc_report(mtc_results)
        logger.info("\n" + report)
        
        # Update results with q-values and significance
        bh_result = mtc_results['benjamini_hochberg']
        details_df = bh_result.test_details.set_index('test_id')
        
        df['q_value'] = np.nan
        df['sig_fdr'] = False
        
        for i, row in df.iterrows():
            test_id = f"{row['asset']}_{row['model']}"
            if test_id in details_df.index:
                df.at[i, 'q_value'] = details_df.at[test_id, 'q_value']
                df.at[i, 'sig_fdr'] = details_df.at[test_id, 'sig_adjusted']
                
        return mtc_results

    def _get_coint_pairs(self):
        coint_cfg = self.config.get('features', {}).get('cointegration', {})
        coint_pairs_raw = coint_cfg.get('pairs', CointegrationAnalyzer.DEFAULT_PAIRS)
        coint_pairs = []
        for p in coint_pairs_raw:
            if len(p) >= 3:
                 coint_pairs.append((f"{p[0]}_level", f"{p[1]}_level", p[2], p[3] if len(p) > 3 else ''))
        return coint_pairs

    def _get_selection_config(self):
        clustering_cfg = self.config.get('features', {}).get('clustering', {})
        selection_cfg_dict = clustering_cfg.get('selection', {})
        if selection_cfg_dict.get('ic_based', {}).get('enabled', False):
            ic_cfg = selection_cfg_dict.get('ic_based', {})
            return SelectionConfig(
                method=SelectionMethod.UNIVARIATE_IC,
                ic_lag_buffer_months=ic_cfg.get('lag_buffer_months', 24),
                ic_min_observations=ic_cfg.get('min_observations', 60)
            )
        else:
            method_str = selection_cfg_dict.get('method', 'centroid')
            return SelectionConfig(method=SelectionMethod(method_str))

    def _pipeline_factory(self, model_instance):
        coint_cfg = self.config.get('features', {}).get('cointegration', {})
        clustering_cfg = self.config.get('features', {}).get('clustering', {})
        # Delegate to standalone
        return create_pipeline_for_model(model_instance, self._get_coint_pairs(), coint_cfg, clustering_cfg)

    def _create_model(self, model_name):
        model_config = self.MODEL_CONFIGS.get(model_name)
        # Delegate to standalone
        return create_model_from_config(model_name, model_config)

    def run_nested_cv_ensemble_evaluation(self, assets: List[str]):
        """Run nested CV to get unbiased ensemble estimates."""
        nested_cfg = self.config.get('ensemble', {}).get('selection', {}).get('nested_cv', {})
        
        evaluator = NestedCVEnsembleEvaluator(
            n_outer_folds=nested_cfg.get('n_outer_folds', 5),
            n_inner_folds=nested_cfg.get('n_inner_folds', 4),
            outer_min_train_months=nested_cfg.get('outer_min_train_months', 120),
            inner_min_train_months=nested_cfg.get('inner_min_train_months', 84),
            ensemble_size=self.config.get('ensemble', {}).get('size', 5)
        )
        
        for asset in assets:
            target_key = f'{asset}_return'
            y = self.targets.get(target_key)
            if y is None: continue
            
            X = self.features
            
            # Use aligner to get consistent indices
            pub_lag = self.config.get('data', {}).get('alfred', {}).get('publication_lag_months', 1)
            from src.preprocessing.alignment import LaggedAligner
            aligner = LaggedAligner(lag_months=pub_lag)
            X_aligned, y_aligned = aligner.align_features_and_targets(X, y.dropna())
            
            # If holdout is enabled, we only use the development set for nested CV
            if self.holdout_enabled:
                X_dev, y_dev, _, _ = self.holdout_manager.split_data(X_aligned, y_aligned)
                X_input, y_input = X_dev, y_dev
            else:
                X_input, y_input = X_aligned, y_aligned
            
            # Exclude ensemble models from factories as they are what we're evaluating
            model_factories = {
                name: lambda n=name: self._create_model(n)
                for name in self.MODEL_CONFIGS.keys()
                if 'Ensemble' not in name
            }
            
            result = evaluator.evaluate(
                X_input, y_input,
                model_factories, self._pipeline_factory,
                asset=asset
            )
            
            logger.info("\n" + format_nested_cv_report(result))
            
            # Save results to a specialized CSV
            nested_results_path = self.experiments_dir / 'cv_results' / f'nested_cv_{asset}.joblib'
            joblib.dump(result, nested_results_path)

    def evaluate_holdout(self, assets: List[str], models: List[str]):
        """
        Evaluate best models on holdout set.
        
        This method is called ONCE, AFTER all model development is complete.
        """
        logger.info("=" * 60)
        logger.info("STEP 3: HOLDOUT EVALUATION (Never-Before-Seen Data)")
        logger.info("=" * 60)
        
        holdout_results = []
        
        for asset in assets:
            # Load the best model (already trained on development set)
            asset_mask = self.results['asset'] == asset
            valid_ic_mask = ~self.results['IC_mean'].isna()
            
            asset_results = self.results[asset_mask & valid_ic_mask]
            if asset_results.empty:
                logger.warning(f"No valid results for {asset}, skipping holdout")
                continue
                
            best_model_row = asset_results.sort_values('IC_mean', ascending=False).iloc[0]
            
            model_name = best_model_row['model']
            model_path = self.experiments_dir / 'models' / f'{asset}_{model_name}.joblib'
            
            if not model_path.exists():
                logger.warning(f"Model not found for {asset}: {model_path}")
                continue
            
            model = joblib.load(model_path)
            
            # Get holdout data
            target_key = f'{asset}_return'
            y = self.targets.get(target_key)
            if y is None:
                continue
                
            # Alignment and Holdout Split
            pub_lag = self.config.get('data', {}).get('alfred', {}).get('publication_lag_months', 1)
            aligner = LaggedAligner(lag_months=pub_lag)
            X, y_aligned = aligner.align_features_and_targets(self.features, y.dropna())
            _, _, X_holdout, y_holdout = self.holdout_manager.split_data(X, y_aligned)
            
            if len(y_holdout) == 0:
                logger.warning(f"No holdout data for {asset}")
                continue

            # Generate predictions on holdout
            predictions = model.predict(X_holdout)
            
            # Compute metrics
            from evaluation.inference import compute_ic_with_inference
            horizon = self.config.get('models', {}).get('targets', {}).get('returns', {}).get('horizon_months', 24)
            inference = compute_ic_with_inference(
                y_holdout, 
                pd.Series(predictions, index=y_holdout.index),
                horizon=horizon
            )
            
            cv_ic = best_model_row['IC_mean']
            degradation = (cv_ic - inference.estimate) / cv_ic if cv_ic > 0 else np.nan
            
            holdout_results.append({
                'asset': asset,
                'model': model_name,
                'holdout_IC': inference.estimate,
                'holdout_IC_t_stat': inference.t_stat_nw,
                'holdout_IC_p_value': inference.p_value_nw,
                'holdout_IC_significant': inference.p_value_nw < 0.05,
                'cv_IC': cv_ic,
                'IC_degradation': degradation,
                'n_holdout_obs': len(y_holdout),
                'n_independent_obs': len(y_holdout) // horizon
            })
            
            logger.info(
                f"{asset:<5} | {model_name:<15} | "
                f"CV IC: {cv_ic:.3f} → Holdout IC: {inference.estimate:.3f} "
                f"(p={inference.p_value_nw:.3f}, deg={degradation:.1%})"
            )
        
        if holdout_results:
            # Save holdout results
            holdout_df = pd.DataFrame(holdout_results)
            holdout_df.to_csv(
                self.experiments_dir / 'cv_results' / 'holdout_results.csv',
                index=False
            )
            self.holdout_results = holdout_df
            logger.info(f"Holdout results saved to {self.experiments_dir / 'cv_results' / 'holdout_results.csv'}")
        else:
            logger.warning("No holdout results generated")

    def _should_skip_lstm(self, X: pd.DataFrame, y: pd.Series, cv_folds: list) -> Tuple[bool, str]:
        """Determine if LSTM should be skipped using enhanced validation."""
        lstm_cfg = self.config.get('models', {}).get('neural', {}).get('lstm', {})
        if not lstm_cfg:
            return True, "LSTM configuration not found"
            
        req = SequenceRequirements(
            seq_length=lstm_cfg.get('sequence', {}).get('length', 12),
            stride=lstm_cfg.get('sequence', {}).get('stride', 1),
            min_total_sequences=lstm_cfg.get('min_requirements', {}).get('min_total_sequences', 50),
            min_train_sequences=lstm_cfg.get('min_requirements', {}).get('min_train_sequences', 30),
            min_val_sequences=lstm_cfg.get('min_requirements', {}).get('min_val_sequences', 10),
            safety_margin_pct=lstm_cfg.get('min_requirements', {}).get('safety_margin_pct', 0.10)
        )
        
        validator = LSTMSequenceValidator(req)
        
        # 1. Quick overall check (estimated)
        overall_result = validator.validate(X, y)
        if not overall_result.is_valid:
            return True, overall_result.failure_reason
            
        # 2. Detailed per-fold check
        fold_results = validator.validate_per_fold(X, y, cv_folds)
        invalid_folds = [f"Fold {i}" for i, r in fold_results.items() if not r['is_valid']]
        
        if invalid_folds:
            return True, f"Insufficient sequences in: {', '.join(invalid_folds)}"
            
        return False, None

    def _log_lstm_skip(self, asset: str, reason: str):
        """Record skipped LSTM for final reporting."""
        if not hasattr(self, 'lstm_skips'):
            self.lstm_skips = []
        self.lstm_skips.append({
            'asset': asset,
            'reason': reason,
            'timestamp': datetime.now()
        })
    
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
        print("\nBest Model per Asset (with benchmarking):")
        display_cols = ['asset', 'model', 'IC_mean', 'IC_rating', 'IC_significant', 'implied_sharpe']
        # Map boolean to checkmark for cleaner output
        print_df = best_models[display_cols].copy()
        print_df['IC_significant'] = print_df['IC_significant'].map({True: '✓', False: '✗'})
        print(print_df.to_string(index=False))
        
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
            
            try:
                model_path = self.experiments_dir / 'models' / f'{asset}_{model_name}.joblib'
                
                # Use a defensive check for existence and readability
                try:
                    can_access = model_path.exists() and os.access(model_path, os.R_OK)
                except PermissionError:
                    can_access = False
                    
                if can_access:
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
                            if 'Ensemble' in model_name:
                                model_type = 'ensemble'
                            elif any(t in model_name for t in ['forest', 'boost', 'gbm']):
                                model_type = 'tree'
                            elif any(t in model_name for t in ['mlp', 'lstm']):
                                model_type = 'neural'
                            else:
                                model_type = 'linear'
                                
                            analyzer.compute_shap_values(model, X, model_type=model_type)
                            
                            # Save SHAP
                            np.save(self.experiments_dir / 'shap' / f'{asset}_{model_name}_shap.npy', analyzer.shap_values)
                            
                            # Save Monitoring
                            monitoring_sheet = analyzer.generate_monitoring_sheet(X)
                            monitoring_sheet.to_csv(self.experiments_dir / 'reports' / f'{asset}_monitoring.csv', index=False)
                            logger.info(f"Generated monitoring for {asset}")
                else:
                    logger.warning(f"Skipping SHAP for {asset}: Model file {model_path} not found or not readable")
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
    parser.add_argument('--no-holdout', action='store_true', help='Disable holdout testing')
    parser.add_argument('--holdout-start', type=str, default=None, help='Explicit holdout start date (YYYY-MM-DD)')
    parser.add_argument('--holdout-only', action='store_true', help='Only run holdout evaluation')
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("ASSET-SPECIFIC MACRO REGIME DETECTION SYSTEM")
    logger.info("Model Tournament")
    logger.info("=" * 60)
    
    tournament = ModelTournament()
    
    if args.no_holdout:
        tournament.holdout_enabled = False
        tournament.holdout_manager = None
    elif args.holdout_start:
        tournament.holdout_enabled = True
        holdout_cfg = tournament.config.get('validation', {}).get('holdout', {})
        tournament.holdout_manager = HoldoutManager(
            method='date',
            holdout_start=args.holdout_start,
            min_holdout_months=holdout_cfg.get('min_holdout_months', 48),
            min_independent_obs=holdout_cfg.get('min_independent_obs', 20),
            forecast_horizon=tournament.config.get('models', {}).get('targets', {}).get('returns', {}).get('horizon_months', 24)
        )
    
    # Run feature pipeline
    feature_file = tournament.experiments_dir / 'features' / f'features_{datetime.now().strftime("%Y%m%d")}.parquet'
    
    # Run feature pipeline
    feature_file = tournament.experiments_dir / 'features' / f'features_{datetime.now().strftime("%Y%m%d")}.parquet'
    
    # Always try to load existing features first
    if feature_file.exists():
        logger.info(f"Features found at {feature_file}. Loading...")
        tournament.features = pd.read_parquet(feature_file)
    
    # If not found and NOT eval-only, run pipeline
    if tournament.features is None and not (args.eval_only or args.holdout_only):
        tournament.run_feature_pipeline(args.fred_api_key)
    
    if args.features_only:
        logger.info("Feature pipeline complete. Exiting.")
        return

    if args.holdout_only:
        if tournament.features is None:
            logger.error("Features not available for holdout evaluation. Run feature pipeline first.")
            return
        tournament.prepare_targets()
        # Load existing results to know which models were best
        results_path = tournament.experiments_dir / 'cv_results' / 'tournament_results.csv'
        if results_path.exists():
            tournament.results = pd.read_csv(results_path)
            tournament.evaluate_holdout(args.assets or tournament.ASSETS, args.models or ['all'])
        else:
            logger.error("No tournament results found. Run full tournament first.")
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
