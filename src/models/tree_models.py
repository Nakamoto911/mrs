"""
Tree-Based Models Module
========================
Random Forest, XGBoost, LightGBM, and CatBoost implementations.

Part of the Asset-Specific Macro Regime Detection System
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
import logging
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator

logger = logging.getLogger(__name__)


class TreeModelWrapper(BaseEstimator):
    """Wrapper for tree-based models with consistent interface."""
    
    def __init__(self, model_type: str = 'xgboost', **kwargs):
        """
        Initialize tree model.
        
        Args:
            model_type: 'random_forest', 'xgboost', 'lightgbm', or 'catboost'
            **kwargs: Model-specific parameters
        """
        self.model_type = model_type
        self.kwargs = kwargs
        self.model_ = None
        self.feature_names_ = None
        self.fill_values_ = None
        self.fitted_ = False
        
    def _create_model(self):
        """Create the underlying model."""
        if self.model_type == 'random_forest':
            from sklearn.ensemble import RandomForestRegressor
            return RandomForestRegressor(**self.kwargs)
        
        elif self.model_type == 'xgboost':
            try:
                import xgboost as xgb
                params = {
                    'n_estimators': self.kwargs.get('n_estimators', 100),
                    'max_depth': self.kwargs.get('max_depth', 6),
                    'learning_rate': self.kwargs.get('learning_rate', 0.1),
                    'min_child_weight': self.kwargs.get('min_child_weight', 1),
                    'subsample': self.kwargs.get('subsample', 0.8),
                    'colsample_bytree': self.kwargs.get('colsample_bytree', 0.8),
                    'random_state': self.kwargs.get('random_state', 42),
                    'n_jobs': self.kwargs.get('n_jobs', -1),
                    'verbosity': 0,
                }
                return xgb.XGBRegressor(**params)
            except ImportError:
                logger.warning("XGBoost not available, using RandomForest")
                from sklearn.ensemble import RandomForestRegressor
                return RandomForestRegressor(n_estimators=100, random_state=42)
        
        elif self.model_type == 'lightgbm':
            try:
                import lightgbm as lgb
                params = {
                    'n_estimators': self.kwargs.get('n_estimators', 100),
                    'max_depth': self.kwargs.get('max_depth', -1),
                    'learning_rate': self.kwargs.get('learning_rate', 0.1),
                    'num_leaves': self.kwargs.get('num_leaves', 31),
                    'random_state': self.kwargs.get('random_state', 42),
                    'n_jobs': self.kwargs.get('n_jobs', -1),
                    'verbosity': -1,
                }
                return lgb.LGBMRegressor(**params)
            except ImportError:
                logger.warning("LightGBM not available, using RandomForest")
                from sklearn.ensemble import RandomForestRegressor
                return RandomForestRegressor(n_estimators=100, random_state=42)
        
        elif self.model_type == 'catboost':
            try:
                from catboost import CatBoostRegressor
                params = {
                    'iterations': self.kwargs.get('iterations', 100),
                    'depth': self.kwargs.get('depth', 6),
                    'learning_rate': self.kwargs.get('learning_rate', 0.1),
                    'random_seed': self.kwargs.get('random_state', 42),
                    'verbose': False,
                }
                return CatBoostRegressor(**params)
            except ImportError:
                logger.warning("CatBoost not available, using RandomForest")
                from sklearn.ensemble import RandomForestRegressor
                # Filter kwargs for RF
                rf_params = {k: v for k, v in self.kwargs.items() if k in ['n_estimators', 'max_depth', 'min_samples_split', 'min_samples_leaf', 'n_jobs', 'random_state']}
                return RandomForestRegressor(**rf_params)
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'TreeModelWrapper':
        """
        Fit the model.
        
        Args:
            X: Feature DataFrame
            y: Target series
            
        Returns:
            self
        """
        self.feature_names_ = X.columns.tolist()
        
        # 1. Align features and target
        y_clean = y.dropna()
        common_idx = X.index.intersection(y_clean.index)
        X_sync = X.loc[common_idx]
        y_sync = y_clean.loc[common_idx]
        
        if len(y_sync) < 20:
             raise ValueError(f"Insufficient total samples ({len(y_sync)}) for alignment")
 
        # 2. Prune all-NaN features (critical for imputation)
        all_nan_features = X_sync.columns[X_sync.isna().all()].tolist()
        if all_nan_features:
            logger.debug(f"Removing {len(all_nan_features)} all-NaN features")
            
        self.feature_names_ = [c for c in X_sync.columns if c not in all_nan_features]
        X_filtered = X_sync[self.feature_names_]
        
        # 3. Strict Rolling Imputation (No Look-ahead)
        # 1. Compute rolling stats (shifted by 1)
        rolling_medians = X_filtered.expanding(min_periods=1).median().shift(1)
        
        # 2. Impute with rolling medians, then bfill for the first row, then 0.0 as final fallback
        X_imputed = X_filtered.fillna(rolling_medians).bfill().fillna(0.0)
        
        # 3. Store strict PIT medians for Inference usage (final state of the rolling window)
        self.fill_values_ = X_filtered.median().fillna(0.0)
        
        if len(X_imputed) < 20:
            raise ValueError(f"Insufficient data for fitting: {len(X_imputed)}")
        
        # Create and fit model
        self.model_ = self._create_model()
        self.model_.fit(X_imputed, y_sync)
        
        self.fitted_ = True
        return self
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Generate predictions.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Predictions as Series
        """
        if not self.fitted_:
            raise ValueError("Model not fitted")
        
        # Filter features
        X_filtered = X[self.feature_names_]
        
        # Impute
        X_filled = X_filtered.fillna(self.fill_values_)
        
        predictions = self.model_.predict(X_filled)
        
        return pd.Series(predictions, index=X.index)
    
    def get_feature_importance(self) -> pd.Series:
        """Get feature importance from model."""
        if not self.fitted_:
            raise ValueError("Model not fitted")
        
        if hasattr(self.model_, 'feature_importances_'):
            importance = self.model_.feature_importances_
        else:
            importance = np.ones(len(self.feature_names_)) / len(self.feature_names_)
        
        return pd.Series(importance, index=self.feature_names_).sort_values(ascending=False)


class RegimeClassifier:
    """Tree-based classifier for regime prediction."""
    
    def __init__(self, model_type: str = 'lightgbm', **kwargs):
        """
        Initialize classifier.
        
        Args:
            model_type: 'random_forest', 'xgboost', 'lightgbm', or 'catboost'
            **kwargs: Model-specific parameters
        """
        self.model_type = model_type
        self.kwargs = kwargs
        self.model = None
        self.feature_names = None
    
    def _create_model(self):
        """Create the underlying classifier."""
        if self.model_type == 'lightgbm':
            try:
                import lightgbm as lgb
                params = {
                    'n_estimators': self.kwargs.get('n_estimators', 100),
                    'max_depth': self.kwargs.get('max_depth', -1),
                    'learning_rate': self.kwargs.get('learning_rate', 0.1),
                    'num_leaves': self.kwargs.get('num_leaves', 31),
                    'random_state': self.kwargs.get('random_state', 42),
                    'n_jobs': self.kwargs.get('n_jobs', -1),
                    'verbosity': -1,
                    'objective': 'binary',
                }
                return lgb.LGBMClassifier(**params)
            except ImportError:
                from sklearn.ensemble import RandomForestClassifier
                return RandomForestClassifier(n_estimators=100, random_state=42)
        
        elif self.model_type == 'xgboost':
            try:
                import xgboost as xgb
                params = {
                    'n_estimators': self.kwargs.get('n_estimators', 100),
                    'max_depth': self.kwargs.get('max_depth', 6),
                    'learning_rate': self.kwargs.get('learning_rate', 0.1),
                    'random_state': self.kwargs.get('random_state', 42),
                    'n_jobs': self.kwargs.get('n_jobs', -1),
                    'verbosity': 0,
                    'objective': 'binary:logistic',
                }
                return xgb.XGBClassifier(**params)
            except ImportError:
                from sklearn.ensemble import RandomForestClassifier
                return RandomForestClassifier(n_estimators=100, random_state=42)
        
        else:
            from sklearn.ensemble import RandomForestClassifier
            return RandomForestClassifier(**self.kwargs)
    
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'RegimeClassifier':
        """Fit the classifier."""
        self.feature_names = X.columns.tolist()
        
        mask = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X.loc[mask]
        y_clean = y.loc[mask]
        
        self.model = self._create_model()
        self.model.fit(X_clean, y_clean)
        
        return self
    
    def predict_proba(self, X: pd.DataFrame) -> pd.Series:
        """Predict probability of bullish regime."""
        if self.model is None:
            raise ValueError("Model not fitted")
        
        X_filled = X.fillna(0)
        proba = self.model.predict_proba(X_filled)[:, 1]
        
        return pd.Series(proba, index=X.index)


def create_tree_model(model_type: str, **kwargs) -> TreeModelWrapper:
    """Factory function for tree models."""
    return TreeModelWrapper(model_type, **kwargs)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test
    np.random.seed(42)
    n = 200
    
    X = pd.DataFrame({
        'feature1': np.random.randn(n),
        'feature2': np.random.randn(n),
        'feature3': np.random.randn(n),
    })
    
    y = 0.5 * X['feature1'] + 0.3 * X['feature2'] ** 2 + np.random.randn(n) * 0.1
    y = pd.Series(y, name='target')
    
    # Test XGBoost
    model = create_tree_model('xgboost', n_estimators=50, max_depth=3)
    model.fit(X, y)
    
    print("Feature Importance:")
    print(model.get_feature_importance())
