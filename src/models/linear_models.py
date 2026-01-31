"""
Linear Models Module
====================
Ridge, Lasso, Elastic Net, and VECM implementations.

Part of the Asset-Specific Macro Regime Detection System
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from preprocessing import TimeSeriesScaler, PointInTimeImputer
import logging

logger = logging.getLogger(__name__)


class LinearModelWrapper(BaseEstimator):
    """Wrapper for sklearn linear models with consistent interface."""
    
    MODELS = {
        'ridge': Ridge,
        'lasso': Lasso,
        'elastic_net': ElasticNet,
    }
    
    def __init__(self, model_type: str = 'ridge', **kwargs):
        """
        Initialize linear model.
        
        Args:
            model_type: 'ridge', 'lasso', or 'elastic_net'
            **kwargs: Model-specific parameters
        """
        if model_type not in self.MODELS:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.model_type = model_type
        self.kwargs = kwargs
        self.model_ = None
        self.scaler_ = StandardScaler()
        self.fill_values_ = None
        self.feature_names_ = None
        self.selected_features_ = None
        self.fitted_ = False
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'LinearModelWrapper':
        """
        Fit the model with robust preprocessing.
        
        Args:
            X: Feature DataFrame
            y: Target series
            
        Returns:
            self
        """
        self.feature_names_ = X.columns.tolist()
        
        # 1. Align features and target (remove NaNs from target first)
        y_clean = y.dropna()
        common_idx = X.index.intersection(y_clean.index)
        X_sync = X.loc[common_idx]
        y_sync = y_clean.loc[common_idx]
        
        if len(y_sync) < 20:
             raise ValueError(f"Insufficient total samples ({len(y_sync)}) for alignment")
        
        # 2. Variance & Validity Filtering (remove constant and all-NaN features)
        # Drop all-NaN columns first to avoid SimpleImputer issues
        all_nan_features = X_sync.columns[X_sync.isna().all()].tolist()
        if all_nan_features:
            logger.debug(f"Removing {len(all_nan_features)} all-NaN features")
            
        variances = X_sync.drop(columns=all_nan_features).var()
        constant_features = variances[variances == 0].index.tolist()
        if constant_features:
            logger.debug(f"Removing {len(constant_features)} constant features")
            
        initial_selected = [c for c in X_sync.columns if c not in all_nan_features and c not in constant_features]
        initial_selected = [c for c in X_sync.columns if c not in all_nan_features and c not in constant_features]
        X_filtered = X_sync[initial_selected]
        
        # 3. Degrees of Freedom Guard: N > K + 20
        # If K is too large, prune features by NaN count then variance
        n_samples = len(X_filtered)
        max_k = n_samples - 20
        
        if len(initial_selected) > max_k:
            logger.debug(f"Pruning features to satisfy DOF: N={n_samples}, K_init={len(initial_selected)}, Max_K={max_k}")
            
            # Rank features by NaN count (ascending) and then Variance (descending)
            nan_counts = X_filtered.isna().sum()
            vars = X_filtered.var()
            
            ranking = pd.DataFrame({
                'nan_count': nan_counts,
                'variance': vars
            }).sort_values(['nan_count', 'variance'], ascending=[True, False])
            
            self.selected_features_ = ranking.index[:max_k].tolist()
            X_filtered = X_filtered[self.selected_features_]
            logger.debug(f"Successfully pruned to {len(self.selected_features_)} features")
        else:
            self.selected_features_ = initial_selected
            
        if len(self.selected_features_) == 0:
            raise ValueError(f"No valid features remaining after pruning (N={n_samples})")

        X_filtered = X_filtered[self.selected_features_]

        # 4. Point-in-Time Imputation
        self.imputer_ = PointInTimeImputer(strategy='median')
        X_imputed = self.imputer_.transform_expanding(X_filtered)
        
        # 5. Point-in-Time Scaling
        self.scaler_ = TimeSeriesScaler(method='standard')
        X_scaled = self.scaler_.fit_transform_rolling(X_imputed)
        
        # 6. Fit model
        self.model_ = self.MODELS[self.model_type](**self.kwargs)
        
        # FINAL SAFETY: Align X and y and ensure no NaNs remain
        final_mask = ~(X_scaled.isna().any(axis=1) | y_sync.isna())
        if final_mask.sum() < 20:
            raise ValueError(f"Insufficient valid data after scaling (N={final_mask.sum()})")
            
        self.model_.fit(X_scaled[final_mask], y_sync[final_mask])
        
        self.fitted_ = True
        return self
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Generate predictions using the same robust pipeline.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Predictions as Series
        """
        if not self.fitted_:
            raise ValueError("Model not fitted")
        
        # Filter to selected features
        X_filtered = X[self.selected_features_]
        
        # Impute and Scale (Using frozen state from Fit)
        X_imputed = self.imputer_.transform(X_filtered)
        X_scaled = self.scaler_.transform(X_imputed)
        
        predictions = self.model_.predict(X_scaled)
        
        return pd.Series(predictions, index=X.index)
    
    def get_coefficients(self) -> pd.Series:
        """Get model coefficients for selected features."""
        if not self.fitted_:
            raise ValueError("Model not fitted")
        
        return pd.Series(self.model_.coef_, index=self.selected_features_)
    
    def get_feature_importance(self) -> pd.Series:
        """Get absolute coefficients as feature importance."""
        return self.get_coefficients().abs().sort_values(ascending=False)


class VECMWrapper:
    """Vector Error Correction Model wrapper for cointegrated systems."""
    
    def __init__(self, max_lag: int = 12, coint_rank: int = 1):
        """
        Initialize VECM.
        
        Args:
            max_lag: Maximum lag order
            coint_rank: Cointegration rank
        """
        self.max_lag = max_lag
        self.coint_rank = coint_rank
        self.model = None
        
    def fit(self, df: pd.DataFrame) -> 'VECMWrapper':
        """
        Fit VECM model.
        
        Args:
            df: DataFrame with multiple time series
            
        Returns:
            self
        """
        try:
            from statsmodels.tsa.vector_ar.vecm import VECM as StatsVECM
            
            # Clean data
            df_clean = df.dropna()
            
            if len(df_clean) < self.max_lag + 20:
                raise ValueError("Insufficient data for VECM")
            
            # Fit VECM
            self.model = StatsVECM(df_clean, k_ar_diff=self.max_lag, coint_rank=self.coint_rank)
            self.fit_result = self.model.fit()
            
        except ImportError:
            logger.warning("statsmodels VECM not available")
            self.model = None
        except Exception as e:
            logger.warning(f"VECM fitting failed: {e}")
            self.model = None
        
        return self
    
    def predict(self, steps: int = 1) -> Optional[pd.DataFrame]:
        """
        Generate forecasts.
        
        Args:
            steps: Number of steps ahead
            
        Returns:
            Forecast DataFrame
        """
        if self.model is None:
            return None
        
        try:
            forecast = self.fit_result.predict(steps=steps)
            return pd.DataFrame(forecast, columns=self.model.endog_names)
        except Exception as e:
            logger.warning(f"VECM prediction failed: {e}")
            return None


def create_linear_model(model_type: str, alpha: float = 1.0, 
                       l1_ratio: float = 0.5, **kwargs) -> LinearModelWrapper:
    """
    Factory function for linear models.
    
    Args:
        model_type: 'ridge', 'lasso', or 'elastic_net'
        alpha: Regularization strength
        l1_ratio: L1 ratio for elastic net
        **kwargs: Additional parameters for the underlying model
        
    Returns:
        LinearModelWrapper instance
    """
    if model_type == 'ridge':
        return LinearModelWrapper('ridge', alpha=alpha, **kwargs)
    elif model_type == 'lasso':
        return LinearModelWrapper('lasso', alpha=alpha, **kwargs)
    elif model_type == 'elastic_net':
        return LinearModelWrapper('elastic_net', alpha=alpha, l1_ratio=l1_ratio, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


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
    
    y = 0.5 * X['feature1'] + 0.3 * X['feature2'] + np.random.randn(n) * 0.1
    y = pd.Series(y, name='target')
    
    # Test Ridge
    model = create_linear_model('ridge', alpha=1.0)
    model.fit(X, y)
    
    print("Coefficients:")
    print(model.get_coefficients())
    
    print("\nFeature Importance:")
    print(model.get_feature_importance())
