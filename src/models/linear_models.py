"""
Linear Models Module
====================
Ridge, Lasso, Elastic Net, and VECM implementations.

Part of the Asset-Specific Macro Regime Detection System
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from sklearn.linear_model import Ridge, Lasso, ElasticNet, RidgeCV, LassoCV, ElasticNetCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import logging

logger = logging.getLogger(__name__)


class LinearModelWrapper:
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
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        
    def fit(self, X: pd.DataFrame, y: pd.Series) -> 'LinearModelWrapper':
        """
        Fit the model.
        
        Args:
            X: Feature DataFrame
            y: Target series
            
        Returns:
            self
        """
        self.feature_names = X.columns.tolist()
        
        # Handle missing values
        mask = ~(X.isna().any(axis=1) | y.isna())
        X_clean = X.loc[mask]
        y_clean = y.loc[mask]
        
        if len(X_clean) < 20:
            raise ValueError("Insufficient data for fitting")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_clean)
        
        # Fit model
        self.model = self.MODELS[self.model_type](**self.kwargs)
        self.model.fit(X_scaled, y_clean)
        
        return self
    
    def predict(self, X: pd.DataFrame) -> pd.Series:
        """
        Generate predictions.
        
        Args:
            X: Feature DataFrame
            
        Returns:
            Predictions as Series
        """
        if self.model is None:
            raise ValueError("Model not fitted")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        return pd.Series(predictions, index=X.index)
    
    def get_coefficients(self) -> pd.Series:
        """Get model coefficients."""
        if self.model is None:
            raise ValueError("Model not fitted")
        
        return pd.Series(self.model.coef_, index=self.feature_names)
    
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
                       l1_ratio: float = 0.5) -> LinearModelWrapper:
    """
    Factory function for linear models.
    
    Args:
        model_type: 'ridge', 'lasso', or 'elastic_net'
        alpha: Regularization strength
        l1_ratio: L1 ratio for elastic net
        
    Returns:
        LinearModelWrapper instance
    """
    if model_type == 'ridge':
        return LinearModelWrapper('ridge', alpha=alpha)
    elif model_type == 'lasso':
        return LinearModelWrapper('lasso', alpha=alpha)
    elif model_type == 'elastic_net':
        return LinearModelWrapper('elastic_net', alpha=alpha, l1_ratio=l1_ratio)
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
