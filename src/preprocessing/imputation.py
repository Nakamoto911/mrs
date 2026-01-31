"""
Point-in-Time Imputer
====================
Implements impartial imputation that prevents lookahead bias.
Supports global median fallback only from training data.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, Dict, List
from sklearn.base import BaseEstimator, TransformerMixin
import logging

logger = logging.getLogger(__name__)


class PointInTimeImputer(BaseEstimator, TransformerMixin):
    """
    Imputer that prevents lookahead bias.
    
    During training, it can perform rolling/expanding imputation.
    During out-of-sample use, it uses a fixed 'fill_values' dictionary 
    learned from the final state of the training set.
    """
    
    def __init__(
        self, 
        strategy: str = "median",
        fallback_value: float = 0.0,
        window_type: str = "fixed"
    ):
        """
        Args:
            strategy: 'median', 'mean', or 'zero'
            fallback_value: Value to use if everything is NaN
            window_type: 'fixed' (standard sklearn style) or 'expanding'
        """
        self.strategy = strategy
        self.fallback_value = fallback_value
        self.window_type = window_type
        
        self.fill_values_ = None
        self.feature_names_in_ = None
        self.fitted_ = False

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        """
        Learn the fill values from the training data.
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist()
            X_data = X.replace([np.inf, -np.inf], np.nan)
        else:
            X_df = pd.DataFrame(X)
            X_data = X_df.replace([np.inf, -np.inf], np.nan)

        if self.strategy == "median":
            self.fill_values_ = X_data.median().to_dict()
        elif self.strategy == "mean":
            self.fill_values_ = X_data.mean().to_dict()
        elif self.strategy == "zero":
            self.fill_values_ = {col: 0.0 for col in X_data.columns}
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
            
        # Handle columns with all NaN
        for col in X_data.columns:
            if pd.isna(self.fill_values_.get(col)):
                self.fill_values_[col] = self.fallback_value
                
        self.fitted_ = True
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """
        Apply fixed fill values (no lookahead).
        """
        if not self.fitted_:
            raise ValueError("Imputer must be fitted before transform.")

        is_df = isinstance(X, pd.DataFrame)
        if is_df:
            # Replace Infs with NaNs, then fill with learned statistics, then final fallback
            X_out = X.replace([np.inf, -np.inf], np.nan).fillna(self.fill_values_).fillna(self.fallback_value)
        else:
            X_df = pd.DataFrame(X)
            if self.feature_names_in_ is not None:
                X_df.columns = self.feature_names_in_
            X_out_df = X_df.replace([np.inf, -np.inf], np.nan).fillna(self.fill_values_).fillna(self.fallback_value)
            X_out = X_out_df.values
            
        return X_out

    def transform_expanding(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Point-in-time imputation using expanding window statistics.
        Each row is imputed using information available up to that row.
        """
        # Clean input
        X_clean = X.replace([np.inf, -np.inf], np.nan)
        
        if self.strategy == "median":
            stats = X_clean.expanding().median().shift(1)
        elif self.strategy == "mean":
            stats = X_clean.expanding().mean().shift(1)
        else:
            # Fallback to zeros for 'zero' strategy
            stats = X_clean.copy()
            stats.iloc[:, :] = 0.0
            
        # Fill first row (which shift(1) leaves NaN) and any stats NAs with fallback
        stats = stats.fillna(self.fallback_value)
        
        # Row-wise fill
        X_out = X_clean.copy()
        mask = X_out.isna()
        X_out[mask] = stats[mask]
        
        # Final safety fill
        X_out = X_out.fillna(self.fallback_value)
        
        # Update fitted state with final values
        if self.strategy == "median":
            self.fill_values_ = X_clean.median().to_dict()
        elif self.strategy == "mean":
            self.fill_values_ = X_clean.mean().to_dict()
        self.fitted_ = True
        
        return X_out
