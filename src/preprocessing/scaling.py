"""
Time-Series Scaler
==================
Implements robust, point-in-time scaling to prevent lookahead bias.
Supports expanding and rolling window scaling.
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, Dict, Any
from sklearn.base import BaseEstimator, TransformerMixin
import logging

logger = logging.getLogger(__name__)


class TimeSeriesScaler(BaseEstimator, TransformerMixin):
    """
    Scaler designed for time-series data that prevents lookahead bias.
    
    It saves the state (mean, std) from the training data and applies it
    to out-of-sample data. It can also perform 'expanding' or 'rolling' 
    scaling during the training phase if desired, though typically it 
    behaves like StandardScaler but with explicit state management.
    """
    
    def __init__(
        self, 
        method: str = "standard", 
        window_type: str = "expanding",
        min_periods: int = 36,
        rolling_window: Optional[int] = None
    ):
        """
        Args:
            method: 'standard' (mean/std) or 'robust' (median/IQR)
            window_type: 'expanding', 'rolling', or 'fixed'
            min_periods: Minimum samples before scaling is applied
            rolling_window: Size of rolling window (if window_type='rolling')
        """
        self.method = method
        self.window_type = window_type
        self.min_periods = min_periods
        self.rolling_window = rolling_window
        
        self.mean_ = None
        self.scale_ = None
        self.feature_names_in_ = None
        self.n_features_in_ = None
        self.fitted_ = False

    def fit(self, X: Union[pd.DataFrame, np.ndarray], y=None):
        """
        Fit the scaler on the training data.
        In time-series context, this calculates the FINAL state of the training set.
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns.tolist()
            self.n_features_in_ = len(self.feature_names_in_)
            X_data = X.values
        else:
            X_data = X
            self.n_features_in_ = X.shape[1]

        if self.method == "standard":
            self.mean_ = np.nanmean(X_data, axis=0)
            self.scale_ = np.nanstd(X_data, axis=0)
            # Avoid division by zero
            self.scale_[self.scale_ == 0] = 1.0
        elif self.method == "robust":
            self.mean_ = np.nanmedian(X_data, axis=0)
            q75, q25 = np.nanpercentile(X_data, [75, 25], axis=0)
            self.scale_ = q75 - q25
            self.scale_[self.scale_ == 0] = 1.0
        else:
            raise ValueError(f"Unknown scaling method: {self.method}")
            
        self.fitted_ = True
        return self

    def transform(self, X: Union[pd.DataFrame, np.ndarray]) -> Union[pd.DataFrame, np.ndarray]:
        """
        Apply the learned scaling to the data.
        """
        if not self.fitted_:
            raise ValueError("Scaler must be fitted before transform.")

        is_df = isinstance(X, pd.DataFrame)
        if is_df:
            X_data = X.values
        else:
            X_data = X

        X_scaled = (X_data - self.mean_) / self.scale_
        
        if is_df:
            return pd.DataFrame(X_scaled, index=X.index, columns=X.columns)
        return X_scaled

    def fit_transform_rolling(self, X: Any) -> pd.DataFrame:
        """
        Performs point-in-time scaling for backtesting/training.
        Each row is scaled using ONLY data available up to that row.
        """
        if not isinstance(X, pd.DataFrame):
            if self.feature_names_in_ is not None:
                X = pd.DataFrame(X, columns=self.feature_names_in_)
            else:
                X = pd.DataFrame(X)

        if self.method == "standard":
            if self.window_type == "expanding":
                mean = X.expanding(min_periods=self.min_periods).mean()
                std = X.expanding(min_periods=self.min_periods).std()
            else:
                window = self.rolling_window or 60
                mean = X.rolling(window=window, min_periods=self.min_periods).mean()
                std = X.rolling(window=window, min_periods=self.min_periods).std()
        elif self.method == "robust":
            if self.window_type == "expanding":
                mean = X.expanding(min_periods=self.min_periods).median()
                q75 = X.expanding(min_periods=self.min_periods).quantile(0.75)
                q25 = X.expanding(min_periods=self.min_periods).quantile(0.25)
                std = q75 - q25
            else:
                window = self.rolling_window or 60
                mean = X.rolling(window=window, min_periods=self.min_periods).median()
                q75 = X.rolling(window=window, min_periods=self.min_periods).quantile(0.75)
                q25 = X.rolling(window=window, min_periods=self.min_periods).quantile(0.25)
                std = q75 - q25
        else:
            raise ValueError(f"Unknown scaling method: {self.method}")

        std = std.replace(0, 1.0)
        
        # We shift(1) because at time t, we only have statistics from t-1.
        # This leaves the first row as NaN. We fill it with the first available stats.
        shifted_mean = mean.shift(1).bfill()
        shifted_std = std.shift(1).bfill()
        
        X_scaled = (X - shifted_mean) / shifted_std
        
        # Final safety fill for anything else
        X_scaled = X_scaled.fillna(0.0)
        
        # Update fitted state with the LAST known values (for future OOS use)
        if hasattr(mean, 'iloc') and len(mean) > 0:
            self.mean_ = mean.iloc[-1].values
            self.scale_ = std.iloc[-1].values
        self.fitted_ = True
        
        return X_scaled
