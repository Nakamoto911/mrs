"""
Holdout Testing Module
======================
Implements strict temporal holdout for unbiased model evaluation.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class HoldoutSplit:
    """Container for holdout split information."""
    development_end: pd.Timestamp
    holdout_start: pd.Timestamp
    holdout_end: pd.Timestamp
    n_development_months: int
    n_holdout_months: int
    n_independent_holdout_obs: int  # Accounts for horizon overlap


class HoldoutManager:
    """
    Manages strict temporal holdout for unbiased evaluation.
    
    Key principle: The holdout period is NEVER touched during development.
    This includes feature selection, hyperparameter tuning, and model selection.
    """
    
    def __init__(
        self,
        method: str = "percentage",
        holdout_pct: float = 0.15,
        holdout_start: Optional[str] = None,
        min_holdout_months: int = 48,
        min_independent_obs: int = 20,
        forecast_horizon: int = 24
    ):
        """
        Initialize holdout manager.
        
        Args:
            method: "percentage" or "date"
            holdout_pct: Fraction of data to hold out (if method="percentage")
            holdout_start: Start date for holdout (if method="date")
            min_holdout_months: Minimum required holdout period
            min_independent_obs: Minimum non-overlapping observations
            forecast_horizon: Forecast horizon in months (for overlap calculation)
        """
        self.method = method
        self.holdout_pct = holdout_pct
        self.holdout_start = pd.Timestamp(holdout_start) if holdout_start else None
        self.min_holdout_months = min_holdout_months
        self.min_independent_obs = min_independent_obs
        self.forecast_horizon = forecast_horizon
        
        self._split_info: Optional[HoldoutSplit] = None
    
    def compute_split(self, data_index: pd.DatetimeIndex) -> HoldoutSplit:
        """
        Compute the holdout split point.
        
        Args:
            data_index: DatetimeIndex of the full dataset
            
        Returns:
            HoldoutSplit with split information
        """
        data_index = data_index.sort_values()
        n_total = len(data_index)
        
        if self.method == "percentage":
            n_holdout = int(n_total * self.holdout_pct)
            split_idx = n_total - n_holdout
            holdout_start = data_index[split_idx]
        else:  # date-based
            if self.holdout_start is None:
                raise ValueError("holdout_start required for date-based method")
            holdout_start = self.holdout_start
            split_idx = data_index.searchsorted(holdout_start)
            n_holdout = n_total - split_idx
        
        # Validate minimum requirements
        if n_holdout < self.min_holdout_months:
            raise ValueError(
                f"Holdout period ({n_holdout} months) is less than minimum "
                f"required ({self.min_holdout_months} months)"
            )
        
        n_independent = n_holdout // self.forecast_horizon
        if n_independent < self.min_independent_obs:
            logger.warning(
                f"Only {n_independent} independent observations in holdout "
                f"(minimum recommended: {self.min_independent_obs})"
            )
        
        self._split_info = HoldoutSplit(
            development_end=data_index[split_idx - 1],
            holdout_start=holdout_start,
            holdout_end=data_index[-1],
            n_development_months=split_idx,
            n_holdout_months=n_holdout,
            n_independent_holdout_obs=n_independent
        )
        
        logger.info(
            f"Holdout split: Development ends {self._split_info.development_end.strftime('%Y-%m')}, "
            f"Holdout starts {self._split_info.holdout_start.strftime('%Y-%m')} "
            f"({n_holdout} months, {n_independent} independent obs)"
        )
        
        return self._split_info
    
    def split_data(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        """
        Split data into development and holdout sets.
        
        Args:
            X: Feature DataFrame
            y: Target Series
            
        Returns:
            (X_dev, y_dev, X_holdout, y_holdout)
        """
        if self._split_info is None:
            self.compute_split(X.index)
        
        # Development set: everything before holdout
        dev_mask = X.index < self._split_info.holdout_start
        X_dev = X.loc[dev_mask]
        y_dev = y.loc[dev_mask]
        
        # Holdout set: everything from holdout_start onwards
        holdout_mask = X.index >= self._split_info.holdout_start
        X_holdout = X.loc[holdout_mask]
        y_holdout = y.loc[holdout_mask]
        
        return X_dev, y_dev, X_holdout, y_holdout
    
    def get_split_info(self) -> Optional[HoldoutSplit]:
        """Return the current split information."""
        return self._split_info


def validate_holdout_never_touched(
    model_training_dates: pd.DatetimeIndex,
    holdout_start: pd.Timestamp
) -> bool:
    """
    Verify that no training data leaked into holdout period.
    
    Args:
        model_training_dates: Dates used in model training
        holdout_start: Start of holdout period
        
    Returns:
        True if no leakage detected
        
    Raises:
        ValueError if leakage detected
    """
    leaked_dates = model_training_dates[model_training_dates >= holdout_start]
    
    if len(leaked_dates) > 0:
        raise ValueError(
            f"DATA LEAKAGE DETECTED: {len(leaked_dates)} training observations "
            f"are in the holdout period (first: {leaked_dates[0]})"
        )
    
    return True
