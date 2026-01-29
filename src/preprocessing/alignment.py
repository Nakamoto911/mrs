import pandas as pd
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class LaggedAligner:
    """
    Aligns features to targets by shifting feature availability forward.
    This remediates publication lag look-ahead bias by ensuring that 
    at any decision date T, only data available at or before T is used.
    """
    def __init__(self, lag_months: int = 1):
        """
        Initialize the aligner.
        
        Args:
            lag_months: Number of months to shift features forward.
                        Typical FRED-MD publication lag is 1 month.
        """
        self.lag_months = lag_months

    def align_features_and_targets(self, features: pd.DataFrame, targets: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Aligns features to targets by shifting feature availability forward.
        
        Logic:
        1. Feature Index (Reference Date) -> Shift forward by lag_months -> Availability Date.
        2. Merge with Target Index (Trade Date) on Availability Date.
        
        Example (lag_months=1):
        - A feature dated 2020-01-01 (Jan reference) becomes available on 2020-02-01.
        - A target dated 2020-02-01 (Feb trade date) represents returns from Feb 1 to Mar 1.
        - The model will now use Jan macro data to predict Feb-Mar returns.
        
        Args:
            features: Feature DataFrame indexed by Reference Date (Start of Month).
            targets: Target Series indexed by Trade Date (Start of Month).
            
        Returns:
            Tuple of (X_aligned, y_aligned)
        """
        if features.empty or targets.empty:
            logger.warning("Empty features or targets provided for alignment.")
            return features, targets

        # 1. Shift Feature Index forward to Availability Date
        available_features = features.copy()
        
        # Ensure indices are datetime if they aren't already
        if not isinstance(available_features.index, pd.DatetimeIndex):
            available_features.index = pd.to_datetime(available_features.index)
        if not isinstance(targets.index, pd.DatetimeIndex):
            targets.index = pd.to_datetime(targets.index)
            
        available_features.index = available_features.index + pd.DateOffset(months=self.lag_months)
        
        # 2. Intersection with Targets
        common_idx = available_features.index.intersection(targets.index)
        
        if len(common_idx) == 0:
            logger.error(f"No overlapping dates found after {self.lag_months} month lag alignment.")
            return pd.DataFrame(), pd.Series()
            
        # 3. Slice and Return
        X_aligned = available_features.loc[common_idx]
        y_aligned = targets.loc[common_idx]
        
        logger.debug(f"Aligned {len(X_aligned)} samples with {self.lag_months} month lag.")
        return X_aligned, y_aligned
