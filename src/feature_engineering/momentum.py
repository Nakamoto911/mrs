"""
Momentum Features Module
========================
Generates momentum and change features across multiple time horizons.

Part of the Asset-Specific Macro Regime Detection System
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class MomentumFeatureGenerator:
    """Generates momentum features for all stationary series."""
    
    DEFAULT_WINDOWS = [3, 6, 12]
    
    def __init__(self, windows: Optional[List[int]] = None,
                 include_acceleration: bool = True,
                 include_zscore: bool = True,
                 min_periods: int = 12):
        self.windows = windows or self.DEFAULT_WINDOWS
        self.include_acceleration = include_acceleration
        self.include_zscore = include_zscore
        self.min_periods = min_periods
    
    def compute_change(self, series: pd.Series, window: int) -> pd.Series:
        """Compute simple change over window."""
        return series.diff(window)
    
    def compute_rate_of_change(self, series: pd.Series, window: int) -> pd.Series:
        """Compute rate of change."""
        shifted = series.shift(window)
        return (series - shifted) / shifted.abs().replace(0, np.nan)
    
    def compute_acceleration(self, series: pd.Series, window: int) -> pd.Series:
        """Compute acceleration (change of change)."""
        return series.diff(window).diff(window)
    
    def compute_rolling_zscore(self, series: pd.Series, window: int) -> pd.Series:
        """Compute rolling z-score."""
        rolling_mean = series.rolling(window=window, min_periods=self.min_periods).mean()
        rolling_std = series.rolling(window=window, min_periods=self.min_periods).std()
        return (series - rolling_mean) / rolling_std.replace(0, np.nan)
    
    def generate_features(self, df: pd.DataFrame,
                         columns: Optional[List[str]] = None) -> pd.DataFrame:
        """Generate momentum features for DataFrame."""
        columns = columns or df.columns.tolist()
        columns = [c for c in columns if c in df.columns]
        
        features_list = []
        
        for col in columns:
            series = df[col]
            col_features = pd.DataFrame(index=df.index)
            
            for window in self.windows:
                col_features[f"{col}_chg_{window}M"] = self.compute_change(series, window)
                
                if self.include_acceleration:
                    col_features[f"{col}_accel_{window}M"] = self.compute_acceleration(series, window)
                
                if self.include_zscore:
                    col_features[f"{col}_zscore_{window}M"] = self.compute_rolling_zscore(series, window)
            
            features_list.append(col_features)
        
        result = pd.concat(features_list, axis=1)
        logger.info(f"Generated {len(result.columns)} momentum features from {len(columns)} columns")
        return result


def generate_all_momentum_features(df: pd.DataFrame,
                                   windows: List[int] = [3, 6, 12]) -> pd.DataFrame:
    """Convenience function to generate all momentum features."""
    generator = MomentumFeatureGenerator(windows=windows)
    return generator.generate_features(df)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test
    np.random.seed(42)
    dates = pd.date_range('2010-01-01', periods=100, freq='ME')
    df = pd.DataFrame({
        'feature1': np.random.randn(100).cumsum(),
        'feature2': np.random.randn(100),
    }, index=dates)
    
    features = generate_all_momentum_features(df)
    print(features.columns.tolist())
