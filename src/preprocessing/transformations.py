"""
Transformations Module
======================
Applies FRED-MD transformation codes and custom transformations.

Part of the Asset-Specific Macro Regime Detection System
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


class FREDMDTransformer:
    """
    Applies FRED-MD transformation codes to make series stationary.
    
    Transformation codes from McCracken & Ng (2016):
    1 - no transformation
    2 - first difference: Δx_t
    3 - second difference: Δ²x_t
    4 - log: log(x_t)
    5 - log first difference: Δlog(x_t) (growth rate)
    6 - log second difference: Δ²log(x_t) (acceleration)
    7 - percent change: (x_t/x_{t-1} - 1)
    """
    
    TRANSFORM_NAMES = {
        1: "none",
        2: "first_diff",
        3: "second_diff",
        4: "log",
        5: "log_first_diff",
        6: "log_second_diff",
        7: "pct_change"
    }
    
    def __init__(self, transform_codes: Optional[Dict[str, int]] = None):
        """
        Initialize transformer.
        
        Args:
            transform_codes: Dictionary mapping variable names to transform codes
        """
        self.transform_codes = transform_codes or {}
    
    def set_transform_codes(self, codes: Dict[str, int]):
        """Set transformation codes."""
        self.transform_codes = codes
    
    def _safe_log(self, series: pd.Series) -> pd.Series:
        """
        Compute log, handling non-positive values.
        
        Args:
            series: Input series
            
        Returns:
            Log of series (NaN for non-positive values)
        """
        return np.log(series.replace(0, np.nan).clip(lower=1e-10))
    
    def transform_series(self, series: pd.Series, code: int) -> pd.Series:
        """
        Apply transformation code to a series.
        
        Args:
            series: Input series
            code: Transformation code (1-7)
            
        Returns:
            Transformed series
        """
        if code == 1:
            # No transformation
            return series
        elif code == 2:
            # First difference
            return series.diff()
        elif code == 3:
            # Second difference
            return series.diff().diff()
        elif code == 4:
            # Log
            return self._safe_log(series)
        elif code == 5:
            # Log first difference (growth rate)
            return self._safe_log(series).diff()
        elif code == 6:
            # Log second difference (acceleration)
            return self._safe_log(series).diff().diff()
        elif code == 7:
            # Percent change
            return series.pct_change()
        else:
            logger.warning(f"Unknown transform code {code}, using no transformation")
            return series
    
    def transform_dataframe(self, df: pd.DataFrame, 
                           preserve_levels: bool = False) -> pd.DataFrame:
        """
        Apply transformations to all columns in DataFrame.
        
        Args:
            df: Input DataFrame
            preserve_levels: If True, also return level-stationary series
            
        Returns:
            Transformed DataFrame
        """
        transformed_cols = []
        levels_cols = []
        
        for col in df.columns:
            code = self.transform_codes.get(col, 1)
            
            try:
                s = self.transform_series(df[col], code)
                s.name = col
                transformed_cols.append(s)
                
                # Preserve levels for stationary series (code 1)
                if preserve_levels and code == 1:
                    l = df[col].copy()
                    l.name = col
                    levels_cols.append(l)
                    
            except Exception as e:
                logger.warning(f"Error transforming {col}: {e}")
                s = pd.Series(np.nan, index=df.index, name=col)
                transformed_cols.append(s)
        
        transformed = pd.concat(transformed_cols, axis=1) if transformed_cols else pd.DataFrame(index=df.index)
        
        if preserve_levels:
            levels = pd.concat(levels_cols, axis=1) if levels_cols else pd.DataFrame(index=df.index)
            return transformed, levels
        return transformed
    
    def inverse_transform_series(self, series: pd.Series, code: int, 
                                 original: Optional[pd.Series] = None) -> pd.Series:
        """
        Inverse transformation (for forecasts).
        
        Note: Some transformations are not invertible without original data.
        
        Args:
            series: Transformed series
            code: Original transformation code
            original: Original untransformed series (needed for some transforms)
            
        Returns:
            Inverse-transformed series
        """
        if code == 1:
            return series
        elif code == 2:
            # Inverse of first difference: cumsum
            if original is not None:
                return series.cumsum() + original.iloc[0]
            return series.cumsum()
        elif code == 3:
            # Inverse of second difference: double cumsum
            if original is not None:
                first_diff = series.cumsum() + original.diff().iloc[1]
                return first_diff.cumsum() + original.iloc[0]
            return series.cumsum().cumsum()
        elif code == 4:
            # Inverse of log
            return np.exp(series)
        elif code == 5:
            # Inverse of log diff
            if original is not None:
                return np.exp(series.cumsum() + np.log(original.iloc[0]))
            return np.exp(series.cumsum())
        elif code == 6:
            # Inverse of log second diff
            if original is not None:
                log_diff = series.cumsum() + np.log(original).diff().iloc[1]
                return np.exp(log_diff.cumsum() + np.log(original.iloc[0]))
            return np.exp(series.cumsum().cumsum())
        elif code == 7:
            # Inverse of pct change
            if original is not None:
                return (1 + series).cumprod() * original.iloc[0]
            return (1 + series).cumprod()
        else:
            return series


class CustomTransformer:
    """
    Custom transformations for additional feature engineering.
    """
    
    @staticmethod
    def compute_growth_rate(series: pd.Series, periods: int = 1, 
                           annualize: bool = False) -> pd.Series:
        """
        Compute growth rate (log difference).
        
        Args:
            series: Price/level series
            periods: Number of periods
            annualize: Whether to annualize
            
        Returns:
            Growth rate series
        """
        growth = np.log(series / series.shift(periods))
        
        if annualize:
            growth = growth * (12 / periods)  # Assuming monthly data
        
        return growth
    
    @staticmethod
    def compute_acceleration(series: pd.Series, periods: int = 1) -> pd.Series:
        """
        Compute acceleration (second difference of log).
        
        Args:
            series: Price/level series
            periods: Number of periods
            
        Returns:
            Acceleration series
        """
        log_series = np.log(series.replace(0, np.nan))
        return log_series.diff(periods).diff(periods)
    
    @staticmethod
    def compute_momentum(series: pd.Series, fast_window: int = 3, 
                        slow_window: int = 12) -> pd.Series:
        """
        Compute momentum as difference between fast and slow moving averages.
        
        Args:
            series: Input series
            fast_window: Fast MA window
            slow_window: Slow MA window
            
        Returns:
            Momentum series
        """
        fast_ma = series.rolling(window=fast_window).mean()
        slow_ma = series.rolling(window=slow_window).mean()
        return fast_ma - slow_ma
    
    @staticmethod
    def compute_z_score(series: pd.Series, window: Optional[int] = None) -> pd.Series:
        """
        Compute z-score (standardization).
        
        Args:
            series: Input series
            window: Rolling window (full sample if None)
            
        Returns:
            Z-score series
        """
        if window is None:
            mean = series.expanding().mean()
            std = series.expanding().std()
        else:
            mean = series.rolling(window=window).mean()
            std = series.rolling(window=window).std()
        
        return (series - mean) / std.replace(0, np.nan)
    
    @staticmethod
    def compute_percentile(series: pd.Series, window: Optional[int] = None) -> pd.Series:
        """
        Compute rolling percentile rank.
        
        Args:
            series: Input series
            window: Rolling window (full sample if None)
            
        Returns:
            Percentile rank series (0-100)
        """
        def percentile_rank(x):
            if len(x) < 2:
                return np.nan
            return (x.rank().iloc[-1] - 1) / (len(x) - 1) * 100
        
        if window is None:
            return series.expanding().apply(percentile_rank, raw=False)
        else:
            return series.rolling(window=window).apply(percentile_rank, raw=False)
    
    @staticmethod
    def compute_volatility(series: pd.Series, window: int = 6, 
                          annualize: bool = True) -> pd.Series:
        """
        Compute rolling volatility.
        
        Args:
            series: Return series
            window: Rolling window
            annualize: Whether to annualize
            
        Returns:
            Volatility series
        """
        vol = series.rolling(window=window).std()
        
        if annualize:
            vol = vol * np.sqrt(12)  # Assuming monthly data
        
        return vol
    
    @staticmethod
    def compute_drawdown(series: pd.Series) -> pd.Series:
        """
        Compute drawdown from peak.
        
        Args:
            series: Price/level series
            
        Returns:
            Drawdown series (negative values)
        """
        rolling_max = series.expanding().max()
        drawdown = (series - rolling_max) / rolling_max
        return drawdown
    
    @staticmethod
    def compute_rate_of_change(series: pd.Series, periods: int = 12) -> pd.Series:
        """
        Compute rate of change.
        
        Args:
            series: Input series
            periods: Number of periods
            
        Returns:
            Rate of change series
        """
        return (series - series.shift(periods)) / series.shift(periods).replace(0, np.nan)


def standardize_features(df: pd.DataFrame, method: str = 'zscore',
                        window: Optional[int] = None) -> pd.DataFrame:
    """
    Standardize all features in DataFrame.
    
    Args:
        df: Input DataFrame
        method: 'zscore', 'minmax', or 'robust'
        window: Rolling window (full sample if None)
        
    Returns:
        Standardized DataFrame
    """
    cols = []
    
    for col in df.columns:
        series = df[col]
        
        if method == 'zscore':
            if window is None:
                mean = series.expanding().mean()
                std = series.expanding().std()
            else:
                mean = series.rolling(window=window).mean()
                std = series.rolling(window=window).std()
            s = (series - mean) / std.replace(0, np.nan)
            
        elif method == 'minmax':
            if window is None:
                min_val = series.expanding().min()
                max_val = series.expanding().max()
            else:
                min_val = series.rolling(window=window).min()
                max_val = series.rolling(window=window).max()
            range_val = max_val - min_val
            s = (series - min_val) / range_val.replace(0, np.nan)
            
        elif method == 'robust':
            if window is None:
                median = series.expanding().median()
                iqr = series.expanding().quantile(0.75) - series.expanding().quantile(0.25)
            else:
                median = series.rolling(window=window).median()
                iqr = series.rolling(window=window).quantile(0.75) - series.rolling(window=window).quantile(0.25)
            s = (series - median) / iqr.replace(0, np.nan)
        
        else:
            raise ValueError(f"Unknown standardization method: {method}")
            
        s.name = col
        cols.append(s)
    
    return pd.concat(cols, axis=1) if cols else pd.DataFrame(index=df.index)


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Generate test data
    np.random.seed(42)
    n = 100
    
    df = pd.DataFrame({
        'level': np.random.randn(n).cumsum() + 100,  # Random walk with drift
        'growth': np.random.randn(n) * 0.01,  # Already a growth rate
        'rate': np.random.randn(n) * 0.5 + 5,  # Interest rate-like
    })
    
    # Test FRED-MD transformer
    transformer = FREDMDTransformer({
        'level': 5,   # Log diff
        'growth': 1,  # No transform
        'rate': 2,    # First diff
    })
    
    transformed = transformer.transform_dataframe(df)
    print("Transformed data:")
    print(transformed.head(10))
    
    # Test custom transformations
    custom = CustomTransformer()
    
    series = df['level']
    print(f"\nOriginal series (first 5): {series.head().values}")
    print(f"Growth rate: {custom.compute_growth_rate(series).head().values}")
    print(f"Z-score: {custom.compute_z_score(series).head().values}")
    print(f"Percentile: {custom.compute_percentile(series).head().values}")
