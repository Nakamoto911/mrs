"""
Quintile Features Module
========================
Creates regime-level quintile indicators for key variables.

Part of the Asset-Specific Macro Regime Detection System

Rationale: A 10Y yield declining from 5% (Q5) has different implications
than declining from 2% (Q1). Standard change features cannot capture this.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union
import logging

logger = logging.getLogger(__name__)


class QuintileFeatureGenerator:
    """
    Generates quintile-based features for regime detection.
    
    For each key variable, computes historical quintiles and creates
    either one-hot encoded indicators or z-scores within regime.
    """
    
    # Default key variables for quintile features
    DEFAULT_VARIABLES = [
        'GS10',          # 10-Year Treasury
        'FEDFUNDS',      # Fed Funds Rate
        'TB3MS',         # 3-Month Treasury
        'GS1',           # 1-Year Treasury
        'GS5',           # 5-Year Treasury
        'BAA_10Y_Spread', # Credit Spread (BAA-10Y)  
        'AAA_10Y_Spread', # Credit Spread (AAA-10Y)
        'VIXCLSx',       # VIX
        'UNRATE',        # Unemployment Rate
        'UMCSENTx',      # Consumer Sentiment
        'HOUST',         # Housing Starts
        'M2REAL',        # Real M2
        'INDPRO',        # Industrial Production
        'PERMIT',        # Building Permits
        'DPCERA3M086SBEA',  # Real Consumption
    ]
    
    def __init__(self, 
                 n_quintiles: int = 5,
                 variables: Optional[List[str]] = None,
                 encoding: str = 'one_hot',
                 expanding_window: bool = True,
                 min_observations: int = 60):
        """
        Initialize quintile feature generator.
        
        Args:
            n_quintiles: Number of quantile bins (default 5 for quintiles)
            variables: List of variables to create quintile features for
            encoding: 'one_hot' or 'z_score_within_regime'
            expanding_window: Use expanding window (True) or full sample (False)
            min_observations: Minimum observations before computing quintiles
        """
        self.n_quintiles = n_quintiles
        self.variables = variables or self.DEFAULT_VARIABLES
        self.encoding = encoding
        self.expanding_window = expanding_window
        self.min_observations = min_observations
        
        # Store quintile boundaries for interpretation
        self.quintile_boundaries: Dict[str, List[float]] = {}
    
    def _compute_quintile_rank(self, series: pd.Series) -> pd.Series:
        """
        Compute quintile rank for each observation.
        
        Args:
            series: Input time series
            
        Returns:
            Series with quintile ranks (1 to n_quintiles)
        """
        if self.expanding_window:
            # Use optimized expanding window to avoid look-ahead bias
            # Formula: Percentile of Score (Strict) = (Count of history < current) / len(history)
            # Using rank(method='min') gives 1 + count of elements strictly smaller.
            # So count(history < current) = rank - 1.
            # len(history) = total_count - 1.
            
            expanding = series.expanding(min_periods=self.min_observations)
            rank = expanding.rank(method='min')
            count = expanding.count()
            
            # Avoid division by zero (though min_observations >= 60)
            score = (rank - 1) / (count - 1).replace(0, np.nan)
            
            # Map to 1-5
            quintile = (score * self.n_quintiles).apply(np.floor) + 1
            return quintile.clip(upper=self.n_quintiles)
        else:
            # Use full sample (not recommended for production)
            quintiles = pd.qcut(
                series.rank(method='first'),
                q=self.n_quintiles,
                labels=range(1, self.n_quintiles + 1)
            )
            return quintiles.astype(float)
    
    def _one_hot_encode(self, quintile_series: pd.Series, 
                        base_name: str) -> pd.DataFrame:
        """
        One-hot encode quintile ranks.
        
        Args:
            quintile_series: Series with quintile ranks
            base_name: Base name for output columns
            
        Returns:
            DataFrame with one-hot encoded columns
        """
        cols = []
        
        for q in range(1, self.n_quintiles + 1):
            col_name = f"{base_name}_Q{q}"
            s = (quintile_series == q).astype(float)
            # Set NaN where original was NaN
            s.loc[quintile_series.isna()] = np.nan
            s.name = col_name
            cols.append(s)
        
        return pd.concat(cols, axis=1) if cols else pd.DataFrame(index=quintile_series.index)
    
    def _compute_z_score_within_regime(self, series: pd.Series,
                                       quintile_series: pd.Series,
                                       base_name: str) -> pd.DataFrame:
        """
        Compute z-score within each quintile regime.
        
        Args:
            series: Original value series
            quintile_series: Quintile rank series
            base_name: Base name for output columns
            
        Returns:
            DataFrame with regime-conditional z-scores
        """
        cols = []
        
        for q in range(1, self.n_quintiles + 1):
            col_name = f"{base_name}_Q{q}_zscore"
            
            # Mask for this quintile
            mask = quintile_series == q
            
            if mask.sum() > 1:
                # Z-score within quintile observations
                quintile_mean = series[mask].expanding().mean()
                quintile_std = series[mask].expanding().std()
                
                zscore = pd.Series(np.nan, index=series.index)
                zscore[mask] = (series[mask] - quintile_mean) / quintile_std.replace(0, np.nan)
                
                zscore.name = col_name
                cols.append(zscore)
        
        return pd.concat(cols, axis=1) if cols else pd.DataFrame(index=series.index)
    
    def generate_features(self, df: pd.DataFrame,
                         variables: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Generate quintile features for specified variables.
        
        Args:
            df: Input DataFrame with raw data
            variables: Variables to process (uses defaults if None)
            
        Returns:
            DataFrame with quintile features
        """
        variables = variables or self.variables
        
        # Filter to available variables
        available_vars = [v for v in variables if v in df.columns]
        missing_vars = [v for v in variables if v not in df.columns]
        
        if missing_vars:
            logger.info(f"Note: {len(missing_vars)} variables not found in vintage data (expected for early dates): {missing_vars[:5]}...")
        
        if not available_vars:
            logger.warning("No variables available for quintile features")
            return pd.DataFrame(index=df.index)
        
        features_list = []
        
        for var in available_vars:
            logger.debug(f"Generating quintile features for {var}")
            
            series = df[var]
            
            # Compute quintile ranks
            quintile_ranks = self._compute_quintile_rank(series)
            
            # Store boundaries for interpretation
            if not self.expanding_window:
                boundaries = series.quantile([i/self.n_quintiles 
                                             for i in range(self.n_quintiles + 1)]).tolist()
                self.quintile_boundaries[var] = boundaries
            
            # Generate encoded features
            if self.encoding == 'one_hot':
                features = self._one_hot_encode(quintile_ranks, var)
            elif self.encoding == 'z_score_within_regime':
                features = self._compute_z_score_within_regime(series, quintile_ranks, var)
            else:
                raise ValueError(f"Unknown encoding: {self.encoding}")
            
            # Also add raw quintile rank
            features[f"{var}_quintile"] = quintile_ranks
            
            features_list.append(features)
        
        result = pd.concat(features_list, axis=1)
        
        logger.info(f"Generated {len(result.columns)} quintile features from {len(available_vars)} variables")
        return result
    
    def generate_quintile_changes(self, quintile_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate features for quintile regime changes.
        
        Captures when a variable transitions from one quintile to another.
        
        Args:
            quintile_df: DataFrame with quintile features
            
        Returns:
            DataFrame with quintile change features
        """
        changes_list = []
        
        # Find quintile rank columns
        rank_cols = [c for c in quintile_df.columns if c.endswith('_quintile')]
        
        for col in rank_cols:
            base_name = col.replace('_quintile', '')
            
            # Quintile change (can be -4 to +4 for quintiles)
            chg = quintile_df[col].diff()
            chg.name = f"{base_name}_quintile_chg"
            changes_list.append(chg)
            
            # Regime shift indicators
            up = (quintile_df[col].diff() > 0).astype(float)
            up.name = f"{base_name}_regime_up"
            changes_list.append(up)
            
            down = (quintile_df[col].diff() < 0).astype(float)
            down.name = f"{base_name}_regime_down"
            changes_list.append(down)
            
            # Extreme regime indicators
            ex_high = (quintile_df[col] == self.n_quintiles).astype(float)
            ex_high.name = f"{base_name}_extreme_high"
            changes_list.append(ex_high)
            
            ex_low = (quintile_df[col] == 1).astype(float)
            ex_low.name = f"{base_name}_extreme_low"
            changes_list.append(ex_low)
        
        return pd.concat(changes_list, axis=1) if changes_list else pd.DataFrame(index=quintile_df.index)
    
    def get_current_regimes(self, df: pd.DataFrame,
                           variables: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get current quintile regime for each variable.
        
        Useful for monitoring dashboard.
        
        Args:
            df: Input DataFrame
            variables: Variables to check
            
        Returns:
            DataFrame with current regime status
        """
        variables = variables or self.variables
        available_vars = [v for v in variables if v in df.columns]
        
        regimes = []
        
        for var in available_vars:
            series = df[var]
            quintile = self._compute_quintile_rank(series)
            
            current_quintile = quintile.iloc[-1] if not quintile.isna().iloc[-1] else np.nan
            current_value = series.iloc[-1]
            
            # Compute percentile
            pct = (series.iloc[:-1] < current_value).mean() * 100
            
            regimes.append({
                'Variable': var,
                'Current_Value': current_value,
                'Quintile': current_quintile,
                'Percentile': pct,
                'Regime': self._quintile_to_regime(current_quintile)
            })
        
        return pd.DataFrame(regimes)
    
    def _quintile_to_regime(self, quintile: float) -> str:
        """Convert quintile number to regime label."""
        if pd.isna(quintile):
            return 'Unknown'
        
        labels = {
            1: 'Very Low',
            2: 'Low', 
            3: 'Neutral',
            4: 'High',
            5: 'Very High'
        }
        
        return labels.get(int(quintile), 'Unknown')


def generate_all_quintile_features(df: pd.DataFrame,
                                   variables: Optional[List[str]] = None,
                                   n_quintiles: int = 5,
                                   include_changes: bool = True) -> pd.DataFrame:
    """
    Convenience function to generate all quintile features.
    
    Args:
        df: Input DataFrame
        variables: Variables to process
        n_quintiles: Number of quantile bins
        include_changes: Whether to include change features
        
    Returns:
        DataFrame with all quintile features
    """
    generator = QuintileFeatureGenerator(
        n_quintiles=n_quintiles,
        variables=variables,
        encoding='one_hot'
    )
    
    features = generator.generate_features(df, variables)
    
    if include_changes:
        changes = generator.generate_quintile_changes(features)
        features = pd.concat([features, changes], axis=1)
    
    return features


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    np.random.seed(42)
    n = 200
    dates = pd.date_range('2010-01-01', periods=n, freq='ME')
    
    df = pd.DataFrame({
        'GS10': 2 + np.cumsum(np.random.randn(n) * 0.1),
        'FEDFUNDS': np.clip(1 + np.cumsum(np.random.randn(n) * 0.05), 0, 10),
        'VIXCLSx': 15 + np.abs(np.random.randn(n)) * 10,
        'UNRATE': 5 + np.cumsum(np.random.randn(n) * 0.1),
    }, index=dates)
    
    # Generate features
    generator = QuintileFeatureGenerator(encoding='one_hot')
    features = generator.generate_features(df, ['GS10', 'FEDFUNDS', 'VIXCLSx', 'UNRATE'])
    
    print("Generated features:")
    print(features.columns.tolist())
    print("\nSample:")
    print(features.tail())
    
    # Current regimes
    print("\nCurrent Regimes:")
    print(generator.get_current_regimes(df, ['GS10', 'FEDFUNDS', 'VIXCLSx', 'UNRATE']))
