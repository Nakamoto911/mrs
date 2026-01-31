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
    """Generates quintile-based features for regime detection robustly."""
    DEFAULT_VARIABLES = [
        'GS10', 'FEDFUNDS', 'TB3MS', 'GS1', 'GS5', 'BAA_10Y_Spread', 'AAA_10Y_Spread',
        'VIXCLSx', 'UNRATE', 'UMCSENTx', 'HOUST', 'M2REAL', 'INDPRO', 'PERMIT', 'DPCERA3M086SBEA'
    ]
    def __init__(self, n_quintiles=5, variables=None, encoding='one_hot', min_observations=60):
        self.n_quintiles = n_quintiles
        self.variables = variables or self.DEFAULT_VARIABLES
        self.encoding = encoding
        self.min_observations = min_observations

    def _compute_quintile_rank(self, series: pd.Series) -> pd.Series:
        """Robust expanding rank-based quintile assignment."""
        expanding = series.expanding(min_periods=self.min_observations)
        ranks = expanding.rank(method='min')
        counts = expanding.count()
        
        # Avoid division by zero
        score = (ranks - 1) / (counts - 1).replace(0, np.nan)
        
        # Map to 1-N, handling 1.0 boundary
        quintile = (score * self.n_quintiles).apply(np.floor) + 1
        return quintile.clip(1, self.n_quintiles)

    def generate_features(self, df: pd.DataFrame, variables=None) -> pd.DataFrame:
        available_vars = [v for v in (variables or self.variables) if v in df.columns]
        features_list = []
        for var in available_vars:
            quintiles = self._compute_quintile_rank(df[var])
            if self.encoding == 'one_hot':
                encoded = pd.concat([(quintiles == q).astype(float).rename(f"{var}_Q{q}") for q in range(1, self.n_quintiles+1)], axis=1)
                encoded.loc[quintiles.isna()] = np.nan
            else: encoded = pd.DataFrame(index=df.index)
            encoded[f"{var}_quintile"] = quintiles
            features_list.append(encoded)
        return pd.concat(features_list, axis=1) if features_list else pd.DataFrame(index=df.index)

def compute_quintile_metrics(y_true, y_pred, n_quantiles=5):
    """Computes Q5-Q1 spread and monotonicity."""
    common = pd.concat([y_true, y_pred], axis=1).dropna()
    if len(common) < n_quantiles * 2: return {'quintile_spread': 0.0, 'monotonicity': 0.0}
    
    # Use qcut for evaluation
    q = pd.qcut(common.iloc[:, 1].rank(method='first'), q=n_quantiles, labels=False) + 1
    returns = common.iloc[:, 0]
    
    means = [returns[q == i].mean() for i in range(1, n_quantiles+1)]
    spread = means[-1] - means[0]
    from scipy.stats import spearmanr
    mono = spearmanr(range(n_quantiles), means)[0]
    
    return {'quintile_spread': spread, 'monotonicity': mono, 'q_means': means}

def generate_all_quintile_features(df: pd.DataFrame, variables: Optional[List[str]] = None, n_quintiles: int = 5) -> pd.DataFrame:
    """Convenience function to generate all quintile features."""
    generator = QuintileFeatureGenerator(n_quintiles=n_quintiles, variables=variables)
    return generator.generate_features(df)
