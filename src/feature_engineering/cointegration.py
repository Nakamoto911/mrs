"""
Cointegration Module
====================
Performs cointegration analysis and generates Error Correction Term features.

Part of the Asset-Specific Macro Regime Detection System

Cointegration captures long-run equilibrium relationships:
- Nominal GDP vs M2 (quantity theory of money)
- 10Y Yield vs CPI (Fisher hypothesis)
- Industrial Production vs Employment (Okun's law)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
import logging
import warnings

from sklearn.base import BaseEstimator, TransformerMixin

from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.stattools import coint

from .cointegration_validator import (
    CointegrationValidator,
    CointegrationTestResult,
    CointegrationStatus
)

logger = logging.getLogger(__name__)


@dataclass
class CointegrationResult:
    """Results from cointegration analysis."""
    series1: str
    series2: str
    pair_name: str
    is_cointegrated: bool
    n_cointegrating_vectors: int
    cointegrating_vector: Optional[np.ndarray]
    eigenvalues: np.ndarray
    trace_stats: np.ndarray
    trace_critical_values: np.ndarray
    max_eig_stats: np.ndarray
    max_eig_critical_values: np.ndarray


class CointegrationAnalyzer(BaseEstimator, TransformerMixin):
    """
    Performs Johansen cointegration tests and generates ECT features.
    Can be used as a scikit-learn transformer.
    """
    
    # Theory-driven pairs for cointegration testing
    DEFAULT_PAIRS = [
        # (series1, series2, name, economic_theory)
        ('INDPRO', 'M2SL', 'gdp_m2', 'quantity_theory'),
        ('RPI', 'M2SL', 'income_m2', 'quantity_theory'),
        ('GS10', 'CPIAUCSL', 'yields_inflation', 'fisher_hypothesis'),
        ('GS5', 'CPIAUCSL', 'yields5_inflation', 'fisher_hypothesis'),
        ('INDPRO', 'PAYEMS', 'output_employment', 'okun_law'),
        ('HOUST', 'GS10', 'housing_rates', 'housing_market'),
        ('PERMIT', 'GS10', 'permits_rates', 'housing_market'),
        ('S&P 500', 'INDPRO', 'stocks_gdp', 'equity_macro'),
        ('DPCERA3M086SBEA', 'RPI', 'consumption_income', 'permanent_income'),
    ]
    
    def __init__(self, 
                 pairs: Optional[List[Tuple]] = None,
                 max_lag: int = 12,
                 significance: float = 0.05,
                 det_order: int = 0,
                 validate: bool = True,
                 min_observations: int = 120,
                 stability_threshold: float = 0.70,
                 allow_theory_override: bool = True):
        """
        Initialize cointegration analyzer.
        
        Args:
            pairs: List of pairs to test (uses defaults if None)
            max_lag: Maximum lag order for Johansen test
            significance: Significance level for tests
            det_order: Deterministic term order (-1: no const/trend, 0: const, 1: const+trend)
            validate: Whether to statistically validate pairs before use
            min_observations: Minimum observations for validation
            stability_threshold: Threshold for rolling stability
            allow_theory_override: Whether to allow theory override for strong priors
        """
        self.pairs = pairs or self.DEFAULT_PAIRS
        self.max_lag = max_lag
        self.significance = significance
        self.det_order = det_order
        self.validate = validate
        
        self.validator = CointegrationValidator(
            significance_level=significance,
            min_observations=min_observations,
            stability_threshold=stability_threshold,
            allow_theory_override=allow_theory_override
        )
        
        # Store results
        self.results: Dict[str, CointegrationResult] = {}
        self.validation_results: Dict[str, CointegrationTestResult] = {}
        self.ect_features: Dict[str, pd.Series] = {}
        
        # Scikit-learn parameters
        self.validated_pairs: List[Tuple] = []
        self.vectors_: Dict[str, np.ndarray] = {}
        self.fitted_ = False
    
    def _get_critical_value_index(self) -> int:
        """Get index for critical values based on significance level."""
        if self.significance <= 0.01:
            return 2  # 99%
        elif self.significance <= 0.05:
            return 1  # 95%
        else:
            return 0  # 90%
    
    def test_pair_johansen(self, series1: pd.Series, series2: pd.Series,
                          name1: str, name2: str, pair_name: str) -> CointegrationResult:
        """
        Perform Johansen cointegration test on a pair.
        
        Args:
            series1: First series
            series2: Second series
            name1: Name of first series
            name2: Name of second series
            pair_name: Name for the pair
            
        Returns:
            CointegrationResult
        """
        # Combine and clean data
        data = pd.concat([series1, series2], axis=1).dropna()
        
        if len(data) < self.max_lag + 20:
            logger.warning(f"Insufficient data for {pair_name}: {len(data)} observations")
            return CointegrationResult(
                series1=name1,
                series2=name2,
                pair_name=pair_name,
                is_cointegrated=False,
                n_cointegrating_vectors=0,
                cointegrating_vector=None,
                eigenvalues=np.array([]),
                trace_stats=np.array([]),
                trace_critical_values=np.array([]),
                max_eig_stats=np.array([]),
                max_eig_critical_values=np.array([])
            )
        
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                
                # Johansen test
                result = coint_johansen(data.values, det_order=self.det_order, k_ar_diff=self.max_lag)
            
            # Get critical value index
            cv_idx = self._get_critical_value_index()
            
            # Count cointegrating vectors using trace statistic
            trace_stats = result.lr1
            trace_cv = result.cvt[:, cv_idx]
            n_coint = sum(trace_stats > trace_cv)
            
            is_cointegrated = n_coint > 0
            
            # Extract cointegrating vector if exists
            coint_vector = None
            if is_cointegrated:
                # Normalize by first element
                coint_vector = result.evec[:, 0]
                if coint_vector[0] != 0:
                    coint_vector = coint_vector / coint_vector[0]
            
            return CointegrationResult(
                series1=name1,
                series2=name2,
                pair_name=pair_name,
                is_cointegrated=is_cointegrated,
                n_cointegrating_vectors=n_coint,
                cointegrating_vector=coint_vector,
                eigenvalues=result.eig,
                trace_stats=trace_stats,
                trace_critical_values=trace_cv,
                max_eig_stats=result.lr2,
                max_eig_critical_values=result.cvm[:, cv_idx]
            )
            
        except Exception as e:
            logger.warning(f"Johansen test failed for {pair_name}: {e}")
            return CointegrationResult(
                series1=name1,
                series2=name2,
                pair_name=pair_name,
                is_cointegrated=False,
                n_cointegrating_vectors=0,
                cointegrating_vector=None,
                eigenvalues=np.array([]),
                trace_stats=np.array([]),
                trace_critical_values=np.array([]),
                max_eig_stats=np.array([]),
                max_eig_critical_values=np.array([])
            )
    
    def test_pair_engle_granger(self, series1: pd.Series, series2: pd.Series,
                                name1: str, name2: str, pair_name: str) -> Tuple[bool, float]:
        """
        Perform Engle-Granger cointegration test (simpler alternative).
        
        Args:
            series1: First series
            series2: Second series
            name1: Name of first series
            name2: Name of second series
            pair_name: Name for the pair
            
        Returns:
            Tuple of (is_cointegrated, p_value)
        """
        # Combine and clean
        data = pd.concat([series1, series2], axis=1).dropna()
        
        if len(data) < 20:
            return False, 1.0
        
        try:
            score, pvalue, _ = coint(data.iloc[:, 0], data.iloc[:, 1])
            is_cointegrated = pvalue < self.significance
            return is_cointegrated, pvalue
        except Exception as e:
            logger.warning(f"Engle-Granger test failed for {pair_name}: {e}")
            return False, 1.0
    
    def compute_ect(self, series1: pd.Series, series2: pd.Series,
                   coint_vector: np.ndarray, pair_name: str) -> pd.Series:
        """
        Compute Error Correction Term from cointegrating vector.
        
        ECT = series1 - β * series2
        
        Args:
            series1: First series
            series2: Second series
            coint_vector: Cointegrating vector [1, -β]
            pair_name: Name for output
            
        Returns:
            ECT series
        """
        # Combine data
        data = pd.concat([series1, series2], axis=1)
        
        # ECT = X1 - β*X2 (where coint_vector = [1, -β])
        beta = -coint_vector[1] if len(coint_vector) > 1 else 0
        ect = series1 - beta * series2
        ect.name = f"ECT_{pair_name}"
        
        return ect
    
    def analyze_pairs(self, df: pd.DataFrame,
                     pairs: Optional[List[Tuple]] = None) -> Dict[str, CointegrationResult]:
        """
        Analyze cointegration for all specified pairs.
        
        Args:
            df: DataFrame with level data
            pairs: Pairs to test (uses defaults if None)
            
        Returns:
            Dictionary of CointegrationResult
        """
        pairs = pairs or self.pairs
        self.results = {}
        
        for pair_config in pairs:
            if len(pair_config) >= 3:
                series1_name, series2_name, pair_name = pair_config[:3]
            else:
                continue
            
            # Check if series exist
            if series1_name not in df.columns or series2_name not in df.columns:
                logger.debug(f"Missing data for pair {pair_name}")
                continue
            
            # Test cointegration
            logger.debug(f"Testing cointegration for {pair_name}")
            result = self.test_pair_johansen(
                df[series1_name], df[series2_name],
                series1_name, series2_name, pair_name
            )
            
            self.results[pair_name] = result
            
            # Compute ECT if cointegrated
            if result.is_cointegrated and result.cointegrating_vector is not None:
                ect = self.compute_ect(
                    df[series1_name], df[series2_name],
                    result.cointegrating_vector, pair_name
                )
                self.ect_features[pair_name] = ect
        
        # Summary
        n_coint = sum(1 for r in self.results.values() if r.is_cointegrated)
        logger.debug(f"Found {n_coint}/{len(self.results)} cointegrated pairs")
        
        return self.results

    def fit(self, X: pd.DataFrame, y: Any = None):
        """
        Scikit-learn fit method. Performs cointegration validation.
        
        Args:
            X: DataFrame with level data
            y: Ignored
            
        Returns:
            self
        """
        if hasattr(X, 'columns'):
            self.feature_names_in_ = X.columns.tolist()
            
        # Reset state for fresh fit
        self.results = {}
        self.validation_results = {}
        self.vectors_ = {}
        self.ect_features = {}
            
        if self.validate:
            # Run validation tests
            self.validation_results = self.validator.test_all_pairs(X, self.pairs)
            
            # Get validated pairs only
            self.validated_pairs = self.validator.get_validated_pairs(
                self.validation_results
            )
            
            logger.debug(
                f"Cointegration validation: {len(self.validated_pairs)} of "
                f"{len(self.pairs)} pairs validated"
            )
        else:
            # Legacy behavior: use all pairs
            self.validated_pairs = [
                (p[0], p[1], p[2]) for p in self.pairs
            ]
            
        # Analyze the validated pairs (compute vectors)
        self.analyze_pairs(X, pairs=self.validated_pairs)
        
        # Store vectors for transform
        self.vectors_ = {}
        for name, res in self.results.items():
            if res.is_cointegrated and res.cointegrating_vector is not None:
                self.vectors_[name] = res.cointegrating_vector
        
        self.fitted_ = True
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Scikit-learn transform method.
        Computes ECT features using stored vectors.
        
        Args:
            X: DataFrame with level data
            
        Returns:
            DataFrame with original features PLUS ECT features
        """
        if not self.fitted_:
            raise ValueError("Transformer must be fitted before calling transform.")
            
        features = X.copy()
        
        for pair_name, coint_vector in self.vectors_.items():
            # Find the series names from results (matched by pair_name)
            if pair_name not in self.results:
                continue
                
            res = self.results[pair_name]
            series1_name = res.series1
            series2_name = res.series2
            
            if series1_name not in X.columns or series2_name not in X.columns:
                continue
            
            # Compute ECT
            ect = self.compute_ect(X[series1_name], X[series2_name], coint_vector, pair_name)
            
            # Level ECT
            features[f"ECT_{pair_name}"] = ect
            
            # Z-score of ECT (Expanding or fixed?)
            # User spec: "Computes Z-scores using expanding window statistics (to remain safe) or fixed mean/std from train set."
            # Expanding is safer for look-ahead.
            ect_zscore = (ect - ect.expanding().mean()) / ect.expanding().std()
            features[f"ECT_{pair_name}_zscore"] = ect_zscore
            
            # Add changes
            for window in [1, 3, 6]:
                features[f"ECT_{pair_name}_chg_{window}M"] = ect.diff(window)
        
        # Drop raw level columns to ensure downstream models only see stationary data
        cols_to_drop = [c for c in features.columns if c.endswith('_level')]
        if cols_to_drop:
            features = features.drop(columns=cols_to_drop)
            
        return features
    
    def generate_ect_features(self, df: pd.DataFrame,
                             include_changes: bool = True,
                             windows: List[int] = [1, 3, 6]) -> pd.DataFrame:
        """
        Generate ECT features for all cointegrated pairs.
        
        Args:
            df: DataFrame with level data
            include_changes: Whether to include ECT changes
            windows: Windows for change calculation
            
        Returns:
            DataFrame with ECT features
        """
        if not self.results:
            self.analyze_pairs(df)
        
        features = pd.DataFrame(index=df.index)
        
        for pair_name, ect in self.ect_features.items():
            # Align index
            ect_aligned = ect.reindex(df.index)
            
            # Level ECT
            features[f"ECT_{pair_name}"] = ect_aligned
            
            # Z-score of ECT
            ect_zscore = (ect_aligned - ect_aligned.expanding().mean()) / ect_aligned.expanding().std()
            features[f"ECT_{pair_name}_zscore"] = ect_zscore
            
            if include_changes:
                for window in windows:
                    features[f"ECT_{pair_name}_chg_{window}M"] = ect_aligned.diff(window)
        
        logger.info(f"Generated {len(features.columns)} ECT features")
        return features
    
    def summary_dataframe(self) -> pd.DataFrame:
        """
        Create summary DataFrame of cointegration results.
        
        Returns:
            Summary DataFrame
        """
        rows = []
        for pair_name, result in self.results.items():
            rows.append({
                'Pair': pair_name,
                'Series1': result.series1,
                'Series2': result.series2,
                'Cointegrated': result.is_cointegrated,
                'N_Vectors': result.n_cointegrating_vectors,
                'Trace_Stat': result.trace_stats[0] if len(result.trace_stats) > 0 else np.nan,
                'Trace_CV_95': result.trace_critical_values[0] if len(result.trace_critical_values) > 0 else np.nan,
                'Beta': -result.cointegrating_vector[1] if result.cointegrating_vector is not None and len(result.cointegrating_vector) > 1 else np.nan
            })
        
        return pd.DataFrame(rows).sort_values('Cointegrated', ascending=False)


def generate_cointegration_features(df: pd.DataFrame,
                                   pairs: Optional[List[Tuple]] = None,
                                   max_lag: int = 12,
                                   significance: float = 0.05) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience function to generate cointegration features.
    
    Args:
        df: DataFrame with level data
        pairs: Pairs to test
        max_lag: Maximum lag order
        significance: Significance level
        
    Returns:
        Tuple of (ECT features DataFrame, summary DataFrame)
    """
    analyzer = CointegrationAnalyzer(
        pairs=pairs,
        max_lag=max_lag,
        significance=significance
    )
    
    analyzer.analyze_pairs(df)
    features = analyzer.generate_ect_features(df)
    summary = analyzer.summary_dataframe()
    
    return features, summary


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create sample cointegrated data
    np.random.seed(42)
    n = 200
    dates = pd.date_range('2010-01-01', periods=n, freq='ME')
    
    # Generate cointegrated series: Y = 2*X + noise
    x = np.cumsum(np.random.randn(n))  # Random walk
    y = 2 * x + np.random.randn(n) * 0.5  # Cointegrated with X
    z = np.cumsum(np.random.randn(n))  # Independent random walk
    
    df = pd.DataFrame({
        'series_x': x,
        'series_y': y,
        'series_z': z,
    }, index=dates)
    
    # Test cointegration
    analyzer = CointegrationAnalyzer(
        pairs=[
            ('series_x', 'series_y', 'xy_pair', 'test'),
            ('series_x', 'series_z', 'xz_pair', 'test'),
        ]
    )
    
    results = analyzer.analyze_pairs(df)
    
    print("Cointegration Results:")
    print(analyzer.summary_dataframe())
    
    print("\nECT Features:")
    ect_features = analyzer.generate_ect_features(df)
    print(ect_features.tail())
