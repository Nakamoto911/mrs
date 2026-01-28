"""
Stationarity Testing Module
===========================
Implements ADF and KPSS tests for stationarity detection.

Part of the Asset-Specific Macro Regime Detection System
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

from statsmodels.tsa.stattools import adfuller, kpss

logger = logging.getLogger(__name__)


@dataclass
class StationarityResult:
    """Results from stationarity tests."""
    series_name: str
    adf_statistic: float
    adf_pvalue: float
    adf_critical_values: Dict[str, float]
    adf_is_stationary: bool
    kpss_statistic: float
    kpss_pvalue: float
    kpss_critical_values: Dict[str, float]
    kpss_is_stationary: bool
    is_stationary: bool  # Combined conclusion
    recommendation: str


class StationarityTester:
    """
    Performs stationarity tests on time series.
    
    Uses both ADF (null: unit root) and KPSS (null: stationary) tests
    for robust stationarity detection.
    """
    
    def __init__(self, adf_significance: float = 0.05, kpss_significance: float = 0.05):
        """
        Initialize stationarity tester.
        
        Args:
            adf_significance: Significance level for ADF test
            kpss_significance: Significance level for KPSS test
        """
        self.adf_significance = adf_significance
        self.kpss_significance = kpss_significance
    
    def test_adf(self, series: pd.Series, maxlag: Optional[int] = None,
                 regression: str = 'c') -> Tuple[float, float, Dict[str, float], bool]:
        """
        Perform Augmented Dickey-Fuller test.
        
        H0: Unit root exists (non-stationary)
        H1: No unit root (stationary)
        
        Reject H0 if p-value < significance → Series is stationary
        
        Args:
            series: Time series to test
            maxlag: Maximum lag order (auto-selected if None)
            regression: Regression type ('c' for constant, 'ct' for constant + trend)
            
        Returns:
            Tuple of (statistic, p-value, critical_values, is_stationary)
        """
        # Remove NaN values
        clean_series = series.dropna()
        
        if len(clean_series) < 20:
            logger.warning(f"Series too short for ADF test: {len(clean_series)} observations")
            return np.nan, 1.0, {}, False
        
        try:
            result = adfuller(clean_series, maxlag=maxlag, regression=regression, autolag='AIC')
            
            statistic = result[0]
            pvalue = result[1]
            critical_values = result[4]
            
            is_stationary = pvalue < self.adf_significance
            
            return statistic, pvalue, critical_values, is_stationary
        
        except Exception as e:
            logger.warning(f"ADF test failed: {e}")
            return np.nan, 1.0, {}, False
    
    def test_kpss(self, series: pd.Series, regression: str = 'c',
                  nlags: str = 'auto') -> Tuple[float, float, Dict[str, float], bool]:
        """
        Perform KPSS test.
        
        H0: Series is stationary
        H1: Series has unit root (non-stationary)
        
        Reject H0 if p-value < significance → Series is NON-stationary
        
        Args:
            series: Time series to test
            regression: 'c' for level stationary, 'ct' for trend stationary
            nlags: Number of lags or 'auto'
            
        Returns:
            Tuple of (statistic, p-value, critical_values, is_stationary)
        """
        # Remove NaN values
        clean_series = series.dropna()
        
        if len(clean_series) < 20:
            logger.warning(f"Series too short for KPSS test: {len(clean_series)} observations")
            return np.nan, 0.0, {}, False
        
        try:
            # Suppress KPSS interpolation warning
            import warnings
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', message='.*p-value is smaller than.*')
                warnings.filterwarnings('ignore', message='.*p-value is greater than.*')
                
                result = kpss(clean_series, regression=regression, nlags=nlags)
            
            statistic = result[0]
            pvalue = result[1]
            critical_values = result[3]
            
            # KPSS: FAIL to reject H0 means stationary
            # i.e., stationary if p-value >= significance
            is_stationary = pvalue >= self.kpss_significance
            
            return statistic, pvalue, critical_values, is_stationary
        
        except Exception as e:
            logger.warning(f"KPSS test failed: {e}")
            return np.nan, 0.0, {}, False
    
    def test_series(self, series: pd.Series, name: Optional[str] = None) -> StationarityResult:
        """
        Perform both ADF and KPSS tests and provide combined conclusion.
        
        Decision matrix:
        - ADF: Stationary, KPSS: Stationary → Stationary (high confidence)
        - ADF: Stationary, KPSS: Not Stationary → Trend stationary (difference recommended)
        - ADF: Not Stationary, KPSS: Stationary → Difference stationary (needs transform)
        - ADF: Not Stationary, KPSS: Not Stationary → Non-stationary (needs transform)
        
        Args:
            series: Time series to test
            name: Series name (uses series.name if None)
            
        Returns:
            StationarityResult with test results and recommendation
        """
        name = name or series.name or "unknown"
        
        # Run ADF test
        adf_stat, adf_pval, adf_crit, adf_stationary = self.test_adf(series)
        
        # Run KPSS test
        kpss_stat, kpss_pval, kpss_crit, kpss_stationary = self.test_kpss(series)
        
        # Combined conclusion and recommendation
        if adf_stationary and kpss_stationary:
            is_stationary = True
            recommendation = "Stationary - no transformation needed"
        elif adf_stationary and not kpss_stationary:
            is_stationary = False
            recommendation = "Trend stationary - consider detrending or differencing"
        elif not adf_stationary and kpss_stationary:
            is_stationary = False
            recommendation = "Difference stationary - apply first difference"
        else:
            is_stationary = False
            recommendation = "Non-stationary - apply first or second difference"
        
        return StationarityResult(
            series_name=name,
            adf_statistic=adf_stat,
            adf_pvalue=adf_pval,
            adf_critical_values=adf_crit,
            adf_is_stationary=adf_stationary,
            kpss_statistic=kpss_stat,
            kpss_pvalue=kpss_pval,
            kpss_critical_values=kpss_crit,
            kpss_is_stationary=kpss_stationary,
            is_stationary=is_stationary,
            recommendation=recommendation
        )
    
    def test_dataframe(self, df: pd.DataFrame, n_jobs: int = -1,
                       progress: bool = True) -> Dict[str, StationarityResult]:
        """
        Test all columns in a DataFrame for stationarity.
        
        Args:
            df: DataFrame with time series as columns
            n_jobs: Number of parallel jobs (-1 for all cores)
            progress: Whether to show progress
            
        Returns:
            Dictionary mapping column names to StationarityResult
        """
        logger.info(f"Testing {len(df.columns)} series for stationarity...")
        
        results = {}
        
        if n_jobs == 1:
            # Sequential execution
            for i, col in enumerate(df.columns):
                if progress and i % 20 == 0:
                    logger.info(f"Progress: {i}/{len(df.columns)}")
                results[col] = self.test_series(df[col], col)
        else:
            # Parallel execution
            import multiprocessing
            if n_jobs == -1:
                n_jobs = multiprocessing.cpu_count()
            
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                futures = {
                    executor.submit(self.test_series, df[col], col): col
                    for col in df.columns
                }
                
                completed = 0
                for future in as_completed(futures):
                    col = futures[future]
                    try:
                        results[col] = future.result()
                    except Exception as e:
                        logger.warning(f"Error testing {col}: {e}")
                    
                    completed += 1
                    if progress and completed % 20 == 0:
                        logger.info(f"Progress: {completed}/{len(df.columns)}")
        
        return results
    
    def get_stationary_columns(self, results: Dict[str, StationarityResult]) -> List[str]:
        """
        Get list of columns that are stationary in levels.
        
        Args:
            results: Dictionary of stationarity results
            
        Returns:
            List of stationary column names
        """
        return [name for name, result in results.items() if result.is_stationary]
    
    def get_nonstationary_columns(self, results: Dict[str, StationarityResult]) -> List[str]:
        """
        Get list of columns that are not stationary in levels.
        
        Args:
            results: Dictionary of stationarity results
            
        Returns:
            List of non-stationary column names
        """
        return [name for name, result in results.items() if not result.is_stationary]
    
    def summary_dataframe(self, results: Dict[str, StationarityResult]) -> pd.DataFrame:
        """
        Create summary DataFrame of stationarity test results.
        
        Args:
            results: Dictionary of stationarity results
            
        Returns:
            Summary DataFrame
        """
        rows = []
        for name, result in results.items():
            rows.append({
                'Series': result.series_name,
                'ADF Statistic': result.adf_statistic,
                'ADF p-value': result.adf_pvalue,
                'ADF Stationary': result.adf_is_stationary,
                'KPSS Statistic': result.kpss_statistic,
                'KPSS p-value': result.kpss_pvalue,
                'KPSS Stationary': result.kpss_is_stationary,
                'Overall Stationary': result.is_stationary,
                'Recommendation': result.recommendation
            })
        
        return pd.DataFrame(rows).sort_values('Overall Stationary', ascending=False)


def identify_level_stationary_features(df: pd.DataFrame, 
                                       adf_significance: float = 0.05,
                                       kpss_significance: float = 0.05) -> Tuple[List[str], pd.DataFrame]:
    """
    Convenience function to identify level-stationary features.
    
    Args:
        df: DataFrame with features
        adf_significance: ADF test significance level
        kpss_significance: KPSS test significance level
        
    Returns:
        Tuple of (list of stationary column names, summary DataFrame)
    """
    tester = StationarityTester(adf_significance, kpss_significance)
    results = tester.test_dataframe(df)
    
    stationary_cols = tester.get_stationary_columns(results)
    summary = tester.summary_dataframe(results)
    
    logger.info(f"Found {len(stationary_cols)} stationary columns out of {len(df.columns)}")
    
    return stationary_cols, summary


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Generate test data
    np.random.seed(42)
    n = 200
    
    # Stationary series (white noise)
    stationary = pd.Series(np.random.randn(n), name='stationary')
    
    # Non-stationary series (random walk)
    random_walk = pd.Series(np.cumsum(np.random.randn(n)), name='random_walk')
    
    # Trend stationary
    trend_stationary = pd.Series(np.arange(n) * 0.1 + np.random.randn(n), name='trend_stationary')
    
    # Test
    tester = StationarityTester()
    
    print("Testing stationary series:")
    result = tester.test_series(stationary)
    print(f"  Is stationary: {result.is_stationary}")
    print(f"  Recommendation: {result.recommendation}")
    
    print("\nTesting random walk:")
    result = tester.test_series(random_walk)
    print(f"  Is stationary: {result.is_stationary}")
    print(f"  Recommendation: {result.recommendation}")
    
    print("\nTesting trend stationary:")
    result = tester.test_series(trend_stationary)
    print(f"  Is stationary: {result.is_stationary}")
    print(f"  Recommendation: {result.recommendation}")
