"""
Cointegration Validation Module
===============================
Statistical testing and validation of cointegration relationships.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.stattools import adfuller, coint

logger = logging.getLogger(__name__)


class CointegrationStatus(Enum):
    """Status of a cointegration relationship."""
    VALIDATED = "validated"           # Passes Johansen test
    REJECTED = "rejected"             # Fails Johansen test
    UNSTABLE = "unstable"             # Passes in some periods, fails in others
    INSUFFICIENT_DATA = "insufficient"  # Not enough observations
    THEORY_OVERRIDE = "theory_override"  # Included despite failing test


@dataclass
class CointegrationTestResult:
    """Result of cointegration testing for a pair."""
    pair_name: str
    series1: str
    series2: str
    theory: str
    
    # Johansen test results
    johansen_trace_stat: float
    johansen_trace_cv_5pct: float
    johansen_max_eigen_stat: float
    johansen_max_eigen_cv_5pct: float
    
    # Engle-Granger test (alternative)
    eg_stat: float
    eg_pvalue: float
    
    # Diagnostics
    n_observations: int
    sample_start: pd.Timestamp
    sample_end: pd.Timestamp
    
    # Conclusion
    status: CointegrationStatus
    coint_rank: int  # 0, 1, or 2
    include_in_model: bool
    
    # ECT characteristics (if cointegrated)
    ect_half_life: Optional[float] = None  # Mean reversion speed
    ect_stationarity_pvalue: Optional[float] = None
    
    # Warnings
    warnings: List[str] = None
    
    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


@dataclass 
class RollingCointegrationResult:
    """Result of rolling cointegration stability analysis."""
    pair_name: str
    n_windows: int
    n_cointegrated: int
    stability_ratio: float  # % of windows where cointegration holds
    is_stable: bool  # stability_ratio > threshold


class CointegrationValidator:
    """
    Validates cointegration relationships with statistical rigor.
    
    Performs:
    1. Johansen trace and max-eigenvalue tests
    2. Engle-Granger two-step test (alternative)
    3. Rolling window stability analysis
    4. ECT stationarity verification
    """
    
    # Default theoretical pairs with economic rationale
    DEFAULT_PAIRS = [
        ("GDPC1", "M2SL", "quantity_theory", "M2 velocity stability"),
        ("GS10", "CPIAUCSL", "fisher_hypothesis", "Real rate stationarity"),
        ("INDPRO", "PAYEMS", "okun_law", "Output-employment linkage"),
        ("HOUST", "MORTGAGE30US", "housing_rates", "Housing demand elasticity"),
        ("SP500", "SP500_DIV", "gordon_growth", "Present value relation"),
        ("PCEC", "DSPIC96", "consumption_income", "Permanent income hypothesis"),
    ]
    
    # Pairs with strong theoretical priors (include even if test marginally fails)
    THEORY_OVERRIDE_PAIRS = [
        "consumption_income",  # Very strong theoretical backing
    ]
    
    # Class-level cache to avoid re-calculating expensive tests across models/folds
    # Key: (pair_name, start_date, end_date, n_obs)
    # Value: CointegrationTestResult
    _cache: Dict[Tuple, CointegrationTestResult] = {}
    
    def __init__(
        self,
        significance_level: float = 0.05,
        min_observations: int = 120,
        stability_threshold: float = 0.70,
        allow_theory_override: bool = True
    ):
        """
        Initialize validator.
        
        Args:
            significance_level: p-value threshold for tests
            min_observations: Minimum sample size for testing
            stability_threshold: Minimum ratio of windows with cointegration
            allow_theory_override: Whether to include pairs with strong priors
        """
        self.significance_level = significance_level
        self.min_observations = min_observations
        self.stability_threshold = stability_threshold
        self.allow_theory_override = allow_theory_override
    
    def test_pair(
        self,
        df: pd.DataFrame,
        series1: str,
        series2: str,
        pair_name: str,
        theory: str
    ) -> CointegrationTestResult:
        """
        Test a single pair for cointegration.
        
        Args:
            df: DataFrame containing both series
            series1: First series name
            series2: Second series name
            pair_name: Descriptive name for the pair
            theory: Economic theory behind the relationship
            
        Returns:
            CointegrationTestResult with all diagnostics
        """
        warnings = []
        
        # Check data availability
        if series1 not in df.columns or series2 not in df.columns:
            return CointegrationTestResult(
                pair_name=pair_name,
                series1=series1,
                series2=series2,
                theory=theory,
                johansen_trace_stat=np.nan,
                johansen_trace_cv_5pct=np.nan,
                johansen_max_eigen_stat=np.nan,
                johansen_max_eigen_cv_5pct=np.nan,
                eg_stat=np.nan,
                eg_pvalue=np.nan,
                n_observations=0,
                sample_start=pd.NaT,
                sample_end=pd.NaT,
                status=CointegrationStatus.INSUFFICIENT_DATA,
                coint_rank=0,
                include_in_model=False,
                warnings=[f"Series {series1} or {series2} not found in data"]
            )
        
        # Prepare data
        data = df[[series1, series2]].dropna()
        n_obs = len(data)
        
        # Check cache
        if n_obs > 0:
            start_date = data.index.min()
            end_date = data.index.max()
            # Use timestamp value for hashing if valid, else None
            start_ts = start_date.value if hasattr(start_date, 'value') else start_date
            end_ts = end_date.value if hasattr(end_date, 'value') else end_date
            
            cache_key = (pair_name, start_ts, end_ts, n_obs)
            
            if cache_key in self._cache:
                return self._cache[cache_key]
        else:
            cache_key = None
        
        if n_obs < self.min_observations:
            return CointegrationTestResult(
                pair_name=pair_name,
                series1=series1,
                series2=series2,
                theory=theory,
                johansen_trace_stat=np.nan,
                johansen_trace_cv_5pct=np.nan,
                johansen_max_eigen_stat=np.nan,
                johansen_max_eigen_cv_5pct=np.nan,
                eg_stat=np.nan,
                eg_pvalue=np.nan,
                n_observations=n_obs,
                sample_start=data.index.min(),
                sample_end=data.index.max(),
                status=CointegrationStatus.INSUFFICIENT_DATA,
                coint_rank=0,
                include_in_model=False,
                warnings=[f"Insufficient observations: {n_obs} < {self.min_observations}"]
            )
        
        # Johansen test
        try:
            # det_order=0: constant in cointegrating relation
            # k_ar_diff: number of lagged differences (use 2 as default)
            johansen_result = coint_johansen(data.values, det_order=0, k_ar_diff=2)
            
            trace_stat = johansen_result.lr1[0]  # r=0 vs r>=1
            trace_cv = johansen_result.cvt[0, 1]  # 5% critical value
            
            max_eigen_stat = johansen_result.lr2[0]
            max_eigen_cv = johansen_result.cvm[0, 1]
            
            # Determine cointegration rank
            if trace_stat > trace_cv:
                coint_rank = 1  # At least one cointegrating vector
            else:
                coint_rank = 0
                
        except Exception as e:
            logger.warning(f"Johansen test failed for {pair_name}: {e}")
            trace_stat = np.nan
            trace_cv = np.nan
            max_eigen_stat = np.nan
            max_eigen_cv = np.nan
            coint_rank = 0
            warnings.append(f"Johansen test error: {str(e)}")
        
        # Engle-Granger test (alternative)
        try:
            eg_stat, eg_pvalue, _ = coint(data[series1], data[series2])
        except Exception as e:
            logger.warning(f"Engle-Granger test failed for {pair_name}: {e}")
            eg_stat = np.nan
            eg_pvalue = np.nan
            warnings.append(f"Engle-Granger test error: {str(e)}")
        
        # Determine status
        johansen_passes = trace_stat > trace_cv if not np.isnan(trace_stat) else False
        eg_passes = eg_pvalue < self.significance_level if not np.isnan(eg_pvalue) else False
        
        # ECT analysis if cointegrated
        ect_half_life = None
        ect_stationarity_pvalue = None
        
        if johansen_passes or eg_passes:
            try:
                # Compute ECT as residual from cointegrating regression
                from sklearn.linear_model import LinearRegression
                lr = LinearRegression()
                lr.fit(data[[series2]], data[series1])
                ect = data[series1] - lr.predict(data[[series2]]).flatten()
                
                # Test ECT stationarity
                adf_result = adfuller(ect.dropna(), maxlag=12, autolag='AIC')
                ect_stationarity_pvalue = adf_result[1]
                
                # Estimate half-life of mean reversion
                # AR(1) on ECT: ect_t = rho * ect_{t-1} + epsilon
                ect_lag = ect.shift(1).dropna()
                ect_current = ect.iloc[1:]
                if len(ect_lag) > 10:
                    rho = np.corrcoef(ect_lag, ect_current)[0, 1]
                    if 0 < rho < 1:
                        ect_half_life = -np.log(2) / np.log(rho)
                    else:
                        ect_half_life = np.nan
                        warnings.append("ECT shows no mean reversion (rho >= 1)")
                        
            except Exception as e:
                warnings.append(f"ECT analysis error: {str(e)}")
        
        # Final status determination
        if johansen_passes and eg_passes:
            status = CointegrationStatus.VALIDATED
            include = True
        elif johansen_passes or eg_passes:
            status = CointegrationStatus.VALIDATED  # One test sufficient
            include = True
            warnings.append("Tests disagree; included based on one passing")
        elif self.allow_theory_override and pair_name in self.THEORY_OVERRIDE_PAIRS:
            status = CointegrationStatus.THEORY_OVERRIDE
            include = True
            warnings.append(f"Included due to strong theoretical prior despite failing tests")
        else:
            status = CointegrationStatus.REJECTED
            include = False
        
        # Additional warnings
        if johansen_passes and ect_stationarity_pvalue is not None:
            if ect_stationarity_pvalue > self.significance_level:
                warnings.append(
                    f"ECT not stationary (ADF p={ect_stationarity_pvalue:.3f}). "
                    "Cointegration may be spurious."
                )
        
        if ect_half_life is not None and ect_half_life > 60:
            warnings.append(
                f"ECT half-life is {ect_half_life:.0f} months. "
                "Very slow mean reversion may limit predictive value."
            )
        
        result = CointegrationTestResult(
            pair_name=pair_name,
            series1=series1,
            series2=series2,
            theory=theory,
            johansen_trace_stat=trace_stat,
            johansen_trace_cv_5pct=trace_cv,
            johansen_max_eigen_stat=max_eigen_stat,
            johansen_max_eigen_cv_5pct=max_eigen_cv,
            eg_stat=eg_stat,
            eg_pvalue=eg_pvalue,
            n_observations=n_obs,
            sample_start=data.index.min(),
            sample_end=data.index.max(),
            status=status,
            coint_rank=coint_rank,
            include_in_model=include,
            ect_half_life=ect_half_life,
            ect_stationarity_pvalue=ect_stationarity_pvalue,
            warnings=warnings
        )
        
        # Cache the result if we have a valid key
        if cache_key is not None:
            self._cache[cache_key] = result
            
        return result
    
    def test_all_pairs(
        self,
        df: pd.DataFrame,
        pairs: Optional[List[Tuple]] = None
    ) -> Dict[str, CointegrationTestResult]:
        """
        Test all specified pairs for cointegration.
        
        Args:
            df: DataFrame with all series
            pairs: List of (series1, series2, name, theory) tuples
            
        Returns:
            Dictionary mapping pair names to test results
        """
        if pairs is None:
            pairs = self.DEFAULT_PAIRS
        
        results = {}
        
        for pair_spec in pairs:
            if len(pair_spec) == 4:
                series1, series2, pair_name, theory = pair_spec
            else:
                series1, series2, pair_name = pair_spec[:3]
                theory = "unspecified"
            
            result = self.test_pair(df, series1, series2, pair_name, theory)
            results[pair_name] = result
            
            # Log result
            status_str = result.status.value
            logger.debug(
                f"Cointegration test: {pair_name} ({series1}, {series2}) -> "
                f"{status_str.upper()}"
            )
            
            if result.warnings:
                for w in result.warnings:
                    logger.debug(f"  {pair_name}: {w}")
        
        return results
    
    def rolling_stability_test(
        self,
        df: pd.DataFrame,
        series1: str,
        series2: str,
        pair_name: str,
        window_size: int = 120,
        step_size: int = 12
    ) -> RollingCointegrationResult:
        """
        Test cointegration stability over rolling windows.
        
        Args:
            df: DataFrame with series
            series1, series2: Series names
            pair_name: Pair identifier
            window_size: Rolling window size (months)
            step_size: Step between windows (months)
            
        Returns:
            RollingCointegrationResult with stability metrics
        """
        data = df[[series1, series2]].dropna()
        
        n_windows = 0
        n_cointegrated = 0
        
        for start in range(0, len(data) - window_size, step_size):
            window = data.iloc[start:start + window_size]
            
            try:
                _, pvalue, _ = coint(window[series1], window[series2])
                n_windows += 1
                if pvalue < self.significance_level:
                    n_cointegrated += 1
            except:
                continue
        
        stability_ratio = n_cointegrated / n_windows if n_windows > 0 else 0.0
        
        return RollingCointegrationResult(
            pair_name=pair_name,
            n_windows=n_windows,
            n_cointegrated=n_cointegrated,
            stability_ratio=stability_ratio,
            is_stable=stability_ratio >= self.stability_threshold
        )
    
    def get_validated_pairs(
        self,
        results: Dict[str, CointegrationTestResult]
    ) -> List[Tuple[str, str, str]]:
        """
        Get list of validated pairs to include in the model.
        
        Returns:
            List of (series1, series2, pair_name) tuples
        """
        validated = []
        for pair_name, result in results.items():
            if result.include_in_model:
                validated.append((result.series1, result.series2, pair_name))
        
        return validated


def format_cointegration_report(
    results: Dict[str, CointegrationTestResult]
) -> str:
    """Format cointegration test results as a report."""
    lines = [
        "=" * 70,
        "COINTEGRATION VALIDATION REPORT",
        "=" * 70,
        "",
        f"{'Pair':<25} {'Status':<15} {'Johansen':<12} {'E-G p-val':<10} {'Include':<8}",
        "-" * 70
    ]
    
    for pair_name, r in results.items():
        johansen_str = (
            f"{r.johansen_trace_stat:.1f}/{r.johansen_trace_cv_5pct:.1f}"
            if not np.isnan(r.johansen_trace_stat) else "N/A"
        )
        eg_str = f"{r.eg_pvalue:.3f}" if not np.isnan(r.eg_pvalue) else "N/A"
        include_str = "✓" if r.include_in_model else "✗"
        
        lines.append(
            f"{pair_name:<25} {r.status.value:<15} {johansen_str:<12} {eg_str:<10} {include_str:<8}"
        )
    
    lines.append("-" * 70)
    
    # Summary
    n_validated = sum(1 for r in results.values() if r.status == CointegrationStatus.VALIDATED)
    n_rejected = sum(1 for r in results.values() if r.status == CointegrationStatus.REJECTED)
    n_override = sum(1 for r in results.values() if r.status == CointegrationStatus.THEORY_OVERRIDE)
    
    lines.extend([
        "",
        f"Validated: {n_validated}  |  Rejected: {n_rejected}  |  Theory Override: {n_override}",
        ""
    ])
    
    # Warnings
    all_warnings = []
    for pair_name, r in results.items():
        for w in r.warnings:
            all_warnings.append(f"  [{pair_name}] {w}")
    
    if all_warnings:
        lines.append("Warnings:")
        lines.extend(all_warnings)
        lines.append("")
    
    lines.append("=" * 70)
    
    return "\n".join(lines)
