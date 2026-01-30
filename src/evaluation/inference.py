"""
Statistical Inference Module for Overlapping Observations
=========================================================
Implements Newey-West, Hansen-Hodrick, and block bootstrap methods
for valid inference with overlapping forecast horizons.
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import spearmanr, t as t_dist
from typing import Tuple, Optional, Dict
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class InferenceResult:
    """Container for inference results with overlapping observations."""
    estimate: float           # Point estimate (IC or correlation)
    se_raw: float            # Naive standard error (biased)
    se_nw: float             # Newey-West adjusted standard error
    se_hh: Optional[float]   # Hansen-Hodrick standard error (if computed)
    t_stat_nw: float         # t-statistic using NW SE
    p_value_nw: float        # Two-sided p-value using NW SE
    ci_lower: float          # 95% CI lower bound
    ci_upper: float          # 95% CI upper bound
    effective_n: int         # Effective sample size estimate
    n_obs: int               # Actual number of observations
    horizon: int             # Forecast horizon (months)
    method: str              # Primary method used


def compute_newey_west_se(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizon: int,
    kernel: str = 'bartlett'
) -> Tuple[float, float]:
    """
    Compute Newey-West standard error for correlation/IC.
    
    The Newey-West estimator accounts for heteroskedasticity and 
    autocorrelation up to a specified lag (here, horizon - 1).
    
    Args:
        y_true: Realized values
        y_pred: Predicted values
        horizon: Forecast horizon (determines max lag)
        kernel: Kernel type ('bartlett', 'parzen', 'quadratic')
        
    Returns:
        Tuple of (standard_error, variance)
        
    References:
        Newey, W.K. and West, K.D. (1987). "A Simple, Positive Semi-Definite,
        Heteroskedasticity and Autocorrelation Consistent Covariance Matrix."
        Econometrica, 55(3), 703-708.
    """
    n = len(y_true)
    max_lag = horizon - 1  # h-1 lags for h-period overlapping returns
    
    # Check for constant input
    if np.all(y_true == y_true[0]) or np.all(y_pred == y_pred[0]):
        return np.nan, np.nan
        
    # Compute ranks for Spearman correlation
    rank_true = stats.rankdata(y_true)
    rank_pred = stats.rankdata(y_pred)
    
    # Standardize ranks
    rank_true = (rank_true - rank_true.mean()) / rank_true.std()
    rank_pred = (rank_pred - rank_pred.mean()) / rank_pred.std()
    
    # Cross-product (influence function for correlation)
    z = rank_true * rank_pred
    z_centered = z - z.mean()
    
    # Compute autocovariances
    gamma = np.zeros(max_lag + 1)
    for j in range(max_lag + 1):
        if j == 0:
            gamma[j] = np.mean(z_centered ** 2)
        else:
            gamma[j] = np.mean(z_centered[j:] * z_centered[:-j])
    
    # Apply kernel weights
    if kernel == 'bartlett':
        weights = 1 - np.arange(max_lag + 1) / (max_lag + 1)
    elif kernel == 'parzen':
        k = np.arange(max_lag + 1) / (max_lag + 1)
        weights = np.where(
            k <= 0.5,
            1 - 6 * k**2 + 6 * k**3,
            2 * (1 - k)**3
        )
    elif kernel == 'quadratic':
        k = np.arange(max_lag + 1) / (max_lag + 1)
        weights = 25 / (12 * np.pi**2 * k**2) * (
            np.sin(6 * np.pi * k / 5) / (6 * np.pi * k / 5) - 
            np.cos(6 * np.pi * k / 5)
        )
        weights[0] = 1.0
    else:
        raise ValueError(f"Unknown kernel: {kernel}")
    
    # Long-run variance estimate
    long_run_var = gamma[0] + 2 * np.sum(weights[1:] * gamma[1:])
    
    # Standard error
    se = np.sqrt(max(0, long_run_var) / n)
    
    return se, long_run_var


def compute_hansen_hodrick_se(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizon: int
) -> Tuple[float, float]:
    """
    Compute Hansen-Hodrick standard error for overlapping observations.
    
    The HH estimator is specifically designed for overlapping forecast
    errors and uses a truncated kernel at exactly h-1 lags.
    
    Args:
        y_true: Realized values
        y_pred: Predicted values
        horizon: Forecast horizon
        
    Returns:
        Tuple of (standard_error, variance)
        
    References:
        Hansen, L.P. and Hodrick, R.J. (1980). "Forward Exchange Rates as
        Optimal Predictors of Future Spot Rates." Journal of Political
        Economy, 88(5), 829-853.
    """
    n = len(y_true)
    max_lag = horizon - 1
    
    # Check for constant input
    if np.all(y_true == y_true[0]) or np.all(y_pred == y_pred[0]):
        return np.nan, np.nan
        
    # Compute ranks for Spearman
    rank_true = stats.rankdata(y_true)
    rank_pred = stats.rankdata(y_pred)
    
    # Standardize
    rank_true = (rank_true - rank_true.mean()) / rank_true.std()
    rank_pred = (rank_pred - rank_pred.mean()) / rank_pred.std()
    
    # Cross-product
    z = rank_true * rank_pred
    z_centered = z - z.mean()
    
    # Hansen-Hodrick: equal weights up to h-1, then zero
    gamma_0 = np.mean(z_centered ** 2)
    gamma_sum = 0
    
    for j in range(1, min(max_lag + 1, n)):
        gamma_j = np.mean(z_centered[j:] * z_centered[:-j])
        gamma_sum += gamma_j
    
    # HH variance: gamma_0 + 2 * gamma_sum
    hh_var = gamma_0 + 2 * gamma_sum
    se = np.sqrt(max(0, hh_var) / n)
    
    return se, hh_var


def block_bootstrap_ic(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizon: int,
    n_bootstrap: int = 1000,
    block_size: Optional[int] = None,
    seed: int = 42
) -> Tuple[float, np.ndarray]:
    """
    Block bootstrap for IC standard error and confidence intervals.
    
    Uses non-overlapping blocks of size equal to the forecast horizon
    to preserve the dependence structure.
    
    Args:
        y_true: Realized values
        y_pred: Predicted values
        horizon: Forecast horizon (used as default block size)
        n_bootstrap: Number of bootstrap replications
        block_size: Block size (defaults to horizon)
        seed: Random seed
        
    Returns:
        Tuple of (bootstrap_se, bootstrap_distribution)
    """
    np.random.seed(seed)
    n = len(y_true)
    
    if block_size is None:
        block_size = horizon
    
    n_blocks = n // block_size
    
    if n_blocks < 5:
        logger.warning(f"Only {n_blocks} blocks available. Bootstrap may be unreliable.")
    
    bootstrap_ics = []
    
    for _ in range(n_bootstrap):
        # Sample blocks with replacement
        block_indices = np.random.randint(0, n_blocks, size=n_blocks)
        
        # Construct bootstrap sample
        indices = []
        for b in block_indices:
            start = b * block_size
            end = start + block_size
            indices.extend(range(start, end))
        
        indices = np.array(indices[:n])  # Trim to original size
        
        # Compute IC on bootstrap sample
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        
        ic_boot, _ = spearmanr(y_true_boot, y_pred_boot)
        bootstrap_ics.append(ic_boot)
    
    bootstrap_ics = np.array(bootstrap_ics)
    bootstrap_se = np.std(bootstrap_ics)
    
    return bootstrap_se, bootstrap_ics


def compute_effective_sample_size(n_obs: int, horizon: int) -> int:
    """
    Estimate effective sample size for overlapping observations.
    
    For h-period overlapping returns with n observations,
    the effective sample size is approximately n/h.
    
    Args:
        n_obs: Number of observations
        horizon: Forecast horizon
        
    Returns:
        Effective sample size
    """
    # Conservative estimate: n / h
    # More precise would require estimating autocorrelation structure
    return max(1, n_obs // horizon)


def compute_ic_with_inference(
    y_true: pd.Series,
    y_pred: pd.Series,
    horizon: int = 24,
    method: str = 'newey_west',
    alpha: float = 0.05,
    bootstrap_reps: int = 1000
) -> InferenceResult:
    """
    Compute Information Coefficient with proper statistical inference.
    
    This is the main entry point for IC computation with overlapping
    observations. It provides standard errors, t-statistics, p-values,
    and confidence intervals that account for serial correlation.
    
    Args:
        y_true: Realized values
        y_pred: Predicted values
        horizon: Forecast horizon in months
        method: 'newey_west', 'hansen_hodrick', or 'bootstrap'
        alpha: Significance level for confidence intervals
        bootstrap_reps: Number of bootstrap replications (if method='bootstrap')
        
    Returns:
        InferenceResult with all inference statistics
    """
    # Clean data
    mask = ~(y_true.isna() | y_pred.isna())
    y_true_clean = y_true[mask].values
    y_pred_clean = y_pred[mask].values
    
    n = len(y_true_clean)
    
    if n < horizon + 10:
        logger.warning(f"Sample size ({n}) barely exceeds horizon ({horizon})")
    
    # Check for constant input
    if np.all(y_true_clean == y_true_clean[0]) or np.all(y_pred_clean == y_pred_clean[0]):
        msg = "Constant input detected in inference (likely Lasso collapse). Returning NaNs."
        # Use a module-level attribute to track logged warnings if not already present
        if not hasattr(compute_ic_with_inference, '_logged_warnings'):
            compute_ic_with_inference._logged_warnings = set()
        
        if msg not in compute_ic_with_inference._logged_warnings:
            logger.warning(msg)
            compute_ic_with_inference._logged_warnings.add(msg)
            
        return InferenceResult(
            estimate=0.0, # IC is effectively 0 for a constant prediction
            se_raw=np.nan, se_nw=np.nan, se_hh=np.nan,
            t_stat_nw=np.nan, p_value_nw=np.nan,
            ci_lower=np.nan, ci_upper=np.nan,
            effective_n=compute_effective_sample_size(n, horizon),
            n_obs=n, horizon=horizon, method=method
        )

    # Point estimate
    ic, _ = spearmanr(y_true_clean, y_pred_clean)
    
    # Naive SE (assuming independence - WRONG but included for comparison)
    se_raw = np.sqrt((1 - ic**2) / (n - 2)) if n > 2 else np.nan
    
    # If sample size is too small for robust inference, skip complex SEs
    if n <= horizon:
        logger.debug(f"Sample size ({n}) <= horizon ({horizon}). Skipping robust SE calculation.")
        se_nw = np.nan
        se_hh = np.nan
        se_primary = np.nan
    else:
        # Newey-West SE
        se_nw, _ = compute_newey_west_se(y_true_clean, y_pred_clean, horizon)
        
        # Hansen-Hodrick SE
        se_hh, _ = compute_hansen_hodrick_se(y_true_clean, y_pred_clean, horizon)
        
        # Select primary SE based on method
        if method == 'newey_west':
            se_primary = se_nw
        elif method == 'hansen_hodrick':
            se_primary = se_hh
        elif method == 'bootstrap':
            se_boot, boot_dist = block_bootstrap_ic(
                y_true_clean, y_pred_clean, horizon, bootstrap_reps
            )
            se_primary = se_boot
            # Override CI with bootstrap percentiles
            ci_lower = np.percentile(boot_dist, 100 * alpha / 2)
            ci_upper = np.percentile(boot_dist, 100 * (1 - alpha / 2))
            return InferenceResult(
                estimate=ic, se_raw=se_raw, se_nw=np.nan, se_hh=np.nan,
                t_stat_nw=np.nan, p_value_nw=np.nan,
                ci_lower=ci_lower, ci_upper=ci_upper,
                effective_n=compute_effective_sample_size(n, horizon),
                n_obs=n, horizon=horizon, method=method
            )
        else:
            raise ValueError(f"Unknown method: {method}")
    
    # Effective sample size
    effective_n = compute_effective_sample_size(n, horizon)
    df = max(1, effective_n - 2)
    
    # T-statistic and p-value (using Newey-West as default)
    if se_nw > 0:
        t_stat = ic / se_nw
        p_value = 2 * (1 - t_dist.cdf(abs(t_stat), df))
    else:
        t_stat = np.nan
        p_value = np.nan
    
    # Confidence interval (if not bootstrap)
    if method != 'bootstrap':
        if se_primary > 0:
            t_crit = t_dist.ppf(1 - alpha / 2, df)
            ci_lower = ic - t_crit * se_primary
            ci_upper = ic + t_crit * se_primary
        else:
            ci_lower = np.nan
            ci_upper = np.nan
    
    return InferenceResult(
        estimate=ic,
        se_raw=se_raw,
        se_nw=se_nw,
        se_hh=se_hh,
        t_stat_nw=t_stat,
        p_value_nw=p_value,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        effective_n=effective_n,
        n_obs=n,
        horizon=horizon,
        method=method
    )
