"""
Effective Sample Size Estimation
================================
Proper accounting for overlapping observations in statistical inference.
"""

import numpy as np
import pandas as pd
from typing import Optional, Dict, Tuple, List
from dataclasses import dataclass
from enum import Enum
import logging
from scipy import stats

logger = logging.getLogger(__name__)


class ESSMethod(Enum):
    """Methods for effective sample size estimation."""
    SIMPLE = "simple"
    NEWEY_WEST = "newey_west"
    HANSEN_HODRICK = "hansen_hodrick"
    AUTOCORRELATION = "autocorr"


@dataclass
class EffectiveSampleSizeResult:
    """Result of effective sample size estimation."""
    method: str
    n_observations: int
    n_effective: float
    adjustment_ratio: float  # n_eff / n_obs
    details: Dict
    
    def __str__(self):
        return (
            f"ESS ({self.method}): {self.n_effective:.1f} "
            f"(from {self.n_observations} obs, ratio={self.adjustment_ratio:.2%})"
        )


class EffectiveSampleSizeEstimator:
    """
    Estimates effective sample size accounting for autocorrelation.
    """
    
    def __init__(
        self,
        method: str = "newey_west",
        horizon: int = 24,
        bandwidth_method: str = "auto",
        fixed_bandwidth: Optional[int] = None,
        kernel: str = "bartlett",
        max_lags: int = 36,
        significance_threshold: float = 0.05
    ):
        self.method = ESSMethod(method)
        self.horizon = horizon
        self.bandwidth_method = bandwidth_method
        self.fixed_bandwidth = fixed_bandwidth
        self.kernel = kernel
        self.max_lags = max_lags
        self.significance_threshold = significance_threshold
    
    def estimate_simple(self, n_obs: int) -> EffectiveSampleSizeResult:
        """Simple horizon-based adjustment."""
        n_eff = max(1.0, n_obs / self.horizon)
        
        return EffectiveSampleSizeResult(
            method="simple",
            n_observations=n_obs,
            n_effective=n_eff,
            adjustment_ratio=n_eff / n_obs if n_obs > 0 else 0,
            details={"horizon": self.horizon}
        )
    
    def _compute_autocorrelations(
        self,
        residuals: np.ndarray,
        max_lag: int
    ) -> np.ndarray:
        """Compute autocorrelations up to max_lag."""
        n = len(residuals)
        if n == 0:
            return np.zeros(max_lag + 1)
            
        mean = np.mean(residuals)
        var = np.var(residuals)
        
        if var == 0:
            return np.zeros(max_lag + 1)
        
        autocorrs = np.zeros(max_lag + 1)
        autocorrs[0] = 1.0
        
        for lag in range(1, min(max_lag + 1, n)):
            cov = np.sum((residuals[lag:] - mean) * (residuals[:-lag] - mean)) / n
            autocorrs[lag] = cov / var
        
        return autocorrs
    
    def _andrews_bandwidth(self, residuals: np.ndarray) -> int:
        """
        Andrews (1991) automatic bandwidth selection.
        """
        n = len(residuals)
        
        # Estimate AR(1) coefficient
        if n < 3:
            return 1
        
        autocorrs = self._compute_autocorrelations(residuals, 1)
        rho_hat = autocorrs[1]
        
        # Bound rho away from 1
        rho_hat = np.clip(rho_hat, -0.99, 0.99)
        
        if abs(rho_hat) < 0.01:
            return 1
        
        # Andrews formula for Bartlett kernel
        alpha = (4 * rho_hat**2) / ((1 - rho_hat)**2 * (1 + rho_hat)**2)
        bandwidth = 1.1447 * (alpha * n)**(1/3)
        
        return max(1, int(np.ceil(bandwidth)))
    
    def _bartlett_kernel(self, x: float) -> float:
        """Bartlett (triangular) kernel."""
        if abs(x) <= 1:
            return 1 - abs(x)
        return 0
    
    def _parzen_kernel(self, x: float) -> float:
        """Parzen kernel."""
        ax = abs(x)
        if ax <= 0.5:
            return 1 - 6 * ax**2 + 6 * ax**3
        elif ax <= 1:
            return 2 * (1 - ax)**3
        return 0
    
    def estimate_newey_west(
        self,
        residuals: np.ndarray
    ) -> EffectiveSampleSizeResult:
        """
        Newey-West bandwidth-based effective sample size.
        """
        n = len(residuals)
        
        if n < 10:
            return self.estimate_simple(n)
        
        # Determine bandwidth
        if self.bandwidth_method == "auto":
            bandwidth = self._andrews_bandwidth(residuals)
        elif self.bandwidth_method == "fixed" and self.fixed_bandwidth:
            bandwidth = self.fixed_bandwidth
        else:
            bandwidth = max(1, int(n ** (1/3)))
        
        # Compute autocorrelations
        autocorrs = self._compute_autocorrelations(residuals, bandwidth)
        
        # Select kernel
        if self.kernel == "bartlett":
            kernel_fn = self._bartlett_kernel
        elif self.kernel == "parzen":
            kernel_fn = self._parzen_kernel
        else:
            kernel_fn = self._bartlett_kernel
        
        # Compute HAC adjustment factor
        rho_sum = 0
        for lag in range(1, bandwidth + 1):
            if lag < len(autocorrs):
                weight = kernel_fn(lag / (bandwidth + 1))
                rho_sum += 2 * weight * autocorrs[lag]
        
        hac_factor = 1 + rho_sum
        n_eff = n / max(hac_factor, 1)
        
        return EffectiveSampleSizeResult(
            method="newey_west",
            n_observations=n,
            n_effective=n_eff,
            adjustment_ratio=n_eff / n,
            details={
                "bandwidth": bandwidth,
                "hac_factor": hac_factor,
                "rho_1": autocorrs[1] if len(autocorrs) > 1 else None,
                "kernel": self.kernel
            }
        )
    
    def estimate_hansen_hodrick(
        self,
        residuals: np.ndarray
    ) -> EffectiveSampleSizeResult:
        """
        Hansen-Hodrick (1980) effective sample size.
        """
        n = len(residuals)
        h = self.horizon
        
        if n < h:
            return self.estimate_simple(n)
        
        # Compute autocorrelations up to h-1
        autocorrs = self._compute_autocorrelations(residuals, h)
        
        # Hansen-Hodrick variance multiplier
        multiplier = 1
        for k in range(1, min(h, len(autocorrs))):
            weight = 1 - k / h
            multiplier += 2 * weight * autocorrs[k]
        
        n_eff = n / max(multiplier, 1)
        
        return EffectiveSampleSizeResult(
            method="hansen_hodrick",
            n_observations=n,
            n_effective=n_eff,
            adjustment_ratio=n_eff / n,
            details={
                "horizon": h,
                "variance_multiplier": multiplier,
                "significant_autocorrs": [
                    (k, autocorrs[k]) for k in range(1, min(h, len(autocorrs)))
                    if abs(autocorrs[k]) > 2 / np.sqrt(n)
                ]
            }
        )
    
    def estimate_autocorrelation_based(
        self,
        residuals: np.ndarray
    ) -> EffectiveSampleSizeResult:
        """
        Direct autocorrelation-based effective sample size.
        """
        n = len(residuals)
        
        if n < 10:
            return self.estimate_simple(n)
        
        # Significance threshold
        sig_threshold = 2 / np.sqrt(n)
        
        # Compute autocorrelations
        autocorrs = self._compute_autocorrelations(residuals, self.max_lags)
        
        # Find first insignificant lag
        dependence_length = 1
        for lag in range(1, len(autocorrs)):
            if abs(autocorrs[lag]) > sig_threshold:
                dependence_length = lag + 1
            else:
                if lag + 1 < len(autocorrs) and abs(autocorrs[lag + 1]) <= sig_threshold:
                    break
        
        n_eff = n / dependence_length
        
        return EffectiveSampleSizeResult(
            method="autocorr",
            n_observations=n,
            n_effective=n_eff,
            adjustment_ratio=n_eff / n,
            details={
                "dependence_length": dependence_length,
                "significance_threshold": sig_threshold,
                "first_5_autocorrs": autocorrs[1:6].tolist() if len(autocorrs) > 1 else []
            }
        )
    
    def estimate(
        self,
        residuals: Optional[np.ndarray] = None,
        n_obs: Optional[int] = None
    ) -> EffectiveSampleSizeResult:
        """
        Estimate effective sample size using configured method.
        """
        if self.method == ESSMethod.SIMPLE:
            if n_obs is None and residuals is not None:
                n_obs = len(residuals)
            if n_obs is None:
                raise ValueError("n_obs required for simple method")
            return self.estimate_simple(n_obs)
        
        if residuals is None:
            raise ValueError(f"residuals required for {self.method.value} method")
        
        if self.method == ESSMethod.NEWEY_WEST:
            return self.estimate_newey_west(residuals)
        elif self.method == ESSMethod.HANSEN_HODRICK:
            return self.estimate_hansen_hodrick(residuals)
        elif self.method == ESSMethod.AUTOCORRELATION:
            return self.estimate_autocorrelation_based(residuals)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def estimate_all_methods(
        self,
        residuals: np.ndarray
    ) -> Dict[str, EffectiveSampleSizeResult]:
        """Estimate using all available methods for comparison."""
        n = len(residuals)
        
        results = {
            "simple": self.estimate_simple(n),
            "newey_west": self.estimate_newey_west(residuals),
            "hansen_hodrick": self.estimate_hansen_hodrick(residuals),
            "autocorr": self.estimate_autocorrelation_based(residuals)
        }
        
        return results


def format_ess_comparison(results: Dict[str, EffectiveSampleSizeResult]) -> str:
    """Format comparison of ESS methods."""
    lines = [
        "=" * 60,
        "EFFECTIVE SAMPLE SIZE COMPARISON",
        "=" * 60,
        "",
        f"{'Method':<20} {'N_eff':<10} {'Ratio':<10} {'Notes':<20}",
        "-" * 60,
    ]
    
    for method, result in sorted(results.items(), key=lambda x: x[1].n_effective):
        notes = ""
        if 'bandwidth' in result.details:
            notes = f"bw={result.details['bandwidth']}"
        elif 'dependence_length' in result.details:
            notes = f"dep={result.details['dependence_length']}"
        
        lines.append(
            f"{method:<20} {result.n_effective:<10.1f} "
            f"{result.adjustment_ratio:<10.1%} {notes:<20}"
        )
    
    lines.append("=" * 60)
    return "\n".join(lines)
