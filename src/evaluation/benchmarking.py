"""
Benchmarking Module
===================
Validates model performance against academic benchmarks and naive strategies.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, List
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class AssetClass(Enum):
    EQUITIES = "equities"
    BONDS = "bonds"
    COMMODITIES = "commodities"


@dataclass
class ThresholdSet:
    """IC thresholds for an asset class."""
    ic_excellent: float
    ic_good: float
    ic_acceptable: float
    ic_minimum: float
    ic_suspicious: float


# Literature-calibrated thresholds
ASSET_THRESHOLDS = {
    AssetClass.EQUITIES: ThresholdSet(
        ic_excellent=0.12,
        ic_good=0.08,
        ic_acceptable=0.05,
        ic_minimum=0.03,
        ic_suspicious=0.20
    ),
    AssetClass.BONDS: ThresholdSet(
        ic_excellent=0.18,
        ic_good=0.12,
        ic_acceptable=0.08,
        ic_minimum=0.05,
        ic_suspicious=0.30
    ),
    AssetClass.COMMODITIES: ThresholdSet(
        ic_excellent=0.10,
        ic_good=0.06,
        ic_acceptable=0.04,
        ic_minimum=0.02,
        ic_suspicious=0.18
    )
}

# Mapping from asset codes to classes
ASSET_CLASS_MAP = {
    'SPX': AssetClass.EQUITIES,
    'BOND': AssetClass.BONDS,
    'GOLD': AssetClass.COMMODITIES
}


@dataclass
class BenchmarkResult:
    """Result of benchmarking a model's IC."""
    asset: str
    asset_class: AssetClass
    ic: float
    ic_t_stat: float
    ic_p_value: float
    rating: str                      # 'excellent', 'good', 'acceptable', 'minimum', 'poor'
    is_suspicious: bool              # IC too good to be true
    is_significant: bool             # p < 0.05
    implied_sharpe: float            # Economic significance
    vs_naive_sharpe: float           # Sharpe improvement over buy-and-hold
    warnings: List[str]              # Any red flags


def get_thresholds_for_asset(asset: str) -> ThresholdSet:
    """Get appropriate thresholds for an asset."""
    asset_class = ASSET_CLASS_MAP.get(asset.upper(), AssetClass.EQUITIES)
    return ASSET_THRESHOLDS[asset_class]


def rate_ic(ic: float, thresholds: ThresholdSet) -> str:
    """Rate an IC value against thresholds."""
    if ic >= thresholds.ic_excellent:
        return 'excellent'
    elif ic >= thresholds.ic_good:
        return 'good'
    elif ic >= thresholds.ic_acceptable:
        return 'acceptable'
    elif ic >= thresholds.ic_minimum:
        return 'minimum'
    else:
        return 'poor'


def compute_implied_sharpe(
    ic: float,
    turnover_per_year: float = 2.0,
    transaction_cost_bps: float = 10.0
) -> float:
    """
    Compute implied Sharpe ratio from IC using Grinold's Fundamental Law.
    
    IR ≈ IC × √(Breadth)
    
    For a monthly signal with semi-annual rebalancing:
    - Breadth ≈ 2 trades per year per asset
    - Adjusted for transaction costs
    
    Args:
        ic: Information Coefficient
        turnover_per_year: Number of round-trip trades per year
        transaction_cost_bps: One-way transaction cost in basis points
        
    Returns:
        Implied annualized Sharpe ratio (after costs)
    """
    # Gross IR
    breadth = turnover_per_year
    gross_ir = ic * np.sqrt(breadth * 12)  # Monthly signal
    
    # Cost drag (rough approximation)
    # If IC is on monthly returns, transaction costs reduce Sharpe
    cost_drag = 2 * turnover_per_year * transaction_cost_bps / 10000 * np.sqrt(12)
    
    net_ir = max(0, gross_ir - cost_drag)
    
    return float(net_ir)


def compute_naive_benchmark_sharpe(returns: pd.Series) -> float:
    """
    Compute Sharpe ratio of buy-and-hold (naive benchmark).
    
    Args:
        returns: Series of returns
        
    Returns:
        Annualized Sharpe ratio
    """
    if returns.empty:
        return 0.0
        
    mean_ret = returns.mean()
    std_ret = returns.std()
    
    if std_ret == 0 or np.isnan(std_ret):
        return 0.0
    
    # Assume monthly returns, annualize
    # NOTE: If the input series is already 24M returns, np.sqrt(12) might be wrong,
    # but the specification assumes monthly returns here.
    sharpe = (mean_ret / std_ret) * np.sqrt(12)
    
    return float(sharpe)


def benchmark_model(
    ic: float,
    ic_t_stat: float,
    ic_p_value: float,
    asset: str,
    returns: Optional[pd.Series] = None,
    ic_in_sample: Optional[float] = None,
    ic_std: Optional[float] = None
) -> BenchmarkResult:
    """
    Benchmark a model's performance against literature standards.
    
    Args:
        ic: Out-of-sample IC
        ic_t_stat: t-statistic for IC
        ic_p_value: p-value for IC
        asset: Asset code (SPX, BOND, GOLD)
        returns: Historical returns for naive benchmark
        ic_in_sample: In-sample IC (for overfitting check)
        ic_std: Standard deviation of IC across folds
        
    Returns:
        BenchmarkResult with ratings and warnings
    """
    asset_class = ASSET_CLASS_MAP.get(asset.upper(), AssetClass.EQUITIES)
    thresholds = ASSET_THRESHOLDS[asset_class]
    
    warnings = []
    
    # Rating
    rating = rate_ic(ic, thresholds)
    
    # Suspicious check
    is_suspicious = ic > thresholds.ic_suspicious
    if is_suspicious:
        warnings.append(
            f"IC of {ic:.3f} exceeds suspicious threshold of {thresholds.ic_suspicious:.2f}. "
            "Possible data leakage, look-ahead bias, or overfitting."
        )
    
    # Statistical significance
    is_significant = ic_p_value < 0.05
    if not is_significant and ic >= thresholds.ic_acceptable:
        warnings.append(
            f"IC of {ic:.3f} rated '{rating}' but not statistically significant (p={ic_p_value:.3f}). "
            "May be spurious."
        )
    
    # In-sample vs out-of-sample check
    if ic_in_sample is not None and ic > 0:
        ratio = ic_in_sample / ic
        if ratio > 2.0:
            warnings.append(
                f"In-sample IC ({ic_in_sample:.3f}) is {ratio:.1f}x higher than OOS IC ({ic:.3f}). "
                "Likely overfitting."
            )
    
    # IC stability check
    if ic_std is not None and ic > 0:
        cv = ic_std / ic
        if cv > 1.0:
            warnings.append(
                f"IC coefficient of variation is {cv:.1f} (std={ic_std:.3f}, mean={ic:.3f}). "
                "Signal is unstable across time periods."
            )
    
    # Economic significance
    implied_sharpe = compute_implied_sharpe(ic)
    
    # Naive benchmark
    naive_sharpe = 0.0
    if returns is not None:
        naive_sharpe = compute_naive_benchmark_sharpe(returns)
    
    vs_naive = implied_sharpe - naive_sharpe
    
    if implied_sharpe < 0.20 and rating in ['excellent', 'good']:
        warnings.append(
            f"IC rated '{rating}' but implied Sharpe ({implied_sharpe:.2f}) is low. "
            "May not be economically meaningful after costs."
        )
    
    return BenchmarkResult(
        asset=asset,
        asset_class=asset_class,
        ic=ic,
        ic_t_stat=ic_t_stat,
        ic_p_value=ic_p_value,
        rating=rating,
        is_suspicious=is_suspicious,
        is_significant=is_significant,
        implied_sharpe=implied_sharpe,
        vs_naive_sharpe=vs_naive,
        warnings=warnings
    )


def format_benchmark_report(result: BenchmarkResult) -> str:
    """Format benchmark result as human-readable report."""
    lines = [
        f"{'='*60}",
        f"BENCHMARK REPORT: {result.asset}",
        f"{'='*60}",
        f"",
        f"IC Performance:",
        f"  IC:           {result.ic:.4f}",
        f"  t-statistic:  {result.ic_t_stat:.2f}",
        f"  p-value:      {result.ic_p_value:.4f}",
        f"  Rating:       {result.rating.upper()}",
        f"  Significant:  {'Yes' if result.is_significant else 'No'}",
        f"",
        f"Economic Significance:",
        f"  Implied Sharpe: {result.implied_sharpe:.2f}",
        f"  vs Naive:       {'+' if result.vs_naive_sharpe > 0 else ''}{result.vs_naive_sharpe:.2f}",
        f"",
    ]
    
    if result.is_suspicious:
        lines.append("⚠️  WARNING: IC SUSPICIOUSLY HIGH")
        lines.append("")
    
    if result.warnings:
        lines.append("Warnings:")
        for w in result.warnings:
            lines.append(f"  • {w}")
        lines.append("")
    
    lines.append(f"{'='*60}")
    
    return "\n".join(lines)


def create_benchmark_table(results: List[BenchmarkResult]) -> pd.DataFrame:
    """Create summary table from multiple benchmark results."""
    rows = []
    for r in results:
        rows.append({
            'Asset': r.asset,
            'IC': r.ic,
            't-stat': r.ic_t_stat,
            'p-value': r.ic_p_value,
            'Rating': r.rating,
            'Significant': '✓' if r.is_significant else '',
            'Implied SR': r.implied_sharpe,
            'Warnings': len(r.warnings)
        })
    
    return pd.DataFrame(rows)
