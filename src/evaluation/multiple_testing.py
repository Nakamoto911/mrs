"""
Multiple Testing Correction
===========================
Adjusts for multiple comparisons to control false discovery.
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging

logger = logging.getLogger(__name__)


class MTCMethod(Enum):
    """Multiple testing correction methods."""
    BONFERRONI = "bonferroni"
    HOLM = "holm"
    BENJAMINI_HOCHBERG = "benjamini_hochberg"
    ROMANO_WOLF = "romano_wolf"
    NONE = "none"


@dataclass
class HypothesisTest:
    """Individual hypothesis test result."""
    test_id: str
    asset: str
    model: str
    target: str
    p_value: float
    ic_estimate: float
    t_statistic: float


@dataclass
class MTCResult:
    """Result after multiple testing correction."""
    method: str
    n_tests: int
    alpha_original: float
    alpha_adjusted: Optional[float]  # For FWER methods
    fdr_level: Optional[float]       # For FDR methods
    n_significant_original: int
    n_significant_adjusted: int
    significant_tests: List[str]
    test_details: pd.DataFrame


class MultipleTestingCorrector:
    """
    Applies multiple testing corrections to a family of hypothesis tests.
    """
    
    def __init__(
        self,
        fwer_alpha: float = 0.05,
        fdr_alpha: float = 0.10,
        n_bootstrap: int = 1000,
        seed: int = 42
    ):
        self.fwer_alpha = fwer_alpha
        self.fdr_alpha = fdr_alpha
        self.n_bootstrap = n_bootstrap
        self.seed = seed
    
    def bonferroni(
        self,
        tests: List[HypothesisTest]
    ) -> MTCResult:
        """
        Bonferroni correction: α_adj = α / m
        """
        m = len(tests)
        alpha_adj = self.fwer_alpha / m if m > 0 else self.fwer_alpha
        
        significant = [t for t in tests if t.p_value < alpha_adj]
        
        return MTCResult(
            method="bonferroni",
            n_tests=m,
            alpha_original=self.fwer_alpha,
            alpha_adjusted=alpha_adj,
            fdr_level=None,
            n_significant_original=sum(1 for t in tests if t.p_value < self.fwer_alpha),
            n_significant_adjusted=len(significant),
            significant_tests=[t.test_id for t in significant],
            test_details=self._create_details_df(tests, threshold=alpha_adj)
        )
    
    def holm(
        self,
        tests: List[HypothesisTest]
    ) -> MTCResult:
        """
        Holm-Bonferroni stepdown procedure.
        """
        m = len(tests)
        if m == 0:
            return MTCResult("holm", 0, self.fwer_alpha, self.fwer_alpha, None, 0, 0, [], pd.DataFrame())
            
        # Sort by p-value
        sorted_tests = sorted(tests, key=lambda t: t.p_value)
        
        significant = []
        holm_thresholds = {}
        for i, test in enumerate(sorted_tests):
            threshold = self.fwer_alpha / (m - i)
            holm_thresholds[test.test_id] = threshold
            
            if test.p_value < threshold:
                significant.append(test)
            else:
                # Once we fail to reject, all subsequent tests are not significant
                for t in sorted_tests[i+1:]:
                    holm_thresholds[t.test_id] = self.fwer_alpha / (m - (sorted_tests.index(t)))
                break
        
        return MTCResult(
            method="holm",
            n_tests=m,
            alpha_original=self.fwer_alpha,
            alpha_adjusted=self.fwer_alpha / m,
            fdr_level=None,
            n_significant_original=sum(1 for t in tests if t.p_value < self.fwer_alpha),
            n_significant_adjusted=len(significant),
            significant_tests=[t.test_id for t in significant],
            test_details=self._create_details_df(
                tests, 
                holm_thresholds=holm_thresholds
            )
        )
    
    def benjamini_hochberg(
        self,
        tests: List[HypothesisTest]
    ) -> MTCResult:
        """
        Benjamini-Hochberg procedure for FDR control.
        """
        m = len(tests)
        if m == 0:
            return MTCResult("benjamini_hochberg", 0, self.fwer_alpha, None, self.fdr_alpha, 0, 0, [], pd.DataFrame())

        # Sort by p-value
        sorted_tests = sorted(tests, key=lambda t: t.p_value)
        
        # Find largest k such that P_(k) <= (k/m) * α
        max_k = 0
        for k in range(1, m + 1):
            threshold = (k / m) * self.fdr_alpha
            if sorted_tests[k - 1].p_value <= threshold:
                max_k = k
        
        significant = sorted_tests[:max_k]
        
        # Compute q-values (adjusted p-values)
        q_values = {}
        if m > 0:
            sorted_p = [t.p_value for t in sorted_tests]
            current_q = 1.0
            for i in range(m - 1, -1, -1):
                raw_q = sorted_p[i] * m / (i + 1)
                current_q = min(current_q, raw_q)
                q_values[sorted_tests[i].test_id] = min(current_q, 1.0)
        
        return MTCResult(
            method="benjamini_hochberg",
            n_tests=m,
            alpha_original=self.fwer_alpha,
            alpha_adjusted=None,
            fdr_level=self.fdr_alpha,
            n_significant_original=sum(1 for t in tests if t.p_value < self.fwer_alpha),
            n_significant_adjusted=len(significant),
            significant_tests=[t.test_id for t in significant],
            test_details=self._create_details_df(tests, q_values=q_values)
        )
    
    def romano_wolf(
        self,
        tests: List[HypothesisTest],
        predictions_matrix: Optional[np.ndarray] = None,
        actuals: Optional[np.ndarray] = None
    ) -> MTCResult:
        """
        Romano-Wolf stepdown procedure with bootstrap.
        """
        if predictions_matrix is None or actuals is None:
            logger.warning("Romano-Wolf requires predictions_matrix and actuals. Falling back to Holm.")
            return self.holm(tests)
        
        m = len(tests)
        n = len(actuals)
        if m == 0 or n == 0:
            return self.holm(tests)
            
        np.random.seed(self.seed)
        
        # Original test statistics
        t_stats = np.array([t.t_statistic for t in tests])
        
        # This is a computationally expensive method. Simplified version for now.
        # True Romano-Wolf requires multiple bootstrap rounds in a stepdown fashion.
        # We'll use a single-step bootstrap for critical value to bound the max statistic.
        
        boot_max_stats = []
        for _ in range(self.n_bootstrap):
            idx = np.random.choice(n, n, replace=True)
            boot_t = []
            for j in range(m):
                y_true_boot = actuals[idx]
                y_pred_boot = predictions_matrix[idx, j]
                
                # Fast IC and SE
                from scipy.stats import spearmanr
                ic_boot, _ = spearmanr(y_true_boot, y_pred_boot)
                se_boot = np.sqrt((1 - ic_boot**2) / (n - 2)) if n > 2 else 1.0
                boot_t_stat = ic_boot / se_boot if se_boot > 0 else 0
                
                # Maximize deviation from original (null distribution of max statistic)
                boot_t.append(abs(boot_t_stat - (t_stats[j] if not np.isnan(t_stats[j]) else 0)))
            
            boot_max_stats.append(max(boot_t) if boot_t else 0)
            
        critical_value = np.percentile(boot_max_stats, 100 * (1 - self.fwer_alpha))
        
        significant_mask = np.abs(t_stats) > critical_value
        significant = [tests[i] for i in range(m) if significant_mask[i]]
        
        return MTCResult(
            method="romano_wolf",
            n_tests=m,
            alpha_original=self.fwer_alpha,
            alpha_adjusted=None,
            fdr_level=None,
            n_significant_original=sum(1 for t in tests if t.p_value < self.fwer_alpha),
            n_significant_adjusted=len(significant),
            significant_tests=[t.test_id for t in significant],
            test_details=self._create_details_df(tests, significant_mask=significant_mask)
        )
    
    def _create_details_df(
        self,
        tests: List[HypothesisTest],
        threshold: float = None,
        holm_thresholds: Dict = None,
        q_values: Dict = None,
        significant_mask: np.ndarray = None
    ) -> pd.DataFrame:
        """Create detailed results DataFrame."""
        rows = []
        for i, t in enumerate(tests):
            row = {
                'test_id': t.test_id,
                'asset': t.asset,
                'model': t.model,
                'target': t.target,
                'ic': t.ic_estimate,
                't_stat': t.t_statistic,
                'p_value': t.p_value,
                'sig_original': t.p_value < self.fwer_alpha
            }
            
            if threshold is not None:
                row['threshold'] = threshold
                row['sig_adjusted'] = t.p_value < threshold
            elif holm_thresholds is not None:
                row['threshold'] = holm_thresholds.get(t.test_id, self.fwer_alpha)
                row['sig_adjusted'] = t.p_value < row['threshold']
            elif q_values is not None:
                row['q_value'] = q_values.get(t.test_id, 1.0)
                row['sig_adjusted'] = row['q_value'] < self.fdr_alpha
            elif significant_mask is not None:
                row['sig_adjusted'] = significant_mask[i]
            else:
                row['sig_adjusted'] = t.p_value < self.fwer_alpha
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def correct_all_methods(
        self,
        tests: List[HypothesisTest],
        predictions_matrix: Optional[np.ndarray] = None,
        actuals: Optional[np.ndarray] = None
    ) -> Dict[str, MTCResult]:
        """Apply all correction methods for comparison."""
        if not tests:
            return {}
            
        results = {
            'none': MTCResult(
                method="none",
                n_tests=len(tests),
                alpha_original=self.fwer_alpha,
                alpha_adjusted=self.fwer_alpha,
                fdr_level=None,
                n_significant_original=sum(1 for t in tests if t.p_value < self.fwer_alpha),
                n_significant_adjusted=sum(1 for t in tests if t.p_value < self.fwer_alpha),
                significant_tests=[t.test_id for t in tests if t.p_value < self.fwer_alpha],
                test_details=self._create_details_df(tests, threshold=self.fwer_alpha)
            ),
            'bonferroni': self.bonferroni(tests),
            'holm': self.holm(tests),
            'benjamini_hochberg': self.benjamini_hochberg(tests),
        }
        
        if predictions_matrix is not None and actuals is not None:
            results['romano_wolf'] = self.romano_wolf(tests, predictions_matrix, actuals)
        
        return results


def format_mtc_report(results: Dict[str, MTCResult]) -> str:
    """Format multiple testing correction comparison report."""
    if not results:
        return "No tests performed."
        
    lines = [
        "=" * 70,
        "MULTIPLE TESTING CORRECTION COMPARISON",
        "=" * 70,
        "",
        f"{'Method':<25} {'α/FDR':<10} {'Sig (orig)':<12} {'Sig (adj)':<12} {'Reduction':<10}",
        "-" * 70,
    ]
    
    for method, result in results.items():
        level = result.alpha_adjusted if result.alpha_adjusted else result.fdr_level
        level_str = f"{level:.4f}" if level and level < 0.01 else f"{level:.3f}" if level else "N/A"
        
        reduction = ""
        if result.n_significant_original > 0:
            reduction = f"{(1 - result.n_significant_adjusted/result.n_significant_original)*100:.0f}%"
        else:
            reduction = "0%"
        
        lines.append(
            f"{method:<25} {level_str:<10} "
            f"{result.n_significant_original:<12} {result.n_significant_adjusted:<12} "
            f"{reduction:<10}"
        )
    
    # Significant tests after B-H (recommended)
    bh_result = results.get('benjamini_hochberg')
    if bh_result and bh_result.significant_tests:
        lines.extend([
            "",
            "Significant after Benjamini-Hochberg (FDR control):",
            "-" * 40,
        ])
        for test_id in bh_result.significant_tests[:10]:
            lines.append(f"  ✓ {test_id}")
        if len(bh_result.significant_tests) > 10:
            lines.append(f"  ... and {len(bh_result.significant_tests) - 10} more")
    
    lines.extend([
        "",
        "RECOMMENDATION:",
        "  - Use Benjamini-Hochberg for exploratory analysis",
        "  - Use Bonferroni/Holm for confirmatory claims",
        "=" * 70,
    ])
    
    return "\n".join(lines)
