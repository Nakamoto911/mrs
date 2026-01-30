"""
Evaluation Module
=================
Cross-validation, metrics, and SHAP analysis.
"""

from .inference import (
    InferenceResult,
    compute_ic_with_inference,
    compute_newey_west_se,
    compute_hansen_hodrick_se,
    block_bootstrap_ic,
    compute_effective_sample_size
)

from .cross_validation import (
    TimeSeriesCV,
    CVFold,
    CVResult,
    CrossValidator
)

from .metrics import (
    compute_ic,
    compute_rmse,
    compute_mae,
    compute_r2_oos,
    compute_hit_rate,
    compute_directional_mae,
    compute_revision_risk,
    compute_feature_stability,
    compute_all_metrics,
    check_deployment_criteria
)

from .shap_analysis import (
    SHAPAnalyzer,
    compute_shap_for_model
)

from .benchmarking import (
    benchmark_model,
    BenchmarkResult,
    get_thresholds_for_asset,
    compute_implied_sharpe,
    format_benchmark_report,
    create_benchmark_table,
    AssetClass,
    ASSET_THRESHOLDS
)

__all__ = [
    'TimeSeriesCV',
    'CVFold',
    'CVResult',
    'CrossValidator',
    'compute_ic',
    'compute_rmse',
    'compute_mae',
    'compute_r2_oos',
    'compute_hit_rate',
    'compute_directional_mae',
    'compute_revision_risk',
    'compute_feature_stability',
    'compute_all_metrics',
    'check_deployment_criteria',
    'SHAPAnalyzer',
    'compute_shap_for_model',
    'InferenceResult',
    'compute_ic_with_inference',
    'compute_newey_west_se',
    'compute_hansen_hodrick_se',
    'block_bootstrap_ic',
    'compute_effective_sample_size',
    'benchmark_model',
    'BenchmarkResult',
    'get_thresholds_for_asset',
    'compute_implied_sharpe',
    'format_benchmark_report',
    'create_benchmark_table',
    'AssetClass',
    'ASSET_THRESHOLDS'
]
