"""
Evaluation Module
=================
Cross-validation, metrics, and SHAP analysis.
"""

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
    'compute_shap_for_model'
]
