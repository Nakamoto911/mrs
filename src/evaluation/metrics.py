"""
Metrics Module
==============
Evaluation metrics for model performance assessment.

Part of the Asset-Specific Macro Regime Detection System
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple
from scipy.stats import spearmanr, pearsonr
import logging

logger = logging.getLogger(__name__)


def compute_ic(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Compute Information Coefficient (Spearman rank correlation).
    
    IC measures the ability to correctly rank forecasts.
    IC > 0.15 indicates economically significant predictive power.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        Information Coefficient
    """
    # Remove NaN
    mask = ~(y_true.isna() | y_pred.isna())
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) < 3:
        return np.nan
    
    ic, _ = spearmanr(y_true_clean, y_pred_clean)
    return ic


def compute_rmse(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Compute Root Mean Squared Error.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        RMSE
    """
    mask = ~(y_true.isna() | y_pred.isna())
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) < 1:
        return np.nan
    
    return np.sqrt(np.mean((y_true_clean - y_pred_clean) ** 2))


def compute_mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Compute Mean Absolute Error.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        MAE
    """
    mask = ~(y_true.isna() | y_pred.isna())
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) < 1:
        return np.nan
    
    return np.mean(np.abs(y_true_clean - y_pred_clean))


def compute_r2_oos(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Compute out-of-sample R-squared.
    
    R²_OOS = 1 - MSE(pred) / MSE(mean)
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        Out-of-sample R²
    """
    mask = ~(y_true.isna() | y_pred.isna())
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) < 2:
        return np.nan
    
    mse_pred = np.mean((y_true_clean - y_pred_clean) ** 2)
    mse_mean = np.mean((y_true_clean - y_true_clean.mean()) ** 2)
    
    if mse_mean == 0:
        return np.nan
    
    return 1 - mse_pred / mse_mean


def compute_hit_rate(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Compute hit rate (directional accuracy).
    
    Percentage of correct directional forecasts.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        Hit rate (0-1)
    """
    mask = ~(y_true.isna() | y_pred.isna())
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    if len(y_true_clean) < 1:
        return np.nan
    
    actual_sign = np.sign(y_true_clean)
    pred_sign = np.sign(y_pred_clean)
    
    return np.mean(actual_sign == pred_sign)


def compute_directional_mae(y_true: pd.Series, y_pred: pd.Series) -> float:
    """
    Compute MAE only when direction is correct.
    
    Measures forecast quality given correct direction.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        Directional MAE
    """
    mask = ~(y_true.isna() | y_pred.isna())
    y_true_clean = y_true[mask]
    y_pred_clean = y_pred[mask]
    
    # Filter to correct direction only
    correct_dir = np.sign(y_true_clean) == np.sign(y_pred_clean)
    
    if correct_dir.sum() < 1:
        return np.nan
    
    return np.mean(np.abs(y_true_clean[correct_dir] - y_pred_clean[correct_dir]))


def compute_revision_risk(ic_revised: float, ic_realtime: float) -> float:
    """
    Compute revision risk (IC degradation from revised to real-time data).
    
    Revision Risk = (IC_revised - IC_realtime) / IC_revised
    
    Args:
        ic_revised: IC on revised (final) data
        ic_realtime: IC on real-time (ALFRED) data
        
    Returns:
        Revision risk (0-1, higher = more degradation)
    """
    if ic_revised <= 0 or np.isnan(ic_revised):
        return np.nan
    
    return (ic_revised - ic_realtime) / ic_revised


def compute_feature_stability(shap_revised: pd.Series, shap_realtime: pd.Series) -> float:
    """
    Compute feature stability (correlation of SHAP values).
    
    Args:
        shap_revised: SHAP values from revised data
        shap_realtime: SHAP values from real-time data
        
    Returns:
        Feature stability score (0-1)
    """
    # Align features
    common_features = shap_revised.index.intersection(shap_realtime.index)
    
    if len(common_features) < 3:
        return np.nan
    
    revised_aligned = shap_revised.loc[common_features]
    realtime_aligned = shap_realtime.loc[common_features]
    
    corr, _ = spearmanr(revised_aligned, realtime_aligned)
    return corr


def compute_all_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    """
    Compute all evaluation metrics.
    
    Args:
        y_true: Actual values
        y_pred: Predicted values
        
    Returns:
        Dictionary of metrics
    """
    return {
        'IC': compute_ic(y_true, y_pred),
        'RMSE': compute_rmse(y_true, y_pred),
        'MAE': compute_mae(y_true, y_pred),
        'R2_OOS': compute_r2_oos(y_true, y_pred),
        'hit_rate': compute_hit_rate(y_true, y_pred),
        'directional_MAE': compute_directional_mae(y_true, y_pred)
    }


def check_deployment_criteria(ic_realtime: float, revision_risk: float,
                              feature_stability: float,
                              ic_threshold: float = 0.15,
                              risk_threshold: float = 0.30,
                              stability_threshold: float = 0.70) -> Tuple[bool, Dict[str, bool]]:
    """
    Check if model meets deployment criteria.
    
    Criteria:
    - IC (real-time) > 0.15
    - Revision Risk < 30%
    - Feature Stability > 0.70
    
    Args:
        ic_realtime: IC on real-time data
        revision_risk: Revision risk metric
        feature_stability: Feature stability score
        ic_threshold: Minimum IC threshold
        risk_threshold: Maximum revision risk threshold
        stability_threshold: Minimum feature stability threshold
        
    Returns:
        Tuple of (passes_all, individual_checks)
    """
    checks = {
        'ic_realtime': ic_realtime > ic_threshold,
        'revision_risk': revision_risk < risk_threshold,
        'feature_stability': feature_stability > stability_threshold
    }
    
    passes_all = all(checks.values())
    
    return passes_all, checks


if __name__ == "__main__":
    # Test metrics
    np.random.seed(42)
    n = 100
    
    y_true = pd.Series(np.random.randn(n))
    y_pred = y_true + np.random.randn(n) * 0.3  # Noisy predictions
    
    metrics = compute_all_metrics(y_true, y_pred)
    
    print("Metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value:.4f}")
    
    # Test deployment criteria
    passes, checks = check_deployment_criteria(
        ic_realtime=0.20,
        revision_risk=0.18,
        feature_stability=0.85
    )
    
    print(f"\nDeployment check: {'PASS' if passes else 'FAIL'}")
    for check, result in checks.items():
        print(f"  {check}: {'✓' if result else '✗'}")
