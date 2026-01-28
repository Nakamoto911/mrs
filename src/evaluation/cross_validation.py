"""
Cross-Validation Module
=======================
Time-series cross-validation with expanding windows.

Part of the Asset-Specific Macro Regime Detection System
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Generator, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class CVFold:
    """Represents a single CV fold."""
    fold_id: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp
    val_start: pd.Timestamp
    val_end: pd.Timestamp
    train_indices: np.ndarray
    val_indices: np.ndarray


class TimeSeriesCV:
    """
    Time-series cross-validation with expanding windows.
    
    Features:
    - Expanding training window (no data leakage)
    - Gap between train and validation (for forecast horizon)
    - Semi-annual steps (matching rebalancing frequency)
    """
    
    def __init__(self,
                 min_train_months: int = 120,
                 validation_months: int = 12,
                 step_months: int = 6,
                 gap_months: int = 0):
        """
        Initialize time-series CV.
        
        Args:
            min_train_months: Minimum training period (months)
            validation_months: Validation window size (months)
            step_months: Step size between folds (months)
            gap_months: Gap between train end and validation start
        """
        self.min_train_months = min_train_months
        self.validation_months = validation_months
        self.step_months = step_months
        self.gap_months = gap_months
    
    def split(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> Generator[CVFold, None, None]:
        """
        Generate CV folds.
        
        Args:
            X: Feature DataFrame with DatetimeIndex
            y: Optional target (not used, for sklearn compatibility)
            
        Yields:
            CVFold objects
        """
        if not isinstance(X.index, pd.DatetimeIndex):
            raise ValueError("X must have DatetimeIndex")
        
        dates = X.index
        min_date = dates.min()
        max_date = dates.max()
        
        # First validation start date
        first_val_start = min_date + pd.DateOffset(months=self.min_train_months + self.gap_months)
        
        fold_id = 0
        val_start = first_val_start
        
        while val_start + pd.DateOffset(months=self.validation_months) <= max_date:
            # Define fold boundaries
            train_end = val_start - pd.DateOffset(months=self.gap_months) - pd.DateOffset(days=1)
            val_end = val_start + pd.DateOffset(months=self.validation_months) - pd.DateOffset(days=1)
            
            # Get indices
            train_mask = (dates >= min_date) & (dates <= train_end)
            val_mask = (dates >= val_start) & (dates <= val_end)
            
            train_indices = np.where(train_mask)[0]
            val_indices = np.where(val_mask)[0]
            
            if len(train_indices) > 0 and len(val_indices) > 0:
                yield CVFold(
                    fold_id=fold_id,
                    train_start=min_date,
                    train_end=train_end,
                    val_start=val_start,
                    val_end=val_end,
                    train_indices=train_indices,
                    val_indices=val_indices
                )
                fold_id += 1
            
            # Move to next fold
            val_start = val_start + pd.DateOffset(months=self.step_months)
    
    def get_n_splits(self, X: pd.DataFrame) -> int:
        """Get number of folds."""
        return sum(1 for _ in self.split(X))


@dataclass
class CVResult:
    """Results from cross-validation."""
    model_name: str
    asset: str
    target: str
    n_folds: int
    metrics: Dict[str, float]
    fold_metrics: List[Dict[str, float]]
    feature_importance: Optional[pd.Series]
    predictions: Optional[pd.DataFrame]


class CrossValidator:
    """Performs cross-validation for model evaluation."""
    
    def __init__(self, cv: Optional[TimeSeriesCV] = None, n_jobs: int = 1):
        """
        Initialize cross-validator.
        
        Args:
            cv: TimeSeriesCV instance
            n_jobs: Number of parallel jobs
        """
        self.cv = cv or TimeSeriesCV()
        self.n_jobs = n_jobs
    
    def evaluate(self, model: Any, X: pd.DataFrame, y: pd.Series,
                 model_name: str = 'model', asset: str = 'unknown',
                 target: str = 'return') -> CVResult:
        """
        Evaluate model using cross-validation.
        
        Args:
            model: Model with fit/predict interface
            X: Feature DataFrame
            y: Target Series
            model_name: Model name for reporting
            asset: Asset name
            target: Target type ('return' or 'volatility')
            
        Returns:
            CVResult with aggregated metrics
        """
        fold_metrics = []
        all_predictions = []
        all_importances = []
        
        for fold in self.cv.split(X):
            # Get data for fold
            X_train = X.iloc[fold.train_indices]
            y_train = y.iloc[fold.train_indices]
            X_val = X.iloc[fold.val_indices]
            y_val = y.iloc[fold.val_indices]
            
            try:
                # Clone model (if possible) or use as-is
                fold_model = self._clone_model(model)
                
                # Fit
                fold_model.fit(X_train, y_train)
                
                # Predict
                predictions = fold_model.predict(X_val)
                
                # Compute metrics
                metrics = self._compute_metrics(y_val, predictions, target)
                metrics['fold_id'] = fold.fold_id
                fold_metrics.append(metrics)
                
                # Store predictions
                pred_df = pd.DataFrame({
                    'actual': y_val.values,
                    'predicted': predictions.values,
                    'fold': fold.fold_id
                }, index=y_val.index)
                all_predictions.append(pred_df)
                
                # Feature importance
                if hasattr(fold_model, 'get_feature_importance'):
                    importance = fold_model.get_feature_importance()
                    all_importances.append(importance)
                
            except Exception as e:
                logger.warning(f"Fold {fold.fold_id} failed: {e}")
                continue
        
        if not fold_metrics:
            raise ValueError("All CV folds failed")
        
        # Aggregate metrics
        agg_metrics = self._aggregate_metrics(fold_metrics)
        
        # Aggregate feature importance
        agg_importance = None
        if all_importances:
            importance_df = pd.DataFrame(all_importances)
            agg_importance = importance_df.mean().sort_values(ascending=False)
        
        # Combine predictions
        predictions_df = pd.concat(all_predictions) if all_predictions else None
        
        return CVResult(
            model_name=model_name,
            asset=asset,
            target=target,
            n_folds=len(fold_metrics),
            metrics=agg_metrics,
            fold_metrics=fold_metrics,
            feature_importance=agg_importance,
            predictions=predictions_df
        )
    
    def _clone_model(self, model: Any) -> Any:
        """Clone model if possible."""
        try:
            from sklearn.base import clone
            return clone(model)
        except:
            # Return the same model instance
            return model
    
    def _compute_metrics(self, y_true: pd.Series, y_pred: pd.Series,
                        target: str) -> Dict[str, float]:
        """Compute evaluation metrics."""
        from scipy.stats import spearmanr
        
        # Remove NaN
        mask = ~(y_true.isna() | y_pred.isna())
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        if len(y_true_clean) < 2:
            return {'IC': np.nan, 'RMSE': np.nan, 'MAE': np.nan, 'hit_rate': np.nan}
        
        metrics = {}
        
        if len(np.unique(y_true_clean)) > 1 and len(np.unique(y_pred_clean)) > 1:
            ic, _ = spearmanr(y_true_clean, y_pred_clean)
        else:
            ic = 0.0  # Constant input results in zero correlation
        metrics['IC'] = ic
        
        # RMSE
        rmse = np.sqrt(np.mean((y_true_clean - y_pred_clean) ** 2))
        metrics['RMSE'] = rmse
        
        # MAE
        mae = np.mean(np.abs(y_true_clean - y_pred_clean))
        metrics['MAE'] = mae
        
        # Hit rate (directional accuracy)
        actual_sign = np.sign(y_true_clean)
        pred_sign = np.sign(y_pred_clean)
        hit_rate = np.mean(actual_sign == pred_sign)
        metrics['hit_rate'] = hit_rate
        
        return metrics
    
    def _aggregate_metrics(self, fold_metrics: List[Dict]) -> Dict[str, float]:
        """Aggregate metrics across folds."""
        agg = {}
        
        # Get metric names (excluding fold_id)
        metric_names = [k for k in fold_metrics[0].keys() if k != 'fold_id']
        
        for metric in metric_names:
            values = [f[metric] for f in fold_metrics if not np.isnan(f.get(metric, np.nan))]
            if values:
                agg[f"{metric}_mean"] = np.mean(values)
                agg[f"{metric}_std"] = np.std(values)
        
        return agg


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test
    np.random.seed(42)
    n = 200
    dates = pd.date_range('2010-01-01', periods=n, freq='ME')
    
    X = pd.DataFrame({
        'feature1': np.random.randn(n),
        'feature2': np.random.randn(n),
    }, index=dates)
    
    y = pd.Series(0.5 * X['feature1'] + np.random.randn(n) * 0.1, index=dates)
    
    # Test CV
    cv = TimeSeriesCV(min_train_months=60, validation_months=12, step_months=6)
    
    print(f"Number of folds: {cv.get_n_splits(X)}")
    
    for fold in cv.split(X):
        print(f"Fold {fold.fold_id}: Train {fold.train_start.date()} to {fold.train_end.date()}, "
              f"Val {fold.val_start.date()} to {fold.val_end.date()}")
