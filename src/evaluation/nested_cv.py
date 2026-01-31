"""
Nested Cross-Validation for Ensemble Selection
===============================================
Implements unbiased ensemble evaluation with proper train/test separation
for model selection.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass
import logging
from pathlib import Path
import joblib

logger = logging.getLogger(__name__)


@dataclass
class OuterFold:
    """Represents an outer CV fold."""
    fold_id: int
    outer_train_start: pd.Timestamp
    outer_train_end: pd.Timestamp
    outer_test_start: pd.Timestamp
    outer_test_end: pd.Timestamp
    outer_train_indices: np.ndarray
    outer_test_indices: np.ndarray


@dataclass
class InnerFoldResult:
    """Result from inner CV for a single model."""
    model_name: str
    inner_ic_mean: float
    inner_ic_std: float
    n_inner_folds: int


@dataclass
class OuterFoldResult:
    """Result from outer CV evaluation."""
    fold_id: int
    selected_models: List[str]
    ensemble_ic: float
    ensemble_ic_pvalue: float
    individual_model_ics: Dict[str, float]
    n_test_obs: int


@dataclass
class NestedCVResult:
    """Complete nested CV result."""
    asset: str
    n_outer_folds: int
    ensemble_ic_mean: float
    ensemble_ic_std: float
    ensemble_ic_pvalue: float
    selection_stability: float  # How often each model was selected
    model_selection_counts: Dict[str, int]
    outer_fold_results: List[OuterFoldResult]


class NestedCVEnsembleEvaluator:
    """
    Evaluates ensemble selection using nested cross-validation.
    
    Outer loop: Evaluates the ensemble selection strategy
    Inner loop: Selects which models to include in ensemble
    """
    
    def __init__(
        self,
        n_outer_folds: int = 5,
        n_inner_folds: int = 4,
        outer_min_train_months: int = 120,
        inner_min_train_months: int = 84,
        ensemble_size: int = 5,
        forecast_horizon: int = 24
    ):
        self.n_outer_folds = n_outer_folds
        self.n_inner_folds = n_inner_folds
        self.outer_min_train_months = outer_min_train_months
        self.inner_min_train_months = inner_min_train_months
        self.ensemble_size = ensemble_size
        self.forecast_horizon = forecast_horizon
    
    def generate_outer_folds(self, X: pd.DataFrame) -> List[OuterFold]:
        """Generate outer CV folds with temporal splits."""
        dates = X.index.sort_values()
        n_total = len(dates)
        if n_total == 0:
            return []
            
        min_date = dates.min()
        
        min_outer_train = self.outer_min_train_months
        outer_test_months = 24
        
        folds = []
        total_months = (dates.max() - min_date).days / 30.44
        usable_months = total_months - min_outer_train - outer_test_months
        
        if usable_months <= 0:
            logger.warning(f"Insufficient data for {self.n_outer_folds} outer folds. Found {total_months:.1f} months.")
            # Fallback: single fold with 80/20 split
            usable_months = total_months * 0.2
            fold_step = 0
            n_folds = 1
        else:
            fold_step = usable_months / max(1, self.n_outer_folds - 1)
            n_folds = self.n_outer_folds
        
        for i in range(n_folds):
            outer_train_end_offset = min_outer_train + i * fold_step
            outer_train_end = min_date + pd.DateOffset(months=int(outer_train_end_offset))
            outer_test_start = outer_train_end + pd.DateOffset(days=1)
            outer_test_end = min(
                outer_test_start + pd.DateOffset(months=outer_test_months),
                dates.max()
            )
            
            train_mask = (dates >= min_date) & (dates <= outer_train_end)
            test_mask = (dates >= outer_test_start) & (dates <= outer_test_end)
            
            if train_mask.sum() > 20 and test_mask.sum() > 5:
                folds.append(OuterFold(
                    fold_id=i,
                    outer_train_start=min_date,
                    outer_train_end=outer_train_end,
                    outer_test_start=outer_test_start,
                    outer_test_end=outer_test_end,
                    outer_train_indices=np.where(train_mask)[0],
                    outer_test_indices=np.where(test_mask)[0]
                ))
        
        logger.info(f"Generated {len(folds)} outer folds")
        return folds
    
    def run_inner_cv(
        self,
        X_outer_train: pd.DataFrame,
        y_outer_train: pd.Series,
        model_factories: Dict[str, Callable[[], Any]],
        pipeline_factory: Callable[[Any], Any]
    ) -> Dict[str, InnerFoldResult]:
        """Run inner CV to evaluate each model on outer training set."""
        from .cross_validation import TimeSeriesCV, CrossValidator
        
        inner_cv = TimeSeriesCV(
            min_train_months=self.inner_min_train_months,
            validation_months=12,
            step_months=6
        )
        validator = CrossValidator(inner_cv)
        
        inner_results = {}
        
        for model_name, model_factory in model_factories.items():
            try:
                # Fresh model and pipeline for this outer fold
                model = model_factory()
                pipeline = pipeline_factory(model)
                
                result = validator.evaluate(
                    pipeline, X_outer_train, y_outer_train,
                    model_name=model_name, asset="inner", target="return"
                )
                
                inner_results[model_name] = InnerFoldResult(
                    model_name=model_name,
                    inner_ic_mean=result.metrics.get('IC_mean', np.nan),
                    inner_ic_std=result.metrics.get('IC_std', np.nan),
                    n_inner_folds=result.n_folds
                )
            except Exception as e:
                logger.warning(f"Inner CV failed for {model_name}: {e}")
                inner_results[model_name] = InnerFoldResult(
                    model_name=model_name,
                    inner_ic_mean=np.nan,
                    inner_ic_std=np.nan,
                    n_inner_folds=0
                )
        
        return inner_results
    
    def select_ensemble_models(
        self,
        inner_results: Dict[str, InnerFoldResult]
    ) -> List[str]:
        """Select top N models based on inner CV performance."""
        valid_results = [
            (name, r) for name, r in inner_results.items()
            if not np.isnan(r.inner_ic_mean)
        ]
        
        if not valid_results:
            return []
            
        sorted_results = sorted(
            valid_results,
            key=lambda x: x[1].inner_ic_mean,
            reverse=True
        )
        
        selected = [name for name, _ in sorted_results[:self.ensemble_size]]
        logger.debug(f"Selected models based on inner CV: {selected}")
        return selected
    
    def evaluate_ensemble_on_outer_test(
        self,
        X_outer_train: pd.DataFrame,
        y_outer_train: pd.Series,
        X_outer_test: pd.DataFrame,
        y_outer_test: pd.Series,
        selected_models: List[str],
        model_factories: Dict[str, Callable[[], Any]],
        pipeline_factory: Callable[[Any], Any]
    ) -> Tuple[float, float, Dict[str, float]]:
        """Train selected models on outer train, evaluate on outer test."""
        from .inference import compute_ic_with_inference
        from scipy.stats import spearmanr
        
        predictions = {}
        individual_ics = {}
        
        y_test_clean = y_outer_test.dropna()
        if len(y_test_clean) == 0:
            return np.nan, np.nan, {}
            
        for model_name in selected_models:
            try:
                model = model_factories[model_name]()
                pipeline = pipeline_factory(model)
                pipeline.fit(X_outer_train, y_outer_train)
                
                preds = pipeline.predict(X_outer_test)
                
                if isinstance(preds, (pd.Series, pd.DataFrame)):
                    preds_val = preds.values.flatten()
                else:
                    preds_val = preds.flatten()
                    
                predictions[model_name] = preds_val
                
                # Compute individual IC on outer test
                ic, _ = spearmanr(
                    y_test_clean.values,
                    preds_val[y_outer_test.notna()]
                )
                individual_ics[model_name] = ic
            except Exception as e:
                logger.warning(f"Failed to evaluate {model_name} on outer test: {e}")
        
        if not predictions:
            return np.nan, np.nan, {}
        
        # Average predictions
        pred_matrix = np.column_stack(list(predictions.values()))
        ensemble_pred = np.mean(pred_matrix, axis=1)
        ensemble_pred_series = pd.Series(ensemble_pred, index=y_outer_test.index)
        
        inference = compute_ic_with_inference(
            y_outer_test, ensemble_pred_series, horizon=self.forecast_horizon
        )
        
        return inference.estimate, inference.p_value_nw, individual_ics
    
    def evaluate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_factories: Dict[str, Callable[[], Any]],
        pipeline_factory: Callable[[Any], Any],
        asset: str = "unknown"
    ) -> NestedCVResult:
        """Run complete nested CV evaluation."""
        logger.info(f"Running nested CV for {asset}")
        
        outer_folds = self.generate_outer_folds(X)
        if not outer_folds:
            return NestedCVResult(asset, 0, np.nan, np.nan, np.nan, 0, {}, [])
            
        outer_results = []
        model_selection_counts = {name: 0 for name in model_factories.keys()}
        
        for fold in outer_folds:
            logger.info(f"  Outer fold {fold.fold_id} ({fold.outer_test_start.date()} to {fold.outer_test_end.date()})")
            
            X_outer_train = X.iloc[fold.outer_train_indices]
            y_outer_train = y.iloc[fold.outer_train_indices]
            X_outer_test = X.iloc[fold.outer_test_indices]
            y_outer_test = y.iloc[fold.outer_test_indices]
            
            # Step 1: Run inner CV to select models
            inner_results = self.run_inner_cv(
                X_outer_train, y_outer_train,
                model_factories, pipeline_factory
            )
            
            # Step 2: Select models based on inner CV
            selected_models = self.select_ensemble_models(inner_results)
            
            if not selected_models:
                logger.warning(f"  No models selected for outer fold {fold.fold_id}")
                continue
                
            for model in selected_models:
                model_selection_counts[model] += 1
            
            # Step 3: Evaluate on outer test
            ensemble_ic, ensemble_pvalue, individual_ics = self.evaluate_ensemble_on_outer_test(
                X_outer_train, y_outer_train,
                X_outer_test, y_outer_test,
                selected_models, model_factories, pipeline_factory
            )
            
            outer_results.append(OuterFoldResult(
                fold_id=fold.fold_id,
                selected_models=selected_models,
                ensemble_ic=ensemble_ic,
                ensemble_ic_pvalue=ensemble_pvalue,
                individual_model_ics=individual_ics,
                n_test_obs=len(y_outer_test.dropna())
            ))
        
        # Aggregate results
        ics = [r.ensemble_ic for r in outer_results if not np.isnan(r.ensemble_ic)]
        pvalues = [r.ensemble_ic_pvalue for r in outer_results if not np.isnan(r.ensemble_ic_pvalue)]
        
        # Compute selection stability
        max_selections = len(outer_results)
        if max_selections > 0:
            selection_rates = [count / max_selections for count in model_selection_counts.values()]
            top_n_rates = sorted(selection_rates, reverse=True)[:self.ensemble_size]
            selection_stability = np.mean(top_n_rates)
        else:
            selection_stability = 0.0
        
        return NestedCVResult(
            asset=asset,
            n_outer_folds=len(outer_results),
            ensemble_ic_mean=np.mean(ics) if ics else np.nan,
            ensemble_ic_std=np.std(ics) if ics else np.nan,
            ensemble_ic_pvalue=np.mean(pvalues) if pvalues else np.nan,
            selection_stability=selection_stability,
            model_selection_counts=model_selection_counts,
            outer_fold_results=outer_results
        )


def format_nested_cv_report(result: NestedCVResult) -> str:
    """Format nested CV result as human-readable report."""
    if result.n_outer_folds == 0:
        return f"Nested CV failed for {result.asset}: No valid folds."
        
    lines = [
        "=" * 60,
        f"NESTED CV ENSEMBLE EVALUATION: {result.asset}",
        "=" * 60,
        "",
        f"Ensemble IC (unbiased): {result.ensemble_ic_mean:.3f} ± {result.ensemble_ic_std:.3f}",
        f"Ensemble p-value: {result.ensemble_ic_pvalue:.3f}",
        f"Selection stability: {result.selection_stability:.1%}",
        "",
        "Model Selection Frequency:",
        "-" * 40,
    ]
    
    sorted_counts = sorted(
        result.model_selection_counts.items(),
        key=lambda x: x[1], reverse=True
    )
    
    for model, count in sorted_counts:
        if count == 0: continue
        pct = count / result.n_outer_folds * 100
        bar = "█" * int(pct / 5)
        lines.append(f"  {model:<15} {count}/{result.n_outer_folds} ({pct:5.1f}%) {bar}")
    
    lines.append("=" * 60)
    return "\n".join(lines)
