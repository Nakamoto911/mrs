"""
SHAP Analysis Module
====================
SHAP-based feature importance and dominant driver identification.

Part of the Asset-Specific Macro Regime Detection System
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class SHAPAnalyzer:
    """
    SHAP-based feature importance analysis.
    
    Identifies dominant macro drivers for each asset-regime combination.
    """
    
    def __init__(self, n_top_features: int = 10):
        """
        Initialize SHAP analyzer.
        
        Args:
            n_top_features: Number of top features to report
        """
        self.n_top_features = n_top_features
        self.shap_values = None
        self.feature_names = None
        self.explainer = None
    
    def compute_shap_values(self, model: Any, X: pd.DataFrame,
                           model_type: str = 'tree') -> np.ndarray:
        """
        Compute SHAP values for model predictions.
        
        Args:
            model: Fitted model
            X: Feature DataFrame
            model_type: 'tree', 'linear', or 'kernel'
            
        Returns:
            SHAP values array (n_samples x n_features)
        """
        try:
            import shap
            from sklearn.pipeline import Pipeline
        except ImportError:
            logger.warning("SHAP not available, using permutation importance")
            return self._fallback_importance(model, X)
        
        # Handle Pipeline
        if isinstance(model, Pipeline):
            try:
                # Transform features using all steps except the last one
                if len(model.steps) > 1:
                    logger.info("Transforming features using pipeline pre-processors for SHAP...")
                    transformers = Pipeline(model.steps[:-1])
                    X = transformers.transform(X)
                
                # Get the final estimator
                model = model.steps[-1][1]
                logger.info(f"Extracted estimator from pipeline: {type(model).__name__}")
                
            except Exception as e:
                logger.warning(f"Failed to process pipeline for SHAP: {e}")
                return self._fallback_importance(model, X)

                # Get the final estimator
                model = model.steps[-1][1]
                logger.info(f"Extracted estimator from pipeline: {type(model).__name__}")
                
            except Exception as e:
                logger.warning(f"Failed to process pipeline for SHAP: {e}")
                return self._fallback_importance(model, X)

        # Handle EnsembleModel (Optimization)
        # Check if model has 'estimators' attribute (duck typing for EnsembleModel)
        if hasattr(model, 'estimators') and hasattr(model, 'weights') and model_type == 'ensemble':
            logger.info(f"Detected EnsembleModel with {len(model.estimators)} estimators. Using optimized component-wise SHAP.")
            try:

                shap_values_list = []
                feature_names_list = []
                weights = model.weights if model.weights is not None else [1.0 / len(model.estimators)] * len(model.estimators)
                
                for i, estimator in enumerate(model.estimators):
                    logger.info(f"Computing SHAP for ensemble constituent {i+1}/{len(model.estimators)} ({type(estimator).__name__})...")
                    
                    # Determine constituent model type
                    # Look inside pipeline if needed
                    check_est = estimator
                    if isinstance(estimator, Pipeline):
                        # Use the final estimator for type checking
                        if len(estimator.steps) > 0:
                            check_est = estimator.steps[-1][1]
                    
                    est_str = str(type(check_est)).lower()
                    constituent_type = 'linear' # Default safe fallback
                    
                    if any(t in est_str for t in ['forest', 'boost', 'gbm', 'tree']):
                         constituent_type = 'tree'
                    elif any(t in est_str for t in ['neural', 'mlp', 'lstm', 'torch', 'keras']):
                         constituent_type = 'neural'
                    
                    # Recursive call for constituent
                    sub_analyzer = SHAPAnalyzer(n_top_features=self.n_top_features)
                    # Note: compute_shap_values modifies sub_analyzer.feature_names
                    sub_shap = sub_analyzer.compute_shap_values(estimator, X, model_type=constituent_type)
                    
                    shap_values_list.append(sub_shap)
                    feature_names_list.append(sub_analyzer.feature_names)
                
                # Validation: Ensure all constituents produced compatible SHAP values
                if not shap_values_list:
                    raise ValueError("No SHAP values computed for ensemble")
                
                # Check shapes
                base_shape = shap_values_list[0].shape
                base_features = feature_names_list[0]
                
                for i, (sv, fn) in enumerate(zip(shap_values_list[1:], feature_names_list[1:])):
                    if sv.shape != base_shape:
                        logger.warning(f"Shape mismatch in ensemble SHAP: Model 0 {base_shape} vs Model {i+1} {sv.shape}. This implies different feature selection.")
                        # If shapes differ, we cannot average simply. 
                        # We would need to align by feature name.
                        # For now, let's assume they match (as they should if using same pipeline on same data).
                        pass
                        
                # Update self.feature_names to match the transformed features
                self.feature_names = base_features
                
                # Weighted average
                logger.info("Aggregating SHAP values from ensemble constituents...")
                weighted_shap = np.zeros_like(shap_values_list[0])
                for sv, w in zip(shap_values_list, weights):
                    weighted_shap += sv * w
                
                self.shap_values = weighted_shap
                    
                return self.shap_values
                
            except Exception as e:
                logger.warning(f"Optimized Ensemble SHAP failed: {e}. Falling back to KernelExplainer.")
                # Fall through to default logic

        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            X_clean = X.fillna(0)
        else:
            # Handle numpy array (e.g. from StandardScaler)
            self.feature_names = [f"feature_{i}" for i in range(X.shape[1])]
            X_clean = pd.DataFrame(X, columns=self.feature_names).fillna(0)
        
        try:
            if model_type == 'tree':
                # Tree-based models (XGBoost, LightGBM, RandomForest)
                if hasattr(model, 'model'):
                    # Wrapper model (some implementations)
                    underlying = model.model
                elif hasattr(model, 'model_'):
                    # Wrapper model (TreeModelWrapper)
                    underlying = model.model_
                else:
                    underlying = model
                
                # Ensure we have the actual estimator (sometimes wrappers wrap wrappers)
                if isinstance(underlying, Pipeline):
                     logger.warning("Nested pipeline detected in wrapper, attempting to extract final step")
                     underlying = underlying.steps[-1][1]

                self.explainer = shap.TreeExplainer(underlying)
                self.shap_values = self.explainer.shap_values(X_clean)
            
            elif model_type == 'neural':
                # Neural networks (MLP, LSTM)
                # For PyTorch wrappers, we use KernelExplainer for robustness
                def predict_fn(x):
                    # Ensure input is DataFrame with correct feature names
                    df = pd.DataFrame(x, columns=self.feature_names)
                    if hasattr(model, 'predict'):
                        preds = model.predict(df)
                        # Ensure output is 1D numpy array
                        if isinstance(preds, pd.Series):
                            return preds.values
                        return preds
                    return model(torch.FloatTensor(x)).detach().numpy()
                
                # Use smaller background dataset for speed
                # 50 samples is a standard balance for performance
                background = shap.kmeans(X_clean, 50)
                self.explainer = shap.KernelExplainer(predict_fn, background)
                
                # Performance Optimization for Neural Models:
                # 1. Subsample foreground to the most recent 100 samples if needed
                # 2. Limit the number of permutations for Kernel SHAP
                if len(X_clean) > 100:
                    logger.info(f"Subsampling foreground data from {len(X_clean)} to last 100 samples for neural SHAP speed...")
                    X_foreground = X_clean.iloc[-100:]
                else:
                    X_foreground = X_clean
                
                self.shap_values = self.explainer.shap_values(X_foreground, nsamples=100)
                
                # Performance Optimization: Pad results to maintain index alignment
                if len(X_clean) > 100:
                    logger.info("Padding SHAP values with zeros for older samples to maintain alignment...")
                    padded_shap = np.zeros((len(X_clean), X_clean.shape[1]))
                    padded_shap[-100:] = self.shap_values
                    self.shap_values = padded_shap
                
                return self.shap_values
            
            else:
                # Kernel SHAP (fallback for any other model)
                def predict_fn(x):
                    if hasattr(model, 'predict'):
                        preds = model.predict(pd.DataFrame(x, columns=self.feature_names))
                        if isinstance(preds, pd.Series):
                            return preds.values
                        return preds
                    return model(x)
                
                background = shap.kmeans(X_clean, 50)
                self.explainer = shap.KernelExplainer(predict_fn, background)
                self.shap_values = self.explainer.shap_values(X_clean)
            
            return self.shap_values
        
        except Exception as e:
            logger.warning(f"SHAP computation failed: {e}")
            return self._fallback_importance(model, X_clean if 'X_clean' in locals() else X)
    
    def _fallback_importance(self, model: Any, X: pd.DataFrame) -> np.ndarray:
        """
        Fallback to model's native feature importance.
        
        Args:
            model: Fitted model
            X: Feature DataFrame
            
        Returns:
            Importance array
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
            n_features = len(X.columns)
        else:
            n_features = X.shape[1]
            self.feature_names = [f"feature_{i}" for i in range(n_features)]
            importance = model.get_feature_importance()
            # Convert to SHAP-like format (repeat for each sample)
            return np.tile(importance.values, (len(X), 1))
        
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            return np.tile(importance, (len(X), 1))
        
        if hasattr(model, 'coef_'):
            importance = np.abs(model.coef_)
            return np.tile(importance, (len(X), 1))
        
        # Default: uniform importance
        n_features = len(X.columns)
        return np.ones((len(X), n_features)) / n_features
    
    def get_mean_abs_shap(self) -> pd.Series:
        """
        Get mean absolute SHAP values (overall feature importance).
        
        Returns:
            Series with mean |SHAP| per feature, sorted descending
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed")
        
        mean_abs = np.abs(self.shap_values).mean(axis=0)
        
        return pd.Series(mean_abs, index=self.feature_names).sort_values(ascending=False)
    
    def get_top_features(self, n: Optional[int] = None) -> pd.DataFrame:
        """
        Get top N features by SHAP importance.
        
        Args:
            n: Number of features (uses default if None)
            
        Returns:
            DataFrame with top features and their SHAP values
        """
        n = n or self.n_top_features
        
        mean_shap = self.get_mean_abs_shap()
        top_features = mean_shap.head(n)
        
        return pd.DataFrame({
            'Feature': top_features.index,
            'Mean_Abs_SHAP': top_features.values,
            'Rank': range(1, n + 1)
        }).set_index('Rank')
    
    def get_regime_conditional_shap(self, regime_labels: pd.Series) -> Dict[str, pd.Series]:
        """
        Compute SHAP values separately for bullish and bearish regimes.
        
        Args:
            regime_labels: Series with regime labels (1=bullish, 0=bearish)
            
        Returns:
            Dictionary with 'bullish' and 'bearish' SHAP values
        """
        if self.shap_values is None:
            raise ValueError("SHAP values not computed")
        
        # Align indices
        common_idx = regime_labels.index.intersection(
            pd.Index(range(len(self.shap_values)))
        )
        
        results = {}
        
        for regime_name, regime_val in [('bullish', 1), ('bearish', 0)]:
            mask = regime_labels == regime_val
            if mask.sum() > 0:
                regime_shap = self.shap_values[mask]
                mean_abs = np.abs(regime_shap).mean(axis=0)
                results[regime_name] = pd.Series(
                    mean_abs, index=self.feature_names
                ).sort_values(ascending=False)
        
        return results
    
    def generate_monitoring_sheet(self, X: pd.DataFrame,
                                  percentiles: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Generate dominant driver monitoring sheet.
        
        Args:
            X: Current feature values
            percentiles: Historical percentile DataFrame
            
        Returns:
            Monitoring sheet DataFrame
        """
        top_features = self.get_top_features()
        
        rows = []
        for rank, row in top_features.iterrows():
            feature = row['Feature']
            shap_val = row['Mean_Abs_SHAP']
            
            # Get current value
            current_val = X[feature].iloc[-1] if feature in X.columns else np.nan
            
            # Get percentile
            if percentiles is not None and feature in percentiles.columns:
                current_pct = percentiles[feature].iloc[-1]
            else:
                # Compute expanding percentile
                if feature in X.columns:
                    series = X[feature].dropna()
                    current_pct = (series < current_val).sum() / len(series) * 100
                else:
                    current_pct = np.nan
            
            # Determine signal
            signal = self._get_signal(current_pct)
            
            rows.append({
                'Rank': rank,
                'Feature': feature,
                'SHAP': shap_val,
                'Current_Value': current_val,
                'Percentile': current_pct,
                'Signal': signal
            })
        
        return pd.DataFrame(rows)
    
    def _get_signal(self, percentile: float) -> str:
        """Convert percentile to signal."""
        if np.isnan(percentile):
            return '? Unknown'
        elif percentile >= 80:
            return '↑ Very High'
        elif percentile >= 60:
            return '↗ High'
        elif percentile >= 40:
            return '→ Neutral'
        elif percentile >= 20:
            return '↘ Low'
        else:
            return '↓ Very Low'


def compute_shap_for_model(model: Any, X: pd.DataFrame,
                          model_type: str = 'tree',
                          n_top: int = 10) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Convenience function to compute SHAP values and get top features.
    
    Args:
        model: Fitted model
        X: Feature DataFrame
        model_type: Model type for SHAP
        n_top: Number of top features
        
    Returns:
        Tuple of (top_features DataFrame, mean_abs_shap Series)
    """
    analyzer = SHAPAnalyzer(n_top_features=n_top)
    analyzer.compute_shap_values(model, X, model_type)
    
    top_features = analyzer.get_top_features()
    mean_shap = analyzer.get_mean_abs_shap()
    
    return top_features, mean_shap


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test with sklearn model
    np.random.seed(42)
    n = 200
    
    X = pd.DataFrame({
        'feature1': np.random.randn(n),
        'feature2': np.random.randn(n),
        'feature3': np.random.randn(n),
        'feature4': np.random.randn(n),
        'feature5': np.random.randn(n),
    })
    
    y = 0.5 * X['feature1'] + 0.3 * X['feature2'] + np.random.randn(n) * 0.1
    
    # Train a simple model
    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    
    # Compute SHAP
    analyzer = SHAPAnalyzer()
    analyzer.compute_shap_values(model, X, 'tree')
    
    print("Top Features:")
    print(analyzer.get_top_features())
    
    print("\nMonitoring Sheet:")
    print(analyzer.generate_monitoring_sheet(X))
