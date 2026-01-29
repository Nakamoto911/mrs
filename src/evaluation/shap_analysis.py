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
            
            elif model_type == 'linear':
                # Linear models
                if hasattr(model, 'model'):
                    underlying = model.model
                else:
                    underlying = model
                
                # Scale features if scaler available
                if hasattr(model, 'scaler') and model.scaler is not None:
                    X_scaled = model.scaler.transform(X_clean)
                else:
                    X_scaled = X_clean.values
                
                self.explainer = shap.LinearExplainer(underlying, X_scaled)
                self.shap_values = self.explainer.shap_values(X_scaled)
            
            else:
                # Kernel SHAP (slowest but works for any model)
                def predict_fn(x):
                    if hasattr(model, 'predict'):
                        return model.predict(pd.DataFrame(x, columns=self.feature_names))
                    return model(x)
                
                # Use smaller background dataset for speed
                background = shap.kmeans(X_clean, 50)
                self.explainer = shap.KernelExplainer(predict_fn, background)
                self.shap_values = self.explainer.shap_values(X_clean)
            
            return self.shap_values
        
        except Exception as e:
            logger.warning(f"SHAP computation failed: {e}")
            return self._fallback_importance(model, X)
    
    def _fallback_importance(self, model: Any, X: pd.DataFrame) -> np.ndarray:
        """
        Fallback to model's native feature importance.
        
        Args:
            model: Fitted model
            X: Feature DataFrame
            
        Returns:
            Importance array
        """
        self.feature_names = X.columns.tolist()
        
        if hasattr(model, 'get_feature_importance'):
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
