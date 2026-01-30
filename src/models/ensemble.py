"""
Ensemble Model Wrapper
======================
Scikit-learn compatible wrapper for ensemble models to support SHAP analysis.
"""

from typing import List, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin

class EnsembleModel(BaseEstimator, RegressorMixin):
    """
    Ensemble model that averages predictions from multiple base estimators.
    Follows scikit-learn API for compatibility with SHAP.
    """
    
    def __init__(self, estimators: List[Any], weights: List[float] = None):
        """
        Initialize ensemble.
        
        Args:
            estimators: List of fitted model objects (sklearn pipelines or estimators)
            weights: Optional weights for averaging (default: uniform)
        """
        self.estimators = estimators
        self.weights = weights
        self._fitted = True # Ensembles are constructed from already fitted models
        
    def fit(self, X, y):
        """
        Fit - no-op for this ensemble as it uses pre-trained models.
        Included for API compatibility.
        """
        return self
        
    def predict(self, X):
        """
        Predict by averaging base model predictions.
        """
        predictions = []
        for model in self.estimators:
            pred = model.predict(X)
            predictions.append(pred)
            
        return np.average(predictions, axis=0, weights=self.weights)
        
    def __sklearn_is_fitted__(self):
        return True
