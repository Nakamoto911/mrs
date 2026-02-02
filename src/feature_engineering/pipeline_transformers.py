import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class QuintileTransformer(BaseEstimator, TransformerMixin):
    """
    Generates quintile dummy variables based strictly on training data distribution.
    Prevents look-ahead bias by learning thresholds (cuts) only from fit().
    """
    def __init__(self, n_quintiles=5, variables='all'):
        self.n_quintiles = n_quintiles
        self.variables = variables
        self.bins_ = {}
        self.feature_names_in_ = None

    def fit(self, X, y=None):
        # Validate input
        if isinstance(X, pd.DataFrame):
            self.feature_names_in_ = X.columns
            vars_to_process = X.columns if self.variables == 'all' else self.variables
        else:
            # If X is numpy array, assume all columns
            vars_to_process = range(X.shape[1])

        # Learn thresholds from Training Data ONLY
        for col in vars_to_process:
            try:
                series = X[col] if isinstance(X, pd.DataFrame) else X[:, col]
                # Calculate quantiles (e.g., 0, 0.2, 0.4, 0.6, 0.8, 1.0)
                # We use linspace to get the edges
                _, bins = pd.qcut(series, self.n_quintiles, retbins=True, duplicates='drop')
                self.bins_[col] = bins
            except Exception:
                # Skip columns that can't be binned (e.g. constant values)
                continue
        return self

    def transform(self, X):
        X_in = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        new_features = []
        
        for col, bins in self.bins_.items():
            if col not in X_in.columns:
                continue
                
            # Apply thresholds learned in fit()
            # This assigns 0..4 based on the bins
            binned = pd.cut(X_in[col], bins=bins, labels=False, include_lowest=True)
            
            # Create One-Hot features
            # e.g., CPI_Q5 (Highest Quintile)
            for q in range(self.n_quintiles):
                # We typically care most about extremes: Q1 (Low) and Q5 (High)
                # But let's generate all to match previous logic, or just 1 and 5.
                # Naming convention: {Col}_Q{q+1}
                col_name = f"{col}_Q{q+1}"
                feature = (binned == q).astype(int)
                feature.name = col_name
                new_features.append(feature)
        
        # Concatenate all new features at once to avoid fragmentation
        if new_features:
            X_new = pd.concat(new_features, axis=1)
            X_out = pd.concat([X_in, X_new], axis=1)
        else:
            X_out = X_in
            
        return X_out
