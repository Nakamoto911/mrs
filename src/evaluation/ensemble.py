"""
Ensemble Evaluation Module
==========================
Logic for loading, aligning, and averaging multiple model predictions.

Part of the Asset-Specific Macro Regime Detection System
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)

class EnsembleEvaluator:
    """
    Encapsulates logic for creating ensemble predictions from multiple models.
    """
    
    def __init__(self):
        pass
    
    def load_and_average(self, file_paths: List[Path]) -> pd.DataFrame:
        """
        Load multiple prediction CSVs, align them by date, and compute the average signal.
        
        Args:
            file_paths: List of paths to prediction CSV files 
                        (expected columns: ['date', 'actual', 'predicted'])
                        NOTE: assumed to have date index or 'date' column.
        
        Returns:
            DataFrame with columns ['predicted', 'actual']
        """
        if not file_paths:
            raise ValueError("No file paths provided for ensemble")
            
        dfs = []
        for path in file_paths:
            if not path.exists():
                logger.warning(f"Prediction file not found: {path}")
                continue
                
            df = pd.read_csv(path, index_col=0)
            
            # Ensure Date index
            if not isinstance(df.index, pd.DatetimeIndex):
                try:
                    df.index = pd.to_datetime(df.index)
                except:
                    logger.error(f"Could not parse date index for {path}")
                    continue
            
            # Drop any NaT/NaN indices that might cause join/KeyErrors
            if df.index.isna().any():
                df = df.loc[df.index.notna()]
            
            dfs.append(df)
            
        if not dfs:
            raise ValueError("No valid prediction files loaded")
            
        # Perform Inner Join to ensure only overlapping valid periods are included
        # We start with the first DF and join the others
        ensemble_df = dfs[0][['predicted', 'actual']].copy()
        ensemble_df = ensemble_df.rename(columns={'predicted': 'pred_0'})
        
        for i, df in enumerate(dfs[1:], 1):
            ensemble_df = ensemble_df.join(df[['predicted']].rename(columns={'predicted': f'pred_{i}'}), how='inner')
            
        # Compute row-wise mean of the predicted columns
        pred_cols = [c for c in ensemble_df.columns if c.startswith('pred_')]
        ensemble_df['predicted'] = ensemble_df[pred_cols].mean(axis=1)
        
        # Keep only the final predicted and actual
        result = ensemble_df[['predicted', 'actual']].copy()
        
        return result

    def compute_ensemble_metrics(self, ensemble_df: pd.DataFrame) -> dict:
        """
        Compute standard metrics for the ensemble signal.
        """
        from scipy.stats import spearmanr
        
        y_true = ensemble_df['actual']
        y_pred = ensemble_df['predicted']
        
        # Remove NaN
        mask = ~(y_true.isna() | y_pred.isna())
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        metrics = {}
        
        if len(y_true_clean) < 2:
            return {'IC_mean': np.nan, 'RMSE_mean': np.nan, 'hit_rate_mean': np.nan}
            
        # IC (Information Coefficient)
        if len(np.unique(y_true_clean)) > 1 and len(np.unique(y_pred_clean)) > 1:
            ic, _ = spearmanr(y_true_clean, y_pred_clean)
        else:
            ic = 0.0
        metrics['IC_mean'] = ic
        
        # RMSE
        rmse = np.sqrt(np.mean((y_true_clean - y_pred_clean) ** 2))
        metrics['RMSE_mean'] = rmse
        
        # Hit rate
        actual_sign = np.sign(y_true_clean)
        pred_sign = np.sign(y_pred_clean)
        hit_rate = np.mean(actual_sign == pred_sign)
        metrics['hit_rate_mean'] = hit_rate
        
        return metrics
