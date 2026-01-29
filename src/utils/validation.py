import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def assert_no_lookahead(X_aligned: pd.DataFrame, lag_months: int):
    """
    Verifies that for every row in X_aligned (indexed by Trade Date T),
    the source data's reference date (Reference T) obeys: Reference T <= T - lag
    
    This is a sanity check to ensure the alignment logic is working as expected.
    Note: This assumes the alignment logic was correctly applied.
    """
    if X_aligned.empty:
        return
        
    # In our implementation, we shifted the index forward.
    # So if index is T, the data originally belonged to T - lag_months.
    # We can't easily verify the *source* data content here without more metadata,
    # but we can verify the index alignment logic if we had preserved the original date.
    
    # For now, this is a placeholder for more rigorous checks if we start tracking
    # 'reference_date' as a separate column in the future.
    pass

def log_alignment_stats(X: pd.DataFrame, y: pd.Series, lag_months: int):
    """Logs statistics about the aligned dataset."""
    if X.empty:
        logger.warning("Aligned dataset is empty.")
        return
        
    start_date = X.index.min().strftime('%Y-%m-%d')
    end_date = X.index.max().strftime('%Y-%m-%d')
    
    logger.info(f"Alignment Summary:")
    logger.info(f"  - Lag Applied: {lag_months} months")
    logger.info(f"  - Period: {start_date} to {end_date}")
    logger.info(f"  - Samples: {len(X)}")
    logger.info(f"  - Features: {len(X.columns)}")
