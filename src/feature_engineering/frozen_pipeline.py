"""
Frozen Feature Pipeline
=======================
Ensures consistent feature generation for validation using point-in-time data.
"""

import pandas as pd
import numpy as np
from typing import List, Optional, Dict
import logging

from src.preprocessing.transformations import FREDMDTransformer
from src.feature_engineering.ratios import generate_all_ratio_features
from src.feature_engineering.quintiles import generate_all_quintile_features
from src.feature_engineering.momentum import generate_all_momentum_features
from src.feature_engineering.cointegration import generate_cointegration_features

logger = logging.getLogger(__name__)

class FrozenFeaturePipeline:
    """
    Wraps existing feature generators to strictly reproduce features for a frozen model.
    """
    def __init__(self, required_features: List[str], transform_codes: Optional[Dict[str, int]] = None):
        """
        Args:
            required_features: List of feature names the model expects.
            transform_codes: Dictionary of transformer codes for FRED-MD series.
        """
        self.required_features = required_features
        self.transform_codes = transform_codes or {}

    def transform_vintage(self, vintage_df: pd.DataFrame) -> pd.DataFrame:
        """
        Runs the feature engineering pipeline and filters to required features.
        """
        logger.debug(f"Transforming vintage with {len(vintage_df)} rows")
        
        # 1. Level Generation: Generate quintiles on Raw vintage data for ALL columns (Spec 02)
        # Apply Stationarity Check bypass: Levels are valid even on non-stationary raw data
        quintiles = generate_all_quintile_features(vintage_df, variables="all")
        
        # 2. Slope Generation: Apply stationarity transformations to Raw data
        transformer = FREDMDTransformer(self.transform_codes)
        transformed = transformer.transform_dataframe(vintage_df)
        
        # 3. Ratios (also Slope features)
        ratios = generate_all_ratio_features(vintage_df)
        
        # 4. Cointegration (ECT)
        try:
            ect_features, _ = generate_cointegration_features(vintage_df)
        except Exception as e:
            logger.warning(f"Error generating cointegration features: {e}")
            ect_features = pd.DataFrame(index=vintage_df.index)
            
        # 5. Momentum (Uses transformed series)
        momentum = generate_all_momentum_features(transformed)
        
        # Combine all to create the full ~750 feature pool
        all_generated = pd.concat([
            transformed,    # Slope: Stationarity transformed
            ratios,         # Slope: Ratios
            quintiles,      # Level: Quintiles (all columns)
            ect_features,   # Level/Slope: Cointegration
            momentum        # Slope: Momentum
        ], axis=1)
        
        # Remove duplicates
        all_generated = all_generated.loc[:, ~all_generated.columns.duplicated()]
        
        # 6. Filter to required features and handle missing
        collected_features = {}
        
        missing_count = 0
        for feat in self.required_features:
            if feat in all_generated.columns:
                collected_features[feat] = all_generated[feat]
            else:
                # Handle feature not present in this vintage (e.g. series didn't exist yet)
                collected_features[feat] = pd.Series(0.0, index=vintage_df.index)
                missing_count += 1
                
        if missing_count > 0:
            logger.debug(f"Note: {missing_count} features filled with 0 due to historical data gaps.")
            
        final_features = pd.DataFrame(collected_features, index=vintage_df.index)
        
        # Order columns to match required_features
        final_features = final_features[self.required_features]
        
        # Final cleanup: fill NaNs appearing in generated features (e.g. due to lags)
        # We fill with the last available value (PIT) then 0
        final_features = final_features.ffill().fillna(0.0)
        
        return final_features
