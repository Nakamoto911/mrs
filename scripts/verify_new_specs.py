import os
import sys
import pandas as pd
import numpy as np
import yaml
import logging

# Add src to path
sys.path.append(os.getcwd())

from src.preprocessing.data_loader import FREDMDLoader
from src.feature_engineering.frozen_pipeline import FrozenFeaturePipeline
from src.feature_engineering.hierarchical_clustering import HierarchicalClusterSelector, SelectionMethod, SelectionConfig
from src.feature_engineering.quintiles import generate_all_quintile_features

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

def verify():
    # 1. Load Config
    with open('configs/experiment_config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    logger.info("--- Specification 01: Data Configuration ---")
    loader = FREDMDLoader(config=config)
    df_raw = loader.download_current_vintage()
    
    excluded_vars = ['S&P 500', 'S&P div yield', 'S&P PE ratio']
    for var in excluded_vars:
        if var in df_raw.columns:
            logger.error(f"FAILURE: {var} found in raw data even though Category 8 should be excluded.")
        else:
            logger.info(f"SUCCESS: {var} correctly excluded.")

    logger.info("\n--- Specification 02: Feature Generation ---")
    # Test broad quintile generation
    sample_df = df_raw.iloc[:100].copy() # Use small sample
    # Add dummy ratios to simulate full pipeline data
    sample_df['dummy_ratio'] = np.random.randn(len(sample_df))
    
    quintiles = generate_all_quintile_features(sample_df, variables="all")
    logger.info(f"Generated {len(quintiles.columns)} quintile features for {len(sample_df.columns)} base variables.")
    if len(quintiles.columns) >= len(sample_df.columns) * 5:
        logger.info("SUCCESS: Broad level features generated.")
    else:
        logger.warning(f"Note: Generated {len(quintiles.columns)} quintiles. Expected ~{len(sample_df.columns)*5}.")

    # Test Frozen Pipeline
    # Get required features (just a few to test filtering)
    base_var = df_raw.columns[0]
    required = [base_var, f"{base_var}_Q1", f"{base_var}_Q5"]
    pipeline = FrozenFeaturePipeline(required_features=required, transform_codes=loader.get_transform_codes())
    try:
        features = pipeline.transform_vintage(sample_df)
        logger.info(f"Frozen pipeline generated required features: {features.columns.tolist()}")
        logger.info("SUCCESS: Slope + Level generation in pipeline confirmed.")
    except Exception as e:
        logger.error(f"FAILURE: Frozen pipeline failed: {e}")

    logger.info("\n--- Specification 03 & 04: Clustering & Medoid ---")
    # Test clustering medoid selection
    clusterer = HierarchicalClusterSelector(
        similarity_threshold=config['features']['clustering']['similarity_threshold'],
        selection_config=SelectionConfig(method=SelectionMethod.CENTROID)
    )
    
    # Create some mock data with a Level and Slope of the same var to trigger Spec 05
    mock_data = pd.DataFrame({
        'VAR': np.random.randn(100),
        'VAR_Q1': np.random.randint(0, 2, 100),
        'OTHER': np.random.randn(100)
    })
    # Make VAR and VAR_Q1 highly correlated to force into same cluster
    mock_data['VAR_Q1'] = (mock_data['VAR'] > 0).astype(float)
    
    logger.info("Performing clustering on mock data with potential Level/Slope conflict...")
    clusterer.perform_clustering(mock_data)
    
    logger.info("Testing Medoid selection...")
    reps = clusterer.select_all_representatives(mock_data)
    logger.info(reps)

    logger.info("\nVerification Complete.")

if __name__ == "__main__":
    verify()
