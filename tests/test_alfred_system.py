"""
Verification script for ALFRED Validation System
"""
import pandas as pd
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.preprocessing.alfred_loader import ALFREDVintageLoader
from src.feature_engineering.frozen_pipeline import FrozenFeaturePipeline
from src.evaluation.alfred_validation import ALFREDValidator

def test_loader():
    print("Testing ALFREDVintageLoader...")
    loader = ALFREDVintageLoader("data/raw/vintages")
    vintages = loader.get_available_vintages()
    print(f"Available vintages: {vintages}")
    
    if not vintages:
        print("No vintages found. Skipping load test.")
        return None
        
    df = loader.load_vintage(vintages[0])
    print(f"Loaded vintage {vintages[0]} with shape {df.shape}")
    return df

def test_pipeline(sample_df):
    if sample_df is None:
        return
        
    print("\nTesting FrozenFeaturePipeline...")
    # Mock some required features
    required = ['RPI', 'INDPRO', 'M2_GDP_Ratio', 'GS10_chg_12M']
    
    loader = ALFREDVintageLoader("data/raw/vintages")
    transform_codes = loader.get_transform_codes(pd.Timestamp("2024-12-31"))

    pipeline = FrozenFeaturePipeline(required, transform_codes)
    features = pipeline.transform_vintage(sample_df)
    
    print(f"Generated features shape: {features.shape}")
    print(f"Columns: {features.columns.tolist()}")
    
    for feat in required:
        if feat in features.columns:
            print(f"Feature {feat}: OK")
        else:
            print(f"Feature {feat}: MISSING (should be filled with 0)")

def test_validator():
    print("\nTesting ALFREDValidator...")
    # Find a model to test
    models_dir = Path("experiments/models")
    model_files = list(models_dir.glob("*.pkl"))
    
    if not model_files:
        print("No models found. Skipping validator test.")
        return
        
    model_path = str(model_files[0])
    print(f"Testing with model: {model_path}")
    
    # Validation dates (only use dates we have vintages for)
    dates = [pd.Timestamp("2024-12-31")]
    
    try:
        validator = ALFREDValidator(model_path, dates)
        print(f"Validator initialized. Features needed: {len(validator.features_needed)}")
        
        # Mock true returns
        true_returns = pd.Series(np.random.randn(len(dates)), index=dates)
        
        results = validator.run_validation(true_returns)
        print(f"Validation results shape: {results.shape}")
        print(results)
        
        metrics = validator.compute_metrics(results, revised_ic=0.25)
        print(f"Metrics: {metrics}")
        
    except Exception as e:
        print(f"Validator test failed: {e}")

if __name__ == "__main__":
    df = test_loader()
    test_pipeline(df)
    test_validator()
