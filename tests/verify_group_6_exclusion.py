import os
import sys
import yaml
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

from src.preprocessing.data_loader import FREDMDLoader

def verify_exclusion():
    print("Testing Group 6 Exclusion...")
    
    # Load config
    config_path = Path("configs/experiment_config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Instantiate loader
    loader = FREDMDLoader(data_dir="data/raw", config=config)
    
    # 1. Check if category mapping loaded
    print(f"Total mappings: {len(loader.category_mapping)}")
    
    # Identify group 6 series from appendix directly for verification
    appendix_path = Path("data/fred_md/FRED-MD_historic_appendix.csv")
    df_app = pd.read_csv(appendix_path, encoding='latin-1')
    group_6_series = df_app[df_app['group'] == 6]['fred'].tolist()
    print(f"Found {len(group_6_series)} series in Group 6 in appendix")

    # 2. Check if mapping correctly contains group 6
    found_in_mapping = [s for s in group_6_series if s in loader.category_mapping]
    print(f"Found {len(found_in_mapping)} Group 6 series in loader's mapping")
    
    # 3. Load data and verify exclusion
    try:
        df = loader.download_current_vintage()
        print(f"Loaded data shape: {df.shape}")
        
        # Check if any group 6 series are in the final columns
        present_group_6 = [s for s in group_6_series if s in df.columns]
        
        if not present_group_6:
            print("SUCCESS: No Group 6 series found in loaded data.")
        else:
            print(f"FAILURE: Found {len(present_group_6)} Group 6 series in data: {present_group_6}")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error during data loading: {e}")
        sys.exit(1)

if __name__ == "__main__":
    verify_exclusion()
