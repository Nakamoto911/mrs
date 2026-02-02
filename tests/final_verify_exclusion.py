
import sys
import yaml
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from preprocessing import FREDMDLoader

def verify():
    # Load config
    with open('configs/experiment_config.yaml') as f:
        config = yaml.safe_load(f)
    
    print(f"Exclude categories in config: {config['data']['fred_md']['exclude_categories']}")
    
    # Instantiate loader WITH config
    loader = FREDMDLoader(data_dir="data/raw", config=config)
    
    # Download/Load current vintage
    df = loader.download_current_vintage()
    
    # Check if T10YFFM is in DF
    if 'T10YFFM' in df.columns:
        print("FAILURE: T10YFFM still found in data!")
    else:
        print("SUCCESS: T10YFFM successfully excluded.")
        
    # Check if any other Group 6 variables are there
    group_6_vars = ['FEDFUNDS', 'CP3Mx', 'TB3MS', 'TB6MS', 'GS1', 'GS5', 'GS10', 'AAA', 'BAA', 'COMPAPFFx', 'TB3SMFFM', 'TB6SMFFM', 'T1YFFM', 'T5YFFM', 'AAAFFM', 'BAAFFM']
    found = [v for v in group_6_vars if v in df.columns]
    if found:
        print(f"FAILURE: Other Group 6 variables found: {found}")
    else:
        print("SUCCESS: All checked Group 6 variables excluded.")

if __name__ == "__main__":
    verify()
