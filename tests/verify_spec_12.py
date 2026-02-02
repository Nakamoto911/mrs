import yaml
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

def verify_spec_12_integration():
    print("Starting Spec 12 Verification...")
    
    # 1. Load Config
    config_path = "configs/experiment_config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    # 2. Check Momentum Windows Config
    print(f"Checking momentum_windows in config: {config['features']['momentum_windows']}")
    assert config['features']['momentum_windows'] == [6, 12, 18], f"Expected [6, 12, 18], got {config['features']['momentum_windows']}"
    
    # 3. Check Holdout Config
    print(f"Checking holdout enabled in config: {config['validation']['holdout']['enabled']}")
    assert config['validation']['holdout']['enabled'] is False, "Expected holdout enabled to be False"

    # 4. Check Momentum Defaults (Import check)
    from feature_engineering.momentum import MomentumFeatureGenerator, generate_all_momentum_features
    print(f"Checking MomentumFeatureGenerator.DEFAULT_WINDOWS: {MomentumFeatureGenerator.DEFAULT_WINDOWS}")
    assert MomentumFeatureGenerator.DEFAULT_WINDOWS == [6, 12, 18], f"Expected [6, 12, 18], got {MomentumFeatureGenerator.DEFAULT_WINDOWS}"
    
    # Check function signature default
    import inspect
    sig = inspect.signature(generate_all_momentum_features)
    default_windows = sig.parameters['windows'].default
    print(f"Checking generate_all_momentum_features default windows: {default_windows}")
    assert default_windows == [6, 12, 18], f"Expected [6, 12, 18], got {default_windows}"
    
    print("✓ Spec 12 Integration Verified Successfully!")

if __name__ == "__main__":
    try:
        verify_spec_12_integration()
    except AssertionError as e:
        print(f"✗ Verification Failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Error during verification: {e}")
        sys.exit(1)
