import joblib
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(os.getcwd()) / "src"))

model_path = "experiments/models/BOND_xgboost.pkl"
model = joblib.load(model_path)

print(f"Model type: {type(model)}")
print(f"Attributes: {dir(model)}")

if hasattr(model, 'feature_names_in_'):
    print(f"feature_names_in_: {model.feature_names_in_}")
elif hasattr(model, 'get_booster'):
    booster = model.get_booster()
    print(f"Booster feature names: {booster.feature_names}")
