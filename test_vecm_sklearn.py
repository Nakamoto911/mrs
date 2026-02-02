
import pandas as pd
import numpy as np
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.getcwd(), 'src'))

from models.linear_models import VECMWrapper
from sklearn.utils.validation import check_is_fitted
from sklearn.base import clone

def test_vecm_compliance():
    # 1. Instantiate
    model = VECMWrapper(max_lag=2, coint_rank=1)
    
    # 2. Check inheritance (implicitly done by clone or check_estimator, but let's just run fit)
    print("Inheritance check: BaseEstimator?", hasattr(model, 'get_params'))
    
    # 3. Create dummy data (High Dimensionality)
    n = 100
    n_features = 50
    X = pd.DataFrame(np.random.randn(n, n_features), columns=[f'f{i}' for i in range(n_features)])
    # Make f0 and f1 actually useful
    X['f0'] = X['f0'].cumsum()
    X['f1'] = X['f1'].cumsum()
    y = 0.5 * X['f0'] + 0.3 * X['f1'] + np.random.randn(n) * 0.1
    y = pd.Series(y, name='target')
    
    # 4. Fit
    print(f"Fitting model with {n_features} features (expecting reduction)...")
    model.fit(X, y)
    print("Model fitted.")
    if hasattr(model, 'selected_features_'):
        print(f"Selected {len(model.selected_features_)} features: {model.selected_features_}")
    
    # 5. Check is_fitted
    print("Checking is_fitted...")
    try:
        check_is_fitted(model)
        print("check_is_fitted passed directly.")
    except Exception as e:
        print(f"check_is_fitted failed: {e}")
        
    # 6. Predict
    print("Predicting...")
    preds = model.predict(X)
    print(f"Predictions shape: {preds.shape}")
    print(f"Predictions head: {preds.head()}")
    
    # 7. Check cloning (sklearn requirement)
    print("Cloning...")
    model_clone = clone(model)
    print("Clone successful.")
    
    # 8. Test Singular Matrix / Failure case (empty data or bad data)
    print("Testing failure resilience...")
    model_fail = VECMWrapper(max_lag=100) # Too many lags for data
    try:
        model_fail.fit(X, y)
        print("Fit called on bad configuration.")
        check_is_fitted(model_fail)
        print("check_is_fitted passed on failed model.")
        preds_fail = model_fail.predict(X)
        print(f"Failed model predictions (should be 0s): {preds_fail.head()}")
    except Exception as e:
        print(f"Failure test raised exception: {e}")

if __name__ == "__main__":
    test_vecm_compliance()
