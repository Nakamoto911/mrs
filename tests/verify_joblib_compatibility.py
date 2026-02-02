import joblib
import pandas as pd
import numpy as np
from joblib import Parallel, delayed
import time
import logging

# Setup basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def mock_train_model(model_type, asset, seed):
    """
    Simulate training a model to check for pickling/multiprocessing issues.
    """
    logger.info(f"Starting {model_type} for {asset} (seed={seed})")
    
    # Simulate work
    time.sleep(0.5)
    
    # Return a result that might be problematic to pickle if not careful
    result = {
        'model_type': model_type,
        'asset': asset,
        'status': 'success',
        'data': np.random.rand(10, 10)
    }
    
    logger.info(f"Finished {model_type} for {asset}")
    return result

def real_model_check():
    """
    Try to instantiate and train actual lightweight versions of the models 
    if libraries are available.
    """
    import xgboost as xgb
    from sklearn.linear_model import Ridge
    # torch import inside to catch availability
    try:
        import torch
        has_torch = True
    except ImportError:
        has_torch = False
        
    def train_xgboost():
        X = np.random.rand(100, 10)
        y = np.random.rand(100)
        model = xgb.XGBRegressor(n_estimators=10, n_jobs=1) # force single thread per process
        model.fit(X, y)
        return "XGBoost Done"

    def train_sklearn():
        X = np.random.rand(100, 10)
        y = np.random.rand(100)
        model = Ridge()
        model.fit(X, y)
        return "Sklearn Done"
        
    def train_torch():
        if not has_torch: return "Torch Missing"
        # Simple linear model
        model = torch.nn.Linear(10, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
        data = torch.randn(10, 10)
        target = torch.randn(10, 1)
        
        # Train loop
        for _ in range(5):
            optimizer.zero_grad()
            out = model(data)
            loss = torch.nn.functional.mse_loss(out, target)
            loss.backward()
            optimizer.step()
        return "Torch Done"

    tasks = [
        ('xgboost', train_xgboost),
        ('sklearn', train_sklearn),
        ('torch', train_torch)
    ]
    
    # Run in parallel
    print("Running actual model tests in parallel...")
    results = Parallel(n_jobs=2)(
        delayed(func)() for name, func in tasks
    )
    print("Results:", results)

if __name__ == "__main__":
    print("Testing Joblib Parallelism...")
    
    # 1. Basic Serializable Check
    print("\n--- Basic Serialization Check ---")
    assets = ['SPX', 'BOND', 'GOLD']
    models = ['m1', 'm2', 'm3']
    
    try:
        results = Parallel(n_jobs=2)(
            delayed(mock_train_model)(m, a, i) 
            for i, (m, a) in enumerate(zip(models, assets))
        )
        print("Basic check passed.")
    except Exception as e:
        print(f"Basic check failed: {e}")
        
    # 2. Real Model Check
    print("\n--- Real Model Library Check ---")
    try:
        real_model_check()
    except Exception as e:
        print(f"Real model check failed: {e}")
        import traceback
        traceback.print_exc()
