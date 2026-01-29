"""
ALFRED Real-Time Validation
===========================
Executes historical simulations using point-in-time vintage data.
"""

import pandas as pd
import numpy as np
import joblib
import logging
from typing import List, Dict, Optional
from scipy.stats import spearmanr
from pathlib import Path
import json

from src.preprocessing.alfred_loader import ALFREDVintageLoader
from src.feature_engineering.frozen_pipeline import FrozenFeaturePipeline

logger = logging.getLogger(__name__)

class EnsembleWrapper:
    """
    Wraps multiple models to provide a single 'predict' interface 
    for the validation script.
    """
    def __init__(self, models):
        self.models = models
        
        # Aggregate features from all constituent models
        all_features = set()
        for m in self.models:
            feats = []
            if hasattr(m, 'feature_names_in_'):
                feats = m.feature_names_in_
            elif hasattr(m, 'feature_names'):
                feats = m.feature_names
            elif hasattr(m, 'features'):
                feats = m.features
            
            if hasattr(feats, 'tolist'):
                all_features.update(feats.tolist())
            else:
                all_features.update(feats)
        
        self.feature_names_in_ = sorted(list(all_features))

    def predict(self, X):
        """
        Runs prediction on all constituent models and returns the average.
        """
        preds = []
        for model in self.models:
            # Handle both Series and Array returns
            p = model.predict(X)
            if hasattr(p, 'values'):
                p = p.values
            preds.append(p)
        
        # Average
        return np.mean(preds, axis=0)

class ALFREDValidator:
    """
    Core engine for running historical simulation loops using ALFRED vintages.
    """
    def __init__(self, 
                 model_path: str, 
                 validation_dates: List[pd.Timestamp],
                 vintages_dir: str = "data/raw/vintages",
                 transform_codes_path: str = "data/raw/fred_md_transforms.csv"):
        """
        Args:
            model_path: Path to the .pkl or .json file of the trained model.
            validation_dates: List of semi-annual dates for simulation.
            vintages_dir: Directory containing vintage CSVs.
            transform_codes_path: Path to FRED-MD transformation codes.
        """
        model_p = Path(model_path)
        if model_p.suffix == '.json':
            logger.info(f"Detected Ensemble Manifest: {model_p.name}")
            with open(model_p, 'r') as f:
                manifest = json.load(f)
            
            constituent_names = manifest['models']
            loaded_models = []
            
            # Assumes models are in the same directory or standard models directory
            models_dir = model_p.parent
            
            # Parse asset from filename (e.g., GOLD_Ensemble_Top5.json)
            asset = model_p.stem.split('_')[0]
            
            for m_name in constituent_names:
                c_path = models_dir / f"{asset}_{m_name}.pkl"
                if not c_path.exists():
                     raise FileNotFoundError(f"Ensemble constituent missing: {c_path}")
                loaded_models.append(joblib.load(c_path))
                
            self.model = EnsembleWrapper(loaded_models)
            logger.info(f"Successfully loaded Ensemble with {len(loaded_models)} models")
        else:
            self.model = joblib.load(model_path)
        
        # Identify features needed by the model
        if hasattr(self.model, 'feature_names_in_'):
            self.features_needed = self.model.feature_names_in_
        elif hasattr(self.model, 'feature_names'):
            self.features_needed = self.model.feature_names
        elif hasattr(self.model, 'features'):
            self.features_needed = self.model.features
        else:
            raise AttributeError("Model does not have 'feature_names_in_', 'feature_names', or 'features' attribute.")
            
        # Convert to list if it's a numpy array
        if hasattr(self.features_needed, 'tolist'):
            self.features_needed = self.features_needed.tolist()
            
        self.loader = ALFREDVintageLoader(vintages_dir)
        
        # Load transform codes
        self.transform_codes = {}
        if Path(transform_codes_path).exists():
            t_df = pd.read_csv(transform_codes_path)
            for col in t_df.columns:
                try:
                    self.transform_codes[col] = int(t_df.iloc[0][col])
                except:
                    self.transform_codes[col] = 1
        elif validation_dates:
            # Try to get from the first vintage
            self.transform_codes = self.loader.get_transform_codes(validation_dates[0])
                    
        self.pipeline = FrozenFeaturePipeline(self.features_needed, self.transform_codes)
        self.validation_dates = sorted(validation_dates)

    def run_validation(self, true_returns: pd.Series) -> pd.DataFrame:
        """
        Execute the historical simulation loop.
        
        Args:
            true_returns: Series of realized returns (target variable) to compare against.
            
        Returns:
            DataFrame with [Date, Forecast_RealTime, Realized_Later]
        """
        results = []
        
        for t in self.validation_dates:
            logger.info(f"Simulating point-in-time for: {t.strftime('%Y-%m-%d')}")
            
            try:
                # 1. Load Vintage(t)
                vintage_df = self.loader.load_vintage(t)
                
                # 2. Generate features X_t (using FrozenPipeline)
                X_t = self.pipeline.transform_vintage(vintage_df)
                
                # We need the most recent observation in the vintage for prediction
                # Usually the last row in X_t
                X_latest = X_t.tail(1)
                
                # 3. Predict Y_{t+h} (Forecast)
                prediction_result = self.model.predict(X_latest)
                if hasattr(prediction_result, 'iloc'):
                    forecast = prediction_result.iloc[0]
                else:
                    forecast = prediction_result[0]
                
                # 4. Get realized return (from true_returns)
                realized = np.nan
                if t in true_returns.index:
                    realized = true_returns.loc[t]
                
                results.append({
                    'Date': t,
                    'Forecast_RealTime': forecast,
                    'Realized_Return': realized
                })
                
            except Exception as e:
                logger.error(f"Error validating date {t}: {e}")
                
        return pd.DataFrame(results)

    def compute_metrics(self, validation_results: pd.DataFrame, revised_ic: float) -> Dict:
        """
        Compute Revision Risk metrics.
        
        Args:
            validation_results: Output from run_validation.
            revised_ic: The IC achieved during the Discovery Phase (on revised data).
            
        Returns:
            Dictionary with Real-Time IC, Revision Risk, etc.
        """
        # Remove NaNs
        clean_results = validation_results.dropna(subset=['Realized_Return'])
        
        if len(clean_results) < 2:
            return {
                'RealTime_IC': 0.0,
                'Revision_Risk': 1.0,
                'Status': 'Insufficient Data'
            }
            
        # 1. Real-Time IC
        rt_ic, _ = spearmanr(clean_results['Forecast_RealTime'], clean_results['Realized_Return'])
        
        # 2. Revision Risk
        # (IC_Revised - IC_RealTime) / IC_Revised
        rev_risk = 0.0
        if abs(revised_ic) > 1e-6:
            rev_risk = (revised_ic - rt_ic) / revised_ic
            
        return {
            'RealTime_IC': rt_ic,
            'Revised_IC': revised_ic,
            'Revision_Risk': rev_risk,
            'Sample_Size': len(clean_results)
        }
