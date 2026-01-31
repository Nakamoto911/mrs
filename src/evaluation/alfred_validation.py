import numpy as np
import pandas as pd
import json
import joblib
import logging
from pathlib import Path
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)

class NBERRecessionDates:
    """NBER recession date manager."""
    OFFICIAL_DATES = [
        ("1980-01-01", "1980-07-31"), ("1981-07-01", "1982-11-30"),
        ("1990-07-01", "1991-03-31"), ("2001-03-01", "2001-11-30"),
        ("2007-12-01", "2009-06-30"), ("2020-02-01", "2020-04-30"),
    ]
    def __init__(self, custom_dates=None):
        dates = custom_dates or self.OFFICIAL_DATES
        self.recession_periods = [(pd.Timestamp(s), pd.Timestamp(e)) for s, e in dates]
    def is_recession(self, date):
        for s, e in self.recession_periods:
            if s <= date <= e: return True
        return False
    def get_regime_labels(self, dates):
        return pd.Series(['recession' if self.is_recession(d) else 'expansion' for d in dates], index=dates)

from models.ensemble import EnsembleModel

class ALFREDValidator:
    """Core engine for running historical simulation loops using ALFRED vintages."""
    def __init__(self, model_path, validation_dates, vintages_dir="data/raw/vintages", transform_codes_path="data/raw/fred_md_transforms.csv"):
        # ... (keep existing init code) ...
        model_p = Path(model_path)
        if model_p.suffix == '.json':
            with open(model_p, 'r') as f: manifest = json.load(f)
            constituent_names = manifest['models']
            loaded_models = []
            models_dir = model_p.parent
            asset = model_p.stem.split('_')[0]
            for m_name in constituent_names:
                c_path = models_dir / f"{asset}_{m_name}.joblib"
                if not c_path.exists(): 
                   # Fallback to .pkl
                   c_path = models_dir / f"{asset}_{m_name}.pkl"
                if not c_path.exists(): raise FileNotFoundError(f"Ensemble constituent missing: {c_path}")
                loaded_models.append(joblib.load(c_path))
            self.model = EnsembleModel(loaded_models)
        else:
            self.model = joblib.load(model_path)
        
        if hasattr(self.model, 'feature_names_in_'): self.features_needed = self.model.feature_names_in_
        elif hasattr(self.model, 'feature_names'): self.features_needed = self.model.feature_names
        elif hasattr(self.model, 'features'): self.features_needed = self.model.features
        elif hasattr(self.model, 'steps'):
            try:
                if hasattr(self.model.steps[0][1], 'feature_names_in_'): self.features_needed = self.model.steps[0][1].feature_names_in_
                elif hasattr(self.model.steps[-1][1], 'feature_names_in_'): self.features_needed = self.model.steps[-1][1].feature_names_in_
                else: raise AttributeError("Could not find feature_names_in_ in pipeline steps")
            except Exception as e: raise AttributeError(f"Failed to extract features from Pipeline: {e}")
        else: raise AttributeError("Model features missing.")
        
        if hasattr(self.features_needed, 'tolist'): self.features_needed = self.features_needed.tolist()
        self.loader = ALFREDVintageLoader(vintages_dir)
        self.transform_codes = {}
        if Path(transform_codes_path).exists():
            t_df = pd.read_csv(transform_codes_path)
            for col in t_df.columns:
                try: self.transform_codes[col] = int(t_df.iloc[0][col])
                except: self.transform_codes[col] = 1
        elif validation_dates: self.transform_codes = self.loader.get_transform_codes(validation_dates[0])
        self.pipeline = FrozenFeaturePipeline(self.features_needed, self.transform_codes)
        self.validation_dates = sorted(validation_dates)
        self.nber = NBERRecessionDates()

    def run_validation(self, true_returns: pd.Series) -> pd.DataFrame:
        results = []
        for t in self.validation_dates:
            try:
                vintage_df = self.loader.load_vintage(t)
                X_t = self.pipeline.transform_vintage(vintage_df)
                X_latest = X_t.tail(1)
                prediction_result = self.model.predict(X_latest)
                forecast = prediction_result.iloc[0] if hasattr(prediction_result, 'iloc') else prediction_result[0]
                results.append({'Date': t, 'Forecast_RealTime': forecast, 'Realized_Return': true_returns.loc[t] if t in true_returns.index else np.nan})
            except Exception as e: logger.error(f"Error {t}: {e}")
        return pd.DataFrame(results)

    def compute_metrics(self, validation_results: pd.DataFrame, revised_ic: float) -> Dict:
        clean = validation_results.dropna(subset=['Realized_Return'])
        if len(clean) < 2: return {'RealTime_IC': 0.0, 'Revision_Risk': 1.0, 'Status': 'Insufficient Data'}
        
        rt_ic, _ = spearmanr(clean['Forecast_RealTime'], clean['Realized_Return'])
        rev_risk = (revised_ic - rt_ic) / revised_ic if abs(revised_ic) > 1e-6 else 0.0
        
        # Regime-Conditional Analysis
        labels = self.nber.get_regime_labels(clean['Date'])
        recessions = clean[labels == 'recession']
        expansions = clean[labels == 'expansion']
        
        rec_ic = spearmanr(recessions['Forecast_RealTime'], recessions['Realized_Return'])[0] if len(recessions) >= 5 else np.nan
        exp_ic = spearmanr(expansions['Forecast_RealTime'], expansions['Realized_Return'])[0] if len(expansions) >= 5 else np.nan
        
        return {
            'RealTime_IC': rt_ic, 'Revised_IC': revised_ic, 'Revision_Risk': rev_risk,
            'Recession_IC': rec_ic, 'Expansion_IC': exp_ic, 'Sample_Size': len(clean)
        }
