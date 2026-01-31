"""
Cointegration Prior Weighting Module
====================================
Implements Bayesian-inspired weighting for cointegration relationships.
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Union
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class PriorConfig:
    """Configuration for cointegration priors."""
    pair_priors: Dict[str, float]
    evidence_method: str = "logistic"
    logistic_k: float = 10.0
    logistic_p0: float = 0.10
    stability_enabled: bool = True
    rolling_window_months: int = 120
    min_stable_ratio: float = 0.5
    min_weight_threshold: float = 0.3

@dataclass
class CointegrationWeight:
    """Weight calculation result for a cointegration pair."""
    pair_name: str
    prior_weight: float
    evidence_factor: float
    stability_factor: float
    final_weight: float
    include_in_model: bool
    components_breakdown: Dict[str, float]

class CointegrationPriorWeighter:
    """Computes Bayesian-inspired weights for cointegration pairs."""
    DEFAULT_PRIORS = {
        'consumption_income': 0.8, 'fisher_hypothesis': 0.6, 'yields_inflation': 0.6,
        'quantity_theory': 0.5, 'gdp_m2': 0.5, 'income_m2': 0.5,
        'okun_law': 0.4, 'output_employment': 0.4,
        'housing_rates': 0.3, 'permits_rates': 0.3, 'stocks_gdp': 0.2,
    }
    
    def __init__(self, config: Optional[Union[Dict, PriorConfig]] = None):
        if config is None:
            self.config = PriorConfig(pair_priors=self.DEFAULT_PRIORS)
        elif isinstance(config, dict):
            # Map dictionary keys even if they don't exactly match dataclass fields
            priors = config.get('pair_priors', config.get('priors', self.DEFAULT_PRIORS))
            self.config = PriorConfig(
                pair_priors=priors,
                evidence_method=config.get('evidence_method', 'logistic'),
                logistic_k=config.get('logistic_k', 10.0),
                logistic_p0=config.get('logistic_p0', 0.10),
                stability_enabled=config.get('stability_enabled', True),
                rolling_window_months=config.get('rolling_window_months', 120),
                min_stable_ratio=config.get('min_stable_ratio', 0.5),
                min_weight_threshold=config.get('min_weight_threshold', 0.3)
            )
        else:
            self.config = config
        self.weights: Dict[str, CointegrationWeight] = {}
        
    # Class-level cache for rolling cointegration tests
    # Key: (variable1, variable2, start_date, end_date)
    # Value: p-value
    _rolling_test_cache: Dict[Tuple, float] = {}

    def _compute_evidence_factor(self, johansen_pvalue: float, eg_pvalue: float) -> float:
        p = min(johansen_pvalue, eg_pvalue)
        if np.isnan(p): return 0.0
        if self.config.evidence_method == "logistic":
            return 1.0 / (1.0 + np.exp(self.config.logistic_k * (p - self.config.logistic_p0)))
        return max(0, 1 - p / 0.2)

    def _compute_stability_factor(self, series1: pd.Series, series2: pd.Series) -> float:
        if not self.config.stability_enabled: return 1.0
        from statsmodels.tsa.stattools import coint
        
        # Ensure alignment
        data = pd.concat([series1, series2], axis=1).dropna()
        window = self.config.rolling_window_months
        
        if len(data) < window + 24: return 0.5
        
        n_windows = 0
        n_cointegrated = 0
        
        # Step size 12 months
        step = 12
        
        for start in range(0, len(data) - window, step):
            # Define window range
            w_start_idx = data.index[start]
            w_end_idx = data.index[start + window - 1] # Inclusive
            
            # Cache key
            key = (str(series1.name), str(series2.name), w_start_idx.value, w_end_idx.value)
            
            # Check cache
            if key in self._rolling_test_cache:
                p_val = self._rolling_test_cache[key]
            else:
                w_data = data.iloc[start:start + window]
                try:
                    # Run test
                    _, p_val, _ = coint(w_data.iloc[:, 0], w_data.iloc[:, 1])
                    self._rolling_test_cache[key] = p_val
                except:
                    p_val = 1.0 # Fail
                    self._rolling_test_cache[key] = 1.0
            
            n_windows += 1
            if p_val < 0.10: 
                n_cointegrated += 1
                
        if n_windows == 0: return 0.5
        ratio = n_cointegrated / n_windows
        return 0.8 + 0.2 * (ratio - 0.5) / 0.5 if ratio >= 0.5 else 0.8 * ratio / 0.5

    def compute_weight(self, pair_name: str, johansen_pvalue: float, eg_pvalue: float, series1=None, series2=None) -> CointegrationWeight:
        prior = self.config.pair_priors.get(pair_name, 0.5)
        evidence = self._compute_evidence_factor(johansen_pvalue, eg_pvalue)
        stability = self._compute_stability_factor(series1, series2) if series1 is not None else 1.0
        final = (prior ** 0.3) * (evidence ** 0.5) * (stability ** 0.2)
        include = final >= self.config.min_weight_threshold
        result = CointegrationWeight(pair_name, prior, evidence, stability, final, include, {'prior': prior, 'evidence': evidence, 'stability': stability})
        self.weights[pair_name] = result
        return result
