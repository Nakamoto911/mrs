# Technical Appendix B: Model Framework & Validation Protocol

**Document 3 of 4** | **Version 1.0** | **January 27, 2026**

---

## Contents

1. [Asset-Specific Regime Detection](#asset-specific-regime-detection)
2. [Model Tournament Framework](#model-tournament-framework)
3. [Evaluation Metrics & Selection Criteria](#evaluation-metrics--selection-criteria)
4. [ALFRED Validation Protocol](#alfred-validation-protocol)
5. [Dominant Driver Monitoring System](#dominant-driver-monitoring-system)

---

## Asset-Specific Regime Detection

### 1.1 Three-Stage Pipeline

**Stage 1: Historical Regime Labeling**

Statistical Jump Models classify each month as bullish (high return, low vol) or bearish (low return, high vol) for each asset SEPARATELY. Regimes discovered from asset behavior, not imposed from macro conditions.

**Stage 2: Dominant Driver Identification**

Train regime-conditional XGBoost models. Compute SHAP values to identify top 5–10 macro features per asset-regime combination. Example: S&P bullish driven by M2/GDP, bearish by credit spreads.

**Stage 3: Regime Forecasting**

LightGBM classifier predicts 6-month ahead regime probability using current macro state. Provides forward-looking regime classification for portfolio decisions.

### 1.2 Why Asset-Specific vs. Universal Regimes

| Aspect | Universal Regimes | Asset-Specific Regimes |
|---|---|---|
| Assumption | 3–4 regimes affect all assets | Each asset has unique sensitivities |
| Example | Expansion regime → all bullish | Equities bullish, bonds neutral |
| Academic Support | Traditional approach | 2024 research shows 15–30% better IC |

---

## Model Tournament Framework

### 2.1 Competing Model Families

**Linear Models:**
- Ridge Regression — L2 penalty, interpretable coefficients (Benchmark)
- Lasso — L1 penalty with very weak regularization ($\alpha=0.0001$) to avoid zero-signal collapse in sparse regimes. Requires `max_iter=100,000` for convergence.
- Elastic Net — L1+L2 ($\alpha=0.0001$, `l1_ratio=0.5`), handles correlated features. Requires `max_iter=100,000`.
- VECM — If cointegration detected, captures mean-reversion.

**Tree-Based Ensembles:**
- Random Forest — Robust to overfitting, feature importance via Gini. Configured with `n_jobs=1` to prevent deadlock, `n_estimators=20`, and `max_depth=4` for regularization.
- XGBoost — State-of-the-art accuracy, handles non-linearities.
- LightGBM — Faster, better with large feature sets.
- CatBoost — Handles categorical features (regime indicators).

**Neural Networks:**
- Feedforward MLP — Multi-layer perceptron with dropout. Requires robust scaling and imputation to identical standards as linear models.
- LSTM — Temporal dependencies, 12–24 month sequences.
- Temporal Convolutional Network — Alternative to LSTM.

### 2.2 Hyperparameter Optimization

Optuna Bayesian optimization: 100 trials per model, 5-fold time-series CV

- **Linear:** Regularization strength α, mixing parameter (Elastic Net)
- **Tree-based:** Learning rate, max depth, min samples/leaf, n_estimators
- **Neural Nets:** Layer sizes, dropout rate, learning rate, batch size

### 2.4 Ensemble Strategy: Average Top 5

To reduce model-specific variance and improve out-of-sample stability, the system implements a post-hoc **Variance Reduction** strategy:

1. **Prediction Persistence:** Every model in the tournament persists its Out-of-Sample (OOS) predictions to `experiments/predictions/`.
2. **IC Ranking:** For each asset, all models are ranked by their mean Information Coefficient (IC).
3. **Top 5 Selection:** The top $N$ models (default: 5) are selected.
4. **Signal Averaging:** An "Ensemble" model is created by computing the row-wise mean of the signals from the selected models.
5. **Averaging Logic:**
   - **Alignment:** Perform an Inner Join on the Date index to ensure only overlapping valid periods are included.
   - **Weighting:** Equal weighting is applied to all selected models.
6. **Ensemble Metrics:** The averaged signal is evaluated as a distinct model and added to the leaderboard as `Ensemble_Top5`.

### 2.3 Robust Training Layer

To maintain estimation integrity across the high-dimensional (744-feature) space, a standardized preprocessing layer is enforced within the training pipeline for **all model families** (Linear, Tree, Neural):

**Data Integrity Protocols:**
- **Sparse Feature Imputation:** **Strict Rolling Imputation** (Expanding Median) applied to features with < 20% missingness. By calculating medians on an expanding window and shifting by 1 month, we ensure that row $t$ is imputed using only data from $t-1$, eliminating look-ahead bias. If no history is available for a feature at $t$, it is filled with **0.0** (neutral prior) to ensure informational PIT validity. This applies to all model families (Linear, Tree, Neural). Series exceeding the missingness threshold are pruned PIT (Point-In-Time) to prevent signal dilution.
- **Variance Filtering:** Automatic detection and exclusion of zero-variance or near-constant features that degrade `StandardScaler` performance.
- **Degrees of Freedom Guard:** Hard requirement of $N > K + 20$ (samples > features + 20) for valid estimation. If violated in any CV fold, the system enforces **Automated PIT Feature Pruning**, ranking and keeping only the top $(N-20)$ features based on maximum data density (lowest NaNs) and secondarily by cross-sectional variance.

---

## Evaluation Metrics & Selection Criteria

### 3.1 Time-Series Cross-Validation

- **Expanding window:** Minimum 120 months, grows each fold
- **Validation window:** 12 months rolling
- **Step size:** 6 months (semi-annual rebalancing)
- **Total folds:** 10+ depending on data history
- **Metrics computed:** Per fold, averaged for stability

### 3.2 Performance Metrics

| Metric | Definition | Primary For | Target |
|---|---|---|---|
| IC | Spearman rank correlation | Return forecasts | > 0.20 |
| RMSE | Root mean squared error | Volatility forecasts | Minimize |
| MAE | Mean absolute error | Robustness check | Minimize |
| R² (OOS) | Out-of-sample R-squared | Variance explained | > 0.10 |
| Hit Rate | % correct directional forecasts | Regime detection | > 60% |
| Directional MAE | MAE when direction correct | Quality given correct sign | Lower |

### 3.3 Model Selection Criterion

- **For return forecasts:** Information Coefficient (IC) is primary — measures rank ordering
- **For volatility forecasts:** RMSE is primary — magnitude accuracy matters
- **Tie-break:** Prefer simpler models (VECM > XGBoost > LSTM) for interpretability

---

## ALFRED Validation Protocol

### 4.1 Validation Workflow

1. **For each semi-annual date (2000–2024):** 48 rebalancing periods
2. **Determine data availability:** Assume 1-month publication lag
   - June 30 rebalancing → Uses data through April 30
3. **Download ALFRED vintage:** `vintage_dates='2015-05-31'`
4. **Apply feature engineering** to real-time vintage
5. **Load frozen Discovery Phase models** (hyperparameters fixed)
6. **Generate forecasts**, store for evaluation
7. **Compute metrics** on real-time data
8. **Compare to Discovery Phase** (revised data) results

### 4.2 Revision Risk Analysis

- **Revision Risk (IC)** = (IC_revised − IC_realtime) / IC_revised
- **Feature Stability Score** = Correlation(SHAP_revised, SHAP_realtime)

**Deployment Criteria:**
- IC(realtime) > 0.15
- Revision Risk < 30%
- Feature Stability > 0.70

### 4.3 Expected Results

| Asset | Model | IC_Revised | IC_Realtime | Rev_Risk | Deploy? |
|---|---|---|---|---|---|
| S&P 500 | XGBoost | 0.28 | 0.22 | 21% | ✓ |
| 10Y Bond | Elastic Net | 0.32 | 0.29 | 9% | ✓ |
| Gold | Ridge | 0.16 | 0.13 | 19% | ✓ |

---

## Dominant Driver Monitoring System

### 5.1 SHAP-Based Feature Importance

For best model per asset, compute SHAP values across all predictions. Rank features by mean absolute SHAP value. Top 5–10 = dominant drivers.

### 5.2 Regime-Conditional Analysis

Compute SHAP rankings separately for:

- **Bullish periods:** Which features matter when regime is favorable
- **Bearish periods:** Which features matter during stress
- **Overall:** Combined across all periods

### 5.3 Monitoring Sheet Template

**Example: S&P 500 Dominant Drivers**

| Rank | Feature | SHAP | Current Pct | Signal |
|---|---|---|---|---|
| 1 | M2/GDP (Liquidity) | 0.42 | 85th | ↑ Bullish |
| 2 | Real 10Y Yield | 0.38 | 50th | → Neutral |
| 3 | Credit Spread | 0.31 | 15th | ↑ Bullish |
| 4 | VIX | 0.24 | 45th | → Neutral |
| 5 | IP Growth | 0.19 | 30th | ↓ Bearish |

**Interpretation:** Overall Regime = **Cautiously Bullish**. Liquidity and credit supportive, but monitor IP for deterioration.

---

*End of Technical Appendix B*
