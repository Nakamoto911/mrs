# Asset-Specific Macro Regime Detection System

## Executive Overview & Research Strategy

**Document 1 of 4** | **Version 1.0** | **January 27, 2026**

---

## Document Suite Overview

This research methodology is organized into four interconnected documents:

- **Document 1: Executive Overview** (this document)  
  Strategic objectives, core innovations, expected outcomes, and high-level methodology

- **Document 2: Technical Appendix A — Data & Feature Engineering**  
  FRED-MD data sources, historical extension to 1959, 7-step feature engineering pipeline, validated cointegration (Johansen/Engle-Granger), hierarchical clustering, ALFRED vintages.

- **Document 3: Technical Appendix B — Model Framework & Validation**  
  Asset-specific regime detection, model tournament protocol, SHAP analysis, ALFRED validation, revision risk metrics, performance benchmarks

- **Document 4: Technical Appendix C — Implementation Guide**  
  Streamlit UI specifications, modular execution modes, computational timing, code examples, deployment workflow, deliverables checklist

---

## Executive Summary

This research system discovers which macroeconomic variables drive asset returns and volatility across different market regimes. Unlike traditional approaches assuming universal economic regimes affect all assets identically, this system detects **asset-specific regimes** based on each asset's unique macro sensitivities.

### Core Objective

Build an automated model discovery platform identifying, for each asset (US Equities, Bonds, Gold):

- Top 5–10 macroeconomic variables (dominant drivers) predicting future returns and volatility
- How these drivers differ between bullish and bearish regimes
### 2.3 Robust Training Layer

To maintain estimation integrity across the high-dimensional (~750-feature) space, a standardized preprocessing layer is enforced within the training pipeline for **all model families** (Linear, Tree, Neural):
- Which feature engineering approaches (levels, ratios, cointegration, quintiles) are most predictive
- Which model families (linear, tree-based, neural networks) perform best
- How robust relationships are to data revisions (real-time vs. revised data)

### System Output: Real-Time Monitoring

**Example: S&P 500 Current Regime — Cautiously Bullish**

| Rank | Dominant Macro Driver | SHAP Value | Current Signal |
|------|---|---|---|
| 1 | M2/GDP Ratio (Liquidity) | 0.42 | ↑ Bullish (85th percentile) |
| 2 | Real 10Y Yield | 0.38 | → Neutral (50th percentile) |
| 3 | Credit Spread (BAA-10Y) | 0.31 | ↑ Bullish (15th pct, tight) |
| 4 | VIX Level | 0.24 | → Neutral (45th percentile) |
| 5 | Industrial Production Growth | 0.19 | ↓ Bearish (30th percentile) |

**Interpretation:** Liquidity and credit conditions supportive, but weakening industrial activity is a concern. Monitor IP closely for deterioration.

---

## Methodological Innovations

### 1. Asset-Specific Regime Detection

Each asset gets independent bullish/bearish classification via Statistical Jump Models on return/volatility patterns. Regimes discovered bottom-up from asset behavior, not imposed top-down from GDP/inflation. Academic research (2024) shows asset-specific approaches outperform universal models by 15–30%.

### 2. Hierarchical Clustering Eliminates Substitution Instability

Groups correlated features (GDP Growth, IP Growth, Income Growth all measure "activity") at 0.40 similarity threshold (Super-Clustering). Selects ONE representative per cluster using Medoid Selection. Ensures dominant drivers represent broad economic forces, not statistical variations. Result: Stable SHAP values, clear interpretation (top 10 = 10 distinct economic themes).

### 3. Two-Phase Data Strategy: Revised → Real-Time

- **DISCOVERY** on lagged revised FRED-MD: Find true relationships without measurement noise
- **VALIDATION** on ALFRED real-time vintages: Test with data actually available to investors
- Compute Revision Risk (typical IC degradation 15–25%)
- Deploy only models with IC(real-time) exceeding asset-specific "Acceptable" thresholds and Revision Risk < 30%

### 4. Regime-Level Quintile Features

Beyond growth rates, include historical quintile indicators. GS10 declining from 5% (Q5) vs. 2% (Q1) has different implications. Captures regime-dependent dynamics that change features alone miss.

### 5. 12-Month Strategic Forecast Horizon

For horizons with annual rebalancing: 12M balances macro signal extraction with forecast reliability and provides the industry standard for tactical benchmarks.

---

## Expected Performance Benchmarks (Literature-Calibrated)

| Asset Class | Excellent (Top Decile) | Good (Meaningful) | Acceptable (Detectable) | Deployment criteria |
|---|---|---|---|---|
| **Equities (SPX)** | IC > 0.12 | IC > 0.08 | IC > 0.05 | IC > 0.05, Risk < 30% |
| **Bonds (BOND)** | IC > 0.18 | IC > 0.12 | IC > 0.08 | IC > 0.08, Risk < 30% |
| **Commodities (GOLD)** | IC > 0.10 | IC > 0.06 | IC > 0.04 | IC > 0.04, Risk < 30% |

**Notes:** IC (Information Coefficient) measures rank-ordering predictive power. These thresholds are calibrated to empirical benchmarks (Welch & Goyal, 2008; Campbell & Thompson, 2008) for 12-month horizons. Bonds achieve higher ICs due to term structure persistence; Commodities are the most noise-dominated.

---

## System Architecture: 5-Stage Research Platform

### Stage 1: Data Acquisition & Feature Engineering

Download FRED-MD (128 variables) + ETF prices via independent `data_acquisition.py` script. Save to human-readable CSVs. **Categories 6 (Interest and Exchange Rates) & 8 (Stock Market) are strictly excluded.** Extend asset history to 1959 using spliced FRED-MD proxies (`S&P 500`, `GS10`, `PPICMM`). Generate ~750 features through 7-step pipeline. Apply hierarchical clustering @ 0.40 threshold. Result: ~250–300 cluster representatives. **Time:** 5–10 min for data update, 15–20 min for feature engineering. *(See Appendix A for complete details)*

### Stage 2: Asset-Specific Regime Detection

Statistical Jump Models label historical bullish/bearish periods per asset. Train regime-conditional XGBoost models. Compute SHAP values to identify top 5–10 dominant drivers per asset-regime. LightGBM forecasts 6-month ahead regime probabilities. *(See Appendix B Section 1)*

### Stage 3: Model Tournament & Selection

10+ models compete: Ridge, Lasso, Elastic Net, VECM, Random Forest, XGBoost, LightGBM, LSTM V2. Time-series CV with expanding windows. Optuna hyperparameter optimization (100 trials). Select best via IC for returns, RMSE for volatility. **Time:** 6–9 hours full tournament. *(See Appendix B Section 2–3)*

### Stage 4: ALFRED Real-Time Validation

Re-run selected models on ALFRED vintages (2000–2024, 48 semi-annual dates). Measure revision risk. Compare feature stability. Deploy only if IC(real-time) > 0.15 AND Revision Risk < 30%. **Time:** ~20 hours total, parallelizable to 4–5 hours. *(See Appendix B Section 4)*

### Stage 5: Production Deployment & Monitoring

Streamlit dashboard (5 pages): Experiment config, model comparison, feature importance, dominant drivers monitoring, data inspector. Includes "Update Data" feature to refresh CSVs. Monthly updates: 30–40 min. Quarterly retraining: 3–4 hours. *(See Appendix C)*

---

## System Deliverables

- **Automated model tournament scripts** — Train all models, compute metrics, rank performance
- **Feature engineering pipeline** — ~250–300 cluster representatives from FRED-MD
- **Dominant driver reports** — Top 5–10 macro features per asset with SHAP values and current signals
- **Interactive Streamlit dashboard** — Model comparison, feature importance, regime monitoring
- **ALFRED validation report** — Revision risk analysis, feature stability, deployment recommendations
- **Production model registry** — Saved best models with versioning for reproducible forecasts
- **Real-time monitoring sheets** — Current regime classification based on dominant driver levels

---

## Technical Details: See Appendices

This executive overview provides strategic context. For implementation details, consult:

### Appendix A: Data & Feature Engineering
FRED-MD structure, historical extension methodology, 7-step pipeline (preserve stationary levels, FRED-MD transforms, macro ratios, quintile features, validated cointegration, momentum, cross-asset), hierarchical clustering algorithm, ALFRED vintage construction.

### Appendix B: Model Framework & Validation
Statistical Jump Model regime labeling, SHAP dominant driver identification, model tournament specifications (hyperparameters, CV protocol), ALFRED validation workflow, revision risk metrics, performance comparison tables, feature stability analysis

### Appendix C: Implementation Guide
Streamlit UI page-by-page specifications, modular execution modes (full/single-asset/single-model), result caching strategy, CLI interface design, parallel execution setup, computational timing tables, deployment workflow, monitoring dashboard examples

---

*End of Executive Overview*
