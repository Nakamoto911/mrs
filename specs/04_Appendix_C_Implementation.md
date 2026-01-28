# Technical Appendix C: Implementation & Deployment Guide

**Document 4 of 4** | **Version 1.0** | **January 27, 2026**

---

## Contents

1. [Modular Execution Framework](#modular-execution-framework)
2. [Streamlit User Interface Specifications](#streamlit-user-interface-specifications)
3. [Code Organization & Architecture](#code-organization--architecture)
4. [Computational Timing & Optimization](#computational-timing--optimization)
5. [Deployment Workflow & Checklist](#deployment-workflow--checklist)
6. [Final Deliverables Checklist](#final-deliverables-checklist)

---

## Modular Execution Framework

### 1.1 Execution Modes

| Mode | Command Example | Time (Sequential) | Use Case |
|---|---|---|---|
| Full Tournament | `--assets all --models all` | 6–9 hours | Initial discovery |
| Single Asset | `--asset SPX` | 2–3 hours | Asset-specific iteration |
| Single Model | `--model xgboost` | 1.5–2.5 hours | Model-specific tuning |
| Single Asset-Model | `--asset SPX --model lstm` | 20–40 min | Quick experiments |
| Features Only | `--features-only` | 1.5–2 hours | Debug pipeline |
| Evaluation Only | `--eval-only` | 15–30 min | Re-compute metrics |

### 1.2 Result Caching Strategy

- **Feature matrices:** Parquet files (`experiments/features/YYYYMMDD.parquet`)
- **Trained models:** Joblib/pickle (`experiments/models/SPX_xgboost_YYYYMMDD.pkl`)
- **CV results:** JSON (`experiments/cv_results/SPX_xgboost_cv.json`)
- **SHAP values:** NumPy arrays (`experiments/shap/SPX_xgboost_shap.npy`)
- **Hyperparameter history:** Optuna SQLite database
- **Regime labels:** CSV (`experiments/regimes/SPX_regime_labels.csv`)

### 1.3 Parallel Execution

- **Asset-level:** Run 3 assets on different CPU cores (reduces wall time 60%)
- **Model-level:** Train multiple models in parallel within asset
- **CV-fold:** joblib/multiprocessing for cross-validation
- **Hyperparameter search:** Optuna supports concurrent trials

**Example:** 16-core machine → 3 assets × 3 cores each, 7 cores for hyperparam search

---

## Streamlit User Interface Specifications

### 2.1 Why Streamlit

| Framework | Pros | Cons | Verdict |
|---|---|---|---|
| Streamlit | Pure Python, rapid prototyping, auto-reload | Less customization | ✓ **RECOMMENDED** |
| Dash | More control, production-ready | Steeper learning curve | Alternative |
| Gradio | Great for model demos | Too focused on inference | Not ideal |
| Flask/React | Full control, scalable | Requires frontend dev | Overkill |

### 2.2 Five-Page Application

**Page 1: Experiment Configuration**

Multi-select assets, checkboxes for models, radio buttons for target variable, sliders for hyperparameter budget, "Launch Experiment" button with progress bar

**Page 2: Model Comparison Dashboard**

Dropdown to select asset, performance metrics table (IC, RMSE, R², Hit Rate), heatmap (models × metrics color-coded), time-series of OOS performance, forecast vs. realized scatter plot, export to Excel

**Page 3: Feature Importance Explorer**

Top 20 features bar chart (SHAP values), tabs for Overall/Bullish/Bearish regimes, sortable table of all features, time-series feature stability plot, waterfall chart for specific predictions, export CSV

**Page 4: Dominant Drivers Monitoring**

Real-time monitoring sheet (rank, feature, SHAP, current percentile, signal), alert indicators for extreme ranges (>95th or <5th percentile), regime summary (Bullish/Bearish/Neutral), historical regime timeline

**Page 5: Data & Feature Inspector**

Upload custom CSV option, feature matrix preview (first/last 10 rows), correlation heatmap, missing data report, stationarity test results, cointegration report with β coefficients

---

## Code Organization & Architecture

Recommended project structure:

```
macro_regime_system/
├── data/
│   ├── fred_md/                 # FRED-MD downloads
│   ├── asset_prices/            # ETF prices
│   ├── alfred_vintages/         # Real-time data
│   └── processed/               # Cached features
├── src/
│   ├── preprocessing/
│   │   ├── data_loader.py
│   │   ├── stationarity.py
│   │   └── transformations.py
│   ├── feature_engineering/
│   │   ├── ratios.py
│   │   ├── quintiles.py
│   │   ├── cointegration.py
│   │   ├── momentum.py
│   │   └── hierarchical_clustering.py
│   ├── models/
│   │   ├── linear_models.py
│   │   ├── tree_models.py
│   │   └── neural_nets.py
│   ├── evaluation/
│   │   ├── cross_validation.py
│   │   ├── metrics.py
│   │   └── shap_analysis.py
│   └── visualization/
│       ├── plotting.py
│       └── dashboards.py
├── streamlit_app/
│   ├── app.py
│   ├── pages/
│   │   ├── 1_Experiment_Config.py
│   │   ├── 2_Model_Comparison.py
│   │   ├── 3_Feature_Importance.py
│   │   ├── 4_Dominant_Drivers.py
│   │   └── 5_Data_Inspector.py
│   └── utils/
│       ├── data_loader.py
│       └── backend.py
├── experiments/
│   ├── features/
│   ├── models/
│   ├── cv_results/
│   └── shap/
├── configs/
│   └── experiment_config.yaml
└── run_tournament.py
```

---

## Computational Timing & Optimization

| Component | Per Asset | × 3 Assets | Parallelized |
|---|---|---|---|
| Linear Models | 10–15 min | 30–45 min | 15–20 min |
| Tree Models | 30–45 min | 90–135 min | 35–50 min |
| Neural Networks | 45–60 min | 135–180 min | 50–70 min |
| SHAP Computation | 20–30 min | 60–90 min | 25–35 min |
| Regime Analysis | 15–20 min | 45–60 min | 20–25 min |
| Report Generation | 5–10 min | 15–30 min | 10–15 min |
| **FULL TOURNAMENT** | **125–180 min** | **375–540 min** | **155–215 min** |

**Optimization Strategies:**

- **GPU acceleration:** LSTM training drops from 60 min to 15 min per asset
- **Caching:** Load pre-computed features, avoid re-transformation
- **Incremental learning:** Monthly updates use existing models, quarterly full retraining
- **Smart hyperparameter init:** Use previous best as Optuna starting point

---

## Deployment Workflow & Checklist

### 5.1 Initial Setup (Week 1)

- [ ] Install dependencies: pandas, numpy, scikit-learn, xgboost, shap, streamlit
- [ ] Download FRED-MD current vintage
- [ ] Extend asset returns to 1959
- [ ] Run feature engineering pipeline (1.5–2h)
- [ ] Run full model tournament (6–9h)
- [ ] Generate initial reports

### 5.2 ALFRED Validation (Week 2)

- [ ] Download 48 ALFRED vintages (2000–2024)
- [ ] Run validation workflow (~20h, parallelizable to 4–5h)
- [ ] Compute revision risk metrics
- [ ] Verify deployment criteria met (IC > 0.15, Risk < 30%)
- [ ] Document feature stability

### 5.3 Production Deployment (Week 3)

- [ ] Launch Streamlit app: `streamlit run app.py`
- [ ] Configure monthly data update schedule
- [ ] Set up monitoring alerts (features in extreme ranges)
- [ ] Train team on monitoring sheet interpretation
- [ ] Document handoff procedures

### 5.4 Ongoing Maintenance

- **Monthly (30–40 min):** Download new FRED-MD + ETF prices, update features, generate forecasts
- **Quarterly (3–4 hours):** Re-train models with expanded data, update SHAP values
- **Annually:** Full ALFRED validation, review deployment criteria, update methodology if needed

---

## Final Deliverables Checklist

- [ ] Automated model tournament scripts
- [ ] Feature engineering pipeline (250–300 cluster representatives)
- [ ] Dominant driver reports (top 5–10 per asset with SHAP values)
- [ ] Interactive Streamlit dashboard (5 pages)
- [ ] ALFRED validation report (revision risk, feature stability)
- [ ] Production model registry (versioned, reproducible)
- [ ] Real-time monitoring sheets (current regime + signals)
- [ ] User documentation (methodology, interpretation guide)
- [ ] Code repository with README and examples
- [ ] Deployment guide for IT team

---

*End of Technical Appendix C*

---

*End of Document Suite*
