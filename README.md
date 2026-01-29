# Asset-Specific Macro Regime Detection System

A quantitative research platform for discovering which macroeconomic variables drive asset returns across different market regimes.

## Key Innovations

1. **Asset-Specific Regime Detection**: Each asset gets independent bullish/bearish classification
2. **Hierarchical Clustering**: Eliminates feature substitution instability at 0.80 similarity threshold
3. **Two-Phase Data Strategy**: Discovery on lagged revised data, validation on ALFRED real-time vintages
4. **Regime-Level Quintile Features**: Captures regime-dependent dynamics

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Acquire data (independent step)
python src/preprocessing/data_acquisition.py

# Run feature pipeline
python run_tournament.py --features-only

# Run full tournament
python run_tournament.py --assets all --models all

# Perform ALFRED Real-Time Validation (identifies Revision Risk)
python run_validation.py --asset SPX --model xgboost

# Batch validate all assets/models and generate summary report
python run_validation.py

# Launch dashboard
cd streamlit_app && streamlit run app.py
```

## Project Structure

```
macro_regime_system/
├── configs/                      # Configuration files
├── src/

│   ├── preprocessing/            # Data loading & transforms
│   │   ├── data_acquisition.py   # Independent data fetcher
│   │   └── data_loader.py        # Data reader & validator
│   ├── feature_engineering/      # Feature generation
│   ├── models/                   # Model implementations
│   └── evaluation/               # Evaluation & analysis
├── streamlit_app/                # Interactive dashboard
├── experiments/                  # Output storage
├── run_tournament.py             # Discovery phase entry point
└── run_validation.py             # ALFRED validation phase entry point
```

## Feature Pipeline

1. Acquire Data (FRED-MD + Assets) via independent script
2. Apply transformations for stationarity
3. Generate macro ratios
4. Create quintile features
5. Cointegration analysis (ECT)
6. Momentum features
7. **Hierarchical clustering → ~250-300 features**

## Models

- **Linear**: Ridge, Lasso, Elastic Net, VECM
- **Tree-Based**: XGBoost, LightGBM, Random Forest
- **Neural**: MLP, LSTM

## Expected Performance

| Asset | IC (Revised) | IC (Real-Time) | Deploy Criteria |
|-------|--------------|----------------|-----------------|
| SPX   | 0.20-0.28    | 0.15-0.22      | IC > 0.15, Risk < 30% |
| Bond  | 0.28-0.35    | 0.24-0.30      | IC > 0.15, Risk < 30% |
| Gold  | 0.12-0.20    | 0.10-0.16      | IC > 0.15, Risk < 30% |

## Dashboard Pages

1. **Dominant Drivers**: Real-time monitoring of top 5-10 macro features
2. **Model Comparison**: Performance metrics across models
3. **Feature Explorer**: SHAP-based importance by regime
4. **Experiment Config**: Launch training experiments
5. **Data Inspector**: Feature matrix and correlations

## License

Proprietary - See LICENSE.txt
