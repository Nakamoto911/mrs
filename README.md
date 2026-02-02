# Asset-Specific Macro Regime Detection System

A quantitative research platform for discovering which macroeconomic variables drive asset returns across different market regimes.

## Key Innovations

1. **Causality-First Data Selection**: Explicitly excludes FRED-MD Category 8 (Stock Market) and Category 6 (Interest and Exchange Rates) to prevent circularity and ensure focus on lead indicators.
1. **Asset-Specific Regime Detection**: Each asset gets independent bullish/bearish classification.
2. **Super-Clustering**: Eliminates feature redundancy at 0.40 similarity threshold using **Medoid Selection** for maximum stability.
3. **Two-Phase Data Strategy**: Discovery on lagged revised data, validation on ALFRED real-time vintages.
4. **Broad Feature Generation**: Generates ~750 Slope (Transformed) and Level (Quintile) features for all variables.
5. **Orthogonality Verification**: Automated checks ensure non-linear "Level" and linear "Slope" features remain distinct.
6. **Empirical Cointegration Validation**: Statistical verification of theoretical relationships (Johansen/Engle-Granger) to prevent spurious macro signals.
7. **Dynamic Data Strategy**: Automated identification of series via FRED-MD appendix to enable robust category-based filtering.

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
├── scripts/                      # Utility and verification scripts
│   └── verification/             # Scripts to verify pipeline components
├── tests/                        # Unit and Integration tests
├── logs/                         # Execution logs
├── streamlit_app/                # Interactive dashboard
├── experiments/                  # Output storage
├── run_tournament.py             # Discovery phase entry point
└── run_validation.py             # ALFRED validation phase entry point
```

## Feature Pipeline

1. Acquire Data (FRED-MD + Assets) - **Categories 6 (Interest Rates) & 8 (Stocks) Excluded**
2. Apply transformations for stationarity (Slopes)
3. Generate macro ratios
4. Create quintile features for ALL variables (Levels)
5. Validated Cointegration analysis (ECT)
6. Momentum features
7. **Super-Clustering (0.40) → ~250-300 broad factor representatives**

## Models

- **Linear**: Ridge, Lasso, Elastic Net, VECM
- **Tree-Based**: XGBoost, LightGBM, Random Forest
- **Neural**: MLP, LSTM

## Understanding IC Thresholds

This system uses literature-calibrated IC thresholds that reflect realistic 
expectations for macro-timing models at 12-month horizons:

| Asset Class | Excellent | Good | Acceptable | Minimum |
|-------------|-----------|------|------------|---------|
| Equities    | > 0.12    | > 0.08 | > 0.05   | > 0.03  |
| Bonds       | > 0.18    | > 0.12 | > 0.08   | > 0.05  |
| Commodities | > 0.10    | > 0.06 | > 0.04   | > 0.02  |

**Important**: An IC of 0.30+ would be extraordinary and triggers investigation for 
data leakage. Most successful quantitative strategies operate with ICs in the 
0.02-0.10 range at these horizons.

References:
- Welch & Goyal (2008): "A Comprehensive Look at The Empirical Performance of Equity Premium Prediction"
- Campbell & Thompson (2008): "Predicting Excess Stock Returns Out of Sample"

## Expected Performance Benchmarks

| Asset | Rating: Excellent | Rating: Good | Rating: Acceptable |
|-------|-------------------|--------------|--------------------|
| SPX   | > 0.12            | > 0.08       | > 0.05             |
| Bond  | > 0.18            | > 0.12       | > 0.08             |
| Gold  | > 0.10            | > 0.06       | > 0.04             |

## Dashboard Pages

1. **Dominant Drivers**: Real-time monitoring of top 5-10 macro features
2. **Model Comparison**: Performance metrics across models
3. **Feature Explorer**: SHAP-based importance by regime
4. **Experiment Config**: Launch training experiments
5. **Data Inspector**: Feature matrix and correlations

## License

Proprietary - See LICENSE.txt
