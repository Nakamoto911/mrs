# Asset-Specific Macro Regime Detection System
## System Architecture & Methodology Master Document

**Version 1.1** | **February 2, 2026**

---

## 1. System Overview

### 1.1 Strategic Objective

The **Asset-Specific Macro Regime Detection System** is an automated model discovery platform designed to identify, for each asset (US Equities, Bonds, Gold), the specific macroeconomic variables that drive returns and volatility. Unlike traditional approaches that assume universal economic regimes affect all assets identically, this system detects **asset-specific regimes** based on each asset’s unique macro sensitivities. It utilizes the identification of the current and shifting economic regimes to determine which asset types to prioritize, as different asset classes react positively or negatively based on the prevailing economic conditions.

The system answers three critical questions for every asset class:
1.  **What drives the market?** Identification of the top 5–10 dominant macroeconomic drivers (out of ~750 candidates).
2.  **How do drivers change?** Detection of distinct sensitivities in bullish vs. bearish regimes (e.g., Liquidity drives Bull markets; Credit Spreads drive Bear markets).
3.  **Is the signal real?** Rigorous validation using Real-Time (ALFRED) data to ensure signals were visible to investors at the time, not artifacts of data revisions.

### 1.2 Core Methodological Innovations

1.  **Asset-Specific Regime Detection**: Instead of imposing top-down regimes (e.g., NBER Recessions), each asset gets independent bullish/bearish classification via Statistical Jump Models. Academic research (2024) indicates this approach outperforms universal models by 15–30%.
2.  **Hierarchical Clustering**: To solve "Substitution Instability" (where models arbitrarily swap correlated proxies like GDP vs. IP), features are grouped into **20 Orthogonal Factors**. Medoid selection ensures dominant drivers represent stable economic forces rather than statistical noise.
3.  **Two-Phase Data Strategy**:
    *   **Phase 1 (Discovery)**: Uses Revised Data to identify true economic relationships.
    *   **Phase 2 (Validation)**: Uses Real-Time (ALFRED) Vintages to verify deployability.
4.  **12-Month Strategic Horizon**: A focus on annual rebalancing balances macro signal extraction with forecast reliability, aligning with tactical benchmark standards.

### 1.3 System Output: Real-Time Monitoring Example

**Example: S&P 500 Current Regime — Cautiously Bullish**

| Rank | Dominant Macro Driver | SHAP Value | Current Signal |
|---|---|---|---|
| 1 | M2/GDP Ratio (Liquidity) | 0.42 | ↑ Bullish (85th percentile) |
| 2 | Real 10Y Yield | 0.38 | → Neutral (50th percentile) |
| 3 | Credit Spread (BAA-10Y) | 0.31 | ↑ Bullish (15th pct, tight) |
| 4 | VIX Level | 0.24 | → Neutral (45th percentile) |
| 5 | Industrial Production Growth | 0.19 | ↓ Bearish (30th percentile) |

*Interpretation: Liquidity and credit conditions are supportive, but weakening industrial activity is a risk factor.*

---

## 2. Data Infrastructure (The "Input")

The foundation of the system is a rigorous pipeline that transforms raw economic data into predictive signals while strictly preventing look-ahead bias.

### 2.1 Data Sources

#### Primary Source: FRED-MD Database
-   **Content**: 128 monthly macroeconomic time series (Output, Labor, Housing, Money, Prices, etc.).
-   **History**: Extended to 1959 using spliced proxies.
-   **Exclusions**: Categories 6 (Interest/Exchange Rates) and 8 (Stock Market) are **excluded as raw features** to prevent causality leakage.
    *   *Note:* Specific variables from these categories (e.g., `GS10`, `FEDFUNDS`) are retained *temporarily* solely to calculate derived features (Real Rates, Spreads) and are dropped from the final feature set if they appear in raw form.
-   **Files**: `historical_fred-md.zip` (1999–2014), `historical-vintages-of-fred-md-*.zip` (2015–Present).

#### Asset Return Proxies
| Asset | Modern Data | Historical Proxy | Method |
|---|---|---|---|
| S&P 500 | SPY (1993+) | FRED-MD `S&P 500` | Spliced at overlap |
| 10Y Bond | IEF (2002+) | FRED-MD `GS10` | Yield-to-Return conversion |
| Gold | GLD (2004+) | FRED-MD `PPICMM` | Proxy Splicing |

### 2.2 Two-Phase Data Strategy

1.  **Discovery Phase (Revised Data)**
    *   **Objective**: Find *true* economic relationships.
    *   **Protocol**: Uses final revised data shifted by the 1-month **Publication Lag**. This ensures that the training labels capture the "ground truth" of the economy, even if that truth was only fully revealed later.

2.  **Validation Phase (Real-Time ALFRED Vintages)**
    *   **Objective**: Test real-world viability.
    *   **Protocol**: Reconstructs the exact dataset available to an investor on a specific date $t$ using **Point-In-Time (PIT)** reconstruction mechanics (see Section 2.4).

### 2.3 The 7-Step Feature Engineering Pipeline

The system generates ~750 features through a robust pipeline. Steps 0-6 are stateless (pre-computable), while Step 7 is state-dependent (computed within the CV loop).

**Step 0: Data Acquisition & Pre-processing**
-   Downloads FRED-MD and asset prices.
-   Extends history to 1959.
-   Applies **Lagged Alignment** (default = 1 month) to account for publication delay.

**Step 1: Stationary Levels Preservation**
-   Runs ADF/KPSS tests to identify and preserve ~20–30 naturally stationary series (e.g., Unemployment Rate, VIX, Spreads) *before* transformation.

**Step 2: FRED-MD Transformations**
-   Applies standardized transformations (Codes 1–7) based on McCracken & Ng (2016):
    *   Code 2: $\Delta x$ (First difference)
    *   Code 5: $\Delta \log(x)$ (Growth rates)
    *   Code 6: $\Delta^2 \log(x)$ (Acceleration)

**Step 3: Macro Ratios**
-   Constructs ~50–100 economic ratios using preserved levels:
    *   **Liquidity**: M2 / GDP
    *   **Valuation**: P/E Ratios
    *   **Real Rates**: Nominal Rates - Inflation

**Step 4: Momentum Features**
-   Computes 3M, 6M, and 12M changes for all stationary series.

**Step 5: Cross-Asset Features**
-   Rolling correlations (6M, 12M) and relative strength ratios (e.g., SPX/Gold).

**Step 6: State-Dependent Features (PIT Pipeline)**
*Note: These steps are wrapped in transformers and fitted strictly on training folds.*

*   **Step 6.1: Regime-Level Quintiles**: Captures non-linear dynamics (e.g., Interest rates falling from 5% vs. 1%).
*   **Step 6.2: Cointegration & ECTs**: Validates long-run relationships using **Bayesian Prior Weighting**:
    $$W = (Prior^{0.3}) \cdot (Evidence^{0.5}) \cdot (Stability^{0.2})$$
    Terms with $W < 0.3$ are zeroed.

**Step 7: Hierarchical Clustering & Feature Selection**
To eliminate the "Substitution Instability Problem" (features serving as interchangeable proxies):
1.  Compute Spearman correlation matrix.
2.  Convert to distance: $D = 1 - |\text{correlation}|$.
3.  Perform Hierarchical Clustering (Average Linkage).
4.  **Cut Threshold**: Similarity 0.40 (Distance 0.60).
5.  **Selection**: Choose **ONE** representative per cluster using **Medoid Selection** (feature closest to the cluster center).
6.  **Result**: Reduces ~750 features to ~250–300 distinct, orthogonal economic factors.

### 2.4 Point-In-Time (PIT) Reconstruction Rules

For Model Validation (Phase 2), a single observation for date $t$ is reconstructed as follows:
1.  **Vintage Selection**: Select the file $V$ closest to $t$ where $V_{date} \le t$ (e.g., for `2015-06-30`, use `2015-06.csv`).
2.  **Full Re-Processing**: Load the *entire* raw history from that vintage.
3.  **Pipeline Execution**: Run Steps 1–7 anew on this vintage history.
4.  **Terminal Extraction**: Isolate the last row as the feature vector $X_t$.

---

## 3. The Modeling Engine (The "Brain")

### 3.1 Asset-Specific Regime Detection

The system employs a bottom-up approach to regime definition:
1.  **Regime Labeling**: A **Statistical Jump Model** analyzes the return and volatility series of *each* asset separately to classify months as "Bullish" (High Return/Low Vol) or "Bearish" (Low Return/High Vol).
2.  **Regime Forecasting**: A LightGBM classifier predicts the probability of the *next* 6-month regime state ($P(R_{t+6})$) using current macro features.

### 3.2 Robust Training Layer

To manage high dimensionality and ensure stability, all models utilize a standardized preprocessing layer:
-   **Sparse Imputation**: **Strict Rolling Imputation** (Expanding Median) shifted by 1 month. Missing values at the start of history are filled with 0.0 (neutral prior).
-   **Variance Filtering**: Removal of constant/near-constant features.
-   **Degrees of Freedom Guard**: Enforces $N > K + 20$. If violated, **Automated PIT Feature Pruning** retains only the top $(N-20$ features based on data density.

### 3.3 Model Tournament Framework

The system utilizes a "Tournament" of competing model families, optimized via Optuna (100 trials, 5-fold Time-Series CV).

**Linear Models (Benchmarks)**
-   **Ridge / Lasso / Elastic Net**: Benchmark linear proxies. Lasso requires `max_iter=100,000`.
-   **VECM**: Captures cointegrated mean-reversion.

**Tree-Based Ensembles**
-   **Random Forest**: `n_jobs=1`, `max_depth=4` for regularization.
-   **XGBoost**: State-of-the-art gradient boosting.
-   **LightGBM**: Optimized for speed and large feature sets.

**Neural Networks**
-   **LSTM V2**: Features "Adaptive" sequence strategies, temporal attention, and MC Dropout for uncertainty estimation.

**Ensemble Strategy**
The "Champion" model is rarely a single algorithm. The system averages the signals of the **Top 5** performing models (ranked by IC) to create a `Ensemble_Top5` meta-model, which typically offers the highest stability.

---

## 4. Validation & Risk Management

### 4.1 ALFRED Validation Protocol

Any model identified as a "Champion" in the Discovery Phase must survive the **ALFRED Validation Protocol**:
1.  **Re-Run**: The frozen model configuration is tested on 48 semi-annual ALFRED vintages (2000–2024).
2.  **Verify**: Does the signal generated using *real-time* data match the signal generated using *revised* data?

### 4.2 Metrics & Thresholds

**Key Performance Metrics**
-   **Information Coefficient (IC)**: Spearman rank correlation between signal and forward returns (Primary metric for returns).
-   **RMSE**: Root Mean Squared Error (Primary metric for volatility).
-   **Revision Risk**: The degradation in performance due to data revisions.
    $$ \text{Revision Risk} = \frac{IC_{\text{Revised}} - IC_{\text{RealTime}}}{IC_{\text{Revised}}} $$

**Revision Risk Criteria**
-   **Acceptable Risk**: $< 30\%$ degradation.
-   **Feature Stability**: Correlation between Revised and Real-Time SHAP values $> 0.70$.

**Literature-Calibrated Benchmarks**
Based on academic standards (Welch & Goyal, 2008), the following thresholds determine deployability:

| Asset Class | Good (Meaningful) | Acceptable (Detectable) | Deployment Cutoff |
|---|---|---|---|
| **Equities (SPX)** | IC > 0.08 | IC > 0.05 | **IC > 0.05** |
| **Bonds (BOND)** | IC > 0.12 | IC > 0.08 | **IC > 0.08** |
| **Commodities** | IC > 0.06 | IC > 0.04 | **IC > 0.04** |

---

## 5. Implementation & Operations

### 5.1 Code Organization

The codebase follows a modular structure facilitating extension and maintenance:

```text
macro_regime_system/
├── data/
│   ├── fred_md/                 # FRED-MD downloads
│   ├── alfred_vintages/         # Real-Time Vintages
│   └── processed/               # Parquet Features
├── src/
│   ├── feature_engineering/     # The 7-Step Pipeline
│   │   ├── transformations.py
│   │   ├── cointegration.py
│   │   └── hierarchical_clustering.py
│   ├── models/                  # Model Definitions
│   └── evaluation/              # Cross-Validation & SHAP
├── streamlit_app/               # UI Components
├── experiments/                 # Outputs (Models, Logs, Predictions)
└── run_tournament.py            # Main Entry Point
```

### 5.2 Execution Workflows

The `run_tournament.py` script supports modular execution flags:
-   **Full Discovery**: `python run_tournament.py --assets all --models all`
-   **Single Cycle**: `python run_tournament.py --asset SPX --model xgboost`
-   **Feature Debug**: `python run_tournament.py --features-only`

### 5.3 Production Interface (Streamlit)

A 5-page **Streamlit** application serves as the control center:
1.  **Experiment Config**: Launch new tournaments and continuous integration steps.
2.  **Model Comparison**: Leaderboards highlighting the "Ensemble Top 5".
3.  **Feature Importance**: Waterfall charts and SHAP summaries for Bull/Bear regimes.
4.  **Dominant Drivers**: Real-time monitoring dashboard showing current macro signal percentiles.
5.  **Data Inspector**: Audit logs for Cointegration Stability and PIT data availability.

### 5.4 Deployment Workflow
1.  **Weekly**: Automated refresh of FRED-MD/Asset data.
2.  **Monthly**: Generate new forecasts using frozen models.
3.  **Quarterly**: Full model re-training and cluster re-generation.
4.  **Annually**: Comprehensive ALFRED Validation run to detect structural breaks in data reliability.
