# Technical Appendix A: Data Infrastructure & Feature Engineering

**Document 2 of 4** | **Version 1.0** | **January 27, 2026**

---

## Contents

1. [Data Sources & Acquisition](#data-sources--acquisition)
   - 1.1 [FRED-MD Database](#11-fred-md-database)
   - 1.2 [Asset Return Data — Historical Proxy Strategy](#12-asset-return-data--historical-proxy-strategy)
   - 1.3 [FRED Historical Vintage Reconstruction](#13-fred-historical-vintage-reconstruction-alfred)
2. [Two-Phase Data Strategy](#two-phase-data-strategy)
3. [Six-Step Feature Engineering Pipeline (Stateless)](#six-step-feature-engineering-pipeline-stateless)
4. [Step 7: State-Dependent Features (PIT Pipeline)](#step-7-state-dependent-features-pit-pipeline)
   - 4.1 [Cointegration & Error Correction Terms](#41-cointegration--error-correction-terms)
   - 4.2 [Hierarchical Clustering & Feature Selection](#42-hierarchical-clustering--feature-selection)
5. [Execution Timing & Requirements](#execution-timing--requirements)

---

## Data Sources & Acquisition

### 1.1 FRED-MD Database

**Primary Source:** Federal Reserve Economic Data — Monthly Database (FRED-MD)

- 128 macroeconomic time series
- Monthly frequency, 1959–present
- Developed by McCracken & Ng (2016)
- Standardized transformation codes for stationarity
- 8 categories: Output/Income (17), Labor (32), Housing (10), Consumption/Orders (14), Money/Credit (14), Interest Rates/Spreads (22), Prices (21), Stock Market (3)

**Historical Vintages (ALFRED):**
- **1999-08 to 2014-12:** `historical_fred-md.zip`
- **2015-01 to 2024-12:** `historical-vintages-of-fred-md-2015-01-to-2024-12.zip`
- **2025-01 to Present:** Individual monthly CSVs (e.g., `2025-01.csv`, `current.csv`)
- **Source:** [St. Louis Fed Research / McCracken Databases](https://www.stlouisfed.org/research/economists/mccracken/fred-databases)

### 1.2 Asset Return Data — Historical Proxy Strategy

| Asset | Modern Data | Historical Proxy | Method |
|---|---|---|---|
| S&P 500 | SPY (1993+) | FRED-MD `S&P 500` | Spliced at overlap |
| 10Y Bond | IEF (2002+) | FRED-MD `GS10` (Synthetic Return) | Yield-to-Return conversion |
| Gold | GLD (2004+) | FRED-MD `PPICMM` (PPI Metals) | Proxy Splicing |

### 1.3 FRED Historical Vintage Reconstruction (ALFRED)

To prevent look-ahead bias during validation, the system reconstructs a **Point-In-Time (PIT)** dataset using historical vintages. This ensures models are tested ONLY on data that was available at the time of prediction.

#### 1.3.1 Acquisition & Storage
- **Bulk Archives:** ZIP files containing monthly CSVs from 1999 to 2024 are downloaded and extracted to `data/raw/vintages/`.
- **Recent Vintages:** Individual monthly CSVs (2025+) are scraped directly from the St. Louis Fed research page.
- **Naming Convention:** Files are stored as `YYYY-MM.csv`, representing the vintage released in that month.

#### 1.3.2 Point-In-Time Alignment
For any target validation date $t$, the system identifies the "Relevant Vintage":
1. Finds the closest available vintage file $V$ where $V_{date} \le t$.
2. Example: For a validation date of `2015-06-30`, the loader selects `2015-06.csv`.
3. **Publication Lag:** Since FRED-MD vintages are published monthly, the `2015-06.csv` file typically contains data ending in `2015-05` (May). This naturally enforces a 1-month publication lag for macroeconomic data.

#### 1.3.3 Feature Reconstruction Methodology
A single PIT observation for date $t$ is NOT simply a row from a table; it is the result of a full reconstruction:
1. **Load Raw Vintage:** The entire history of raw data present in `V.csv` is loaded.
2. **Apply Pipeline:** The full Seven-Step Feature Engineering Pipeline (Transformations, Ratios, Quintiles, Cointegration, Momentum) is executed on the *entire historical series* within that vintage.
3. **Terminal Selection:** The last row of this fully processed dataset is extracted as the feature vector $X_t$.
4. **Alias Resolution:** The system handles variable name changes (e.g., mapping `VXOCLSx` to the modern `VIXCLSx`) to ensure consistency across 25 years of vintage history.

---

## Two-Phase Data Strategy

### 2.1 Phase 1: Discovery (Revised Data)

**Objective:** Find TRUE economic relationships

**Data:** Final Revised FRED-MD (Current Vintage) aligned via **Lagged Alignment Protocol**.

- **Lagged Alignment:** Features are shifted forward by `publication_lag_months` (default = 1). To predict returns starting Feb 1, the model only sees data dated Jan 1 or earlier.
- **Why:** GDP revisions can be 1–2% — preliminary data adds noise. Discovery finds the "true" signals.
- **Example:** Initial GDP = 2.0%, Final = 3.5% → the 3.5% drove returns.
- **Benefits:** Enables clean SHAP analysis — importance reflects economics, not data quality.
- **Process:** Download latest FRED-MD → Apply Lagged Alignment → Train all models → Identify dominant drivers.

### 2.2 Phase 2: Validation (ALFRED Real-Time)

**Objective:** Test real-world viability

**Data:** ALFRED Vintages (exact data available on each historical date)

- ALFRED = Archival Federal Reserve Economic Data
- Preserves exact vintage published on each date
- For each semi-annual date (2000–2024): Download vintage for all 128 series
- Assume 1-month publication lag (June 30 uses data through April 30)
- **API:** `fred.get_series(..., vintage_dates='2015-05-31')`
- Re-run Discovery models with real-time data
- Compare IC(revised) vs. IC(realtime), compute Revision Risk

### 4.2 Revision Risk Analysis

- **Revision Risk (IC)** = (IC_revised − IC_realtime) / IC_revised
- **Regime-Conditional Revision**: Analyzes risk during NBER recessions vs. expansions to detect cyclical data quality decay.
- **Feature Stability Score** = Correlation(SHAP_revised, SHAP_realtime)

**Deployment Criteria:**
- **IC(realtime)**: Must exceed asset-specific "Acceptable" threshold (Equities > 0.05, Bonds > 0.08, Gold > 0.04)
- **Revision Risk**: < 30%
- **Feature Stability**: > 0.70

---

## Six-Step Feature Engineering Pipeline (Stateless)

Automated pipeline: ~600 raw features generated pointwise or using strictly expanding windows. This base feature set is pre-computed before the CV loop.

### Step 0: Data Acquisition

- **Script:** Independent `data_acquisition.py` module
- **Action:** Downloads latest FRED-MD vintage (128 series), Historical Vintages, and Asset Prices (Yahoo: SPY, IEF, GLD + FRED Proxies)
- **Output:** Human-readable CSVs in `data/raw/` (`fred_md.csv`, `assets.csv`, `vintages/`)
- **Process:**
    - Fetch FRED-MD current vintage
    - Fetch FRED-MD historical vintages (Download & Unzip Archives + Monthly CSVs)
    - Fetch Asset Prices & Calculate Returns
    - Extend to 1959 using FRED-MD proxies (consistent with Section 1.2)
    - **Proxy Strategy:**
        - **Equity:** Uses `S&P 500` column from FRED-MD.
        - **Bonds:** Synthetic returns calculated from `GS10` (10-Year Yield).
        - **Gold:** Uses `PPICMM` (Producer Price Index: Metals) as a long-term proxy.
    - Save raw data to proper CSV format
- **Result:** ~792 months of data (Jan 1959 – present) ready for loading

### Step 1: Preserve Stationary Levels

- Run ADF + KPSS stationarity tests on all 128 variables
- Identify ~20–30 level-stationary series (unemployment, rates, spreads, VIX)
- Save these separately BEFORE applying transformations
- **Critical:** Needed for ratio construction (e.g., M2/GDP requires level GDP)
- **Result:** 'stationary_levels' dataset preserved

### Step 2: Apply FRED-MD Transformations

| Code | Transformation | Examples |
|---|---|---|
| 1 | No transformation (stationary) | Unemployment, Fed Funds, VIX |
| 2 | First difference: Δx | Level series needing differencing |
| 5 | Growth rate: Δlog(x) | GDP, IP, Employment, M2 |
| 6 | Acceleration: Δ²log(x) | Highly persistent series |

**Result:** ~128 stationary transformed series

### Step 3: Generate Macro Ratios

- **Liquidity Ratios:** M2/GDP, M2/Personal Income, Reserves/GDP
- **Leverage Ratios:** Total Debt/GDP, Consumer Debt/Disposable Income
- **Real Variables:** Nominal rates − Inflation (real yields, real Fed Funds)
- **Valuation Ratios:** P/E ratio / growth measures
- **Activity Ratios:** IP/Employment, Retail Sales/Personal Income
- Test each ratio for stationarity (many are cointegration relationships)
- **Result:** ~50–100 ratio features

### Step 3.5: Create Regime-Level Quintile Features

- Select ~15–20 key stationary variables (GS10, Fed Funds, spreads, VIX, unemployment)
- Compute historical quintiles using a **Robust Expanding Rank** methodology:
  - **Rank-Based**: $Score = (Rank - 1) / (Count - 1)$ using expanding windows.
  - **Boundary Safety**: Clips final quintiles to $[1, N]$ to handle exact 1.0 scores.
  - **Burn-In**: 60 Months (5 Years) minimum history required.
  - **Evaluation**: Computers **Monotonicity** (Spearman correlation of quintile means) and Q5-Q1 spread.
- Create features: One-hot encoding.
- **Result**: ~40–60 quintile features with high integrity.

### Step 4: Cointegration Analysis & Error Correction Terms (MOVED)
*This step is now part of the **State-Dependent Features (PIT Pipeline)**. See Section 4.*

### Step 5: Momentum Features

For all stationary series: Compute 3M, 6M, 12M changes

**Result:** ~500–600 momentum features

### Step 6: Cross-Asset Features

- Rolling correlations (6M, 12M): Equity–Bond, Equity–Gold, Bond–Gold
- Relative strength: SPX/Gold ratio, SPX/Bond ratio
- Volatility ratios: Equity Vol/Bond Vol
- **Result:** ~20–30 cross-asset features

---

## Step 7: State-Dependent Features (PIT Pipeline)

To eliminate look-ahead bias, features whose calculation depends on the global dataset (Stateful Features) are wrapped in `scikit-learn` transformers and fitted ONLY on the training fold within the cross-validation loop.

### 4.1 Cointegration & Error Correction Terms (Bayesian Weighting)

Cointegration captures long-run equilibrium relationships. The system implements a **Bayesian Prior Weighting** scheme to replace binary overrides:

#### 1. Bayesian Weighting
Weights are computed as: $W = (Prior^{0.3}) \cdot (Evidence^{0.5}) \cdot (Stability^{0.2})$
- **Prior**: Theoretical strength (e.g., PIH = 0.8, Fisher = 0.6).
- **Evidence**: Logistic mapping of Johansen/EG p-values.
- **Stability**: Historical presence of cointegration over rolling slices.

#### 2. Weighted Generation
- ECTs are multiplied by their final weight before being fed into models.
- Features with weights below **0.3** are automatically zeroed.

#### 3. ECT Characteristics
- ECT Z-scores are computed using **strictly expanding windows**.
- Half-life of mean reversion is monitored to detect "frozen" relationships.

**Theoretically Motivated Pairs:**
| Name | Series 1 | Series 2 | Economic Theory |
|---|---|---|---|
| quantity_theory | GDPC1 | M2SL | M2 velocity stability (Often rejected) |
| fisher_hypothesis | GS10 | CPIAUCSL | Real rate stationarity |
| okun_law | INDPRO | PAYEMS | Output-employment linkage (Robust) |
| housing_rates | HOUST | MORTGAGE30US | Housing demand elasticity |
| consumption_income | PCEC | DSPIC96 | Permanent Income Hypothesis |

### 4.2 Hierarchical Clustering & Feature Selection

### 4.1 The Substitution Instability Problem

GDP Growth, IP Growth, Income Growth all measure "economic activity" with correlation > 0.85. Models arbitrarily select one, causing:

- **Unstable SHAP values:** IP SHAP = 0.30 or 0.05 depending on whether GDP included
- **Unclear interpretation:** Top 10 features = 3 economic forces × 3 statistical variants each
- **Random feature selection** across different runs/time periods
- **Features are SUBSTITUTES not COMPLEMENTS**

### 4.3 Inference Statistics
All metrics now include:
- **Adjusted IC**: Spearman correlation with HAC standard errors.
- **Adjusted t-stat**: $t = IC / SE_{NW}$ using effective degrees of freedom ($N/h$).

## Robustness Suite

To prevent spurious discoveries, all "Champion" models undergo a final robustness battery:
- **Placebo Tests**: Target shuffling to compute empirical $p$-values.
- **Economic Significance**: Returns adjusted for realistic 10bps transaction costs and turnover.
- **Subsample Stability**: Split-half IC check to ensure results aren't driven by single outliers.

### 4.2 The Solution: Hierarchical Clustering

1. **Step 7.1:** Compute Spearman correlation matrix (600×600 features)
2. **Step 7.2:** Convert to distance: Distance = 1 − |correlation|
   - Using absolute value ensures negative correlations treated as redundancy
3. **Step 7.3:** Hierarchical clustering with average linkage
   - Why average? Balances Single (too aggressive) vs. Complete (too conservative)
4. **Step 7.4:** Cut dendrogram at similarity threshold 0.80 (distance 0.20)
   - Features with |corr| > 0.80 → Same cluster (high economic redundancy)
5. **Step 7.5:** Select ONE representative per cluster:
   - **Small clusters (2–3):** Highest univariate IC
   - **Large clusters (10+):** Highest correlation to cluster centroid
   - **Medium (4–9):** Centroid method, validate with IC

**Result:** ~250–300 features, each representing DISTINCT economic force. By fitting the clusterer on each training fold, we ensure that feature selection is based solely on correlations available at that specific point in time.

### 4.3 Expected Cluster Structure

| Cluster ID | Economic Force | Size | Representative (Example) |
|---|---|---|---|
| 1 | Real Economic Activity | 25–30 | Industrial Production Growth |
| 2 | Labor Market Health | 15–20 | Unemployment Rate |
| 3 | Inflation / Prices | 20–25 | CPI YoY |
| 4 | Short-Term Rates | 8–12 | Fed Funds Rate |
| 5 | Credit Conditions | 10–15 | BAA-AAA Spread |
| 6 | Liquidity / Money | 12–18 | M2/GDP Ratio |
| 7 | Market Volatility | 6–10 | VIX |
| ... | 40–60 total clusters | Varies | ~250–300 total |

### 4.4 Benefits vs. Simple Correlation Threshold

| Aspect | Old (corr > 0.995) | New (Clustering @ 0.80) |
|---|---|---|
| Features Removed | ~50–100 (near-duplicates) | ~300–350 (all redundancy) |
| Substitution Stability | Poor — many substitutes remain | Excellent — one per force |
| SHAP Stability | Low — rankings change across runs | High — stable dominant drivers |
| Interpretability | Top 10 = 3 forces × 3 variants | Top 10 = 10 distinct forces |


- Update rolling features: 5 min

---

*End of Technical Appendix A*
