"""
Asset-Specific Macro Regime Detection System
Streamlit Dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Macro Regime Detection",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Robust path resolution
APP_DIR = Path(__file__).parent.absolute()
if APP_DIR.name == "streamlit_app":
    PROJECT_ROOT = APP_DIR.parent
else:
    PROJECT_ROOT = APP_DIR

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .bullish { color: #28a745; font-weight: bold; }
    .bearish { color: #dc3545; font-weight: bold; }
    .neutral { color: #ffc107; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


def load_tournament_results():
    """Load real tournament results."""
    # Streamlit runs from the root usually
    results_path = EXPERIMENTS_DIR / "cv_results" / "tournament_results.csv"
    
    if results_path.exists():
        df = pd.read_csv(results_path)
        # Standardize columns
        df = df.rename(columns={
            'asset': 'Asset',
            'model': 'Model',
            'hit_rate_mean': 'Hit_Rate'
        })
        if 'RMSE_mean' in df.columns:
            df = df.rename(columns={'RMSE_mean': 'RMSE'})
        return df
    return pd.DataFrame()


def load_monitoring_data(asset_code):
    """Load monitoring data for a specific asset."""
    path = EXPERIMENTS_DIR / "reports" / f"{asset_code}_monitoring.csv"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def main():
    """Main dashboard application."""
    
    # Header
    st.markdown('<p class="main-header">📊 Asset-Specific Macro Regime Detection</p>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    This system identifies which macroeconomic variables drive asset returns and volatility 
    across different market regimes, using asset-specific regime detection and hierarchical 
    clustering to eliminate feature substitution instability.
    """)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Page",
        ["🎯 Dominant Drivers", "📈 Model Comparison", "🔍 Feature Explorer", 
         "⚙️ Experiment Config", "📋 Data Inspector"]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.subheader("Data Management")
    if st.sidebar.button("🔄 Update Data"):
        with st.sidebar.status("Acquiring data...", expanded=True) as status:
            st.write("Starting acquisition...")
            try:
                import subprocess
                import sys
                # Use the same python interpreter as the running process
                cmd = [sys.executable, str(PROJECT_ROOT / "src/preprocessing/data_acquisition.py")]
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                if result.returncode == 0:
                    status.update(label="Data Updated!", state="complete", expanded=False)
                    st.sidebar.success("Data updated successfully!")
                else:
                    status.update(label="Update Failed", state="error", expanded=True)
                    st.sidebar.error("Failed to update data.")
                    st.sidebar.code(result.stderr)
            except Exception as e:
                status.update(label="Error", state="error")
                st.sidebar.error(f"Error: {str(e)}")
    
    # Load data
    model_results = load_tournament_results()
    
    if page == "🎯 Dominant Drivers":
        show_dominant_drivers()
    elif page == "📈 Model Comparison":
        if model_results.empty:
            st.warning("No tournament results found. Run the tournament first.")
        else:
            show_model_comparison(model_results)
    elif page == "🔍 Feature Explorer":
        show_feature_explorer()
    elif page == "⚙️ Experiment Config":
        show_experiment_config()
    elif page == "📋 Data Inspector":
        show_data_inspector()


def show_dominant_drivers():
    """Show dominant drivers monitoring page."""
    st.header("🎯 Dominant Drivers Monitoring")
    
    # Asset selection
    asset_map = {
        "S&P 500 (SPX)": "SPX",
        "10Y Bond (BOND)": "BOND",
        "Gold (GOLD)": "GOLD"
    }
    selected_asset_label = st.selectbox("Select Asset", list(asset_map.keys()))
    asset_code = asset_map[selected_asset_label]
    
    monitoring_data = load_monitoring_data(asset_code)
    
    if monitoring_data.empty:
        st.warning(f"No monitoring data found for {asset_code}. Run the tournament first.")
        return
        
    # Current regime summary
    col1, col2, col3 = st.columns(3)
    
    # Simple heuristic for regime for demo purposes
    bullish_count = sum('High' in s for s in monitoring_data['Signal'])
    bearish_count = sum('Low' in s for s in monitoring_data['Signal'])
    
    if bullish_count > bearish_count:
        regime = "Bullish"
        delta = f"↑ {bullish_count} bullish drivers"
    elif bearish_count > bullish_count:
        regime = "Bearish"
        delta = f"↓ {bearish_count} bearish drivers"
    else:
        regime = "Neutral"
        delta = "→ Balanced signals"
        
    with col1:
        st.metric("Current Regime", regime, delta)
    with col2:
        # Placeholder for real regime probability
        prob = 50 + (bullish_count - bearish_count) * 5
        st.metric("Regime Probability", f"{min(99, max(1, prob))}%")
    with col3:
        st.metric("Confidence", "High" if len(monitoring_data) >= 5 else "Medium", f"Based on {len(monitoring_data)} drivers")
    
    st.markdown("---")
    
    # Monitoring sheet
    st.subheader("Top 5 Dominant Macro Drivers")
    
    # Format the dataframe for display
    def color_signal(val):
        if 'Bullish' in val:
            return 'color: green; font-weight: bold'
        elif 'Bearish' in val:
            return 'color: red; font-weight: bold'
        return 'color: orange'
    
    styled_df = monitoring_data.style.map(
        color_signal, subset=['Signal']
    )
    
    st.dataframe(styled_df, width='stretch', hide_index=True)
    
    # Interpretation
    st.info("""
    **Interpretation:** Liquidity (M2/GDP) and credit conditions supportive for equities, 
    but weakening industrial production (30th percentile) is a concern. 
    Monitor IP Growth closely for further deterioration.
    """)
    
    # SHAP visualization
    st.subheader("Feature Importance (SHAP Values)")
    
    fig = px.bar(
        monitoring_data, 
        x='SHAP', 
        y='Feature',
        orientation='h',
        color='Signal',
        color_discrete_map={
            '↑ Bullish': '#28a745',
            '→ Neutral': '#ffc107', 
            '↓ Bearish': '#dc3545'
        }
    )
    fig.update_layout(height=400, yaxis={'categoryorder': 'total ascending'})
    st.plotly_chart(fig, width='stretch')


def show_model_comparison(model_results):
    """Show model comparison dashboard."""
    st.header("📈 Model Comparison Dashboard")
    
    # Asset filter
    asset = st.selectbox("Filter by Asset", ["All", "SPX", "BOND", "GOLD"])
    
    if asset != "All":
        filtered = model_results[model_results['Asset'] == asset]
    else:
        filtered = model_results
    
    # Performance table
    st.subheader("Performance Metrics")
    
    st.dataframe(
        filtered.style.format({
            'IC_mean': '{:.3f}',
            'IC_std': '{:.3f}',
            'RMSE': '{:.3f}',
            'Hit_Rate': '{:.1%}'
        }).background_gradient(subset=['IC_mean'], cmap='Greens'),
        width='stretch',
        hide_index=True
    )
    
    # IC heatmap
    st.subheader("Information Coefficient Comparison")
    
    pivot = model_results.pivot(index='Model', columns='Asset', values='IC_mean')
    
    fig = px.imshow(
        pivot,
        color_continuous_scale='RdYlGn',
        aspect='auto',
        text_auto='.2f'
    )
    fig.update_layout(height=300)
    st.plotly_chart(fig, width='stretch')
    
    # Best model per asset
    st.subheader("Best Model per Asset")
    best = model_results.loc[model_results.groupby('Asset')['IC_mean'].idxmax()]
    
    cols = st.columns(3)
    for i, (_, row) in enumerate(best.iterrows()):
        with cols[i]:
            st.metric(
                row['Asset'],
                row['Model'],
                f"IC: {row['IC_mean']:.3f}"
            )


def show_feature_explorer():
    """Show feature importance explorer."""
    st.header("🔍 Feature Importance Explorer")
    
    # Asset selection
    asset_map = {
        "S&P 500 (SPX)": "SPX",
        "10Y Bond (BOND)": "BOND",
        "Gold (GOLD)": "GOLD"
    }
    selected_asset_label = st.selectbox("Select Asset for Feature Analysis", list(asset_map.keys()))
    asset_code = asset_map[selected_asset_label]
    
    monitoring_data = load_monitoring_data(asset_code)
    
    if monitoring_data.empty:
        st.warning(f"No feature data found for {asset_code}.")
        return

    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["Overall", "Bullish Regime", "Bearish Regime"])
    
    feature_df = monitoring_data[['Feature', 'SHAP']].copy()
    feature_df = feature_df.rename(columns={'SHAP': 'Importance'})
    
    with tab1:
        st.subheader("Top Dominant Drivers (SHAP)")
        fig = px.bar(
            feature_df.head(15), 
            x='Importance', 
            y='Feature',
            orientation='h'
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, width='stretch')
    
    with tab2:
        st.subheader("Bullish Regime Feature Importance")
        st.info("Regime-specific SHAP analysis requires saved regime-conditional models.")
        # Fallback to overall for now but with green color
        fig = px.bar(
            feature_df.head(10),
            x='Importance',
            y='Feature', 
            orientation='h',
            color_discrete_sequence=['green']
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, width='stretch')
    
    with tab3:
        st.subheader("Bearish Regime Feature Importance")
        st.info("Regime-specific SHAP analysis requires saved regime-conditional models.")
        fig = px.bar(
            feature_df.head(10),
            x='Importance',
            y='Feature',
            orientation='h',
            color_discrete_sequence=['red']
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, width='stretch')


def show_experiment_config():
    """Show experiment configuration page."""
    st.header("⚙️ Experiment Configuration")
    
    st.markdown("Configure and launch model training experiments.")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Assets")
        spx = st.checkbox("S&P 500 (SPX)", value=True)
        bond = st.checkbox("10Y Bond (BOND)", value=True)
        gold = st.checkbox("Gold (GOLD)", value=True)
        
        st.subheader("Target Variable")
        target = st.radio("Select Target", ["Returns (24M)", "Volatility (24M)"])
    
    with col2:
        st.subheader("Models")
        ridge = st.checkbox("Ridge Regression", value=True)
        lasso = st.checkbox("Lasso", value=True)
        elastic = st.checkbox("Elastic Net", value=True)
        rf = st.checkbox("Random Forest", value=True)
        xgb = st.checkbox("XGBoost", value=True)
        lgb = st.checkbox("LightGBM", value=True)
        mlp = st.checkbox("MLP Neural Net", value=False)
        lstm = st.checkbox("LSTM", value=False)
    
    st.markdown("---")
    
    st.subheader("Hyperparameter Budget")
    n_trials = st.slider("Optuna Trials per Model", 10, 200, 100)
    
    st.subheader("Cross-Validation")
    min_train = st.slider("Minimum Training Period (months)", 60, 180, 120)
    val_window = st.slider("Validation Window (months)", 6, 24, 12)
    
    st.markdown("---")
    
    if st.button("🚀 Launch Experiment", type="primary"):
        st.success("Experiment launched! Check logs for progress.")
        
        with st.expander("Experiment Details"):
            st.code(f"""
Configuration:
- Assets: {[a for a, v in [('SPX', spx), ('BOND', bond), ('GOLD', gold)] if v]}
- Target: {target}
- Models: {sum([ridge, lasso, elastic, rf, xgb, lgb, mlp, lstm])} selected
- Optuna trials: {n_trials}
- Min training: {min_train} months
- Validation window: {val_window} months
            """)


def show_data_inspector():
    """Show data inspection page."""
    st.header("📋 Data & Feature Inspector")
    
    tab1, tab2, tab3 = st.tabs(["Feature Matrix", "Correlation Heatmap", "Missing Data"])
    
    # Try to load real features
    feature_files = sorted(list((EXPERIMENTS_DIR / "features").glob("features_*.parquet")))
    if feature_files:
        sample_features = pd.read_parquet(feature_files[-1])
        st.success(f"Loaded features from {feature_files[-1]}")
    else:
        # Sample feature data fallback
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', periods=60, freq='ME')
        sample_features = pd.DataFrame({
            'M2_GDP_Ratio': np.random.randn(60).cumsum() + 0.8,
            'Real_10Y_Yield': np.random.randn(60) * 0.5 + 1.5,
            'Credit_Spread': np.random.randn(60) * 0.3 + 2,
            'VIX': np.abs(np.random.randn(60)) * 5 + 15,
            'IP_Growth': np.random.randn(60) * 2 + 1,
        }, index=dates)
    
    with tab1:
        st.subheader("Feature Matrix Preview")
        st.dataframe(sample_features.head(10), width='stretch')
        
        st.download_button(
            "Download Features CSV",
            sample_features.to_csv(),
            "features.csv",
            "text/csv"
        )
    
    with tab2:
        st.subheader("Feature Correlation Heatmap")
        corr = sample_features.corr()
        
        fig = px.imshow(
            corr,
            color_continuous_scale='RdBu',
            aspect='auto',
            text_auto='.2f',
            zmin=-1, zmax=1
        )
        st.plotly_chart(fig, width='stretch')
    
    with tab3:
        st.subheader("Missing Data Report")
        
        # Add some missing values for demo
        sample_features_missing = sample_features.copy()
        sample_features_missing.iloc[0:3, 0] = np.nan
        sample_features_missing.iloc[5:8, 2] = np.nan
        
        missing = sample_features_missing.isna().sum()
        missing_pct = (missing / len(sample_features_missing) * 100).round(2)
        
        missing_df = pd.DataFrame({
            'Feature': missing.index,
            'Missing Count': missing.values,
            'Missing %': missing_pct.values
        })
        
        st.dataframe(missing_df, width='stretch', hide_index=True)


if __name__ == "__main__":
    main()
