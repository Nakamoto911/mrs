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


def load_sample_data():
    """Load sample data for demonstration."""
    np.random.seed(42)
    
    # Sample monitoring data
    monitoring_data = pd.DataFrame({
        'Rank': [1, 2, 3, 4, 5],
        'Feature': ['M2/GDP Ratio', 'Real 10Y Yield', 'Credit Spread (BAA-10Y)', 
                   'VIX Level', 'IP Growth'],
        'SHAP': [0.42, 0.38, 0.31, 0.24, 0.19],
        'Current_Value': [0.85, 1.2, 1.8, 18.5, 2.1],
        'Percentile': [85, 50, 15, 45, 30],
        'Signal': ['↑ Bullish', '→ Neutral', '↑ Bullish', '→ Neutral', '↓ Bearish']
    })
    
    # Sample model results
    model_results = pd.DataFrame({
        'Asset': ['SPX', 'SPX', 'SPX', 'BOND', 'BOND', 'BOND', 'GOLD', 'GOLD', 'GOLD'],
        'Model': ['XGBoost', 'Ridge', 'LightGBM'] * 3,
        'IC_mean': [0.28, 0.22, 0.26, 0.32, 0.30, 0.31, 0.16, 0.14, 0.15],
        'IC_std': [0.08, 0.06, 0.07, 0.05, 0.04, 0.05, 0.06, 0.05, 0.06],
        'RMSE': [0.12, 0.14, 0.13, 0.08, 0.09, 0.085, 0.15, 0.16, 0.155],
        'Hit_Rate': [0.62, 0.58, 0.61, 0.68, 0.65, 0.67, 0.55, 0.53, 0.54]
    })
    
    return monitoring_data, model_results


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
                cmd = [sys.executable, "src/preprocessing/data_acquisition.py"]
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
    monitoring_data, model_results = load_sample_data()
    
    if page == "🎯 Dominant Drivers":
        show_dominant_drivers(monitoring_data)
    elif page == "📈 Model Comparison":
        show_model_comparison(model_results)
    elif page == "🔍 Feature Explorer":
        show_feature_explorer()
    elif page == "⚙️ Experiment Config":
        show_experiment_config()
    elif page == "📋 Data Inspector":
        show_data_inspector()


def show_dominant_drivers(monitoring_data):
    """Show dominant drivers monitoring page."""
    st.header("🎯 Dominant Drivers Monitoring")
    
    # Asset selection
    asset = st.selectbox("Select Asset", ["S&P 500 (SPX)", "10Y Bond (BOND)", "Gold (GOLD)"])
    
    # Current regime summary
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Current Regime", "Cautiously Bullish", "↑ 2 bullish drivers")
    with col2:
        st.metric("Regime Probability", "68%", "+5% from last month")
    with col3:
        st.metric("Confidence", "High", "Based on 5 dominant drivers")
    
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
    
    styled_df = monitoring_data.style.applymap(
        color_signal, subset=['Signal']
    ).format({
        'SHAP': '{:.2f}',
        'Current_Value': '{:.2f}',
        'Percentile': '{:.0f}th'
    })
    
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
    
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
    st.plotly_chart(fig, use_container_width=True)


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
        use_container_width=True,
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
    st.plotly_chart(fig, use_container_width=True)
    
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
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["Overall", "Bullish Regime", "Bearish Regime"])
    
    # Sample feature importance data
    np.random.seed(42)
    features = [f"Feature_{i}" for i in range(20)]
    importance = np.random.exponential(0.1, 20)
    importance = np.sort(importance)[::-1]
    
    feature_df = pd.DataFrame({
        'Feature': features,
        'Importance': importance
    })
    
    with tab1:
        st.subheader("Overall Feature Importance")
        fig = px.bar(
            feature_df.head(15), 
            x='Importance', 
            y='Feature',
            orientation='h'
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("Bullish Regime Feature Importance")
        # Shuffle for demo
        bullish_df = feature_df.sample(frac=1, random_state=1).head(15)
        fig = px.bar(
            bullish_df,
            x='Importance',
            y='Feature', 
            orientation='h',
            color_discrete_sequence=['green']
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("Bearish Regime Feature Importance")
        bearish_df = feature_df.sample(frac=1, random_state=2).head(15)
        fig = px.bar(
            bearish_df,
            x='Importance',
            y='Feature',
            orientation='h',
            color_discrete_sequence=['red']
        )
        fig.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig, use_container_width=True)


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
    
    # Sample feature data
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
        st.dataframe(sample_features.head(10), use_container_width=True)
        
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
        st.plotly_chart(fig, use_container_width=True)
    
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
        
        st.dataframe(missing_df, use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
