"""
Macro Ratios Module
===================
Generates economically meaningful ratios from FRED-MD data.

Part of the Asset-Specific Macro Regime Detection System
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class MacroRatioGenerator:
    """
    Generates macro ratios from FRED-MD data.
    
    Categories:
    - Liquidity ratios: M2/GDP, Reserves/GDP
    - Leverage ratios: Debt/GDP, Consumer Debt/DPI
    - Real variables: Real yields, Real Fed Funds
    - Valuation ratios: P/E type measures
    - Activity ratios: Output/Employment measures
    """
    
    # Pre-defined ratio configurations
    DEFAULT_RATIOS = {
        'liquidity': [
            # (numerator, denominator, name, is_log_ratio)
            ('M2SL', 'INDPRO', 'M2_GDP_Ratio', True),
            ('M2SL', 'RPI', 'M2_Personal_Income', True),
            ('BOGMBASE', 'INDPRO', 'Reserves_GDP', True),
            ('TOTRESNS', 'INDPRO', 'TotalReserves_GDP', True),
        ],
        'leverage': [
            ('BUSLOANS', 'INDPRO', 'BusinessLoans_GDP', True),
            ('CONSUMER', 'RPI', 'ConsumerDebt_DPI', True),
            ('TOTALSL', 'RPI', 'TotalConsumerCredit_DPI', True),
        ],
        'real_rates': [
            # These are level differences, not log ratios
            ('GS10', 'CPIAUCSL_yoy', 'Real_10Y_Yield', False),
            ('GS5', 'CPIAUCSL_yoy', 'Real_5Y_Yield', False),
            ('GS1', 'CPIAUCSL_yoy', 'Real_1Y_Yield', False),
            ('FEDFUNDS', 'CPIAUCSL_yoy', 'Real_Fed_Funds', False),
            ('TB3MS', 'CPIAUCSL_yoy', 'Real_3M_Rate', False),
        ],
        'activity': [
            ('INDPRO', 'PAYEMS', 'IP_Per_Employee', True),
            ('RPI', 'PAYEMS', 'Income_Per_Employee', True),
            ('DPCERA3M086SBEA', 'PI', 'Consumption_Income_Ratio', True),
        ],
        'spreads': [
            # These are already spreads in FRED-MD
            ('BAA', 'AAA', 'BAA_AAA_Spread', False),
            ('GS10', 'GS1', 'Yield_Curve_10Y_1Y', False),
            ('GS10', 'TB3MS', 'Yield_Curve_10Y_3M', False),
            ('GS5', 'GS1', 'Yield_Curve_5Y_1Y', False),
        ],
    }
    
    def __init__(self, custom_ratios: Optional[Dict] = None):
        """
        Initialize ratio generator.
        
        Args:
            custom_ratios: Additional custom ratios to generate
        """
        self.ratios = self.DEFAULT_RATIOS.copy()
        if custom_ratios:
            self.ratios.update(custom_ratios)
    
    def _compute_yoy_inflation(self, df: pd.DataFrame, 
                               price_col: str = 'CPIAUCSL') -> pd.Series:
        """
        Compute year-over-year inflation rate.
        
        Args:
            df: DataFrame with price data
            price_col: Price index column name
            
        Returns:
            YoY inflation rate
        """
        if price_col not in df.columns:
            return pd.Series(dtype=float)
        
        prices = df[price_col]
        yoy = (prices / prices.shift(12) - 1) * 100
        return yoy
    
    def generate_ratios(self, df: pd.DataFrame, 
                        categories: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Generate macro ratios.
        
        Args:
            df: DataFrame with level data (not transformed)
            categories: List of ratio categories to generate (all if None)
            
        Returns:
            DataFrame with ratio features
        """
        if categories is None:
            categories = list(self.ratios.keys())
        
        outputs = []
        
        # First compute inflation for real rate calculations
        inflation_yoy = self._compute_yoy_inflation(df)
        df_with_inflation = df.copy()
        df_with_inflation['CPIAUCSL_yoy'] = inflation_yoy
        
        for category in categories:
            if category not in self.ratios:
                logger.warning(f"Unknown ratio category: {category}")
                continue
            
            for ratio_config in self.ratios[category]:
                numerator, denominator, name, is_log_ratio = ratio_config
                
                # Check if columns exist
                if numerator not in df_with_inflation.columns:
                    logger.debug(f"Numerator {numerator} not found for {name}")
                    continue
                if denominator not in df_with_inflation.columns:
                    logger.debug(f"Denominator {denominator} not found for {name}")
                    continue
                
                num = df_with_inflation[numerator]
                denom = df_with_inflation[denominator]
                
                try:
                    if is_log_ratio:
                        # Log ratio (for level variables)
                        ratio = np.log(num / denom.replace(0, np.nan))
                    else:
                        # Simple difference (for rates/spreads)
                        ratio = num - denom
                    
                    ratio.name = name
                    outputs.append(ratio)
                    logger.debug(f"Generated ratio: {name}")
                    
                except Exception as e:
                    logger.warning(f"Error computing {name}: {e}")
        
        ratios_df = pd.concat(outputs, axis=1) if outputs else pd.DataFrame(index=df.index)
        logger.info(f"Generated {len(ratios_df.columns)} ratio features")
        return ratios_df
    
    def generate_ratio_changes(self, ratios_df: pd.DataFrame,
                               windows: List[int] = [1, 3, 6, 12]) -> pd.DataFrame:
        """
        Generate changes in ratios over different windows.
        
        Args:
            ratios_df: DataFrame with ratio features
            windows: List of window sizes in months
            
        Returns:
            DataFrame with ratio change features
        """
        changes = []
        
        for col in ratios_df.columns:
            for window in windows:
                change = ratios_df[col].diff(window)
                change.name = f"{col}_chg_{window}M"
                changes.append(change)
        
        return pd.concat(changes, axis=1) if changes else pd.DataFrame(index=ratios_df.index)
    
    def add_custom_ratio(self, category: str, numerator: str, denominator: str,
                         name: str, is_log_ratio: bool = True):
        """
        Add a custom ratio configuration.
        
        Args:
            category: Category name
            numerator: Numerator column name
            denominator: Denominator column name
            name: Output feature name
            is_log_ratio: Whether to use log ratio
        """
        if category not in self.ratios:
            self.ratios[category] = []
        
        self.ratios[category].append((numerator, denominator, name, is_log_ratio))


class SpreadCalculator:
    """
    Calculates various yield spreads and credit spreads.
    """
    
    # Standard spread definitions
    STANDARD_SPREADS = {
        'credit': [
            ('BAA', 'AAA', 'BAA_AAA_Spread'),
            ('BAA', 'GS10', 'BAA_10Y_Spread'),
            ('AAA', 'GS10', 'AAA_10Y_Spread'),
        ],
        'yield_curve': [
            ('GS10', 'GS2', 'YC_10Y_2Y'),
            ('GS10', 'GS1', 'YC_10Y_1Y'),
            ('GS10', 'TB3MS', 'YC_10Y_3M'),
            ('GS5', 'GS2', 'YC_5Y_2Y'),
            ('GS2', 'TB3MS', 'YC_2Y_3M'),
        ],
        'ted_spread': [
            ('TB3MS', 'FEDFUNDS', 'TED_Spread_Proxy'),
        ],
    }
    
    def __init__(self):
        """Initialize spread calculator."""
        self.spreads = self.STANDARD_SPREADS.copy()
    
    def calculate_spreads(self, df: pd.DataFrame,
                         categories: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Calculate spread features.
        
        Args:
            df: DataFrame with rate data
            categories: Categories to calculate
            
        Returns:
            DataFrame with spread features
        """
        if categories is None:
            categories = list(self.spreads.keys())
        
        outputs = []
        
        for category in categories:
            if category not in self.spreads:
                continue
            
            for long_rate, short_rate, name in self.spreads[category]:
                if long_rate not in df.columns or short_rate not in df.columns:
                    continue
                
                spread = df[long_rate] - df[short_rate]
                spread.name = name
                outputs.append(spread)
        
        return pd.concat(outputs, axis=1) if outputs else pd.DataFrame(index=df.index)
    
    def calculate_spread_changes(self, spreads_df: pd.DataFrame,
                                windows: List[int] = [1, 3, 6]) -> pd.DataFrame:
        """
        Calculate changes in spreads.
        
        Args:
            spreads_df: DataFrame with spread data
            windows: Change windows
            
        Returns:
            DataFrame with spread change features
        """
        changes = []
        
        for col in spreads_df.columns:
            for window in windows:
                change = spreads_df[col].diff(window)
                change.name = f"{col}_chg_{window}M"
                changes.append(change)
        
        return pd.concat(changes, axis=1) if changes else pd.DataFrame(index=spreads_df.index)


def generate_all_ratio_features(df: pd.DataFrame,
                               include_changes: bool = True,
                               change_windows: List[int] = [1, 3, 6, 12]) -> pd.DataFrame:
    """
    Convenience function to generate all ratio features.
    
    Args:
        df: Raw FRED-MD data (levels)
        include_changes: Whether to include ratio changes
        change_windows: Windows for change calculations
        
    Returns:
        DataFrame with all ratio features
    """
    # Generate macro ratios
    ratio_gen = MacroRatioGenerator()
    ratios = ratio_gen.generate_ratios(df)
    
    # Generate spreads
    spread_calc = SpreadCalculator()
    spreads = spread_calc.calculate_spreads(df)
    
    # Combine
    all_features = pd.concat([ratios, spreads], axis=1)
    
    if include_changes:
        # Add ratio changes
        ratio_changes = ratio_gen.generate_ratio_changes(ratios, change_windows)
        spread_changes = spread_calc.calculate_spread_changes(spreads, change_windows[:3])
        
        all_features = pd.concat([all_features, ratio_changes, spread_changes], axis=1)
    
    logger.info(f"Generated {len(all_features.columns)} total ratio/spread features")
    return all_features


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Create sample data
    np.random.seed(42)
    n = 100
    dates = pd.date_range('2015-01-01', periods=n, freq='ME')
    
    df = pd.DataFrame({
        'M2SL': np.exp(np.cumsum(np.random.randn(n) * 0.01) + np.log(10000)),
        'GDP': np.exp(np.cumsum(np.random.randn(n) * 0.005) + np.log(15000)),
        'GS10': 2 + np.cumsum(np.random.randn(n) * 0.1),
        'GS1': 1 + np.cumsum(np.random.randn(n) * 0.05),
        'CPIAUCSL': np.exp(np.cumsum(np.random.randn(n) * 0.002) + np.log(250)),
        'BAA': 4 + np.cumsum(np.random.randn(n) * 0.1),
        'AAA': 3 + np.cumsum(np.random.randn(n) * 0.08),
    }, index=dates)
    
    # Generate ratios
    ratio_gen = MacroRatioGenerator()
    ratios = ratio_gen.generate_ratios(df)
    print("Generated ratios:")
    print(ratios.columns.tolist())
    print(ratios.head())
