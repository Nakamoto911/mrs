"""
Data Loader Module
==================
Handles FRED-MD data acquisition, historical extension, and asset price loading.

Part of the Asset-Specific Macro Regime Detection System
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import warnings
import logging

try:
    from fredapi import Fred
except ImportError:
    Fred = None

try:
    import yfinance as yf
except ImportError:
    yf = None

import requests
from io import StringIO

logger = logging.getLogger(__name__)


class FREDMDLoader:
    """
    Loader for FRED-MD macroeconomic database.
    
    Handles downloading, parsing, and transformation code application.
    """
    
    # Stable download URL for FRED-MD current (Dec 2025) - retained for reference
    FRED_MD_URL = "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/monthly/2025-12-md.csv?sc_lang=en&hash=14BCC7AA1D5AB89D3459B69B8AE67D10"
    
    # Category mappings (Specifically Category 8 as requested)
    CATEGORY_MAPPING = {
        'S&P 500': 8,
        'S&P div yield': 8,
        'S&P PE ratio': 8,
        # 'S&P: indust': 8, # Alternative name in some vintages
    }
    
    def __init__(self, data_dir: str = "data/raw", fred_api_key: Optional[str] = None, config: Optional[dict] = None):
        """
        Initialize FRED-MD loader.
        
        Args:
            data_dir: Directory containing the acquired data (fred_md.csv)
            fred_api_key: Not used in this reader-only version
            config: Optional configuration dictionary
        """
        self.data_dir = Path(data_dir)
        self.config = config or {}
        self._raw_data = None
        self._transform_codes = None
        self._transformed_data = None

    def download_current_vintage(self, save: bool = False) -> pd.DataFrame:
        """
        Load the current FRED-MD vintage from local CSV.
        
        Args:
            save: Ignored (kept for compatibility)
            
        Returns:
            DataFrame with FRED-MD data
        """
        csv_path = self.data_dir / "fred_md.csv"
        transforms_path = self.data_dir / "fred_md_transforms.csv"
        
        if not csv_path.exists():
             raise FileNotFoundError(f"FRED-MD data not found at {csv_path}. Please run data_acquisition.py first.")

        logger.info(f"Loading FRED-MD from {csv_path}...")
        
        # Load Data
        df = pd.read_csv(csv_path)
        if 'sasdate' in df.columns:
            df['sasdate'] = pd.to_datetime(df['sasdate'])
            df.set_index('sasdate', inplace=True)
            
        # Apply Category/Variable Exclusions (Spec 01)
        fred_cfg = self.config.get('data', {}).get('fred_md', {})
        exclude_cats = fred_cfg.get('exclude_categories', [])
        exclude_vars = fred_cfg.get('exclude_variables', [])
        
        cols_to_drop = []
        for col in df.columns:
            # Check exclusions
            if col in exclude_vars:
                cols_to_drop.append(col)
                continue
            
            cat = self.CATEGORY_MAPPING.get(col)
            if cat in exclude_cats:
                cols_to_drop.append(col)
                
        if cols_to_drop:
            logger.info(f"Excluding {len(cols_to_drop)} variables based on configuration: {cols_to_drop}")
            df = df.drop(columns=cols_to_drop)

        # Load Transforms
        self._transform_codes = {}
        if transforms_path.exists():
            t_df = pd.read_csv(transforms_path)
            # The transforms csv should have columns matching the data (except date)
            # and one row of codes.
            for col in t_df.columns:
                try:
                    self._transform_codes[col] = int(t_df.iloc[0][col])
                except:
                    self._transform_codes[col] = 1
        else:
             # Fallback: try to infer or default
             logger.warning("Transforms file not found. Using default defaults (1).")
             for col in df.columns:
                 self._transform_codes[col] = 1
        
        self._raw_data = df
        logger.info(f"Loaded {len(df)} months, {len(df.columns)} variables")
        return df
    
    def get_transform_codes(self) -> Dict[str, int]:
        """Get transformation codes for each variable."""
        if self._transform_codes is None:
            raise ValueError("Data not loaded. Call download_current_vintage() first.")
        return self._transform_codes.copy()
    
    def apply_transformations(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Apply FRED-MD transformation codes to make series stationary.
        
        Args:
            df: DataFrame to transform (uses raw data if None)
            
        Returns:
            Transformed DataFrame
        """
        if df is None:
            if self._raw_data is None:
                raise ValueError("No data available. Call download_current_vintage() first.")
            df = self._raw_data.copy()
        
        if self._transform_codes is None:
            raise ValueError("Transform codes not available.")
        
        logger.info("Applying FRED-MD transformations...")
        
        transformed = pd.DataFrame(index=df.index)
        
        for col in df.columns:
            code = self._transform_codes.get(col, 1)
            series = df[col]
            
            try:
                if code == 1:
                    # No transformation
                    transformed[col] = series
                elif code == 2:
                    # First difference
                    transformed[col] = series.diff()
                elif code == 3:
                    # Second difference
                    transformed[col] = series.diff().diff()
                elif code == 4:
                    # Log
                    transformed[col] = np.log(series.replace(0, np.nan))
                elif code == 5:
                    # Log first difference (growth rate)
                    transformed[col] = np.log(series.replace(0, np.nan)).diff()
                elif code == 6:
                    # Log second difference
                    transformed[col] = np.log(series.replace(0, np.nan)).diff().diff()
                elif code == 7:
                    # Percent change
                    transformed[col] = series.pct_change()
                else:
                    logger.warning(f"Unknown transform code {code} for {col}, using no transform")
                    transformed[col] = series
            except Exception as e:
                logger.warning(f"Error transforming {col}: {e}")
                transformed[col] = np.nan
        
        self._transformed_data = transformed
        return transformed
    
    def get_stationary_levels(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Extract variables that are stationary in levels (transform code 1).
        
        These are needed for ratio construction before applying transformations.
        
        Args:
            df: Raw data DataFrame
            
        Returns:
            DataFrame with level-stationary variables
        """
        if df is None:
            if self._raw_data is None:
                raise ValueError("No data available.")
            df = self._raw_data.copy()
        
        if self._transform_codes is None:
            raise ValueError("Transform codes not available.")
        
        level_stationary_cols = [
            col for col, code in self._transform_codes.items() 
            if code == 1 and col in df.columns
        ]
        
        logger.info(f"Found {len(level_stationary_cols)} level-stationary variables")
        return df[level_stationary_cols]


class AssetPriceLoader:
    """
    Loader for asset prices with historical extension.
    
    Handles modern ETF data and historical proxy construction.
    """
    
    # Asset configuration
    ASSETS = {
        'SPX': {
            'name': 'S&P 500',
            'modern_ticker': 'SPY',
            'modern_start': '1993-01-01',
            'fred_series': 'S&P 500',
            'dividend_series': 'SPDYLD',
        },
        'BOND': {
            'name': '10Y Treasury Bond',
            'modern_ticker': 'IEF',
            'modern_start': '2002-07-01',
            'fred_series': 'GS10',
            'duration': 8.5,  # Approximate duration for 10Y
        },
        'GOLD': {
            'name': 'Gold',
            'modern_ticker': 'GLD',
            'modern_start': '2004-11-01',
            'fred_series': 'GOLDAMGBD228NLBM',
            'cpi_series': 'CPIAUCSL',  # For pre-1968 extension
        }
    }
    
    def __init__(self, data_dir: str = "data/raw", fred_api_key: Optional[str] = None):
        """
        Initialize asset price loader.
        
        Args:
            data_dir: Directory containing the acquired data (assets.csv)
            fred_api_key: Not used in this reader-only version
        """
        self.data_dir = Path(data_dir)
        self._all_prices = None

    def load_extended_prices(self, asset: str, start: str = '1959-01-01') -> pd.Series:
        """
        Load asset prices from local CSV.
        
        Args:
            asset: Asset code
            start: Start date filter
            
        Returns:
            Extended monthly price series
        """
        csv_path = self.data_dir / "assets.csv"
        
        if self._all_prices is None:
            if not csv_path.exists():
                raise FileNotFoundError(f"Asset data not found at {csv_path}. Please run data_acquisition.py first.")
            
            logger.info(f"Loading asset prices from {csv_path}...")
            df = pd.read_csv(csv_path)
            if 'Date' in df.columns:
                df['Date'] = pd.to_datetime(df['Date'])
                df.set_index('Date', inplace=True)
            elif 'sasdate' in df.columns: # fallback if users use different key
                df['sasdate'] = pd.to_datetime(df['sasdate'])
                df.set_index('sasdate', inplace=True)
            self._all_prices = df
            
        if asset not in self._all_prices.columns:
            raise ValueError(f"Asset {asset} not found in loaded data.")
            
        series = self._all_prices[asset].dropna()
        series = series[series.index >= start]
        return series        

    
    def compute_returns(self, prices: pd.Series, periods: int = 1) -> pd.Series:
        """
        Compute log returns.
        
        Args:
            prices: Price series
            periods: Number of periods for return calculation
            
        Returns:
            Log return series
        """
        returns = np.log(prices / prices.shift(periods))
        returns.name = f"{prices.name}_ret_{periods}M"
        return returns
    
    def compute_volatility(self, returns: pd.Series, window: int = 6) -> pd.Series:
        """
        Compute rolling realized volatility.
        
        Args:
            returns: Return series
            window: Rolling window in months
            
        Returns:
            Annualized volatility series
        """
        vol = returns.rolling(window=window).std() * np.sqrt(12)
        vol.name = f"{returns.name}_vol_{window}M"
        return vol


class ALFREDLoader:
    """
    Loader for ALFRED (Archival FRED) real-time vintage data.
    
    Used for validation of models with point-in-time data.
    """
    
    def __init__(self, data_dir: str = "data/alfred_vintages", fred_api_key: str = None):
        """
        Initialize ALFRED loader.
        
        Args:
            data_dir: Directory to store vintage data
            fred_api_key: FRED API key (required for ALFRED access)
        """
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        if fred_api_key is None:
            raise ValueError("FRED API key required for ALFRED access")
        
        if Fred is None:
            raise ImportError("fredapi package required for ALFRED access")
        
        self.fred = Fred(api_key=fred_api_key)
    
    def get_vintage_dates(self, start_year: int = 2000, end_year: int = 2024,
                          frequency: str = 'semi-annual') -> List[datetime]:
        """
        Generate list of vintage dates for validation.
        
        Args:
            start_year: Start year
            end_year: End year
            frequency: 'semi-annual' or 'quarterly'
            
        Returns:
            List of vintage dates
        """
        dates = []
        
        for year in range(start_year, end_year + 1):
            if frequency == 'semi-annual':
                # June 30 and December 31
                dates.extend([
                    datetime(year, 6, 30),
                    datetime(year, 12, 31)
                ])
            elif frequency == 'quarterly':
                # End of each quarter
                for month in [3, 6, 9, 12]:
                    dates.append(datetime(year, month, 
                                        28 if month == 2 else 30 if month in [4, 6, 9, 11] else 31))
        
        # Remove future dates
        dates = [d for d in dates if d <= datetime.now()]
        
        return dates
    
    def download_vintage(self, series_id: str, vintage_date: datetime,
                        publication_lag_months: int = 1) -> pd.Series:
        """
        Download a specific vintage of a FRED series.
        
        Args:
            series_id: FRED series ID
            vintage_date: Date of the vintage to download
            publication_lag_months: Assumed publication lag
            
        Returns:
            Series with point-in-time data
        """
        # Account for publication lag
        effective_date = vintage_date - timedelta(days=publication_lag_months * 30)
        vintage_str = vintage_date.strftime('%Y-%m-%d')
        
        try:
            data = self.fred.get_series(
                series_id,
                realtime_start=vintage_str,
                realtime_end=vintage_str
            )
            return data
        except Exception as e:
            logger.warning(f"Could not load vintage {vintage_str} for {series_id}: {e}")
            return pd.Series(dtype=float)
    
    def download_all_vintages(self, series_ids: List[str], 
                             vintage_dates: Optional[List[datetime]] = None,
                             save: bool = True) -> Dict[datetime, pd.DataFrame]:
        """
        Download all vintages for multiple series.
        
        Args:
            series_ids: List of FRED series IDs
            vintage_dates: List of vintage dates (defaults to semi-annual 2000-2024)
            save: Whether to save to disk
            
        Returns:
            Dictionary mapping vintage dates to DataFrames
        """
        if vintage_dates is None:
            vintage_dates = self.get_vintage_dates()
        
        vintages = {}
        
        for vdate in vintage_dates:
            logger.info(f"Downloading vintage {vdate.strftime('%Y-%m-%d')}...")
            
            vintage_data = {}
            for series_id in series_ids:
                data = self.download_vintage(series_id, vdate)
                if not data.empty:
                    vintage_data[series_id] = data
            
            if vintage_data:
                df = pd.DataFrame(vintage_data)
                vintages[vdate] = df
                
                if save:
                    filepath = self.data_dir / f"vintage_{vdate.strftime('%Y%m%d')}.parquet"
                    df.to_parquet(filepath)
        
        return vintages


def load_all_data(config: dict, fred_api_key: Optional[str] = None) -> Dict:
    """
    Convenience function to load all data based on configuration.
    
    Args:
        config: Configuration dictionary
        fred_api_key: Optional FRED API key
        
    Returns:
        Dictionary with all loaded data
    """
    result = {}
    
    # Load FRED-MD
    fred_loader = FREDMDLoader(
        data_dir=config.get('data_dir', 'data/raw'),
        fred_api_key=fred_api_key,
        config=config
    )
    result['fred_md_raw'] = fred_loader.download_current_vintage()
    result['fred_md_transformed'] = fred_loader.apply_transformations()
    result['stationary_levels'] = fred_loader.get_stationary_levels()
    result['transform_codes'] = fred_loader.get_transform_codes()
    
    # Load asset prices
    asset_loader = AssetPriceLoader(
        data_dir=config.get('data_dir', 'data/raw'), # Point to same raw dir by default
        fred_api_key=fred_api_key
    )
    
    result['asset_prices'] = {}
    result['asset_returns'] = {}
    result['asset_volatility'] = {}
    
    for asset in ['SPX', 'BOND', 'GOLD']:
        try:
            prices = asset_loader.load_extended_prices(asset)
            result['asset_prices'][asset] = prices
            
            returns = asset_loader.compute_returns(prices)
            result['asset_returns'][asset] = returns
            
            vol = asset_loader.compute_volatility(returns)
            result['asset_volatility'][asset] = vol
        except Exception as e:
            logger.warning(f"Could not load data for {asset}: {e}")
    
    return result


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)
    
    # Load FRED-MD
    loader = FREDMDLoader()
    df = loader.download_current_vintage()
    print(f"Loaded {len(df)} months, {len(df.columns)} variables")
    
    # Apply transformations
    transformed = loader.apply_transformations()
    print(f"Transformed data shape: {transformed.shape}")
