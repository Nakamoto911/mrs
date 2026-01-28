"""
Data Acquisition Module
=======================
Independent script to fetch FRED-MD (current & historical) and asset data
(Yahoo Finance + FRED Proxies) and save them to human-readable CSVs.

To be run as a standalone step before the main pipeline.
"""

import os
import argparse
import logging
import pandas as pd
import numpy as np
import requests
# import yfinance as yf
from yahooquery import Ticker
from dotenv import load_dotenv
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from io import StringIO
import re



# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Fix yfinance cache issue by setting local cache dir
# yfinance removed
# try:
#     import tempfile
#     cache_dir = Path(tempfile.gettempdir()) / "yfinance_cache"
#     cache_dir.mkdir(parents=True, exist_ok=True)
#     yf.set_tz_cache_location(str(cache_dir))
# except Exception as e:
#     logger.warning(f"Failed to set yfinance cache: {e}")


class DataAcquisitionOrchestrator:
    """
    Orchestrates the data acquisition process:
    1. Fetch FRED-MD current vintage
    2. Fetch FRED-MD historical vintages (optional)
    3. Fetch Asset Prices (Modern + Historical Proxies)
    4. Save everything to human-readable CSVs
    """
    
    FRED_MD_URL = "https://www.stlouisfed.org/-/media/project/frbstl/stlouisfed/research/fred-md/monthly/2025-12-md.csv?sc_lang=en&hash=14BCC7AA1D5AB89D3459B69B8AE67D10"
    RESEARCH_PAGE_URL = "https://www.stlouisfed.org/research/economists/mccracken/fred-databases"
    
    ASSETS = {
        'SPX': {
            'name': 'S&P 500',
            'modern_ticker': 'SPY',
            'modern_start': '1993-01-01',
            'proxy_col': 'S&P 500', 
            'dividend_series': 'SPDYLD',
        },
        'BOND': {
            'name': '10Y Treasury Bond',
            'modern_ticker': 'IEF',
            'modern_start': '2002-07-01',
            'proxy_col': 'GS10', 
            'duration': 7.5,
        },
        'GOLD': {
            'name': 'Gold',
            'modern_ticker': 'GLD',
            'modern_start': '2004-11-01',
            'proxy_col': 'PPICMM', 
            'cpi_series': 'CPIAUCSL',
        }
    }
    
    def __init__(self, output_dir: str = "data/raw"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.vintages_dir = self.output_dir / "vintages"
        self.vintages_dir.mkdir(exist_ok=True)
        self.fred_md_path = self.output_dir / "fred_md.csv"
        

    def _discover_current_fred_md_url(self) -> str:
        """Scrape the McCracken research page to find the latest current.csv URL."""
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36",
            "Referer": self.RESEARCH_PAGE_URL
        }
        try:
            logger.info(f"Discovering latest FRED-MD URL from {self.RESEARCH_PAGE_URL}...")
            response = requests.get(self.RESEARCH_PAGE_URL, headers=headers, timeout=30)
            response.raise_for_status()
            
            # Pattern: href="([^"]+)"[^>]*>current.csv</a>
            matches = re.findall(r'href="([^"]+)"[^>]*>current\.csv</a>', response.text)
            if matches:
                url = matches[0]
                if url.startswith('/'):
                    url = "https://research.stlouisfed.org" + url
                logger.info(f"Discovered URL: {url}")
                return url
            
            # Fallback to monthly pattern YYYY-MM-md.csv
            monthly_matches = re.findall(r'href="([^"]+/monthly/\d{4}-\d{2}-md\.csv[^"]*)"', response.text)
            if monthly_matches:
                url = monthly_matches[0]
                if url.startswith('/'):
                    url = "https://research.stlouisfed.org" + url
                logger.info(f"Discovered monthly URL: {url}")
                return url
                
        except Exception as e:
            logger.warning(f"URL discovery failed: {e}. Using default URL.")
            
        return self.FRED_MD_URL

    def fetch_fred_md_current(self) -> pd.DataFrame:
        """Download and save the current FRED-MD vintage."""
        url = self._discover_current_fred_md_url()
        logger.info(f"Downloading FRED-MD from {url}...")
        
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        try:
            response = requests.get(url, headers=headers, timeout=60)
            response.raise_for_status()
            content = response.text
        except Exception as e:
            logger.error(f"Failed to download FRED-MD: {e}")
            raise

        # Save Raw Content first (safety)
        timestamp = datetime.now().strftime("%Y%m%d")
        raw_path = self.vintages_dir / f"fred_md_raw_{timestamp}.csv"
        with open(raw_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Parse for cleaner CSV
        lines = content.strip().split('\n')
        header = lines[0].split(',')
        transform_line = lines[1].split(',')
        data_lines = lines[2:]
        
        # Create a DataFrame
        df = pd.read_csv(StringIO('\n'.join([lines[0]] + data_lines)))
        
        # We want to keep the transform codes in a separate file or metadata
        # But for 'human readable csv', let's just save the cleaned data
        # and maybe a separate transforms.csv
        
        # Save Transforms
        transforms = pd.DataFrame([transform_line], columns=header)
        transforms.to_csv(self.output_dir / "fred_md_transforms.csv", index=False)
        
        # Save Data
        df.to_csv(self.output_dir / "fred_md.csv", index=False)
        logger.info(f"Saved FRED-MD to {self.output_dir / 'fred_md.csv'}")
        
        return df

    def fetch_fred_md_history(self) -> None:
        """Fetch historical vintages of FRED-MD."""
        # For this prototype, we will simulate or fetch a few key vintages if available online
        # However, McCracken website mainly hosts 'current.csv' and specific monthly files YYYY-MM.csv
        # We can try to download past months based on the pattern discovered.
        
        logger.info("Fetching historical FRED-MD vintages...")
        # Try to download the last 12 months as "vintages"
        current_date = datetime.now()
        
        base_url = "https://files.stlouisfed.org/files/htdocs/fred-md/monthly/"
        # Vintages are usually stored as YYYY-MM.csv
        
        for i in range(12):
            date = current_date - timedelta(days=30*i)
            str_date = date.strftime("%Y-%m")
            url = f"{base_url}{str_date}.csv"
            
            save_path = self.vintages_dir / f"{str_date}.csv"
            if save_path.exists():
                continue
                
            try:
                # logger.info(f"Trying vintage {str_date}...")
                response = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
                if response.status_code == 200:
                    with open(save_path, 'wb') as f:
                        f.write(response.content)
                    # logger.info(f"Downloaded vintage {str_date}")
            except Exception:
                pass


            
        return pd.Series()

    def _load_fred_md_proxies(self) -> pd.DataFrame:
        """Load and calculate proxy series from the local FRED-MD csv."""
        if not self.fred_md_path.exists():
             logger.warning("FRED-MD file not found. Cannot load proxies.")
             return pd.DataFrame()
             
        try:
            # Read FRED-MD, skipping the second row (transform codes)
            df = pd.read_csv(self.fred_md_path)
            # FRED-MD usually has a transform row at index 0 (line 2)
            # We want to keep correct headers, but skip that row
            # If 'sasdate' is in columns, transformations are likely in row 0
            if 'sasdate' in df.columns:
                 df = df.iloc[1:]
                 date_col = 'sasdate'
            else:
                 date_col = 'Unnamed: 0'
                 
            df['date'] = pd.to_datetime(df[date_col], errors='coerce', utc=True)
            df.dropna(subset=['date'], inplace=True)
            df.set_index('date', inplace=True)
            # Make index timezone naive to match yahoo
            df.index = df.index.tz_localize(None)
            
            proxies = pd.DataFrame(index=df.index)
            
            # 1. SPX Proxy (S&P 500)
            if 'S&P 500' in df.columns:
                 proxies['SPX'] = pd.to_numeric(df['S&P 500'], errors='coerce')
            
            # 2. BOND Proxy (Synthetic from GS10)
            if 'GS10' in df.columns:
                yields = pd.to_numeric(df['GS10'], errors='coerce') / 100
                duration = 7.5
                carry = yields.shift(1) / 12
                # Bond Price Change ~= -Duration * Change in Yield
                price_change = -duration * (yields - yields.shift(1))
                bond_ret = (carry + price_change).fillna(0)
                # This is monthly return. Construct Price Index.
                proxies['BOND'] = (1 + bond_ret).cumprod() * 100
                
            # 3. GOLD Proxy (PPICMM)
            if 'PPICMM' in df.columns:
                proxies['GOLD'] = pd.to_numeric(df['PPICMM'], errors='coerce')
                
            return proxies
            
        except Exception as e:
            logger.warning(f"Failed to load FRED-MD proxies: {e}")
            return pd.DataFrame()

    def fetch_asset_prices(self) -> None:
        """Fetch modern data from Yahoo and splice with historical proxies."""
        # Pre-load proxies from FRED-MD
        fred_proxies = self._load_fred_md_proxies()
        
        all_prices = {}
        
        for asset, config in self.ASSETS.items():
            logger.info(f"Processing {asset}...")
            
            # 1. Modern Data (YahooQuery)
            try:
                # Use yahooquery Ticker which is more robust
                t = Ticker(config['modern_ticker'], asynchronous=False)
                # Fetch history
                df_hist = t.history(start=config['modern_start'], interval='1d')
                
                if not df_hist.empty:
                    # yahooquery returns a multi-index (symbol, date) or just date if single symbol
                    # reset index to handle easily
                    df = df_hist.reset_index()
                    
                    # Ensure 'date' is datetime
                    if 'date' in df.columns:
                        df['date'] = pd.to_datetime(df['date'])
                        df = df.set_index('date')
                    elif 'index' in df.columns: # Sometimes it returns 'index' if not named
                        df['index'] = pd.to_datetime(df['index']) 
                        df = df.set_index('index')
                        
                    # Filter just adjclose
                    if 'adjclose' in df.columns:
                         modern = df['adjclose']
                    elif 'close' in df.columns:
                         modern = df['close']
                    else:
                         modern = pd.Series()
                    
                    # Resample to monthly end
                    modern = modern.resample('ME').last()
                    modern.name = asset
                else:
                    logger.warning(f"No modern data for {asset} (empty dataframe)")
                    modern = None
            except Exception as e:
                logger.warning(f"Failed to fetch Yahoo data for {asset}: {e}")
                modern = None
                
            # 2. Historical Proxy (Load from FRED-MD)
            historical = None
            if asset in fred_proxies.columns:
                historical = fred_proxies[asset]
                historical = historical.dropna()
                historical.name = asset
            else:
                 logger.warning(f"Proxy for {asset} not found in FRED-MD data.")
            
            # 3. Splice or Fallback
            final_series = None
            
            # (Fallback to FRED-MD logic is implicitly covered by historical proxy)

            # If we still rely on the historical proxy which comes from FRED and covers 'modern' times mostly
            # (e.g. SP500 on FRED is up to date).
            # So if modern is None, check if historical covers enough.
            
            if modern is not None and historical is not None:
                # Splice logic
                overlap_start = modern.index[0]
                logger.info(f"  Splice for {asset}: {overlap_start.strftime('%Y-%m')} (Modern data starts here, previous is FRED proxy)")
                
                if overlap_start in historical.index:
                    hist_val = historical.asof(overlap_start)
                    scale = modern.iloc[0] / hist_val
                    historical_scaled = historical * scale
                    
                    cut_date = overlap_start - pd.Timedelta(days=15)
                    part1 = historical_scaled[historical_scaled.index < overlap_start]
                    final_series = pd.concat([part1, modern])
                else:
                    final_series = modern 
            elif modern is not None:
                logger.info(f"  Using Yahoo modern data only for {asset} (no historical proxy)")
                final_series = modern
            elif historical is not None:
                logger.info(f"  Using FRED historical proxy only for {asset} (no modern data)")
                final_series = historical
            
            if final_series is not None:
                all_prices[asset] = final_series

        # Save to CSV
        if all_prices:
            # Align all to a common monthly index
            # Join outer to keep full history
            df_assets = pd.DataFrame(all_prices)
            # Forward fill for small gaps if any, but be careful
            df_assets = df_assets.sort_index()
            # Ensure index name
            df_assets.index.name = 'Date'
            
            out_path = self.output_dir / "assets.csv"
            df_assets.to_csv(out_path)
            logger.info(f"Saved asset prices to {out_path}")
        else:
            logger.warning("No asset prices collected!")

    def run(self):
        logger.info("Starting Data Acquisition...")
        self.fetch_fred_md_current()
        self.fetch_fred_md_history()
        self.fetch_asset_prices()
        logger.info("Data Acquisition Complete.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Acquire macro and asset data.")
    parser.add_argument("--output", type=str, default="data/raw", help="Output directory")
    args = parser.parse_args()
    
    orchestrator = DataAcquisitionOrchestrator(output_dir=args.output)
    orchestrator.run()
