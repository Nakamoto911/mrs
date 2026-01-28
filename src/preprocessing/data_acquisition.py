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
import yfinance as yf
from dotenv import load_dotenv
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional
from io import StringIO
import re

try:
    from fredapi import Fred
except ImportError:
    Fred = None

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Fix yfinance cache issue by setting local cache dir
try:
    import tempfile
    cache_dir = Path(tempfile.gettempdir()) / "yfinance_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    yf.set_tz_cache_location(str(cache_dir))
except Exception as e:
    logger.warning(f"Failed to set yfinance cache: {e}")


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
            'fred_series': 'SP500',
            'dividend_series': 'SPDYLD',
        },
        'BOND': {
            'name': '10Y Treasury Bond',
            'modern_ticker': 'IEF',
            'modern_start': '2002-07-01',
            'fred_series': 'GS10',
            'duration': 8.5,
        },
        'GOLD': {
            'name': 'Gold',
            'modern_ticker': 'GLD',
            'modern_start': '2004-11-01',
            'fred_series': 'GOLDAMGBD228NLBM',
            'cpi_series': 'CPIAUCSL',
        }
    }

    def __init__(self, output_dir: str = "data/raw", fred_api_key: Optional[str] = None):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.vintages_dir = self.output_dir / "vintages"
        self.vintages_dir.mkdir(exist_ok=True)
        
        self.fred_api_key = fred_api_key or os.getenv('FRED_API_KEY')
        self.fred = Fred(api_key=self.fred_api_key) if self.fred_api_key and Fred else None
        
        if not self.fred and Fred:
            logger.warning("FRED API key not found. Historical asset proxies requiring FRED might fail if not publicly available.")

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

    def _load_historical_proxy(self, asset: str, start: str = '1959-01-01') -> pd.Series:
        """Fetch historical proxy data from FRED."""
        if not self.fred:
            raise ValueError("FRED API key required for historical proxies")
            
        config = self.ASSETS[asset]
        logger.info(f"Fetching historical proxy for {asset}...")
        
        if asset == 'SPX':
            # S&P 500
            s = self.fred.get_series(config['fred_series'], observation_start=start)
            return s.resample('ME').last()
            
        elif asset == 'BOND':
            # Construct total return from yield
            yields = self.fred.get_series(config['fred_series'], observation_start=start)
            yields = yields.resample('ME').last() / 100
            
            duration = config['duration']
            yield_change = yields.diff()
            monthly_ret = yields/12 - duration * yield_change
            monthly_ret = monthly_ret.fillna(yields/12)
            
            price_index = (1 + monthly_ret).cumprod() * 100
            return price_index
            
        elif asset == 'GOLD':
            gold_start = '1968-04-01'
            gold = self.fred.get_series(config['fred_series'], observation_start=gold_start)
            gold = gold.resample('ME').last()
            
            if start < gold_start:
                cpi = self.fred.get_series(config['cpi_series'], observation_start=start, observation_end='1968-03-31')
                cpi = cpi.resample('ME').last()
                scale = gold.iloc[0] / cpi.iloc[-1]
                gold = pd.concat([cpi * scale, gold])
            return gold
            
        return pd.Series()

    def fetch_asset_prices(self) -> None:
        """Fetch modern data from Yahoo and splice with historical proxies."""
        all_prices = {}
        
        for asset, config in self.ASSETS.items():
            logger.info(f"Processing {asset}...")
            
            # 1. Modern Data
            try:
                modern = yf.download(config['modern_ticker'], start=config['modern_start'], progress=False)
                if not modern.empty:
                    modern = modern['Adj Close'].resample('ME').last()
                    modern.name = asset
                else:
                    logger.warning(f"No modern data for {asset}")
                    modern = None
            except Exception as e:
                logger.warning(f"Failed to fetch Yahoo data for {asset}: {e}")
                modern = None
                
            # 2. Historical Proxy
            historical = None
            try:
                if self.fred:
                    historical = self._load_historical_proxy(asset)
                    historical.name = asset
            except Exception as e:
                logger.warning(f"Failed to fetch historical proxy for {asset}: {e}")
            
            # 3. Splice or Fallback
            final_series = None
            
            # Fallback to FRED for modern data if Yahoo failed (since SP500, GS10 are on FRED)
            if modern is None and self.fred:
                logger.info(f"Falling back to FRED for modern data for {asset}...")
                try:
                    if asset == 'SPX':
                         # SP500 is daily on FRED
                         modern = self.fred.get_series('SP500', observation_start=config['modern_start'])
                         modern = modern.resample('ME').last()
                         modern.name = asset
                    elif asset == 'BOND':
                         # GS10 is monthly
                         modern_yields = self.fred.get_series('GS10', observation_start=config['modern_start'])
                         # We need price index. Re-use historical proxy logic but for modern period?
                         # Or just assume we have the proxy for the full period if we used FRED
                         pass 
                except Exception as e:
                     logger.warning(f"FRED fallback failed for {asset}: {e}")

            # If we still rely on the historical proxy which comes from FRED and covers 'modern' times mostly
            # (e.g. SP500 on FRED is up to date).
            # So if modern is None, check if historical covers enough.
            
            if modern is not None and historical is not None:
                # Splice logic
                overlap_start = modern.index[0]
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
                final_series = modern
            elif historical is not None:
                # Use historical as is (it might be full history if FRED has it)
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
    parser.add_argument("--fred-key", type=str, default=None, help="FRED API Key")
    args = parser.parse_args()
    
    # Load environment variables from .env file
    load_dotenv()
            
    # Priority: CLI arg > Env Var
    api_key = args.fred_key or os.getenv('FRED_API_KEY')
    
    orchestrator = DataAcquisitionOrchestrator(output_dir=args.output, fred_api_key=api_key)
    orchestrator.run()
