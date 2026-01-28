"""
ALFRED Vintage Loader
=====================
Handles loading of historical point-in-time "vintages" from FRED-MD.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class ALFREDVintageLoader:
    """
    Specialized loader for point-in-time data retrieval from stored vintages.
    """
    
    # Map common modern FRED-MD names to older names found in early vintages
    SERIES_ALIASES = {
        'VIXCLSx': 'VXOCLSx',
    }
    
    def __init__(self, vintages_dir: str = "data/raw/vintages"):
        """
        Initialize with path to vintage CSVs.
        """
        self.vintages_dir = Path(vintages_dir)
        if not self.vintages_dir.exists():
            logger.warning(f"Vintages directory {vintages_dir} does not exist.")

    def get_available_vintages(self) -> List[pd.Timestamp]:
        """
        Scan directory and return list of available vintage dates.
        Expected format: YYYY-MM.csv
        """
        if not self.vintages_dir.exists():
            return []
            
        vintages = []
        for file in self.vintages_dir.glob("*.csv"):
            try:
                # Expecting YYYY-MM.csv or similar
                date_str = file.stem
                # Try to parse YYYY-MM
                date = pd.to_datetime(date_str, format='%Y-%m')
                vintages.append(date)
            except ValueError:
                # Handle other naming conventions if necessary, e.g., current.csv or specific dates
                try:
                    date = pd.to_datetime(date_str)
                    vintages.append(date)
                except ValueError:
                    logger.debug(f"Skipping non-date file: {file.name}")
                    
        return sorted(vintages)

    def load_vintage(self, vintage_date: pd.Timestamp) -> pd.DataFrame:
        """
        Load specific vintage. 
        CRITICAL: Handle '1-month publication lag'.
        If vintage_date is '2015-06-30', we look for '2015-06.csv'.
        The data inside will likely end in 2015-05 (May).
        """
        # Format as YYYY-MM to find the file
        vintage_filename = vintage_date.strftime('%Y-%m') + ".csv"
        file_path = self.vintages_dir / vintage_filename
        
        if not file_path.exists():
            available = self.get_available_vintages()
            if not available:
                raise FileNotFoundError(f"No vintages found in {self.vintages_dir}")
            
            # Find the latest available vintage on or before vintage_date
            candidates = [v for v in available if v <= vintage_date]
            if not candidates:
                raise FileNotFoundError(f"No vintage found on or before {vintage_date}")
            
            target_vintage = candidates[-1]
            file_path = self.vintages_dir / (target_vintage.strftime('%Y-%m') + ".csv")
            logger.info(f"Exact vintage {vintage_filename} not found. using closest: {file_path.name}")

        logger.info(f"Loading vintage from {file_path}")
        
        # FRED-MD CSVs have a special 2nd row (index 1) for transformations
        # We read headers first
        header_df = pd.read_csv(file_path, nrows=1)
        
        # Read data skipping the 2nd row
        df = pd.read_csv(file_path, skiprows=[1])
        
        # Standardize index to 'sasdate'
        date_col = 'sasdate' if 'sasdate' in df.columns else 'Date'
        if date_col in df.columns:
            # Handle MM/DD/YYYY format often found in ALFRED
            df[date_col] = pd.to_datetime(df[date_col])
            df.set_index(date_col, inplace=True)
            
        # Apply Aliases
        for modern, old in self.SERIES_ALIASES.items():
            if old in df.columns and modern not in df.columns:
                df[modern] = df[old]
                
        return df

    def get_transform_codes(self, vintage_date: pd.Timestamp) -> Dict[str, int]:
        """
        Extract transform codes from the 2nd row of the vintage file.
        In FRED-MD files, the header is row 0 and the codes are in row 1.
        """
        vintage_filename = vintage_date.strftime('%Y-%m') + ".csv"
        file_path = self.vintages_dir / vintage_filename
        
        if not file_path.exists():
            # Fallback to get_available_vintages logic
            available = self.get_available_vintages()
            if not available: return {}
            candidates = [v for v in available if v <= vintage_date]
            if not candidates: return {}
            file_path = self.vintages_dir / (candidates[-1].strftime('%Y-%m') + ".csv")
            
        try:
            # pd.read_csv with nrows=1 uses row 0 as header and row 1 as the first data row
            df = pd.read_csv(file_path, nrows=1)
            row = df.iloc[0]
            
            transform_codes = {}
            for col in df.columns:
                if col.lower() in ['sasdate', 'date']: continue
                val = row[col]
                try:
                    # Clean up the value (could be float 5.0 or string "5")
                    if pd.isna(val): continue
                    transform_codes[col] = int(float(val))
                except (ValueError, TypeError):
                    continue
            return transform_codes
        except Exception as e:
            logger.warning(f"Error reading transform codes from {file_path}: {e}")
            return {}
