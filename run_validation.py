#!/usr/bin/env python3
"""
ALFRED Validation CLI Runner
============================
Orchestrates the historical simulation for a given asset and model.
"""

import argparse
import logging
import sys
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# Add src to path
# Add src and root to path
root_dir = Path(__file__).parent
sys.path.insert(0, str(root_dir))
sys.path.insert(0, str(root_dir / "src"))

from src.preprocessing.data_loader import AssetPriceLoader
from src.evaluation.alfred_validation import ALFREDValidator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)

def generate_validation_dates(start_year: int = 2000, end_year: int = 2024):
    """Generate semi-annual validation dates."""
    dates = []
    for year in range(start_year, end_year + 1):
        dates.append(pd.Timestamp(f"{year}-06-01"))
        dates.append(pd.Timestamp(f"{year}-12-01"))
    return [d for d in dates if d <= pd.Timestamp.now()]

def run_single_validation(asset, model_name, start_year, returns=None, tournament_results=None):
    """Execute validation for a single asset/model pair."""
    asset = asset.upper()
    model_name = model_name.lower()
    model_path = f"experiments/models/{asset}_{model_name}.pkl"
    
    if not Path(model_path).exists():
        logger.error(f"Model file not found: {model_path}")
        return None

    # Get Revised IC
    revised_ic = 0.0
    if tournament_results is not None:
        match = tournament_results[(tournament_results['asset'] == asset) & (tournament_results['model'] == model_name)]
        if not match.empty:
            revised_ic = match.iloc[0]['IC_mean']

    # Initialize Validator
    validation_dates = generate_validation_dates(start_year)
    validator = ALFREDValidator(model_path=model_path, validation_dates=validation_dates)
    
    # Load returns if not provided
    if returns is None:
        asset_loader = AssetPriceLoader(data_dir="data/raw")
        prices = asset_loader.load_extended_prices(asset)
        horizon = 24
        returns = np.log(prices.shift(-horizon) / prices)
        returns.index = returns.index.to_period('M').to_timestamp()

    # Run Loop
    validation_results = validator.run_validation(returns)
    metrics = validator.compute_metrics(validation_results, revised_ic)
    
    # Check thresholds
    ic_ok = metrics['RealTime_IC'] > 0.15
    risk_ok = metrics['Revision_Risk'] < 0.30
    
    # Save Report
    report_dir = Path("experiments/reports/validation")
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / f"{asset}_{model_name}_validation_report.txt"
    
    with open(report_path, 'w') as f:
        f.write(f"ALFRED REAL-TIME VALIDATION REPORT\n")
        f.write(f"==================================\n")
        f.write(f"Asset: {asset}\n")
        f.write(f"Model: {model_name}\n")
        f.write(f"Validation Period: {start_year}-2024\n\n")
        f.write(f"Performance Metrics:\n")
        f.write(f"Revised IC:  {revised_ic:.4f}\n")
        f.write(f"Real-Time IC: {metrics['RealTime_IC']:.4f}\n")
        f.write(f"Revision Risk: {metrics['Revision_Risk']:.2%}\n\n")
        f.write(f"Verdict: {'DEPLOYABLE' if ic_ok and risk_ok else 'REJECT'}\n")

    return {
        'asset': asset,
        'model': model_name,
        'revised_ic': revised_ic,
        'rt_ic': metrics['RealTime_IC'],
        'rev_risk': metrics['Revision_Risk'],
        'status': 'PASS' if ic_ok and risk_ok else 'FAIL'
    }

def main():
    parser = argparse.ArgumentParser(description='ALFRED Real-Time Validation System')
    parser.add_argument('--asset', type=str, help='Asset to validate (default: ALL)')
    parser.add_argument('--model', type=str, help='Model name (default: ALL)')
    parser.add_argument('--start-year', type=int, default=2000, help='Start year for validation')
    
    args = parser.parse_args()
    
    # Load tournament results once for batch mode
    results_path = "experiments/cv_results/tournament_results.csv"
    tournament_results = pd.read_csv(results_path) if Path(results_path).exists() else None
    
    models_to_run = []
    if args.asset and args.model:
        models_to_run.append((args.asset, args.model))
    else:
        # Batch mode: scan directory
        model_dir = Path("experiments/models")
        for pkl in model_dir.glob("*.pkl"):
            # Expected name format: {ASSET}_{MODEL}.pkl
            name = pkl.stem
            if "_" in name:
                parts = name.split("_")
                asset = parts[0]
                model = "_".join(parts[1:])
                
                # Filter if needed
                if args.asset and asset.upper() != args.asset.upper(): continue
                if args.model and model.lower() != args.model.lower(): continue
                
                models_to_run.append((asset, model))
    
    if not models_to_run:
        logger.error("No models found matching criteria.")
        sys.exit(1)
        
    logger.info(f"Running validation for {len(models_to_run)} models...")
    
    summary_results = []
    for asset, model in models_to_run:
        logger.info(f"--- Validating {asset} / {model} ---")
        res = run_single_validation(asset, model, args.start_year, tournament_results=tournament_results)
        if res:
            summary_results.append(res)
            
    # Save batch summary
    if len(summary_results) > 1:
        summary_df = pd.DataFrame(summary_results)
        summary_path = Path("experiments/reports/validation/batch_validation_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        logger.info(f"Batch validation complete. Summary saved to {summary_path}")
        print("\nBatch Validation Summary:")
        print(summary_df[['asset', 'model', 'rt_ic', 'rev_risk', 'status']].to_string(index=False))

if __name__ == '__main__':
    main()
