"""
Baseline Model Evaluation Script - Day 3
Runs AutoARIMA and ETS on all 50 CiK tasks and computes numeric oracle

Usage:
    python run_baseline_evaluation.py
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add Benchmarks to path
sys.path.insert(0, str(Path(__file__).parent / "Benchmarks"))

from models.autoarima import AutoARIMAModel
from models.ets import ETSModel


def mean_absolute_error(y_true, y_pred):
    """Compute Mean Absolute Error"""
    return np.mean(np.abs(y_true - y_pred))


def load_dataset():
    """Load time series instances and contexts from datasets folder"""
    print("=" * 80)
    print("LOADING DATASET")
    print("=" * 80)
    
    datasets_dir = Path(__file__).parent / "datasets"
    
    # Load time series instances
    ts_df = pd.read_csv(datasets_dir / "ts_instances.csv")
    print(f"✓ Loaded {len(ts_df)} time series instances")
    
    # Load contexts
    with open(datasets_dir / "contexts.json", 'r') as f:
        contexts = json.load(f)
    print(f"✓ Loaded {len(contexts)} context descriptions")
    
    # Parse JSON arrays in history and future columns
    ts_df['history'] = ts_df['history'].apply(json.loads)
    ts_df['future'] = ts_df['future'].apply(json.loads)
    
    # Convert to numpy arrays
    ts_df['history'] = ts_df['history'].apply(np.array)
    ts_df['future'] = ts_df['future'].apply(np.array)
    
    print(f"\nDataset breakdown by type:")
    print(ts_df['type'].value_counts())
    print()
    
    return ts_df, contexts


def run_model_on_task(model, task_id, history, future, context=None):
    """
    Run a single model on a single task
    
    Returns:
        dict with predictions, actual, mae, and status
    """
    try:
        # Fit the model
        model.fit(history)
        
        # Make prediction (horizon = length of future)
        horizon = len(future)
        predictions = model.predict(history, horizon=horizon, context=context)
        
        # Extract mean prediction
        pred_mean = predictions['mean']
        
        # Compute MAE
        mae = mean_absolute_error(future, pred_mean)
        
        return {
            'task_id': task_id,
            'predictions': pred_mean,
            'actual': future,
            'mae': mae,
            'status': 'success'
        }
    
    except Exception as e:
        print(f"    ✗ Error on {task_id}: {str(e)[:100]}")
        return {
            'task_id': task_id,
            'predictions': None,
            'actual': future,
            'mae': np.nan,
            'status': f'error: {str(e)[:100]}'
        }


def run_autoarima_evaluation(ts_df):
    """Run AutoARIMA on all tasks"""
    print("=" * 80)
    print("RUNNING AutoARIMA")
    print("=" * 80)
    
    results = []
    
    for idx, row in tqdm(ts_df.iterrows(), total=len(ts_df), desc="AutoARIMA"):
        task_id = row['id']
        history = row['history']
        future = row['future']
        
        # Create fresh model for each task
        model = AutoARIMAModel(season_length=1, freq='H', name="AutoARIMA")
        
        # Run model
        result = run_model_on_task(model, task_id, history, future)
        results.append(result)
    
    # Create results dataframe
    results_df = pd.DataFrame([
        {
            'task_id': r['task_id'],
            'mae': r['mae'],
            'status': r['status'],
            'horizon': len(r['actual'])
        }
        for r in results
    ])
    
    # Print summary
    print(f"\n{'='*80}")
    print("AutoARIMA Summary:")
    print(f"  Total tasks: {len(results_df)}")
    print(f"  Successful: {(results_df['status'] == 'success').sum()}")
    print(f"  Failed: {(results_df['status'] != 'success').sum()}")
    print(f"  Mean MAE: {results_df['mae'].mean():.4f}")
    print(f"  Median MAE: {results_df['mae'].median():.4f}")
    print(f"  Std MAE: {results_df['mae'].std():.4f}")
    print(f"{'='*80}\n")
    
    return results_df


def run_ets_evaluation(ts_df):
    """Run ETS on all tasks"""
    print("=" * 80)
    print("RUNNING ETS")
    print("=" * 80)
    
    results = []
    
    for idx, row in tqdm(ts_df.iterrows(), total=len(ts_df), desc="ETS"):
        task_id = row['id']
        history = row['history']
        future = row['future']
        
        # Create fresh model for each task
        # Use simple additive trend/seasonal for stability
        model = ETSModel(
            seasonal_periods=1,
            trend='add',
            seasonal=None,
            damped_trend=False,
            freq='H',
            name="ETS"
        )
        
        # Run model
        result = run_model_on_task(model, task_id, history, future)
        results.append(result)
    
    # Create results dataframe
    results_df = pd.DataFrame([
        {
            'task_id': r['task_id'],
            'mae': r['mae'],
            'status': r['status'],
            'horizon': len(r['actual'])
        }
        for r in results
    ])
    
    # Print summary
    print(f"\n{'='*80}")
    print("ETS Summary:")
    print(f"  Total tasks: {len(results_df)}")
    print(f"  Successful: {(results_df['status'] == 'success').sum()}")
    print(f"  Failed: {(results_df['status'] != 'success').sum()}")
    print(f"  Mean MAE: {results_df['mae'].mean():.4f}")
    print(f"  Median MAE: {results_df['mae'].median():.4f}")
    print(f"  Std MAE: {results_df['mae'].std():.4f}")
    print(f"{'='*80}\n")
    
    return results_df


def compute_numeric_oracle(arima_results, ets_results):
    """Compute the numeric oracle - best of ARIMA/ETS per task"""
    print("=" * 80)
    print("COMPUTING NUMERIC ORACLE")
    print("=" * 80)
    
    # Merge results
    merged = arima_results.merge(
        ets_results,
        on='task_id',
        suffixes=('_arima', '_ets')
    )
    
    # Compute oracle (minimum MAE)
    merged['oracle_mae'] = merged[['mae_arima', 'mae_ets']].min(axis=1)
    merged['oracle_model'] = merged[['mae_arima', 'mae_ets']].idxmin(axis=1)
    merged['oracle_model'] = merged['oracle_model'].str.replace('mae_', '')
    
    oracle_df = merged[['task_id', 'oracle_mae', 'oracle_model']]
    
    # Print summary
    print(f"\nNumeric Oracle Summary:")
    print(f"  Mean Oracle MAE: {oracle_df['oracle_mae'].mean():.4f}")
    print(f"  Median Oracle MAE: {oracle_df['oracle_mae'].median():.4f}")
    print(f"\nBest model distribution:")
    print(oracle_df['oracle_model'].value_counts())
    print(f"{'='*80}\n")
    
    return oracle_df


def save_results(arima_results, ets_results, oracle_results):
    """Save all results to CSV files"""
    print("=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Save individual model results
    arima_path = results_dir / "arima_results.csv"
    arima_results.to_csv(arima_path, index=False)
    print(f"✓ Saved AutoARIMA results to {arima_path}")
    
    ets_path = results_dir / "ets_results.csv"
    ets_results.to_csv(ets_path, index=False)
    print(f"✓ Saved ETS results to {ets_path}")
    
    # Save oracle results
    oracle_path = results_dir / "numeric_oracle.csv"
    oracle_results.to_csv(oracle_path, index=False)
    print(f"✓ Saved Numeric Oracle to {oracle_path}")
    
    # Save combined results
    combined = arima_results[['task_id', 'mae']].rename(columns={'mae': 'mae_arima'})
    combined = combined.merge(
        ets_results[['task_id', 'mae']].rename(columns={'mae': 'mae_ets'}),
        on='task_id'
    )
    combined = combined.merge(
        oracle_results[['task_id', 'oracle_mae', 'oracle_model']],
        on='task_id'
    )
    
    combined_path = results_dir / "baseline_comparison.csv"
    combined.to_csv(combined_path, index=False)
    print(f"✓ Saved comparison to {combined_path}")
    
    print(f"{'='*80}\n")


def main():
    """Main evaluation pipeline"""
    print("\n")
    print("#" * 80)
    print("#" + " " * 78 + "#")
    print("#" + "  BASELINE MODEL EVALUATION - DAY 3".center(78) + "#")
    print("#" + "  AutoARIMA & ETS on 50 CiK Tasks".center(78) + "#")
    print("#" + " " * 78 + "#")
    print("#" * 80)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Load dataset
    ts_df, contexts = load_dataset()
    
    # Run AutoARIMA
    arima_results = run_autoarima_evaluation(ts_df)
    
    # Run ETS
    ets_results = run_ets_evaluation(ts_df)
    
    # Compute numeric oracle
    oracle_results = compute_numeric_oracle(arima_results, ets_results)
    
    # Save all results
    save_results(arima_results, ets_results, oracle_results)
    
    # Final summary
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"{'Model':<20} {'Mean MAE':<15} {'Median MAE':<15} {'Success Rate':<15}")
    print("-" * 80)
    
    arima_success = (arima_results['status'] == 'success').mean() * 100
    ets_success = (ets_results['status'] == 'success').mean() * 100
    
    print(f"{'AutoARIMA':<20} {arima_results['mae'].mean():<15.4f} "
          f"{arima_results['mae'].median():<15.4f} {arima_success:<15.1f}%")
    print(f"{'ETS':<20} {ets_results['mae'].mean():<15.4f} "
          f"{ets_results['mae'].median():<15.4f} {ets_success:<15.1f}%")
    print(f"{'Numeric Oracle':<20} {oracle_results['oracle_mae'].mean():<15.4f} "
          f"{oracle_results['oracle_mae'].median():<15.4f} {'100.0':<15}%")
    print("=" * 80)
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n✓ Day 3 baseline evaluation complete!")
    print(f"\nResults saved to: {Path(__file__).parent / 'results'}")
    print()


if __name__ == "__main__":
    main()