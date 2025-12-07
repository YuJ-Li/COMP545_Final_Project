"""
Baseline Model Evaluation Script - Day 3
Runs AutoARIMA and ETS on all 50 CiK tasks and computes numeric oracle

Metrics:
- MAE (Mean Absolute Error) - absolute scale
- nMAE (Normalized MAE) - scale-independent
- DA (Directional Accuracy) - trend prediction accuracy

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


def normalized_mae(y_true, y_pred):
    """
    Compute Normalized MAE
    nMAE = MAE / mean(|actual|)
    """
    mean_actual = np.mean(np.abs(y_true))
    if mean_actual == 0:
        return np.nan
    
    mae = mean_absolute_error(y_true, y_pred)
    return mae / mean_actual


def directional_accuracy(y_true, y_pred, last_value):
    """
    Compute Directional Accuracy
    """
    if len(y_true) == 0 or len(y_pred) == 0:
        return np.nan
    
    # Direction of actual change from last value
    actual_direction = np.sign(y_true - last_value)
    
    # Direction of predicted change from last value
    pred_direction = np.sign(y_pred - last_value)
    
    # Count correct directions (ignoring cases where actual change is 0)
    non_zero_mask = actual_direction != 0
    
    if non_zero_mask.sum() == 0:
        return np.nan  # All values same as last_value
    
    correct = (actual_direction[non_zero_mask] == pred_direction[non_zero_mask]).sum()
    total = non_zero_mask.sum()
    
    return correct / total


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
    
    print(f"\nDataset breakdown by domain:")
    if 'domain' in ts_df.columns:
        print(ts_df['domain'].value_counts().sort_index())
    elif 'type' in ts_df.columns:
        print(ts_df['type'].value_counts())
    else:
        print("No domain or type column found")
    print()
    
    return ts_df, contexts


def run_model_on_task(model, task_id, history, future, context=None):
    """
    Run a single model on a single task
    
    Returns:
        dict with predictions, actual, metrics (MAE, nMAE, DA), and status
    """
    try:
        # Fit the model
        model.fit(history)
        
        # Make prediction (horizon = length of future)
        horizon = len(future)
        predictions = model.predict(history, horizon=horizon, context=context)
        
        # Extract mean prediction
        pred_mean = predictions['mean']
        
        # Compute metrics
        mae = mean_absolute_error(future, pred_mean)
        nmae = normalized_mae(future, pred_mean)
        
        # Compute Directional Accuracy
        last_value = history[-1]
        da = directional_accuracy(future, pred_mean, last_value)
        
        return {
            'task_id': task_id,
            'predictions': pred_mean,
            'actual': future,
            'mae': mae,
            'nmae': nmae,
            'da': da,
            'status': 'success'
        }
    
    except Exception as e:
        print(f"    ✗ Error on {task_id}: {str(e)[:100]}")
        return {
            'task_id': task_id,
            'predictions': None,
            'actual': future,
            'mae': np.nan,
            'nmae': np.nan,
            'da': np.nan,
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
            'nmae': r['nmae'],
            'da': r['da'],
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
    print(f"\n  Metrics (mean ± std):")
    print(f"    MAE:   {results_df['mae'].mean():8.2f} ± {results_df['mae'].std():6.2f}")
    print(f"    nMAE:  {results_df['nmae'].mean():8.4f} ± {results_df['nmae'].std():6.4f}")
    print(f"    DA:    {results_df['da'].mean():8.4f} ± {results_df['da'].std():6.4f}")
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
            'nmae': r['nmae'],
            'da': r['da'],
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
    print(f"\n  Metrics (mean ± std):")
    print(f"    MAE:   {results_df['mae'].mean():8.2f} ± {results_df['mae'].std():6.2f}")
    print(f"    nMAE:  {results_df['nmae'].mean():8.4f} ± {results_df['nmae'].std():6.4f}")
    print(f"    DA:    {results_df['da'].mean():8.4f} ± {results_df['da'].std():6.4f}")
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
    
    # Get all metrics for the oracle model (from whichever model had best MAE)
    merged['oracle_nmae'] = merged.apply(
        lambda row: row['nmae_arima'] if row['oracle_model'] == 'arima' else row['nmae_ets'],
        axis=1
    )
    merged['oracle_da'] = merged.apply(
        lambda row: row['da_arima'] if row['oracle_model'] == 'arima' else row['da_ets'],
        axis=1
    )
    
    oracle_df = merged[['task_id', 'oracle_mae', 'oracle_nmae', 'oracle_da', 'oracle_model']]
    
    # Print summary
    print(f"\nNumeric Oracle Summary:")
    print(f"  Metrics (mean ± std):")
    print(f"    MAE:   {oracle_df['oracle_mae'].mean():8.2f} ± {oracle_df['oracle_mae'].std():6.2f}")
    print(f"    nMAE:  {oracle_df['oracle_nmae'].mean():8.4f} ± {oracle_df['oracle_nmae'].std():6.4f}")
    print(f"    DA:    {oracle_df['oracle_da'].mean():8.4f} ± {oracle_df['oracle_da'].std():6.4f}")
    print(f"\n  Best model distribution:")
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
    
    # Save combined results with all metrics
    combined = arima_results[['task_id', 'mae', 'nmae', 'da']].rename(
        columns={
            'mae': 'mae_arima',
            'nmae': 'nmae_arima',
            'da': 'da_arima'
        }
    )
    combined = combined.merge(
        ets_results[['task_id', 'mae', 'nmae', 'da']].rename(
            columns={
                'mae': 'mae_ets',
                'nmae': 'nmae_ets',
                'da': 'da_ets'
            }
        ),
        on='task_id'
    )
    combined = combined.merge(
        oracle_results[['task_id', 'oracle_mae', 'oracle_nmae', 'oracle_da', 'oracle_model']],
        on='task_id'
    )
    
    combined_path = results_dir / "baseline_comparison.csv"
    combined.to_csv(combined_path, index=False)
    print(f"✓ Saved comparison to {combined_path}")
    
    print(f"{'='*80}\n")


def analyze_by_domain(results_dir):
    """Analyze results by domain to understand scale effects"""
    print("=" * 80)
    print("DOMAIN-LEVEL ANALYSIS")
    print("=" * 80)
    
    try:
        # Load results and metadata
        combined = pd.read_csv(results_dir / "baseline_comparison.csv")
        metadata = pd.read_csv(Path(__file__).parent / "datasets" / "task_metadata.csv")
        
        # Merge with domain info
        if 'domain' in metadata.columns:
            merged = combined.merge(
                metadata[['id', 'domain', 'mean', 'std']], 
                left_on='task_id', 
                right_on='id'
            )
            
            # Group by domain
            domain_stats = merged.groupby('domain').agg({
                'mae_arima': 'mean',
                'nmae_arima': 'mean',
                'da_arima': 'mean',
                'mean': 'mean',
                'std': 'mean'
            }).round(4)
            
            domain_stats = domain_stats.sort_values('mae_arima', ascending=False)
            
            print("\nMetrics by Domain (sorted by MAE):")
            print("="*80)
            print(domain_stats.to_string())
            print("\n✓ High MAE domains likely have large-scale values")
            print("✓ nMAE normalizes these differences - compare nMAE across domains!")
            print(f"{'='*80}\n")
            
    except Exception as e:
        print(f"Could not perform domain analysis: {e}\n")


def main():
    """Main evaluation pipeline"""
    print("\n")
    print("#" * 80)
    print("#" + " " * 78 + "#")
    print("#" + "  BASELINE MODEL EVALUATION - DAY 3".center(78) + "#")
    print("#" + "  AutoARIMA & ETS on 50 CiK Tasks".center(78) + "#")
    print("#" + "  Metrics: MAE, nMAE, DA".center(78) + "#")
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
    results_dir = Path(__file__).parent / "results"
    save_results(arima_results, ets_results, oracle_results)
    
    # Analyze by domain
    analyze_by_domain(results_dir)
    
    # Final summary table
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"\n{'Model':<20} {'MAE':<15} {'nMAE':<15} {'DA':<15}")
    print("-" * 80)
    
    arima_success = (arima_results['status'] == 'success').mean() * 100
    ets_success = (ets_results['status'] == 'success').mean() * 100
    
    print(f"{'AutoARIMA':<20} "
          f"{arima_results['mae'].mean():<15.2f} "
          f"{arima_results['nmae'].mean():<15.4f} "
          f"{arima_results['da'].mean():<15.4f}")
    
    print(f"{'ETS':<20} "
          f"{ets_results['mae'].mean():<15.2f} "
          f"{ets_results['nmae'].mean():<15.4f} "
          f"{ets_results['da'].mean():<15.4f}")
    
    print(f"{'Numeric Oracle':<20} "
          f"{oracle_results['oracle_mae'].mean():<15.2f} "
          f"{oracle_results['oracle_nmae'].mean():<15.4f} "
          f"{oracle_results['oracle_da'].mean():<15.4f}")
    
    print("=" * 80)
    
    print(f"\n💡 METRICS EXPLAINED:")
    print(f"   MAE:  Absolute error - shows real-world impact")
    print(f"   nMAE: Normalized error - enables cross-domain comparison")
    print(f"   DA:   Directional accuracy - trend prediction correctness")
    
    print(f"\n✓ Success rates:")
    print(f"   AutoARIMA: {arima_success:.1f}%")
    print(f"   ETS: {ets_success:.1f}%")
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n✓ Day 3 baseline evaluation complete!")
    print(f"\nResults saved to: {results_dir}")
    print()


if __name__ == "__main__":
    main()
