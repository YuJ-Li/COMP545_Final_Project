"""
Results Viewer - Quick Analysis of Baseline Evaluation

Usage:
    python view_results.py
"""

import pandas as pd
from pathlib import Path
import numpy as np


def load_results():
    """Load all result files"""
    results_dir = Path(__file__).parent / "results"
    
    if not results_dir.exists():
        print("❌ Results directory not found!")
        print("   Run 'python run_baseline_evaluation.py' first")
        return None
    
    # Check which files exist
    files = {
        'comparison': results_dir / "baseline_comparison.csv",
        'arima': results_dir / "arima_results.csv",
        'ets': results_dir / "ets_results.csv",
        'oracle': results_dir / "numeric_oracle.csv"
    }
    
    missing = [name for name, path in files.items() if not path.exists()]
    if missing:
        print(f"❌ Missing result files: {', '.join(missing)}")
        return None
    
    # Load all files
    results = {
        name: pd.read_csv(path) 
        for name, path in files.items()
    }
    
    return results


def display_summary(results):
    """Display high-level summary"""
    print("\n" + "="*80)
    print("BASELINE EVALUATION RESULTS")
    print("="*80)
    
    comp = results['comparison']
    
    # Overall statistics
    print("\n📊 OVERALL PERFORMANCE")
    print("-" * 80)
    print(f"{'Model':<20} {'Mean MAE':<15} {'Median MAE':<15} {'Min MAE':<15} {'Max MAE':<15}")
    print("-" * 80)
    
    for col in ['mae_arima', 'mae_ets', 'oracle_mae']:
        name = col.replace('mae_', '').upper()
        if name == 'ORACLE_MAE':
            name = 'NUMERIC ORACLE'
        
        print(f"{name:<20} {comp[col].mean():<15.2f} {comp[col].median():<15.2f} "
              f"{comp[col].min():<15.2f} {comp[col].max():<15.2f}")
    
    print("-" * 80)
    
    # Model selection
    print("\n🎯 ORACLE MODEL SELECTION")
    print("-" * 80)
    oracle_dist = results['oracle']['oracle_model'].value_counts()
    for model, count in oracle_dist.items():
        pct = (count / len(results['oracle'])) * 100
        print(f"  {model.upper():<15} {count:>3} tasks ({pct:>5.1f}%)")
    print("-" * 80)
    
    # Improvement analysis
    print("\n📈 ORACLE IMPROVEMENT OVER INDIVIDUAL MODELS")
    print("-" * 80)
    arima_vs_oracle = ((comp['mae_arima'] - comp['oracle_mae']) / comp['mae_arima'] * 100).mean()
    ets_vs_oracle = ((comp['mae_ets'] - comp['oracle_mae']) / comp['mae_ets'] * 100).mean()
    
    print(f"  Oracle vs AutoARIMA: {arima_vs_oracle:>6.2f}% better (on average)")
    print(f"  Oracle vs ETS:       {ets_vs_oracle:>6.2f}% better (on average)")
    print("-" * 80)


def display_task_breakdown(results):
    """Display performance by task type"""
    print("\n📋 PERFORMANCE BY TASK TYPE")
    print("-" * 80)
    
    # Load task metadata
    datasets_dir = Path(__file__).parent / "datasets"
    metadata = pd.read_csv(datasets_dir / "task_metadata.csv")
    
    # Merge with results
    comp = results['comparison'].merge(metadata[['task_id', 'task_type']], on='task_id')
    
    # Group by task type
    by_type = comp.groupby('task_type').agg({
        'mae_arima': ['mean', 'std', 'count'],
        'mae_ets': ['mean', 'std'],
        'oracle_mae': ['mean', 'std']
    }).round(2)
    
    print(f"\n{'Task Type':<20} {'Count':<10} {'ARIMA MAE':<15} {'ETS MAE':<15} {'Oracle MAE':<15}")
    print("-" * 80)
    
    for task_type in by_type.index:
        count = int(by_type.loc[task_type, ('mae_arima', 'count')])
        arima_mean = by_type.loc[task_type, ('mae_arima', 'mean')]
        ets_mean = by_type.loc[task_type, ('mae_ets', 'mean')]
        oracle_mean = by_type.loc[task_type, ('oracle_mae', 'mean')]
        
        print(f"{task_type:<20} {count:<10} {arima_mean:<15.2f} {ets_mean:<15.2f} {oracle_mean:<15.2f}")
    
    print("-" * 80)


def display_worst_performers(results, n=10):
    """Show tasks where models struggled most"""
    print(f"\n⚠️  TOP {n} HARDEST TASKS (Highest Oracle MAE)")
    print("-" * 80)
    
    comp = results['comparison'].sort_values('oracle_mae', ascending=False).head(n)
    
    print(f"{'Task ID':<12} {'Oracle MAE':<15} {'Best Model':<15} {'ARIMA MAE':<15} {'ETS MAE':<15}")
    print("-" * 80)
    
    oracle_df = results['oracle']
    for _, row in comp.iterrows():
        task_id = row['task_id']
        oracle_row = oracle_df[oracle_df['task_id'] == task_id].iloc[0]
        best_model = oracle_row['oracle_model'].upper()
        
        print(f"{task_id:<12} {row['oracle_mae']:<15.2f} {best_model:<15} "
              f"{row['mae_arima']:<15.2f} {row['mae_ets']:<15.2f}")
    
    print("-" * 80)


def display_best_performers(results, n=10):
    """Show tasks where models performed best"""
    print(f"\n✅ TOP {n} EASIEST TASKS (Lowest Oracle MAE)")
    print("-" * 80)
    
    comp = results['comparison'].sort_values('oracle_mae', ascending=True).head(n)
    
    print(f"{'Task ID':<12} {'Oracle MAE':<15} {'Best Model':<15} {'ARIMA MAE':<15} {'ETS MAE':<15}")
    print("-" * 80)
    
    oracle_df = results['oracle']
    for _, row in comp.iterrows():
        task_id = row['task_id']
        oracle_row = oracle_df[oracle_df['task_id'] == task_id].iloc[0]
        best_model = oracle_row['oracle_model'].upper()
        
        print(f"{task_id:<12} {row['oracle_mae']:<15.2f} {best_model:<15} "
              f"{row['mae_arima']:<15.2f} {row['mae_ets']:<15.2f}")
    
    print("-" * 80)


def display_failure_analysis(results):
    """Show any failed runs"""
    print("\n🔍 FAILURE ANALYSIS")
    print("-" * 80)
    
    arima_fails = results['arima'][results['arima']['status'] != 'success']
    ets_fails = results['ets'][results['ets']['status'] != 'success']
    
    if len(arima_fails) == 0 and len(ets_fails) == 0:
        print("  ✓ All tasks completed successfully!")
    else:
        if len(arima_fails) > 0:
            print(f"\n  AutoARIMA failures: {len(arima_fails)}")
            for _, row in arima_fails.iterrows():
                print(f"    - {row['task_id']}: {row['status']}")
        
        if len(ets_fails) > 0:
            print(f"\n  ETS failures: {len(ets_fails)}")
            for _, row in ets_fails.iterrows():
                print(f"    - {row['task_id']}: {row['status']}")
    
    print("-" * 80)


def main():
    """Main viewer"""
    print("\n" + "#"*80)
    print("#" + "  BASELINE EVALUATION RESULTS VIEWER".center(78) + "#")
    print("#"*80 + "\n")
    
    # Load results
    results = load_results()
    if results is None:
        return
    
    print("✓ Loaded all result files")
    
    # Display sections
    display_summary(results)
    
    try:
        display_task_breakdown(results)
    except Exception as e:
        print(f"\n⚠️  Could not display task breakdown: {e}")
    
    display_worst_performers(results, n=10)
    display_best_performers(results, n=10)
    display_failure_analysis(results)
    
    print("\n" + "="*80)
    print("✓ Analysis complete!")
    print("="*80 + "\n")


if __name__ == "__main__":
    main()
