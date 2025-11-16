"""
Day 4: Verify & Debug Numeric Results
Visualize baseline results and prepare for LLM experiments

Usage:
    python verify_baseline_results.py
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

sns.set_style("whitegrid")


def load_results():
    """Load all baseline results"""
    results_dir = Path("results")
    
    arima_df = pd.read_csv(results_dir / "arima_results.csv")
    ets_df = pd.read_csv(results_dir / "ets_results.csv")
    oracle_df = pd.read_csv(results_dir / "numeric_oracle.csv")
    combined_df = pd.read_csv(results_dir / "baseline_comparison.csv")
    
    return arima_df, ets_df, oracle_df, combined_df


def load_dataset():
    """Load original time series data"""
    datasets_dir = Path("datasets")
    
    ts_df = pd.read_csv(datasets_dir / "ts_instances.csv")
    with open(datasets_dir / "contexts.json", 'r') as f:
        contexts = json.load(f)
    
    # Parse JSON arrays
    ts_df['history'] = ts_df['history'].apply(json.loads).apply(np.array)
    ts_df['future'] = ts_df['future'].apply(json.loads).apply(np.array)
    
    return ts_df, contexts


def check_for_issues(arima_df, ets_df):
    """Check for NaNs, zeros, or other problems"""
    print("=" * 80)
    print("CHECKING FOR ISSUES")
    print("=" * 80)
    
    issues = []
    
    # Check for NaN MAEs
    arima_nans = arima_df['mae'].isna().sum()
    ets_nans = ets_df['mae'].isna().sum()
    
    if arima_nans > 0:
        issues.append(f"AutoARIMA has {arima_nans} NaN MAE values")
    if ets_nans > 0:
        issues.append(f"ETS has {ets_nans} NaN MAE values")
    
    # Check for extremely large MAEs (potential explosions)
    arima_large = (arima_df['mae'] > 1e6).sum()
    ets_large = (ets_df['mae'] > 1e6).sum()
    
    if arima_large > 0:
        issues.append(f"AutoARIMA has {arima_large} extremely large MAE values (>1e6)")
    if ets_large > 0:
        issues.append(f"ETS has {ets_large} extremely large MAE values (>1e6)")
    
    # Check failure rates
    arima_failures = (arima_df['status'] != 'success').sum()
    ets_failures = (ets_df['status'] != 'success').sum()
    
    if arima_failures > 0:
        issues.append(f"AutoARIMA failed on {arima_failures} tasks")
    if ets_failures > 0:
        issues.append(f"ETS failed on {ets_failures} tasks")
    
    if issues:
        print("\n⚠️  Issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✓ No major issues detected!")
    
    print()
    return issues


def print_summary_stats(arima_df, ets_df, oracle_df):
    """Print detailed summary statistics"""
    print("=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    # Filter out NaNs for statistics
    arima_valid = arima_df[arima_df['mae'].notna()]['mae']
    ets_valid = ets_df[ets_df['mae'].notna()]['mae']
    oracle_valid = oracle_df[oracle_df['oracle_mae'].notna()]['oracle_mae']
    
    stats_df = pd.DataFrame({
        'Model': ['AutoARIMA', 'ETS', 'Numeric Oracle'],
        'Count': [len(arima_valid), len(ets_valid), len(oracle_valid)],
        'Mean MAE': [arima_valid.mean(), ets_valid.mean(), oracle_valid.mean()],
        'Median MAE': [arima_valid.median(), ets_valid.median(), oracle_valid.median()],
        'Std MAE': [arima_valid.std(), ets_valid.std(), oracle_valid.std()],
        'Min MAE': [arima_valid.min(), ets_valid.min(), oracle_valid.min()],
        'Max MAE': [arima_valid.max(), ets_valid.max(), oracle_valid.max()],
    })
    
    print(stats_df.to_string(index=False))
    print()


def plot_random_forecasts(ts_df, arima_df, ets_df, n_plots=10, seed=42):
    """Plot random forecasts to visually inspect quality"""
    print("=" * 80)
    print(f"PLOTTING {n_plots} RANDOM FORECASTS")
    print("=" * 80)
    
    np.random.seed(seed)
    
    # Select random successful tasks
    successful_tasks = arima_df[arima_df['status'] == 'success']['task_id'].values
    if len(successful_tasks) < n_plots:
        print(f"Warning: Only {len(successful_tasks)} successful tasks available")
        n_plots = len(successful_tasks)
    
    random_tasks = np.random.choice(successful_tasks, size=n_plots, replace=False)
    
    # Create plots directory
    plots_dir = Path("results/plots")
    plots_dir.mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(5, 2, figsize=(15, 20))
    axes = axes.flatten()
    
    for idx, task_id in enumerate(random_tasks):
        # Get data
        task_row = ts_df[ts_df['id'] == task_id].iloc[0]
        history = task_row['history']
        future = task_row['future']
        
        # Get predictions (we'd need to recompute or save them)
        # For now, just plot history and future
        
        ax = axes[idx]
        
        # Plot history
        t_hist = np.arange(len(history))
        ax.plot(t_hist, history, 'b-', label='History', alpha=0.7)
        
        # Plot future (ground truth)
        t_future = np.arange(len(history), len(history) + len(future))
        ax.plot(t_future, future, 'g-', label='Ground Truth', linewidth=2)
        
        # Add vertical line at forecast start
        ax.axvline(x=len(history), color='red', linestyle='--', alpha=0.5)
        
        ax.set_title(f"{task_id} ({task_row['type']})")
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = plots_dir / "random_forecasts_verification.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved plots to {plot_path}")
    plt.close()


def plot_mae_distribution(combined_df):
    """Plot MAE distributions for comparison"""
    plots_dir = Path("results/plots")
    plots_dir.mkdir(exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # AutoARIMA
    axes[0].hist(combined_df['mae_arima'].dropna(), bins=30, color='skyblue', edgecolor='black')
    axes[0].set_title('AutoARIMA MAE Distribution')
    axes[0].set_xlabel('MAE')
    axes[0].set_ylabel('Frequency')
    axes[0].axvline(combined_df['mae_arima'].median(), color='red', linestyle='--', 
                    label=f'Median: {combined_df["mae_arima"].median():.2f}')
    axes[0].legend()
    
    # ETS
    axes[1].hist(combined_df['mae_ets'].dropna(), bins=30, color='lightcoral', edgecolor='black')
    axes[1].set_title('ETS MAE Distribution')
    axes[1].set_xlabel('MAE')
    axes[1].set_ylabel('Frequency')
    axes[1].axvline(combined_df['mae_ets'].median(), color='red', linestyle='--',
                    label=f'Median: {combined_df["mae_ets"].median():.2f}')
    axes[1].legend()
    
    # Oracle
    axes[2].hist(combined_df['oracle_mae'].dropna(), bins=30, color='lightgreen', edgecolor='black')
    axes[2].set_title('Numeric Oracle MAE Distribution')
    axes[2].set_xlabel('MAE')
    axes[2].set_ylabel('Frequency')
    axes[2].axvline(combined_df['oracle_mae'].median(), color='red', linestyle='--',
                    label=f'Median: {combined_df["oracle_mae"].median():.2f}')
    axes[2].legend()
    
    plt.tight_layout()
    plot_path = plots_dir / "mae_distributions.png"
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved MAE distributions to {plot_path}")
    plt.close()


def show_llm_prep_status():
    """Show what's ready for LLM experiments"""
    print("\n" + "=" * 80)
    print("LLM EXPERIMENT READINESS")
    print("=" * 80)
    
    print("\n✓ Baseline results complete!")
    print("✓ Numeric oracle computed")
    print("\nNext steps for Day 5:")
    print("  1. Set up Llama 3B model (via Ollama or HuggingFace)")
    print("  2. Write prompt templates:")
    print("     - History-only prompt (no context)")
    print("     - History + context prompt")
    print("  3. Run Llama on first 25 tasks")
    print("\nReady to proceed to Day 5! 🚀")


def main():
    print("\n" + "#" * 80)
    print("#" + " " * 78 + "#")
    print("#" + "  DAY 4: VERIFY & DEBUG NUMERIC RESULTS".center(78) + "#")
    print("#" + " " * 78 + "#")
    print("#" * 80 + "\n")
    
    # Load results
    print("Loading results...")
    arima_df, ets_df, oracle_df, combined_df = load_results()
    ts_df, contexts = load_dataset()
    print(f"✓ Loaded results for {len(arima_df)} tasks\n")
    
    # Check for issues
    issues = check_for_issues(arima_df, ets_df)
    
    # Print summary statistics
    print_summary_stats(arima_df, ets_df, oracle_df)
    
    # Plot MAE distributions
    print("Creating visualizations...")
    plot_mae_distribution(combined_df)
    
    # Plot random forecasts
    plot_random_forecasts(ts_df, arima_df, ets_df, n_plots=10)
    
    # Show LLM prep status
    show_llm_prep_status()
    
    print("\n" + "=" * 80)
    print("✓ Day 4 verification complete!")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()