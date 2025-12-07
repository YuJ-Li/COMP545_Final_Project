"""
Generate Model Performance Summary Table

Standalone script to create model_performance_summary.csv from compiled_comparison.csv
Computes comprehensive statistics across all models and tasks.

Author: Kazi Ashab Rahman
Date: December 2024
Usage: python generate_model_performance_summary.py
"""

import pandas as pd
import numpy as np
from pathlib import Path


def generate_model_performance_summary(results_dir='Results'):
    """
    Generate comprehensive model performance summary table.
    
    Computes:
    - Mean/Median/Std Dev for NMAE and DA
    - Standard Error (SE = Std Dev / sqrt(n)) for Mean NMAE and Mean DA
    - Win rates (which model achieves lowest NMAE per task)
    - Mean improvement over baseline
    - Percentage of tasks where each model beats baseline
    
    Standard Error represents the uncertainty in the estimated mean.
    Smaller SE indicates more precise estimate of the true population mean.
    Formula: SE = σ / √n where σ is std dev and n is sample size.
    
    Parameters:
    -----------
    results_dir : str
        Path to Results directory containing compiled_comparison.csv
    
    Returns:
    --------
    summary_df : pd.DataFrame
        Summary statistics dataframe
    """
    print("="*80)
    print("GENERATING MODEL PERFORMANCE SUMMARY")
    print("="*80)
    
    # Load compiled comparison data
    results_path = Path(results_dir)
    compiled_path = results_path / 'compiled_comparison.csv'
    
    if not compiled_path.exists():
        raise FileNotFoundError(f"Could not find {compiled_path}")
    
    df = pd.read_csv(compiled_path)
    print(f"✓ Loaded {len(df)} tasks from compiled_comparison.csv")
    print(f"  Sample size (n) for SE calculation: {len(df)}")
    print(f"  SE = Std Dev / √n = Std Dev / √{len(df)} = Std Dev / {np.sqrt(len(df)):.2f}")
    
    # Initialize summary dictionary
    summary = {
        'Metric': [],
        'ARIMA': [],
        'ETS': [],
        'Best Baseline': [],
        'Llama 3B': [],
        'Mistral 8x7B': [],
        'GPT-4o': []
    }
    
    print("\n📊 Computing statistics...")

    print("  - NMAE statistics (Mean, Median, Std Dev, Std Error)")
    
    n = len(df)  # Number of tasks
    
    summary['Metric'].extend(['Mean NMAE', 'SE NMAE', 'Median NMAE', 'Std Dev NMAE'])
    
    summary['ARIMA'].extend([
        df['arima_nmae'].mean(),
        df['arima_nmae'].std() / np.sqrt(n),
        df['arima_nmae'].median(),
        df['arima_nmae'].std()
    ])
    
    summary['ETS'].extend([
        df['ets_nmae'].mean(),
        df['ets_nmae'].std() / np.sqrt(n),
        df['ets_nmae'].median(),
        df['ets_nmae'].std()
    ])
    
    summary['Best Baseline'].extend([
        df['best_baseline_nmae'].mean(),
        df['best_baseline_nmae'].std() / np.sqrt(n),
        df['best_baseline_nmae'].median(),
        df['best_baseline_nmae'].std()
    ])
    
    summary['Llama 3B'].extend([
        df['llama_nmae'].mean(),
        df['llama_nmae'].std() / np.sqrt(n),
        df['llama_nmae'].median(),
        df['llama_nmae'].std()
    ])
    
    summary['Mistral 8x7B'].extend([
        df['mistral_nmae'].mean(),
        df['mistral_nmae'].std() / np.sqrt(n),
        df['mistral_nmae'].median(),
        df['mistral_nmae'].std()
    ])
    
    summary['GPT-4o'].extend([
        df['gpt4o_nmae'].mean(),
        df['gpt4o_nmae'].std() / np.sqrt(n),
        df['gpt4o_nmae'].median(),
        df['gpt4o_nmae'].std()
    ])

    print("  - Directional Accuracy statistics (Mean, SE, Median)")
    
    summary['Metric'].extend(['Mean DA', 'SE DA', 'Median DA'])
    
    summary['ARIMA'].extend([
        df['arima_da'].mean(),
        df['arima_da'].std() / np.sqrt(n),
        df['arima_da'].median()
    ])
    
    summary['ETS'].extend([
        df['ets_da'].mean(),
        df['ets_da'].std() / np.sqrt(n),
        df['ets_da'].median()
    ])
    
    summary['Best Baseline'].extend([
        df['best_baseline_da'].mean(),
        df['best_baseline_da'].std() / np.sqrt(n),
        df['best_baseline_da'].median()
    ])
    
    summary['Llama 3B'].extend([
        df['llama_da'].mean(),
        df['llama_da'].std() / np.sqrt(n),
        df['llama_da'].median()
    ])
    
    summary['Mistral 8x7B'].extend([
        df['mistral_da'].mean(),
        df['mistral_da'].std() / np.sqrt(n),
        df['mistral_da'].median()
    ])
    
    summary['GPT-4o'].extend([
        df['gpt4o_da'].mean(),
        df['gpt4o_da'].std() / np.sqrt(n),
        df['gpt4o_da'].median()
    ])

    print("  - Win rates (lowest NMAE per task)")
    
    # Count wins for each model
    arima_wins = (df['best_model'] == 'ARIMA').sum()
    ets_wins = (df['best_model'] == 'ETS').sum()
    llama_wins = (df['best_model'] == 'LLMP').sum()
    mistral_wins = (df['best_model'] == 'MISTRAL').sum()
    gpt4o_wins = (df['best_model'] == 'GPT4O').sum()
    total = len(df)
    
    summary['Metric'].extend(['Win Rate %', 'Win Count'])
    
    summary['ARIMA'].extend([
        100 * arima_wins / total,
        arima_wins
    ])
    
    summary['ETS'].extend([
        100 * ets_wins / total,
        ets_wins
    ])
    
    summary['Best Baseline'].extend(['-', '-'])
    
    summary['Llama 3B'].extend([
        100 * llama_wins / total,
        llama_wins
    ])
    
    summary['Mistral 8x7B'].extend([
        100 * mistral_wins / total,
        mistral_wins
    ])
    
    summary['GPT-4o'].extend([
        100 * gpt4o_wins / total,
        gpt4o_wins
    ])

    print("  - Improvement over baseline metrics")
    
    summary['Metric'].extend(['Mean Improvement (NMAE)', '% Beat Baseline'])
    
    summary['ARIMA'].extend(['-', '-'])
    summary['ETS'].extend(['-', '-'])
    summary['Best Baseline'].extend([0, '-'])
    
    summary['Llama 3B'].extend([
        df['llama_improvement_nmae'].mean(),
        100 * df['llama_beats_baseline'].mean()
    ])
    
    summary['Mistral 8x7B'].extend([
        df['mistral_improvement_nmae'].mean(),
        100 * df['mistral_beats_baseline'].mean()
    ])
    
    summary['GPT-4o'].extend([
        df['gpt4o_improvement_nmae'].mean(),
        100 * df['gpt4o_beats_baseline'].mean()
    ])

    summary_df = pd.DataFrame(summary)
    
    output_path = results_path / 'model_performance_summary.csv'
    summary_df.to_csv(output_path, index=False)
    
    print(f"\n💾 Saved: {output_path}")
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    print("\n" + summary_df.to_string(index=False))
    print("\n" + "="*80)
    
    return summary_df


def display_key_insights(summary_df):
    """Display key insights from the summary statistics."""
    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)
    
    # Extract key values for NMAE
    best_baseline_nmae = summary_df.loc[summary_df['Metric'] == 'Mean NMAE', 'Best Baseline'].values[0]
    best_baseline_se = summary_df.loc[summary_df['Metric'] == 'SE NMAE', 'Best Baseline'].values[0]
    
    llama_nmae = summary_df.loc[summary_df['Metric'] == 'Mean NMAE', 'Llama 3B'].values[0]
    llama_se = summary_df.loc[summary_df['Metric'] == 'SE NMAE', 'Llama 3B'].values[0]
    
    mistral_nmae = summary_df.loc[summary_df['Metric'] == 'Mean NMAE', 'Mistral 8x7B'].values[0]
    mistral_se = summary_df.loc[summary_df['Metric'] == 'SE NMAE', 'Mistral 8x7B'].values[0]
    
    gpt4o_nmae = summary_df.loc[summary_df['Metric'] == 'Mean NMAE', 'GPT-4o'].values[0]
    gpt4o_se = summary_df.loc[summary_df['Metric'] == 'SE NMAE', 'GPT-4o'].values[0]
    
    # Extract key values for DA
    best_baseline_da = summary_df.loc[summary_df['Metric'] == 'Mean DA', 'Best Baseline'].values[0]
    best_baseline_da_se = summary_df.loc[summary_df['Metric'] == 'SE DA', 'Best Baseline'].values[0]
    
    llama_da = summary_df.loc[summary_df['Metric'] == 'Mean DA', 'Llama 3B'].values[0]
    llama_da_se = summary_df.loc[summary_df['Metric'] == 'SE DA', 'Llama 3B'].values[0]
    
    mistral_da = summary_df.loc[summary_df['Metric'] == 'Mean DA', 'Mistral 8x7B'].values[0]
    mistral_da_se = summary_df.loc[summary_df['Metric'] == 'SE DA', 'Mistral 8x7B'].values[0]
    
    gpt4o_da = summary_df.loc[summary_df['Metric'] == 'Mean DA', 'GPT-4o'].values[0]
    gpt4o_da_se = summary_df.loc[summary_df['Metric'] == 'SE DA', 'GPT-4o'].values[0]
    
    # Win rates
    llama_win_rate = summary_df.loc[summary_df['Metric'] == 'Win Rate %', 'Llama 3B'].values[0]
    mistral_win_rate = summary_df.loc[summary_df['Metric'] == 'Win Rate %', 'Mistral 8x7B'].values[0]
    gpt4o_win_rate = summary_df.loc[summary_df['Metric'] == 'Win Rate %', 'GPT-4o'].values[0]
    
    arima_win_rate = summary_df.loc[summary_df['Metric'] == 'Win Rate %', 'ARIMA'].values[0]
    ets_win_rate = summary_df.loc[summary_df['Metric'] == 'Win Rate %', 'ETS'].values[0]
    
    print(f"\n1. MEAN NMAE COMPARISON (± Standard Error):")
    print(f"   Best Baseline: {best_baseline_nmae:.4f} ± {best_baseline_se:.4f}")
    print(f"   Llama 3B:      {llama_nmae:.4f} ± {llama_se:.4f} ({'+' if llama_nmae > best_baseline_nmae else ''}{llama_nmae - best_baseline_nmae:.4f})")
    print(f"   Mistral 8x7B:  {mistral_nmae:.4f} ± {mistral_se:.4f} ({'+' if mistral_nmae > best_baseline_nmae else ''}{mistral_nmae - best_baseline_nmae:.4f})")
    print(f"   GPT-4o mini:   {gpt4o_nmae:.4f} ± {gpt4o_se:.4f} ({'+' if gpt4o_nmae > best_baseline_nmae else ''}{gpt4o_nmae - best_baseline_nmae:.4f})")
    
    print(f"\n2. MEAN DA COMPARISON (± Standard Error):")
    print(f"   Best Baseline: {best_baseline_da:.4f} ± {best_baseline_da_se:.4f}")
    print(f"   Llama 3B:      {llama_da:.4f} ± {llama_da_se:.4f} ({'+' if llama_da > best_baseline_da else ''}{llama_da - best_baseline_da:.4f})")
    print(f"   Mistral 8x7B:  {mistral_da:.4f} ± {mistral_da_se:.4f} ({'+' if mistral_da > best_baseline_da else ''}{mistral_da - best_baseline_da:.4f})")
    print(f"   GPT-4o mini:   {gpt4o_da:.4f} ± {gpt4o_da_se:.4f} ({'+' if gpt4o_da > best_baseline_da else ''}{gpt4o_da - best_baseline_da:.4f})")
    
    print(f"\n3. WIN RATES (Lowest NMAE per task):")
    print(f"   Classical Baselines: {arima_win_rate + ets_win_rate:.1f}% (ARIMA: {arima_win_rate:.1f}% + ETS: {ets_win_rate:.1f}%)")
    print(f"   Best LLM (Mistral):  {mistral_win_rate:.1f}%")
    print(f"   GPT-4o mini:         {gpt4o_win_rate:.1f}%")
    print(f"   Llama 3B:            {llama_win_rate:.1f}%")
    
    print(f"\n4. KEY FINDING:")
    if mistral_nmae < best_baseline_nmae:
        improvement = ((best_baseline_nmae - mistral_nmae) / best_baseline_nmae) * 100
        print(f"   ✓ Mistral improves over baseline by {improvement:.2f}%")
    else:
        degradation = ((mistral_nmae - best_baseline_nmae) / best_baseline_nmae) * 100
        print(f"   ✗ Mistral is WORSE than baseline by {degradation:.2f}%")
        print(f"   → Context benefits do NOT transfer to practical model sizes!")
    
    print("\n" + "="*80)


def main():
    """Main execution function."""
    print("\n" + "#"*80)
    print("#" + " "*78 + "#")
    print("#" + "  MODEL PERFORMANCE SUMMARY GENERATOR".center(78) + "#")
    print("#" + " "*78 + "#")
    print("#"*80)
    print()
    
    # Generate summary
    summary_df = generate_model_performance_summary(results_dir='Results')
    
    # Display key insights
    display_key_insights(summary_df)
    
    print("\n✅ Model performance summary complete!")
    print("\nOutput file: Results/model_performance_summary.csv")
    print()


if __name__ == "__main__":
    main()