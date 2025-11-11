"""
Unified Benchmarking Script

Run AutoARIMA, PatchTST, and any other models on your datasets
with consistent evaluation metrics.

Usage:
    python run_unified_benchmark.py --data finance_data.csv --models autoarima patchtst
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Benchmarks.models.ets import ETSModel
from models import AutoARIMAModel, PatchTSTModel  
from evaluation import TimeSeriesEvaluator    


def load_data(data_path: str) -> tuple:
    """
    Load time series data from CSV.
    
    Expected format:
        - CSV with 'date'/'ds'/'Date' and 'y' columns
        OR
        - CSV with 'Date' and 'Close' (stock data)
        OR
        - CSV with just values (one column)
    
    Returns:
        (data_df, timestamps) tuple
    """
    df = pd.read_csv(data_path)
    
    # Check for date columns (case-insensitive)
    date_candidates = ['date', 'ds', 'Date', 'DATE', 'timestamp', 'Timestamp']
    date_col = None
    for col in date_candidates:
        if col in df.columns:
            date_col = col
            break
    
    if date_col is not None:
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Standardize to 'ds' column name
        if date_col != 'ds':
            df = df.rename(columns={date_col: 'ds'})
        
        # Find target column
        if 'y' not in df.columns:
            # Priority: Close > close > second column
            if 'Close' in df.columns:
                df = df.rename(columns={'Close': 'y'})
            elif 'close' in df.columns:
                df = df.rename(columns={'close': 'y'})
            else:
                # Assume second column is the target
                value_col = [c for c in df.columns if c != 'ds'][0]
                df = df.rename(columns={value_col: 'y'})
        
        # Keep only ds and y columns
        df = df[['ds', 'y']]
        
        return df, df['ds']
    else:
        # No timestamps, just values
        y_values = df.iloc[:, 0].values
        timestamps = pd.date_range(start='2020-01-01', periods=len(y_values), freq='D')
        df = pd.DataFrame({'ds': timestamps, 'y': y_values})
        return df, timestamps


def get_models(model_names, config):
    """
    Initialize models based on names.
    
    Args:
        model_names: List of model names (e.g., ['autoarima', 'patchtst'])
        config: Configuration dict with model parameters
    
    Returns:
        List of TimeSeriesModel instances
    """
    models = []
    
    for name in model_names:
        name_lower = name.lower()
        
        if name_lower == 'autoarima':
            model = AutoARIMAModel(
                season_length=config.get('season_length', 1),
                freq=config.get('freq', 'D'),
                name='AutoARIMA'
            )
            models.append(model)
            
        elif name_lower == 'patchtst':
            model = PatchTSTModel(
                seq_len=config.get('context_length', 512),  # seq_len in PatchTST = context_length
                pred_len=config.get('pred_len', 96),
                patch_len=config.get('patch_size', 16),
                stride=config.get('stride', 8),
                d_model=config.get('d_model', 64),
                n_heads=config.get('n_heads', 4),
                e_layers=config.get('e_layers', 2),
                device=config.get('device', 'cpu'),
                name='PatchTST'
            )
            models.append(model)
            
        elif name_lower == 'ets':  # ← ADD THIS
            model = ETSModel(
                seasonal_periods=config.get('season_length', 1),
                trend='add',
                seasonal='add' if config.get('season_length', 1) > 1 else None,
                damped_trend=False,
                freq=config.get('freq', 'D'),
                name='ETS'
            )
            models.append(model)
            
        else:
            print(f"Warning: Unknown model '{name}', skipping...")
    
    return models


def main():
    parser = argparse.ArgumentParser(
        description="Unified benchmarking for time series forecasting models"
    )
    
    # Data arguments
    parser.add_argument('--data', type=str, required=True,
                       help='Path to CSV file with time series data')
    parser.add_argument('--train_split', type=float, default=0.7,
                       help='Fraction of data to use for training (default: 0.7)')
    
    # Model arguments
    parser.add_argument('--models', type=str, nargs='+', default=['autoarima'],
                       help='Models to evaluate (autoarima, patchtst)')
    parser.add_argument('--season_length', type=int, default=1,
                       help='Seasonal period (1=non-seasonal, 7=weekly, 12=monthly)')
    parser.add_argument('--freq', type=str, default='D',
                       help='Pandas frequency string (D=daily, H=hourly, M=monthly)')
    
    # Evaluation arguments
    parser.add_argument('--context_length', type=int, default=100,
                       help='Length of history to use for predictions')
    parser.add_argument('--horizon', type=int, default=10,
                       help='Forecast horizon (steps ahead)')
    parser.add_argument('--step', type=int, default=10,
                       help='Stride between rolling windows')
    parser.add_argument('--max_windows', type=int, default=None,
                       help='Maximum number of windows to evaluate (for speed testing)')
    
    # Quantiles for probabilistic forecasting
    parser.add_argument('--quantiles', type=str, default='0.1,0.5,0.9',
                       help='Comma-separated quantiles (e.g., "0.1,0.5,0.9")')
    
    # Output arguments
    parser.add_argument('--save_dir', type=str, default='./results',
                       help='Directory to save results')
    parser.add_argument('--experiment_name', type=str, default=None,
                       help='Name for this experiment (default: auto-generated)')
    
    # Hardware
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device for PatchTST (cpu or cuda)')
    
    args = parser.parse_args()
    
    # Parse quantiles
    quantiles = [float(q) for q in args.quantiles.split(',')]
    
    # Generate experiment name if not provided
    if args.experiment_name is None:
        data_name = Path(args.data).stem
        model_names = '_'.join(args.models)
        args.experiment_name = f"{data_name}_{model_names}"
    
    print("=" * 80)
    print("TIME SERIES FORECASTING BENCHMARK")
    print("=" * 80)
    print(f"\nExperiment: {args.experiment_name}")
    print(f"Data: {args.data}")
    print(f"Models: {', '.join(args.models)}")
    print(f"Evaluation: {args.context_length} -> {args.horizon} (step={args.step})")
    print(f"Quantiles: {quantiles}")
    print()
    
    # Load data
    print("Loading data...")
    df, timestamps = load_data(args.data)
    y = df['y'].values
    print(f"Loaded {len(y)} timesteps")
    
    # Split train/test
    train_size = int(len(y) * args.train_split)
    train_data = y[:train_size]
    test_data = y[train_size:]
    print(f"Train: {len(train_data)} timesteps | Test: {len(test_data)} timesteps")
    
    # Configuration for models
    config = {
        'season_length': args.season_length,
        'freq': args.freq,
        'context_length': args.context_length,
        'pred_len': args.horizon,  # PatchTST uses pred_len
        'patch_size': 16,
        'stride': 8,
        'device': args.device,
        # PatchTST model size (smaller for faster training)
        'd_model': 64,
        'n_heads': 4,
        'e_layers': 2,
    }
    
    # Initialize models
    print("\nInitializing models...")
    models = get_models(args.models, config)
    
    if not models:
        print("Error: No valid models specified!")
        return
    
    # Fit models on training data
    print("\nFitting models on training data...")
    for model in models:
        print(f"\nFitting {model.name}...")
        try:
            model.fit(train_data, timestamps=timestamps[:train_size])
            print(f"✓ {model.name} fitted successfully")
        except Exception as e:
            print(f"✗ Error fitting {model.name}: {str(e)}")
    
    # Create evaluator
    evaluator = TimeSeriesEvaluator(quantiles=quantiles)
    
    # Evaluate each model
    print("\n" + "=" * 80)
    print("EVALUATION ON TEST DATA")
    print("=" * 80)
    
    all_results = []
    for model in models:
        print(f"\nEvaluating {model.name}...")
        try:
            results = evaluator.evaluate_model(
                model=model,
                data=test_data,
                train_data=train_data,
                context_length=args.context_length,
                horizon=args.horizon,
                step=args.step,
                max_windows=args.max_windows,
                seasonality=args.season_length,
                verbose=True
            )
            all_results.append(results)
            
            # Save individual model results
            model_dir = Path(args.save_dir) / args.experiment_name / model.name
            evaluator.save_results(results, model_dir, save_predictions=True)
            
        except Exception as e:
            print(f"✗ Error evaluating {model.name}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    # Create comparison table
    if len(all_results) > 1:
        print("\n" + "=" * 80)
        print("MODEL COMPARISON")
        print("=" * 80)
        
        comparison_data = []
        for results in all_results:
            row = {'Model': results['model_name']}
            for metric_name, stats in results['metrics'].items():
                if metric_name != 'prediction_time':
                    row[metric_name] = f"{stats['mean']:.4f} ± {stats['std']:.4f}"
            row['Time (s)'] = f"{results['timing']['total_time']:.2f}"
            comparison_data.append(row)
        
        comparison_df = pd.DataFrame(comparison_data)
        print("\n", comparison_df.to_string(index=False))
        
        # Save comparison
        comparison_path = Path(args.save_dir) / args.experiment_name / 'comparison.csv'
        comparison_df.to_csv(comparison_path, index=False)
        print(f"\nComparison saved to: {comparison_path}")
    
    # Save experiment config
    config_path = Path(args.save_dir) / args.experiment_name / 'config.json'
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    
    print("\n" + "=" * 80)
    print(f"✓ Benchmark complete! Results saved to: {Path(args.save_dir) / args.experiment_name}")
    print("=" * 80)


if __name__ == "__main__":
    main()