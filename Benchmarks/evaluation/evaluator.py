"""
Unified Evaluator for Time Series Forecasting Models

Provides a consistent evaluation pipeline for all models
(AutoARIMA, PatchTST, LLMs, Foundation Models, etc.)
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Union
from pathlib import Path
import json
import time


import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from models.base_model import TimeSeriesModel  # ← Changed
from .metrics import compute_all_metrics        # ← Changed



class TimeSeriesEvaluator:
    """
    Unified evaluator for time series forecasting models.
    
    Features:
    - Rolling window evaluation
    - Multiple metrics (accuracy + calibration)
    - Consistent results format
    - Easy comparison across models
    """
    
    def __init__(self, 
                 metrics: Optional[List[str]] = None,
                 quantiles: Optional[List[float]] = None):
        """
        Args:
            metrics: List of metric names to compute
                    If None, computes all available metrics
            quantiles: Quantiles to evaluate (e.g., [0.1, 0.5, 0.9])
                      Required for calibration metrics
        """
        self.metrics = metrics
        self.quantiles = quantiles if quantiles else [0.1, 0.5, 0.9]
        
    def evaluate_model(self,
                      model: TimeSeriesModel,
                      data: Union[np.ndarray, pd.DataFrame],
                      train_data: Optional[np.ndarray] = None,
                      context_length: int = 100,
                      horizon: int = 10,
                      step: int = 10,
                      max_windows: Optional[int] = None,
                      seasonality: int = 1,
                      verbose: bool = True) -> Dict:
        """
        Evaluate a model on a time series using rolling windows.
        
        Args:
            model: TimeSeriesModel instance
            data: Time series data to evaluate on
            train_data: Training data (for MASE scaling)
            context_length: Length of history to use for predictions
            horizon: Forecast horizon
            step: Stride between rolling windows
            max_windows: Maximum number of windows to evaluate (for speed)
            seasonality: Seasonal period for MASE
            verbose: Print progress
            
        Returns:
            Dictionary with:
                - 'metrics': Dict of averaged metrics
                - 'per_window_metrics': List of metrics per window
                - 'predictions': List of prediction results
                - 'timing': Execution time info
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"Evaluating: {model.name}")
            print(f"{'='*60}")
            print(f"Context length: {context_length}, Horizon: {horizon}, Step: {step}")
        
        # Convert data to numpy if needed
        if isinstance(data, pd.DataFrame):
            y_full = data['y'].values
        else:
            y_full = data
        
        # Use provided training data or first half of data
        if train_data is None:
            train_data = y_full[:len(y_full)//2]
        
        # Generate rolling windows
        start_time = time.time()
        
        windows = []
        predictions = []
        per_window_metrics = []
        
        max_start = len(y_full) - context_length - horizon
        n_windows = 0
        
        for start_idx in range(0, max_start + 1, step):
            if max_windows and n_windows >= max_windows:
                break
                
            end_idx = start_idx + context_length
            history = y_full[start_idx:end_idx]
            future = y_full[end_idx:end_idx + horizon]
            
            # Make prediction
            try:
                pred_start = time.time()
                preds = model.predict(history, horizon, quantiles=self.quantiles)
                pred_time = time.time() - pred_start
                
                # Extract point forecast
                y_pred = preds.get('mean', preds.get('median', None))
                if y_pred is None:
                    if verbose:
                        print(f"Warning: No point forecast available for window {n_windows}")
                    continue
                
                # Compute metrics for this window
                window_metrics = compute_all_metrics(
                    y_true=future,
                    y_pred=y_pred,
                    y_train=train_data,
                    quantile_predictions=preds.get('quantiles', None),
                    seasonality=seasonality
                )
                window_metrics['prediction_time'] = pred_time
                
                per_window_metrics.append(window_metrics)
                predictions.append({
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'history': history,
                    'future': future,
                    'predictions': preds,
                    'metrics': window_metrics
                })
                
                n_windows += 1
                
                if verbose and n_windows % 10 == 0:
                    avg_mae = np.mean([m['MAE'] for m in per_window_metrics])
                    print(f"  Evaluated {n_windows} windows | Avg MAE: {avg_mae:.4f}")
                    
            except Exception as e:
                if verbose:
                    print(f"  Error in window {n_windows}: {str(e)}")
                continue
        
        total_time = time.time() - start_time
        
        # Aggregate metrics across all windows
        aggregated_metrics = {}
        if per_window_metrics:
            metric_names = per_window_metrics[0].keys()
            for metric_name in metric_names:
                values = [m[metric_name] for m in per_window_metrics 
                         if not np.isnan(m[metric_name])]
                if values:
                    aggregated_metrics[metric_name] = {
                        'mean': float(np.mean(values)),
                        'std': float(np.std(values)),
                        'min': float(np.min(values)),
                        'max': float(np.max(values))
                    }
        
        results = {
            'model_name': model.name,
            'metrics': aggregated_metrics,
            'per_window_metrics': per_window_metrics,
            'predictions': predictions,
            'timing': {
                'total_time': total_time,
                'n_windows': n_windows,
                'avg_time_per_window': total_time / n_windows if n_windows > 0 else 0
            },
            'config': {
                'context_length': context_length,
                'horizon': horizon,
                'step': step,
                'seasonality': seasonality,
                'quantiles': self.quantiles
            }
        }
        
        if verbose:
            print(f"\n{'-'*60}")
            print("Results:")
            print(f"{'-'*60}")
            for metric_name, stats in aggregated_metrics.items():
                if metric_name != 'prediction_time':
                    print(f"{metric_name:20s}: {stats['mean']:.4f} ± {stats['std']:.4f}")
            print(f"\nTotal time: {total_time:.2f}s ({n_windows} windows)")
            print(f"{'='*60}\n")
        
        return results
    
    def compare_models(self,
                      models: List[TimeSeriesModel],
                      data: Union[np.ndarray, pd.DataFrame],
                      **kwargs) -> pd.DataFrame:
        """
        Evaluate multiple models and return comparison table.
        
        Args:
            models: List of models to compare
            data: Time series data
            **kwargs: Arguments passed to evaluate_model
            
        Returns:
            DataFrame with models as rows and metrics as columns
        """
        all_results = []
        
        for model in models:
            results = self.evaluate_model(model, data, **kwargs)
            
            # Flatten metrics for comparison
            row = {'Model': model.name}
            for metric_name, stats in results['metrics'].items():
                row[f"{metric_name}_mean"] = stats['mean']
                row[f"{metric_name}_std"] = stats['std']
            row['Time (s)'] = results['timing']['total_time']
            
            all_results.append(row)
        
        df = pd.DataFrame(all_results)
        return df
    
    def save_results(self, 
                    results: Dict, 
                    save_dir: Union[str, Path],
                    save_predictions: bool = False):
        """
        Save evaluation results to disk.
        
        Args:
            results: Results dict from evaluate_model
            save_dir: Directory to save results
            save_predictions: Whether to save individual predictions (can be large)
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Save aggregated metrics
        metrics_df = pd.DataFrame(results['metrics']).T
        metrics_df.to_csv(save_dir / 'metrics.csv')
        
        # Save per-window metrics
        if results['per_window_metrics']:
            per_window_df = pd.DataFrame(results['per_window_metrics'])
            per_window_df.to_csv(save_dir / 'per_window_metrics.csv', index=False)
        
        # Save config and timing
        metadata = {
            'model_name': results['model_name'],
            'config': results['config'],
            'timing': results['timing']
        }
        with open(save_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Optionally save predictions
        if save_predictions and results['predictions']:
            # Save first few predictions as examples
            n_save = min(5, len(results['predictions']))
            example_preds = results['predictions'][:n_save]
            
            # Convert to serializable format
            serializable_preds = []
            for pred in example_preds:
                serializable_preds.append({
                    'start_idx': int(pred['start_idx']),
                    'end_idx': int(pred['end_idx']),
                    'history': pred['history'].tolist(),
                    'future': pred['future'].tolist(),
                    'mean_pred': pred['predictions']['mean'].tolist(),
                    'metrics': pred['metrics']
                })
            
            with open(save_dir / 'example_predictions.json', 'w') as f:
                json.dump(serializable_preds, f, indent=2)
        
        print(f"Results saved to {save_dir}")


# Example usage
if __name__ == "__main__":
    print("Unified Evaluator - Example Usage")
    print("=" * 60)
    
    # Generate synthetic data
    np.random.seed(42)
    t = np.arange(500)
    y = 10 + 0.1 * t + 5 * np.sin(2 * np.pi * t / 50) + np.random.randn(500)
    
    # This would normally be your actual models
    # For demo, we'll create mock models
    from Benchmarks.models.base_model import TimeSeriesModel
    
    class MockModel(TimeSeriesModel):
        def __init__(self, name):
            super().__init__(name)
            self.is_fitted = True
        
        def fit(self, train_data, timestamps=None):
            return self
        
        def predict(self, history, horizon, quantiles=None):
            # Simple linear extrapolation
            trend = (history[-1] - history[0]) / len(history)
            mean_pred = history[-1] + trend * np.arange(1, horizon + 1)
            mean_pred += np.random.randn(horizon) * 0.1  # Add noise
            
            result = {'mean': mean_pred, 'median': mean_pred}
            
            if quantiles:
                result['quantiles'] = {
                    q: mean_pred + (q - 0.5) * 2.0 for q in quantiles
                }
            
            return result
    
    # Create evaluator
    evaluator = TimeSeriesEvaluator(quantiles=[0.1, 0.5, 0.9])
    
    # Create mock models
    model1 = MockModel("Model_A")
    model2 = MockModel("Model_B")
    
    # Evaluate single model
    print("\nEvaluating single model:")
    results = evaluator.evaluate_model(
        model1, 
        y, 
        context_length=50, 
        horizon=10, 
        step=20,
        max_windows=10
    )
    
    # Compare models
    print("\nComparing multiple models:")
    comparison_df = evaluator.compare_models(
        [model1, model2],
        y,
        context_length=50,
        horizon=10,
        step=20,
        max_windows=10,
        verbose=False
    )
    print(comparison_df)
    
    # Save results
    evaluator.save_results(results, save_dir='./example_results')
