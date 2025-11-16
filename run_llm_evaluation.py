"""
LLM Evaluation Script - Day 5
Runs Llama on all 50 CiK tasks with and without context

Usage:
    python run_llm_evaluation.py --method llmp --use-context
    python run_llm_evaluation.py --method llmp --no-context
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import argparse
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Add Benchmarks to path
sys.path.insert(0, str(Path(__file__).parent / "Benchmarks"))

from models.llm.llm_processes import LLMPForecaster
from models.llm.direct_prompt import DirectPrompt


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


def to_df(times, values):
    """Convert times and values to DataFrame for LLM models"""
    return pd.DataFrame({"y": values}, index=pd.to_datetime(times))


def create_timestamps(history, future):
    """Create dummy timestamps for history and future"""
    n_hist = len(history)
    n_fut = len(future)
    
    # Create timestamps (daily frequency as placeholder)
    end = pd.Timestamp.now().normalize()
    hist_times = pd.date_range(end=end, periods=n_hist, freq='H')
    fut_times = pd.date_range(
        start=hist_times[-1] + pd.Timedelta(hours=1), 
        periods=n_fut, 
        freq='H'
    )
    
    return hist_times, fut_times


def run_llm_on_task(llm_model, task_id, history, future, context, use_context, shared_pipe):
    """
    Run LLM model on a single task
    
    Args:
        llm_model: LLMPForecaster or DirectPrompt instance
        task_id: Task identifier
        history: Historical time series
        future: Ground truth future values
        context: Textual context description
        use_context: Whether to use context
        shared_pipe: Shared HuggingFace pipeline
    
    Returns:
        dict with predictions, actual, mae, and status
    """
    try:
        # Create timestamps
        hist_times, fut_times = create_timestamps(history, future)
        
        # Prepare task object
        class Task:
            def __init__(self, past_df, future_df, context_text=""):
                self.past_time = past_df
                self.future_time = future_df
                self.background = context_text if use_context else ""
                self.constraints = ""
                self.scenario = ""
        
        # Create DataFrames
        past_df = to_df(hist_times, history)
        future_df = to_df(fut_times, np.zeros(len(future)))
        
        # Create task
        task = Task(past_df, future_df, context if use_context else "")
        
        # Inject shared pipeline
        llm_model._pipe = shared_pipe
        
        # Run model
        samples, _ = llm_model(task, n_samples=1)
        predictions = samples.mean(axis=0).ravel()
        
        # Handle shape mismatch
        if len(predictions) != len(future):
            if len(predictions) > len(future):
                predictions = predictions[:len(future)]
            else:
                # Pad with last value
                padding = np.full(len(future) - len(predictions), predictions[-1])
                predictions = np.concatenate([predictions, padding])
        
        # Handle NaN values
        if not np.all(np.isfinite(predictions)):
            predictions = np.where(np.isfinite(predictions), predictions, history[-1])
        
        # Compute MAE
        mae = mean_absolute_error(future, predictions)
        
        return {
            'task_id': task_id,
            'predictions': predictions,
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


def build_shared_pipeline(model_id: str):
    """Load HuggingFace pipeline once and reuse"""
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    import torch
    
    print(f"\nLoading model: {model_id}")
    print("This may take a few minutes...")
    
    # Check for MPS (Apple Silicon GPU)
    if torch.backends.mps.is_available():
        device = "mps"
        print("✓ Using Apple Silicon GPU (MPS) - this will be MUCH faster!")
    else:
        device = "cpu"
        print("Using CPU")
    
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    
    mdl = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,  # Use half precision for speed
        low_cpu_mem_usage=True
    ).to(device)  # Move to GPU
    
    if getattr(mdl.config, "pad_token_id", None) is None:
        mdl.config.pad_token_id = tok.pad_token_id
    
    pipe = pipeline("text-generation", model=mdl, tokenizer=tok, device=device)
    print(f"✓ Model loaded on {device}!\n")
    
    return pipe


def run_llm_evaluation(ts_df, contexts, use_context, method='llmp', model_id='meta-llama/Llama-3.2-3B-Instruct'):
    """Run LLM on all tasks"""
    context_str = "WITH CONTEXT" if use_context else "NO CONTEXT"
    method_str = method.upper()
    
    print("=" * 80)
    print(f"RUNNING {method_str} - {context_str}")
    print("=" * 80)
    
    # Build shared pipeline
    shared_pipe = build_shared_pipeline(model_id)
    
    # Create model instance
    if method == 'llmp':
        llm_model = LLMPForecaster(model_id, use_context=use_context, dry_run=True)
    else:  # direct_prompt
        llm_model = DirectPrompt(
            model_id, 
            use_context=use_context, 
            dry_run=True, 
            temperature=0.7
        )
    
    results = []
    
    for idx, row in tqdm(ts_df.iterrows(), total=len(ts_df), desc=f"{method_str}-{context_str}"):
        task_id = row['id']
        history = row['history']
        future = row['future']
        context = contexts.get(task_id, "")
        
        # Run model
        result = run_llm_on_task(
            llm_model, task_id, history, future, context, use_context, shared_pipe
        )
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
    print(f"{method_str} - {context_str} Summary:")
    print(f"  Total tasks: {len(results_df)}")
    print(f"  Successful: {(results_df['status'] == 'success').sum()}")
    print(f"  Failed: {(results_df['status'] != 'success').sum()}")
    print(f"  Mean MAE: {results_df['mae'].mean():.4f}")
    print(f"  Median MAE: {results_df['mae'].median():.4f}")
    print(f"  Std MAE: {results_df['mae'].std():.4f}")
    print(f"{'='*80}\n")
    
    return results_df


def save_results(results_df, use_context, method='llmp'):
    """Save results to CSV"""
    print("=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    # Filename based on context and method
    context_suffix = "with_context" if use_context else "no_context"
    filename = f"{method}_{context_suffix}_results.csv"
    
    result_path = results_dir / filename
    results_df.to_csv(result_path, index=False)
    print(f"✓ Saved results to {result_path}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Run LLM evaluation on CiK tasks')
    parser.add_argument('--method', type=str, default='llmp', 
                       choices=['llmp', 'direct_prompt'],
                       help='LLM forecasting method')
    parser.add_argument('--use-context', dest='use_context', action='store_true',
                       help='Use textual context')
    parser.add_argument('--no-context', dest='use_context', action='store_false',
                       help='Do not use textual context')
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.2-3B-Instruct',
                       help='HuggingFace model ID')
    parser.set_defaults(use_context=True)
    
    args = parser.parse_args()
    
    print("\n")
    print("#" * 80)
    print("#" + " " * 78 + "#")
    print("#" + "  LLM EVALUATION - DAY 5".center(78) + "#")
    context_str = "WITH CONTEXT" if args.use_context else "NO CONTEXT"
    print("#" + f"  {args.method.upper()} - {context_str}".center(78) + "#")
    print("#" + " " * 78 + "#")
    print("#" * 80)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Load dataset
    ts_df, contexts = load_dataset()
    
    # Run LLM evaluation
    results_df = run_llm_evaluation(
        ts_df, contexts, 
        use_context=args.use_context, 
        method=args.method,
        model_id=args.model
    )
    
    # Save results
    save_results(results_df, args.use_context, method=args.method)
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n✓ Day 5 LLM evaluation complete!")
    print(f"\nResults saved to: {Path(__file__).parent / 'results'}")
    print()


if __name__ == "__main__":
    main()