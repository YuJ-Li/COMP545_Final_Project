"""
Standalone Mistral Large 2 Forecasting Script
No dependencies on Benchmarks folder - uses Mistral directly

Usage:
    python run_mistral_standalone.py --domain ConstrainedRandomWalk --use-context
    python run_mistral_standalone.py --domain ConstrainedRandomWalk --no-context
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


def load_domain_tasks(domain_name):
    """Load existing tasks from saved files"""
    print("=" * 80)
    print(f"LOADING TASKS FOR: {domain_name}")
    print("=" * 80)
    
    domain_dir = Path("Results") / domain_name / "datasets"
    
    if not domain_dir.exists():
        print(f"ERROR: Dataset directory not found: {domain_dir}")
        sys.exit(1)
    
    # Load time series
    ts_file = domain_dir / "ts_instances.csv"
    ts_df = pd.read_csv(ts_file)
    print(f"✓ Loaded {len(ts_df)} tasks from {ts_file}")
    
    # Load contexts
    contexts_file = domain_dir / "contexts.json"
    with open(contexts_file, 'r') as f:
        contexts = json.load(f)
    print(f"✓ Loaded {len(contexts)} contexts")
    
    # Parse JSON arrays
    ts_df['history'] = ts_df['history'].apply(json.loads)
    ts_df['future'] = ts_df['future'].apply(json.loads)
    
    tasks_data = []
    for idx, row in ts_df.iterrows():
        tasks_data.append({
            'id': row['id'],
            'history': row['history'],
            'future': row['future'],
            'domain': domain_name
        })
        print(f"  ✓ {row['id']}: {len(row['history'])} history, {len(row['future'])} future")
    
    print(f"\n✓ Successfully loaded {len(tasks_data)} tasks\n")
    return tasks_data, contexts


def mean_absolute_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def normalized_mae(y_true, y_pred):
    mean_actual = np.mean(np.abs(y_true))
    if mean_actual == 0:
        return np.nan
    return mean_absolute_error(y_true, y_pred) / mean_actual


def directional_accuracy(y_true, y_pred, last_value):
    if len(y_true) == 0 or len(y_pred) == 0:
        return np.nan
    
    actual_direction = np.sign(y_true - last_value)
    pred_direction = np.sign(y_pred - last_value)
    non_zero_mask = actual_direction != 0
    
    if non_zero_mask.sum() == 0:
        return np.nan
    
    correct = (actual_direction[non_zero_mask] == pred_direction[non_zero_mask]).sum()
    return correct / non_zero_mask.sum()


def create_forecast_prompt(history, horizon, context=None):
    """Create a simple forecasting prompt"""
    history_sample = history[-50:] if len(history) > 50 else history
    history_str = ", ".join([f"{x:.2f}" for x in history_sample])
    
    if context:
        prompt = f"""You are a time series forecasting expert.

Context: {context}

Historical values (most recent {len(history_sample)}): {history_str}

Task: Predict the next {horizon} values considering the context and pattern.

Output ONLY {horizon} comma-separated numbers, no text. Example: 1.23, 4.56, 7.89

Your {horizon} predictions:"""
    else:
        prompt = f"""You are a time series forecasting expert.

Historical values (most recent {len(history_sample)}): {history_str}

Task: Predict the next {horizon} values based on the pattern.

Output ONLY {horizon} comma-separated numbers, no text. Example: 1.23, 4.56, 7.89

Your {horizon} predictions:"""
    
    return prompt


def forecast_with_mistral(model, tokenizer, history, horizon, context=None):
    """Generate forecast using Mistral"""
    import torch
    try:
        prompt = create_forecast_prompt(history, horizon, context)
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=200,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        predictions = []
        import re
        numbers = re.findall(r'-?\d+\.?\d*', response)
        
        for num_str in numbers[:horizon]:
            try:
                predictions.append(float(num_str))
            except:
                continue
        
        if len(predictions) < horizon:
            last_val = predictions[-1] if predictions else history[-1]
            while len(predictions) < horizon:
                predictions.append(last_val)
        
        return np.array(predictions[:horizon])
    
    except Exception as e:
        print(f"    Error in forecast: {e}")
        return np.full(horizon, history[-1])


def load_mistral_model(model_id):
    """Load Mistral model"""
    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    print(f"\nLoading Mistral: {model_id}")
    print("This may take a few minutes...")
    
    if torch.cuda.is_available():
        device = "cuda"
        print(f"✓ Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("⚠ Using CPU - this will be VERY slow!")
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading model weights...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16 if device == "cuda" else torch.float32,
        device_map="auto" if device == "cuda" else None,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    
    if device == "cpu":
        model = model.to(device)
    
    print(f"✓ Mistral loaded on {device}!\n")
    
    return model, tokenizer


def run_mistral_evaluation(tasks_data, contexts, use_context, domain_name, model_id):
    """Run Mistral on all tasks"""
    import torch
    
    context_str = "WITH CONTEXT" if use_context else "NO CONTEXT"
    
    print("=" * 80)
    print(f"RUNNING MISTRAL - {context_str}")
    print(f"Domain: {domain_name}")
    print("=" * 80)
    
    model, tokenizer = load_mistral_model(model_id)
    
    results = []
    
    for task_data in tqdm(tasks_data, desc=f"Mistral-{context_str}"):
        task_id = task_data['id']
        history = np.array(task_data['history'])
        future = np.array(task_data['future'])
        context = contexts.get(task_id, "") if use_context else None
        
        try:
            horizon = len(future)
            predictions = forecast_with_mistral(model, tokenizer, history, horizon, context)
            
            mae = mean_absolute_error(future, predictions)
            nmae = normalized_mae(future, predictions)
            da = directional_accuracy(future, predictions, history[-1])
            
            results.append({
                'task_id': task_id,
                'mae': mae,
                'nmae': nmae,
                'da': da,
                'status': 'success',
                'horizon': horizon
            })
            
        except Exception as e:
            print(f"    ✗ Error on {task_id}: {str(e)[:100]}")
            results.append({
                'task_id': task_id,
                'mae': np.nan,
                'nmae': np.nan,
                'da': np.nan,
                'status': f'error: {str(e)[:100]}',
                'horizon': len(future)
            })
    
    results_df = pd.DataFrame(results)
    
    print(f"\n{'='*80}")
    print(f"MISTRAL - {context_str} Summary:")
    print(f"  Total: {len(results_df)}")
    print(f"  Success: {(results_df['status'] == 'success').sum()}")
    print(f"  Failed: {(results_df['status'] != 'success').sum()}")
    
    successful = results_df[results_df['status'] == 'success']
    if len(successful) > 0:
        print(f"\n  Metrics (mean ± std):")
        print(f"    MAE:   {successful['mae'].mean():8.2f} ± {successful['mae'].std():6.2f}")
        print(f"    nMAE:  {successful['nmae'].mean():8.4f} ± {successful['nmae'].std():6.4f}")
        print(f"    DA:    {successful['da'].mean():8.4f} ± {successful['da'].std():6.4f}")
    print(f"{'='*80}\n")
    
    return results_df


def save_results(results_df, use_context, domain_name):
    """Save results"""
    print("=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
    domain_dir = Path("Results") / domain_name
    domain_dir.mkdir(parents=True, exist_ok=True)
    
    context_suffix = "with_context" if use_context else "no_context"
    filename = f"mistral_{context_suffix}_results.csv"
    
    result_path = domain_dir / filename
    results_df.to_csv(result_path, index=False)
    print(f"✓ Saved to {result_path}")
    print(f"{'='*80}\n")


def main():
    import torch
    
    parser = argparse.ArgumentParser(description='Run Mistral on one domain')
    parser.add_argument('--domain', type=str, required=True)
    parser.add_argument('--use-context', dest='use_context', action='store_true')
    parser.add_argument('--no-context', dest='use_context', action='store_false')
    parser.add_argument('--model', type=str, default='mistralai/Mistral-Large-Instruct-2407')
    parser.set_defaults(use_context=True)
    
    args = parser.parse_args()
    
    print("\n" + "#" * 80)
    print("#" + " " * 78 + "#")
    print("#" + "  MISTRAL STANDALONE - SINGLE DOMAIN".center(78) + "#")
    print("#" + f"  Domain: {args.domain}".center(78) + "#")
    context_str = "WITH CONTEXT" if args.use_context else "NO CONTEXT"
    print("#" + f"  {context_str}".center(78) + "#")
    print("#" + " " * 78 + "#")
    print("#" * 80)
    print(f"\nStarted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    print(f"\nEnvironment:")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print()
    
    tasks_data, contexts = load_domain_tasks(args.domain)
    
    results_df = run_mistral_evaluation(
        tasks_data, contexts, args.use_context, args.domain, args.model
    )
    
    save_results(results_df, args.use_context, args.domain)
    
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"\nDomain: {args.domain}")
    print(f"Mistral - {context_str}")
    
    successful = results_df[results_df['status'] == 'success']
    if len(successful) > 0:
        print(f"  MAE:   {successful['mae'].mean():.2f}")
        print(f"  nMAE:  {successful['nmae'].mean():.4f}")
        print(f"  DA:    {successful['da'].mean():.4f}")
        print(f"  Success: {len(successful)/len(results_df)*100:.1f}%")
    print("=" * 80)
    
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"✓ Results saved to Results/{args.domain}/\n")


if __name__ == "__main__":
    main()