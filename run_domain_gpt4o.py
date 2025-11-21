"""
Generate tasks for ONE domain and run GPT-4o Mini

Usage:
    python run_domain_gpt4o.py --domain ConstrainedRandomWalk --use-context
    python run_domain_gpt4o.py --domain ConstrainedRandomWalk --no-context
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
from datetime import datetime
from tqdm import tqdm
import time
import warnings
warnings.filterwarnings('ignore')

# Add CiK benchmark to path
cik_path = Path("context-is-key-forecasting")
sys.path.insert(0, str(cik_path))

try:
    from openai import OpenAI
except ImportError:
    print("ERROR: openai package not installed")
    print("Run: pip install openai")
    sys.exit(1)


# DOMAIN MAPPING - matches your generate_cik_dataset.py
DOMAIN_GENERATORS = {
    "SolarPowerProduction": ("MinimalInfoHalfDaySolarForecastTask", "cik_benchmark.tasks.solar_tasks"),
    "UnemploymentCountyUsingSingleStateData": ("UnemploymentCountyUsingSingleStateData", "cik_benchmark.tasks.fred_county_tasks"),
    "SpeedFromLoadTask": ("SpeedFromLoadTask", "cik_benchmark.tasks.causal_chambers"),
    "ElectricityIncreaseInPredictionTask": ("ElectricityIncreaseInPredictionTask", "cik_benchmark.tasks.electricity_tasks"),
    "DecreaseInTraffic": ("DecreaseInTrafficInPredictionTask", "cik_benchmark.tasks.pred_change_tasks"),
    "PredictableSpikes": ("PredictableSpikesInPredTask", "cik_benchmark.tasks.predictable_spikes_in_pred"),
    "OracleePredUnivariateConstraints": ("OraclePredUnivariateConstraintsTask", "cik_benchmark.tasks.predictable_constraints_real_data"),
    "DirectNormalIrradianceFromCloudStatus": ("DirectNormalIrradianceFromCloudStatus", "cik_benchmark.tasks.nsrdb_tasks"),
    "SensorMaintenance": ("SensorMaintenanceInPredictionTask", "cik_benchmark.tasks.sensor_maintenance"),
    "CashDepletedinATMScenario": ("CashDepletedinATMScenarioTask", "cik_benchmark.tasks.nn5_tasks"),
    "LaneClosureAfterShortHistoryMediumBackground": ("LaneClosureAfterShortHistoryMediumBackgroundTask", "cik_benchmark.tasks.pems_tasks"),
    "ImplicitTrafficForecastTaskwithHolidays": ("ImplicitTrafficForecastTaskwithHolidaysInPredictionWindow", "cik_benchmark.tasks.traffic_tasks"),
    "PreedictableGrocerPersistentShockUnivariate": ("PredictableGrocerPersistentShockUnivariateGroceryTask", "cik_benchmark.tasks.predictable_grocer_shocks"),
    "STLPredTrendMultiplierWithMediumDescription": ("STLPredTrendMultiplierWithMediumDescriptionTask", "cik_benchmark.tasks.predictable_stl_shocks"),
    "FullCausalContextExplicitEquationBivarLinSVAR": ("FullCausalContextExplicitEquationBivarLinSVAR", "cik_benchmark.tasks.bivariate_categorical_causal"),
    "ConstrainedRandomWalk": ("ConstrainedRandomWalk", "cik_benchmark.tasks.constrained_forecasts"),
}


def extract_all_context(task):
    """Extract all available context fields from task"""
    context_parts = []
    
    if hasattr(task, "background") and task.background:
        context_parts.append(f"[BACKGROUND] {task.background}")
    
    if hasattr(task, "scenario") and task.scenario:
        context_parts.append(f"[SCENARIO] {task.scenario}")
    
    if hasattr(task, "constraints") and task.constraints:
        context_parts.append(f"[CONSTRAINTS] {task.constraints}")
    
    if hasattr(task, "causal_context") and task.causal_context:
        if callable(task.causal_context):
            try:
                context_parts.append(f"[CAUSAL] {task.causal_context()}")
            except:
                pass
        else:
            context_parts.append(f"[CAUSAL] {task.causal_context}")
    
    return " || ".join(context_parts) if context_parts else "No textual context"


def generate_domain_tasks(domain_name, num_tasks=10, random_seed=42):
    """Generate tasks for a specific domain"""
    
    if domain_name not in DOMAIN_GENERATORS:
        print(f"ERROR: Unknown domain: {domain_name}")
        print(f"Available domains: {list(DOMAIN_GENERATORS.keys())}")
        sys.exit(1)
    
    print("=" * 80)
    print(f"GENERATING TASKS FOR: {domain_name}")
    print("=" * 80)
    
    # Import the task generator
    class_name, module_name = DOMAIN_GENERATORS[domain_name]
    module = __import__(module_name, fromlist=[class_name])
    task_class = getattr(module, class_name)
    
    print(f"✓ Loaded {class_name} from {module_name}")
    print(f"  Generating {num_tasks} tasks...\n")
    
    np.random.seed(random_seed)
    
    tasks_data = []
    contexts = {}
    
    for i in range(num_tasks):
        try:
            seed = random_seed + i
            task = task_class(seed=seed)
            
            # Extract data
            if hasattr(task.past_time, "iloc"):
                history = task.past_time.iloc[:, -1].values.tolist()
                future = task.future_time.iloc[:, -1].values.tolist()
            else:
                history = task.past_time.values.tolist()
                future = task.future_time.values.tolist()
            
            context = extract_all_context(task)
            task_id = f"task_{i:03d}"
            
            tasks_data.append({
                'id': task_id,
                'history': history,
                'future': future,
                'domain': domain_name
            })
            
            contexts[task_id] = context
            
            print(f"  ✓ Generated {task_id}: {len(history)} history, {len(future)} future")
            
        except Exception as e:
            print(f"  ✗ Failed task {i}: {e}")
            continue
    
    print(f"\n✓ Successfully generated {len(tasks_data)} tasks for {domain_name}\n")
    
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


def forecast_with_gpt4o_mini(client, history, horizon, context=None, max_retries=3):
    """Generate forecast using GPT-4o Mini"""
    history_sample = history[-50:] if len(history) > 50 else history
    history_str = ", ".join([f"{x:.4f}" for x in history_sample])
    
    if context:
        prompt = f"""You are a time series forecasting expert.

**Context Information:**
{context}

**Historical Data (most recent {len(history_sample)} values):**
[{history_str}]

**Task:**
Based on the context and historical pattern, predict the next {horizon} values.

**CRITICAL INSTRUCTIONS:**
- Output ONLY {horizon} comma-separated numbers
- No explanation, no text, just numbers
- Use the context information to inform your predictions
- Format: 1.23, 4.56, 7.89, ...

**Your {horizon} predictions:**"""
    else:
        prompt = f"""You are a time series forecasting expert.

**Historical Data (most recent {len(history_sample)} values):**
[{history_str}]

**Task:**
Predict the next {horizon} values based on the observed pattern.

**CRITICAL INSTRUCTIONS:**
- Output ONLY {horizon} comma-separated numbers
- No explanation, no text, just numbers
- Format: 1.23, 4.56, 7.89, ...

**Your {horizon} predictions:**"""
    
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are a precise time series forecasting assistant. Always output exactly the requested number of numeric predictions, separated by commas, with no additional text."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.1,
                max_tokens=500,
                timeout=30
            )
            
            predictions_str = response.choices[0].message.content.strip()
            predictions_str = predictions_str.replace('```', '').replace('json', '').strip()
            
            predictions = []
            for part in predictions_str.split(','):
                try:
                    import re
                    numbers = re.findall(r'-?\d+\.?\d*', part)
                    if numbers:
                        predictions.append(float(numbers[0]))
                except (ValueError, IndexError):
                    continue
            
            if len(predictions) >= horizon:
                return np.array(predictions[:horizon])
            elif len(predictions) > 0:
                while len(predictions) < horizon:
                    predictions.append(predictions[-1])
                return np.array(predictions)
            else:
                raise ValueError(f"Could not parse predictions")
                
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                raise
    
    raise Exception("Max retries exceeded")


def run_gpt4o_mini(tasks_data, contexts, use_context, domain_name):
    """Run GPT-4o Mini on all tasks"""
    context_str = "WITH CONTEXT" if use_context else "NO CONTEXT"
    
    print("=" * 80)
    print(f"RUNNING GPT-4o-MINI - {context_str}")
    print(f"Domain: {domain_name}")
    print("=" * 80)
    
    # Initialize OpenAI client
    api_key = "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"
    if not api_key:
        config_file = Path("openai_key.txt")
        if config_file.exists():
            api_key = config_file.read_text().strip()
            print("✓ Loaded API key from openai_key.txt")
        else:
            print("\nERROR: OPENAI_API_KEY not found!")
            print("Option 1: export OPENAI_API_KEY='your-key'")
            print("Option 2: Create openai_key.txt with your key")
            sys.exit(1)
    
    client = OpenAI(api_key=api_key)
    print(f"✓ OpenAI client initialized\n")
    
    results = []
    
    for task_data in tqdm(tasks_data, desc=f"GPT-4o-Mini"):
        task_id = task_data['id']
        history = np.array(task_data['history'])
        future = np.array(task_data['future'])
        context = contexts.get(task_id, "")
        
        try:
            horizon = len(future)
            context_to_use = context if use_context else None
            
            predictions = forecast_with_gpt4o_mini(client, history, horizon, context_to_use)
            
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
        
        time.sleep(0.15)  # Rate limiting
    
    results_df = pd.DataFrame(results)
    
    print(f"\n{'='*80}")
    print(f"GPT-4o-MINI - {context_str} Summary:")
    print(f"  Total tasks: {len(results_df)}")
    print(f"  Successful: {(results_df['status'] == 'success').sum()}")
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
    """Save results to domain-specific results folder"""
    print("=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)
    
    domain_dir = Path("results") / domain_name
    domain_dir.mkdir(parents=True, exist_ok=True)
    
    context_suffix = "with_context" if use_context else "no_context"
    filename = f"gpt4o_mini_{context_suffix}_results.csv"
    
    result_path = domain_dir / filename
    results_df.to_csv(result_path, index=False)
    print(f"✓ Saved results to {result_path}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(description='Generate tasks and run GPT-4o Mini for ONE domain')
    parser.add_argument('--domain', type=str, required=True,
                       help='Domain name (e.g., ConstrainedRandomWalk)')
    parser.add_argument('--use-context', dest='use_context', action='store_true',
                       help='Use textual context')
    parser.add_argument('--no-context', dest='use_context', action='store_false',
                       help='Do not use textual context')
    parser.add_argument('--num-tasks', type=int, default=10,
                       help='Number of tasks to generate (default: 10)')
    parser.set_defaults(use_context=True)
    
    args = parser.parse_args()
    
    print("\n")
    print("#" * 80)
    print("#" + " " * 78 + "#")
    print("#" + "  GPT-4o MINI - SINGLE DOMAIN".center(78) + "#")
    print("#" + f"  Domain: {args.domain}".center(78) + "#")
    context_str = "WITH CONTEXT" if args.use_context else "NO CONTEXT"
    print("#" + f"  {context_str}".center(78) + "#")
    print("#" + " " * 78 + "#")
    print("#" * 80)
    print(f"\nStarted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Step 1: Generate tasks
    tasks_data, contexts = generate_domain_tasks(args.domain, args.num_tasks)
    
    # Step 2: Run GPT-4o Mini
    results_df = run_gpt4o_mini(tasks_data, contexts, args.use_context, args.domain)
    
    # Step 3: Save results
    save_results(results_df, args.use_context, args.domain)
    
    # Final summary
    print("=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"\nDomain: {args.domain}")
    print(f"GPT-4o-MINI - {context_str}")
    
    successful = results_df[results_df['status'] == 'success']
    if len(successful) > 0:
        print(f"  MAE:   {successful['mae'].mean():.2f}")
        print(f"  nMAE:  {successful['nmae'].mean():.4f}")
        print(f"  DA:    {successful['da'].mean():.4f}")
        print(f"  Success Rate: {len(successful)/len(results_df) * 100:.1f}%")
    print("=" * 80)
    
    print(f"\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n✓ Complete! Results saved to results/{args.domain}/")
    print()


if __name__ == "__main__":
    main()