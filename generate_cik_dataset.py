"""
Day 1: Generate 50 diverse tasks from Context-is-Key benchmark

CiK task generators automatically download underlying datasets (solar, electricity, traffic)
on first use and cache them. We just need to instantiate tasks with different seeds.

This script:
1. Instantiates task generators from CiK (datasets auto-download)
2. Generates 50 tasks with different seeds
3. Saves to datasets/ in required format
4. Creates train/test split (35/15)
5. Plots samples for verification

Author: Kazi
Date: Nov 2024
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# Add CiK benchmark to path
cik_path = Path("context-is-key-forecasting")
sys.path.insert(0, str(cik_path))

print("="*70)
print("Day 1: CiK Task Generation")
print("="*70)
print("\nImporting CiK task generators...")
print("Note: First run will download datasets (solar, electricity, traffic)")
print("This may take 5-10 minutes. Subsequent runs will use cached data.\n")

# Import diverse task generators with CORRECT names
from cik_benchmark.tasks.solar_tasks import (
    MinimalInfoHalfDaySolarForecastTask,
    LocaleInfoHalfDaySolarForecastTask,
    ZenithInfoHalfDaySolarForecastTask,
)
from cik_benchmark.tasks.nsrdb_tasks import (
    DirectNormalIrradianceFromCloudStatus,
    GlobalHorizontalIrradianceFromClearsky,
)
from cik_benchmark.tasks.electricity_tasks import (
    ElectricityIncreaseInPredictionTask,
    ShortNewsElectricityIncreaseInPredictionTask,
    MediumNewsElectricityIncreaseInPredictionTask,
)
from cik_benchmark.tasks.sensor_maintenance import (
    SensorMaintenanceInPredictionTask,
)
from cik_benchmark.tasks.traffic_tasks import (
    ExplicitTrafficForecastTaskwithHolidaysInPredictionWindow,
)
from cik_benchmark.tasks.pred_change_tasks import (
    DecreaseInTrafficInPredictionTask,
)

print("✓ Imports successful!\n")

# Define task generators with target counts (total = 50)
# Format: (TaskClass, count, short_name)
TASK_CONFIG = [
    # Solar tasks - 15 tasks (30%)
    (MinimalInfoHalfDaySolarForecastTask, 5, "solar_minimal"),
    (LocaleInfoHalfDaySolarForecastTask, 5, "solar_locale"),
    (ZenithInfoHalfDaySolarForecastTask, 5, "solar_zenith"),
    
    # NSRDB solar irradiance - 8 tasks (16%)
    (DirectNormalIrradianceFromCloudStatus, 4, "irradiance_dni"),
    (GlobalHorizontalIrradianceFromClearsky, 4, "irradiance_ghi"),
    
    # Electricity - 12 tasks (24%)
    (ElectricityIncreaseInPredictionTask, 6, "electricity_spike"),
    (ShortNewsElectricityIncreaseInPredictionTask, 3, "electricity_short_news"),
    (MediumNewsElectricityIncreaseInPredictionTask, 3, "electricity_med_news"),
    
    # Sensor & Traffic - 9 tasks (18%)
    (SensorMaintenanceInPredictionTask, 5, "sensor_maintenance"),
    (ExplicitTrafficForecastTaskwithHolidaysInPredictionWindow, 4, "traffic_holiday"),
    
    # Traffic decrease - 6 tasks (12%)
    (DecreaseInTrafficInPredictionTask, 6, "traffic_decrease"),
]

# Verify total
total_tasks = sum(count for _, count, _ in TASK_CONFIG)
print(f"Configuration: {total_tasks} total tasks across {len(TASK_CONFIG)} task types\n")


def extract_all_context(task):
    """Extract all available context fields from task"""
    context_parts = []
    
    if hasattr(task, 'background') and task.background:
        context_parts.append(f"[BACKGROUND] {task.background}")
    
    if hasattr(task, 'scenario') and task.scenario:
        context_parts.append(f"[SCENARIO] {task.scenario}")
    
    if hasattr(task, 'constraints') and task.constraints:
        context_parts.append(f"[CONSTRAINTS] {task.constraints}")
        
    if hasattr(task, 'causal_context') and task.causal_context:
        if callable(task.causal_context):
            try:
                context_parts.append(f"[CAUSAL] {task.causal_context()}")
            except:
                pass
        else:
            context_parts.append(f"[CAUSAL] {task.causal_context}")
    
    return " || ".join(context_parts) if context_parts else "No textual context"


def analyze_series_stats(values):
    """Compute statistics for categorization"""
    values = np.array(values)
    
    mean_val = np.mean(values)
    std_val = np.std(values)
    
    # Trend (linear slope)
    x = np.arange(len(values))
    trend = np.polyfit(x, values, 1)[0] if len(values) > 1 else 0
    
    # Volatility (coefficient of variation)
    cv = std_val / (abs(mean_val) + 1e-8)
    
    # Structural break score
    mid = len(values) // 2
    if mid > 2:
        first_mean = np.mean(values[:mid])
        second_mean = np.mean(values[mid:])
        break_score = abs(second_mean - first_mean) / (std_val + 1e-8)
    else:
        break_score = 0
    
    return {
        'mean': float(mean_val),
        'std': float(std_val),
        'trend': float(trend),
        'volatility': float(cv),
        'structural_break': float(break_score)
    }


def categorize_series(stats):
    """Categorize time series based on characteristics"""
    # Simple heuristics
    if stats['structural_break'] > 1.0:
        return 'structural_break'
    elif abs(stats['trend']) > 0.1 * stats['std']:
        return 'trending'
    elif stats['volatility'] > 0.5:
        return 'volatile'
    else:
        return 'seasonal'


def generate_tasks(random_seed=42):
    """
    Generate tasks from CiK benchmark using different task generators
    
    Returns:
    - tasks: List of task instances
    - metadata_df: DataFrame with task metadata
    """
    np.random.seed(random_seed)
    
    tasks = []
    metadata = []
    
    task_id = 0
    
    print("="*70)
    print("TASK GENERATION")
    print("="*70)
    
    for task_class, count, short_name in TASK_CONFIG:
        print(f"\n{task_class.__name__}: Generating {count} tasks...")
        
        for i in range(count):
            try:
                # Generate task with unique seed
                seed = random_seed + task_id
                task = task_class(seed=seed)
                
                # Extract data - use last column which is typically the target
                history = task.past_time.iloc[:, -1].values
                future = task.future_time.iloc[:, -1].values
                context = extract_all_context(task)
                
                # Analyze characteristics
                stats = analyze_series_stats(history)
                category = categorize_series(stats)
                
                # Store task object
                tasks.append(task)
                
                # Store metadata
                metadata.append({
                    'id': f"task_{task_id:03d}",
                    'generator': task_class.__name__,
                    'short_name': short_name,
                    'seed': seed,
                    'history_length': len(history),
                    'future_length': len(future),
                    'category': category,
                    'mean': stats['mean'],
                    'std': stats['std'],
                    'trend': stats['trend'],
                    'volatility': stats['volatility'],
                    'structural_break': stats['structural_break'],
                    'context_length': len(context),
                    'context_preview': context[:150] + "..." if len(context) > 150 else context
                })
                
                task_id += 1
                
                if task_id % 5 == 0:
                    print(f"  Progress: {task_id}/{total_tasks} tasks completed")
                    
            except Exception as e:
                print(f"  ⚠️  Warning: Failed to generate task: {e}")
                continue
    
    print(f"\n✓ Successfully generated {len(tasks)} tasks")
    
    # Create metadata DataFrame
    df_metadata = pd.DataFrame(metadata)
    
    # Print diversity statistics
    print("\n" + "="*70)
    print("DIVERSITY STATISTICS")
    print("="*70)
    print("\nCategory distribution:")
    print(df_metadata['category'].value_counts())
    print(f"\nTask type distribution:")
    print(df_metadata['short_name'].value_counts())
    print(f"\nHistory length: min={df_metadata['history_length'].min()}, "
          f"max={df_metadata['history_length'].max()}, "
          f"mean={df_metadata['history_length'].mean():.1f}")
    print(f"Future length: min={df_metadata['future_length'].min()}, "
          f"max={df_metadata['future_length'].max()}, "
          f"mean={df_metadata['future_length'].mean():.1f}")
    print(f"Context length: min={df_metadata['context_length'].min()}, "
          f"max={df_metadata['context_length'].max()}, "
          f"mean={df_metadata['context_length'].mean():.1f}")
    
    return tasks, df_metadata


def save_tasks(tasks, metadata_df, output_dir='datasets'):
    """
    Save tasks in the required format:
    - ts_instances.csv: Time series data
    - contexts.json: Context information
    - task_metadata.csv: Full metadata
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("SAVING DATA")
    print("="*70)
    
    # Prepare data structures
    ts_instances = []
    contexts = {}
    
    for idx, task in enumerate(tasks):
        task_id = metadata_df.iloc[idx]['id']
        
        # Extract time series data
        history = task.past_time.iloc[:, -1].values.tolist()
        future = task.future_time.iloc[:, -1].values.tolist()
        
        # Store in ts_instances
        ts_instances.append({
            'id': task_id,
            'history': history,
            'future': future,
            'length': len(history) + len(future),
            'type': metadata_df.iloc[idx]['category']
        })
        
        # Extract and store context
        contexts[task_id] = extract_all_context(task)
    
    # Save ts_instances.csv
    df_ts = pd.DataFrame(ts_instances)
    csv_path = os.path.join(output_dir, 'ts_instances.csv')
    
    # Convert lists to string for CSV storage
    df_ts['history'] = df_ts['history'].apply(json.dumps)
    df_ts['future'] = df_ts['future'].apply(json.dumps)
    df_ts.to_csv(csv_path, index=False)
    print(f"✓ Saved time series data to {csv_path}")
    
    # Save contexts.json
    json_path = os.path.join(output_dir, 'contexts.json')
    with open(json_path, 'w') as f:
        json.dump(contexts, f, indent=2)
    print(f"✓ Saved contexts to {json_path}")
    
    # Save metadata
    metadata_path = os.path.join(output_dir, 'task_metadata.csv')
    metadata_df.to_csv(metadata_path, index=False)
    print(f"✓ Saved metadata to {metadata_path}")
    
    return df_ts, contexts


def create_train_test_split(metadata_df, train_size=35, test_size=15, random_seed=42):
    """
    Create stratified train/test split ensuring diversity in both sets
    """
    np.random.seed(random_seed)
    
    print("\n" + "="*70)
    print("TRAIN/TEST SPLIT")
    print("="*70)
    
    # Stratify by category to ensure diversity
    categories = metadata_df['category'].unique()
    
    train_ids = []
    test_ids = []
    
    for category in categories:
        cat_tasks = metadata_df[metadata_df['category'] == category]['id'].values
        
        # Determine how many from this category
        n_cat = len(cat_tasks)
        n_cat_train = int(n_cat * train_size / (train_size + test_size))
        
        # Shuffle and split
        cat_tasks_shuffled = cat_tasks.copy()
        np.random.shuffle(cat_tasks_shuffled)
        
        train_ids.extend(cat_tasks_shuffled[:n_cat_train].tolist())
        test_ids.extend(cat_tasks_shuffled[n_cat_train:].tolist())
    
    # Trim to exact sizes
    np.random.shuffle(train_ids)
    np.random.shuffle(test_ids)
    train_ids = train_ids[:train_size]
    test_ids = test_ids[:test_size]
    
    split = {
        'train': train_ids,
        'test': test_ids,
        'train_size': len(train_ids),
        'test_size': len(test_ids)
    }
    
    # Save split
    split_path = 'datasets/train_test_split.json'
    with open(split_path, 'w') as f:
        json.dump(split, f, indent=2)
    
    print(f"Train: {len(train_ids)} tasks")
    print(f"Test: {len(test_ids)} tasks")
    
    # Show category distribution in each split
    train_meta = metadata_df[metadata_df['id'].isin(train_ids)]
    test_meta = metadata_df[metadata_df['id'].isin(test_ids)]
    
    print(f"\nTrain set category distribution:")
    print(train_meta['category'].value_counts())
    print(f"\nTest set category distribution:")
    print(test_meta['category'].value_counts())
    
    print(f"\n✓ Saved split to {split_path}")
    
    return split


def plot_sample_tasks(tasks, metadata_df, n_samples=5, output_dir='datasets/plots'):
    """
    Plot a few sample tasks for quality verification
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "="*70)
    print("PLOTTING SAMPLES")
    print("="*70)
    
    # Randomly select tasks
    n_samples = min(n_samples, len(tasks))
    sample_indices = np.random.choice(len(tasks), n_samples, replace=False)
    
    print(f"Plotting {n_samples} sample tasks...")
    
    for idx in sample_indices:
        task = tasks[idx]
        task_id = metadata_df.iloc[idx]['id']
        category = metadata_df.iloc[idx]['category']
        generator = metadata_df.iloc[idx]['short_name']
        
        # Create figure
        fig, ax = plt.subplots(figsize=(14, 6))
        
        # Extract data
        history = task.past_time.iloc[:, -1].values
        future = task.future_time.iloc[:, -1].values
        
        # Create time indices
        hist_idx = np.arange(len(history))
        future_idx = np.arange(len(history), len(history) + len(future))
        
        # Plot
        ax.plot(hist_idx, history, 'b-', linewidth=2, label='History', alpha=0.8)
        ax.plot(future_idx, future, 'r-', linewidth=2, label='Future (Ground Truth)', alpha=0.8)
        ax.axvline(x=len(history)-0.5, color='k', linestyle='--', alpha=0.5, linewidth=1.5)
        
        # Add context as title
        context = extract_all_context(task)
        title = f"{task_id} | Type: {generator} | Category: {category}\n"
        title += context[:200] + "..." if len(context) > 200 else context
        ax.set_title(title, fontsize=9, wrap=True)
        ax.set_xlabel('Time Step', fontsize=10)
        ax.set_ylabel('Value', fontsize=10)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save
        plot_path = os.path.join(output_dir, f'{task_id}_sample.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✓ Saved {task_id}")
    
    print(f"\n✓ All sample plots saved to {output_dir}/")


def main():
    """Main execution function"""
    
    print("\nStarting Day 1 task generation...")
    print("This will take 10-30 minutes on first run (downloads datasets)")
    print("Subsequent runs will be fast (<1 min)\n")
    
    # Step 1: Generate tasks
    tasks, metadata_df = generate_tasks(random_seed=42)
    
    # Step 2: Save tasks
    ts_df, contexts = save_tasks(tasks, metadata_df)
    
    # Step 3: Create train/test split
    split = create_train_test_split(metadata_df, train_size=35, test_size=15)
    
    # Step 4: Plot samples
    plot_sample_tasks(tasks, metadata_df, n_samples=5)
    
    print("\n" + "="*70)
    print("DAY 1 COMPLETE!")
    print("="*70)
    print(f"\n✓ Generated {len(tasks)} tasks")
    print(f"✓ Created {split['train_size']} train / {split['test_size']} test split")
    print(f"✓ Saved to datasets/ folder")
    print(f"✓ Sample plots in datasets/plots/")
    print("\nNext: Move to Day 3 (model experiments)")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
