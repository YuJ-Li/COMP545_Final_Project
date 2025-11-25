"""
Day 1: Generate 32 diverse tasks from Context-is-Key benchmark

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

# Add CiK benchmark to path
cik_path = Path("context-is-key-forecasting")
sys.path.insert(0, str(cik_path))

print("="*70)
print("Day 1: CiK Task Generation - 16 Diverse Domains")
print("="*70)
print("\nImporting CiK task generators...")
print("Note: First run will download datasets")
print("This may take 5-10 minutes. Subsequent runs will use cached data.\n")

'''
# Import diverse task generators from DIFFERENT data sources
from cik_benchmark.tasks.solar_tasks import (
    MinimalInfoHalfDaySolarForecastTask,
)
from cik_benchmark.tasks.fred_county_tasks import (
    UnemploymentCountyUsingSingleStateData,
)
from cik_benchmark.tasks.causal_chambers import (
    SpeedFromLoadTask,
)
from cik_benchmark.tasks.electricity_tasks import (
    ElectricityIncreaseInPredictionTask,
)
from cik_benchmark.tasks.pred_change_tasks import (
    DecreaseInTrafficInPredictionTask,
)
from cik_benchmark.tasks.predictable_spikes_in_pred import (
    PredictableSpikesInPredTask,
)
from cik_benchmark.tasks.predictable_constraints_real_data import (
    OraclePredUnivariateConstraintsTask,
)
from cik_benchmark.tasks.nsrdb_tasks import (
    DirectNormalIrradianceFromCloudStatus,
)
from cik_benchmark.tasks.sensor_maintenance import (
    SensorMaintenanceInPredictionTask,
)
from cik_benchmark.tasks.nn5_tasks import (
    CashDepletedinATMScenarioTask,
)
from cik_benchmark.tasks.pems_tasks import (
    LaneClosureAfterShortHistoryMediumBackgroundTask,
)
from cik_benchmark.tasks.traffic_tasks import (
    ImplicitTrafficForecastTaskwithHolidaysInPredictionWindow,
)
from cik_benchmark.tasks.predictable_grocer_shocks import (
    PredictableGrocerPersistentShockUnivariateGroceryTask,
)
from cik_benchmark.tasks.predictable_stl_shocks import (
    STLPredTrendMultiplierWithMediumDescriptionTask,
)
from cik_benchmark.tasks.bivariate_categorical_causal import (
    FullCausalContextExplicitEquationBivarLinSVAR,
)
from cik_benchmark.tasks.constrained_forecasts import (
    ConstrainedRandomWalk,
)

print("✓ Imports successful!\n")
'''
# DIVERSE TASK CONFIGURATION: 16 domains × 2 tasks = 32 total
# Each domain uses a DIFFERENT underlying dataset
TASK_CONFIG = [
    # # 1. Solar power production (half-day forecasts, Georgia plant)
    # # 3 Improving DA && 3 Worse && 4 Equal
    # (MinimalInfoHalfDaySolarForecastTask, 10, "solar_plant"),

    # # 2. Economic data (FRED county unemployment)
    # # 2 Improving DA && 8 Equal
     (UnemploymentCountyUsingSingleStateData, 10, "economic_unemployment"),

    # # 3. Physics experiments (Wind tunnel causal chamber)
    # # 4 Improving DA && 3 Worse && 3 Equal
    # (SpeedFromLoadTask, 10, "wind_tunnel"),

    # # 4. Electricity demand spikes due to heatwave
    # # 4 Improving DA && 3 Worse && 3 Equal
    # (ElectricityIncreaseInPredictionTask, 10, "electricity_heatwave"),

    # # 5. Traffic with accidents/closures causing drops
    # # 1 Improving DA && 4 Worse && 5 Equal
    #(DecreaseInTrafficInPredictionTask, 10, "traffic_drop"),

    # # 6. Electricity with predictable spikes (event-driven)
    # # 3 Improving DA && 3 Worse
    # (PredictableSpikesInPredTask, 10, "electricity_spikes"),

    # # 7. Real data with explicit prediction constraints - 10 tasks
    # # 6 Improving DA && 1 Worse
    # (OraclePredUnivariateConstraintsTask, 10, "real_constraints"),

    # # 8. Solar irradiance from NSRDB (DNI)
    # # 6 Improving DA && 1 Worse
    # (DirectNormalIrradianceFromCloudStatus, 10, "solar_irradiance"),

    # # 9. Sensor maintenance and outages
    # # 1 Improving DA && 4 Worse
    # (SensorMaintenanceInPredictionTask, 10, "sensor_maintenance"),

    # # 10. ATM withdrawals (NN5 dataset)
    # # 1 Improving DA && 3 Worse
    # (CashDepletedinATMScenarioTask, 10, "atm_withdrawals"),

    # # 11. Highway lane closures (PeMS)
    # # 2 Improving DA && 1 Worse
    # (LaneClosureAfterShortHistoryMediumBackgroundTask, 10, "pems_lane_closure"),

    # # 12. Traffic with holidays and covariates
    # # 4 Improving DA && 5 Worse
    # (ImplicitTrafficForecastTaskwithHolidaysInPredictionWindow, 10, "traffic_holidays"),

    # # 13. Grocery sales shocks (Predictable Grocer)
    # # 2 Improving DA && 1 Worse
    # (PredictableGrocerPersistentShockUnivariateGroceryTask, 10, "grocery_sales"),

    # # 14. STL decomposition synthetic shocks
    # # 6 Worse
    # (STLPredTrendMultiplierWithMediumDescriptionTask, 10, "stl_synthetic"),

    # # 15. Bivariate categorical causal relationships
    # # 1 Improving DA && 5 Worse
    # (FullCausalContextExplicitEquationBivarLinSVAR, 10, "bivariate_causal"),

    # 16. Fully synthetic constrained random-walk forecasts
    # Many
    #(ConstrainedRandomWalk, 10, "synthetic_constrained"),
]

# Verify total
total_tasks = sum(count for _, count, _ in TASK_CONFIG)
print(f"Configuration: {total_tasks} total tasks across {len(TASK_CONFIG)} domains\n")

print("Domain breakdown:")
for task_class, count, domain in TASK_CONFIG:
    print(f"  {domain:20s}: {count} tasks ({task_class.__name__})")
print()


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

    return {
        "mean": float(mean_val),
        "std": float(std_val),
        "trend": float(trend),
        "volatility": float(cv),
    }


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

    print("=" * 70)
    print("TASK GENERATION")
    print("=" * 70)

    for task_class, count, domain in TASK_CONFIG:
        print(f"\n{domain.upper()}: {task_class.__name__}")
        print(f"  Generating {count} tasks...")

        for i in range(count):
            try:
                # Generate task with unique seed
                seed = random_seed + task_id
                task = task_class(seed=seed)

                # Extract data - handle different formats
                if hasattr(task.past_time, "iloc"):
                    history = task.past_time.iloc[:, -1].values
                    future = task.future_time.iloc[:, -1].values
                else:
                    history = task.past_time.values
                    future = task.future_time.values

                context = extract_all_context(task)

                # Analyze characteristics
                stats = analyze_series_stats(history)

                # Store task object
                tasks.append(task)

                # Store metadata with DOMAIN label
                metadata.append(
                    {
                        "id": f"task_{task_id:03d}",
                        "domain": domain,  # DOMAIN LABEL for plotting!
                        "generator": task_class.__name__,
                        "seed": seed,
                        "history_length": len(history),
                        "future_length": len(future),
                        "mean": stats["mean"],
                        "std": stats["std"],
                        "trend": stats["trend"],
                        "volatility": stats["volatility"],
                        "context_length": len(context),
                        "context_preview": context[:150] + "..."
                        if len(context) > 150
                        else context,
                    }
                )

                task_id += 1

                if task_id % 10 == 0:
                    print(f"  Progress: {task_id}/{total_tasks} tasks completed")

            except Exception as e:
                print(f"  ⚠️  Warning: Failed to generate task: {e}")
                continue

    print(f"\n✓ Successfully generated {len(tasks)} tasks")

    # Create metadata DataFrame
    df_metadata = pd.DataFrame(metadata)

    # Print diversity statistics
    print("\n" + "=" * 70)
    print("DIVERSITY STATISTICS")
    print("=" * 70)
    print("\nDomain distribution:")
    print(df_metadata["domain"].value_counts().sort_index())
    print(
        f"\nHistory length: min={df_metadata['history_length'].min()}, "
        f"max={df_metadata['history_length'].max()}, "
        f"mean={df_metadata['history_length'].mean():.1f}"
    )
    print(
        f"Future length: min={df_metadata['future_length'].min()}, "
        f"max={df_metadata['future_length'].max()}, "
        f"mean={df_metadata['future_length'].mean():.1f}"
    )
    print(
        f"Context length: min={df_metadata['context_length'].min()}, "
        f"max={df_metadata['context_length'].max()}, "
        f"mean={df_metadata['context_length'].mean():.1f}"
    )

    return tasks, df_metadata


def save_tasks(tasks, metadata_df, output_dir="datasets"):
    """
    Save tasks in the required format:
    - ts_instances.csv: Time series data
    - contexts.json: Context information
    - task_metadata.csv: Full metadata
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("SAVING DATA")
    print("=" * 70)

    # Prepare data structures
    ts_instances = []
    contexts = {}

    for idx, task in enumerate(tasks):
        task_id = metadata_df.iloc[idx]["id"]

        # Extract time series data
        if hasattr(task.past_time, "iloc"):
            history = task.past_time.iloc[:, -1].values.tolist()
            future = task.future_time.iloc[:, -1].values.tolist()
        else:
            history = task.past_time.values.tolist()
            future = task.future_time.values.tolist()

        # Store in ts_instances
        ts_instances.append(
            {
                "id": task_id,
                "history": history,
                "future": future,
                "length": len(history) + len(future),
                "domain": metadata_df.iloc[idx]["domain"],  # Include domain
            }
        )

        # Extract and store context
        contexts[task_id] = extract_all_context(task)

    # Save ts_instances.csv
    df_ts = pd.DataFrame(ts_instances)
    csv_path = os.path.join(output_dir, "ts_instances.csv")

    # Convert lists to string for CSV storage
    df_ts["history"] = df_ts["history"].apply(json.dumps)
    df_ts["future"] = df_ts["future"].apply(json.dumps)
    df_ts.to_csv(csv_path, index=False)
    print(f"✓ Saved time series data to {csv_path}")

    # Save contexts.json
    json_path = os.path.join(output_dir, "contexts.json")
    with open(json_path, "w") as f:
        json.dump(contexts, f, indent=2)
    print(f"✓ Saved contexts to {json_path}")

    # Save metadata
    metadata_path = os.path.join(output_dir, "task_metadata.csv")
    metadata_df.to_csv(metadata_path, index=False)
    print(f"✓ Saved metadata to {metadata_path}")

    return df_ts, contexts


def create_train_test_split(metadata_df, train_size=7, test_size=3, random_seed=42):
    """
    Create stratified train/test split by domain
    """
    np.random.seed(random_seed)

    print("\n" + "=" * 70)
    print("TRAIN/TEST SPLIT (STRATIFIED BY DOMAIN)")
    print("=" * 70)

    train_ids = []
    test_ids = []

    # Stratify by domain
    domains = metadata_df["domain"].unique()

    for domain in domains:
        domain_tasks = metadata_df[metadata_df["domain"] == domain]["id"].values

        # 70% train, 30% test
        n_domain = len(domain_tasks)
        n_domain_train = int(n_domain * 0.7)

        # Shuffle and split
        domain_shuffled = domain_tasks.copy()
        np.random.shuffle(domain_shuffled)

        train_ids.extend(domain_shuffled[:n_domain_train].tolist())
        test_ids.extend(domain_shuffled[n_domain_train:].tolist())

    # Trim to exact sizes (or fewer if not enough tasks)
    np.random.shuffle(train_ids)
    np.random.shuffle(test_ids)
    train_ids = train_ids[:train_size]
    test_ids = test_ids[:test_size]

    split = {
        "train": train_ids,
        "test": test_ids,
        "train_size": len(train_ids),
        "test_size": len(test_ids),
    }

    # Save split
    split_path = "datasets/train_test_split.json"
    with open(split_path, "w") as f:
        json.dump(split, f, indent=2)

    print(f"Train: {len(train_ids)} tasks")
    print(f"Test: {len(test_ids)} tasks")

    # Show domain distribution in each split
    train_meta = metadata_df[metadata_df["id"].isin(train_ids)]
    test_meta = metadata_df[metadata_df["id"].isin(test_ids)]

    print(f"\nTrain set domain distribution:")
    print(train_meta["domain"].value_counts().sort_index())
    print(f"\nTest set domain distribution:")
    print(test_meta["domain"].value_counts().sort_index())

    print(f"\n✓ Saved split to {split_path}")

    return split


def plot_sample_tasks(tasks, metadata_df, n_samples=5, output_dir="datasets/plots"):
    """
    Plot a few sample tasks for quality verification
    """
    os.makedirs(output_dir, exist_ok=True)

    print("\n" + "=" * 70)
    print("PLOTTING SAMPLES")
    print("=" * 70)

    # Randomly select tasks
    n_samples = min(n_samples, len(tasks))
    sample_indices = np.random.choice(len(tasks), n_samples, replace=False)

    print(f"Plotting {n_samples} sample tasks...")

    for idx in sample_indices:
        task = tasks[idx]
        task_id = metadata_df.iloc[idx]["id"]
        domain = metadata_df.iloc[idx]["domain"]

        # Create figure
        fig, ax = plt.subplots(figsize=(14, 6))

        # Extract data
        if hasattr(task.past_time, "iloc"):
            history = task.past_time.iloc[:, -1].values
            future = task.future_time.iloc[:, -1].values
        else:
            history = task.past_time.values
            future = task.future_time.values

        # Create time indices
        hist_idx = np.arange(len(history))
        future_idx = np.arange(len(history), len(history) + len(future))

        # Plot
        ax.plot(hist_idx, history, "b-", linewidth=2, label="History", alpha=0.8)
        ax.plot(
            future_idx,
            future,
            "r-",
            linewidth=2,
            label="Future (Ground Truth)",
            alpha=0.8,
        )
        ax.axvline(
            x=len(history) - 0.5, color="k", linestyle="--", alpha=0.5, linewidth=1.5
        )

        # Add context as title
        context = extract_all_context(task)
        title = f"{task_id} | Domain: {domain}\n"
        title += context[:200] + "..." if len(context) > 200 else context
        ax.set_title(title, fontsize=9, wrap=True)
        ax.set_xlabel("Time Step", fontsize=10)
        ax.set_ylabel("Value", fontsize=10)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        # Save
        plot_path = os.path.join(output_dir, f"{task_id}_{domain}.png")
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
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
    split = create_train_test_split(metadata_df, train_size=7, test_size=3)

    # Step 4: Plot samples
    plot_sample_tasks(tasks, metadata_df, n_samples=5)

    print("\n" + "=" * 70)
    print("DAY 1 COMPLETE!")
    print("=" * 70)
    print(
        f"\n✓ Generated {len(tasks)} tasks from {len(metadata_df['domain'].unique())} domains"
    )
    print(f"✓ Created {split['train_size']} train / {split['test_size']} test split")
    print(f"✓ Saved to datasets/ folder")
    print(f"✓ Sample plots in datasets/plots/")
    print("\n📊 Domain distribution:")
    print(metadata_df["domain"].value_counts().sort_index())
    print("\nNext: Move to Day 3 (model experiments)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
