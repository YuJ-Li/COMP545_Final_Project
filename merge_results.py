"""
merge_results.py

Merges all model result CSVs into a single comparison.csv for each domain.
Computes labels comparing best LLM+context vs best baseline for NMAE and DA.

Usage:
    python merge_results.py

Input (per domain folder):
    - arima_results.csv
    - ets_results.csv
    - llmp_no_context_results.csv
    - llmp_with_context_results.csv
    - gpt4o_mini_no_context_results.csv
    - gpt4o_mini_with_context_results.csv

Output (per domain folder):
    - comparison.csv
"""

import os
import pandas as pd
import numpy as np

# Model CSV files to merge
MODEL_FILES = {
    "arima": "arima_results.csv",
    "ets": "ets_results.csv",
    "llmp_no_context": "llmp_no_context_results.csv",
    "llmp_with_context": "llmp_with_context_results.csv",
    "gpt4o_no_context": "gpt4o_mini_no_context_results.csv",
    "gpt4o_with_context": "gpt4o_mini_with_context_results.csv",
}

# Columns to keep from each model CSV
KEEP_COLS = ["nmae", "da"]


def merge_domain(domain_dir: str) -> pd.DataFrame:
    """
    Merge all model CSVs for a single domain.
    """
    
    merged = None
    
    for model_name, filename in MODEL_FILES.items():
        filepath = os.path.join(domain_dir, filename)
        
        if not os.path.exists(filepath):
            print(f"    [WARN] Missing: {filename}")
            continue
        
        # Load CSV
        df = pd.read_csv(filepath)
        
        # Keep only task_id and metrics
        cols_to_keep = ["task_id"] + [c for c in KEEP_COLS if c in df.columns]
        df = df[cols_to_keep].copy()
        
        # Rename columns with model prefix
        rename_map = {col: f"{model_name}_{col}" for col in KEEP_COLS if col in df.columns}
        df = df.rename(columns=rename_map)
        
        # Merge using inner join to only keep complete tasks
        if merged is None:
            merged = df
        else:
            merged = merged.merge(df, on="task_id", how="inner")
    
    if merged is None:
        raise ValueError(f"No model CSVs found in {domain_dir}")
    
    return merged


def compute_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute all comparison labels for XGBoost training.
    """

    # NMAE: lower is better, so take min
    baseline_nmae_cols = ["arima_nmae", "ets_nmae"]
    available_baseline_nmae = [c for c in baseline_nmae_cols if c in df.columns]
    if available_baseline_nmae:
        df["best_baseline_nmae"] = df[available_baseline_nmae].min(axis=1)
        df["best_baseline_nmae_model"] = df[available_baseline_nmae].idxmin(axis=1).str.replace("_nmae", "")
    
    # DA: higher is better, so take max
    baseline_da_cols = ["arima_da", "ets_da"]
    available_baseline_da = [c for c in baseline_da_cols if c in df.columns]
    if available_baseline_da:
        df["best_baseline_da"] = df[available_baseline_da].max(axis=1)
        df["best_baseline_da_model"] = df[available_baseline_da].idxmax(axis=1).str.replace("_da", "")

    
    # NMAE: lower is better, so take min
    llm_context_nmae_cols = ["llmp_with_context_nmae", "gpt4o_with_context_nmae"]
    available_llm_nmae = [c for c in llm_context_nmae_cols if c in df.columns]
    if available_llm_nmae:
        df["best_llm_context_nmae"] = df[available_llm_nmae].min(axis=1)
        df["best_llm_context_nmae_model"] = df[available_llm_nmae].idxmin(axis=1).str.replace("_nmae", "")
    
    # DA: higher is better, so take max
    llm_context_da_cols = ["llmp_with_context_da", "gpt4o_with_context_da"]
    available_llm_da = [c for c in llm_context_da_cols if c in df.columns]
    if available_llm_da:
        df["best_llm_context_da"] = df[available_llm_da].max(axis=1)
        df["best_llm_context_da_model"] = df[available_llm_da].idxmax(axis=1).str.replace("_da", "")

    
    llm_no_context_nmae_cols = ["llmp_no_context_nmae", "gpt4o_no_context_nmae"]
    available_llm_nc_nmae = [c for c in llm_no_context_nmae_cols if c in df.columns]
    if available_llm_nc_nmae:
        df["best_llm_no_context_nmae"] = df[available_llm_nc_nmae].min(axis=1)
    
    llm_no_context_da_cols = ["llmp_no_context_da", "gpt4o_no_context_da"]
    available_llm_nc_da = [c for c in llm_no_context_da_cols if c in df.columns]
    if available_llm_nc_da:
        df["best_llm_no_context_da"] = df[available_llm_nc_da].max(axis=1)

    
    # NMAE: LLM wins if lower (better)
    if "best_llm_context_nmae" in df.columns and "best_baseline_nmae" in df.columns:
        df["llm_wins_nmae"] = (df["best_llm_context_nmae"] < df["best_baseline_nmae"]).astype(int)
    
    # DA: LLM wins if higher (better)
    if "best_llm_context_da" in df.columns and "best_baseline_da" in df.columns:
        df["llm_wins_da"] = (df["best_llm_context_da"] > df["best_baseline_da"]).astype(int)
    
    # Both: LLM wins on both metrics
    if "llm_wins_nmae" in df.columns and "llm_wins_da" in df.columns:
        df["llm_wins_both"] = ((df["llm_wins_nmae"] == 1) & (df["llm_wins_da"] == 1)).astype(int)
    
    # Either: LLM wins on at least one metric
    if "llm_wins_nmae" in df.columns and "llm_wins_da" in df.columns:
        df["llm_wins_either"] = ((df["llm_wins_nmae"] == 1) | (df["llm_wins_da"] == 1)).astype(int)

    
    # Did context help LLM? (comparing with vs without context)
    if "best_llm_context_nmae" in df.columns and "best_llm_no_context_nmae" in df.columns:
        df["context_helped_nmae"] = (df["best_llm_context_nmae"] < df["best_llm_no_context_nmae"]).astype(int)
    
    if "best_llm_context_da" in df.columns and "best_llm_no_context_da" in df.columns:
        df["context_helped_da"] = (df["best_llm_context_da"] > df["best_llm_no_context_da"]).astype(int)

    
    # Llama vs best baseline
    if "llmp_with_context_nmae" in df.columns and "best_baseline_nmae" in df.columns:
        df["llmp_wins_nmae"] = (df["llmp_with_context_nmae"] < df["best_baseline_nmae"]).astype(int)
    
    if "llmp_with_context_da" in df.columns and "best_baseline_da" in df.columns:
        df["llmp_wins_da"] = (df["llmp_with_context_da"] > df["best_baseline_da"]).astype(int)
    
    # GPT-4o vs best baseline
    if "gpt4o_with_context_nmae" in df.columns and "best_baseline_nmae" in df.columns:
        df["gpt4o_wins_nmae"] = (df["gpt4o_with_context_nmae"] < df["best_baseline_nmae"]).astype(int)
    
    if "gpt4o_with_context_da" in df.columns and "best_baseline_da" in df.columns:
        df["gpt4o_wins_da"] = (df["gpt4o_with_context_da"] > df["best_baseline_da"]).astype(int)

    
    all_nmae_cols = [c for c in df.columns if c.endswith("_nmae") and not c.startswith("best_")]
    if all_nmae_cols:
        df["best_model_nmae"] = df[all_nmae_cols].idxmin(axis=1).str.replace("_nmae", "")
    
    all_da_cols = [c for c in df.columns if c.endswith("_da") and not c.startswith("best_")]
    if all_da_cols:
        df["best_model_da"] = df[all_da_cols].idxmax(axis=1).str.replace("_da", "")
    
    return df

def process_domain(domain_dir: str, domain_name: str) -> bool:
    """
    Process a single domain folder.
    
    Returns:
        True if successful, False otherwise
    """
    try:
        # Merge all model CSVs
        df = merge_domain(domain_dir)
        
        # Compute labels
        df = compute_labels(df)
        
        # Save
        output_path = os.path.join(domain_dir, "comparison.csv")
        df.to_csv(output_path, index=False)
        
        # Summary stats
        n_tasks = len(df)
        llm_wins = df["llm_wins_nmae"].sum() if "llm_wins_nmae" in df.columns else 0
        
        print(f"  [OK] {domain_name}: {n_tasks} tasks, LLM wins {llm_wins}/{n_tasks} on NMAE")
        return True
        
    except Exception as e:
        print(f"  [ERROR] {domain_name}: {e}")
        return False

def main(results_root: str):
    """Process all domain folders."""
    
    print("="*60)
    print("Merge Results (Updated)")
    print("="*60)
    print(f"Results directory: {results_root}")
    print()
    print("Comparing: Best LLM+Context vs Best Baseline")
    print("  - Baselines: ARIMA, ETS")
    print("  - LLMs: Llama-Context, GPT4o-Context")
    print("  - Metrics: NMAE (lower=better), DA (higher=better)")
    print()
    
    success_count = 0
    fail_count = 0
    
    for name in sorted(os.listdir(results_root)):
        domain_dir = os.path.join(results_root, name)
        
        # Skip non-directories and output folders
        if not os.path.isdir(domain_dir):
            continue
        if name in ["xgboost_compiled", "__pycache__"]:
            continue
        
        # Check if it has model CSVs
        has_models = any(
            os.path.exists(os.path.join(domain_dir, f))
            for f in MODEL_FILES.values()
        )
        
        if not has_models:
            print(f"  [SKIP] {name}: No model CSVs found")
            continue
        
        if process_domain(domain_dir, name):
            success_count += 1
        else:
            fail_count += 1
    
    print()
    print("="*60)
    print(f"Done! {success_count} domains processed, {fail_count} failed")
    print("="*60)
    print()
    print("New columns in comparison.csv:")
    print("  - best_baseline_nmae/da: min/max of ARIMA, ETS")
    print("  - best_llm_context_nmae/da: min/max of Llama-C, GPT4o-C")
    print("  - llm_wins_nmae: LLM beat baseline on NMAE?")
    print("  - llm_wins_da: LLM beat baseline on DA?")
    print("  - llm_wins_both: LLM won on both metrics?")
    print("  - context_helped_nmae/da: Did context improve LLM?")

if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_root = os.path.join(base_dir, "results")
    
    main(results_root)