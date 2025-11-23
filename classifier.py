import os
import json
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns

# ============================================
# Scoring configuration
# ============================================
ALPHA = 0.4  # weight for NMAE score
BETA = 0.4   # weight for DA score
GAMMA = 0.2  # weight for cost score

# Relative cost: LLMs are 8x more expensive than ARIMA/ETS
MODEL_COSTS = {
    "arima": 1.0,
    "ets": 1.0,
    "llmp_no_context": 8.0,
    "llmp_with_context": 8.0,
    "gpt4o_mini_no_context": 8.0,
    "gpt4o_mini_with_context": 8.0,
}

# All 6 models in scoring
ALL_MODELS = [
    "arima", "ets", 
    "llmp_no_context", "llmp_with_context",
    "gpt4o_mini_no_context", "gpt4o_mini_with_context"
]

# Model display names for plots
MODEL_DISPLAY_NAMES = {
    "arima": "ARIMA",
    "ets": "ETS",
    "llmp_no_context": "Llama-NC",
    "llmp_with_context": "Llama-C",
    "gpt4o_mini_no_context": "GPT4o-NC",
    "gpt4o_mini_with_context": "GPT4o-C",
}

# Color scheme
MODEL_COLORS = {
    "arima": "#1f77b4",           # blue
    "ets": "#ff7f0e",             # orange
    "llmp_no_context": "#2ca02c",  # green (light)
    "llmp_with_context": "#006400", # green (dark)
    "gpt4o_mini_no_context": "#d62728",  # red (light)
    "gpt4o_mini_with_context": "#8b0000", # red (dark)
}


# ============================================
# I/O helpers
# ============================================
def load_results(csv_path: str, prefix: str) -> pd.DataFrame:
    """Load a results CSV and rename columns with prefix."""
    df = pd.read_csv(csv_path, usecols=["task_id", "nmae", "da"])
    df = df.rename(columns={
        "nmae": f"{prefix}_nmae",
        "da": f"{prefix}_da",
    })
    return df


def load_all_results(results_dir: str) -> pd.DataFrame:
    """Load all 6 CSVs and merge into one DataFrame."""
    files = {
        "arima": "arima_results.csv",
        "ets": "ets_results.csv",
        "llmp_no_context": "llmp_no_context_results.csv",
        "llmp_with_context": "llmp_with_context_results.csv",
        "gpt4o_mini_no_context": "gpt4o_mini_no_context_results.csv",
        "gpt4o_mini_with_context": "gpt4o_mini_with_context_results.csv",
    }
    
    dfs = []
    for model_name, filename in files.items():
        path = os.path.join(results_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing: {path}")
        dfs.append(load_results(path, model_name))
    
    # Merge all on task_id
    merged = dfs[0]
    for df in dfs[1:]:
        merged = merged.merge(df, on="task_id", how="inner")
    
    return merged


# ============================================
# Scoring logic (all 6 models)
# ============================================
def compute_scores(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute weighted scores for all 6 models.
    
    score = ALPHA * nmae_score + BETA * da_score + GAMMA * cost_score
    
    Adds columns:
        - {model}_score for each model
        - selected_model (highest score)
        - context analysis columns
    """
    df = df.copy()
    
    for idx, row in df.iterrows():
        # Collect raw values
        nmae_vals = {m: row[f"{m}_nmae"] for m in ALL_MODELS}
        da_vals = {m: row[f"{m}_da"] for m in ALL_MODELS}
        
        # Normalize NMAE (lower is better -> higher score)
        nmae_min, nmae_max = min(nmae_vals.values()), max(nmae_vals.values())
        nmae_range = nmae_max - nmae_min if nmae_max > nmae_min else 1e-9
        
        # Normalize DA (higher is better -> higher score)
        da_min, da_max = min(da_vals.values()), max(da_vals.values())
        da_range = da_max - da_min if da_max > da_min else 1e-9
        
        # Normalize cost (lower is better -> higher score)
        cost_min = min(MODEL_COSTS.values())
        cost_max = max(MODEL_COSTS.values())
        cost_range = cost_max - cost_min if cost_max > cost_min else 1e-9
        
        scores = {}
        for m in ALL_MODELS:
            nmae_score = 1.0 - (nmae_vals[m] - nmae_min) / nmae_range
            da_score = (da_vals[m] - da_min) / da_range
            cost_score = 1.0 - (MODEL_COSTS[m] - cost_min) / cost_range
            
            scores[m] = ALPHA * nmae_score + BETA * da_score + GAMMA * cost_score
            df.at[idx, f"{m}_score"] = scores[m]
        
        # Best model
        df.at[idx, "selected_model"] = max(scores, key=scores.get)
    
    return df


def add_context_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add columns analyzing whether context helped.
    
    Adds:
        - llmp_context_gain_nmae: positive = context improved NMAE
        - gpt4o_context_gain_nmae: positive = context improved NMAE
        - llmp_context_gain_da: positive = context improved DA
        - gpt4o_context_gain_da: positive = context improved DA
        - llmp_context_helped: binary (did context improve NMAE?)
        - gpt4o_context_helped: binary
        - any_context_helped: binary (did either LLM benefit from context?)
    """
    df = df.copy()
    
    # NMAE gain: no_context - with_context (positive = context is better)
    df["llmp_context_gain_nmae"] = df["llmp_no_context_nmae"] - df["llmp_with_context_nmae"]
    df["gpt4o_context_gain_nmae"] = df["gpt4o_mini_no_context_nmae"] - df["gpt4o_mini_with_context_nmae"]
    
    # DA gain: with_context - no_context (positive = context is better)
    df["llmp_context_gain_da"] = df["llmp_with_context_da"] - df["llmp_no_context_da"]
    df["gpt4o_context_gain_da"] = df["gpt4o_mini_with_context_da"] - df["gpt4o_mini_no_context_da"]
    
    # Binary: did context help? (based on NMAE improvement)
    df["llmp_context_helped"] = (df["llmp_context_gain_nmae"] > 0).astype(int)
    df["gpt4o_context_helped"] = (df["gpt4o_context_gain_nmae"] > 0).astype(int)
    df["any_context_helped"] = ((df["llmp_context_helped"] == 1) | (df["gpt4o_context_helped"] == 1)).astype(int)
    
    # Was context worth it vs ARIMA? (context model beat ARIMA)
    df["llmp_context_beat_arima"] = (df["llmp_with_context_nmae"] < df["arima_nmae"]).astype(int)
    df["gpt4o_context_beat_arima"] = (df["gpt4o_mini_with_context_nmae"] < df["arima_nmae"]).astype(int)
    
    return df


# ============================================
# Domain-level analysis
# ============================================
def analyze_domain(df: pd.DataFrame, domain_name: str) -> dict:
    """Compute summary statistics for a domain."""
    n_tasks = len(df)
    
    # Model selection counts
    selection_counts = df["selected_model"].value_counts().to_dict()
    selection_pcts = {k: 100 * v / n_tasks for k, v in selection_counts.items()}
    
    # Fill in zeros for models that were never selected
    for m in ALL_MODELS:
        if m not in selection_pcts:
            selection_pcts[m] = 0.0
            selection_counts[m] = 0
    
    # Average metrics per model
    avg_nmae = {m: df[f"{m}_nmae"].mean() for m in ALL_MODELS}
    avg_da = {m: df[f"{m}_da"].mean() for m in ALL_MODELS}
    
    # Context analysis
    llmp_context_helped_pct = 100 * df["llmp_context_helped"].mean()
    gpt4o_context_helped_pct = 100 * df["gpt4o_context_helped"].mean()
    any_context_helped_pct = 100 * df["any_context_helped"].mean()
    
    llmp_beat_arima_pct = 100 * df["llmp_context_beat_arima"].mean()
    gpt4o_beat_arima_pct = 100 * df["gpt4o_context_beat_arima"].mean()
    
    return {
        "domain": domain_name,
        "n_tasks": n_tasks,
        "selection_counts": selection_counts,
        "selection_pcts": selection_pcts,
        "avg_nmae": avg_nmae,
        "avg_da": avg_da,
        "llmp_context_helped_pct": llmp_context_helped_pct,
        "gpt4o_context_helped_pct": gpt4o_context_helped_pct,
        "any_context_helped_pct": any_context_helped_pct,
        "llmp_avg_context_gain": df["llmp_context_gain_nmae"].mean(),
        "gpt4o_avg_context_gain": df["gpt4o_context_gain_nmae"].mean(),
        "llmp_beat_arima_pct": llmp_beat_arima_pct,
        "gpt4o_beat_arima_pct": gpt4o_beat_arima_pct,
    }


# ============================================
# Plotting functions
# ============================================
def plot_model_selection(df: pd.DataFrame, domain_name: str, output_dir: str):
    """Pie chart of which model was selected most often."""
    counts = df["selected_model"].value_counts()
    
    # Use display names and colors
    labels = [MODEL_DISPLAY_NAMES.get(m, m) for m in counts.index]
    colors = [MODEL_COLORS.get(m, "gray") for m in counts.index]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.pie(counts.values, labels=labels, autopct='%1.1f%%', colors=colors, startangle=90)
    ax.set_title(f"{domain_name}: Best Model Selection\n(α={ALPHA}, β={BETA}, γ={GAMMA})")
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_selection_pie.png"), dpi=150)
    plt.close()


def plot_avg_nmae_comparison(df: pd.DataFrame, domain_name: str, output_dir: str):
    """Bar chart comparing average NMAE across all 6 models."""
    avg_nmae = [df[f"{m}_nmae"].mean() for m in ALL_MODELS]
    labels = [MODEL_DISPLAY_NAMES.get(m, m) for m in ALL_MODELS]
    colors = [MODEL_COLORS.get(m, "gray") for m in ALL_MODELS]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(range(len(ALL_MODELS)), avg_nmae, color=colors)
    ax.set_xticks(range(len(ALL_MODELS)))
    ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_ylabel("Average NMAE (lower is better)")
    ax.set_title(f"{domain_name}: Average NMAE by Model")
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bar, val in zip(bars, avg_nmae):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "avg_nmae_comparison.png"), dpi=150)
    plt.close()


def plot_context_gain_histogram(df: pd.DataFrame, domain_name: str, output_dir: str):
    """Histograms showing context gain distribution for both LLMs."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Llama
    ax = axes[0]
    ax.hist(df["llmp_context_gain_nmae"], bins=15, color=MODEL_COLORS["llmp_with_context"], 
            alpha=0.7, edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='No gain')
    ax.axvline(df["llmp_context_gain_nmae"].mean(), color='blue', linestyle='-', 
               linewidth=2, label=f'Mean: {df["llmp_context_gain_nmae"].mean():.3f}')
    ax.set_xlabel("NMAE Gain from Context\n(positive = context better)")
    ax.set_ylabel("Number of Tasks")
    ax.set_title(f"Llama: Context Gain\n({100*df['llmp_context_helped'].mean():.1f}% helped)")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # GPT-4o
    ax = axes[1]
    ax.hist(df["gpt4o_context_gain_nmae"], bins=15, color=MODEL_COLORS["gpt4o_mini_with_context"], 
            alpha=0.7, edgecolor='black')
    ax.axvline(0, color='red', linestyle='--', linewidth=2, label='No gain')
    ax.axvline(df["gpt4o_context_gain_nmae"].mean(), color='blue', linestyle='-', 
               linewidth=2, label=f'Mean: {df["gpt4o_context_gain_nmae"].mean():.3f}')
    ax.set_xlabel("NMAE Gain from Context\n(positive = context better)")
    ax.set_ylabel("Number of Tasks")
    ax.set_title(f"GPT-4o-mini: Context Gain\n({100*df['gpt4o_context_helped'].mean():.1f}% helped)")
    ax.legend()
    ax.grid(alpha=0.3)
    
    fig.suptitle(f"{domain_name}: Did Context Help?", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "context_gain_histogram.png"), dpi=150)
    plt.close()


def plot_nmae_da_scatter(df: pd.DataFrame, domain_name: str, output_dir: str):
    """Scatter plot of NMAE vs DA for all models."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for m in ALL_MODELS:
        ax.scatter(df[f"{m}_nmae"], df[f"{m}_da"], 
                   label=MODEL_DISPLAY_NAMES.get(m, m),
                   color=MODEL_COLORS.get(m, "gray"),
                   alpha=0.6, s=50)
    
    ax.set_xlabel("NMAE (lower is better)")
    ax.set_ylabel("Directional Accuracy (higher is better)")
    ax.set_title(f"{domain_name}: NMAE vs DA (All Models)")
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    
    # Limit x-axis to reasonable range (exclude extreme outliers for visibility)
    nmae_95 = df[[f"{m}_nmae" for m in ALL_MODELS]].values.flatten()
    nmae_95 = np.percentile(nmae_95, 95)
    ax.set_xlim(0, min(nmae_95 * 1.2, ax.get_xlim()[1]))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "nmae_da_scatter.png"), dpi=150)
    plt.close()


def plot_model_centroids(df: pd.DataFrame, domain_name: str, output_dir: str):
    """Plot showing mean NMAE vs mean DA per model."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    for m in ALL_MODELS:
        mean_nmae = df[f"{m}_nmae"].mean()
        mean_da = df[f"{m}_da"].mean()
        
        ax.scatter(mean_nmae, mean_da, s=200, 
                   color=MODEL_COLORS.get(m, "gray"),
                   edgecolor='black', linewidth=1.5, zorder=3)
        ax.annotate(MODEL_DISPLAY_NAMES.get(m, m), 
                    (mean_nmae, mean_da),
                    xytext=(8, 8), textcoords='offset points',
                    fontsize=10, fontweight='bold')
    
    ax.set_xlabel("Mean NMAE (lower is better)")
    ax.set_ylabel("Mean DA (higher is better)")
    ax.set_title(f"{domain_name}: Model Performance Summary")
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "model_centroids.png"), dpi=150)
    plt.close()


# ============================================
# Cross-domain analysis
# ============================================
def create_cross_domain_summary(all_stats: list) -> pd.DataFrame:
    """Create summary table across all domains."""
    rows = []
    for stats in all_stats:
        row = {
            "Domain": stats["domain"],
            "Tasks": stats["n_tasks"],
            "ARIMA %": stats["selection_pcts"].get("arima", 0),
            "ETS %": stats["selection_pcts"].get("ets", 0),
            "Llama-NC %": stats["selection_pcts"].get("llmp_no_context", 0),
            "Llama-C %": stats["selection_pcts"].get("llmp_with_context", 0),
            "GPT-NC %": stats["selection_pcts"].get("gpt4o_mini_no_context", 0),
            "GPT-C %": stats["selection_pcts"].get("gpt4o_mini_with_context", 0),
            "Llama Context Helped %": stats["llmp_context_helped_pct"],
            "GPT Context Helped %": stats["gpt4o_context_helped_pct"],
            "Any Context Helped %": stats["any_context_helped_pct"],
            "Llama Beat ARIMA %": stats["llmp_beat_arima_pct"],
            "GPT Beat ARIMA %": stats["gpt4o_beat_arima_pct"],
        }
        rows.append(row)
    
    return pd.DataFrame(rows)


def plot_cross_domain_heatmap(summary_df: pd.DataFrame, output_path: str):
    """Heatmap showing which models won in which domains."""
    # Model selection percentages
    model_cols = ["ARIMA %", "ETS %", "Llama-NC %", "Llama-C %", "GPT-NC %", "GPT-C %"]
    heatmap_data = summary_df[["Domain"] + model_cols].set_index("Domain")
    
    fig, ax = plt.subplots(figsize=(12, max(6, len(summary_df) * 0.5)))
    sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='YlOrRd', 
                cbar_kws={'label': '% tasks where model was best'}, ax=ax)
    ax.set_title(f"Model Selection by Domain\n(α={ALPHA}, β={BETA}, γ={GAMMA})", 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel("Model")
    ax.set_ylabel("Domain")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_context_helpfulness_heatmap(summary_df: pd.DataFrame, output_path: str):
    """Heatmap showing how often context helped per domain."""
    context_cols = ["Llama Context Helped %", "GPT Context Helped %", 
                    "Llama Beat ARIMA %", "GPT Beat ARIMA %"]
    heatmap_data = summary_df[["Domain"] + context_cols].set_index("Domain")
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(summary_df) * 0.5)))
    sns.heatmap(heatmap_data, annot=True, fmt='.1f', cmap='RdYlGn', 
                center=50, cbar_kws={'label': '% of tasks'}, ax=ax)
    ax.set_title("Context Effectiveness by Domain", fontsize=14, fontweight='bold')
    ax.set_xlabel("Metric")
    ax.set_ylabel("Domain")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================
# Main processing pipeline
# ============================================
def process_domain(results_dir: str, domain_name: str) -> dict:
    """Process a single domain: load, score, analyze, plot."""
    print(f"\n{'='*60}")
    print(f"Processing: {domain_name}")
    print('='*60)
    
    # Load all results
    df = load_all_results(results_dir)
    print(f"  Loaded {len(df)} tasks")
    
    # Compute scores and selection
    df = compute_scores(df)
    
    # Add context analysis
    df = add_context_analysis(df)
    
    # Save comparison CSV
    output_path = os.path.join(results_dir, "comparison.csv")
    df.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}")
    
    # Analyze
    stats = analyze_domain(df, domain_name)
    
    # Print summary
    print(f"\n  Model Selection:")
    for m in ALL_MODELS:
        pct = stats["selection_pcts"].get(m, 0)
        if pct > 0:
            print(f"    {MODEL_DISPLAY_NAMES.get(m, m):12s}: {pct:5.1f}%")
    
    print(f"\n  Context Analysis:")
    print(f"    Llama: context helped {stats['llmp_context_helped_pct']:.1f}% of tasks (avg gain: {stats['llmp_avg_context_gain']:.3f})")
    print(f"    GPT-4o: context helped {stats['gpt4o_context_helped_pct']:.1f}% of tasks (avg gain: {stats['gpt4o_avg_context_gain']:.3f})")
    print(f"    Llama beat ARIMA: {stats['llmp_beat_arima_pct']:.1f}%")
    print(f"    GPT-4o beat ARIMA: {stats['gpt4o_beat_arima_pct']:.1f}%")
    
    # Generate plots
    plot_model_selection(df, domain_name, results_dir)
    plot_avg_nmae_comparison(df, domain_name, results_dir)
    plot_context_gain_histogram(df, domain_name, results_dir)
    plot_nmae_da_scatter(df, domain_name, results_dir)
    plot_model_centroids(df, domain_name, results_dir)
    print(f"  Generated 5 plots in: {results_dir}")
    
    return stats


def process_all_domains(results_root: str):
    """Process all domain subfolders and create cross-domain analysis."""
    all_stats = []
    
    # Process each domain
    for name in sorted(os.listdir(results_root)):
        subdir = os.path.join(results_root, name)
        if not os.path.isdir(subdir):
            continue
        
        try:
            stats = process_domain(subdir, name)
            all_stats.append(stats)
        except FileNotFoundError as e:
            print(f"\n  [WARN] Skipping {name}: {e}")
        except Exception as e:
            print(f"\n  [ERROR] Problem in {name}: {e}")
            import traceback
            traceback.print_exc()
    
    if not all_stats:
        print("\nNo domains processed!")
        return
    
    # Cross-domain summary
    print(f"\n{'='*60}")
    print("CROSS-DOMAIN SUMMARY")
    print('='*60)
    
    summary_df = create_cross_domain_summary(all_stats)
    
    # Save summary
    summary_path = os.path.join(results_root, "cross_domain_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"\nSaved: {summary_path}")
    
    # Print summary
    print("\n" + summary_df.to_string(index=False))
    
    # Aggregate stats
    total_tasks = summary_df["Tasks"].sum()
    avg_context_helped = summary_df["Any Context Helped %"].mean()
    avg_llmp_beat_arima = summary_df["Llama Beat ARIMA %"].mean()
    avg_gpt_beat_arima = summary_df["GPT Beat ARIMA %"].mean()
    
    print(f"\n  AGGREGATE STATISTICS:")
    print(f"    Total tasks across all domains: {total_tasks}")
    print(f"    Avg % where context helped (either LLM): {avg_context_helped:.1f}%")
    print(f"    Avg % where Llama-C beat ARIMA: {avg_llmp_beat_arima:.1f}%")
    print(f"    Avg % where GPT-4o-C beat ARIMA: {avg_gpt_beat_arima:.1f}%")
    
    # Cross-domain plots
    heatmap_path = os.path.join(results_root, "cross_domain_model_selection.png")
    plot_cross_domain_heatmap(summary_df, heatmap_path)
    print(f"\nSaved: {heatmap_path}")
    
    context_heatmap_path = os.path.join(results_root, "cross_domain_context_effectiveness.png")
    plot_context_helpfulness_heatmap(summary_df, context_heatmap_path)
    print(f"Saved: {context_heatmap_path}")


# ============================================
# Entry point
# ============================================
if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_root = os.path.join(base_dir, "results")
    
    print(f"Classifier Configuration:")
    print(f"  ALPHA (NMAE weight): {ALPHA}")
    print(f"  BETA (DA weight): {BETA}")
    print(f"  GAMMA (Cost weight): {GAMMA}")
    print(f"  Model costs: {MODEL_COSTS}")
    
    process_all_domains(results_root)