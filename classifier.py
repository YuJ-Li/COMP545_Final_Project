import os
import pandas as pd
from matplotlib import pyplot as plt

# ============================================
# Scoring configuration
# ============================================
ALPHA = 0.4  # weight for NMAE score
BETA = 0.4   # weight for DA score
GAMMA = 0.2  # weight for cost score

# Relative cost: Both LLMs with context are 8x more expensive than ARIMA/ETS
MODEL_COST = {
    "arima": 1.0,
    "ets": 1.0,
    "llmp_with_context": 8.0,
    "gpt4o_mini_with_context": 8.0,  # Same cost as Llama
}

# Models considered in the scoring / selection
MODELS_FOR_SCORING = ["arima", "ets", "llmp_with_context", "gpt4o_mini_with_context"]


# ============================================
# I/O helpers
# ============================================
def load_results(csv_path: str, prefix: str) -> pd.DataFrame:
    """
    Load a results CSV and keep only task_id, nmae, da.
    Rename nmae -> {prefix}_nmae and da -> {prefix}_da.
    """
    df = pd.read_csv(csv_path, usecols=["task_id", "nmae", "da"])
    df = df.rename(
        columns={
            "nmae": f"{prefix}_nmae",
            "da": f"{prefix}_da",
        }
    )
    return df


# ============================================
# Scoring logic
# ============================================
def _score_single_row(row: pd.Series) -> pd.Series:
    """
    For one row (task), compute scores for:
        - arima
        - ets
        - llmp_with_context
        - gpt4o_mini_with_context

    score = ALPHA * nmae_score + BETA * da_score + GAMMA * cost_score

    where:
      - nmae_score: normalized so LOWER NMAE => HIGHER score
      - da_score:   normalized so HIGHER DA  => HIGHER score
      - cost_score: normalized so CHEAPER   => HIGHER score

    Adds:
        arima_score,
        ets_score,
        llmp_with_context_score,
        gpt4o_mini_with_context_score,
        selected_model
    """
    # Collect raw NMAE and DA
    nmae_vals = {m: row[f"{m}_nmae"] for m in MODELS_FOR_SCORING}
    da_vals = {m: row[f"{m}_da"] for m in MODELS_FOR_SCORING}

    # --- normalize NMAE (lower is better) ---
    nmae_min = min(nmae_vals.values())
    nmae_max = max(nmae_vals.values())
    nmae_range = nmae_max - nmae_min
    if nmae_range == 0:
        nmae_range = 1e-9  # avoid division by zero

    # --- normalize DA (higher is better) ---
    da_min = min(da_vals.values())
    da_max = max(da_vals.values())
    da_range = da_max - da_min
    if da_range == 0:
        da_range = 1e-9

    # --- normalize cost (cheaper is better) ---
    cost_min = min(MODEL_COST[m] for m in MODELS_FOR_SCORING)
    cost_max = max(MODEL_COST[m] for m in MODELS_FOR_SCORING)
    cost_range = cost_max - cost_min
    if cost_range == 0:
        cost_range = 1e-9

    scores = {}
    for m in MODELS_FOR_SCORING:
        # NMAE: smaller => better
        nmae_norm = (nmae_vals[m] - nmae_min) / nmae_range   # 0 = best, 1 = worst
        nmae_score = 1.0 - nmae_norm                         # 1 = best, 0 = worst

        # DA: larger => better
        da_norm = (da_vals[m] - da_min) / da_range           # 0 = worst, 1 = best
        da_score = da_norm

        # Cost: smaller => better
        cost_norm = (MODEL_COST[m] - cost_min) / cost_range  # 0 = cheapest, 1 = most expensive
        cost_score = 1.0 - cost_norm                         # 1 = cheapest, 0 = most expensive

        score = ALPHA * nmae_score + BETA * da_score + GAMMA * cost_score
        scores[m] = score

    # Add scores to row
    row["arima_score"] = scores["arima"]
    row["ets_score"] = scores["ets"]
    row["llmp_with_context_score"] = scores["llmp_with_context"]
    row["gpt4o_mini_with_context_score"] = scores["gpt4o_mini_with_context"]

    # Best model = highest score
    row["selected_model"] = max(scores, key=scores.get)

    return row


def add_scores_and_selection(df: pd.DataFrame) -> pd.DataFrame:
    """
    Given a wide-format DataFrame with columns:
        task_id,
        arima_nmae, arima_da,
        ets_nmae, ets_da,
        llmp_no_context_nmae, llmp_no_context_da,
        llmp_with_context_nmae, llmp_with_context_da,
        gpt4o_mini_no_context_nmae, gpt4o_mini_no_context_da,
        gpt4o_mini_with_context_nmae, gpt4o_mini_with_context_da,

    compute scores and add:
        arima_score,
        ets_score,
        llmp_with_context_score,
        gpt4o_mini_with_context_score,
        selected_model
    """
    return df.apply(_score_single_row, axis=1)


# ============================================
# CSV combination per subfolder
# ============================================
def combine_csv_for_dir(results_dir: str) -> str:
    """
    For a given results subfolder, load the 6 CSVs, merge them,
    add scoring columns + selected_model, and write comparison.csv.

    Returns the path to comparison.csv.
    """
    arima_path = os.path.join(results_dir, "arima_results.csv")
    ets_path = os.path.join(results_dir, "ets_results.csv")
    llmp_nc_path = os.path.join(results_dir, "llmp_no_context_results.csv")
    llmp_wc_path = os.path.join(results_dir, "llmp_with_context_results.csv")
    gpt_nc_path = os.path.join(results_dir, "gpt4o_mini_no_context_results.csv")
    gpt_wc_path = os.path.join(results_dir, "gpt4o_mini_with_context_results.csv")

    # Load and rename columns
    arima_df = load_results(arima_path, "arima")
    ets_df = load_results(ets_path, "ets")
    llmp_nc_df = load_results(llmp_nc_path, "llmp_no_context")
    llmp_wc_df = load_results(llmp_wc_path, "llmp_with_context")
    gpt_nc_df = load_results(gpt_nc_path, "gpt4o_mini_no_context")
    gpt_wc_df = load_results(gpt_wc_path, "gpt4o_mini_with_context")

    # Merge on task_id
    merged = (
        arima_df
        .merge(ets_df, on="task_id", how="inner")
        .merge(llmp_nc_df, on="task_id", how="inner")
        .merge(llmp_wc_df, on="task_id", how="inner")
        .merge(gpt_nc_df, on="task_id", how="inner")
        .merge(gpt_wc_df, on="task_id", how="inner")
    )

    # Add score columns + selected_model
    merged = add_scores_and_selection(merged)

    output_path = os.path.join(results_dir, "comparison.csv")
    merged.to_csv(output_path, index=False)

    print(f"[combine_csv_for_dir] Saved combined results to: {output_path}")
    return output_path


# ============================================
# Plotting (no-context models excluded from plots)
# ============================================
def make_long_format(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert wide-format comparison.csv into:
    columns: task_id, model, nmae, da

    Include arima, ets, llmp_with_context, gpt4o_mini_with_context in plots.
    No-context variants stay only in the CSV.
    """
    models_for_plot = ["arima", "ets", "llmp_with_context", "gpt4o_mini_with_context"]
    records = []

    for _, row in df.iterrows():
        for m in models_for_plot:
            records.append({
                "task_id": row["task_id"],
                "model": m,
                "nmae": row[f"{m}_nmae"],
                "da": row[f"{m}_da"],
            })

    return pd.DataFrame(records)


def plot_all_for_dir(results_dir: str, comparison_path: str | None = None) -> None:
    """
    Produces, inside `results_dir`:
        nmae_da_scatter.png   (triple broken x-axis)
        model_centroids.png   (one point per model: mean NMAE vs mean DA)

    Plots ARIMA, ETS, Llama-with-context, GPT-4o-Mini-with-context.
    """
    if comparison_path is None:
        comparison_path = os.path.join(results_dir, "comparison.csv")

    df = pd.read_csv(comparison_path)
    long_df = make_long_format(df)

    # ============================
    # 1) Triple Broken Axis Plot
    # ============================

    zoom1_max = 25      # fine zoom
    zoom2_min = 25
    zoom2_max = 200     # medium zoom
    outlier_min = 220   # outliers start

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, sharey=True, figsize=(15, 5))

    # Color mapping for consistency
    colors = {
        'arima': 'C0',
        'ets': 'C1', 
        'llmp_with_context': 'C2',
        'gpt4o_mini_with_context': 'C3'
    }

    # --- Panel 1: 0–25 ---
    for model, group in long_df.groupby("model"):
        ax1.scatter(group["nmae"], group["da"], label=model, color=colors.get(model), alpha=0.7)
    ax1.set_xlim(0, zoom1_max)
    ax1.set_xlabel("NMAE")
    ax1.set_ylabel("Directional Accuracy (DA)")
    ax1.set_title("0–25 NMAE")

    # --- Panel 2: 25–200 ---
    for model, group in long_df.groupby("model"):
        mid = group[(group["nmae"] >= zoom2_min) & (group["nmae"] <= zoom2_max)]
        if not mid.empty:
            ax2.scatter(mid["nmae"], mid["da"], color=colors.get(model), alpha=0.7)
    ax2.set_xlim(zoom2_min, zoom2_max)
    ax2.set_xlabel("NMAE")
    ax2.set_title("25–200 NMAE")

    # --- Panel 3: >200 ---
    for model, group in long_df.groupby("model"):
        outliers = group[group["nmae"] > zoom2_max]
        if not outliers.empty:
            ax3.scatter(outliers["nmae"], outliers["da"], color=colors.get(model), alpha=0.7)
    ax3.set_xlim(outlier_min, long_df["nmae"].max() * 1.05)
    ax3.set_xlabel("NMAE")
    ax3.set_title(">200 NMAE (outliers)")

    # Hide panel borders between broken axes
    for axis in (ax1, ax2):
        axis.spines["right"].set_visible(False)
    for axis in (ax2, ax3):
        axis.spines["left"].set_visible(False)

    # Tick settings
    ax2.yaxis.set_ticks_position("none")
    ax3.yaxis.set_ticks_position("right")

    # Diagonal break marks
    d = 0.015
    kwargs = dict(color='k', clip_on=False)

    # between ax1 and ax2
    ax1.plot((1 - d, 1 + d), (-d, +d), transform=ax1.transAxes, **kwargs)
    ax1.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=ax1.transAxes, **kwargs)
    ax2.plot((-d, +d), (-d, +d), transform=ax2.transAxes, **kwargs)
    ax2.plot((-d, +d), (1 - d, 1 + d), transform=ax2.transAxes, **kwargs)

    # between ax2 and ax3
    ax2.plot((1 - d, 1 + d), (-d, +d), transform=ax2.transAxes, **kwargs)
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), transform=ax2.transAxes, **kwargs)
    ax3.plot((-d, +d), (-d, +d), transform=ax3.transAxes, **kwargs)
    ax3.plot((-d, +d), (1 - d, 1 + d), transform=ax3.transAxes, **kwargs)

    # Legend (4 models now)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4)

    fig.suptitle("NMAE vs DA per model (per task)", y=0.97)
    plt.tight_layout(rect=[0, 0.1, 1, 0.95])

    scatter_path = os.path.join(results_dir, "nmae_da_scatter.png")
    plt.savefig(scatter_path)
    plt.close()
    print(f"[plot_all_for_dir] Saved: {scatter_path}")

    # ============================
    # 2) Centroid plot (mean NMAE vs mean DA)
    # ============================

    # Ignore the most extreme 1% NMAE when computing "typical" behavior
    nmae_thr = long_df["nmae"].quantile(0.99)
    stats_df = long_df[long_df["nmae"] <= nmae_thr]

    centroids = (
        stats_df
        .groupby("model")[["nmae", "da"]]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(8, 6))
    for _, row in centroids.iterrows():
        plt.scatter(row["nmae"], row["da"], s=100, color=colors.get(row["model"]))
        plt.text(
            row["nmae"],
            row["da"],
            row["model"],
            fontsize=9,
            ha="left",
            va="bottom",
        )

    plt.xlabel("Mean NMAE (typical, filtered)")
    plt.ylabel("Mean DA (typical, filtered)")
    plt.title("Model centroids in NMAE–DA space")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    centroids_path = os.path.join(results_dir, "model_centroids.png")
    plt.savefig(centroids_path)
    plt.close()
    print(f"[plot_all_for_dir] Saved: {centroids_path}")


# ============================================
# Top-level: iterate all subfolders
# ============================================
def process_all_subfolders():
    """
    Look inside base_dir/results, and for each subfolder:
    - build comparison.csv (with scores + selected_model)
    - generate the two plots
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_root = os.path.join(base_dir, "results")

    for name in os.listdir(results_root):
        subdir = os.path.join(results_root, name)
        if not os.path.isdir(subdir):
            continue

        print(f"\n=== Processing {subdir} ===")
        try:
            comparison_path = combine_csv_for_dir(subdir)
            plot_all_for_dir(subdir, comparison_path)
        except FileNotFoundError as e:
            print(f"  [WARN] Skipping {subdir}: {e}")
        except Exception as e:
            print(f"  [ERROR] Problem in {subdir}: {e}")


if __name__ == "__main__":
    process_all_subfolders()