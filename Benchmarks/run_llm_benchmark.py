import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from llm_wrapper import LLMDirectPromptModel, LLMPModel
from evaluator import TimeSeriesEvaluator


# -----------------------------
# Data loader (normalize to ds,y)
# -----------------------------
def load_series(path: str, target_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Standardize date column to ds
    if "ds" not in df.columns:
        if "Date" in df.columns:
            df = df.rename(columns={"Date": "ds"})
        elif "date" in df.columns:
            df = df.rename(columns={"date": "ds"})
        else:
            df = df.rename(columns={df.columns[0]: "ds"})
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in CSV columns: {list(df.columns)}")
    df = df[["ds", target_col]].rename(columns={target_col: "y"})
    df = df.dropna()
    df["ds"] = pd.to_datetime(df["ds"])
    return df


# --------------------
# Model factory
# --------------------
def make_model(kind: str, backend: str):
    if backend == "dp":
        return LLMDirectPromptModel(kind, temperature=1.0, use_context=True)
    elif backend == "llmp":
        return LLMPModel(kind, use_context=True)
    else:
        raise ValueError("backend must be 'dp' or 'llmp'")


# --------------------
# Main
# --------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--target", required=True, help="Target column to evaluate (e.g., Close)")
    ap.add_argument("--model", required=True,
                    choices=["phi2", "tinyllama", "qwen", "qwen2.5-0.5b-instruct"])
    ap.add_argument("--backend", required=True, choices=["dp", "llmp"])
    ap.add_argument("--L", type=int, default=100, help="context length")
    ap.add_argument("--H", type=int, default=10, help="horizon")
    ap.add_argument("--step", type=int, default=10, help="stride between windows")
    ap.add_argument("--seasonality", type=int, default=1)
    ap.add_argument("--max_windows", type=int, default=None,
                    help="cap number of windows for quick runs")
    ap.add_argument("--out_json", default=None,
                    help="save aggregated+per-window results to JSON")
    ap.add_argument("--out_csv", default="results_llm/metrics_summary.csv",
                    help="path to aggregated CSV (one row per model)")
    args = ap.parse_args()

    # Load & split
    df = load_series(args.data, args.target)
    y = df["y"].values
    train_size = max(args.L, int(len(y) * 0.5))
    train_data = y[:train_size]
    test_data = y[train_size:]

    # Build model + evaluator
    model = make_model(args.model, args.backend)
    evaluator = TimeSeriesEvaluator(quantiles=[0.1, 0.5, 0.9])

    # Evaluate (calls metrics.compute_all_metrics)
    results = evaluator.evaluate_model(
        model=model,
        data=test_data,
        train_data=train_data,
        context_length=args.L,
        horizon=args.H,
        step=args.step,
        max_windows=args.max_windows,
        seasonality=args.seasonality,
        verbose=True,
    )

    # ------------
    # Print summary
    # ------------
    print("\n=== Aggregated Metrics (mean ± std) ===")
    metrics = results.get("metrics", {})
    for metric_name, stats in metrics.items():
        print(f"{metric_name:25s}: {stats['mean']:.4f} ± {stats['std']:.4f}")

    # -------------------------
    # Save outputs
    # -------------------------
    out_dir = Path(args.out_csv).parent
    out_dir.mkdir(parents=True, exist_ok=True)

    # Convert to one-row summary (model = row name, metrics = columns)
    row = {"model": args.model}
    for name, stats in metrics.items():
        if "mean" in stats:
            row[name] = stats["mean"]

    df_out = pd.DataFrame([row]).set_index("model")

    # Append or create CSV
    if Path(args.out_csv).exists():
        old = pd.read_csv(args.out_csv, index_col=0)
        # Replace model row if already exists
        old.loc[args.model] = df_out.loc[args.model]
        df_out = old
    df_out.to_csv(args.out_csv)

    print(f"\n✓ Metrics saved to {args.out_csv}")

    # Save JSON (full evaluator output)
    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)

        def _coerce(o):
            if isinstance(o, (np.floating, np.integer)):
                return o.item()
            if isinstance(o, np.ndarray):
                return o.tolist()
            return o

        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2, default=_coerce)
        print("Saved JSON:", args.out_json)


if __name__ == "__main__":
    main()
