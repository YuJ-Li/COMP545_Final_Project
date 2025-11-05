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
    # Find date column
    ds_col = None
    for name in ["ds", "Date", "date"]:
        if name in df.columns:
            ds_col = name
            break
    if ds_col is None:
        ds_col = df.columns[0]

    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in data columns: {list(df.columns)}")

    df = df[[ds_col, target_col]].rename(columns={ds_col: "ds", target_col: "y"})
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.dropna()
    return df


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
    ap.add_argument("--target", required=True, help="target column to predict (e.g., Close)")
    ap.add_argument("--model", required=True,
                    choices=["phi2", "tinyllama", "qwen", "qwen2.5-0.5b-instruct"])
    ap.add_argument("--backend", required=True, choices=["dp", "llmp"])
    ap.add_argument("--L", type=int, default=100, help="context length")
    ap.add_argument("--H", type=int, default=10, help="horizon")
    ap.add_argument("--step", type=int, default=10, help="stride between windows")
    ap.add_argument("--seasonality", type=int, default=1)
    ap.add_argument("--max_windows", type=int, default=None,
                    help="cap number of windows for quick runs")
    ap.add_argument("--save_dir", default="results_llm", help="where to save results")
    args = ap.parse_args()

    # Load target column only
    df = load_series(args.data, args.target)

    # Split train/test
    y = df["y"].values
    train_size = max(args.L, int(len(y) * 0.5))
    train_data = y[:train_size]
    test_data = y[train_size:]

    model = make_model(args.model, args.backend)
    evaluator = TimeSeriesEvaluator(quantiles=[0.1, 0.5, 0.9])

    # Run evaluation
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

    # -------------------------
    # Format output
    # -------------------------
    agg = results.get("metrics", {})
    row = {"model": args.model}
    for metric_name, stats in agg.items():
        if isinstance(stats, dict) and "mean" in stats:
            row[metric_name] = stats["mean"]

    df_out = pd.DataFrame([row]).set_index("model")

    # Save results
    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"metrics_{args.target}.csv"
    df_out.to_csv(out_csv)

    # Save JSON (optional full result)
    out_json = out_dir / f"results_{args.model}_{args.target}.json"
    def _coerce(o):
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
        return o
    with open(out_json, "w") as f:
        json.dump(results, f, indent=2, default=_coerce)

    print(f"\n✓ Finished evaluating '{args.target}' with {args.model} ({args.backend})")
    print(f"Saved metrics CSV: {out_csv}")
    print(f"Saved full JSON:  {out_json}")
    print("\n=== Summary ===")
    print(df_out.round(4))


if __name__ == "__main__":
    main()
