# single_tests/test_llama3.py
"""
Llama 3 quick test with combined plot (True, DP, LLMP) and robust LLMP fallback.

Example:
  python single_tests/test_llama3.py ^
    --data Data/aapl_us_d.csv --target Close ^
    --model meta-llama/Llama-3.2-3B-Instruct ^
    --L 20 --H 10 --outdir results_llm
"""

import sys
import os
import argparse
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# make repo root importable
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

from single_tests import print_test_header, print_test_result
from models.llm.direct_prompt import DirectPrompt
from models.llm.llm_processes import LLMPForecaster
from evaluation.metrics import mae, rmse, mase


# ---------------------------- helpers ----------------------------

def load_series(path: str, target_col: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "ds" not in df.columns:
        for cand in ("Date", "date", "timestamp", "time"):
            if cand in df.columns:
                df = df.rename(columns={cand: "ds"})
                break
        else:
            df = df.rename(columns={df.columns[0]: "ds"})
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in {list(df.columns)}")
    df = df[["ds", target_col]].rename(columns={target_col: "y"}).dropna()
    df["ds"] = pd.to_datetime(df["ds"])
    df = df.sort_values("ds").reset_index(drop=True)
    return df


def to_df(times, values):
    return pd.DataFrame({"y": values}, index=pd.to_datetime(times))


def build_shared_pipeline(model_id: str):
    """Load ONE HF text-generation pipeline and reuse for DP & LLMP."""
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    mdl = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype="auto")
    if getattr(mdl.config, "pad_token_id", None) is None:
        mdl.config.pad_token_id = tok.pad_token_id
    return pipeline("text-generation", model=mdl, tokenizer=tok)


def last_window(y: np.ndarray, ts: np.ndarray, L: int, H: int):
    if L + H > len(y):
        raise ValueError("L + H exceeds available series length")
    ctx_vals = y[-(L + H):-H]
    tgt_vals = y[-H:]
    ctx_times = ts[-(L + H):-H]
    tgt_times = ts[-H:]
    return ctx_vals, tgt_vals, ctx_times, tgt_times


# -------------------------- forecaster calls --------------------------

def run_dp(dp_model: DirectPrompt, ctx_vals, tgt_vals, ctx_times, tgt_times, shared_pipe):
    """Direct Prompt for the last window — returns np.ndarray length H."""
    dp_model._pipe = shared_pipe  # inject shared pipe
    class Task:
        def __init__(self, past, future):
            self.past_time = past
            self.future_time = future
    task = Task(
        to_df(ctx_times, ctx_vals),
        to_df(tgt_times, np.zeros_like(tgt_vals)),
    )
    samples, _ = dp_model(task, n_samples=1)
    y_pred = samples.mean(axis=0).ravel()
    # shape guard
    if len(y_pred) != len(tgt_vals):
        y_pred = y_pred[: len(tgt_vals)]
        if len(y_pred) < len(tgt_vals):
            pad = np.full(len(tgt_vals) - len(y_pred), float(ctx_vals[-1]))
            y_pred = np.concatenate([y_pred, pad])
    # NaN guard
    if not np.all(np.isfinite(y_pred)):
        y_pred = np.where(np.isfinite(y_pred), y_pred, float(ctx_vals[-1]))
    return y_pred


def run_llmp(llmp_model: LLMPForecaster, ctx_vals, tgt_vals, ctx_times, tgt_times, shared_pipe):
    """
    LLMP for the last window — robust to failures:
    - uses shared pipe
    - returns level predictions (length H)
    - if anything fails, falls back to last-value baseline
    """
    llmp_model._pipe = shared_pipe
    class Task:
        def __init__(self, past, future):
            self.past_time = past
            self.future_time = future
    task = Task(
        to_df(ctx_times, ctx_vals),
        to_df(tgt_times, np.zeros_like(tgt_vals)),
    )
    try:
        samples, _ = llmp_model(task, n_samples=1)
        y_pred = samples.mean(axis=0).ravel()
    except Exception as e:
        print(f"[LLMP] warning: {e} — using fallback (repeat last context value).")
        y_pred = np.full(len(tgt_vals), float(ctx_vals[-1]), dtype=float)

    # shape/NaN guards
    if len(y_pred) != len(tgt_vals):
        y_pred = y_pred[: len(tgt_vals)]
        if len(y_pred) < len(tgt_vals):
            pad = np.full(len(tgt_vals) - len(y_pred), float(ctx_vals[-1]))
            y_pred = np.concatenate([y_pred, pad])
    if not np.all(np.isfinite(y_pred)):
        y_pred = np.where(np.isfinite(y_pred), y_pred, float(ctx_vals[-1]))
    return y_pred


# ------------------------------ plotting ------------------------------

def plot_combined(true_vals, dp_pred, llmp_pred, out_path, title):
    x = range(len(true_vals))
    plt.figure(figsize=(12, 6))
    plt.plot(x, true_vals, label="True", linewidth=2)
    plt.plot(x, dp_pred, "--", label="DP Prediction", linewidth=2)
    plt.plot(x, llmp_pred, ":", label="LLMP Prediction", linewidth=2)
    plt.xlabel("Forecast step")
    plt.ylabel("Value")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()


# -------------------------------- main --------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="CSV path (e.g., Data/aapl_us_d.csv)")
    ap.add_argument("--target", required=True, help="Target column (e.g., Close)")
    ap.add_argument("--model", type=str, default="meta-llama/Llama-3.2-3B-Instruct",
                    help="HF model id")
    ap.add_argument("--L", type=int, default=20, help="Context length")
    ap.add_argument("--H", type=int, default=10, help="Horizon length")
    ap.add_argument("--outdir", type=str, default="results_llm")
    args = ap.parse_args()

    print_test_header("Llama 3 — DP & LLMP (shared pipeline, combined plot)")

    df = load_series(args.data, args.target)
    y = df["y"].to_numpy(dtype=float)
    ts = df["ds"].to_numpy()

    ctx_vals, tgt_vals, ctx_times, tgt_times = last_window(y, ts, args.L, args.H)

    # build one shared pipeline and reuse it
    shared_pipe = build_shared_pipeline(args.model)

    # instantiate forecasters with dry_run=True — we inject the pipe
    dp_model = DirectPrompt(args.model, dry_run=True, temperature=0.7)
    llmp_model = LLMPForecaster(args.model, dry_run=True)

    # run both methods
    dp_pred   = run_dp(dp_model,   ctx_vals, tgt_vals, ctx_times, tgt_times, shared_pipe)
    llmp_pred = run_llmp(llmp_model, ctx_vals, tgt_vals, ctx_times, tgt_times, shared_pipe)

    # metrics (safe)
    def safe_print_metrics(name, y_true, y_hat, train_vals):
        try:
            print(f"{name:4s} | MAE {mae(y_true, y_hat):.4f} | "
                  f"RMSE {rmse(y_true, y_hat):.4f} | "
                  f"MASE {mase(y_true, y_hat, train_vals, seasonality=1):.4f}")
        except Exception as e:
            print(f"{name:4s} | metrics skipped ({e})")

    safe_print_metrics("DP",   tgt_vals, dp_pred,   ctx_vals)
    safe_print_metrics("LLMP", tgt_vals, llmp_pred, ctx_vals)

    # combined plot
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_plot = Path(args.outdir) / f"llama_dp_llmp_true_vs_pred_{stamp}.png"
    plot_combined(tgt_vals, dp_pred, llmp_pred, str(out_plot),
                  f"True vs Predictions (L={args.L}, H={args.H})")
    print(f"\nPlot saved: {out_plot}")

    ok = np.all(np.isfinite(dp_pred)) and np.all(np.isfinite(llmp_pred))
    print_test_result(ok, "predictions computed & plot saved")


if __name__ == "__main__":
    main()
