# run_benchmark.py
import argparse
import numpy as np
from typing import Dict, List, Tuple

from data_utils import load_series_csv, make_windows
from hf_llm import load_llm, llm_call
from forecasting_methods import dp_forecast, llmp_forecast


# ---- metrics ----
def mae(yhat, ytrue): return float(np.mean(np.abs(yhat - ytrue)))
def rmse(yhat, ytrue): return float(np.sqrt(np.mean((yhat - ytrue) ** 2)))
def smape(yhat, ytrue):
    num = np.abs(yhat - ytrue)
    den = (np.abs(yhat) + np.abs(ytrue)) / 2.0 + 1e-8
    return 100.0 * float(np.mean(num / den))
def mase(yhat, ytrue, insample, m=1):
    denom = np.mean(np.abs(insample[m:] - insample[:-m])) + 1e-8
    return float(np.mean(np.abs(yhat - ytrue)) / denom)


def evaluate_windows(
    windows: List[Tuple[np.ndarray, np.ndarray]],
    forecaster,
) -> Dict[str, float]:
    """Run a (history -> forecast) function across windows and average metrics."""
    M = {"MAE": [], "RMSE": [], "sMAPE": [], "MASE": []}
    for hist, fut in windows:
        yhat = forecaster(hist, len(fut))
        M["MAE"].append(mae(yhat, fut))
        M["RMSE"].append(rmse(yhat, fut))
        M["sMAPE"].append(smape(yhat, fut))
        M["MASE"].append(mase(yhat, fut, insample=hist))
    return {k: float(np.mean(v)) for k, v in M.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="finance_data.csv",
                    help="CSV with columns: date,y (we read column 'y')")
    ap.add_argument("--L", type=int, default=30, help="history length")
    ap.add_argument("--H", type=int, default=5, help="forecast horizon")
    ap.add_argument("--step", type=int, default=5, help="stride for rolling windows")
    ap.add_argument("--device", type=str, default=None, help="cuda or cpu (auto if None)")
    ap.add_argument("--four_bit", action="store_true", help="enable 4-bit loading (GPU only)")
    ap.add_argument("--max_windows", type=int, default=10, help="subsample windows for speed")

    # NEW: sampling / reproducibility / model control
    ap.add_argument("--temperature", type=float, default=0.7,
                    help=">0 enables sampling; 0.7 recommended for diversity")
    ap.add_argument("--top_p", type=float, default=0.9,
                    help="nucleus sampling proportion")
    ap.add_argument("--seed_base", type=int, default=1234,
                    help="base RNG seed; we offset per-model for reproducible diversity")
    ap.add_argument("--model", type=str, default=None,
                    help="optional single HF model id to run (overrides default list)")
    args = ap.parse_args()

    # Model list (can be overridden via --model)
    if args.model:
        models = [args.model]
    else:
        models = [
            "microsoft/Phi-2",
            "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # 1.1B chat-tuned
            "Qwen/Qwen2.5-0.5B-Instruct"          # 0.5B instruct
        ]

    # Load data + windows
    y = load_series_csv(args.data)
    windows_all = make_windows(y, args.L, args.H, step=args.step)
    windows = windows_all[: args.max_windows]
    print(f"Loaded {len(y)} points; evaluating on {len(windows)} windows "
          f"(L={args.L}, H={args.H}, step={args.step})")

    results = []
    for i, mname in enumerate(models):
        print(f"\n=== Loading model: {mname} (four_bit={args.four_bit}) ===")
        tok, model = load_llm(mname, device=args.device, four_bit=args.four_bit)

        # Per-model seed → distinct yet reproducible sampling
        seed = args.seed_base + i

        # Wrapper that accepts both positional or keyword max token args
        def call(prompt, max_new=None, **kwargs):
            if max_new is None:
                max_new = kwargs.get("max_new_tokens", 64)
            return llm_call(
                model, tok, prompt,
                max_new_tokens=max_new,
                temperature=args.temperature,
                top_p=args.top_p,
                seed=seed,
            )

        # Evaluate DP and LLMP forecasters
        res_dp   = evaluate_windows(windows, lambda h, H: dp_forecast(call, h, H))
        res_llmp = evaluate_windows(windows, lambda h, H: llmp_forecast(call, h, H))

        results.append(("DP", mname, res_dp))
        results.append(("LLMP", mname, res_llmp))

        print(f"  DP   | MAE {res_dp['MAE']:.4f} | RMSE {res_dp['RMSE']:.4f} | "
              f"sMAPE {res_dp['sMAPE']:.2f} | MASE {res_dp['MASE']:.4f}")
        print(f"  LLMP | MAE {res_llmp['MAE']:.4f} | RMSE {res_llmp['RMSE']:.4f} | "
              f"sMAPE {res_llmp['sMAPE']:.2f} | MASE {res_llmp['MASE']:.4f}")

    print("\n=== Summary ===")
    for method, name, met in results:
        print(f"{method:>5} | {name:35s} | MAE {met['MAE']:.4f} | RMSE {met['RMSE']:.4f} | "
              f"sMAPE {met['sMAPE']:.2f} | MASE {met['MASE']:.4f}")


if __name__ == "__main__":
    main()
