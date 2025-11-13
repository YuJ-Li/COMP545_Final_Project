# models/llm/llm_processes.py
"""
LLMPForecaster — step-by-step LLM forecaster (DELTA mode) with robust numeric parsing.

Goals:
  • No straight line: predict deltas each step, then accumulate (y_next = last + delta).
  • No sudden drops/spikes: strict BEGIN/END parsing, reject outliers, average a few samples,
    and soft-clip only *accepted* deltas to a wide band of recent volatility.

Compatible with your existing runner and DirectPrompt.
"""

from __future__ import annotations

import re
import math
from typing import List, Tuple

import numpy as np
from transformers import pipeline

# ----------------------------- parsing utils -----------------------------

_NUM_LINE = r"^\s*[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?\s*$"

def _parse_number_line_block(text: str) -> float | None:
    """Return the first line that is JUST a number between BEGIN and END."""
    start = text.find("BEGIN")
    if start != -1:
        text = text[start + len("BEGIN") :]
    end = text.find("END")
    if end != -1:
        text = text[:end]
    for line in text.splitlines():
        s = line.strip().replace("\u2212", "-")  # normalize unicode minus
        if re.match(_NUM_LINE, s):
            try:
                return float(s)
            except Exception:
                pass
    return None

def _extract_floats(text: str, k: int = 8) -> List[float]:
    """Fallback numeric extractor: first k floats/ints found."""
    text = text.replace("\u2212", "-")
    toks = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    vals: List[float] = []
    for t in toks:
        try:
            vals.append(float(t))
        except Exception:
            continue
        if len(vals) >= k:
            break
    return vals

# ----------------------------- scale & checks -----------------------------

def _recent_scale(levels: List[float], k: int = 10) -> float:
    """
    Volatility proxy: std of recent deltas over last k points.
    Never below 1e-6 to avoid zero bands.
    """
    if len(levels) >= k:
        recent = np.diff(levels[-k:], prepend=levels[-k])
        s = float(np.std(recent))
        return s if s > 1e-6 else 1.0
    return 1.0

def _is_year_like(x: float, lo: int = 1900, hi: int = 2100) -> bool:
    return float(x).is_integer() and lo <= x <= hi

def _accept_delta(d: float, scale: float, sigma: float = 6.0) -> bool:
    """Accept if |delta| within sigma * scale."""
    return np.isfinite(d) and abs(d) <= sigma * max(scale, 1e-6)

# ----------------------------- prompts -----------------------------------

def _format_step_prompt(hist_times, hist_vals, next_time) -> str:
    """
    DELTA mode: ask for the change (delta = next_value - last_value).
    """
    hist = "\n".join(f"{t}: {v:.6g}" for t, v in zip(hist_times, hist_vals))
    return (
        "You are a careful time-series forecaster.\n"
        "Task: Predict the CHANGE (delta) for the next timestamp, where:\n"
        "  delta = next_value - last_value\n"
        "Return ONLY the delta (a single real number). "
        "Print it on its own line between BEGIN and END.\n\n"
        f"History (time: value):\n{hist}\n\n"
        f"Next timestamp: {next_time}\n\nBEGIN\n"
    )

# ----------------------------- main class --------------------------------

class LLMPForecaster:
    """
    Autoregressive (step-by-step) LLM forecaster in DELTA mode.

    • Each step generates a delta; next value = last + delta.
    • Strict number-only parsing between BEGIN/END.
    • Rejects year-like echoes and outliers; averages a few plausible samples.
    • Soft clip only *after* acceptance (wide band), else fallback to 0 delta.
    """

    def __init__(self, model_id: str, use_context: bool = True, dry_run: bool = False):
        self.model_id = model_id
        self.use_context = use_context
        self.dry_run = dry_run
        self._pipe = None
        if not dry_run:
            self._pipe = pipeline(
                "text-generation",
                model=model_id,
                torch_dtype="auto",
                device_map="auto",
            )

    def __call__(self, task, n_samples: int = 1) -> Tuple[np.ndarray, dict]:
        past_df = task.past_time
        fut_df  = task.future_time

        past_times = list(past_df.index)
        seq        = list(map(float, past_df["y"].values))  # growing levels
        fut_times  = list(fut_df.index)
        H          = len(fut_times)

        if self._pipe is None:
            # deterministic stub: repeat-last
            baseline = float(seq[-1]) if seq else 0.0
            return np.full((1, H), baseline, dtype=float), {"backend": "llmp_stub"}

        info = {"backend": "llmp_delta_step"}

        for h in range(H):
            last = seq[-1]
            scale = _recent_scale(seq, k=10)

            accepted: List[float] = []
            try:
                # Two attempts with small batches to keep it fast but not brittle
                attempts = [
                    dict(max_new_tokens=20, do_sample=True, temperature=0.7, top_p=0.9, num_return_sequences=3),
                    dict(max_new_tokens=28, do_sample=True, temperature=0.6, top_p=0.9, num_return_sequences=3),
                ]
                for gen in attempts:
                    prompt = _format_step_prompt(past_times, seq, fut_times[h]) + "END\n"
                    out = self._pipe(prompt, return_full_text=False, **gen)  # list of dicts

                    for o in out:
                        txt = o["generated_text"]
                        # 1) strict parse
                        v = _parse_number_line_block(txt)
                        # 2) fallback: take first plausible float
                        if v is None:
                            floats = _extract_floats(txt, k=6)
                            # filter out years; in delta mode prefer small magnitude
                            floats = [f for f in floats if not _is_year_like(f)]
                            if floats:
                                # choose the one closest to zero (delta should be small)
                                v = min(floats, key=lambda x: abs(x))

                        # validate as a delta
                        if v is not None and _accept_delta(v, scale, sigma=6.0):
                            accepted.append(float(v))

                    if accepted:
                        break  # we have some plausible deltas; stop attempts

            except Exception as e:
                print(f"[LLMP step {h}] warning: {e} — using fallback (delta=0).")

            # decide delta
            if accepted:
                delta = float(np.mean(accepted))
                # very soft post-accept clip (safety only)
                delta = float(np.clip(delta, -6.0 * scale, 6.0 * scale))
            else:
                delta = 0.0  # fallback = no change for this step

            yhat = last + delta
            seq.append(yhat)
            past_times.append(fut_times[h])

        preds = np.array(seq[-H:], dtype=float).reshape(1, H)
        return preds, info
