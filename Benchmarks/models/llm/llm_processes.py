# models/llm/llm_processes.py
# LEVEL-MODE: predict the next VALUE directly (no deltas). Autoregressive over H steps.

import re
import numpy as np
from transformers import pipeline


def _extract_first_value_tag(text: str):
    """
    Extract a single float between <VALUE> and </VALUE>, ignoring any text after </VALUE>.
    """
    if not text:
        return None
    # normalize unicode minus; cut at first closing tag to avoid trailing echo
    text = text.replace("\u2212", "-")
    cut = text.split("</VALUE>", 1)[0]
    m = re.search(
        r"<VALUE>\s*([-+]?(?:\d+(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?)",
        cut,
        re.IGNORECASE | re.DOTALL,
    )
    if m:
        try:
            return float(m.group(1))
        except Exception:
            return None
    return None


def _extract_floats_anywhere(text: str, k: int = 3):
    """Fallback: up to k float-like numbers from text."""
    if not text:
        return []
    text = text.replace("\u2212", "-")
    nums = re.findall(r"[-+]?(?:\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", text)
    out = []
    for n in nums:
        try:
            out.append(float(n))
        except Exception:
            continue
        if len(out) >= k:
            break
    return out


class LLMPForecaster:
    """
    Autoregressive LLM forecaster (LEVEL mode).
    Each step asks the model for the next VALUE (not a delta), parses one number,
    and uses it as the prediction for that timestamp.
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
                device_map="auto",
                torch_dtype="auto",
            )

    # ---------- prompt ----------

    def _format_step_prompt(self, hist_times, hist_vals, next_time) -> str:
        """
        Ask for the NEXT VALUE directly and force a single number between <VALUE> tags.
        Include a tiny few-shot to bias the model to the exact format and to discourage exact copying.
        """
        # provide enough context to infer local trend; last 40 points
        nctx = 40
        hist = "\n".join(f"{t}: {v:.6g}" for t, v in zip(hist_times[-nctx:], hist_vals[-nctx:]))

        return (
            "You are a precise numerical forecaster.\n"
            "Task: Predict the NEXT value for the given time series.\n"
            "Output ONLY one number between <VALUE> and </VALUE>. "
            "No words, no code, no extra text outside the tags.\n"
            "Do NOT simply repeat the last value unless the series is perfectly flat.\n\n"
            "Example:\n"
            "History (time: value):\n"
            "2024-01-01: 100\n"
            "2024-01-02: 101.2\n"
            "2024-01-03: 101.0\n"
            "Next timestamp: 2024-01-04\n"
            "<VALUE>\n"
            "101.08\n"                  # different from 101.0 to teach 'don't copy last'
            "</VALUE>\n\n"
            "History (time: value):\n"
            f"{hist}\n"
            f"Next timestamp: {next_time}\n"
            "<VALUE>\n"
        )

    # ---------- main ----------

    def __call__(self, task, n_samples: int = 1):
        """
        Returns (preds, info) with preds.shape == (1, H)
        """
        past_df = task.past_time
        fut_df = task.future_time

        past_times = list(past_df.index)
        fut_times = list(fut_df.index)
        seq = list(map(float, past_df["y"].values))  # known history (grows with predictions)
        H = len(fut_times)

        if self._pipe is None:
            # dry-run or no model: just repeat last value
            baseline = float(seq[-1]) if seq else 0.0
            return np.full((1, H), baseline, dtype=float), {"backend": "llmp_level_stub"}

        preds = []

        for h in range(H):
            prompt = self._format_step_prompt(past_times, seq, fut_times[h])

            # Mild sampling + repetition penalty to avoid trivial copying of last token value,
            # and average a few samples per step for stability.
            generations = self._pipe(
                prompt,
                return_full_text=False,
                do_sample=True,
                temperature=0.4,        # mild randomness; prevents greedy copy
                top_p=0.95,
                top_k=50,
                repetition_penalty=1.05,
                max_new_tokens=16,      # number + closing tag
                min_new_tokens=1,       # force at least something
                num_return_sequences=3, # we will average
            )

            vals = []
            for i, o in enumerate(generations):
                txt = o["generated_text"]
                v = _extract_first_value_tag(txt)
                if v is None:
                    floats = _extract_floats_anywhere(txt, k=1)
                    v = floats[0] if floats else None
                if v is not None and np.isfinite(v):
                    vals.append(float(v))

            if not vals:
                # last resort: repeat last value (change this if you prefer a different fallback)
                yhat = seq[-1] if seq else 0.0
            else:
                # mean over parsed candidates; median is also fine
                yhat = float(np.mean(vals))

            preds.append(yhat)
            seq.append(yhat)                 # becomes context for next step
            past_times.append(fut_times[h])  # advance the timeline

        preds = np.array(preds, dtype=float).reshape(1, H)
        return preds, {"backend": "llmp_level_step_sampling"}
