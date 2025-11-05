from typing import Tuple, List
import numpy as np
import re

def _format_prompt(hist_times: List[str], hist_vals: List[float], fut_times: List[str]) -> str:
    hist = "\n".join(f"{t}: {v:.6g}" for t, v in zip(hist_times, hist_vals))
    fut  = "\n".join(f"{t}" for t in fut_times)
    return (
        "You are a careful time-series forecaster.\n"
        "Given past values, forecast the next values strictly as plain numbers.\n"
        "Return ONLY the numbers, one per line, no text.\n\n"
        f"History:\n{hist}\n\n"
        f"Timestamps to predict:\n{fut}\n\n"
        "Output:\n"
    )

def _extract_floats(text: str, k: int) -> List[float]:
    # Robust numeric capture: handle commas and unicode minus, avoid malformed tokens
    text = text.replace(",", "")
    text = text.replace("\u2212", "-")
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    vals: List[float] = []
    for tok in nums:
        try:
            vals.append(float(tok))
        except Exception:
            continue
        if len(vals) >= k:
            break
    return vals

class DirectPrompt:
    """Dependency-light forecaster for a flat (single-folder) project layout."""
    def __init__(self, model: str, use_context: bool=True, n_retries: int=3, temperature: float=1.0,
                 dry_run: bool=False, constrained_decoding: bool=False):
        self.model = model
        self.temperature = temperature
        self.n_retries = n_retries
        self.use_context = use_context
        self.dry_run = dry_run
        self._pipe = None
        if not dry_run:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            tok = AutoTokenizer.from_pretrained(model)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            mdl = AutoModelForCausalLM.from_pretrained(model, device_map="auto", dtype="auto")
            if getattr(mdl.config, "pad_token_id", None) is None:
                mdl.config.pad_token_id = tok.pad_token_id
            self._pipe = pipeline("text-generation", model=mdl, tokenizer=tok)

    def __call__(self, task_instance, n_samples: int = 8) -> Tuple[np.ndarray, dict]:
        # Expect .past_time and .future_time DataFrames (index datetime, last column is value)
        hist_times = task_instance.past_time.index.strftime("%Y-%m-%d %H:%M:%S").tolist()
        hist_vals  = task_instance.past_time.values[:, -1].astype(float).tolist()
        fut_times  = task_instance.future_time.index.strftime("%Y-%m-%d %H:%M:%S").tolist()
        H = len(fut_times)

        prompt = _format_prompt(hist_times, hist_vals, fut_times)

        if self._pipe is None:
            samples = [[hist_vals[-1]] * H] * max(n_samples, 1)
        else:
            out = self._pipe(
                prompt,
                max_new_tokens=H * 16,
                temperature=self.temperature,
                do_sample=True,
                num_return_sequences=max(n_samples, 1),  # batch to avoid many GPU calls
                return_full_text=False,                  # only generated text (no prompt slicing)
            )
            samples = []
            for o in out:
                seq = _extract_floats(o["generated_text"], H)
                if len(seq) == H:
                    samples.append(seq)
            if not samples:
                # fall back to a simple naive repeat if parsing fails
                samples = [[hist_vals[-1]] * H]

        arr = np.array(samples, dtype=float)[:, :, None]  # [n_samples, H, 1]
        return arr, {"backend": "direct_prompt"}
