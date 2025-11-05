from typing import Tuple, List
import numpy as np
import re

def _extract_floats(text: str, k: int) -> List[float]:
    # Same robust parser as direct_prompt
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

def _format_step_prompt(hist_times, hist_vals, next_time):
    hist = "\n".join(f"{t}: {v:.6g}" for t, v in zip(hist_times, hist_vals))
    return (
        "You are a careful time-series forecaster.\n"
        "Given past values, predict the next single value ONLY as a number.\n\n"
        f"History:\n{hist}\n\n"
        f"Timestamp to predict: {next_time}\n\n"
        "Output:\n"
    )

class LLMPForecaster:
    """Autoregressive (step-by-step) forecaster for a flat (single-folder) project layout."""
    def __init__(self, llm_type: str, use_context: bool=True, dry_run: bool=False):
        self.model = {
            "phi2": "microsoft/phi-2",
            "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "qwen": "Qwen/Qwen2.5-0.5B-Instruct",
            "qwen2.5-0.5b-instruct": "Qwen/Qwen2.5-0.5B-Instruct",
        }.get(llm_type, llm_type)
        self.use_context = use_context
        self.dry_run = dry_run
        self._pipe = None
        if not dry_run:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            tok = AutoTokenizer.from_pretrained(self.model)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            mdl = AutoModelForCausalLM.from_pretrained(self.model, device_map="auto", dtype="auto")
            if getattr(mdl.config, "pad_token_id", None) is None:
                mdl.config.pad_token_id = tok.pad_token_id
            self._pipe = pipeline("text-generation", model=mdl, tokenizer=tok)

    def __call__(self, task_instance, n_samples: int = 4):
        # Expect .past_time and .future_time DataFrames
        hist_times = task_instance.past_time.index.strftime("%Y-%m-%d %H:%M:%S").tolist()
        hist_vals  = task_instance.past_time.values[:, -1].astype(float).tolist()
        fut_times  = task_instance.future_time.index.strftime("%Y-%m-%d %H:%M:%S").tolist()
        H = len(fut_times)

        samples = []
        info = {"backend":"llmp"}

        for _ in range(max(n_samples, 1)):
            seq = []
            context_times = hist_times[:]
            context_vals  = hist_vals[:]
            for h in range(H):
                if self._pipe is None:
                    yhat = context_vals[-1]
                else:
                    p = _format_step_prompt(context_times, context_vals, fut_times[h])
                    # Deterministic, return only generated text (avoid prompt slicing issues)
                    out = self._pipe(
                        p,
                        max_new_tokens=16,
                        do_sample=False,
                        temperature=1.0,
                        num_return_sequences=1,
                        return_full_text=False,
                    )
                    vals = _extract_floats(out[0]["generated_text"], 1)
                    yhat = vals[0] if vals else context_vals[-1]
                seq.append(yhat)
                context_times.append(fut_times[h])
                context_vals.append(yhat)
            samples.append(seq)

        arr = np.array(samples, dtype=float)[:, :, None]  # [n_samples, H, 1]
        return arr, info
