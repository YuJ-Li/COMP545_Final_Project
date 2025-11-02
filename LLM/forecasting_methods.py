# forecasting_methods.py
import json
import re
from typing import List
import numpy as np

# ------------------------------------------------------------------
# Cleaning & parsing
# ------------------------------------------------------------------

def _clean_llm_output(text: str) -> str:
    """
    Keep only content after the 'answer:' anchor (if present),
    strip code fences, and trim at first illegal char for JSON-like output.
    Handles None or non-string safely.
    """
    if not isinstance(text, str):
        return ""
    t = text
    ans_idx = t.lower().find("answer:")
    if ans_idx != -1:
        t = t[ans_idx + len("answer:"):]
    # remove code fences
    t = re.sub(r"```.*?```", "", t, flags=re.S)
    # trim at first illegal char
    m = re.search(r"[^\d\.\,\-\[\]\sEe\+]", t)
    if m:
        t = t[:m.start()]
    return t.strip()


def _extract_array(text: str) -> List[float]:
    """
    Prefer the last [...] block; else regex all numbers.
    """
    l = text.rfind("[")
    r = text.find("]", l + 1) if l != -1 else -1
    if l != -1 and r != -1:
        try:
            arr = json.loads(text[l:r+1])
            if isinstance(arr, list):
                return [float(x) for x in arr]
        except Exception:
            pass
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    return [float(x) for x in nums]


def _finish_to_exact_H(values: List[float], H: int, hist_last: float) -> np.ndarray:
    """
    Ensure exactly H values by padding/truncating.
    - If empty, seed with last history value.
    - If short, repeat last value until H.
    - If long, take the first H.
    """
    vals = list(values)
    if not vals:
        vals = [hist_last]
    if len(vals) < H:
        vals = vals + [vals[-1]] * (H - len(vals))
    return np.array(vals[:H], dtype=np.float32)


# ------------------------------------------------------------------
# Prompt & call loop
# ------------------------------------------------------------------

_JSON_RULE = "Output ONLY one JSON array of exactly {H} numbers. No prose, no code fences, no units."

def _make_prompt(role: str, history: np.ndarray, H: int) -> str:
    hist_str = ", ".join(f"{x:.4f}" for x in history.tolist())
    # End with 'answer:' so we have a stable anchor for the parser
    return (
        f"{role}\n"
        f"Given the time series (most recent last), {_JSON_RULE.format(H=H)}\n"
        f"history: [{hist_str}]\n"
        f"answer: "
    )


def _call_exact_H(llm_call_fn, base_prompt: str, H: int, tag: str, hist_last: float) -> np.ndarray:
    """
    Enforce EXACTLY H values with a minimal retry strategy:
      - Attempt #1: if len(arr) >= H → truncate & return immediately
      - If len(arr) < H → Attempt #2 with a strict repair prompt
      - If still short → pad/truncate locally and return
    """
    attempts = 0
    got: List[float] = []

    # ---- attempt #1
    attempts += 1
    try:
        text1 = llm_call_fn(base_prompt, max_new_tokens=max(64, H * 24), H=H)
    except Exception as e:
        print(f"[ERR][{tag}#{attempts}] generation failed: {e}")
        text1 = ""
    t1 = _clean_llm_output(text1)
    print(f"[RAW][{tag}#{attempts}] {t1[:180].replace(chr(10),' ')}")
    arr1 = _extract_array(t1)

    if len(arr1) >= H:
        return np.array(arr1[:H], dtype=np.float32)

    got = arr1 or got

    # ---- attempt #2 (only when < H)
    attempts += 1
    repair = (
        f"You returned {len(arr1)} values; return EXACTLY {H} numbers as a single JSON array like "
        f"[v1, v2, ..., v{H}] with no text.\n"
        f"answer: "
    )
    try:
        text2 = llm_call_fn(base_prompt + repair, max_new_tokens=max(64, H * 24), H=H)
    except Exception as e:
        print(f"[ERR][{tag}#{attempts}] generation failed: {e}")
        text2 = ""
    t2 = _clean_llm_output(text2)
    print(f"[RAW][{tag}#{attempts}] {t2[:180].replace(chr(10),' ')}")
    arr2 = _extract_array(t2)

    if len(arr2) >= H:
        return np.array(arr2[:H], dtype=np.float32)

    # ---- final safety
    print(f"[WARN][{tag}] Could not get exactly {H} values after {attempts} attempts; padding/truncating locally.")
    base = arr2 if arr2 else (arr1 if arr1 else [hist_last])
    return _finish_to_exact_H(base, H, hist_last)


# ------------------------------------------------------------------
# Public forecasters
# ------------------------------------------------------------------

def dp_forecast(llm_call_fn, history: np.ndarray, H: int) -> np.ndarray:
    """
    DP: precise / literal extrapolation
    """
    prompt = _make_prompt("You are a precise time-series forecaster.", history, H)
    return _call_exact_H(llm_call_fn, prompt, H, tag="DP", hist_last=float(history[-1]))


def llmp_forecast(llm_call_fn, history: np.ndarray, H: int) -> np.ndarray:
    """
    LLMP: robust / smoothed extrapolation
    """
    prompt = _make_prompt("You are a robust forecaster. Regularize for smoothness; avoid overreacting to noise.", history, H)
    return _call_exact_H(llm_call_fn, prompt, H, tag="LLMP", hist_last=float(history[-1]))
