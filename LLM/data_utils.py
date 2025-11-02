# data_utils.py
from typing import List, Tuple
import numpy as np
import pandas as pd
import re

def load_series_csv(path: str) -> np.ndarray:
    """
    Reads a CSV and returns the numeric target column as a 1D array.
    If your CSV has more than one column, this loads the second column by default.
    Expected columns: date, y
    """
    df = pd.read_csv(path)
    # assume 'y' is the target column
    y = df["y"].astype(float).to_numpy()
    return y

def make_windows(y: np.ndarray, L: int, H: int, step: int = 1) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Produce rolling (history, future) windows."""
    out = []
    for t in range(L, len(y) - H + 1, step):
        hist = y[t - L:t].copy()
        fut = y[t:t + H].copy()
        out.append((hist, fut))
    return out

def zscore(x: np.ndarray):
    mu = float(np.mean(x))
    sd = float(np.std(x) + 1e-8)
    return (x - mu) / sd, mu, sd

def inv_zscore(x: np.ndarray, mu: float, sd: float):
    return x * sd + mu

def extract_numbers(text: str):
    """Robustly extract floats from generated text."""
    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", text)
    return [float(t) for t in nums]
