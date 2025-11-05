from typing import Optional, Dict, List, Union
import numpy as np
import pandas as pd

from base_model import TimeSeriesModel
import direct_prompt as dp
import llm_processes as llmp

HF_ID_MAP = {
    "phi2": "microsoft/phi-2",
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "qwen": "Qwen/Qwen2.5-0.5B-Instruct",
    "qwen2.5-0.5b-instruct": "Qwen/Qwen2.5-0.5B-Instruct",
}

def _coerce_ts(ts):
    if ts is None:
        return None
    if isinstance(ts, pd.DatetimeIndex):
        return ts.astype('datetime64[ns]').astype(str)
    arr = np.array(ts)
    if np.issubdtype(arr.dtype, np.datetime64):
        return arr.astype('datetime64[ns]').astype(str)
    return arr.astype(str)

def _make_dummy_times(n_hist: int, n_fut: int, freq: str = "D"):
    # Create dummy datetime index if none is available (works with evaluator)
    end = pd.Timestamp.now().normalize()
    hist_idx = pd.date_range(end=end, periods=n_hist, freq=freq)
    fut_idx = pd.date_range(start=hist_idx[-1] + pd.tseries.frequencies.to_offset(freq), periods=n_fut, freq=freq)
    return hist_idx, fut_idx

class LLMDirectPromptModel(TimeSeriesModel):
    def __init__(self,
                 model_name: str,
                 temperature: float = 1.0,
                 use_context: bool = True,
                 n_retry: int = 5):
        super().__init__(name=f"LLM-DP[{model_name}]")
        hf_id = HF_ID_MAP.get(model_name.lower(), model_name)
        self.dp = dp.DirectPrompt(
            model=hf_id,
            use_context=use_context,
            n_retries=n_retry,
            temperature=temperature,
            dry_run=False,
            constrained_decoding=True,
        )
        self._train_ts = None

    def fit(self, y: np.ndarray, timestamps: Optional[np.ndarray] = None) -> None:
        # Optional: if timestamps provided by other pipelines, keep them
        self._train_ts = timestamps

    def predict(self,
                history: np.ndarray,
                horizon: int,
                timestamps_future: Optional[np.ndarray] = None,
                quantiles: Optional[List[float]] = None) -> Dict[str, np.ndarray]:
        # Build minimal task with pandas DataFrames
        if self._train_ts is not None:
            train_ts = _coerce_ts(self._train_ts)
            hist_index = pd.to_datetime(train_ts[-len(history):])
            if timestamps_future is not None:
                fut_index = pd.to_datetime(_coerce_ts(timestamps_future))
            else:
                # synthesize future from last hist timestamp
                fut_index = pd.date_range(start=hist_index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
        else:
            # no timestamps available → synthesize both
            hist_index, fut_index = _make_dummy_times(len(history), horizon, freq="D")

        hist_df = pd.DataFrame({"y": history}, index=hist_index)
        fut_df  = pd.DataFrame({"y": np.zeros(horizon)}, index=fut_index)

        class _Task:
            def __init__(self, past, future):
                self.past_time = past
                self.future_time = future
                self.background = ""
                self.constraints = ""
                self.scenario = ""
                self.max_directprompt_batch_size = 4

        task = _Task(hist_df, fut_df)
        samples, _ = self.dp(task, n_samples=4)  # smaller for speed
        preds = samples[:, :, 0]
        mean_pred = preds.mean(axis=0)
        median_pred = np.median(preds, axis=0)
        out = {'mean': mean_pred, 'median': median_pred}
        if quantiles is not None:
            out['quantiles'] = {q: np.quantile(preds, q, axis=0) for q in quantiles}
        return out

class LLMPModel(TimeSeriesModel):
    def __init__(self, model_name: str, use_context: bool = True):
        super().__init__(name=f"LLM-LLMP[{model_name}]")
        llm_type = model_name.lower() if model_name.lower() in HF_ID_MAP else model_name
        self.llmp = llmp.LLMPForecaster(llm_type=llm_type, use_context=use_context, dry_run=False)
        self._train_ts = None

    def fit(self, y: np.ndarray, timestamps: Optional[np.ndarray] = None) -> None:
        self._train_ts = timestamps

    def predict(self,
                history: np.ndarray,
                horizon: int,
                timestamps_future: Optional[np.ndarray] = None,
                quantiles: Optional[List[float]] = None) -> Dict[str, np.ndarray]:
        if self._train_ts is not None:
            train_ts = _coerce_ts(self._train_ts)
            hist_index = pd.to_datetime(train_ts[-len(history):])
            if timestamps_future is not None:
                fut_index = pd.to_datetime(_coerce_ts(timestamps_future))
            else:
                fut_index = pd.date_range(start=hist_index[-1] + pd.Timedelta(days=1), periods=horizon, freq="D")
        else:
            hist_index, fut_index = _make_dummy_times(len(history), horizon, freq="D")

        hist_df = pd.DataFrame({"y": history}, index=hist_index)
        fut_df  = pd.DataFrame({"y": np.zeros(horizon)}, index=fut_index)

        class _Task:
            def __init__(self, past, future):
                self.past_time = past
                self.future_time = future
                self.background = ""
                self.constraints = ""
                self.scenario = ""

        task = _Task(hist_df, fut_df)
        samples, _ = self.llmp(task, n_samples=2)  # smaller for speed
        preds = samples[:, :, 0]
        mean_pred = preds.mean(axis=0)
        median_pred = np.median(preds, axis=0)
        out = {'mean': mean_pred, 'median': median_pred}
        if quantiles is not None:
            out['quantiles'] = {q: np.quantile(preds, q, axis=0) for q in quantiles}
        return out
