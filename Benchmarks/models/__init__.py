"""
Time Series Forecasting Models

Available models:
- AutoARIMAModel: Statistical baseline
- ETSModel: Exponential smoothing baseline
- PatchTSTModel: Deep learning baseline (optional)
"""

from .base_model import TimeSeriesModel, BaselineModel
from .autoarima import AutoARIMAModel
from .ets import ETSModel

# Optional imports
__all__ = [
    'TimeSeriesModel',
    'BaselineModel',
    'AutoARIMAModel',
    'ETSModel',
]

# Try to import PatchTST (optional dependency)
try:
    from .patchtst import PatchTSTModel
    __all__.append('PatchTSTModel')
except Exception:
    pass  # PatchTST not available, skip it