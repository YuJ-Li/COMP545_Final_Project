"""
Base Model Interface for Time Series Forecasting

All forecasting models (AutoARIMA, PatchTST, LLMs, foundation models) 
should inherit from this class to ensure consistent API.
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, List, Union
import numpy as np
import pandas as pd


class TimeSeriesModel(ABC):
    """
    Abstract base class for all time series forecasting models.
    
    This ensures all models have a consistent interface for:
    - Fitting on training data
    - Making point predictions
    - Making probabilistic predictions (quantiles)
    """
    
    def __init__(self, name: str):
        self.name = name
        self.is_fitted = False
    
    @abstractmethod
    def fit(self, 
            train_data: Union[np.ndarray, pd.DataFrame],
            timestamps: Optional[pd.DatetimeIndex] = None):
        """
        Fit the model on training data.
        
        Args:
            train_data: Historical time series data
                - np.ndarray: shape (n_timesteps,) or (n_timesteps, n_features)
                - pd.DataFrame: with 'ds' (datetime) and 'y' (target) columns
            timestamps: Optional datetime index (if train_data is np.ndarray)
        """
        pass
    
    @abstractmethod
    def predict(self, 
                history: Union[np.ndarray, pd.DataFrame],
                horizon: int,
                quantiles: Optional[List[float]] = None) -> Dict[str, np.ndarray]:
        """
        Make predictions for the next 'horizon' timesteps.
        
        Args:
            history: Recent history to condition on
                - np.ndarray: shape (context_length,)
                - pd.DataFrame: with 'ds' and 'y' columns
            horizon: Number of steps to forecast ahead
            quantiles: List of quantiles to predict (e.g., [0.1, 0.5, 0.9])
                      If None, only return point forecast
        
        Returns:
            Dictionary with:
                'mean': np.ndarray of shape (horizon,) - point forecast
                'median': np.ndarray of shape (horizon,) - median forecast (if available)
                'quantiles': Dict[float, np.ndarray] - quantile forecasts
                              e.g., {0.1: array([...]), 0.9: array([...])}
        """
        pass
    
    def predict_rolling(self,
                       data: Union[np.ndarray, pd.DataFrame],
                       context_length: int,
                       horizon: int,
                       step: int = 1,
                       quantiles: Optional[List[float]] = None) -> List[Dict]:
        """
        Make rolling window predictions across a dataset.
        
        Args:
            data: Full time series
            context_length: How much history to use for each prediction
            horizon: Forecast horizon
            step: Stride between rolling windows
            quantiles: Quantiles to predict
            
        Returns:
            List of prediction dictionaries, each containing:
                - 'history': the input history
                - 'future': the true future values
                - 'predictions': output from predict()
                - 'start_idx': where this window starts
        """
        # Convert to numpy if needed
        if isinstance(data, pd.DataFrame):
            y = data['y'].values
        else:
            y = data
        
        results = []
        max_start = len(y) - context_length - horizon
        
        for start_idx in range(0, max_start + 1, step):
            end_idx = start_idx + context_length
            history = y[start_idx:end_idx]
            future = y[end_idx:end_idx + horizon]
            
            # Make prediction
            preds = self.predict(history, horizon, quantiles)
            
            results.append({
                'history': history,
                'future': future,
                'predictions': preds,
                'start_idx': start_idx,
                'end_idx': end_idx
            })
        
        return results
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"


class BaselineModel(TimeSeriesModel):
    """
    Base class for traditional statistical/ML models.
    Handles common preprocessing and data formatting.
    """
    
    def _prepare_data(self, data: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """Convert input data to numpy array."""
        if isinstance(data, pd.DataFrame):
            return data['y'].values
        return data
    
    def _format_predictions(self,
                          mean_pred: np.ndarray,
                          quantile_preds: Optional[Dict[float, np.ndarray]] = None) -> Dict[str, np.ndarray]:
        """
        Format predictions into standard output dictionary.
        
        Args:
            mean_pred: Point forecast
            quantile_preds: Optional quantile forecasts
            
        Returns:
            Standardized prediction dictionary
        """
        result = {
            'mean': mean_pred,
            'median': quantile_preds.get(0.5, mean_pred) if quantile_preds else mean_pred,
        }
        
        if quantile_preds is not None:
            result['quantiles'] = quantile_preds
        
        return result
