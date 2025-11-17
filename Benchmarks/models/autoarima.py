"""
AutoARIMA Model Wrapper for Unified Benchmarking

Uses statsforecast's AutoARIMA for automatic parameter selection
and ARIMA for efficient forecasting.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Union

from statsforecast import StatsForecast
from statsforecast.models import AutoARIMA, ARIMA
from statsforecast.arima import arima_string

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from base_model import BaselineModel


class AutoARIMAModel(BaselineModel):
    """
    AutoARIMA model wrapper for unified benchmarking.
    
    Uses statsforecast's AutoARIMA for automatic parameter selection
    and ARIMA for efficient forecasting.
    """
    
    def __init__(self, 
                 season_length: int = 1,
                 freq: str = 'D',
                 name: str = "AutoARIMA"):
        """
        Args:
            season_length: Seasonal period (e.g., 7 for weekly, 12 for monthly, 24 for hourly)
            freq: Pandas frequency string (e.g., 'D', 'H', 'M')
            name: Model name for identification
        """
        super().__init__(name)
        self.season_length = season_length
        self.freq = freq
        self.fitted_params = None
        self.model = None
        self.sf = None
    
    def fit(self, 
            train_data: Union[np.ndarray, pd.DataFrame],
            timestamps: Optional[pd.DatetimeIndex] = None):
        """
        Fit AutoARIMA to find optimal parameters.
        
        Args:
            train_data: Training time series
            timestamps: Datetime index (required if train_data is np.ndarray)
        """
        # Prepare data
        if isinstance(train_data, np.ndarray):
            if timestamps is None:
                # Create dummy timestamps if none provided
                timestamps = pd.date_range(start='2020-01-01', periods=len(train_data), freq=self.freq)
            df = pd.DataFrame({
                'ds': timestamps,
                'y': train_data,
                'unique_id': 'series_1'
            })
        elif isinstance(train_data, pd.DataFrame):
            df = train_data.copy()
            if 'unique_id' not in df.columns:
                df['unique_id'] = 'series_1'
        else:
            raise ValueError("train_data must be np.ndarray or pd.DataFrame")
        
        # Fit AutoARIMA to find best parameters
        auto_model = AutoARIMA(season_length=self.season_length)
        sf_auto = StatsForecast(
            models=[auto_model],
            freq=self.freq,
            n_jobs=1
        )
        sf_auto.fit(df=df)
        
        # Extract fitted parameters
        arima_params = arima_string(sf_auto.fitted_[0, 0].model_)
        
        # Parse ARIMA parameters
        open_1 = arima_params.find("(")
        close_1 = arima_params.find(")")
        (p, d, q) = [int(i) for i in arima_params[open_1+1:close_1].split(",")]
        
        # Check for seasonal parameters
        if arima_params.find("(", open_1+1) == -1:
            P, D, Q = 0, 0, 0
        else:
            seasonal_start = arima_params.find("(", open_1+1)
            seasonal_end = arima_params.find(")", close_1+1)
            (P, D, Q) = [int(i) for i in arima_params[seasonal_start+1:seasonal_end].split(",")]
        
        # Store parameters
        self.fitted_params = {
            'order': (p, d, q),
            'seasonal_order': (P, D, Q),
            'season_length': self.season_length
        }
        
        # Create ARIMA model with fitted parameters
        self.model = ARIMA(
            order=(p, d, q),
            season_length=self.season_length,
            seasonal_order=(P, D, Q)
        )
        
        self.sf = StatsForecast(
            models=[self.model],
            freq=self.freq,
            n_jobs=1
        )
        
        # Fit the ARIMA model
        self.sf.fit(df=df)
        self.is_fitted = True
        
        print(f"Fitted ARIMA{(p,d,q)} x {(P,D,Q)}[{self.season_length}]")
        
        return self
    
    def predict(self, 
                history: Union[np.ndarray, pd.DataFrame],
                horizon: int,
                context: Optional[str] = None,  # ← ADDED THIS
                quantiles: Optional[List[float]] = None) -> Dict[str, np.ndarray]:
        """
        Make predictions using fitted ARIMA model.
        
        Args:
            history: Recent history to condition on
            horizon: Number of steps to forecast
            context: Optional textual description (IGNORED by ARIMA)
            quantiles: List of quantiles to predict (e.g., [0.1, 0.5, 0.9])
        
        Returns:
            Dictionary with 'mean', 'median', and 'quantiles' (if requested)
        
        Note:
            ARIMA is a statistical model and does not use textual context.
            The context parameter is ignored.
        """
        # Context is ignored for statistical models
        if context is not None:
            pass  # Silently ignore context for traditional models
        
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")
        
        # Prepare history data
        if isinstance(history, np.ndarray):
            timestamps = pd.date_range(start='2020-01-01', periods=len(history), freq=self.freq)
            df = pd.DataFrame({
                'ds': timestamps,
                'y': history,
                'unique_id': 'series_1'
            })
        else:
            df = history.copy()
            if 'unique_id' not in df.columns:
                df['unique_id'] = 'series_1'
        
        # Make forecast
        if quantiles is None:
            # Point forecast only
            forecast_df = self.sf.forecast(df=df, h=horizon)
            mean_pred = forecast_df['ARIMA'].values
            
            return self._format_predictions(mean_pred)
        else:
            # Include quantile predictions
            # Convert quantiles to confidence levels for statsforecast
            # quantiles [0.1, 0.5, 0.9] -> levels [50, 80]
            levels = []
            for q in quantiles:
                if q > 0.5:
                    level = int(200 * (q - 0.5))
                    if level not in levels:
                        levels.append(level)
            
            # Always include median (level=0)
            if 0.5 in quantiles and 0 not in levels:
                levels.append(0)
            
            forecast_df = self.sf.forecast(df=df, h=horizon, level=levels)
            
            # Extract predictions
            mean_pred = forecast_df['ARIMA'].values
            
            # Extract quantiles
            quantile_preds = {}
            for q in quantiles:
                if q == 0.5:
                    # Median is the mean for ARIMA
                    quantile_preds[0.5] = mean_pred
                elif q > 0.5:
                    # Upper quantile
                    level = int(200 * (q - 0.5))
                    col_name = f'ARIMA-hi-{level}'
                    if col_name in forecast_df.columns:
                        quantile_preds[q] = forecast_df[col_name].values
                else:
                    # Lower quantile (mirror from upper)
                    upper_q = 1.0 - q
                    level = int(200 * (upper_q - 0.5))
                    col_name = f'ARIMA-lo-{level}'
                    if col_name in forecast_df.columns:
                        quantile_preds[q] = forecast_df[col_name].values
            
            return self._format_predictions(mean_pred, quantile_preds)
    
    def __repr__(self):
        params_str = ""
        if self.fitted_params:
            p, d, q = self.fitted_params['order']
            P, D, Q = self.fitted_params['seasonal_order']
            params_str = f" ARIMA({p},{d},{q})x({P},{D},{Q})[{self.season_length}]"
        return f"{self.name}{params_str}"


# Example usage and testing
if __name__ == "__main__":
    # Generate synthetic data
    np.random.seed(42)
    t = np.arange(200)
    y = 10 + 0.5 * t + 5 * np.sin(2 * np.pi * t / 12) + np.random.randn(200)
    
    # Split train/test
    train_y = y[:150]
    test_y = y[150:]
    
    # Create and fit model
    model = AutoARIMAModel(season_length=12, freq='M')
    model.fit(train_y)
    
    # Make predictions WITHOUT context (traditional usage)
    history = train_y[-30:]
    preds = model.predict(history, horizon=10, quantiles=[0.1, 0.5, 0.9])
    
    print("\nPredictions without context:")
    print(f"Mean: {preds['mean'][:5]}")
    print(f"Median: {preds['median'][:5]}")
    print(f"Q10: {preds['quantiles'][0.1][:5]}")
    print(f"Q90: {preds['quantiles'][0.9][:5]}")
    
    # Make predictions WITH context (should produce same results)
    preds_with_context = model.predict(
        history, 
        horizon=10, 
        context="This is monthly data with seasonal pattern",  # Ignored!
        quantiles=[0.1, 0.5, 0.9]
    )
    
    print("\nPredictions with context (should be identical):")
    print(f"Mean: {preds_with_context['mean'][:5]}")
    print(f"Predictions match: {np.allclose(preds['mean'], preds_with_context['mean'])}")
    
    # Test rolling predictions
    print("\n\nRolling window predictions:")
    results = model.predict_rolling(y, context_length=30, horizon=5, step=10)
    print(f"Generated {len(results)} rolling windows")
    print(f"First window mean prediction: {results[0]['predictions']['mean']}")