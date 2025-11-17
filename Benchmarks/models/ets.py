"""
ETS (Exponential Smoothing) Model Wrapper

Implements Error-Trend-Seasonality models (also known as Holt-Winters)
for time series forecasting with trend and seasonality components.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Union
import warnings

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.statespace.tools import constrain_stationary_univariate
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    print("Warning: statsmodels not installed. Install with: pip install statsmodels")

from models.base_model import BaselineModel


class ETSModel(BaselineModel):
    """
    Exponential Smoothing (ETS) model wrapper for unified benchmarking.
    
    Also known as Holt-Winters method. Explicitly models:
    - Error (additive or multiplicative)
    - Trend (additive, multiplicative, or none)
    - Seasonality (additive, multiplicative, or none)
    
    Good for:
    - Data with clear trend and/or seasonal patterns
    - Medium-term forecasting
    - Interpretable components
    """
    
    def __init__(self,
                 seasonal_periods: int = 1,
                 trend: Optional[str] = 'add',
                 seasonal: Optional[str] = None,
                 damped_trend: bool = False,
                 initialization_method: str = 'estimated',
                 freq: str = 'D',
                 name: str = "ETS"):
        """
        Initialize ETS model.
        
        Args:
            seasonal_periods: Length of seasonal cycle (e.g., 7 for weekly, 12 for monthly, 24 for hourly)
            trend: Type of trend component
                   - 'add': Additive trend
                   - 'mul': Multiplicative trend
                   - None: No trend
            seasonal: Type of seasonal component
                     - 'add': Additive seasonality
                     - 'mul': Multiplicative seasonality
                     - None: No seasonality
            damped_trend: Whether to use damped trend (reduces trend over time)
            initialization_method: How to initialize components ('estimated', 'heuristic', 'known')
            freq: Pandas frequency string (not used by ETS but kept for consistency)
            name: Model name for identification
        """
        super().__init__(name)
        
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels is required for ETS. Install with: pip install statsmodels")
        
        self.seasonal_periods = seasonal_periods
        self.trend = trend
        self.seasonal = seasonal if seasonal_periods > 1 else None  # No seasonal if period=1
        self.damped_trend = damped_trend
        self.initialization_method = initialization_method
        self.freq = freq
        
        self.model = None
        self.fitted_model = None
        
        # Build model name for display
        trend_str = trend if trend else 'none'
        seasonal_str = seasonal if seasonal else 'none'
        damped_str = '-damped' if damped_trend else ''
        self.model_description = f"ETS({trend_str},{seasonal_str}{damped_str})"
    
    def fit(self,
            train_data: Union[np.ndarray, pd.DataFrame],
            timestamps: Optional[pd.DatetimeIndex] = None):
        """
        Fit ETS model to training data.
        
        Args:
            train_data: Historical time series data
            timestamps: Datetime index (optional, not used by ETS)
        """
        # Prepare data
        y = self._prepare_data(train_data)
        
        if len(y) < self.seasonal_periods * 2:
            warnings.warn(
                f"Training data length ({len(y)}) is less than 2 seasonal cycles "
                f"({self.seasonal_periods * 2}). Model may be unstable."
            )
        
        # Check for non-positive values if using multiplicative components
        if self.trend == 'mul' or self.seasonal == 'mul':
            if np.any(y <= 0):
                warnings.warn(
                    "Data contains non-positive values but multiplicative components specified. "
                    "Switching to additive components."
                )
                self.trend = 'add' if self.trend == 'mul' else self.trend
                self.seasonal = 'add' if self.seasonal == 'mul' else self.seasonal
        
        try:
            # Create and fit model
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore')  # Suppress convergence warnings
                
                self.model = ExponentialSmoothing(
                    y,
                    seasonal_periods=self.seasonal_periods if self.seasonal else None,
                    trend=self.trend,
                    seasonal=self.seasonal,
                    damped_trend=self.damped_trend,
                    initialization_method=self.initialization_method
                )
                
                self.fitted_model = self.model.fit(
                    optimized=True,
                    use_brute=False  # Faster, less thorough optimization
                )
            
            self.is_fitted = True
            
            # Extract and display fitted parameters
            if hasattr(self.fitted_model, 'params'):
                alpha = self.fitted_model.params.get('smoothing_level', None)
                beta = self.fitted_model.params.get('smoothing_trend', None)
                gamma = self.fitted_model.params.get('smoothing_seasonal', None)
                
                params_str = f"α={alpha:.3f}" if alpha else ""
                if beta:
                    params_str += f", β={beta:.3f}"
                if gamma:
                    params_str += f", γ={gamma:.3f}"
                
                print(f"Fitted {self.model_description} with {params_str}")
            else:
                print(f"Fitted {self.model_description}")
            
        except Exception as e:
            print(f"Warning: ETS fitting failed with error: {e}")
            print("Falling back to simple exponential smoothing (no trend/seasonal)")
            
            # Fallback: simple exponential smoothing
            self.model = ExponentialSmoothing(
                y,
                trend=None,
                seasonal=None
            )
            self.fitted_model = self.model.fit()
            self.is_fitted = True
        
        return self
    
    def predict(self,
                history: Union[np.ndarray, pd.DataFrame],
                horizon: int,
                context: Optional[str] = None,
                quantiles: Optional[List[float]] = None) -> Dict[str, np.ndarray]:
        """
        Make predictions using fitted ETS model.
        
        Args:
            history: Recent history (not used by ETS - uses internal state)
            horizon: Number of steps to forecast
            context: Optional textual description (IGNORED by ETS)
            quantiles: List of quantiles to predict (e.g., [0.1, 0.5, 0.9])
        
        Returns:
            Dictionary with 'mean', 'median', and optionally 'quantiles'
        
        Note:
            ETS is a statistical model and does not use textual context.
            The context parameter is ignored.
            
            Unlike ARIMA, ETS does not refit on new history - it uses
            the model state from the last fit() call.
        """
        # Context is ignored for statistical models
        if context is not None:
            pass  # Silently ignore
        
        if not self.is_fitted or self.fitted_model is None:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")
        
        # Make forecast
        try:
            forecast_result = self.fitted_model.forecast(steps=horizon)
            mean_pred = np.array(forecast_result)
            
            # Get prediction intervals if quantiles requested
            if quantiles is not None:
                # ETS can provide prediction intervals via simulation
                quantile_preds = {}
                
                # Generate prediction intervals
                # Note: statsmodels ETS doesn't have built-in quantile prediction
                # We'll use a simple approximation based on forecast variance
                
                # Get forecast with prediction intervals (default 95%)
                forecast_summary = self.fitted_model.summary_frame(alpha=0.05)
                
                if 'mean' in forecast_summary.columns:
                    forecast_mean = forecast_summary['mean'].values[:horizon]
                    
                    # Try to get prediction intervals
                    if 'pi_lower' in forecast_summary.columns and 'pi_upper' in forecast_summary.columns:
                        pi_lower = forecast_summary['pi_lower'].values[:horizon]
                        pi_upper = forecast_summary['pi_upper'].values[:horizon]
                        
                        # Approximate standard deviation from 95% interval
                        # 95% interval is approximately mean ± 1.96*std
                        std_approx = (pi_upper - pi_lower) / (2 * 1.96)
                        
                        # Generate quantiles
                        for q in quantiles:
                            if q == 0.5:
                                quantile_preds[0.5] = mean_pred
                            else:
                                # Use normal approximation
                                from scipy import stats
                                z = stats.norm.ppf(q)
                                quantile_preds[q] = mean_pred + z * std_approx
                    else:
                        # Fallback: use simple scaling
                        for q in quantiles:
                            quantile_preds[q] = mean_pred * (0.5 + (q - 0.5) * 0.3)
                
                return self._format_predictions(mean_pred, quantile_preds)
            
            return self._format_predictions(mean_pred)
            
        except Exception as e:
            print(f"Warning: Prediction failed with error: {e}")
            # Fallback: return last value repeated
            if hasattr(self.fitted_model, 'fittedvalues'):
                fittedvalues = self.fitted_model.fittedvalues
                # Handle both numpy array and pandas Series
                if isinstance(fittedvalues, np.ndarray):
                    last_value = fittedvalues[-1]
                else:
                    last_value = fittedvalues.iloc[-1]
            else:
                last_value = 0.0
            
            mean_pred = np.full(horizon, last_value)
            return self._format_predictions(mean_pred)
    
    def __repr__(self):
        status = "fitted" if self.is_fitted else "not fitted"
        return f"{self.name} - {self.model_description} ({status})"


# Example usage and testing
if __name__ == "__main__":
    print("ETS Model Wrapper - Testing")
    print("=" * 60)
    
    if not STATSMODELS_AVAILABLE:
        print("✗ statsmodels not available. Install with: pip install statsmodels")
    else:
        print("✓ statsmodels available")
        
        # Generate synthetic data with trend and seasonality
        np.random.seed(42)
        n = 200
        t = np.arange(n)
        
        # Components
        trend = 0.5 * t
        seasonal = 10 * np.sin(2 * np.pi * t / 12)  # 12-period seasonality
        noise = 2 * np.random.randn(n)
        y = 100 + trend + seasonal + noise
        
        # Split
        train_y = y[:150]
        test_y = y[150:]
        
        print(f"\nGenerated {n} points with trend and 12-period seasonality")
        print(f"Train: {len(train_y)}, Test: {len(test_y)}")
        
        # Test different configurations
        configs = [
            {'seasonal_periods': 12, 'trend': 'add', 'seasonal': 'add', 'name': 'ETS(A,A)'},
            {'seasonal_periods': 12, 'trend': 'add', 'seasonal': None, 'name': 'ETS(A,N)'},
            {'seasonal_periods': 1, 'trend': 'add', 'seasonal': None, 'name': 'ETS-NoSeasonal'},
        ]
        
        for config in configs:
            print(f"\n{'-'*60}")
            print(f"Testing: {config['name']}")
            print(f"{'-'*60}")
            
            model = ETSModel(**config)
            model.fit(train_y)
            
            # Predict
            preds = model.predict(train_y[-30:], horizon=10, quantiles=[0.1, 0.5, 0.9])
            
            # Compute error
            mae = np.mean(np.abs(test_y[:10] - preds['mean']))
            print(f"\nMAE on test: {mae:.4f}")
            print(f"First 5 predictions: {preds['mean'][:5]}")
            print(f"Quantiles available: {list(preds.get('quantiles', {}).keys())}")
        
        print("\n" + "="*60)
        print("✓ ETS testing complete!")