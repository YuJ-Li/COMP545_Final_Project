"""
Evaluation Metrics for Time Series Forecasting

Includes:
1. Point forecast accuracy metrics (MAE, RMSE, MASE, etc.)
2. Calibration metrics from "Calibration Properties of Time Series Foundation Models" paper
   - PCE (Probabilistic Calibration Error)
   - SIW (Scaled Interval Width)
   - CCE (Centered Calibration Error)
3. Business-oriented metrics
"""

import numpy as np
from typing import Dict, List, Optional


# ============================================================================
# POINT FORECAST ACCURACY METRICS
# ============================================================================

def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Absolute Error
    
    Lower is better. Scale-dependent.
    """
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Error
    
    Lower is better. Penalizes large errors more than MAE.
    Scale-dependent.
    """
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Mean Squared Error
    
    Lower is better. Scale-dependent.
    """
    return float(np.mean((y_true - y_pred) ** 2))


def mape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """
    Mean Absolute Percentage Error
    
    Lower is better. Scale-independent.
    Note: Undefined when y_true contains zeros. Use sMAPE instead.
    """
    return float(100.0 * np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))))


def smape(y_true: np.ndarray, y_pred: np.ndarray, epsilon: float = 1e-8) -> float:
    """
    Symmetric Mean Absolute Percentage Error
    
    Lower is better. Scale-independent. Range: [0, 200]
    More robust than MAPE when y_true is near zero.
    """
    numerator = np.abs(y_true - y_pred)
    denominator = (np.abs(y_true) + np.abs(y_pred)) / 2.0 + epsilon
    return float(100.0 * np.mean(numerator / denominator))


def mase(y_true: np.ndarray, 
         y_pred: np.ndarray, 
         y_train: np.ndarray,
         seasonality: int = 1) -> float:
    """
    Mean Absolute Scaled Error
    
    Lower is better. Scale-independent.
    MASE < 1: Better than naive forecast
    MASE = 1: Same as naive forecast
    MASE > 1: Worse than naive forecast
    
    Args:
        y_true: True values
        y_pred: Predicted values
        y_train: Training data (used to compute scaling factor)
        seasonality: Seasonal period (1 for non-seasonal data)
    """
    # Calculate scaling factor from naive forecast on training data
    naive_mae = np.mean(np.abs(y_train[seasonality:] - y_train[:-seasonality]))
    
    if naive_mae == 0:
        # Avoid division by zero
        return float('inf') if np.any(y_true != y_pred) else 0.0
    
    mae_value = mae(y_true, y_pred)
    return float(mae_value / naive_mae)


# ============================================================================
# CALIBRATION METRICS (from the paper)
# ============================================================================

def probabilistic_calibration_error(
    y_true: np.ndarray,
    quantile_predictions: Dict[float, np.ndarray],
    p: int = 1
) -> float:
    """
    Probabilistic Calibration Error (PCE)
    
    Measures the difference between empirical and predicted CDFs.
    Lower is better.
    
    Range: [0, 0.5]
    - < 0.05: Well-calibrated
    - 0.15-0.2: Poorly calibrated
    
    From: "Calibration Properties of Time Series Foundation Models"
    
    Args:
        y_true: True values, shape (H,) where H is forecast horizon
        quantile_predictions: Dict mapping quantile (e.g., 0.1, 0.5, 0.9) 
                             to predictions of shape (H,)
        p: Norm parameter (1 for L1 norm, 2 for L2 norm)
    
    Returns:
        PCE value
    """
    quantiles = sorted(quantile_predictions.keys())
    H = len(y_true)
    
    errors = []
    for q in quantiles:
        y_pred_q = quantile_predictions[q]
        
        # Empirical probability: fraction of true values <= predicted quantile
        empirical_prob = np.mean(y_true <= y_pred_q)
        
        # Error: |expected quantile - empirical probability|
        error = np.abs(q - empirical_prob)
        errors.append(error)
    
    # Average over all quantiles
    pce = np.mean(errors) ** p
    return float(pce)


def scaled_interval_width(
    quantile_predictions: Dict[float, np.ndarray],
    y_train: np.ndarray,
    confidence_levels: Optional[List[float]] = None
) -> float:
    """
    Scaled Interval Width (SIW)
    
    Measures the sharpness/confidence of predictions.
    Lower = more confident (narrower intervals)
    Higher = less confident (wider intervals)
    
    From: "Calibration Properties of Time Series Foundation Models"
    
    Args:
        quantile_predictions: Dict of quantile predictions
        y_train: Training data (used for scaling)
        confidence_levels: List of confidence levels to evaluate (e.g., [0.8, 0.9])
                          If None, uses [0.2, 0.4, 0.6, 0.8]
    
    Returns:
        Average SIW across confidence levels
    """
    if confidence_levels is None:
        confidence_levels = [0.2, 0.4, 0.6, 0.8]
    
    quantiles = sorted(quantile_predictions.keys())
    H = len(next(iter(quantile_predictions.values())))
    
    # Compute interval width on training data for scaling
    train_std = np.std(y_train)
    if train_std == 0:
        train_std = 1.0  # Avoid division by zero
    
    siw_values = []
    for confidence in confidence_levels:
        # Determine lower and upper quantiles for this confidence level
        q_low = (1 - confidence) / 2
        q_high = 1 - q_low
        
        # Find closest quantiles in predictions
        q_low_actual = min(quantiles, key=lambda q: abs(q - q_low))
        q_high_actual = min(quantiles, key=lambda q: abs(q - q_high))
        
        if q_low_actual in quantile_predictions and q_high_actual in quantile_predictions:
            pred_low = quantile_predictions[q_low_actual]
            pred_high = quantile_predictions[q_high_actual]
            
            # Average interval width across horizon
            interval_width = np.mean(pred_high - pred_low)
            
            # Scale by training data spread
            scaled_width = interval_width / train_std
            siw_values.append(scaled_width)
    
    return float(np.mean(siw_values)) if siw_values else float('nan')


def centered_calibration_error(
    y_true: np.ndarray,
    quantile_predictions: Dict[float, np.ndarray],
    confidence_levels: Optional[List[float]] = None
) -> float:
    """
    Centered Calibration Error (CCE)
    
    Measures systematic over/under-confidence.
    
    Interpretation:
    - Positive CCE + Low SIW = Overconfident (intervals too narrow)
    - Negative CCE + High SIW = Underconfident (intervals too wide)
    - Near zero = Well-calibrated
    
    From: "Calibration Properties of Time Series Foundation Models"
    
    Args:
        y_true: True values, shape (H,)
        quantile_predictions: Dict of quantile predictions
        confidence_levels: Confidence levels to evaluate
    
    Returns:
        Average CCE across confidence levels
    """
    if confidence_levels is None:
        confidence_levels = [0.2, 0.4, 0.6, 0.8]
    
    quantiles = sorted(quantile_predictions.keys())
    H = len(y_true)
    
    cce_values = []
    for confidence in confidence_levels:
        # Determine interval bounds
        q_low = (1 - confidence) / 2
        q_high = 1 - q_low
        
        # Find closest quantiles
        q_low_actual = min(quantiles, key=lambda q: abs(q - q_low))
        q_high_actual = min(quantiles, key=lambda q: abs(q - q_high))
        
        if q_low_actual in quantile_predictions and q_high_actual in quantile_predictions:
            pred_low = quantile_predictions[q_low_actual]
            pred_high = quantile_predictions[q_high_actual]
            
            # Fraction of true values inside predicted interval
            in_interval = np.mean((y_true >= pred_low) & (y_true <= pred_high))
            
            # CCE: expected coverage - actual coverage
            cce = confidence - in_interval
            cce_values.append(cce)
    
    return float(np.mean(cce_values)) if cce_values else float('nan')


# ============================================================================
# BUSINESS-ORIENTED METRICS
# ============================================================================

def directional_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Directional Accuracy
    
    Measures how often the model correctly predicts the direction of change.
    Higher is better. Range: [0, 1]
    
    Critical for trading and decision-making applications.
    """
    if len(y_true) <= 1:
        return float('nan')
    
    # Compute direction of change
    true_direction = np.sign(y_true[1:] - y_true[:-1])
    pred_direction = np.sign(y_pred[1:] - y_pred[:-1])
    
    # Fraction of correct direction predictions
    correct = np.sum(true_direction == pred_direction)
    total = len(true_direction)
    
    return float(correct / total) if total > 0 else float('nan')


def coverage_rate(
    y_true: np.ndarray,
    quantile_predictions: Dict[float, np.ndarray],
    confidence: float = 0.9
) -> float:
    """
    Coverage Rate
    
    Measures how often true values fall within predicted confidence intervals.
    Should be close to the confidence level for well-calibrated models.
    
    Args:
        y_true: True values
        quantile_predictions: Quantile predictions
        confidence: Target confidence level (e.g., 0.9 for 90% intervals)
    
    Returns:
        Actual coverage rate
    """
    q_low = (1 - confidence) / 2
    q_high = 1 - q_low
    
    quantiles = sorted(quantile_predictions.keys())
    q_low_actual = min(quantiles, key=lambda q: abs(q - q_low))
    q_high_actual = min(quantiles, key=lambda q: abs(q - q_high))
    
    if q_low_actual not in quantile_predictions or q_high_actual not in quantile_predictions:
        return float('nan')
    
    pred_low = quantile_predictions[q_low_actual]
    pred_high = quantile_predictions[q_high_actual]
    
    coverage = np.mean((y_true >= pred_low) & (y_true <= pred_high))
    return float(coverage)


# ============================================================================
# CONVENIENCE FUNCTION
# ============================================================================

def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_train: Optional[np.ndarray] = None,
    quantile_predictions: Optional[Dict[float, np.ndarray]] = None,
    seasonality: int = 1
) -> Dict[str, float]:
    """
    Compute all available metrics for a set of predictions.
    
    Args:
        y_true: True values
        y_pred: Point predictions (mean/median)
        y_train: Training data (needed for MASE, SIW)
        quantile_predictions: Optional quantile predictions for calibration metrics
        seasonality: Seasonal period for MASE
    
    Returns:
        Dictionary of metric name -> value
    """
    metrics = {
        'MAE': mae(y_true, y_pred),
        'RMSE': rmse(y_true, y_pred),
        'MSE': mse(y_true, y_pred),
        'sMAPE': smape(y_true, y_pred),
        'Directional_Accuracy': directional_accuracy(y_true, y_pred),
    }
    
    # Add MASE if training data available
    if y_train is not None:
        try:
            metrics['MASE'] = mase(y_true, y_pred, y_train, seasonality)
        except:
            metrics['MASE'] = float('nan')
    
    # Add calibration metrics if quantile predictions available
    if quantile_predictions is not None and len(quantile_predictions) > 0:
        try:
            metrics['PCE'] = probabilistic_calibration_error(y_true, quantile_predictions)
        except:
            metrics['PCE'] = float('nan')
        
        if y_train is not None:
            try:
                metrics['SIW'] = scaled_interval_width(quantile_predictions, y_train)
            except:
                metrics['SIW'] = float('nan')
        
        try:
            metrics['CCE'] = centered_calibration_error(y_true, quantile_predictions)
        except:
            metrics['CCE'] = float('nan')
        
        try:
            metrics['Coverage_90'] = coverage_rate(y_true, quantile_predictions, confidence=0.9)
        except:
            metrics['Coverage_90'] = float('nan')
    
    return metrics


# Example usage
if __name__ == "__main__":
    # Generate synthetic data
    np.random.seed(42)
    n = 100
    y_train = np.sin(np.linspace(0, 4*np.pi, 200)) + 0.1 * np.random.randn(200)
    y_true = np.sin(np.linspace(4*np.pi, 5*np.pi, n)) + 0.1 * np.random.randn(n)
    y_pred = np.sin(np.linspace(4*np.pi, 5*np.pi, n)) + 0.15 * np.random.randn(n)
    
    # Simulate quantile predictions
    quantile_preds = {
        0.1: y_pred - 0.3,
        0.5: y_pred,
        0.9: y_pred + 0.3
    }
    
    # Compute all metrics
    metrics = compute_all_metrics(
        y_true=y_true,
        y_pred=y_pred,
        y_train=y_train,
        quantile_predictions=quantile_preds
    )
    
    print("Metric Results:")
    print("=" * 50)
    for name, value in metrics.items():
        print(f"{name:25s}: {value:.4f}")
