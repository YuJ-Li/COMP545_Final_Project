"""
Single Model Tests

This directory contains standalone tests for individual models.
Each test file can be run independently to verify a specific model works correctly.

Available tests:
- test_autoarima.py: Test AutoARIMA on synthetic data
- test_ets.py: Test ETS (Exponential Smoothing) on synthetic data
- test_patchtst.py: Test PatchTST (when implemented)
- test_chronos.py: Test Chronos (when implemented)
- test_llama3.py: Test Llama 3 (when implemented)

Usage:
    python single_tests/test_autoarima.py
    python single_tests/test_ets.py
"""

import sys
import os

# Add parent directory to path so tests can import from models/evaluation
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

# Test utilities (optional - can be used by test scripts)

def print_test_header(test_name):
    """Print a formatted header for test output"""
    print("\n" + "="*60)
    print(f"{test_name}")
    print("="*60 + "\n")


def print_test_result(passed, message=""):
    """Print test pass/fail result"""
    if passed:
        print(f"\n✓ Test PASSED {message}")
    else:
        print(f"\n✗ Test FAILED {message}")


def generate_synthetic_data(n=500, trend=True, seasonal=True, 
                           seasonal_period=50, noise_std=2.0, seed=42):
    """
    Generate synthetic time series for testing.
    
    Args:
        n: Number of data points
        trend: Include linear trend
        seasonal: Include seasonal component
        seasonal_period: Period of seasonality
        noise_std: Standard deviation of noise
        seed: Random seed for reproducibility
    
    Returns:
        numpy array of synthetic time series
    """
    import numpy as np
    
    np.random.seed(seed)
    t = np.arange(n)
    
    y = np.zeros(n) + 100  # Base level
    
    if trend:
        y += 0.5 * t
    
    if seasonal:
        y += 10 * np.sin(2 * np.pi * t / seasonal_period)
    
    y += noise_std * np.random.randn(n)
    
    return y


__all__ = [
    'print_test_header',
    'print_test_result', 
    'generate_synthetic_data'
]
