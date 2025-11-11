"""
Test ETS (Exponential Smoothing) Model
"""

from single_tests import print_test_header, generate_synthetic_data
from models.ets import ETSModel
from evaluation.metrics import mae, rmse
import numpy as np

print_test_header("Testing ETS Model")

# Generate data
y = generate_synthetic_data(n=200, seasonal_period=12, seed=42)

print(f"Generated {len(y)} points with 12-period seasonality")

# Split
train = y[:150]
test = y[150:160]

print(f"Train: {len(train)}, Test: {len(test)}")

# Test configurations
configs = [
    {'seasonal_periods': 12, 'trend': 'add', 'seasonal': 'add', 'name': 'ETS(A,A)'},
    {'seasonal_periods': 12, 'trend': 'add', 'seasonal': None, 'name': 'ETS(A,N)'},
    {'seasonal_periods': 1, 'trend': 'add', 'seasonal': None, 'name': 'ETS-NoSeasonal'},
]

results = []

for config in configs:
    print(f"\n{'-'*60}")
    print(f"Testing: {config['name']}")
    print(f"{'-'*60}")
    
    model = ETSModel(**config)
    model.fit(train)
    
    # Predict
    preds = model.predict(train[-30:], horizon=10, quantiles=[0.1, 0.5, 0.9])
    
    # Compute error
    mae_val = mae(test, preds['mean'])
    rmse_val = rmse(test, preds['mean'])
    
    print(f"\nMAE:  {mae_val:.4f}")
    print(f"RMSE: {rmse_val:.4f}")
    print(f"First 5 predictions: {preds['mean'][:5]}")
    
    results.append({
        'name': config['name'],
        'mae': mae_val,
        'rmse': rmse_val
    })

# Summary
print("\n" + "="*60)
print("Summary")
print("="*60)
print(f"{'Model':<20} {'MAE':<10} {'RMSE':<10}")
print("-"*60)
for r in results:
    print(f"{r['name']:<20} {r['mae']:<10.4f} {r['rmse']:<10.4f}")

best = min(results, key=lambda x: x['mae'])
print(f"\n✓ Best model: {best['name']} (MAE: {best['mae']:.4f})")

print("\n" + "="*60)
print("✓ ETS testing complete!")
print("="*60)