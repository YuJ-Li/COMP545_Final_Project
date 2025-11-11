"""
Quick test of AutoARIMA model
"""

from single_tests import print_test_header, generate_synthetic_data
from models.autoarima import AutoARIMAModel
from evaluation.metrics import mae, rmse, mase
import matplotlib.pyplot as plt
import numpy as np

print_test_header("Testing AutoARIMA Model")

# Generate synthetic data using utility
y = generate_synthetic_data(n=500, seasonal_period=50, seed=42)

print(f"Generated {len(y)} points with trend and seasonality")

# Split train/test
train_size = 400
train_data = y[:train_size]
test_data = y[train_size:]

print(f"Train: {len(train_data)} points")
print(f"Test: {len(test_data)} points")

# Create and fit model
print("\n" + "-"*60)
print("Fitting AutoARIMA...")
print("-"*60)

model = AutoARIMAModel(
    season_length=50,
    freq='D',
    name='AutoARIMA-Test'
)

model.fit(train_data)

# Make predictions
print("\nMaking predictions...")
horizon = len(test_data)
predictions = model.predict(
    history=train_data[-100:],
    horizon=horizon
)

y_pred = predictions['mean']

# Compute metrics
print("\n" + "="*60)
print("Results")
print("="*60)

mae_val = mae(test_data, y_pred)
rmse_val = rmse(test_data, y_pred)
mase_val = mase(test_data, y_pred, train_data, seasonality=50)

print(f"MAE:  {mae_val:.4f}")
print(f"RMSE: {rmse_val:.4f}")
print(f"MASE: {mase_val:.4f}")

if mase_val < 1.0:
    print("\n✓ MASE < 1.0: Better than naive forecast!")
else:
    print(f"\n⚠ MASE = {mase_val:.2f}")

# Plot
plt.figure(figsize=(12, 6))
plot_start = max(0, train_size - 200)
plt.plot(range(plot_start, train_size), y[plot_start:train_size], 
         label='Training', color='blue', alpha=0.7)
plt.plot(range(train_size, train_size + horizon), test_data, 
         label='True', color='green', linewidth=2)
plt.plot(range(train_size, train_size + horizon), y_pred, 
         label='Predictions', color='red', linestyle='--', linewidth=2)
plt.axvline(x=train_size, color='black', linestyle=':', label='Split')
plt.xlabel('Time')
plt.ylabel('Value')
plt.title('AutoARIMA Test: Synthetic Data')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('single_tests/autoarima_test.png', dpi=150)
print("\n✓ Plot saved as 'single_tests/autoarima_test.png'")

print("\n" + "="*60)
print("✓ AutoARIMA test complete!")
print("="*60)