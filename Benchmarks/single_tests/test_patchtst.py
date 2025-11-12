"""
Test PatchTST Model
Matches AutoARIMA/ETS test format exactly
"""

import os
import sys
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.patchtst import PatchTSTModel
from evaluation.metrics import mae, rmse, mase
import numpy as np
import torch


print("="*60)
print("Testing PatchTST Model")
print("="*60)

# Generate synthetic data - MATCHING AutoARIMA/ETS test exactly
np.random.seed(42)
n = 500
t = np.arange(n)

# Components - MATCHING AutoARIMA/ETS
trend = 0.05 * t
seasonal = 10 * np.sin(2 * np.pi * t / 50)
noise = 2 * np.random.randn(n)
y = 100 + trend + seasonal + noise

print(f"\nGenerated {n} points with trend and seasonality")

# Split train/test - MATCHING AutoARIMA/ETS
train_size = 400
train_data = y[:train_size]
test_data = y[train_size:]

print(f"Train: {len(train_data)} points")
print(f"Test: {len(test_data)} points")

# Create and fit model
print("\n" + "-"*60)
print("Training PatchTST...")
print("-"*60)

# PatchTST parameters
# seq_len: how much history to look at (context window)
# pred_len: how far ahead to predict (must match our horizon)
# patch_len: size of each patch (smaller = more detail, larger = more efficiency)
# stride: overlap between patches

model = PatchTSTModel(
    seq_len=96,        # Use last 96 timesteps as input
    pred_len=100,      # Predict next 100 timesteps (matches our test set)
    patch_len=16,      # Each patch is 16 timesteps
    stride=8,          # 50% overlap between patches
    d_model=128,       # Model dimension
    n_heads=8,         # Number of attention heads
    e_layers=3,        # Number of transformer layers
    device='cuda' if torch.cuda.is_available() else 'cpu',
    name='PatchTST-Test'
)

print(f"Device: {model.device}")

# Train the model (THIS IS THE KEY DIFFERENCE FROM STATISTICAL MODELS)
model.fit(
    train_data, 
    epochs=30,           # Number of training epochs
    batch_size=16,       # Batch size for training
    learning_rate=0.001, # Learning rate
    patience=5,          # Early stopping patience
    verbose=True         # Show training progress
)

# Make predictions
print("\nMaking predictions...")
horizon = len(test_data)

# PatchTST uses the last seq_len points to predict
predictions = model.predict(
    history=train_data,  # Pass full training data
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

# Plot - MATCHING AutoARIMA/ETS style exactly
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
plt.title('PatchTST Test: Synthetic Data')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save plot
script_dir = os.path.dirname(os.path.abspath(__file__))
save_path = os.path.join(script_dir, 'patchtst_test.png')
plt.savefig(save_path, dpi=150)

print(f"\n✓ Plot saved as '{save_path}'")

print("\n" + "="*60)
print("✓ PatchTST test complete!")
print("="*60)
