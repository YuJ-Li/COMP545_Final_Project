"""
PatchTST Wrapper - Official Repo Integration

Integrates with PatchTST from: https://github.com/yuqinie98/PatchTST
Uses: Benchmarks/patchtst/PatchTST_supervised/

This wrapper uses the official PatchTST model and training pipeline.
"""

import numpy as np
import pandas as pd
from typing import Optional, List, Dict, Union
import torch
import sys
import os
from pathlib import Path
import argparse

# Add PatchTST to path
PATCHTST_PATH = Path(__file__).parent / 'patchtst' / 'PatchTST_supervised'
sys.path.insert(0, str(PATCHTST_PATH))

try:
    from models.PatchTST import Model as PatchTSTModel_Official
    from layers.PatchTST_backbone import PatchTST_backbone
    PATCHTST_AVAILABLE = True
except ImportError:
    print("Warning: PatchTST model not found. Using placeholder.")
    PATCHTST_AVAILABLE = False

from base_model import BaselineModel


class PatchTSTModel(BaselineModel):
    """
    PatchTST model wrapper using official implementation.
    
    Based on: "A Time Series is Worth 64 Words: Long-term Forecasting with Transformers"
    Paper: https://arxiv.org/abs/2211.14730
    Code: https://github.com/yuqinie98/PatchTST
    """
    
    def __init__(self,
                 seq_len: int = 336,           # Input sequence length
                 pred_len: int = 96,           # Prediction length
                 patch_len: int = 16,          # Patch length
                 stride: int = 8,              # Patch stride
                 d_model: int = 128,           # Model dimension
                 n_heads: int = 16,            # Number of heads
                 e_layers: int = 3,            # Number of encoder layers
                 d_ff: int = 256,              # Dimension of fcn
                 dropout: float = 0.2,
                 fc_dropout: float = 0.2,
                 head_dropout: float = 0.0,
                 individual: bool = False,     # Individual head; True for multivariate
                 revin: bool = True,           # RevIN normalization
                 affine: bool = False,         # RevIN-affine
                 subtract_last: bool = False,  # Subtract last value
                 decomposition: bool = False,  # Decomposition
                 kernel_size: int = 25,        # Decomposition kernel size
                 device: str = 'cpu',
                 name: str = "PatchTST"):
        """
        Initialize PatchTST model with official implementation.
        
        Args:
            seq_len: Length of input sequence
            pred_len: Length of prediction
            patch_len: Length of each patch (subseries)
            stride: Stride for patching
            d_model: Dimension of model
            n_heads: Number of attention heads
            e_layers: Number of encoder layers
            d_ff: Dimension of feedforward network
            dropout: Dropout rate
            fc_dropout: Fully connected layer dropout
            head_dropout: Prediction head dropout
            individual: Use individual heads for each variate
            revin: Use Reversible Instance Normalization
            affine: Use affine transformation in RevIN
            subtract_last: Subtract last value (alternative normalization)
            decomposition: Use series decomposition
            kernel_size: Kernel size for decomposition
            device: 'cpu' or 'cuda'
        """
        super().__init__(name)
        
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.n_heads = n_heads
        self.e_layers = e_layers
        self.d_ff = d_ff
        self.dropout = dropout
        self.fc_dropout = fc_dropout
        self.head_dropout = head_dropout
        self.individual = individual
        self.revin = revin
        self.affine = affine
        self.subtract_last = subtract_last
        self.decomposition = decomposition
        self.kernel_size = kernel_size
        self.device = device
        
        self.model = None
        self.enc_in = None  # Will be set during fit
        
        if not PATCHTST_AVAILABLE:
            print(f"[{self.name}] WARNING: PatchTST not available, using placeholder")
    
    def _create_configs(self, enc_in: int):
        """Create configuration namespace for PatchTST model."""
        configs = argparse.Namespace(
            # Task
            task_name='long_term_forecast',
            is_training=1,
            
            # Data
            enc_in=enc_in,
            dec_in=enc_in,
            c_out=enc_in,
            
            # Model architecture
            seq_len=self.seq_len,
            label_len=0,  # Not used for forecasting
            pred_len=self.pred_len,
            
            # PatchTST specifics
            patch_len=self.patch_len,
            stride=self.stride,
            
            # Encoder
            e_layers=self.e_layers,
            d_model=self.d_model,
            n_heads=self.n_heads,
            d_ff=self.d_ff,
            
            # Dropout
            dropout=self.dropout,
            fc_dropout=self.fc_dropout,
            head_dropout=self.head_dropout,
            
            # RevIN
            revin=self.revin,
            affine=self.affine,
            subtract_last=self.subtract_last,
            
            # Decomposition
            decomposition=self.decomposition,
            kernel_size=self.kernel_size,
            
            # Other
            individual=self.individual,
            padding_patch='end',  # Can be 'end' or None
            
            # Not used but required by model
            output_attention=False,
            embed='timeF',
            freq='h',
        )
        return configs
    
    def fit(self, 
            train_data: Union[np.ndarray, pd.DataFrame],
            timestamps: Optional[pd.DatetimeIndex] = None,
            epochs: int = 10,
            batch_size: int = 32,
            learning_rate: float = 0.0001,
            patience: int = 3,
            verbose: bool = True):
        """
        Train PatchTST model.
        
        Args:
            train_data: Training time series
            timestamps: Not used (PatchTST doesn't need timestamps)
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            patience: Early stopping patience
            verbose: Print training progress
        """
        # Prepare data
        y = self._prepare_data(train_data)
        
        # Ensure 2D: (n_timesteps, n_features)
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        n_timesteps, n_features = y.shape
        self.enc_in = n_features
        
        if verbose:
            print(f"[{self.name}] Training data: {n_timesteps} timesteps, {n_features} features")
        
        if not PATCHTST_AVAILABLE:
            print(f"[{self.name}] PatchTST not available - skipping training")
            self.is_fitted = True
            return self
        
        # Create model
        configs = self._create_configs(n_features)
        self.model = PatchTSTModel_Official(configs).to(self.device)
        
        if verbose:
            n_params = sum(p.numel() for p in self.model.parameters())
            print(f"[{self.name}] Model created with {n_params:,} parameters")
            print(f"[{self.name}] Config: seq_len={self.seq_len}, pred_len={self.pred_len}, "
                  f"patch_len={self.patch_len}")
        
        # Prepare training data (create sliding windows)
        X_train = []
        y_train = []
        
        for i in range(len(y) - self.seq_len - self.pred_len + 1):
            X_train.append(y[i:i + self.seq_len])
            y_train.append(y[i + self.seq_len:i + self.seq_len + self.pred_len])
        
        X_train = np.array(X_train)  # (n_samples, seq_len, n_features)
        y_train = np.array(y_train)  # (n_samples, pred_len, n_features)
        
        if verbose:
            print(f"[{self.name}] Created {len(X_train)} training samples")
        
        # Convert to PyTorch tensors
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.FloatTensor(y_train).to(self.device)
        
        # Training setup
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training loop
        self.model.train()
        best_loss = float('inf')
        patience_counter = 0
        
        n_batches = len(X_train) // batch_size
        
        for epoch in range(epochs):
            epoch_loss = 0
            
            # Shuffle data each epoch
            perm = torch.randperm(len(X_train))
            X_shuffled = X_train[perm]
            y_shuffled = y_train[perm]
            
            for i in range(0, len(X_train), batch_size):
                batch_x = X_shuffled[i:i + batch_size]
                batch_y = y_shuffled[i:i + batch_size]
                
                optimizer.zero_grad()
                
                # Forward pass
                # PatchTST expects: (batch, seq_len, n_features)
                outputs = self.model(batch_x)  # Returns: (batch, pred_len, n_features)
                
                loss = criterion(outputs, batch_y)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            avg_loss = epoch_loss / n_batches
            
            if verbose and (epoch + 1) % 2 == 0:
                print(f"[{self.name}] Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    if verbose:
                        print(f"[{self.name}] Early stopping at epoch {epoch+1}")
                    break
        
        self.is_fitted = True
        if verbose:
            print(f"[{self.name}] Training complete! Best loss: {best_loss:.6f}")
        
        return self
    
    def predict(self, 
                history: Union[np.ndarray, pd.DataFrame],
                horizon: int,
                quantiles: Optional[List[float]] = None) -> Dict[str, np.ndarray]:
        """
        Make predictions using trained PatchTST.
        
        Args:
            history: Input sequence (will use last seq_len points)
            horizon: Number of steps to predict (will be truncated/padded to pred_len)
            quantiles: Not supported by PatchTST (returns point forecasts only)
        
        Returns:
            Dictionary with 'mean' predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction. Call fit() first.")
        
        # Prepare history
        y = self._prepare_data(history)
        
        if y.ndim == 1:
            y = y.reshape(-1, 1)
        
        # Take last seq_len points
        if len(y) >= self.seq_len:
            y_input = y[-self.seq_len:]
        else:
            # Pad if needed
            pad_length = self.seq_len - len(y)
            y_input = np.vstack([np.tile(y[0], (pad_length, 1)), y])
        
        if not PATCHTST_AVAILABLE or self.model is None:
            # Fallback: simple extrapolation
            if quantiles is not None:
                print(f"[{self.name}] Warning: Quantiles not supported, returning point forecast")
            
            # Linear extrapolation
            last_val = y_input[-1, 0]
            trend = (y_input[-1, 0] - y_input[0, 0]) / len(y_input)
            preds = last_val + trend * np.arange(1, horizon + 1)
            
            return self._format_predictions(preds)
        
        # Actual prediction
        self.model.eval()
        with torch.no_grad():
            # Add batch dimension: (1, seq_len, n_features)
            x = torch.FloatTensor(y_input).unsqueeze(0).to(self.device)
            
            # Forward pass
            output = self.model(x)  # Shape: (1, pred_len, n_features)
            
            # Extract predictions
            preds = output.squeeze(0).cpu().numpy()  # (pred_len, n_features)
            
            # If univariate, squeeze feature dimension
            if preds.shape[1] == 1:
                preds = preds.squeeze(1)  # (pred_len,)
            
            # Adjust for requested horizon
            if horizon < self.pred_len:
                preds = preds[:horizon]
            elif horizon > self.pred_len:
                # Extend with last predicted value
                if preds.ndim == 1:
                    extension = np.full(horizon - self.pred_len, preds[-1])
                    preds = np.concatenate([preds, extension])
                else:
                    extension = np.tile(preds[-1:], (horizon - self.pred_len, 1))
                    preds = np.vstack([preds, extension])
        
        # Handle quantiles (simple heuristic since PatchTST is deterministic)
        if quantiles is not None:
            print(f"[{self.name}] Note: Generating approximate quantiles (model is deterministic)")
            hist_std = np.std(y)
            quantile_preds = {}
            for q in quantiles:
                z = np.percentile(np.random.randn(10000), q * 100)
                quantile_preds[q] = preds + z * hist_std * 0.3
            
            return self._format_predictions(preds, quantile_preds)
        
        return self._format_predictions(preds)
    
    def __repr__(self):
        status = "fitted" if self.is_fitted else "not fitted"
        return (f"{self.name}(seq_len={self.seq_len}, pred_len={self.pred_len}, "
                f"patch_len={self.patch_len}, d_model={self.d_model}, {status})")


# Test if imports work
if __name__ == "__main__":
    print("PatchTST Wrapper - Official Repo Integration")
    print("=" * 60)
    
    if PATCHTST_AVAILABLE:
        print("✓ PatchTST model successfully imported!")
    else:
        print("✗ PatchTST model not found.")
        print("  Make sure the directory structure is correct:")
        print("  Benchmarks/patchtst/PatchTST_supervised/models/PatchTST.py")
    
    print("\nTesting wrapper interface...")
    
    # Test with synthetic data
    np.random.seed(42)
    y = np.sin(np.linspace(0, 10*np.pi, 1000)) + 0.1*np.random.randn(1000)
    
    model = PatchTSTModel(
        seq_len=96,
        pred_len=24,
        patch_len=16,
        stride=8,
    )
    
    print(f"\nModel: {model}")
    
    if PATCHTST_AVAILABLE:
        print("\nFitting model on 800 timesteps...")
        model.fit(y[:800], epochs=5, batch_size=32, verbose=True)
        
        print("\nMaking predictions...")
        preds = model.predict(y[700:800], horizon=24, quantiles=[0.1, 0.5, 0.9])
        
        print(f"\nPrediction shape: {preds['mean'].shape}")
        print(f"First 5 predictions: {preds['mean'][:5]}")
        print(f"Quantiles available: {list(preds['quantiles'].keys()) if 'quantiles' in preds else 'None'}")
    else:
        print("\nSkipping fit/predict tests (PatchTST not available)")
    
    print("\n" + "=" * 60)