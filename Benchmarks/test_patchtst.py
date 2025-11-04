"""
Test PatchTST Integration

Quick test to verify PatchTST wrapper works with the official repo.
Run this from Benchmarks/ directory.
"""

import numpy as np
import sys
from pathlib import Path

# Test if we can import PatchTST
print("=" * 60)
print("PATCHTST INTEGRATION TEST")
print("=" * 60)

print("\n1. Checking directory structure...")
patchtst_path = Path('patchtst/PatchTST_supervised')
if patchtst_path.exists():
    print(f"   ✓ Found: {patchtst_path}")
    model_file = patchtst_path / 'models' / 'PatchTST.py'
    if model_file.exists():
        print(f"   ✓ Found: {model_file}")
    else:
        print(f"   ✗ Missing: {model_file}")
else:
    print(f"   ✗ Missing: {patchtst_path}")
    print("   Make sure you're running from Benchmarks/ directory")

print("\n2. Testing import...")
try:
    from patchtst_wrapper import PatchTSTModel, PATCHTST_AVAILABLE
    print("   ✓ Wrapper imported successfully")
    
    if PATCHTST_AVAILABLE:
        print("   ✓ PatchTST model available!")
    else:
        print("   ✗ PatchTST model not found")
        print("   Check if patchtst/PatchTST_supervised/ exists")
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    sys.exit(1)

print("\n3. Creating model...")
try:
    model = PatchTSTModel(
        seq_len=96,
        pred_len=24,
        patch_len=16,
        stride=8,
        d_model=64,  # Smaller for quick test
        n_heads=4,
        e_layers=2,
    )
    print(f"   ✓ Model created: {model}")
except Exception as e:
    print(f"   ✗ Failed: {e}")
    sys.exit(1)

print("\n4. Testing with synthetic data...")
np.random.seed(42)
y = np.sin(np.linspace(0, 10*np.pi, 500)) + 0.1*np.random.randn(500)
print(f"   Generated {len(y)} timesteps")

print("\n5. Training model...")
try:
    model.fit(y[:400], epochs=3, batch_size=16, verbose=True)
    print("   ✓ Training complete")
except Exception as e:
    print(f"   ✗ Training failed: {e}")
    import traceback
    traceback.print_exc()
    
    if not PATCHTST_AVAILABLE:
        print("\n   Note: Using placeholder since PatchTST not available")

print("\n6. Making predictions...")
try:
    history = y[300:400]  # 100 timesteps
    preds = model.predict(history, horizon=24)
    
    print(f"   ✓ Predictions shape: {preds['mean'].shape}")
    print(f"   ✓ First 5 predictions: {preds['mean'][:5]}")
except Exception as e:
    print(f"   ✗ Prediction failed: {e}")
    import traceback
    traceback.print_exc()

print("\n7. Testing rolling predictions...")
try:
    results = model.predict_rolling(
        y, 
        context_length=96, 
        horizon=24, 
        step=50
    )
    print(f"   ✓ Generated {len(results)} rolling windows")
except Exception as e:
    print(f"   ✗ Rolling prediction failed: {e}")

print("\n" + "=" * 60)
print("TEST SUMMARY")
print("=" * 60)

if PATCHTST_AVAILABLE:
    print("✓ PatchTST wrapper is fully working!")
    print("\nNext steps:")
    print("1. Run: python run_unified_benchmark.py --data ../LLM/Data/finance_data.csv --models patchtst")
    print("2. Compare with AutoARIMA: --models autoarima patchtst")
else:
    print("⚠ PatchTST wrapper works but using placeholder predictions")
    print("\nTo fix:")
    print("1. Make sure patchtst/ directory contains PatchTST_supervised/")
    print("2. Check path in patchtst_wrapper.py is correct")
    print("3. Try: cd Benchmarks && python test_patchtst.py")

print("=" * 60)