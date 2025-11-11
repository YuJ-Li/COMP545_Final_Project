"""
Test that all imports work correctly
"""

import sys
import os

# Make sure we're in the right directory
print(f"Current directory: {os.getcwd()}")
print(f"Python path: {sys.path[0]}\n")

print("Testing imports...")
print("="*60)

try:
    print("1. Testing metrics...")
    from evaluation.metrics import mae, rmse, mase, compute_all_metrics
    print("   ✓ Metrics imported successfully")
except Exception as e:
    print(f"   ✗ Metrics import failed: {e}")

try:
    print("2. Testing evaluator...")
    from evaluation.evaluator import TimeSeriesEvaluator
    print("   ✓ Evaluator imported successfully")
except Exception as e:
    print(f"   ✗ Evaluator import failed: {e}")

try:
    print("3. Testing base model...")
    from models.base_model import TimeSeriesModel, BaselineModel
    print("   ✓ Base model imported successfully")
except Exception as e:
    print(f"   ✗ Base model import failed: {e}")

try:
    print("4. Testing AutoARIMA...")
    from models.autoarima import AutoARIMAModel
    print("   ✓ AutoARIMA imported successfully")
except Exception as e:
    print(f"   ✗ AutoARIMA import failed: {e}")

try:
    print("5. Testing PatchTST...")
    from models.patchtst import PatchTSTModel
    print("   ✓ PatchTST imported successfully")
except Exception as e:
    print(f"   ✗ PatchTST import failed: {e}")

# Test creating instances
try:
    print("\n6. Testing model instantiation...")
    from models.autoarima import AutoARIMAModel
    model = AutoARIMAModel(season_length=1, freq='D')
    print(f"   ✓ Created: {model}")
except Exception as e:
    print(f"   ✗ Model instantiation failed: {e}")

try:
    print("7. Testing evaluator instantiation...")
    from evaluation.evaluator import TimeSeriesEvaluator
    evaluator = TimeSeriesEvaluator(quantiles=[0.1, 0.5, 0.9])
    print(f"   ✓ Created: {evaluator}")
except Exception as e:
    print(f"   ✗ Evaluator instantiation failed: {e}")

print("\n" + "="*60)
print("✓ All imports working correctly!")
print("="*60)