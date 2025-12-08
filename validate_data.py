"""
Data Validation Script for XGBoost Classifier

Run this BEFORE build_classifier.py to verify your data structure is correct.
This will catch common issues early and save you debugging time!

Usage:
    python validate_data.py
"""

import pandas as pd
import numpy as np
import os
import json
from pathlib import Path

def check_file_exists(filepath, description):
    """Check if a file exists and report."""
    if os.path.exists(filepath):
        print(f"✓ {description}: {filepath}")
        return True
    else:
        print(f"✗ {description} NOT FOUND: {filepath}")
        return False

def validate_compiled_comparison(filepath='results/compiled_comparison.csv'):
    """Validate the main compiled comparison file."""
    print("\n" + "=" * 80)
    print("1. VALIDATING COMPILED COMPARISON FILE")
    print("=" * 80)
    
    if not check_file_exists(filepath, "Compiled comparison"):
        return False
    
    df = pd.read_csv(filepath)
    print(f"✓ Loaded {len(df)} tasks")
    
    # Check required columns
    required_cols = [
        'task_id', 'domain',
        'arima_nmae', 'arima_da',
        'ets_nmae', 'ets_da',
        'best_baseline_nmae', 'best_baseline_da',
        'mistral_nmae', 'mistral_da'
    ]
    
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        print(f"\n✗ MISSING REQUIRED COLUMNS: {missing_cols}")
        print(f"   Available columns: {list(df.columns)}")
        return False
    
    print(f"✓ All required columns present")
    
    # Check for missing values
    missing_counts = df[required_cols].isnull().sum()
    if missing_counts.sum() > 0:
        print(f"\n  WARNING: Missing values detected:")
        for col, count in missing_counts[missing_counts > 0].items():
            print(f"   {col}: {count} missing values")
    else:
        print(f"✓ No missing values in required columns")
    
    # Check target variable
    if 'mistral_beats_baseline' in df.columns:
        positive_pct = df['mistral_beats_baseline'].mean() * 100
        print(f"\n✓ Target variable exists")
        print(f"   Positive class: {df['mistral_beats_baseline'].sum()} ({positive_pct:.1f}%)")
        
        if positive_pct < 20:
            print(f"  WARNING: Very low positive class rate - classifier may struggle")
        elif positive_pct > 80:
            print(f"  WARNING: Very high positive class rate - classifier may struggle")
    else:
        print(f"\n Target 'mistral_beats_baseline' not found (will be created)")
    
    # Check domains
    print(f"\n✓ Domains found: {df['domain'].nunique()}")
    domain_counts = df['domain'].value_counts()
    print(f"   Tasks per domain:")
    for domain, count in domain_counts.items():
        print(f"   - {domain}: {count} tasks")
    
    return True

def validate_domain_data(results_dir='results'):
    """Validate metadata and context files for each domain."""
    print("\n" + "=" * 80)
    print("2. VALIDATING DOMAIN-SPECIFIC DATA")
    print("=" * 80)
    
    # Get list of domain folders
    if not os.path.exists(results_dir):
        print(f"✗ Results directory not found: {results_dir}")
        return False
    
    domain_folders = [d for d in os.listdir(results_dir) 
                     if os.path.isdir(os.path.join(results_dir, d))]
    
    print(f"✓ Found {len(domain_folders)} domain folders")
    
    metadata_ok = 0
    contexts_ok = 0
    
    for domain in domain_folders:
        print(f"\n--- {domain} ---")
        
        # Check metadata
        metadata_path = os.path.join(results_dir, domain, 'datasets', 'task_metadata.csv')
        if os.path.exists(metadata_path):
            df_meta = pd.read_csv(metadata_path)
            print(f"✓ Metadata: {len(df_meta)} tasks")
            
            # Check for key TS features
            expected_features = ['mean', 'std', 'trend', 'seasonality_strength', 'volatility']
            available = [f for f in expected_features if f in df_meta.columns]
            print(f"  Features: {len(available)}/{len(expected_features)} expected features found")
            
            if len(available) < len(expected_features):
                missing = [f for f in expected_features if f not in df_meta.columns]
                print(f"    Missing: {missing}")
            
            metadata_ok += 1
        else:
            print(f"✗ Metadata NOT FOUND: {metadata_path}")
        
        # Check contexts
        contexts_path = os.path.join(results_dir, domain, 'datasets', 'contexts.json')
        if os.path.exists(contexts_path):
            with open(contexts_path, 'r') as f:
                contexts = json.load(f)
            print(f"✓ Contexts: {len(contexts)} tasks")
            
            # Sample a context to verify format
            sample_task = list(contexts.keys())[0]
            sample_context = contexts[sample_task]
            print(f"  Sample context length: {len(sample_context)} chars")
            
            if len(sample_context) < 10:
                print(f"    WARNING: Very short context - may not be informative")
            
            contexts_ok += 1
        else:
            print(f"✗ Contexts NOT FOUND: {contexts_path}")
    
    print(f"\n{'=' * 80}")
    print(f"Domain Data Summary:")
    print(f"  Domains with metadata: {metadata_ok}/{len(domain_folders)}")
    print(f"  Domains with contexts: {contexts_ok}/{len(domain_folders)}")
    
    if metadata_ok < len(domain_folders) or contexts_ok < len(domain_folders):
        print(f"\n  WARNING: Some domains are missing data")
        print(f"   Classifier will still run but may have reduced accuracy")
    
    return True

def validate_output_directory(output_dir='results/classifier_outputs'):
    """Check if output directory exists or can be created."""
    print("\n" + "=" * 80)
    print("3. VALIDATING OUTPUT DIRECTORY")
    print("=" * 80)
    
    if os.path.exists(output_dir):
        print(f"✓ Output directory exists: {output_dir}")
        
        # Check if writable
        test_file = os.path.join(output_dir, '.test_write')
        try:
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
            print(f"✓ Directory is writable")
        except:
            print(f"✗ Directory is NOT writable - check permissions")
            return False
    else:
        print(f"  WARNING: Output directory doesn't exist: {output_dir}")
        print(f"   Will be created automatically")
        
        # Try to create it
        try:
            os.makedirs(output_dir, exist_ok=True)
            print(f"✓ Successfully created output directory")
        except Exception as e:
            print(f"✗ Could not create directory: {e}")
            return False
    
    return True

def estimate_data_size():
    """Estimate expected train/test sizes."""
    print("\n" + "=" * 80)
    print("4. ESTIMATING DATA SPLIT")
    print("=" * 80)
    
    try:
        df = pd.read_csv('results/compiled_comparison.csv')
        n_tasks = len(df)
        n_train = int(n_tasks * 0.8)
        n_test = n_tasks - n_train
        
        print(f"✓ Total tasks: {n_tasks}")
        print(f"✓ Expected training set: {n_train} tasks")
        print(f"✓ Expected test set: {n_test} tasks")
        
        if n_train < 50:
            print(f"\n WARNING: Very small training set ({n_train} tasks)")
            print(f"   Consider collecting more data for better classifier performance")
        
        if n_test < 20:
            print(f"\n WARNING: Very small test set ({n_test} tasks)")
            print(f"   Evaluation metrics may be noisy")
        
    except:
        print(f" WARNING: Could not load compiled comparison file")
    
    return True

def main():
    """Run all validation checks."""
    print("\n" + "=" * 80)
    print(" DATA VALIDATION FOR XGBOOST CLASSIFIER")
    print("=" * 80)
    print("\nThis script checks your data structure before running the classifier.")
    print("Fix any ✗ errors before proceeding!\n")
    
    # Run all checks
    check1 = validate_compiled_comparison()
    check2 = validate_domain_data()
    check3 = validate_output_directory()
    check4 = estimate_data_size()
    
    # Final summary
    print("\n" + "=" * 80)
    print(" VALIDATION SUMMARY")
    print("=" * 80)
    
    if check1 and check2 and check3:
        print("\n✓✓✓ ALL CHECKS PASSED ✓✓✓")
        print("\nYou're ready to run build_classifier.py!")
        print("\nNext step:")
        print("  python build_classifier.py")
    else:
        print("\n✗✗✗ SOME CHECKS FAILED ✗✗✗")
        print("\nPlease fix the issues above before running the classifier.")
        print("\nCommon fixes:")
        print("  1. Make sure you're in the project root directory")
        print("  2. Check that results/compiled_comparison.csv exists")
        print("  3. Verify domain folders have datasets/ subdirectories")
        print("  4. Ensure task_metadata.csv and contexts.json exist for each domain")
    
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()