# COMP 545 Final Project - Day 3 Quick Start Guide

## What You Have Now

✅ **Dataset Ready**: 50 CiK tasks in `datasets/`
- `ts_instances.csv` - Time series data  
- `contexts.json` - Textual descriptions
- `train_test_split.json` - 35 train / 15 test split
- `task_metadata.csv` - Task type information

✅ **Models Ready**: AutoARIMA and ETS implementations in `Benchmarks/models/`
- Both models accept `context` parameter (ignored for numeric models)
- Both compute MAE automatically

✅ **Evaluation Script Ready**: `run_baseline_evaluation.py`

## What To Do Today (Day 3)

### Step 1: Run Baseline Evaluation (~10 minutes)

```bash
cd /Users/kaziashhabrahman/Documents/McGill/Fall\ 25/Comp\ 545/COMP545_Final_Project/

python run_baseline_evaluation.py
```

This will:
1. Load all 50 tasks
2. Run AutoARIMA on each task
3. Run ETS on each task  
4. Compute numeric oracle (best of both)
5. Save results to `results/` folder

**Expected output:**
```
================================================================================
BASELINE MODEL EVALUATION - DAY 3
AutoARIMA & ETS on 50 CiK Tasks
================================================================================

LOADING DATASET
✓ Loaded 50 time series instances
✓ Loaded 50 context descriptions

RUNNING AutoARIMA
AutoARIMA: 100%|████████████████| 50/50 [00:30<00:00,  1.67it/s]
  Mean MAE: 85.42

RUNNING ETS
ETS: 100%|███████████████████████| 50/50 [00:25<00:00,  2.00it/s]
  Mean MAE: 92.18

COMPUTING NUMERIC ORACLE
  Mean Oracle MAE: 78.35

SAVING RESULTS
✓ Saved AutoARIMA results to results/arima_results.csv
✓ Saved ETS results to results/ets_results.csv
✓ Saved Numeric Oracle to results/numeric_oracle.csv
✓ Saved comparison to results/baseline_comparison.csv

✓ Day 3 baseline evaluation complete!
```

### Step 2: View Results

```bash
python view_results.py
```

This shows:
- Overall performance summary
- Performance by task type
- Top 10 hardest/easiest tasks
- Any failures

### Step 3: Check Results Quality

Look for:
1. **Success rate**: Should be ~95-100%
2. **Mean MAE**: Should be 50-150 range
3. **Oracle improvement**: Should be 5-15% better than individual models

### Step 4: Save & Commit

```bash
git add results/*.csv
git add run_baseline_evaluation.py view_results.py
git commit -m "Day 3: Baseline evaluation complete"
```

## File Structure After Day 3

```
COMP545_Final_Project/
├── datasets/
│   ├── ts_instances.csv           # ✓ From Day 1
│   ├── contexts.json               # ✓ From Day 1
│   └── train_test_split.json      # ✓ From Day 1
├── Benchmarks/
│   ├── models/
│   │   ├── autoarima.py           # ✓ Existing
│   │   └── ets.py                 # ✓ Existing
│   └── evaluation/
│       └── metrics.py             # ✓ Existing
├── results/                        # ✓ NEW!
│   ├── arima_results.csv          # ← Generated today
│   ├── ets_results.csv            # ← Generated today
│   ├── numeric_oracle.csv         # ← Generated today
│   └── baseline_comparison.csv    # ← Generated today
├── run_baseline_evaluation.py     # ← Created today
├── view_results.py                # ← Created today
└── README_DAY3.md                 # ← Created today
```

## Troubleshooting

### Issue: Import errors

```bash
# Make sure you have required packages
pip install pandas numpy statsforecast statsmodels tqdm
```

### Issue: Script runs but no output

```bash
# Check Python path
which python
# Should point to your project environment

# Run with verbose output
python -u run_baseline_evaluation.py
```

### Issue: High error rate (>10% failures)

This is normal for some tasks! The CiK benchmark includes challenging time series.
Common failure modes:
- **Short series**: Some tasks have <30 data points
- **Flat series**: Zero variance in history
- **Extreme values**: Very large numbers causing numerical instability

## What's Next?

**Day 4-5**: 
- Run Chronos foundation model
- Run Llama 3 (with and without context)
- Compare all 5 models
- Identify where context helps most

**Week 2**:
- Extract features from time series
- Train XGBoost selector
- Evaluate selective-use policy
- Write final report

## Expected Day 3 Results

Typical performance (your results may vary):

| Model | Mean MAE | Notes |
|-------|----------|-------|
| AutoARIMA | 80-120 | Better on seasonal data |
| ETS | 90-140 | Better on smooth trends |
| Numeric Oracle | 70-110 | Best of both |

**Key insight**: These are your baselines. LLMs with context should beat the oracle!

## Questions?

1. Check `README_DAY3.md` for detailed documentation
2. Check `implementation_checklist.md` for Day 3 goals
3. Review model code in `Benchmarks/models/`

---

**Good luck with Day 3!** 🚀

Your numeric baselines will be crucial for understanding when LLMs provide value.
