# Day 3: Baseline Model Evaluation

## Overview
This script runs **AutoARIMA** and **ETS** models on all 50 CiK forecasting tasks and computes the **numeric oracle** (best of the two models per task).

## Quick Start

```bash
# From project root
python run_baseline_evaluation.py
```

## What the Script Does

### 1. **Load Dataset**
- Reads `datasets/ts_instances.csv` containing 50 time series
- Loads `datasets/contexts.json` with textual descriptions
- Parses JSON arrays into numpy format

### 2. **Run AutoARIMA**
- Fits AutoARIMA model to each task's history
- Makes predictions for the future horizon
- Computes Mean Absolute Error (MAE)
- Tracks success/failure status

### 3. **Run ETS (Exponential Smoothing)**
- Fits ETS model with additive trend
- Makes predictions for each task
- Computes MAE
- Handles edge cases gracefully

### 4. **Compute Numeric Oracle**
- For each task, selects the model with lower MAE
- This represents the "best achievable" performance with numeric models alone
- Shows how much room there is for improvement with LLMs + context

### 5. **Save Results**
All results are saved to `results/` directory:

- `arima_results.csv` - AutoARIMA performance on each task
- `ets_results.csv` - ETS performance on each task  
- `numeric_oracle.csv` - Best numeric model per task
- `baseline_comparison.csv` - Combined comparison table

## Expected Runtime
- **~5-10 minutes** for 50 tasks (depends on CPU)
- Progress bars show real-time status

## Output Format

### arima_results.csv / ets_results.csv
```
task_id,mae,status,horizon
task_000,45.23,success,67
task_001,123.45,success,64
...
```

### baseline_comparison.csv
```
task_id,mae_arima,mae_ets,oracle_mae,oracle_model
task_000,45.23,52.18,45.23,arima
task_001,123.45,98.76,98.76,ets
...
```

## Expected Performance

Based on typical CiK tasks:

| Model | Expected Mean MAE | Notes |
|-------|------------------|-------|
| AutoARIMA | 50-150 | Good on seasonal/trending data |
| ETS | 60-180 | Good on smooth trends |
| Numeric Oracle | 45-130 | Best of both worlds |

## Key Findings to Look For

1. **Which model wins more often?**
   - Look at `oracle_model` column distribution
   
2. **Task types where models struggle**
   - High MAE on `volatile` and `structural_break` types expected
   
3. **Baseline for LLM comparison**
   - These results establish the "numeric-only" performance
   - LLMs with context should beat this oracle

## Troubleshooting

### Import Errors
```bash
# Make sure you're in the project root
cd /Users/kaziashhabrahman/Documents/McGill/Fall\ 25/Comp\ 545/COMP545_Final_Project/

# Install missing packages
pip install tqdm statsforecast statsmodels pandas numpy
```

### Memory Issues
If you run out of memory, modify the script to process in batches:
```python
# Line 100: change ts_df.iterrows() to 
for idx, row in ts_df.head(10).iterrows():
```

## Next Steps (Day 4-5)

After running this script:

1. **Analyze results**: Look at which task types are hardest
2. **Prepare for LLMs**: Use these results as baseline
3. **Week 2**: Train selector to predict when context helps

## Questions?

Check the implementation checklist at `implementation_checklist.md` (Day 3 section) for more details.

---
**COMP 545 Final Project** | McGill University | Fall 2024
