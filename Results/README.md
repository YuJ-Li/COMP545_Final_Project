# Results Directory

This directory will contain baseline evaluation results after running `run_baseline_evaluation.py`.

## Expected Files

After running the baseline evaluation:

- `arima_results.csv` - AutoARIMA performance on all 50 tasks
- `ets_results.csv` - ETS performance on all 50 tasks
- `numeric_oracle.csv` - Best numeric model per task
- `baseline_comparison.csv` - Side-by-side comparison

## Usage

```bash
# Run baseline evaluation
python run_baseline_evaluation.py

# View results
python view_results.py
```
