# COMP545_Final_Project
## Generate dataset
```angular2html
python generate_cik_dataset.py
```

## Run Baseline (Arima && ETS)
```angular2html
python run_baseline_evaluation.py
```

## Run Llama
```angular2html
python run_llm_evaluation.py --method dp --use-context 
```
Note that you can use `--method llmp` and `--no-context` as options as well

## Run mistral
Run `run_all_domains_mistral.sh` for complete run;
If a run on a specific domain is needed, please consult usage within `run_mistral_standalone.py`

## Run GPT4o
Run `run_all_domains_gpt4o.sh` for complete run;
If a run on a specific domain is needed, please consult usage within `run_domain_gpt4o.py`

## Run classifier
```angular2html
python xgboost_selector.py
```

## Results
Results can bee found in `/results`, for further analysis, please follow the `REEADME.md` in `/results`