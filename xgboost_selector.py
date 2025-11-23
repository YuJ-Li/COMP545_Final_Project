import os
import json
import re
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

# ============================================
# Configuration
# ============================================
MODEL_COSTS = {
    "arima": 1.0,
    "ets": 1.0,
    "llmp_no_context": 8.0,
    "llmp_with_context": 8.0,
    "gpt4o_mini_no_context": 8.0,
    "gpt4o_mini_with_context": 8.0,
}

TEST_SIZE = 0.2
RANDOM_SEED = 42

# Target: What are we predicting?
TARGET_COL = "llmp_context_beat_arima"


# ============================================
# Feature Extraction
# ============================================
def extract_context_features(context_text: str) -> dict:
    """Extract features from context text."""
    features = {}
    
    features["context_char_length"] = len(context_text)
    features["context_word_count"] = len(context_text.split())
    features["context_sentence_count"] = (
        context_text.count('.') + context_text.count('!') + context_text.count('?')
    )
    
    text_lower = context_text.lower()
    
    constraint_words = ['must', 'cannot', 'never', 'always', 'constraint', 
                        'limit', 'maximum', 'minimum', 'required', 'forbidden',
                        'no ', 'zero', 'none']
    features["has_constraints"] = int(any(w in text_lower for w in constraint_words))
    features["constraint_count"] = sum(text_lower.count(w) for w in constraint_words)
    
    event_words = ['after', 'before', 'during', 'when', 'until', 'from', 
                   'spike', 'drop', 'change', 'shift', 'disruption', 'event',
                   'period', 'days', 'hours', 'weeks']
    features["has_events"] = int(any(w in text_lower for w in event_words))
    features["event_count"] = sum(text_lower.count(w) for w in event_words)
    
    causal_words = ['because', 'due to', 'caused by', 'results in', 'leads to',
                    'therefore', 'consequently', 'resulting']
    features["has_causality"] = int(any(w in text_lower for w in causal_words))
    
    features["mentions_zero"] = int('zero' in text_lower or ' 0 ' in context_text or 'no ' in text_lower)
    features["numeric_mentions"] = len(re.findall(r'\d+', context_text))
    
    features["has_scenario"] = int('[scenario]' in text_lower)
    features["has_background"] = int('[background]' in text_lower)
    
    return features


# ============================================
# Data Loading
# ============================================
def load_single_domain(domain_dir: str, domain_name: str) -> pd.DataFrame:
    """Load all data for a single domain."""
    
    comparison_path = os.path.join(domain_dir, "comparison.csv")
    datasets_dir = os.path.join(domain_dir, "datasets")
    metadata_path = os.path.join(datasets_dir, "task_metadata.csv")
    contexts_path = os.path.join(datasets_dir, "contexts.json")
    
    if not os.path.exists(comparison_path):
        raise FileNotFoundError(f"Missing: {comparison_path}")
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(f"Missing: {metadata_path}")
    if not os.path.exists(contexts_path):
        raise FileNotFoundError(f"Missing: {contexts_path}")
    
    comparison_df = pd.read_csv(comparison_path)
    
    metadata_df = pd.read_csv(metadata_path)
    metadata_df = metadata_df.rename(columns={"id": "task_id"})
    
    with open(contexts_path, 'r') as f:
        contexts = json.load(f)
    
    context_features = []
    for task_id, context_text in contexts.items():
        features = extract_context_features(context_text)
        features["task_id"] = task_id
        context_features.append(features)
    context_df = pd.DataFrame(context_features)
    
    df = comparison_df.merge(metadata_df, on="task_id", how="inner")
    df = df.merge(context_df, on="task_id", how="inner")
    df["domain"] = domain_name
    
    return df


def load_all_domains(results_root: str) -> pd.DataFrame:
    """Load and combine data from all domain folders."""
    all_dfs = []
    
    for name in sorted(os.listdir(results_root)):
        domain_dir = os.path.join(results_root, name)
        
        if not os.path.isdir(domain_dir):
            continue
        if name == "xgboost_compiled":
            continue
        
        datasets_dir = os.path.join(domain_dir, "datasets")
        if not os.path.exists(datasets_dir):
            print(f"  [SKIP] {name}: No datasets/ folder")
            continue
        
        try:
            df = load_single_domain(domain_dir, name)
            all_dfs.append(df)
            print(f"  [OK] {name}: {len(df)} tasks")
        except Exception as e:
            print(f"  [ERROR] {name}: {e}")
    
    if not all_dfs:
        raise ValueError("No domains loaded!")
    
    combined = pd.concat(all_dfs, ignore_index=True)
    return combined


# ============================================
# Feature Selection
# ============================================
FEATURE_COLS = [
    "mean", "std", "trend", "volatility", "history_length", "future_length",
    "context_length",
    "context_char_length", "context_word_count", "context_sentence_count",
    "has_constraints", "constraint_count",
    "has_events", "event_count",
    "has_causality", "mentions_zero", "numeric_mentions",
    "has_scenario", "has_background",
    "arima_nmae", "arima_da",
    "ets_nmae", "ets_da",
]


def get_available_features(df: pd.DataFrame) -> list:
    """Return features that exist in the dataframe."""
    return [f for f in FEATURE_COLS if f in df.columns]


# ============================================
# Training and Evaluation
# ============================================
def train_xgboost(X_train, y_train):
    """Train XGBoost classifier."""
    clf = XGBClassifier(
        max_depth=4,
        n_estimators=100,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_SEED,
        eval_metric='logloss',
        use_label_encoder=False
    )
    clf.fit(X_train, y_train)
    return clf


def evaluate_classifier(clf, X_test, y_test) -> dict:
    """Evaluate classifier performance."""
    y_pred = clf.predict(X_test)
    y_pred_proba = clf.predict_proba(X_test)[:, 1] if len(clf.classes_) == 2 else None
    
    results = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "y_test": y_test.values,
        "y_pred": y_pred,
    }
    
    if y_pred_proba is not None and len(np.unique(y_test)) > 1:
        results["roc_auc"] = roc_auc_score(y_test, y_pred_proba)
    else:
        results["roc_auc"] = None
    
    return results


def get_feature_importance(clf, feature_cols: list) -> pd.DataFrame:
    """Get feature importance from trained classifier."""
    return pd.DataFrame({
        "feature": feature_cols,
        "importance": clf.feature_importances_
    }).sort_values("importance", ascending=False)


# ============================================
# Policy Comparison (FIXED VERSION)
# ============================================
def compare_policies(test_df: pd.DataFrame, clf, feature_cols: list) -> pd.DataFrame:
    """Compare 4 policies on test set."""
    
    # IMPORTANT: Reset index and work with a clean copy
    df = test_df.reset_index(drop=True).copy()
    n_tasks = len(df)
    
    # Determine which LLM columns to use
    if "llmp" in TARGET_COL:
        llm_nmae_col = "llmp_with_context_nmae"
        llm_da_col = "llmp_with_context_da"
    else:
        llm_nmae_col = "gpt4o_mini_with_context_nmae"
        llm_da_col = "gpt4o_mini_with_context_da"
    
    # Policy 1: Always ARIMA
    arima_nmae = float(df["arima_nmae"].mean())
    arima_da = float(df["arima_da"].mean())
    
    # Policy 2: Always LLM with context
    llm_nmae = float(df[llm_nmae_col].mean())
    llm_da = float(df[llm_da_col].mean())
    
    # Policy 3: XGBoost Selector
    X_test = df[feature_cols].fillna(0)
    predictions = clf.predict(X_test)
    
    selector_nmae_list = []
    selector_da_list = []
    selector_cost = 0.0
    
    for i in range(n_tasks):
        if predictions[i] == 1:  # Use LLM
            selector_nmae_list.append(float(df.loc[i, llm_nmae_col]))
            selector_da_list.append(float(df.loc[i, llm_da_col]))
            selector_cost += MODEL_COSTS["llmp_with_context"]
        else:  # Use ARIMA
            selector_nmae_list.append(float(df.loc[i, "arima_nmae"]))
            selector_da_list.append(float(df.loc[i, "arima_da"]))
            selector_cost += MODEL_COSTS["arima"]
    
    selector_nmae = float(np.mean(selector_nmae_list))
    selector_da = float(np.mean(selector_da_list))
    selector_pct_llm = 100.0 * sum(predictions) / n_tasks
    
    # Policy 4: Oracle (perfect hindsight)
    oracle_nmae_list = []
    oracle_da_list = []
    oracle_cost = 0.0
    oracle_llm_count = 0
    
    for i in range(n_tasks):
        llm_is_better = df.loc[i, llm_nmae_col] < df.loc[i, "arima_nmae"]
        if llm_is_better:
            oracle_nmae_list.append(float(df.loc[i, llm_nmae_col]))
            oracle_da_list.append(float(df.loc[i, llm_da_col]))
            oracle_cost += MODEL_COSTS["llmp_with_context"]
            oracle_llm_count += 1
        else:
            oracle_nmae_list.append(float(df.loc[i, "arima_nmae"]))
            oracle_da_list.append(float(df.loc[i, "arima_da"]))
            oracle_cost += MODEL_COSTS["arima"]
    
    oracle_nmae = float(np.mean(oracle_nmae_list))
    oracle_da = float(np.mean(oracle_da_list))
    oracle_pct_llm = 100.0 * oracle_llm_count / n_tasks
    
    # Build results
    always_llm_cost = n_tasks * MODEL_COSTS["llmp_with_context"]
    
    policies = [
        {
            "Policy": "Always ARIMA",
            "Mean NMAE": arima_nmae,
            "Mean DA": arima_da,
            "% Using LLM": 0.0,
            "Total Cost": float(n_tasks * MODEL_COSTS["arima"]),
            "Cost vs Always-LLM": f"{100*n_tasks*MODEL_COSTS['arima']/always_llm_cost:.0f}%"
        },
        {
            "Policy": "Always LLM+Context",
            "Mean NMAE": llm_nmae,
            "Mean DA": llm_da,
            "% Using LLM": 100.0,
            "Total Cost": float(always_llm_cost),
            "Cost vs Always-LLM": "100%"
        },
        {
            "Policy": "XGBoost Selector",
            "Mean NMAE": selector_nmae,
            "Mean DA": selector_da,
            "% Using LLM": selector_pct_llm,
            "Total Cost": float(selector_cost),
            "Cost vs Always-LLM": f"{100*selector_cost/always_llm_cost:.0f}%"
        },
        {
            "Policy": "Oracle (Perfect)",
            "Mean NMAE": oracle_nmae,
            "Mean DA": oracle_da,
            "% Using LLM": oracle_pct_llm,
            "Total Cost": float(oracle_cost),
            "Cost vs Always-LLM": f"{100*oracle_cost/always_llm_cost:.0f}%"
        }
    ]
    
    return pd.DataFrame(policies)


# ============================================
# Per-Domain Breakdown
# ============================================
def analyze_per_domain(test_df: pd.DataFrame, clf, feature_cols: list) -> pd.DataFrame:
    """Analyze classifier performance per domain."""
    rows = []
    
    for domain in test_df["domain"].unique():
        domain_df = test_df[test_df["domain"] == domain].reset_index(drop=True)
        n_tasks = len(domain_df)
        
        if n_tasks == 0:
            continue
        
        X_domain = domain_df[feature_cols].fillna(0)
        y_domain = domain_df[TARGET_COL]
        y_pred = clf.predict(X_domain)
        
        acc = accuracy_score(y_domain, y_pred) if len(y_domain) > 0 else 0
        
        rows.append({
            "Domain": domain,
            "Test Tasks": n_tasks,
            "Accuracy": acc,
            "Actual % LLM Wins": 100 * y_domain.mean(),
            "Predicted % LLM": 100 * y_pred.mean(),
        })
    
    return pd.DataFrame(rows).sort_values("Domain")


# ============================================
# Plotting
# ============================================
def plot_feature_importance(importance_df: pd.DataFrame, output_path: str, top_n: int = 15):
    """Plot top N most important features."""
    top_features = importance_df.head(top_n)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ['steelblue' if imp > 0 else 'lightgray' for imp in top_features["importance"]]
    ax.barh(range(len(top_features)), top_features["importance"], color=colors)
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features["feature"])
    ax.invert_yaxis()
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Features for Predicting: {TARGET_COL}")
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, output_path: str):
    """Plot confusion matrix."""
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Predict: ARIMA', 'Predict: LLM'],
                yticklabels=['Actual: ARIMA wins', 'Actual: LLM wins'])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix\nTarget: {TARGET_COL}")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_policy_comparison(policies_df: pd.DataFrame, output_path: str):
    """Bar chart comparing policies."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    
    # NMAE comparison
    ax = axes[0]
    nmae_values = policies_df["Mean NMAE"].values
    bars = ax.bar(range(len(policies_df)), nmae_values, color=colors)
    ax.set_xticks(range(len(policies_df)))
    ax.set_xticklabels(policies_df["Policy"], rotation=45, ha='right')
    ax.set_ylabel("Mean NMAE (lower is better)")
    ax.set_title("Policy Comparison: Accuracy")
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, nmae_values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{val:.2f}', ha='center', va='bottom', fontsize=9)
    
    # Cost comparison
    ax = axes[1]
    cost_values = policies_df["Total Cost"].values
    bars = ax.bar(range(len(policies_df)), cost_values, color=colors)
    ax.set_xticks(range(len(policies_df)))
    ax.set_xticklabels(policies_df["Policy"], rotation=45, ha='right')
    ax.set_ylabel("Total Cost")
    ax.set_title("Policy Comparison: Cost")
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_per_domain_accuracy(domain_df: pd.DataFrame, output_path: str):
    """Bar chart of accuracy per domain."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = range(len(domain_df))
    ax.bar(x, domain_df["Accuracy"], color='steelblue', alpha=0.8)
    ax.axhline(y=0.5, color='red', linestyle='--', label='Random baseline')
    ax.set_xticks(x)
    ax.set_xticklabels(domain_df["Domain"], rotation=45, ha='right')
    ax.set_ylabel("Accuracy")
    ax.set_title("XGBoost Accuracy by Domain (on Test Set)")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


# ============================================
# Main Pipeline
# ============================================
def main(results_root: str):
    """Main pipeline."""
    
    output_dir = os.path.join(results_root, "xgboost_compiled")
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*60)
    print("XGBoost Pooled Classifier (FIXED)")
    print("="*60)
    print(f"Target variable: {TARGET_COL}")
    print(f"Test size: {TEST_SIZE*100:.0f}%")
    print(f"Output directory: {output_dir}")
    print()
    
    # Load all domains
    print("Loading domains...")
    df = load_all_domains(results_root)
    print(f"\nTotal tasks loaded: {len(df)}")
    print(f"Domains: {df['domain'].nunique()}")
    
    # Check target column
    if TARGET_COL not in df.columns:
        raise ValueError(f"Target column '{TARGET_COL}' not found. Run classifier_v2.py first!")
    
    # Get features
    feature_cols = get_available_features(df)
    print(f"Features available: {len(feature_cols)}")
    
    # Class balance
    positive_rate = df[TARGET_COL].mean()
    print(f"Class balance: {positive_rate*100:.1f}% positive (LLM wins)")
    
    # Train/test split
    print(f"\nSplitting data...")
    train_df, test_df = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    print(f"Train: {len(train_df)} tasks")
    print(f"Test: {len(test_df)} tasks")
    
    # Prepare features
    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df[TARGET_COL]
    X_test = test_df[feature_cols].fillna(0)
    y_test = test_df[TARGET_COL]
    
    # Check class balance
    train_positive = y_train.sum()
    train_negative = len(y_train) - train_positive
    print(f"Train class balance: {train_positive} positive, {train_negative} negative")
    
    if train_positive == 0 or train_negative == 0:
        print("\n[ERROR] Only one class in training data.")
        return
    
    # Train
    print("\nTraining XGBoost...")
    clf = train_xgboost(X_train, y_train)
    
    # Evaluate
    print("\nEvaluating...")
    eval_results = evaluate_classifier(clf, X_test, y_test)
    
    print(f"\n{'='*60}")
    print("CLASSIFICATION RESULTS")
    print('='*60)
    print(f"Accuracy:  {eval_results['accuracy']:.3f}")
    print(f"Precision: {eval_results['precision']:.3f}")
    print(f"Recall:    {eval_results['recall']:.3f}")
    print(f"F1 Score:  {eval_results['f1']:.3f}")
    if eval_results['roc_auc']:
        print(f"ROC-AUC:   {eval_results['roc_auc']:.3f}")
    
    # Feature importance
    importance_df = get_feature_importance(clf, feature_cols)
    print(f"\n{'='*60}")
    print("TOP 10 FEATURES")
    print('='*60)
    for _, row in importance_df.head(10).iterrows():
        print(f"  {row['feature']:25s}: {row['importance']:.4f}")
    
    # Policy comparison
    policies_df = compare_policies(test_df, clf, feature_cols)
    print(f"\n{'='*60}")
    print("POLICY COMPARISON (on test set)")
    print('='*60)
    print(policies_df.to_string(index=False))
    
    # Per-domain breakdown
    domain_results = analyze_per_domain(test_df, clf, feature_cols)
    print(f"\n{'='*60}")
    print("PER-DOMAIN ACCURACY (on test set)")
    print('='*60)
    print(domain_results.to_string(index=False))
    
    # Save outputs
    print(f"\n{'='*60}")
    print("SAVING OUTPUTS")
    print('='*60)
    
    importance_df.to_csv(os.path.join(output_dir, "feature_importance.csv"), index=False)
    policies_df.to_csv(os.path.join(output_dir, "policy_comparison.csv"), index=False)
    domain_results.to_csv(os.path.join(output_dir, "per_domain_accuracy.csv"), index=False)
    
    test_df_with_pred = test_df.copy()
    test_df_with_pred["xgb_prediction"] = clf.predict(X_test)
    test_df_with_pred.to_csv(os.path.join(output_dir, "test_predictions.csv"), index=False)
    
    print(f"  Saved: feature_importance.csv")
    print(f"  Saved: policy_comparison.csv")
    print(f"  Saved: per_domain_accuracy.csv")
    print(f"  Saved: test_predictions.csv")
    
    # Plots
    plot_feature_importance(importance_df, os.path.join(output_dir, "feature_importance.png"))
    plot_confusion_matrix(eval_results['confusion_matrix'], os.path.join(output_dir, "confusion_matrix.png"))
    plot_policy_comparison(policies_df, os.path.join(output_dir, "policy_comparison.png"))
    plot_per_domain_accuracy(domain_results, os.path.join(output_dir, "per_domain_accuracy.png"))
    
    print(f"  Saved: feature_importance.png")
    print(f"  Saved: confusion_matrix.png")
    print(f"  Saved: policy_comparison.png")
    print(f"  Saved: per_domain_accuracy.png")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print('='*60)
    print(f"Classifier accuracy: {eval_results['accuracy']*100:.1f}%")
    
    selector_row = policies_df[policies_df["Policy"] == "XGBoost Selector"].iloc[0]
    print(f"XGBoost Selector: Mean NMAE = {selector_row['Mean NMAE']:.3f}")
    print(f"XGBoost Selector uses LLM on {selector_row['% Using LLM']:.1f}% of tasks")
    print(f"Cost savings vs Always-LLM: {selector_row['Cost vs Always-LLM']}")
    
    print(f"\nAll outputs saved to: {output_dir}")


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_root = os.path.join(base_dir, "results")
    
    main(results_root)