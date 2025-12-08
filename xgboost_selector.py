import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix,
    classification_report
)
import xgboost as xgb
import os
import json
import re
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style for plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

def load_time_series_metadata(results_dir='results'):
    """
    Load time series metadata from all domain folders and merge into single dataframe.
    """
    print("=" * 80)
    print("TASK 1: Loading Time Series Metadata")
    print("=" * 80)
    
    all_metadata = []
    domain_folders = [d for d in os.listdir(results_dir) 
                     if os.path.isdir(os.path.join(results_dir, d))]
    
    print(f"Found {len(domain_folders)} domain folders")
    
    for domain in domain_folders:
        metadata_path = os.path.join(results_dir, domain, 'datasets', 'task_metadata.csv')
        
        if not os.path.exists(metadata_path):
            print(f"  WARNING: No metadata found for {domain}")
            continue
        
        # Load metadata for this domain
        df_meta = pd.read_csv(metadata_path)
        
        # Standardize column names: 'id' -> 'task_id'
        if 'id' in df_meta.columns and 'task_id' not in df_meta.columns:
            df_meta['task_id'] = df_meta['id']
        
        # Ensure domain column exists
        if 'domain' not in df_meta.columns:
            df_meta['domain'] = domain
        
        all_metadata.append(df_meta)
        print(f"✓ Loaded {len(df_meta)} tasks from {domain}")
    
    # Combine all metadata
    df_metadata = pd.concat(all_metadata, ignore_index=True)
    print(f"\n✓ Total metadata loaded: {len(df_metadata)} tasks")
    print(f"✓ Metadata columns: {list(df_metadata.columns)}")
    
    return df_metadata

def extract_context_features(context_text):
    """
    Extract features from a single context string.
    """
    if pd.isna(context_text) or context_text == "":
        # Return zeros if no context
        return {
            'context_char_length': 0,
            'context_word_count': 0,
            'context_sentence_count': 0,
            'avg_word_length': 0,
            'has_constraint': 0,
            'has_future_event': 0,
            'has_number': 0,
            'has_temporal': 0,
            'has_causal': 0,
            'has_negation': 0,
            'mentions_constraint': 0,
            'mentions_event': 0,
            'mentions_physical': 0
        }
    
    text_lower = context_text.lower()
    
    # Basic text statistics
    char_length = len(context_text)
    words = text_lower.split()
    word_count = len(words)
    sentence_count = len(re.split(r'[.!?]+', context_text))
    avg_word_length = np.mean([len(w) for w in words]) if words else 0
    
    # Keyword indicators
    constraint_words = ['must', 'cannot', 'never', 'always', 'only', 'required', 'forbidden']
    future_words = ['will', 'upcoming', 'scheduled', 'planned', 'future', 'next']
    temporal_words = ['night', 'day', 'weekend', 'holiday', 'season', 'morning', 'evening', 'hour']
    causal_words = ['because', 'due to', 'caused by', 'results in', 'leads to']
    negation_words = ['no', 'not', 'none', 'zero', 'never', 'nothing']
    physical_words = ['physics', 'physical', 'energy', 'power', 'force', 'capacity', 'limit']
    
    has_constraint = int(any(word in text_lower for word in constraint_words))
    has_future_event = int(any(word in text_lower for word in future_words))
    has_number = int(bool(re.search(r'\d', context_text)))
    has_temporal = int(any(word in text_lower for word in temporal_words))
    has_causal = int(any(word in text_lower for word in causal_words))
    has_negation = int(any(word in text_lower for word in negation_words))
    
    return {
        'context_char_length': char_length,
        'context_word_count': word_count,
        'context_sentence_count': sentence_count,
        'avg_word_length': avg_word_length,
        'has_constraint': has_constraint,
        'has_future_event': has_future_event,
        'has_number': has_number,
        'has_temporal': has_temporal,
        'has_causal': has_causal,
        'has_negation': has_negation,
        'mentions_constraint': has_constraint,
        'mentions_event': has_future_event,
        'mentions_physical': int(any(word in text_lower for word in physical_words))
    }


def load_context_features(results_dir='results'):
    """
    Load contexts from all domain folders and extract features.
    """
    print("\n" + "=" * 80)
    print("TASK 2: Extracting Context Features")
    print("=" * 80)
    
    all_contexts = {}
    domain_folders = [d for d in os.listdir(results_dir) 
                     if os.path.isdir(os.path.join(results_dir, d))]
    
    for domain in domain_folders:
        contexts_path = os.path.join(results_dir, domain, 'datasets', 'contexts.json')
        
        if not os.path.exists(contexts_path):
            print(f"  WARNING: No contexts found for {domain}")
            continue
        
        # Load contexts for this domain
        with open(contexts_path, 'r') as f:
            domain_contexts = json.load(f)
        
        all_contexts.update(domain_contexts)
        print(f"✓ Loaded {len(domain_contexts)} contexts from {domain}")
    
    # Extract features for each context
    context_features = []
    for task_id, context_text in all_contexts.items():
        features = extract_context_features(context_text)
        features['task_id'] = task_id  # Use task_id to match with main dataframe
        context_features.append(features)
    
    df_context = pd.DataFrame(context_features)
    print(f"\n✓ Total context features extracted: {len(df_context)} tasks")
    
    # Show what features were extracted
    feature_cols = [col for col in df_context.columns if col != 'task_id']
    print(f"✓ Context feature columns: {feature_cols}")
    
    return df_context

def encode_domain(df):
    """
    Encode domain as numeric using LabelEncoder.
    """
    print("\n" + "=" * 80)
    print("TASK 3: Encoding Domain")
    print("=" * 80)
    
    le = LabelEncoder()
    df['domain_encoded'] = le.fit_transform(df['domain'])
    
    print(f"✓ Encoded {len(le.classes_)} unique domains")
    print(f"✓ Domain mapping:")
    for i, domain in enumerate(le.classes_):
        count = (df['domain'] == domain).sum()
        print(f"   {domain}: {i} ({count} tasks)")
    
    return df, le

def define_targets(df):
    """
    Create classification targets based on NMAE and DA metrics.
    
    Target 1 (PRIMARY): mistral_beats_baseline (NMAE)
    Target 2 (SECONDARY): mistral_beats_baseline_da (DA)
    Target 3 (OPTIONAL): mistral_beats_baseline_both (NMAE + DA)
    """
    print("\n" + "=" * 80)
    print("TASK 4: Defining Classification Targets")
    print("=" * 80)
    
    # Target 1: NMAE-based (already in data)
    target1_col = 'mistral_beats_baseline'
    if target1_col not in df.columns:
        print(" WARNING: 'mistral_beats_baseline' not in data, creating it")
        df[target1_col] = (df['mistral_nmae'] < df['best_baseline_nmae']).astype(int)
    
    positive_pct = df[target1_col].mean() * 100
    print(f"✓ Target 1 (NMAE): {target1_col}")
    print(f"   Positive class: {df[target1_col].sum()} ({positive_pct:.1f}%)")
    print(f"   Negative class: {(df[target1_col] == 0).sum()} ({100-positive_pct:.1f}%)")
    
    # Target 2: DA-based
    if 'mistral_beats_baseline_da' not in df.columns:
        df['mistral_beats_baseline_da'] = (df['mistral_da'] > df['best_baseline_da']).astype(int)
    
    positive_pct_da = df['mistral_beats_baseline_da'].mean() * 100
    print(f"\n✓ Target 2 (DA): mistral_beats_baseline_da")
    print(f"   Positive class: {df['mistral_beats_baseline_da'].sum()} ({positive_pct_da:.1f}%)")
    
    # Target 3: Combined (both NMAE and DA)
    if 'mistral_beats_baseline_both' not in df.columns:
        df['mistral_beats_baseline_both'] = (
            (df['mistral_beats_baseline'] == 1) & 
            (df['mistral_beats_baseline_da'] == 1)
        ).astype(int)
    
    positive_pct_both = df['mistral_beats_baseline_both'].mean() * 100
    print(f"\n✓ Target 3 (Both): mistral_beats_baseline_both")
    print(f"   Positive class: {df['mistral_beats_baseline_both'].sum()} ({positive_pct_both:.1f}%)")
    
    return df

def define_features(df):
    """
    Define feature columns to use for classification.
    
    Includes:
    - Time series properties (whatever is available in metadata)
    - Context properties (extracted from text)
    - Baseline performance metrics
    - Domain encoding
    
    Excludes (to avoid data leakage):
    - LLM results (mistral_nmae, etc.)
    - Improvement metrics
    - Best model indicators
    """
    print("\n" + "=" * 80)
    print("TASK 5: Defining Feature Set")
    print("=" * 80)
    
    # Time series features - flexible list based on what's actually in data
    potential_ts_features = [
        'mean', 'std', 'min', 'max',
        'history_length', 'future_length',
        'trend', 'seasonality_strength', 'volatility',
        'acf_1', 'pacf_1', 'seed'
    ]
    
    ts_features = [f for f in potential_ts_features if f in df.columns]
    
    # Context features
    potential_context_features = [
        'context_char_length', 'context_word_count', 'context_sentence_count',
        'avg_word_length', 'has_constraint', 'has_future_event', 'has_number',
        'has_temporal', 'has_causal', 'has_negation', 'mentions_constraint',
        'mentions_event', 'mentions_physical', 'context_length'
    ]
    
    context_features = [f for f in potential_context_features if f in df.columns]
    
    # Baseline performance features
    potential_baseline_features = [
        'arima_nmae', 'arima_da',
        'ets_nmae', 'ets_da',
        'best_baseline_nmae', 'best_baseline_da'
    ]
    
    baseline_features = [f for f in potential_baseline_features if f in df.columns]
    
    # Domain encoding
    domain_features = ['domain_encoded'] if 'domain_encoded' in df.columns else []
    
    # Combine all features
    all_features = ts_features + context_features + baseline_features + domain_features
    
    # Remove any duplicates
    all_features = list(dict.fromkeys(all_features))
    
    print(f"✓ Total features defined: {len(all_features)}")
    print(f"\n✓ Feature breakdown:")
    print(f"   Time series: {len(ts_features)} - {ts_features}")
    print(f"   Context: {len(context_features)} - {context_features[:5]}{'...' if len(context_features) > 5 else ''}")
    print(f"   Baseline: {len(baseline_features)} - {baseline_features}")
    print(f"   Domain: {len(domain_features)}")
    
    if len(all_features) < 10:
        print(f"\n  WARNING: Only {len(all_features)} features found")
        print(f"   This may limit classifier performance")
        print(f"   Available columns: {list(df.columns)[:20]}...")
    
    return all_features


def train_classifier(df, feature_cols, target_col, test_size=0.2, random_state=42):
    """
    Split data and train XGBoost classifier.
    """
    print("\n" + "=" * 80)
    print("TASK 6 & 7: Train/Test Split and Model Training")
    print("=" * 80)
    
    # Prepare features and target
    X = df[feature_cols].fillna(0)  # Handle any missing values
    y = df[target_col]
    
    print(f"✓ Feature matrix shape: {X.shape}")
    print(f"✓ Target distribution:")
    print(f"   Positive: {y.sum()} ({y.mean():.1%})")
    print(f"   Negative: {(y == 0).sum()} ({(1-y.mean()):.1%})")
    
    # Train/test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )
    
    print(f"\n✓ Split sizes:")
    print(f"   Training: {len(X_train)} tasks")
    print(f"   Test: {len(X_test)} tasks")
    print(f"\n✓ Class balance preserved:")
    print(f"   Train positive: {y_train.mean():.1%}")
    print(f"   Test positive: {y_test.mean():.1%}")
    
    # Train XGBoost classifier with class weight adjustment
    print(f"\n✓ Training XGBoost classifier...")
    
    # Calculate scale_pos_weight to handle class imbalance
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"   Class imbalance ratio: {scale_pos_weight:.2f}:1 (negative:positive)")
    print(f"   Using scale_pos_weight={scale_pos_weight:.2f} to boost positive class")
    
    clf = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,  # Handle class imbalance
        random_state=random_state,
        eval_metric='logloss'
    )
    
    clf.fit(X_train, y_train)
    print(f"✓ Model trained successfully")
    
    return clf, X_train, X_test, y_train, y_test


def optimize_threshold(clf, X_train, y_train, X_test, y_test, metric='f1'):
    """
    Find optimal decision threshold that maximizes performance.
    """
    print("\n" + "=" * 80)
    print("OPTIMIZING DECISION THRESHOLD")
    print("=" * 80)
    
    # Get probabilities
    train_proba = clf.predict_proba(X_train)[:, 1]
    test_proba = clf.predict_proba(X_test)[:, 1]
    
    # Try different thresholds
    thresholds = np.arange(0.1, 0.9, 0.05)
    best_threshold = 0.5
    best_score = 0
    results = []
    
    print(f"\n✓ Searching thresholds from 0.10 to 0.85...")
    print(f"   Optimizing for: {metric}")
    
    for threshold in thresholds:
        y_pred_train = (train_proba >= threshold).astype(int)
        
        # Compute metric on training set
        if metric == 'f1':
            score = f1_score(y_train, y_pred_train, zero_division=0)
        elif metric == 'accuracy':
            score = accuracy_score(y_train, y_pred_train)
        elif metric == 'balanced_accuracy':
            from sklearn.metrics import balanced_accuracy_score
            score = balanced_accuracy_score(y_train, y_pred_train)
        else:  # Default to F1
            score = f1_score(y_train, y_pred_train, zero_division=0)
        
        results.append({
            'threshold': threshold,
            'train_score': score,
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'train_f1': f1_score(y_train, y_pred_train, zero_division=0)
        })
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    print(f"\n✓ Optimal threshold: {best_threshold:.2f}")
    print(f"   Training {metric}: {best_score:.3f}")
    
    # Evaluate on test set with optimal threshold
    y_pred_test = (test_proba >= best_threshold).astype(int)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test, zero_division=0)
    test_precision = precision_score(y_test, y_pred_test, zero_division=0)
    test_recall = recall_score(y_test, y_pred_test, zero_division=0)
    
    print(f"\n✓ Test set performance at threshold={best_threshold:.2f}:")
    print(f"   Accuracy:  {test_accuracy:.3f}")
    print(f"   Precision: {test_precision:.3f}")
    print(f"   Recall:    {test_recall:.3f}")
    print(f"   F1 Score:  {test_f1:.3f}")
    
    # Compare to default threshold
    y_pred_default = (test_proba >= 0.5).astype(int)
    default_f1 = f1_score(y_test, y_pred_default, zero_division=0)
    improvement = (test_f1 - default_f1) / default_f1 * 100 if default_f1 > 0 else 0
    
    print(f"\n✓ Improvement over default (0.5) threshold:")
    print(f"   Default F1: {default_f1:.3f}")
    print(f"   Optimized F1: {test_f1:.3f}")
    print(f"   Gain: {improvement:+.1f}%")
    
    return best_threshold


def evaluate_classifier(clf, X_test, y_test, threshold=0.5):
    """
    Evaluate classifier and compute metrics.
    """
    print("\n" + "=" * 80)
    print("TASK 8: Evaluating Classifier")
    print("=" * 80)
    
    print(f"Using decision threshold: {threshold:.2f}")
    
    # Get predictions with custom threshold
    y_pred_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Compute metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_pred_proba),
        'threshold': threshold
    }
    
    print(f"\n✓ Classification Metrics:")
    print(f"   Accuracy:  {metrics['accuracy']:.3f}")
    print(f"   Precision: {metrics['precision']:.3f}")
    print(f"   Recall:    {metrics['recall']:.3f}")
    print(f"   F1 Score:  {metrics['f1']:.3f}")
    print(f"   ROC-AUC:   {metrics['roc_auc']:.3f}")
    
    print(f"\n✓ Classification Report:")
    print(classification_report(y_test, y_pred, 
                               target_names=['Baseline Wins', 'Mistral Wins']))
    
    return metrics, y_pred, y_pred_proba


def plot_roc_curve(y_test, y_pred_proba, output_dir='results/classifier_outputs'):
    """Generate publication-quality ROC curve with smooth styling."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import roc_curve, roc_auc_score
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Set publication style
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    fig, ax = plt.subplots(figsize=(9, 8))
    
    # Main ROC curve - smooth and professional
    ax.plot(fpr, tpr, color='#0066CC', linewidth=3.5, 
            label=f'XGBoost Selector (AUC = {auc:.3f})', zorder=3,
            solid_capstyle='round')
    
    # Diagonal reference line
    ax.plot([0, 1], [0, 1], color='#666666', linewidth=2.5, 
            linestyle='--', alpha=0.6,
            label='Random Classifier (AUC = 0.500)', zorder=2)
    
    # Fill area under ROC curve with gradient effect
    ax.fill_between(fpr, tpr, alpha=0.2, color='#0066CC', zorder=1)
    
    # Axes limits with small padding
    ax.set_xlim([-0.01, 1.01])
    ax.set_ylim([-0.01, 1.01])
    
    # Labels with professional styling
    ax.set_xlabel('False Positive Rate', fontsize=16, fontweight='600', 
                  labelpad=12)
    ax.set_ylabel('True Positive Rate', fontsize=16, fontweight='600',
                  labelpad=12)
    ax.set_title('ROC Curve: Selective LLM Deployment', 
                 fontsize=18, fontweight='700', pad=20)
    
    # Grid styling - subtle and professional
    ax.grid(True, alpha=0.15, linestyle='-', linewidth=0.8, color='#CCCCCC')
    ax.set_axisbelow(True)
    
    # Legend with professional styling
    legend = ax.legend(loc='lower right', fontsize=13, frameon=True, 
                       fancybox=True, shadow=False, 
                       framealpha=0.95, edgecolor='#CCCCCC')
    legend.get_frame().set_linewidth(1.5)
    
    # Tick styling
    ax.tick_params(axis='both', which='major', labelsize=13, 
                   length=6, width=1.2, direction='out',
                   color='#333333')
    
    # Make ticks appear at nice intervals
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    
    # Spine styling
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('#333333')
    
    # Add subtle background color
    ax.set_facecolor('#FAFAFA')
    fig.patch.set_facecolor('white')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'roc_curve_nmae.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"✓ Saved ROC curve to {output_path}")


def plot_feature_importance(clf, feature_cols, output_dir='results/classifier_outputs', top_n=11):
    """Generate publication-quality feature importance plot."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Set publication style
    plt.style.use('seaborn-v0_8-paper')
    
    importance = clf.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_cols,
        'Importance': importance
    }).sort_values('Importance', ascending=False)
    
    # Save to CSV
    csv_path = os.path.join(output_dir, 'feature_importance_nmae.csv')
    feature_importance_df.to_csv(csv_path, index=False)
    print(f"✓ Saved feature importance to {csv_path}")
    
    # Plot top N features
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.set_style("whitegrid", {'axes.facecolor': 'white'})
    top_features = feature_importance_df.head(top_n)

    # Add full border
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(1.3)
        spine.set_edgecolor("black")
    
    # Create color gradient (darker = more important)
    # Custom hex color list (your palette)
    hex_colors = [
        "#003f5c", "#2f4b7c", "#665191", "#a05195",
        "#d45087", "#f95d6a", "#ff7c43", "#ffa600"
    ]

    # Build a custom gradient colormap
    from matplotlib.colors import LinearSegmentedColormap

    custom_cmap = LinearSegmentedColormap.from_list("custom", hex_colors, N=256)

    # Sample smoothly across the gradient for however many bars we have
    colors = custom_cmap(np.linspace(0.05, 0.95, len(top_features)))[::-1]
    
    # Horizontal bar chart
    bars = ax.barh(range(len(top_features)), top_features['Importance'], 
                   color=colors, edgecolor='black', linewidth=0.8, alpha=0.85)
    
    # Feature names on y-axis
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(top_features['Feature'], fontsize=11)
    
    # Labels and title
    ax.set_xlabel('Feature Importance (Gain)', fontsize=14, fontweight='bold')
    ax.set_title(f'Top {top_n} Most Predictive Features for LLM Selection', 
                 fontsize=16, fontweight='bold', pad=20)
    
    # Grid
    ax.grid(True, axis='x', alpha=0.25, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)
    
    # Add value labels on bars
    for i, (bar, val) in enumerate(zip(bars, top_features['Importance'])):
        ax.text(val + 0.002, i, f'{val:.3f}', 
                va='center', fontsize=9, fontweight='bold')
    
    # Invert y-axis so highest importance is on top
    ax.invert_yaxis()
    
    # Tick styling
    ax.tick_params(axis='both', which='major', labelsize=10)
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'feature_importance_nmae.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    
    print(f"✓ Saved feature importance plot to {output_path}")
    
    return feature_importance_df


def generate_visualizations(clf, X_test, y_test, y_pred, y_pred_proba, feature_cols, 
                          output_dir='results/classifier_outputs'):
    """Generate all visualization plots (publication quality)."""
    print("\n" + "=" * 80)
    print("TASK 9: Generating Visualizations")
    print("=" * 80)
    
    # Only generate ROC curve and feature importance
    plot_roc_curve(y_test, y_pred_proba, output_dir)
    feature_importance_df = plot_feature_importance(clf, feature_cols, output_dir)
    
    return feature_importance_df

def evaluate_policies(df_full, clf, feature_cols, threshold=0.5, output_dir='results/classifier_outputs'):
    """
    Compare 4 policies:
    1. Always-Baseline: Always use best baseline (ARIMA or ETS)
    2. Always-LLM: Always use Mistral with context
    3. XGBoost Selector: Use classifier to decide
    4. Oracle: Hindsight optimal (always pick best model)
    """
    print("\n" + "=" * 80)
    print("TASK 10: Policy Comparison")
    print("=" * 80)
    
    print(f"Using decision threshold: {threshold:.2f}")
    
    # Get classifier predictions for all tasks with custom threshold
    X_all = df_full[feature_cols].fillna(0)
    selector_proba = clf.predict_proba(X_all)[:, 1]
    selector_predictions = (selector_proba >= threshold).astype(int)
    
    # Add predictions to dataframe
    df_full['selector_prediction'] = selector_predictions
    df_full['selector_proba'] = selector_proba
    
    # Policy 1: Always use baseline
    df_full['policy1_nmae'] = df_full['best_baseline_nmae']
    policy1_nmae = df_full['policy1_nmae'].mean()
    policy1_llm_usage = 0.0
    
    # Policy 2: Always use Mistral
    df_full['policy2_nmae'] = df_full['mistral_nmae']
    policy2_nmae = df_full['policy2_nmae'].mean()
    policy2_llm_usage = 1.0
    
    # Policy 3: Use XGBoost selector
    df_full['policy3_nmae'] = np.where(
        df_full['selector_prediction'] == 1,
        df_full['mistral_nmae'],
        df_full['best_baseline_nmae']
    )
    policy3_nmae = df_full['policy3_nmae'].mean()
    policy3_llm_usage = df_full['selector_prediction'].mean()
    
    # Policy 4: Oracle (always pick best)
    df_full['policy4_nmae'] = np.minimum(
        df_full['mistral_nmae'],
        df_full['best_baseline_nmae']
    )
    policy4_nmae = df_full['policy4_nmae'].mean()
    policy4_llm_usage = df_full['mistral_beats_baseline'].mean()
    
    # Create comparison table
    policy_df = pd.DataFrame({
        'Policy': ['Always-Baseline', 'Always-LLM', 'XGBoost Selector', 'Oracle'],
        'Mean NMAE': [policy1_nmae, policy2_nmae, policy3_nmae, policy4_nmae],
        'LLM Usage (%)': [policy1_llm_usage * 100, policy2_llm_usage * 100, 
                         policy3_llm_usage * 100, policy4_llm_usage * 100]
    })
    
    # Calculate improvement metrics
    policy_df['vs Baseline (%)'] = (
        (policy_df['Mean NMAE'] - policy1_nmae) / policy1_nmae * 100
    )
    
    policy_df['Oracle Captured (%)'] = (
        (policy1_nmae - policy_df['Mean NMAE']) / (policy1_nmae - policy4_nmae) * 100
    )
    
    # Save results
    csv_path = os.path.join(output_dir, 'policy_comparison_nmae.csv')
    policy_df.to_csv(csv_path, index=False)
    print(f"✓ Saved policy comparison to {csv_path}")
    
    print(f"\n✓ Policy Comparison Results:")
    print(policy_df.to_string(index=False))
    
    # Visualize policy comparison
    plot_policy_comparison(policy_df, output_dir)
    
    return policy_df, df_full


def plot_policy_comparison(policy_df, output_dir='results/classifier_outputs'):
    """Generate publication-quality policy comparison visualization."""
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    # Set publication style
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Define professional color scheme
    colors = ['#003f5c', '#7a5195', '#ef5675', '#ffa600']  # Dark Blue, Purple, Pink, Orange
    
    # Bar chart
    x_pos = np.arange(len(policy_df))
    bars = ax.bar(x_pos, policy_df['Mean NMAE'], color=colors, 
                  edgecolor='black', linewidth=1.5, alpha=0.88, width=0.65)
    
    # Add NMAE value labels on top of bars
    for i, (bar, val) in enumerate(zip(bars, policy_df['Mean NMAE'])):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.015,
                f'{val:.3f}',
                ha='center', va='bottom', fontsize=14, fontweight='700')
    
    # Add LLM usage percentage INSIDE bars (near top)
    for i, (bar, usage) in enumerate(zip(bars, policy_df['LLM Usage (%)'])):
        height = bar.get_height()
        # Position text inside bar, near the top
        y_pos = height * 0.85 if height > 0.3 else height * 0.5
        
        ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{usage:.1f}%\nLLM Usage',
                ha='center', va='center', fontsize=11, fontweight='600',
                color='white', 
                bbox=dict(boxstyle='round,pad=0.4', 
                         facecolor='black', alpha=0.6, edgecolor='none'))
    
    # Labels and title
    ax.set_ylabel('Mean NMAE (lower is better)', fontsize=16, fontweight='600', labelpad=12)
    ax.set_xlabel('Deployment Policy', fontsize=16, fontweight='600', labelpad=12)
    ax.set_title('Policy Comparison: Forecast Accuracy vs. Computational Cost', 
                 fontsize=18, fontweight='700', pad=20)
    
    # X-axis
    ax.set_xticks(x_pos)
    ax.set_xticklabels(policy_df['Policy'], fontsize=13, fontweight='600')
    
    # Grid - horizontal only, subtle
    ax.grid(True, alpha=0.2, axis='y', linestyle='-', linewidth=0.9, color='#CCCCCC')
    ax.set_axisbelow(True)
    
    # Add horizontal line at baseline for reference
    baseline_nmae = policy_df.loc[0, 'Mean NMAE']
    ax.axhline(y=baseline_nmae, color='#333333', linestyle=':', 
               linewidth=2, alpha=0.6, label='Baseline Performance', zorder=1)
    
    # Legend with professional styling
    legend = ax.legend(loc='upper right', fontsize=12, frameon=True, 
                      fancybox=True, shadow=False, framealpha=0.95,
                      edgecolor='#CCCCCC')
    legend.get_frame().set_linewidth(1.5)
    
    # Tick styling
    ax.tick_params(axis='both', which='major', labelsize=12,
                   length=6, width=1.2, color='#333333')
    
    # Spine styling
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('#333333')
    
    # Add padding to y-axis for cleaner look
    y_min, y_max = ax.get_ylim()
    ax.set_ylim(0, y_max + 0.12)
    
    # Subtle background
    ax.set_facecolor('#FAFAFA')
    fig.patch.set_facecolor('white')
    
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, 'policy_comparison_nmae.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.close()
    
    print(f"✓ Saved policy comparison plot to {output_path}")

def per_domain_analysis(df_full, output_dir='results/classifier_outputs'):
    """Analyze selector performance by domain."""
    print("\n" + "=" * 80)
    print("TASK 11: Per-Domain Analysis")
    print("=" * 80)
    
    domain_analysis = df_full.groupby('domain').agg({
        'best_baseline_nmae': 'mean',
        'mistral_nmae': 'mean',
        'policy3_nmae': 'mean',  # Selector
        'policy4_nmae': 'mean',  # Oracle
        'mistral_beats_baseline': 'mean'
    }).round(4)
    
    domain_analysis.columns = ['Baseline', 'Mistral', 'Selector', 'Oracle', 'Mistral Win Rate']
    domain_analysis['Selector Gap to Oracle'] = (
        domain_analysis['Selector'] - domain_analysis['Oracle']
    ).round(4)
    
    # Save to CSV
    csv_path = os.path.join(output_dir, 'per_domain_analysis.csv')
    domain_analysis.to_csv(csv_path)
    print(f"✓ Saved per-domain analysis to {csv_path}")
    
    print(f"\n✓ Per-Domain Results:")
    print(domain_analysis.to_string())
    
    return domain_analysis

def generate_summary_report(metrics, policy_df, feature_importance_df, y_test, 
                          output_dir='results/classifier_outputs'):
    """Generate comprehensive summary report."""
    print("\n" + "=" * 80)
    print("TASK 12: Generating Summary Report")
    print("=" * 80)
    
    # Extract key values
    policy1_nmae = policy_df.loc[0, 'Mean NMAE']
    policy2_nmae = policy_df.loc[1, 'Mean NMAE']
    policy3_nmae = policy_df.loc[2, 'Mean NMAE']
    policy4_nmae = policy_df.loc[3, 'Mean NMAE']
    
    policy3_llm_usage = policy_df.loc[2, 'LLM Usage (%)'] / 100
    policy4_llm_usage = policy_df.loc[3, 'LLM Usage (%)'] / 100
    
    oracle_captured = policy_df.loc[2, 'Oracle Captured (%)']
    
    report = f"""
================================================================================
XGBOOST CLASSIFIER SUMMARY: SELECTIVE LLM USE
================================================================================

MODEL CONFIGURATION
--------------------------------------------------------------------------------
Decision Threshold: {metrics.get('threshold', 0.5):.2f} (optimized from default 0.5)
Class Weight: scale_pos_weight applied to handle imbalance
Features: {len(feature_importance_df)} total

CLASSIFICATION PERFORMANCE (NMAE Target)
--------------------------------------------------------------------------------
Test Set Size: {len(y_test)}
Accuracy:      {metrics['accuracy']:.3f}
Precision:     {metrics['precision']:.3f}
Recall:        {metrics['recall']:.3f}
F1 Score:      {metrics['f1']:.3f}
ROC-AUC:       {metrics['roc_auc']:.3f}

Class Balance:
  Positive (Mistral wins): {y_test.sum()} ({y_test.mean():.1%})
  Negative (Baseline wins): {len(y_test) - y_test.sum()} ({1-y_test.mean():.1%})

POLICY COMPARISON
--------------------------------------------------------------------------------
{policy_df.to_string(index=False)}

KEY INSIGHTS
--------------------------------------------------------------------------------
1. Selector vs Always-Baseline: {policy1_nmae - policy3_nmae:.4f} NMAE improvement
   ({(policy1_nmae - policy3_nmae) / policy1_nmae * 100:.1f}% better)

2. Selector vs Always-LLM: Uses LLM on {policy3_llm_usage:.1%} of tasks
   (saves {(1 - policy3_llm_usage) * 100:.1f}% of computational cost)

3. Oracle Benefit Captured: {oracle_captured:.1f}%
   (achieves {oracle_captured:.1f}% of theoretically optimal improvement)

4. LLM Usage Rate: {policy3_llm_usage:.1%}
   Oracle Rate:    {policy4_llm_usage:.1%}
   Match:          {'Yes' if abs(policy3_llm_usage - policy4_llm_usage) < 0.05 else 'No'}

TOP 5 MOST IMPORTANT FEATURES
--------------------------------------------------------------------------------
{feature_importance_df.head(5).to_string(index=False)}

RECOMMENDATION
--------------------------------------------------------------------------------
"""
    
    if policy3_nmae < policy1_nmae:
        improvement = (policy1_nmae - policy3_nmae) / policy1_nmae * 100
        report += f"✓ Selector provides {improvement:.1f}% improvement over baseline\n"
        report += f"✓ Uses LLM on only {policy3_llm_usage:.1%} of tasks\n"
        report += f"✓ Captures {oracle_captured:.1f}% of oracle benefit\n"
        report += "\nRECOMMENDATION: Use selector for deployment\n"
    else:
        report += "✗ Selector does not improve over always using baseline\n"
        report += "\nRECOMMENDATION: Stick with baseline methods\n"
    
    report += """
================================================================================
"""
    
    # Save report
    report_path = os.path.join(output_dir, 'classifier_summary.txt')
    with open(report_path, 'w') as f:
        f.write(report)
    
    print(f"✓ Saved summary report to {report_path}")
    print(report)
    
    return report

def main(random_seed=42):
    """Main execution function."""
    print("\n" + "=" * 80)
    print(" XGBOOST CLASSIFIER: SELECTIVE LLM USE FOR TIME SERIES FORECASTING")
    print("=" * 80)
    print("\nAuthor: Kazi Ashab Rahman")
    print("Course: COMP 545 - Advanced Machine Learning")
    print("Project: Selective Context Use for Cost-Effective LLM Forecasting")
    print("=" * 80)
    
    # Configuration
    RESULTS_DIR = 'results'
    OUTPUT_DIR = 'results/classifier_outputs'
    COMPILED_CSV = os.path.join(RESULTS_DIR, 'compiled_comparison.csv')
    TARGET_COL = 'mistral_beats_baseline'
    RANDOM_STATE = random_seed
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load main results
    print(f"\n✓ Loading compiled comparison from {COMPILED_CSV}")
    df = pd.read_csv(COMPILED_CSV)
    print(f"✓ Loaded {len(df)} tasks")
    
    # Standardize column names: 'id' -> 'task_id' if needed
    if 'id' in df.columns and 'task_id' not in df.columns:
        df['task_id'] = df['id']
        print(f"✓ Standardized 'id' column to 'task_id'")
    
    # Task 1: Load time series metadata
    df_metadata = load_time_series_metadata(RESULTS_DIR)
    
    # Task 2: Extract context features
    df_context = load_context_features(RESULTS_DIR)
    
    # Merge everything
    print("\n" + "=" * 80)
    print("Merging All Data")
    print("=" * 80)
    
    print(f"Main df shape: {df.shape}, columns: {list(df.columns)[:10]}...")
    print(f"Metadata shape: {df_metadata.shape}, columns: {list(df_metadata.columns)[:10]}...")
    print(f"Context shape: {df_context.shape}, columns: {list(df_context.columns)[:10]}...")
    
    # Merge metadata
    df = df.merge(df_metadata, on=['task_id', 'domain'], how='left', suffixes=('', '_meta'))
    print(f"✓ After merging metadata: {df.shape}")
    
    # Check for missing values after metadata merge
    missing_after_meta = df['mean'].isna().sum() if 'mean' in df.columns else len(df)
    if missing_after_meta > 0:
        print(f"  WARNING: {missing_after_meta} tasks have no metadata")
    
    # Merge context
    df = df.merge(df_context, on='task_id', how='left', suffixes=('', '_ctx'))
    print(f"✓ After merging context: {df.shape}")
    
    # Check for missing values after context merge
    missing_after_ctx = df['context_char_length'].isna().sum() if 'context_char_length' in df.columns else len(df)
    if missing_after_ctx > 0:
        print(f" WARNING: {missing_after_ctx} tasks have no context features")
    
    print(f"\n✓ Final dataframe shape: {df.shape}")
    print(f"✓ Columns: {len(df.columns)}")
    
    # Task 3: Encode domain
    df, label_encoder = encode_domain(df)
    
    # Task 4: Define targets
    df = define_targets(df)
    
    # Task 5: Define features
    feature_cols = define_features(df)
    
    # Task 6 & 7: Train/test split and train model
    clf, X_train, X_test, y_train, y_test = train_classifier(
        df, feature_cols, TARGET_COL, random_state=RANDOM_STATE
    )
    
    # Optimize decision threshold
    optimal_threshold = optimize_threshold(clf, X_train, y_train, X_test, y_test, metric='f1')
    
    # Task 8: Evaluate classifier with optimized threshold
    metrics, y_pred, y_pred_proba = evaluate_classifier(clf, X_test, y_test, threshold=optimal_threshold)
    
    # Task 9: Generate visualizations
    feature_importance_df = generate_visualizations(
        clf, X_test, y_test, y_pred, y_pred_proba, feature_cols, OUTPUT_DIR
    )
    
    # Task 10: Policy comparison with optimized threshold
    policy_df, df_full = evaluate_policies(df, clf, feature_cols, threshold=optimal_threshold, output_dir=OUTPUT_DIR)
    
    # Task 11: Per-domain analysis
    domain_analysis = per_domain_analysis(df_full, OUTPUT_DIR)
    
    # Task 12: Generate summary report
    report = generate_summary_report(
        metrics, policy_df, feature_importance_df, y_test, OUTPUT_DIR
    )
    
    print("\n" + "=" * 80)
    print(" CLASSIFIER BUILD COMPLETE!")
    print("=" * 80)
    print(f"\n✓ All outputs saved to: {OUTPUT_DIR}")
    print(f"\nGenerated files:")
    print(f"  - confusion_matrix_nmae.png")
    print(f"  - roc_curve_nmae.png")
    print(f"  - pr_curve_nmae.png")
    print(f"  - feature_importance_nmae.png")
    print(f"  - feature_importance_nmae.csv")
    print(f"  - policy_comparison_nmae.png")
    print(f"  - policy_comparison_nmae.csv")
    print(f"  - per_domain_analysis.csv")
    print(f"  - classifier_summary.txt")
    
    return clf, df_full, metrics, policy_df, feature_importance_df

if __name__ == "__main__":
    # Run 5 times with different seeds
    seeds = [42, 123, 456, 789, 1011]
    all_results = []
    
    print("\n" + "=" * 80)
    print(" RUNNING 5 ITERATIONS WITH DIFFERENT RANDOM SEEDS")
    print("=" * 80)
    
    for i, seed in enumerate(seeds, 1):
        print(f"\n{'='*80}")
        print(f" ITERATION {i}/5 - Random Seed: {seed}")
        print(f"{'='*80}\n")
        
        # Modify the RANDOM_STATE in main()
        clf, df_full, metrics, policy_df, feature_importance_df = main(random_seed=seed)
        
        all_results.append({
            'seed': seed,
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1'],
            'roc_auc': metrics['roc_auc'],
            'selector_nmae': policy_df.loc[2, 'Mean NMAE'],
            'oracle_captured': policy_df.loc[2, 'Oracle Captured (%)']
        })
    
    # Compute statistics across runs
    results_df = pd.DataFrame(all_results)
    
    # Mean and Std
    acc_mean, acc_std = results_df['accuracy'].mean(), results_df['accuracy'].std()
    prec_mean, prec_std = results_df['precision'].mean(), results_df['precision'].std()
    rec_mean, rec_std = results_df['recall'].mean(), results_df['recall'].std()
    f1_mean, f1_std = results_df['f1'].mean(), results_df['f1'].std()
    auc_mean, auc_std = results_df['roc_auc'].mean(), results_df['roc_auc'].std()
    nmae_mean, nmae_std = results_df['selector_nmae'].mean(), results_df['selector_nmae'].std()
    oracle_mean, oracle_std = results_df['oracle_captured'].mean(), results_df['oracle_captured'].std()
    
    # Add mean ± std summary row
    results_df.loc[len(results_df)] = {
        'seed': 'mean ± std',
        'accuracy': f"{acc_mean:.3f} ± {acc_std:.3f}",
        'precision': f"{prec_mean:.3f} ± {prec_std:.3f}",
        'recall': f"{rec_mean:.3f} ± {rec_std:.3f}",
        'f1': f"{f1_mean:.3f} ± {f1_std:.3f}",
        'roc_auc': f"{auc_mean:.3f} ± {auc_std:.3f}",
        'selector_nmae': f"{nmae_mean:.3f} ± {nmae_std:.3f}",
        'oracle_captured': f"{oracle_mean:.1f}% ± {oracle_std:.1f}%"
    }
    
    print("\n" + "=" * 80)
    print(" FINAL RESULTS ACROSS 5 RUNS")
    print("=" * 80)
    print(f"\nAccuracy:        {acc_mean:.3f} ± {acc_std:.3f}")
    print(f"Precision:       {prec_mean:.3f} ± {prec_std:.3f}")
    print(f"Recall:          {rec_mean:.3f} ± {rec_std:.3f}")
    print(f"F1 Score:        {f1_mean:.3f} ± {f1_std:.3f}")
    print(f"ROC-AUC:         {auc_mean:.3f} ± {auc_std:.3f}")
    print(f"\nSelector NMAE:   {nmae_mean:.3f} ± {nmae_std:.3f}")
    print(f"Oracle Captured: {oracle_mean:.1f}% ± {oracle_std:.1f}%")
    
    # Save summary
    results_df.to_csv('results/classifier_outputs/multi_run_summary.csv', index=False)
    print(f"\n💾 Saved multi-run summary to results/classifier_outputs/multi_run_summary.csv")
    print("=" * 80 + "\n")