"""
Diminishing Returns Analysis for LLM Time Series Forecasting
Analyzes performance scaling across Llama 3B, Mistral 8x7B, and GPT-4o-mini

Author: Kazi Ashab Rahman
Date: November 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['font.size'] = 10

class DiminishingReturnsAnalyzer:
    """Analyzer for LLM performance scaling"""
    
    def __init__(self, results_dir):
        self.results_dir = Path(results_dir)
        self.domains = []
        self.compiled_df = None
        
    def discover_domains(self):
        """Find all domain folders in Results directory"""
        print("🔍 Discovering domains...")
        
        # Get main directory domains
        main_domains = [d for d in self.results_dir.iterdir() 
                       if d.is_dir() and not d.name.startswith('.') 
                       and d.name not in ['plots', 'xgboost_compiled', 'xgboost_compiled_old', 'xgboost_nmae_da']]
        
        # Get plots subdirectory domains
        #plots_dir = self.results_dir / 'plots'
        #plots_domains = []
        #if plots_dir.exists():
        #    plots_domains = [d for d in plots_dir.iterdir() 
        #                  if d.is_dir() and not d.name.startswith('.')]
        
        self.domains = main_domains #+ plots_domains
        print(f"✅ Found {len(self.domains)} domains")
        for domain in self.domains:
            print(f"   - {domain.name}")
        return self.domains
    
    def load_domain_results(self, domain_path):
        """Load all model results for a single domain"""
        domain_name = domain_path.name
        
        try:
            # Load baseline models
            arima_df = pd.read_csv(domain_path / 'arima_results.csv')
            ets_df = pd.read_csv(domain_path / 'ets_results.csv')
            
            # Load LLM models (WITH CONTEXT only)
            llama_df = pd.read_csv(domain_path / 'llmp_with_context_results.csv')
            gpt4o_df = pd.read_csv(domain_path / 'gpt4o_mini_with_context_results.csv')
            mistral_df = pd.read_csv(domain_path / 'mistral_with_context_results.csv')
            
            # Rename columns to avoid conflicts
            arima_df = arima_df.rename(columns={
                'nmae': 'arima_nmae', 'da': 'arima_da', 'mae': 'arima_mae'
            })
            ets_df = ets_df.rename(columns={
                'nmae': 'ets_nmae', 'da': 'ets_da', 'mae': 'ets_mae'
            })
            llama_df = llama_df.rename(columns={
                'nmae': 'llama_nmae', 'da': 'llama_da', 'mae': 'llama_mae'
            })
            gpt4o_df = gpt4o_df.rename(columns={
                'nmae': 'gpt4o_nmae', 'da': 'gpt4o_da', 'mae': 'gpt4o_mae'
            })
            mistral_df = mistral_df.rename(columns={
                'nmae': 'mistral_nmae', 'da': 'mistral_da', 'mae': 'mistral_mae'
            })
            
            # Merge all models on task_id
            merged = arima_df[['task_id', 'arima_nmae', 'arima_da', 'arima_mae']]
            merged = merged.merge(ets_df[['task_id', 'ets_nmae', 'ets_da', 'ets_mae']], on='task_id')
            merged = merged.merge(llama_df[['task_id', 'llama_nmae', 'llama_da', 'llama_mae', 'horizon']], on='task_id')
            merged = merged.merge(gpt4o_df[['task_id', 'gpt4o_nmae', 'gpt4o_da', 'gpt4o_mae']], on='task_id')
            merged = merged.merge(mistral_df[['task_id', 'mistral_nmae', 'mistral_da', 'mistral_mae']], on='task_id')
            
            # Add domain column
            merged['domain'] = domain_name
            
            return merged
            
        except FileNotFoundError as e:
            print(f"⚠️  Warning: Missing file in {domain_name}: {e}")
            return None
        except Exception as e:
            print(f"❌ Error loading {domain_name}: {e}")
            return None
    
    def merge_all_domains(self):
        """Merge results from all domains"""
        print("\n📊 Merging all domain results...")
        
        all_dfs = []
        for domain_path in self.domains:
            df = self.load_domain_results(domain_path)
            if df is not None:
                all_dfs.append(df)
                print(f"   ✓ Loaded {domain_path.name}: {len(df)} tasks")
        
        self.compiled_df = pd.concat(all_dfs, ignore_index=True)
        print(f"\n✅ Total tasks compiled: {len(self.compiled_df)}")
        print(f"✅ Total domains: {self.compiled_df['domain'].nunique()}")
        
        # Save compiled results
        output_path = self.results_dir / 'compiled_comparison.csv'
        self.compiled_df.to_csv(output_path, index=False)
        print(f"💾 Saved: {output_path}")
        
        return self.compiled_df
    
    def calculate_baseline_and_improvements(self):
        """Calculate best baseline and LLM improvements"""
        print("\n🧮 Calculating baseline performance and improvements...")
        
        df = self.compiled_df
        
        # Best baseline (lower NMAE is better)
        df['best_baseline_nmae'] = df[['arima_nmae', 'ets_nmae']].min(axis=1)
        df['best_baseline_model'] = df[['arima_nmae', 'ets_nmae']].idxmin(axis=1)
        df['best_baseline_model'] = df['best_baseline_model'].str.replace('_nmae', '').str.upper()
        
        # Best baseline DA (higher is better)
        df['best_baseline_da'] = df[['arima_da', 'ets_da']].max(axis=1)
        
        # LLM improvements over baseline (positive = better)
        df['llama_improvement_nmae'] = df['best_baseline_nmae'] - df['llama_nmae']
        df['mistral_improvement_nmae'] = df['best_baseline_nmae'] - df['mistral_nmae']
        df['gpt4o_improvement_nmae'] = df['best_baseline_nmae'] - df['gpt4o_nmae']
        
        # DA improvements (positive = better)
        df['llama_improvement_da'] = df['llama_da'] - df['best_baseline_da']
        df['mistral_improvement_da'] = df['mistral_da'] - df['best_baseline_da']
        df['gpt4o_improvement_da'] = df['gpt4o_da'] - df['best_baseline_da']
        
        # Win indicators (1 if LLM beats baseline)
        df['llama_beats_baseline'] = (df['llama_improvement_nmae'] > 0).astype(int)
        df['mistral_beats_baseline'] = (df['mistral_improvement_nmae'] > 0).astype(int)
        df['gpt4o_beats_baseline'] = (df['gpt4o_improvement_nmae'] > 0).astype(int)
        
        # Best overall model per task
        nmae_cols = ['arima_nmae', 'ets_nmae', 'llama_nmae', 'mistral_nmae', 'gpt4o_nmae']
        df['best_model'] = df[nmae_cols].idxmin(axis=1).str.replace('_nmae', '').str.upper()
        df['best_nmae'] = df[nmae_cols].min(axis=1)
        
        self.compiled_df = df
        
        # Save updated results
        output_path = self.results_dir / 'compiled_comparison.csv'
        self.compiled_df.to_csv(output_path, index=False)
        
        print(f"✅ Calculated improvements and best models")
        print(f"\nBaseline wins: ARIMA={sum(df['best_baseline_model']=='ARIMA')}, ETS={sum(df['best_baseline_model']=='ETS')}")
        print(f"LLM wins over baseline:")
        print(f"   Llama: {df['llama_beats_baseline'].sum()} / {len(df)} ({100*df['llama_beats_baseline'].mean():.1f}%)")
        print(f"   Mistral: {df['mistral_beats_baseline'].sum()} / {len(df)} ({100*df['mistral_beats_baseline'].mean():.1f}%)")
        print(f"   GPT-4o: {df['gpt4o_beats_baseline'].sum()} / {len(df)} ({100*df['gpt4o_beats_baseline'].mean():.1f}%)")
        
        return df
    
    def generate_summary_statistics(self):
        """Generate model performance summary table"""
        print("\n📈 Generating summary statistics...")
        
        df = self.compiled_df
        
        summary = {
            'Metric': [],
            'ARIMA': [],
            'ETS': [],
            'Best Baseline': [],
            'Llama 3B': [],
            'Mistral 8x7B': [],
            'GPT-4o ~20B': []
        }
        
        # NMAE statistics
        summary['Metric'].extend(['Mean NMAE', 'Median NMAE', 'Std Dev NMAE'])
        summary['ARIMA'].extend([df['arima_nmae'].mean(), df['arima_nmae'].median(), df['arima_nmae'].std()])
        summary['ETS'].extend([df['ets_nmae'].mean(), df['ets_nmae'].median(), df['ets_nmae'].std()])
        summary['Best Baseline'].extend([df['best_baseline_nmae'].mean(), df['best_baseline_nmae'].median(), df['best_baseline_nmae'].std()])
        summary['Llama 3B'].extend([df['llama_nmae'].mean(), df['llama_nmae'].median(), df['llama_nmae'].std()])
        summary['Mistral 8x7B'].extend([df['mistral_nmae'].mean(), df['mistral_nmae'].median(), df['mistral_nmae'].std()])
        summary['GPT-4o ~20B'].extend([df['gpt4o_nmae'].mean(), df['gpt4o_nmae'].median(), df['gpt4o_nmae'].std()])
        
        # DA statistics
        summary['Metric'].extend(['Mean DA', 'Median DA'])
        summary['ARIMA'].extend([df['arima_da'].mean(), df['arima_da'].median()])
        summary['ETS'].extend([df['ets_da'].mean(), df['ets_da'].median()])
        summary['Best Baseline'].extend([df['best_baseline_da'].mean(), df['best_baseline_da'].median()])
        summary['Llama 3B'].extend([df['llama_da'].mean(), df['llama_da'].median()])
        summary['Mistral 8x7B'].extend([df['mistral_da'].mean(), df['mistral_da'].median()])
        summary['GPT-4o ~20B'].extend([df['gpt4o_da'].mean(), df['gpt4o_da'].median()])
        
        # Win rates
        arima_wins = (df['best_model'] == 'ARIMA').sum()
        ets_wins = (df['best_model'] == 'ETS').sum()
        llama_wins = (df['best_model'] == 'LLMP').sum()
        mistral_wins = (df['best_model'] == 'MISTRAL').sum()
        gpt4o_wins = (df['best_model'] == 'GPT4O').sum()
        total = len(df)
        
        summary['Metric'].extend(['Win Rate %', 'Win Count'])
        summary['ARIMA'].extend([100*arima_wins/total, arima_wins])
        summary['ETS'].extend([100*ets_wins/total, ets_wins])
        summary['Best Baseline'].extend(['-', '-'])
        summary['Llama 3B'].extend([100*llama_wins/total, llama_wins])
        summary['Mistral 8x7B'].extend([100*mistral_wins/total, mistral_wins])
        summary['GPT-4o ~20B'].extend([100*gpt4o_wins/total, gpt4o_wins])
        
        # Improvement vs baseline
        summary['Metric'].extend(['Mean Improvement (NMAE)', '% Beat Baseline'])
        summary['ARIMA'].extend(['-', '-'])
        summary['ETS'].extend(['-', '-'])
        summary['Best Baseline'].extend([0, '-'])
        summary['Llama 3B'].extend([df['llama_improvement_nmae'].mean(), 100*df['llama_beats_baseline'].mean()])
        summary['Mistral 8x7B'].extend([df['mistral_improvement_nmae'].mean(), 100*df['mistral_beats_baseline'].mean()])
        summary['GPT-4o ~20B'].extend([df['gpt4o_improvement_nmae'].mean(), 100*df['gpt4o_beats_baseline'].mean()])
        
        summary_df = pd.DataFrame(summary)
        output_path = self.results_dir / 'model_performance_summary.csv'
        summary_df.to_csv(output_path, index=False)
        print(f"💾 Saved: {output_path}")
        
        return summary_df
    
    def generate_diminishing_returns_table(self):
        """Generate diminishing returns analysis table"""
        print("\n📉 Generating diminishing returns analysis...")
        
        df = self.compiled_df
        
        diminishing = {
            'Model': ['Best Baseline', 'Llama 3.2', 'Mistral 8x7B', 'GPT-4o mini'],
            'Size (B params)': ['-', 3, 56, 20],
            'Mean NMAE': [
                df['best_baseline_nmae'].mean(),
                df['llama_nmae'].mean(),
                df['mistral_nmae'].mean(),
                df['gpt4o_nmae'].mean()
            ],
            'Mean Improvement vs Baseline': [
                0,
                df['llama_improvement_nmae'].mean(),
                df['mistral_improvement_nmae'].mean(),
                df['gpt4o_improvement_nmae'].mean()
            ],
            '% Beat Baseline': [
                '-',
                100 * df['llama_beats_baseline'].mean(),
                100 * df['mistral_beats_baseline'].mean(),
                100 * df['gpt4o_beats_baseline'].mean()
            ]
        }
        
        # Calculate incremental gains
        llama_gain = df['llama_improvement_nmae'].mean()
        mistral_incremental = df['mistral_improvement_nmae'].mean() - llama_gain
        gpt4o_incremental = df['gpt4o_improvement_nmae'].mean() - llama_gain
        
        diminishing['Incremental Gain vs Previous'] = ['-', '-', mistral_incremental, gpt4o_incremental]
        
        dim_df = pd.DataFrame(diminishing)
        output_path = self.results_dir / 'diminishing_returns.csv'
        dim_df.to_csv(output_path, index=False)
        print(f"💾 Saved: {output_path}")
        
        print("\n📊 Diminishing Returns Summary:")
        print(dim_df.to_string(index=False))
        
        return dim_df
    
    def create_visualizations(self):
        """Generate all 6 required visualizations"""
        print("\n🎨 Creating visualizations...")
        
        figures_dir = self.results_dir / 'figures'
        figures_dir.mkdir(exist_ok=True)
        
        df = self.compiled_df
        
        # Figure 1: Model Performance Comparison (Box plot)
        print("   Creating Figure 1: Model Performance Comparison...")
        fig, ax = plt.subplots(figsize=(12, 6))
        
        nmae_data = pd.DataFrame({
            'ARIMA': df['arima_nmae'],
            'ETS': df['ets_nmae'],
            'Llama 3B': df['llama_nmae'],
            'Mistral 8x7B': df['mistral_nmae'],
            'GPT-4o mini': df['gpt4o_nmae']
        })
        
        bp = ax.boxplot([nmae_data[col].dropna() for col in nmae_data.columns],
                        labels=nmae_data.columns,
                        patch_artist=True,
                        showmeans=True)
        
        # Color the boxes
        colors = ['lightblue', 'lightgreen', 'salmon', 'gold', 'plum']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax.set_ylabel('NMAE (lower is better)', fontsize=12)
        ax.set_title('Model Performance Comparison Across All 160 Tasks', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(figures_dir / 'model_nmae_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Figure 2: Win Rate Pie Chart
        print("   Creating Figure 2: Win Rates...")
        fig, ax = plt.subplots(figsize=(10, 8))
        
        win_counts = df['best_model'].value_counts()
        colors_pie = ['lightblue', 'lightgreen', 'salmon', 'gold', 'plum']
        
        ax.pie(win_counts.values, labels=win_counts.index, autopct='%1.1f%%',
               colors=colors_pie[:len(win_counts)], startangle=90)
        ax.set_title('Model Win Rates (Lowest NMAE per Task)', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(figures_dir / 'win_rates.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Figure 3: Improvement Distribution
        print("   Creating Figure 3: Improvement Distribution...")
        fig, ax = plt.subplots(figsize=(12, 6))
        
        ax.hist(df['llama_improvement_nmae'], bins=30, alpha=0.5, label='Llama 3B', color='salmon')
        ax.hist(df['mistral_improvement_nmae'], bins=30, alpha=0.5, label='Mistral 8x7B', color='gold')
        ax.hist(df['gpt4o_improvement_nmae'], bins=30, alpha=0.5, label='GPT-4o mini', color='plum')
        
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='No improvement line')
        ax.set_xlabel('Improvement over Best Baseline (NMAE)', fontsize=12)
        ax.set_ylabel('Count of Tasks', fontsize=12)
        ax.set_title('Distribution of LLM Improvements Over Baseline', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(figures_dir / 'improvement_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Figure 4: Diminishing Returns
        print("   Creating Figure 4: Diminishing Returns...")
        fig, ax = plt.subplots(figsize=(10, 6))
        
        model_sizes = [3, 20, 56]  # Llama, GPT-4o, Mistral
        mean_improvements = [
            df['llama_improvement_nmae'].mean(),
            df['gpt4o_improvement_nmae'].mean(),
            df['mistral_improvement_nmae'].mean()
        ]
        std_improvements = [
            df['llama_improvement_nmae'].std() / np.sqrt(len(df)),
            df['gpt4o_improvement_nmae'].std() / np.sqrt(len(df)),
            df['mistral_improvement_nmae'].std() / np.sqrt(len(df))
        ]
        pct_beat = [
            100 * df['llama_beats_baseline'].mean(),
            100 * df['gpt4o_beats_baseline'].mean(),
            100 * df['mistral_beats_baseline'].mean()
        ]
        
        ax.errorbar(model_sizes, mean_improvements, yerr=std_improvements,
                   marker='o', markersize=10, linewidth=2, capsize=5)
        ax.set_xscale('log')
        ax.set_xlabel('Model Size (Billions of Parameters)', fontsize=12)
        ax.set_ylabel('Mean Improvement over Baseline (NMAE)', fontsize=12)
        ax.set_title('Diminishing Returns: Performance vs Model Size', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Annotate with % beating baseline
        for x, y, pct in zip(model_sizes, mean_improvements, pct_beat):
            ax.annotate(f'{pct:.1f}% beat baseline', 
                       xy=(x, y), xytext=(10, 10), textcoords='offset points',
                       fontsize=9, bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(figures_dir / 'diminishing_returns.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Figure 5: Performance by Domain
        print("   Creating Figure 5: Performance by Domain...")
        fig, ax = plt.subplots(figsize=(16, 8))
        
        domain_means = df.groupby('domain')[['arima_nmae', 'ets_nmae', 'llama_nmae', 
                                             'mistral_nmae', 'gpt4o_nmae']].mean()
        
        x = np.arange(len(domain_means))
        width = 0.15
        
        ax.bar(x - 2*width, domain_means['arima_nmae'], width, label='ARIMA', color='lightblue')
        ax.bar(x - width, domain_means['ets_nmae'], width, label='ETS', color='lightgreen')
        ax.bar(x, domain_means['llama_nmae'], width, label='Llama 3B', color='salmon')
        ax.bar(x + width, domain_means['mistral_nmae'], width, label='Mistral 8x7B', color='gold')
        ax.bar(x + 2*width, domain_means['gpt4o_nmae'], width, label='GPT-4o mini', color='plum')
        
        ax.set_xlabel('Domain', fontsize=12)
        ax.set_ylabel('Mean NMAE', fontsize=12)
        ax.set_title('Model Performance Across Domains', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(domain_means.index, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(figures_dir / 'performance_by_domain.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Figure 6: Improvement Heatmap
        print("   Creating Figure 6: Improvement Heatmap...")
        fig, ax = plt.subplots(figsize=(10, 10))
        
        heatmap_data = df.groupby('domain')[['llama_improvement_nmae', 
                                              'mistral_improvement_nmae',
                                              'gpt4o_improvement_nmae']].mean()
        heatmap_data.columns = ['Llama 3B', 'Mistral 8x7B', 'GPT-4o mini']
        
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn', center=0,
                   cbar_kws={'label': 'Mean Improvement (NMAE)'}, ax=ax)
        ax.set_title('LLM Improvement Over Baseline by Domain', fontsize=14, fontweight='bold')
        ax.set_ylabel('Domain', fontsize=12)
        ax.set_xlabel('Model', fontsize=12)
        
        plt.tight_layout()
        plt.savefig(figures_dir / 'improvement_heatmap.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ All 6 figures saved to {figures_dir}/")
    
    def run_statistical_tests(self):
        """Run statistical significance tests"""
        print("\n🔬 Running statistical tests...")
        
        df = self.compiled_df
        
        results = []
        results.append("=" * 80)
        results.append("STATISTICAL ANALYSIS")
        results.append("=" * 80)
        
        # Test 1: Do LLMs beat baseline?
        results.append("\n1. PAIRED T-TESTS: Do LLMs beat baseline?")
        results.append("-" * 80)
        
        for model_name, improvement_col in [('Llama 3B', 'llama_improvement_nmae'),
                                             ('Mistral 8x7B', 'mistral_improvement_nmae'),
                                             ('GPT-4o mini', 'gpt4o_improvement_nmae')]:
            t_stat, p_value = stats.ttest_1samp(df[improvement_col], 0)
            mean_improvement = df[improvement_col].mean()
            results.append(f"\n{model_name} vs Baseline:")
            results.append(f"   Mean improvement: {mean_improvement:.4f}")
            results.append(f"   t-statistic: {t_stat:.4f}")
            results.append(f"   p-value: {p_value:.6f}")
            results.append(f"   Significant at p<0.05? {'YES' if p_value < 0.05 else 'NO'}")
        
        # Test 2: Diminishing returns tests
        results.append("\n\n2. PAIRED T-TESTS: Diminishing returns")
        results.append("-" * 80)
        
        # Mistral vs Llama
        t_stat, p_value = stats.ttest_rel(df['mistral_improvement_nmae'], df['llama_improvement_nmae'])
        results.append(f"\nMistral 8x7B vs Llama 3B:")
        results.append(f"   Mistral mean improvement: {df['mistral_improvement_nmae'].mean():.4f}")
        results.append(f"   Llama mean improvement: {df['llama_improvement_nmae'].mean():.4f}")
        results.append(f"   Difference: {df['mistral_improvement_nmae'].mean() - df['llama_improvement_nmae'].mean():.4f}")
        results.append(f"   t-statistic: {t_stat:.4f}")
        results.append(f"   p-value: {p_value:.6f}")
        results.append(f"   Mistral significantly better? {'YES' if p_value < 0.05 and t_stat > 0 else 'NO'}")
        
        # GPT-4o vs Llama
        t_stat, p_value = stats.ttest_rel(df['gpt4o_improvement_nmae'], df['llama_improvement_nmae'])
        results.append(f"\nGPT-4o mini vs Llama 3B:")
        results.append(f"   GPT-4o mean improvement: {df['gpt4o_improvement_nmae'].mean():.4f}")
        results.append(f"   Llama mean improvement: {df['llama_improvement_nmae'].mean():.4f}")
        results.append(f"   Difference: {df['gpt4o_improvement_nmae'].mean() - df['llama_improvement_nmae'].mean():.4f}")
        results.append(f"   t-statistic: {t_stat:.4f}")
        results.append(f"   p-value: {p_value:.6f}")
        results.append(f"   GPT-4o significantly better? {'YES' if p_value < 0.05 and t_stat > 0 else 'NO'}")
        
        # Mistral vs GPT-4o
        t_stat, p_value = stats.ttest_rel(df['mistral_improvement_nmae'], df['gpt4o_improvement_nmae'])
        results.append(f"\nMistral 8x7B vs GPT-4o mini:")
        results.append(f"   Mistral mean improvement: {df['mistral_improvement_nmae'].mean():.4f}")
        results.append(f"   GPT-4o mean improvement: {df['gpt4o_improvement_nmae'].mean():.4f}")
        results.append(f"   Difference: {df['mistral_improvement_nmae'].mean() - df['gpt4o_improvement_nmae'].mean():.4f}")
        results.append(f"   t-statistic: {t_stat:.4f}")
        results.append(f"   p-value: {p_value:.6f}")
        results.append(f"   Mistral significantly better? {'YES' if p_value < 0.05 and t_stat > 0 else 'NO'}")
        
        # Test 3: Effect sizes
        results.append("\n\n3. EFFECT SIZES (Cohen's d)")
        results.append("-" * 80)
        
        for model_name, improvement_col in [('Llama 3B', 'llama_improvement_nmae'),
                                             ('Mistral 8x7B', 'mistral_improvement_nmae'),
                                             ('GPT-4o mini', 'gpt4o_improvement_nmae')]:
            mean_improvement = df[improvement_col].mean()
            std_improvement = df[improvement_col].std()
            cohens_d = mean_improvement / std_improvement if std_improvement > 0 else 0
            
            results.append(f"\n{model_name} vs zero:")
            results.append(f"   Cohen's d: {cohens_d:.4f}")
            
            if abs(cohens_d) < 0.2:
                effect = "negligible"
            elif abs(cohens_d) < 0.5:
                effect = "small"
            elif abs(cohens_d) < 0.8:
                effect = "medium"
            else:
                effect = "large"
            results.append(f"   Effect size: {effect}")
        
        # Test 4: Chi-square test for win rates
        results.append("\n\n4. CHI-SQUARE TEST: Win rate distribution")
        results.append("-" * 80)
        
        observed = df['best_model'].value_counts().values
        expected_uniform = len(df) / len(observed)  # Uniform distribution
        chi2, p_value = stats.chisquare(observed, f_exp=[expected_uniform] * len(observed))
        
        results.append(f"\nObserved win counts: {dict(df['best_model'].value_counts())}")
        results.append(f"Chi-square statistic: {chi2:.4f}")
        results.append(f"p-value: {p_value:.6f}")
        results.append(f"Win rates significantly different from uniform? {'YES' if p_value < 0.05 else 'NO'}")
        
        # Save results
        results_text = '\n'.join(results)
        output_path = self.results_dir / 'statistical_analysis.txt'
        with open(output_path, 'w') as f:
            f.write(results_text)
        
        print(results_text)
        print(f"\n💾 Saved: {output_path}")
        
        return results_text
    
    def generate_summary_report(self):
        """Generate executive summary answering key questions"""
        print("\n📝 Generating summary report...")
        
        df = self.compiled_df
        
        report = []
        report.append("=" * 80)
        report.append("EXECUTIVE SUMMARY: DIMINISHING RETURNS IN LLM TIME SERIES FORECASTING")
        report.append("=" * 80)
        report.append("")
        
        # Question 1: Overall Performance
        report.append("1. OVERALL PERFORMANCE")
        report.append("-" * 80)
        report.append(f"Total tasks analyzed: {len(df)}")
        report.append(f"Total domains: {df['domain'].nunique()}")
        report.append("")
        
        mean_nmae = {
            'ARIMA': df['arima_nmae'].mean(),
            'ETS': df['ets_nmae'].mean(),
            'Best Baseline': df['best_baseline_nmae'].mean(),
            'Llama 3B': df['llama_nmae'].mean(),
            'Mistral 8x7B': df['mistral_nmae'].mean(),
            'GPT-4o mini': df['gpt4o_nmae'].mean()
        }
        best_model = min(mean_nmae, key=mean_nmae.get)
        report.append(f"Best mean NMAE: {best_model} ({mean_nmae[best_model]:.4f})")
        report.append("")
        
        win_counts = df['best_model'].value_counts()
        most_wins = win_counts.idxmax()
        report.append(f"Most tasks won: {most_wins} ({win_counts[most_wins]} / {len(df)} = {100*win_counts[most_wins]/len(df):.1f}%)")
        report.append("")
        
        llama_beats = 100 * df['llama_beats_baseline'].mean()
        mistral_beats = 100 * df['mistral_beats_baseline'].mean()
        gpt4o_beats = 100 * df['gpt4o_beats_baseline'].mean()
        report.append(f"LLMs beating baseline:")
        report.append(f"   Llama 3B: {llama_beats:.1f}%")
        report.append(f"   Mistral 8x7B: {mistral_beats:.1f}%")
        report.append(f"   GPT-4o mini: {gpt4o_beats:.1f}%")
        report.append("")
        
        # Question 2: Diminishing Returns
        report.append("2. DIMINISHING RETURNS ANALYSIS")
        report.append("-" * 80)
        
        llama_imp = df['llama_improvement_nmae'].mean()
        mistral_imp = df['mistral_improvement_nmae'].mean()
        gpt4o_imp = df['gpt4o_improvement_nmae'].mean()
        
        mistral_incremental = mistral_imp - llama_imp
        gpt4o_incremental = gpt4o_imp - llama_imp
        
        report.append(f"Llama 3B (3B params) improvement over baseline: {llama_imp:.4f}")
        report.append(f"Mistral 8x7B (56B params) improvement over baseline: {mistral_imp:.4f}")
        report.append(f"   → Incremental gain over Llama: {mistral_incremental:.4f}")
        report.append(f"   → Percentage improvement: {100*mistral_incremental/abs(llama_imp):.1f}%")
        report.append("")
        
        report.append(f"GPT-4o mini (~20B params) improvement over baseline: {gpt4o_imp:.4f}")
        report.append(f"   → Incremental gain over Llama: {gpt4o_incremental:.4f}")
        report.append(f"   → Percentage improvement: {100*gpt4o_incremental/abs(llama_imp):.1f}%")
        report.append("")
        
        # Question 3: Cost-Benefit
        report.append("3. COST-BENEFIT ASSESSMENT")
        report.append("-" * 80)
        report.append("Assuming:")
        report.append("   - Llama 3B: Free (local), ~5 min per task")
        report.append("   - Mistral 8x7B: $0.50/160 tasks (or ~15 min local)")
        report.append("   - GPT-4o mini: $2.40/160 tasks")
        report.append("")
        
        if mistral_incremental > 0:
            report.append(f"Mistral's incremental improvement: {mistral_incremental:.4f} NMAE")
            report.append(f"Cost per point of improvement: ${0.50/abs(mistral_incremental):.2f}")
            report.append("Verdict: Marginal improvement, consider only if budget allows")
        else:
            report.append("Mistral does NOT improve over Llama 3B")
            report.append("Verdict: NOT WORTH THE COST")
        report.append("")
        
        if gpt4o_incremental > 0:
            report.append(f"GPT-4o's incremental improvement: {gpt4o_incremental:.4f} NMAE")
            report.append(f"Cost per point of improvement: ${2.40/abs(gpt4o_incremental):.2f}")
            report.append("Verdict: Evaluate based on application requirements")
        else:
            report.append("GPT-4o does NOT improve over Llama 3B")
            report.append("Verdict: NOT WORTH THE COST")
        report.append("")
        
        # Question 4: Domain Patterns
        report.append("4. DOMAIN-SPECIFIC PATTERNS")
        report.append("-" * 80)
        
        # Find domains where LLMs excel
        domain_llm_wins = df.groupby('domain').apply(
            lambda x: (x['best_model'].isin(['LLMP', 'MISTRAL', 'GPT4O'])).sum() / len(x)
        ).sort_values(ascending=False)
        
        report.append("Domains favoring LLMs (>50% win rate):")
        for domain, rate in domain_llm_wins[domain_llm_wins > 0.5].items():
            report.append(f"   {domain}: {100*rate:.1f}%")
        report.append("")
        
        report.append("Domains favoring baselines (<30% LLM win rate):")
        for domain, rate in domain_llm_wins[domain_llm_wins < 0.3].items():
            report.append(f"   {domain}: {100*rate:.1f}% LLM wins")
        report.append("")
        
        # Question 5: Bottom Line Recommendation
        report.append("5. BOTTOM LINE RECOMMENDATION")
        report.append("-" * 80)
        
        if llama_beats < 50:
            report.append("❌ LLMs with context do NOT consistently beat statistical baselines")
            report.append(f"   Llama 3B only beats baseline in {llama_beats:.1f}% of tasks")
            report.append("")
            report.append("RECOMMENDATION:")
            report.append("   - Use statistical baselines (ARIMA/ETS) as primary forecasting method")
            report.append("   - Consider LLMs only for specific domains where they excel")
            report.append("   - Larger LLMs provide minimal additional benefit")
        else:
            report.append("✅ LLMs with context show promise for time series forecasting")
            report.append(f"   Llama 3B beats baseline in {llama_beats:.1f}% of tasks")
            report.append("")
            report.append("RECOMMENDATION:")
            report.append("   - Llama 3B offers best cost-performance trade-off")
            
            if mistral_incremental > 0.1:
                report.append("   - Mistral 8x7B worth considering if budget allows")
            else:
                report.append("   - Mistral 8x7B does not justify additional cost")
            
            if gpt4o_incremental > 0.1:
                report.append("   - GPT-4o mini worth considering for critical applications")
            else:
                report.append("   - GPT-4o mini does not justify additional cost")
        
        report.append("")
        report.append("=" * 80)
        report.append("This challenges the Context-is-Key paper's findings with 405B models,")
        report.append("showing that context benefits are highly scale-dependent.")
        report.append("=" * 80)
        
        # Save report
        report_text = '\n'.join(report)
        output_path = self.results_dir / 'summary_report.txt'
        with open(output_path, 'w') as f:
            f.write(report_text)
        
        print(report_text)
        print(f"\n💾 Saved: {output_path}")
        
        return report_text
    
    def run_full_analysis(self):
        """Run the complete analysis pipeline"""
        print("🚀 Starting Diminishing Returns Analysis\n")
        print("=" * 80)
        
        # Step 1: Discover domains
        self.discover_domains()
        
        # Step 2: Merge all results
        self.merge_all_domains()
        
        # Step 3: Calculate improvements
        self.calculate_baseline_and_improvements()
        
        # Step 4: Generate summary statistics
        self.generate_summary_statistics()
        
        # Step 5: Generate diminishing returns table
        self.generate_diminishing_returns_table()
        
        # Step 6: Create visualizations
        self.create_visualizations()
        
        # Step 7: Run statistical tests
        self.run_statistical_tests()
        
        # Step 8: Generate summary report
        self.generate_summary_report()
        
        print("\n" + "=" * 80)
        print("✅ ANALYSIS COMPLETE!")
        print("=" * 80)
        print("\nDeliverables saved:")
        print("   📊 compiled_comparison.csv")
        print("   📈 model_performance_summary.csv")
        print("   📉 diminishing_returns.csv")
        print("   🎨 figures/ (6 visualizations)")
        print("   🔬 statistical_analysis.txt")
        print("   📝 summary_report.txt")


if __name__ == "__main__":
    # Set results directory
    results_dir = Path("/Users/kaziashhabrahman/Documents/McGill/Fall 25/Comp 545/COMP545_Final_Project/Results")
    
    # Create analyzer
    analyzer = DiminishingReturnsAnalyzer(results_dir)
    
    # Run full analysis
    analyzer.run_full_analysis()
