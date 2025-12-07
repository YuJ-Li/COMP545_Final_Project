"""
Publication-Ready Win Rates Pie Chart
Black boxes with white text for labels and percentages
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Use the same style as other plots
plt.style.use('seaborn-v0_8-whitegrid')

def create_publication_pie_chart(results_dir):
    """
    Create a publication-ready pie chart showing model win rates.
    Labels inside pie slices with black background boxes.
    """
    
    # Load data
    results_dir = Path(results_dir)
    df = pd.read_csv(results_dir / 'compiled_comparison.csv')
    
    # Get win counts
    win_counts = df['best_model'].value_counts()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Model name mapping (from data to display names)
    name_mapping = {
        'ARIMA': 'Arima',
        'ETS': 'ETS',
        'MISTRAL': 'Mistral',
        'GPT4O': 'GPT 4o mini',
        'LLAMA': 'Llama 3B'
    }
    
    # Updated color scheme with your specified colors
    colors = {
        'Arima': '#003f5c',         # Dark blue
        'ETS': '#58508d',            # Purple
        'Mistral': '#bc5090',        # Pink/magenta
        'GPT 4o mini': '#ff6361',    # Coral/red
        'Llama 3B': '#ffa600'        # Orange
    }
    
    # Map data model names to display names
    display_names = [name_mapping.get(model, model) for model in win_counts.index]
    
    # Get colors for each model
    pie_colors = [colors.get(name, '#CCCCCC') for name in display_names]
    
    # Calculate percentages
    total = win_counts.sum()
    percentages = (win_counts.values / total) * 100
    
    # Create custom labels with model name + percentage
    custom_labels = [f'{name}\n{pct:.1f}%' 
                     for name, pct in zip(display_names, percentages)]
    
    # Create pie chart WITHOUT autopct (we'll add custom text)
    wedges, texts = ax.pie(
        win_counts.values,
        labels=None,  # We'll add custom labels
        colors=pie_colors,
        startangle=90,
        textprops={'fontsize': 16}
    )
    
    # Add custom text labels with black background boxes
    for i, (wedge, label) in enumerate(zip(wedges, custom_labels)):
        # Get the center angle of the wedge
        angle = (wedge.theta2 - wedge.theta1) / 2.0 + wedge.theta1
        
        # Calculate position (slightly inward from edge)
        x = 0.65 * np.cos(np.radians(angle))
        y = 0.65 * np.sin(np.radians(angle))
        
        # Add text with black background box
        ax.text(
            x, y, label,
            ha='center', va='center',
            fontsize=16,
            fontweight='bold',
            color='white',
            bbox=dict(
                boxstyle='round,pad=0.5',
                facecolor='black',
                edgecolor='none',
                alpha=0.8
            )
        )
    
    # Title
    ax.set_title(
        'Model Win Rates (Lowest NMAE per Task)',
        fontsize=20,
        fontweight='bold',
        pad=20
    )
    
    # Equal aspect ratio
    ax.axis('equal')
    
    # Clean background
    ax.set_facecolor('white')
    fig.patch.set_facecolor('white')
    
    # Save high-quality versions
    output_path_png = results_dir / 'win_rates_publication.png'
    plt.savefig(
        output_path_png,
        dpi=300,
        bbox_inches='tight',
        facecolor='white'
    )
    print(f"✓ Saved PNG: {output_path_png}")
    
    # PDF for LaTeX
    output_path_pdf = results_dir / 'win_rates_publication.pdf'
    plt.savefig(
        output_path_pdf,
        bbox_inches='tight',
        facecolor='white'
    )
    print(f"✓ Saved PDF: {output_path_pdf}")
    
    plt.close()
    
    # Print statistics with display names
    print("\n" + "="*50)
    print("WIN RATE SUMMARY")
    print("="*50)
    for model, count, name in zip(win_counts.index, win_counts.values, display_names):
        percentage = (count / total) * 100
        print(f"{name:15s}: {count:3d} tasks ({percentage:5.1f}%)")
    print("="*50)


if __name__ == "__main__":
    results_dir = Path("/Users/kaziashhabrahman/Documents/McGill/Fall 25/Comp 545/COMP545_Final_Project/Results")
    
    print("Creating publication-ready pie chart...")
    create_publication_pie_chart(results_dir)
    print("\n✅ Done!")