"""
analysis/01_eda.py — Exploratory Data Analysis
==============================================

PURPOSE:
    To understand the shape, distribution, and patterns of our dataset
    before we throw complex Machine Learning at it. 
    
    In a real job, EDA is where you find data errors, outliers, and the
    business narrative. We generate 5+ key charts here to answer
    critical business questions about fraud patterns.

AUTHOR: Sisirath Chaloor Kuppadan | github.com/sisirathck
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Add the root directory to Python's path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data_loader import load_joined
from src.visualizations import set_professional_style, COLOR_LEGIT, COLOR_FRAUD, COLOR_ACCENT

# Ensure the outputs directory exists
OUTPUT_DIR = os.path.join('outputs', 'figures', 'eda')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_fraud_distribution(df):
    """Answers: How severe is our class imbalance?"""
    print("  → Plotting Fraud Distribution...")
    fig, ax = plt.subplots(figsize=(8, 6))
    
    counts = df['is_fraud'].value_counts()
    percentages = 100 * counts / len(df)
    
    sns.barplot(
        x=['Legitimate (0)', 'Fraud (1)'], 
        y=counts.values,
        palette=[COLOR_LEGIT, COLOR_FRAUD],
        ax=ax
    )
    
    set_professional_style(ax, 'Overall Class Distribution', 'Transaction Status', 'Number of Transactions')
    
    for i, p in enumerate(ax.patches):
        height = p.get_height()
        ax.text(
            p.get_x() + p.get_width() / 2., 
            height + (max(counts) * 0.02),
            f'{percentages.iloc[i]:.1f}%',
            ha='center', va='bottom', fontweight='bold', fontsize=12
        )
        
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '01_class_distribution.png'), dpi=300)
    plt.close()


def plot_fraud_by_category(df):
    """Answers: Which merchant categories are targeted most by fraudsters?"""
    print("  → Plotting Fraud by Category...")
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Calculate fraud rate (%) per category
    fraud_rates = df.groupby('merchant_category')['is_fraud'].mean() * 100
    fraud_rates = fraud_rates.sort_values(ascending=False)
    
    # Create a custom palette: Highlight the highest risk categories in Red
    pal = [COLOR_FRAUD if rate > 30 else COLOR_ACCENT if rate > 10 else COLOR_LEGIT for rate in fraud_rates.values]
    
    sns.barplot(x=fraud_rates.index, y=fraud_rates.values, palette=pal, ax=ax)
    
    set_professional_style(ax, 'Fraud Risk by Merchant Category', 'Merchant Category', 'Fraud Rate (%)')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '02_fraud_by_category.png'), dpi=300)
    plt.close()


def plot_fraud_by_hour(df):
    """Answers: What time of day is fraud most likely to occur?"""
    print("  → Plotting Fraud by Hour...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    fraud_rates = df.groupby('transaction_hour')['is_fraud'].mean() * 100
    
    # Plot as a line chart to show the temporal trend
    ax.plot(fraud_rates.index, fraud_rates.values, color=COLOR_FRAUD, lw=3, marker='o', markersize=8)
    
    # Highlight the "Late Night" danger zone (11 PM - 3 AM)
    ax.axvspan(22.5, 23.5, color=COLOR_ACCENT, alpha=0.2)
    ax.axvspan(-0.5, 3.5, color=COLOR_ACCENT, alpha=0.2, label='Late Night Risk Zone')
    
    set_professional_style(ax, 'Fraud Rate by Time of Day', 'Hour of Day (0-23)', 'Fraud Rate (%)')
    ax.set_xticks(range(0, 24))
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '03_fraud_by_hour.png'), dpi=300)
    plt.close()


def plot_amount_distribution(df):
    """Answers: Do fraudsters steal larger amounts on average?"""
    print("  → Plotting Amount Distribution...")
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # We use log scale because amounts vary wildly (₹10 to ₹100,000+)
    # Log scale compresses the x-axis so we can see the distributions clearly
    sns.kdeplot(data=df[df['is_fraud'] == 0], x='amount', log_scale=True, 
                color=COLOR_LEGIT, fill=True, alpha=0.5, label='Legitimate', ax=ax)
    sns.kdeplot(data=df[df['is_fraud'] == 1], x='amount', log_scale=True, 
                color=COLOR_FRAUD, fill=True, alpha=0.5, label='Fraud', ax=ax)
    
    set_professional_style(ax, 'Transaction Amount Distribution (Log Scale)', 'Amount (INR) - Log Scale', 'Density')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '04_amount_distribution.png'), dpi=300)
    plt.close()


def plot_correlation_heatmap(df):
    """Answers: How are our numerical variables related to each other?"""
    print("  → Plotting Correlation Heatmap...")
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Select only continuous numerical columns
    cols = ['amount', 'transaction_hour', 'distance_from_home_km', 'num_prev_transactions_24h', 'age', 'credit_score', 'monthly_income_inr', 'is_fraud']
    corr = df[cols].corr()
    
    # Custom diverging colormap
    cmap = sns.diverging_palette(230, 20, as_cmap=True)
    
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap, center=0, 
                square=True, linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
    
    ax.set_title("Feature Correlation Heatmap", pad=20, fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '05_correlation_heatmap.png'), dpi=300)
    plt.close()


def main():
    print("\n" + "="*50)
    print("  📊 Running Exploratory Data Analysis (EDA)")
    print("="*50)
    
    df = load_joined()
    
    plot_fraud_distribution(df)
    plot_fraud_by_category(df)
    plot_fraud_by_hour(df)
    plot_amount_distribution(df)
    plot_correlation_heatmap(df)
    
    print("\n  ✅ EDA Complete! All charts saved to outputs/figures/eda/\n")

if __name__ == "__main__":
    main()
