"""
src/visualizations.py — Data & Model Visualization
==================================================

PURPOSE:
    Provides reusable functions to generate beautiful, portfolio-ready charts.
    We use Matplotlib and Seaborn for plotting.
    
    Aesthetic choices (like using specific colors and hiding gridlines)
    are made to ensure the charts look professional in a GitHub README or
    a Streamlit dashboard.

AUTHOR: Sisirath Chaloor Kuppadan | github.com/sisirathck
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
import os

# ── Global Aesthetic Settings ──────────────────────────────────
# Set a beautiful, clean style for all plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("notebook", font_scale=1.1)

# Custom color palette (Professional deep blues and alerts)
COLOR_LEGIT = "#2E86AB"  # Deep blue for normal behavior
COLOR_FRAUD = "#D62828"  # Strong red for fraud
COLOR_ACCENT = "#F4A261" # Orange for highlights

def set_professional_style(ax, title, xlabel, ylabel):
    """Applies a clean, modern aesthetic to a given matplotlib axis."""
    ax.set_title(title, pad=15, fontweight='bold', fontsize=14)
    ax.set_xlabel(xlabel, fontweight='bold')
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle='--', alpha=0.6)
    ax.grid(axis='x', visible=False)


def plot_class_distribution(y, save_path=None):
    """
    Plots a bar chart showing the severe imbalance between Legit and Fraud.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    counts = y.value_counts()
    percentages = 100 * counts / len(y)
    
    sns.barplot(
        x=['Legitimate (0)', 'Fraud (1)'], 
        y=counts.values,
        palette=[COLOR_LEGIT, COLOR_FRAUD],
        ax=ax
    )
    
    set_professional_style(ax, 'Class Imbalance in Financial Data', 'Class', 'Number of Transactions')
    
    # Add percentage labels on top of the bars
    for i, p in enumerate(ax.patches):
        height = p.get_height()
        ax.text(
            p.get_x() + p.get_width() / 2., 
            height + (max(counts) * 0.02),
            f'{percentages.iloc[i]:.1f}%',
            ha='center', va='bottom', fontweight='bold', fontsize=12
        )
        
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return fig


def plot_roc_curve(y_true, y_prob, save_path=None):
    """
    Plots the Receiver Operating Characteristic (ROC) Curve.
    
    WHAT IT SHOWS:
    How well the model separates the two classes. 
    A perfect model hugs the top-left corner (AUC = 1.0).
    A random guess model follows the diagonal line (AUC = 0.5).
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot the actual curve
    ax.plot(fpr, tpr, color=COLOR_FRAUD, lw=3, label=f'Model ROC (AUC = {roc_auc:.3f})')
    
    # Plot the random guess line
    ax.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--', label='Random Guessing')
    
    set_professional_style(ax, 'Receiver Operating Characteristic (ROC)', 'False Positive Rate', 'True Positive Rate')
    ax.legend(loc='lower right', frameon=True, shadow=True)
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return fig


def plot_precision_recall_curve(y_true, y_prob, save_path=None):
    """
    Plots the Precision-Recall Curve.
    
    WHY DO WE NEED THIS?
    Because fraud data is imbalanced! ROC curves can be overly optimistic
    when the positive class (fraud) is rare. PR curves show the true
    trade-off: if we want to catch ALL fraud (high recall), how many 
    legitimate transactions do we accidentally block (low precision)?
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    ax.plot(recall, precision, color=COLOR_LEGIT, lw=3, label='Precision-Recall Curve')
    
    set_professional_style(ax, 'Precision-Recall Trade-off', 'Recall (Caught Fraud %)', 'Precision (Correct Flag %)')
    
    # Baseline is the proportion of positive class
    baseline = sum(y_true) / len(y_true)
    ax.axhline(y=baseline, color='gray', linestyle='--', label=f'Baseline ({baseline:.3f})')
    
    ax.legend(loc='lower left', frameon=True, shadow=True)
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return fig


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """
    Plots a beautiful Confusion Matrix heatmap.
    
    WHAT IT SHOWS:
    Exactly where the model made mistakes.
    Top-Left: True Negatives (Legit allowed)
    Top-Right: False Positives (Legit blocked - angry customer!)
    Bottom-Left: False Negatives (Fraud missed - money lost!)
    Bottom-Right: True Positives (Fraud caught - success!)
    """
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(7, 6))
    
    # Custom colormap that goes from white to our deep blue
    cmap = sns.light_palette(COLOR_LEGIT, as_cmap=True)
    
    sns.heatmap(
        cm, annot=True, fmt='d', cmap=cmap, cbar=False,
        annot_kws={'size': 16, 'weight': 'bold'},
        xticklabels=['Predicted Legit', 'Predicted Fraud'],
        yticklabels=['Actual Legit', 'Actual Fraud'],
        ax=ax
    )
    
    ax.set_title("Confusion Matrix", pad=15, fontweight='bold', fontsize=14)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return fig
