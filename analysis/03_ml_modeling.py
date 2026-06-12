"""
analysis/03_ml_modeling.py — Machine Learning Pipeline & Explainability
=======================================================================

PURPOSE:
    This is the core ML script. It:
    1. Loads the engineered features.
    2. Trains our custom "Scratch" Logistic Regression model.
    3. Trains the Scikit-learn production model.
    4. Evaluates them using professional metrics (ROC-AUC, Precision, Recall).
    5. Uses SHAP to explain the model's decisions (crucial for EU AI Act compliance).

AUTHOR: Sisirath Chaloor Kuppadan | github.com/sisirathck
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import shap

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_loader import load_joined
from src.feature_engineering import prepare_ml_dataset
from src.model import LogisticRegressionScratch, handle_class_imbalance, train_sklearn_model
from src.visualizations import plot_roc_curve, plot_precision_recall_curve, plot_confusion_matrix

OUTPUT_DIR = os.path.join('outputs', 'figures', 'ml')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def evaluate_model(model_name, y_true, y_pred, y_prob):
    """Prints a clean summary of standard classification metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_prob)
    
    print(f"\n  [{model_name}] Performance Metrics:")
    print(f"  -------------------------------------")
    print(f"  Accuracy  : {acc:.4f} (Can be misleading for imbalanced data)")
    print(f"  Precision : {prec:.4f} (When it flags fraud, how often is it right?)")
    print(f"  Recall    : {rec:.4f} (Out of all real fraud, how much did we catch?)")
    print(f"  F1 Score  : {f1:.4f} (Harmonic mean of Precision & Recall)")
    print(f"  ROC-AUC   : {roc_auc:.4f} (Overall ability to separate classes)")
    print(f"  -------------------------------------\n")


def run_shap_explainability(model, X_train, X_test):
    """
    DOMAIN LOGIC (AI Act Compliance):
    "Black box" AI is illegal in many financial decisions. We must be able to 
    explain exactly WHY a transaction was declined. SHAP values mathematically 
    distribute the "blame" (or credit) for a prediction among its features.
    """
    print("  → Generating SHAP Explainability plots (this takes a moment)...")
    
    # We use a random sample of the background data to speed up computation
    background = shap.sample(X_train, 100)
    
    # We use LinearExplainer explicitly because it is mathematically exact and blazing fast
    # for Logistic Regression models.
    explainer = shap.LinearExplainer(model, background)
    
    # Explain a subset of test transactions
    test_sample = X_test.sample(200, random_state=42)
    shap_values = explainer(test_sample)
    
    # 1. Global Feature Importance Plot (Summary Plot)
    plt.figure(figsize=(10, 6))
    # show=False allows us to save it
    shap.summary_plot(shap_values, test_sample, show=False)
    plt.title("SHAP Feature Importance (Global Impact)", pad=20, fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '04_shap_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("  ✓ SHAP summary plot generated.")


def main():
    print("\n" + "="*60)
    print("  🤖 Running Machine Learning Pipeline (03_ml_modeling.py)")
    print("="*60)
    
    # 1. Data Prep
    print("  [1/5] Loading and Engineering Features...")
    df = load_joined()
    X, y = prepare_ml_dataset(df)
    
    print("  [2/5] Splitting and Balancing Data (SMOTE)...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train_bal, y_train_bal = handle_class_imbalance(X_train, y_train)
    
    # 2. Scratch Model
    print("\n  [3/5] Training Custom 'From Scratch' Logistic Regression...")
    # Because we added feature scaling, learning rate can be much larger and it will converge nicely
    scratch_model = LogisticRegressionScratch(learning_rate=0.5, num_iterations=500)
    scratch_model.fit(X_train_bal, y_train_bal)
    
    y_prob_scratch = scratch_model.predict_proba(X_test)[:, 1]
    y_pred_scratch = scratch_model.predict(X_test)
    evaluate_model("Scratch NumPy Model", y_test, y_pred_scratch, y_prob_scratch)
    
    # 3. Scikit-Learn Model
    print("  [4/5] Training Scikit-Learn Production Model...")
    sklearn_model = train_sklearn_model(X_train_bal, y_train_bal)
    
    y_prob_sklearn = sklearn_model.predict_proba(X_test)[:, 1]
    y_pred_sklearn = sklearn_model.predict(X_test)
    evaluate_model("Scikit-Learn Model", y_test, y_pred_sklearn, y_prob_sklearn)
    
    # 4. Generate Core Visualizations (Using Sklearn model as the final)
    print("  [5/5] Generating Visualizations...")
    
    fig_roc = plot_roc_curve(y_test, y_prob_sklearn, save_path=os.path.join(OUTPUT_DIR, '01_roc_curve.png'))
    fig_pr = plot_precision_recall_curve(y_test, y_prob_sklearn, save_path=os.path.join(OUTPUT_DIR, '02_pr_curve.png'))
    fig_cm = plot_confusion_matrix(y_test, y_pred_sklearn, save_path=os.path.join(OUTPUT_DIR, '03_confusion_matrix.png'))
    
    # 5. SHAP
    run_shap_explainability(sklearn_model, X_train_bal, X_test)
    
    print("\n  ✅ ML Pipeline Complete! All charts saved to outputs/figures/ml/\n")

if __name__ == "__main__":
    main()
