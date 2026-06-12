"""
src/model.py — Machine Learning Models
======================================

Contains the core ML estimators for FraudShield:
  1. LogisticRegressionScratch — NumPy-only implementation for transparency and
     educational/regulatory validation against the production model.
  2. train_sklearn_model — Production Scikit-learn Logistic Regression pipeline.
  3. handle_class_imbalance — SMOTE-based resampling for imbalanced fraud data.

Design rationale for Logistic Regression:
    Logistic Regression is the preferred choice for financial fraud detection
    under interpretability mandates (EU AI Act, RBI guidelines). Unlike
    ensemble or deep learning models, it produces calibrated probabilities
    and supports exact SHAP attribution at inference time — both critical
    for regulatory audit trails.

Author: Sisirath Chaloor Kuppadan | github.com/sisirathck
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression as SklearnLR
from imblearn.over_sampling import SMOTE
import warnings


class LogisticRegressionScratch:
    """
    NumPy-only implementation of Logistic Regression with gradient descent.

    Implements the standard binary classification formulation:
        z = X @ w + b
        P(y=1|X) = σ(z) = 1 / (1 + e^-z)

    Loss function: Binary Cross-Entropy
        L = -1/N * Σ [ y·log(p) + (1-y)·log(1-p) ]

    Weight updates (gradient descent):
        ∂L/∂w = (1/N) · Xᵀ · (p - y)
        ∂L/∂b = (1/N) · Σ(p - y)
        w ← w - α·∂L/∂w
        b ← b - α·∂L/∂b

    Used to validate that the Scikit-learn production model is performing
    as expected, and to demonstrate full mathematical transparency of the
    decision boundary.
    """

    def __init__(self, learning_rate=0.01, num_iterations=1000):
        """
        Parameters:
            learning_rate (float): Step size for gradient descent updates.
            num_iterations (int):  Number of training epochs.
        """
        self.learning_rate  = learning_rate
        self.num_iterations = num_iterations
        self.weights        = None
        self.bias           = None
        self.loss_history   = []

    def _sigmoid(self, z):
        """Sigmoid activation: maps any real value to (0, 1)."""
        z = np.clip(z, -250, 250)
        return 1 / (1 + np.exp(-z))

    def _compute_loss(self, y_true, y_pred):
        """Binary cross-entropy loss."""
        epsilon = 1e-15
        y_pred  = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def fit(self, X, y):
        """
        Trains the model via batch gradient descent.

        Parameters:
            X: Feature matrix (n_samples × n_features)
            y: Binary target vector (1 = fraud, 0 = legitimate)
        """
        if isinstance(X, (pd.DataFrame, pd.Series)):
            X = X.values
        if isinstance(y, (pd.DataFrame, pd.Series)):
            y = y.values

        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias    = 0

        for i in range(self.num_iterations):
            z      = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(z)

            loss = self._compute_loss(y, y_pred)
            self.loss_history.append(loss)

            dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
            db = (1 / n_samples) * np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw
            self.bias    -= self.learning_rate * db

            if i % (self.num_iterations // 10) == 0:
                print(f"  Iteration {i:4d} | Loss: {loss:.4f}")

        print(f"  ✓ Training complete. Final Loss: {self.loss_history[-1]:.4f}")
        return self

    def predict_proba(self, X):
        """
        Returns fraud probability estimates.

        Returns:
            np.ndarray: Shape (n_samples, 2) — columns [P(legit), P(fraud)]
        """
        if isinstance(X, pd.DataFrame):
            X = X.values

        z          = np.dot(X, self.weights) + self.bias
        prob_fraud = self._sigmoid(z)
        return np.column_stack((1 - prob_fraud, prob_fraud))

    def predict(self, X, threshold=0.5):
        """
        Converts probability estimates to binary labels using a decision threshold.

        Parameters:
            threshold (float): Probability cutoff above which a transaction is
                               classified as fraud. Adjusting this trades
                               precision for recall.
        """
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def handle_class_imbalance(X_train, y_train):
    """
    Applies SMOTE (Synthetic Minority Over-sampling Technique) to address
    the severe class imbalance inherent in fraud datasets (~2% fraud rate).

    A model trained on raw imbalanced data achieves high accuracy by predicting
    'legitimate' for all transactions, effectively catching zero fraud. SMOTE
    generates synthetic minority-class samples in feature space to balance
    the training distribution without simple duplication.

    Parameters:
        X_train: Training feature matrix
        y_train: Training target vector

    Returns:
        Tuple[pd.DataFrame, pd.Series]: Resampled X and y with balanced classes.
    """
    print(f"  Original dataset shape: {y_train.value_counts().to_dict()}")

    smote = SMOTE(random_state=42)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)

    if isinstance(y_resampled, np.ndarray):
        y_resampled = pd.Series(y_resampled, name='is_fraud')
    if isinstance(X_resampled, np.ndarray):
        X_resampled = pd.DataFrame(X_resampled, columns=X_train.columns)

    print(f"  Resampled dataset shape: {y_resampled.value_counts().to_dict()}")
    return X_resampled, y_resampled


def train_sklearn_model(X_train, y_train):
    """
    Trains a Scikit-learn Logistic Regression model for production inference.

    Used as the primary scoring model in the Streamlit dashboard. The scratch
    implementation in LogisticRegressionScratch is retained as a validation
    reference and transparency demonstration.

    Parameters:
        X_train: SMOTE-balanced training features
        y_train: SMOTE-balanced training labels

    Returns:
        sklearn.linear_model.LogisticRegression: Fitted estimator.
    """
    model = SklearnLR(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model
