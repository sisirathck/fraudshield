"""
src/feature_engineering.py — ML Feature Creation
==================================================

Transforms raw transaction and customer records into a machine-learning-ready
feature matrix. All features are designed from financial domain knowledge
rather than generic statistical transformations.

Feature engineering decisions:
  - Temporal flags capture the documented risk elevation in late-night hours
    (11PM–3AM) when fraud monitoring staffing is reduced.
  - The velocity×amount interaction term captures the compounding risk of
    high transaction frequency combined with high individual amounts — a
    pattern strongly associated with card-compromise draining.
  - One-hot encoding of merchant_category and transaction_type preserves
    the domain-specific fraud risk differentials across payment contexts.

Author: Sisirath Chaloor Kuppadan | github.com/sisirathck
"""

import pandas as pd
import numpy as np
import warnings

warnings.filterwarnings('ignore', category=FutureWarning)


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates binary temporal risk flags from transaction timestamps.

    Late-night hours (11PM–3AM) carry elevated fraud risk due to reduced
    real-time monitoring coverage at payment processors and card issuers.

    Parameters:
        df (pd.DataFrame): Transaction DataFrame with 'transaction_hour' column.

    Returns:
        pd.DataFrame: DataFrame with 'is_late_night' feature appended.
    """
    df['is_late_night'] = df['transaction_hour'].isin([23, 0, 1, 2, 3]).astype(int)
    return df


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates interaction features that capture compounding risk signals.

    High transaction velocity alone and high transaction amounts alone are
    moderate risk signals. Their product captures the card-draining pattern —
    a stolen card used rapidly for high-value purchases before being blocked.

    Parameters:
        df (pd.DataFrame): Transaction DataFrame with amount and velocity columns.

    Returns:
        pd.DataFrame: DataFrame with 'velocity_amount_interaction' appended.
    """
    df['velocity_amount_interaction'] = df['num_prev_transactions_24h'] * df['amount']
    return df


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies one-hot encoding to merchant_category and transaction_type.

    drop_first=True prevents multicollinearity in the Logistic Regression
    design matrix (the dummy variable trap).

    Parameters:
        df (pd.DataFrame): Transaction DataFrame with categorical columns.

    Returns:
        pd.DataFrame: DataFrame with dummy variable columns appended.
    """
    cat_dummies  = pd.get_dummies(df['merchant_category'], prefix='category', drop_first=True, dtype=int)
    type_dummies = pd.get_dummies(df['transaction_type'],  prefix='type',     drop_first=True, dtype=int)
    df = pd.concat([df, cat_dummies, type_dummies], axis=1)
    return df


def prepare_ml_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master feature engineering pipeline. Runs all transformation steps and
    returns a scaled feature matrix X and target vector y.

    Feature scaling (StandardScaler) is applied to continuous columns to ensure
    gradient descent convergence — without scaling, features with large magnitudes
    (e.g. amount in INR) dominate the weight updates.

    Parameters:
        df (pd.DataFrame): Joined transaction + customer DataFrame.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: (X scaled features, y target labels)
    """
    from sklearn.preprocessing import StandardScaler

    print("  → Engineering temporal features...")
    df = create_temporal_features(df)

    print("  → Engineering interaction features...")
    df = create_interaction_features(df)

    print("  → Encoding categorical variables...")
    df = encode_categorical_features(df)

    base_features = [
        'amount',
        'is_foreign',
        'distance_from_home_km',
        'num_prev_transactions_24h',
        'is_weekend',
        'is_late_night',
        'velocity_amount_interaction'
    ]

    dummy_cols   = [c for c in df.columns if c.startswith('category_') or c.startswith('type_')]
    all_features = base_features + dummy_cols

    print(f"  ✓ Feature engineering complete: {len(all_features)} ML features generated.")

    X = df[all_features].copy()
    y = df['is_fraud'].copy()

    scaler          = StandardScaler()
    continuous_cols = ['amount', 'distance_from_home_km', 'num_prev_transactions_24h', 'velocity_amount_interaction']
    X[continuous_cols] = scaler.fit_transform(X[continuous_cols])

    return X, y
