"""
generate_data.py — FraudShield Dataset Generator
==================================================

Generates a realistic synthetic financial transactions dataset and stores
it in a SQLite database. Run this before executing any analysis scripts.

Usage:
    python data/generate_data.py

Output:
    data/fraudshield.db — SQLite database containing:
      • customers    (10,000 rows) — unique bank customer profiles
      • transactions (100,000 rows) — payment transactions with fraud labels

Design Rationale — Synthetic Data:
    Real fraud datasets contain cardholder PII that cannot be shared publicly.
    Synthetic generation also demonstrates domain expertise: the fraud patterns
    embedded here mirror real financial risk rules used by payment processors.

Fraud Risk Factors Encoded:
    • Late-night transactions (11PM–3AM)  → 4× higher fraud risk
    • Foreign merchant                    → 6× higher fraud risk
    • Electronics / luxury categories     → 3× higher fraud risk
    • High amounts (> ₹10,000)           → 2× higher fraud risk
    • Zero previous transactions in 24h  → 1.5× higher (stolen card pattern)
    • Distance from home > 200km         → 2× higher fraud risk

Author: Sisirath Chaloor Kuppadan
GitHub: https://github.com/sisirathck/fraudshield
"""

import sqlite3
import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta
import random

# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED      = 42
NUM_CUSTOMERS    = 10_000
NUM_TRANSACTIONS = 100_000

np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


# ============================================================
# DATA DEFINITIONS
# ============================================================

# Indian cities (mix of metro and tier-2)
CITIES = [
    "Bengaluru", "Mumbai", "Delhi", "Hyderabad", "Chennai",
    "Kolkata", "Pune", "Ahmedabad", "Jaipur", "Surat",
    "Lucknow", "Nagpur", "Visakhapatnam", "Bhopal", "Kochi",
    "Patna", "Vadodara", "Ludhiana", "Agra", "Nashik"
]

CITY_TO_STATE = {
    "Bengaluru": "Karnataka", "Mumbai": "Maharashtra", "Delhi": "Delhi",
    "Hyderabad": "Telangana", "Chennai": "Tamil Nadu", "Kolkata": "West Bengal",
    "Pune": "Maharashtra", "Ahmedabad": "Gujarat", "Jaipur": "Rajasthan",
    "Surat": "Gujarat", "Lucknow": "Uttar Pradesh", "Nagpur": "Maharashtra",
    "Visakhapatnam": "Andhra Pradesh", "Bhopal": "Madhya Pradesh", "Kochi": "Kerala",
    "Patna": "Bihar", "Vadodara": "Gujarat", "Ludhiana": "Punjab",
    "Agra": "Uttar Pradesh", "Nashik": "Maharashtra"
}

# Merchant categories: typical INR amount and relative fraud risk multiplier
MERCHANT_CATEGORIES = {
    "grocery":       {"typical_amount": 800,   "fraud_multiplier": 0.5},
    "restaurant":    {"typical_amount": 1200,  "fraud_multiplier": 1.0},
    "fuel":          {"typical_amount": 2500,  "fraud_multiplier": 1.3},
    "healthcare":    {"typical_amount": 4000,  "fraud_multiplier": 0.8},
    "education":     {"typical_amount": 8000,  "fraud_multiplier": 0.6},
    "entertainment": {"typical_amount": 2000,  "fraud_multiplier": 1.4},
    "online_retail": {"typical_amount": 3500,  "fraud_multiplier": 2.5},
    "electronics":   {"typical_amount": 18000, "fraud_multiplier": 3.2},
    "travel":        {"typical_amount": 15000, "fraud_multiplier": 2.8},
    "luxury_goods":  {"typical_amount": 30000, "fraud_multiplier": 3.5},
}

MERCHANTS = {
    "grocery":       ["BigBasket", "DMart", "Reliance Fresh", "More Supermarket", "Spencer's", "Zepto"],
    "restaurant":    ["Zomato Order", "Swiggy Order", "Domino's", "McDonald's", "KFC India", "Starbucks"],
    "fuel":          ["Indian Oil", "HP Petrol", "Bharat Petroleum", "Shell India", "BPCL"],
    "healthcare":    ["Apollo Hospital", "Fortis Hospital", "Max Healthcare", "Practo", "PharmEasy"],
    "education":     ["Coursera", "Udemy", "BYJU'S", "Manipal University", "LinkedIn Learning"],
    "entertainment": ["BookMyShow", "PVR Cinemas", "Netflix India", "Amazon Prime", "Spotify India"],
    "online_retail": ["Amazon India", "Flipkart", "Myntra", "Meesho", "Nykaa", "Ajio"],
    "electronics":   ["Croma", "Vijay Sales", "Reliance Digital", "Apple Store India", "Samsung India"],
    "travel":        ["MakeMyTrip", "Goibibo", "IndiGo Airlines", "Air India", "Ola Cabs", "Uber India"],
    "luxury_goods":  ["Tanishq", "Malabar Gold", "Tata CLiQ Luxury", "Nykaa Fashion", "Tiffany & Co"],
}

TRANSACTION_TYPES = {
    "POS":           {"probability": 0.35, "risk_multiplier": 0.8},
    "online":        {"probability": 0.30, "risk_multiplier": 2.2},
    "ATM":           {"probability": 0.10, "risk_multiplier": 1.5},
    "bank_transfer": {"probability": 0.15, "risk_multiplier": 0.6},
    "contactless":   {"probability": 0.10, "risk_multiplier": 1.1},
}

FIRST_NAMES = [
    "Aarav", "Vivaan", "Aditya", "Arjun", "Sai", "Rohan", "Ishaan", "Karan",
    "Priya", "Saanvi", "Ananya", "Diya", "Kavya", "Meera", "Sneha", "Pooja",
    "Rahul", "Amit", "Neha", "Riya", "Divya", "Nikhil", "Deepa", "Rajesh",
    "Suresh", "Lakshmi", "Sunita", "Vijay", "Geeta", "Ravi", "Suman", "Aryan"
]

LAST_NAMES = [
    "Sharma", "Verma", "Patel", "Kumar", "Singh", "Gupta", "Joshi", "Nair",
    "Reddy", "Mishra", "Rao", "Iyer", "Malhotra", "Chopra", "Mehta", "Shah",
    "Chauhan", "Jain", "Agarwal", "Srivastava", "Pandey", "Das", "Bose", "Mukherjee",
    "Krishnan", "Menon", "Pillai", "Naidu", "Chandra", "Saxena", "Tiwari", "Dubey"
]


# ============================================================
# FUNCTION 1: Generate Customer Records
# ============================================================

def generate_customers(n: int) -> pd.DataFrame:
    """
    Creates n synthetic bank customer profiles.

    Maintains a separate customers table to support SQL JOIN queries and
    customer-level fraud analysis — matching real banking system architecture
    where customer profiles are stored separately from transaction ledgers.

    Parameters:
        n (int): Number of customer records to create

    Returns:
        pd.DataFrame: Customer table with one row per unique customer
    """

    print(f"  → Generating {n:,} customer records...")

    customer_ids = [f"C{str(i).zfill(5)}" for i in range(1, n + 1)]

    first = np.random.choice(FIRST_NAMES, n)
    last  = np.random.choice(LAST_NAMES, n)
    names = [f"{f} {l}" for f, l in zip(first, last)]

    ages     = np.clip(np.random.normal(38, 12, n).astype(int), 18, 75)
    genders  = np.random.choice(["Male", "Female"], n, p=[0.53, 0.47])
    cities   = np.random.choice(CITIES, n)
    states   = [CITY_TO_STATE[c] for c in cities]

    account_types = np.random.choice(
        ["savings", "current", "credit"],
        n,
        p=[0.62, 0.28, 0.10]
    )

    credit_scores = np.clip(np.random.normal(700, 85, n).astype(int), 300, 900)

    start_date = datetime(2010, 1, 1)
    total_days = (datetime(2023, 1, 1) - start_date).days
    open_dates = [
        (start_date + timedelta(days=int(d))).strftime('%Y-%m-%d')
        for d in np.random.randint(0, total_days, n)
    ]

    # Log-normal income distribution: most customers earn moderate amounts, a few earn very high
    incomes = np.random.lognormal(mean=10.8, sigma=0.7, size=n).astype(int)
    incomes = np.clip(incomes, 15_000, 800_000)

    customers = pd.DataFrame({
        'customer_id':         customer_ids,
        'full_name':           names,
        'age':                 ages,
        'gender':              genders,
        'city':                cities,
        'state':               states,
        'account_type':        account_types,
        'credit_score':        credit_scores,
        'account_opened_date': open_dates,
        'monthly_income_inr':  incomes,
    })

    print(f"  ✓ Customers ready. Shape: {customers.shape}")
    return customers


# ============================================================
# FUNCTION 2: Compute Fraud Probability
# ============================================================

def compute_fraud_probability(
    amount:          float,
    hour:            int,
    is_foreign:      int,
    distance_km:     float,
    prev_txn_24h:    int,
    category:        str,
    txn_type:        str,
    credit_score:    int
) -> float:
    """
    Computes the probability that a given transaction is fraudulent using
    a log-odds (logistic) approach that mirrors the mathematical structure
    of a Logistic Regression classifier.

    Starts from a baseline fraud rate of ~2% (log-odds = -3.89) and adjusts
    based on domain-driven risk factors. The final log-odds is converted to
    a probability via the sigmoid function: P = 1 / (1 + e^-z)

    This mimics what a trained Logistic Regression model does — the key
    difference is that here the weights are domain-engineered rather than
    learned from data.

    Parameters:
        amount       : Transaction amount in INR
        hour         : Hour of day (0-23)
        is_foreign   : 1 if foreign merchant, 0 if domestic
        distance_km  : Distance from customer's home address in km
        prev_txn_24h : Number of prior transactions in the past 24 hours
        category     : Merchant category (e.g., "electronics", "grocery")
        txn_type     : Payment channel (e.g., "online", "POS")
        credit_score : Customer's CIBIL credit score (300-900)

    Returns:
        float: Fraud probability between 0 and 1
    """

    # Baseline: log-odds for ~2% fraud rate = log(0.02/0.98) ≈ -3.89
    log_odds = -3.89

    # Risk Factor 1: Transaction Amount
    # High-value transactions are preferred targets for card fraud
    if amount > 25000:    log_odds += 2.0
    elif amount > 10000:  log_odds += 1.4
    elif amount > 5000:   log_odds += 0.8
    elif amount > 2000:   log_odds += 0.3
    elif amount < 100:    log_odds += 0.4  # Small "test charge" pattern

    # Risk Factor 2: Time of Day
    # Fraud monitoring coverage is thinnest between 11PM and 3AM
    if   hour in [23, 0, 1, 2, 3]:  log_odds += 1.6
    elif hour in [4, 5, 22]:         log_odds += 0.7
    elif 9 <= hour <= 18:            log_odds -= 0.4  # Business hours: lower risk

    # Risk Factor 3: Foreign Transaction
    # Card issued in India used at foreign merchant = strongest single signal
    if is_foreign:
        log_odds += 2.2

    # Risk Factor 4: Distance from Home
    if   distance_km > 500:  log_odds += 1.4
    elif distance_km > 200:  log_odds += 0.8
    elif distance_km > 100:  log_odds += 0.4

    # Risk Factor 5: Transaction Velocity
    # High frequency within 24h often indicates card compromise
    if   prev_txn_24h == 0:       log_odds += 0.5   # First-use pattern
    elif prev_txn_24h >= 10:      log_odds += 1.8   # Very high velocity
    elif prev_txn_24h >= 6:       log_odds += 1.0
    elif 1 <= prev_txn_24h <= 3:  log_odds -= 0.3   # Normal usage: lower risk

    # Risk Factor 6: Merchant Category
    category_adjustments = {
        "grocery":       -0.6,
        "education":     -0.5,
        "healthcare":    -0.3,
        "restaurant":     0.0,
        "fuel":           0.2,
        "entertainment":  0.4,
        "online_retail":  1.0,
        "travel":         1.1,
        "electronics":    1.4,
        "luxury_goods":   1.6,
    }
    log_odds += category_adjustments.get(category, 0)

    # Risk Factor 7: Payment Channel
    # Card-not-present (online) is the highest-risk channel
    type_adjustments = {
        "bank_transfer": -0.5,
        "POS":           -0.2,
        "contactless":    0.1,
        "ATM":            0.4,
        "online":         0.9,
    }
    log_odds += type_adjustments.get(txn_type, 0)

    # Risk Factor 8: Credit Score
    if   credit_score < 500:  log_odds += 0.5
    elif credit_score > 780:  log_odds -= 0.4

    # Sigmoid: σ(z) = 1 / (1 + e^-z)
    return 1.0 / (1.0 + np.exp(-np.clip(log_odds, -10, 10)))


# ============================================================
# FUNCTION 3: Generate Transaction Records
# ============================================================

def generate_transactions(customers: pd.DataFrame, n: int) -> pd.DataFrame:
    """
    Generates n synthetic payment transactions, each linked to a customer.

    Pipeline:
        1. Assign each transaction to a random customer
        2. Generate timestamps spanning 2 years (Jan 2023 – Dec 2024)
        3. Sample merchant, amount, and transaction type
        4. Compute fraud probability from domain-driven risk factors
        5. Draw fraud label from Bernoulli(p) distribution

    Parameters:
        customers (pd.DataFrame): Customer table from generate_customers()
        n (int): Number of transactions to generate

    Returns:
        pd.DataFrame: Transaction table with one row per transaction
    """

    print(f"  → Generating {n:,} transactions...")

    cust_idx = np.random.randint(0, len(customers), n)
    selected = customers.iloc[cust_idx].reset_index(drop=True)

    txn_ids = [f"T{str(i).zfill(7)}" for i in range(1, n + 1)]

    # Timestamps: uniformly distributed across 2 years
    start_ts   = datetime(2023, 1, 1)
    end_ts     = datetime(2024, 12, 31, 23, 59, 59)
    range_secs = int((end_ts - start_ts).total_seconds())

    random_secs = np.random.randint(0, range_secs, n)
    timestamps  = [
        (start_ts + timedelta(seconds=int(s))).strftime('%Y-%m-%d %H:%M:%S')
        for s in random_secs
    ]

    ts_series  = pd.to_datetime(timestamps)
    hours      = ts_series.hour.values
    dow        = ts_series.dayofweek.values
    is_weekend = (dow >= 5).astype(int)

    categories     = np.random.choice(list(MERCHANT_CATEGORIES.keys()), n)
    merchant_names = [np.random.choice(MERCHANTS[cat]) for cat in categories]

    # Log-normal amount distribution per category
    amounts = np.array([
        round(max(10.0, np.random.lognormal(
            mean=np.log(MERCHANT_CATEGORIES[cat]["typical_amount"]),
            sigma=0.65
        )), 2)
        for cat in categories
    ])

    type_names = list(TRANSACTION_TYPES.keys())
    type_probs = [TRANSACTION_TYPES[t]["probability"] for t in type_names]
    txn_types  = np.random.choice(type_names, n, p=type_probs)

    is_foreign = np.random.binomial(1, 0.05, n)

    # Exponential distance distribution: most transactions are nearby
    distances = np.random.exponential(scale=25, size=n)
    distances = np.where(is_foreign,
                         np.random.uniform(800, 15000, n),
                         distances)
    distances = distances.round(1)

    prev_24h = np.clip(np.random.poisson(lam=2, size=n), 0, 25)

    print("  → Computing fraud probability for each transaction...")
    fraud_probs  = np.zeros(n)
    fraud_labels = np.zeros(n, dtype=int)

    for i in range(n):
        p = compute_fraud_probability(
            amount       = amounts[i],
            hour         = hours[i],
            is_foreign   = is_foreign[i],
            distance_km  = distances[i],
            prev_txn_24h = prev_24h[i],
            category     = categories[i],
            txn_type     = txn_types[i],
            credit_score = int(selected.iloc[i]['credit_score'])
        )
        fraud_probs[i]  = round(p, 5)
        fraud_labels[i] = np.random.binomial(1, p)

    transactions = pd.DataFrame({
        'transaction_id':            txn_ids,
        'customer_id':               selected['customer_id'].values,
        'timestamp':                 timestamps,
        'transaction_hour':          hours,
        'day_of_week':               dow,
        'is_weekend':                is_weekend,
        'amount':                    amounts,
        'merchant_name':             merchant_names,
        'merchant_category':         categories,
        'transaction_type':          txn_types,
        'is_foreign':                is_foreign,
        'distance_from_home_km':     distances,
        'num_prev_transactions_24h': prev_24h,
        'fraud_probability':         fraud_probs,
        'is_fraud':                  fraud_labels,
    })

    fraud_rate = fraud_labels.mean() * 100
    print(f"  ✓ Transactions ready. Fraud rate: {fraud_rate:.2f}% ({fraud_labels.sum():,} fraud cases)")
    return transactions


# ============================================================
# FUNCTION 4: Save to SQLite Database
# ============================================================

def save_to_database(customers: pd.DataFrame, transactions: pd.DataFrame, db_path: str):
    """
    Saves customer and transaction DataFrames to a SQLite database file.

    Creates database indexes on frequently queried columns (customer_id,
    is_fraud, merchant_category, timestamp) for fast analytical query performance.

    Parameters:
        customers    : Customer profiles DataFrame
        transactions : Transaction records DataFrame
        db_path      : Output path for the .db file
    """

    print(f"\n  → Saving to database: {db_path}")

    conn = sqlite3.connect(db_path)
    customers.to_sql('customers', conn, if_exists='replace', index=False)
    print(f"  ✓ Table 'customers' saved  ({len(customers):,} rows)")

    transactions.to_sql('transactions', conn, if_exists='replace', index=False)
    print(f"  ✓ Table 'transactions' saved ({len(transactions):,} rows)")

    # Create indexes for analytical query performance
    print("  → Creating indexes...")
    cursor = conn.cursor()
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_txn_customer  ON transactions(customer_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_txn_fraud     ON transactions(is_fraud)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_txn_category  ON transactions(merchant_category)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_txn_timestamp ON transactions(timestamp)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_cust_id       ON customers(customer_id)")
    conn.commit()
    print("  ✓ Indexes created")

    # Verification
    cursor.execute("SELECT COUNT(*) FROM customers")
    n_cust = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM transactions")
    n_txn = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(*) FROM transactions WHERE is_fraud = 1")
    n_fraud = cursor.fetchone()[0]

    conn.close()

    fraud_pct = n_fraud / n_txn * 100
    legit_pct = 100 - fraud_pct

    print(f"\n  {'─'*45}")
    print(f"  DATABASE SUMMARY")
    print(f"  {'─'*45}")
    print(f"  Customers:          {n_cust:>10,}")
    print(f"  Transactions:       {n_txn:>10,}")
    print(f"  Fraudulent:         {n_fraud:>10,}   ({fraud_pct:.2f}%)")
    print(f"  Legitimate:         {n_txn-n_fraud:>10,}   ({legit_pct:.2f}%)")
    print(f"  {'─'*45}")
    print(f"  File: {os.path.abspath(db_path)}")


# ============================================================
# MAIN — Entry Point
# ============================================================

def main():
    """
    Executes the full data generation pipeline:
        Generate Customers → Generate Transactions → Save to Database
    """

    print("=" * 55)
    print("  🛡️  FRAUDSHIELD — Data Generation Pipeline")
    print("=" * 55)

    this_dir    = os.path.dirname(os.path.abspath(__file__))
    project_dir = os.path.dirname(this_dir)

    os.makedirs(os.path.join(project_dir, 'outputs', 'figures'), exist_ok=True)

    db_path = os.path.join(this_dir, 'fraudshield.db')

    print("\n[STEP 1] Generating customer data...")
    customers_df = generate_customers(NUM_CUSTOMERS)

    print("\n[STEP 2] Generating transaction data...")
    transactions_df = generate_transactions(customers_df, NUM_TRANSACTIONS)

    print("\n[STEP 3] Saving to SQLite database...")
    save_to_database(customers_df, transactions_df, db_path)

    print("\n" + "=" * 55)
    print("  ✅ Data generation complete!")
    print("\n  Run scripts in order:")
    print("  1.  python analysis/01_eda.py")
    print("  2.  python analysis/02_sql_analysis.py")
    print("  3.  python analysis/03_ml_modeling.py")
    print("  4.  streamlit run app.py")
    print("=" * 55)


if __name__ == "__main__":
    main()
