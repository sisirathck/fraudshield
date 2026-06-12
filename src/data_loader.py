"""
src/data_loader.py — Database Loading Functions
================================================

Centralises all SQLite connection and query logic for the FraudShield project.
Ensures a single point of maintenance for database paths and type conversions.

Author: Sisirath Chaloor Kuppadan | github.com/sisirathck
"""

import sqlite3
import pandas as pd
import os


# Resolve database path relative to this module's location
_THIS_DIR    = os.path.dirname(os.path.abspath(__file__))   # .../fraudshield/src/
_PROJECT_DIR = os.path.dirname(_THIS_DIR)                   # .../fraudshield/
DB_PATH      = os.path.join(_PROJECT_DIR, 'data', 'fraudshield.db')


def get_db_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """
    Returns an open SQLite connection to the FraudShield database.

    Parameters:
        db_path (str): Path to the .db file. Defaults to the project database.

    Returns:
        sqlite3.Connection: Active connection object.

    Raises:
        FileNotFoundError: If the database has not been generated yet.
    """
    if not os.path.exists(db_path):
        raise FileNotFoundError(
            f"\n❌ Database not found at: {db_path}\n"
            "   Run:  python data/generate_data.py\n"
        )
    return sqlite3.connect(db_path)


def load_transactions(db_path: str = DB_PATH, limit: int = None) -> pd.DataFrame:
    """
    Loads the transactions table into a pandas DataFrame.

    Parameters:
        db_path (str): Path to the database. Default: project database.
        limit (int):   If provided, restricts the result to the first N rows.

    Returns:
        pd.DataFrame: Transaction records with correctly typed columns.
    """
    conn = get_db_connection(db_path)

    query = "SELECT * FROM transactions"
    if limit:
        query += f" LIMIT {limit}"

    df = pd.read_sql(query, conn)
    conn.close()

    df['timestamp'] = pd.to_datetime(df['timestamp'])

    for col in ['is_fraud', 'is_foreign', 'is_weekend']:
        df[col] = df[col].astype(int)

    print(f"  ✓ Loaded {len(df):,} transactions  | Fraud: {df['is_fraud'].sum():,} ({df['is_fraud'].mean()*100:.2f}%)")
    return df


def load_customers(db_path: str = DB_PATH) -> pd.DataFrame:
    """
    Loads the customers table into a pandas DataFrame.

    Returns:
        pd.DataFrame: Customer profile records.
    """
    conn = get_db_connection(db_path)
    df   = pd.read_sql("SELECT * FROM customers", conn)
    conn.close()

    df['account_opened_date'] = pd.to_datetime(df['account_opened_date'])

    print(f"  ✓ Loaded {len(df):,} customer records")
    return df


def load_joined(db_path: str = DB_PATH) -> pd.DataFrame:
    """
    Returns transactions INNER JOINed with customer profiles — a denormalised
    view used for feature engineering and ML model training.

    Returns:
        pd.DataFrame: Combined transaction + customer table.
    """
    conn = get_db_connection(db_path)

    query = """
        SELECT
            t.*,
            c.full_name,
            c.age,
            c.gender,
            c.city,
            c.state,
            c.account_type,
            c.credit_score,
            c.account_opened_date,
            c.monthly_income_inr
        FROM transactions t
        INNER JOIN customers c
            ON t.customer_id = c.customer_id
    """

    df = pd.read_sql(query, conn)
    conn.close()

    df['timestamp']           = pd.to_datetime(df['timestamp'])
    df['account_opened_date'] = pd.to_datetime(df['account_opened_date'])

    print(f"  ✓ Loaded joined table: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


def run_query(sql: str, db_path: str = DB_PATH) -> pd.DataFrame:
    """
    Executes an arbitrary SQL SELECT statement and returns the result.

    Parameters:
        sql (str):     A valid SQL SELECT statement.
        db_path (str): Path to the database.

    Returns:
        pd.DataFrame: Query result.
    """
    conn = get_db_connection(db_path)
    df   = pd.read_sql(sql, conn)
    conn.close()
    return df


def get_table_info(db_path: str = DB_PATH) -> None:
    """
    Prints a summary of all tables and their columns in the database.
    """
    conn   = get_db_connection(db_path)
    cursor = conn.cursor()

    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]

    print("\n📋 DATABASE STRUCTURE")
    print("─" * 40)

    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]

        cursor.execute(f"PRAGMA table_info({table})")
        columns  = cursor.fetchall()
        col_names = [col[1] for col in columns]

        print(f"\n  Table: {table}")
        print(f"  Rows : {count:,}")
        print(f"  Cols : {', '.join(col_names)}")

    conn.close()
