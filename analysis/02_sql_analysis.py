"""
analysis/02_sql_analysis.py — SQL-Based Data Analysis
=====================================================

PURPOSE:
    Demonstrates how to use SQL directly from Python.
    While pandas (used in 01_eda.py) is great for data stored in memory,
    SQL is essential when datasets are too large to fit in RAM.
    
    This script extracts results from specific queries defined in
    sql/analysis_queries.sql and visualizes them.

AUTHOR: Sisirath Chaloor Kuppadan | github.com/sisirathck
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data_loader import run_query
from src.visualizations import set_professional_style, COLOR_LEGIT, COLOR_FRAUD, COLOR_ACCENT

OUTPUT_DIR = os.path.join('outputs', 'figures', 'sql')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def plot_sql_fraud_by_distance():
    print("  → Running SQL: Fraud Rate by Distance...")
    sql = """
    SELECT
        CASE
            WHEN distance_from_home_km < 5      THEN '0–5km   (Home area)'
            WHEN distance_from_home_km < 25     THEN '5–25km  (Local)'
            WHEN distance_from_home_km < 100    THEN '25–100km (City/Region)'
            WHEN distance_from_home_km < 500    THEN '100–500km (State/Neighbour)'
            ELSE                                     '500km+  (Far / International)'
        END AS distance_bucket,
        ROUND(AVG(is_fraud) * 100, 2)   AS fraud_rate_pct
    FROM transactions
    GROUP BY distance_bucket
    ORDER BY MIN(distance_from_home_km);
    """
    df = run_query(sql)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x='distance_bucket', y='fraud_rate_pct', data=df, color=COLOR_LEGIT, ax=ax)
    
    set_professional_style(ax, 'Fraud Rate by Distance from Home (SQL)', 'Distance Bucket', 'Fraud Rate (%)')
    plt.xticks(rotation=45, ha='right')
    
    # Add values on top of bars
    for i, p in enumerate(ax.patches):
        ax.annotate(f'{p.get_height():.1f}%', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '01_sql_distance_fraud.png'), dpi=300)
    plt.close()


def plot_sql_fraud_by_velocity():
    print("  → Running SQL: Fraud Rate by Velocity...")
    sql = """
    SELECT
        CASE
            WHEN num_prev_transactions_24h = 0   THEN '0 — First txn today'
            WHEN num_prev_transactions_24h <= 2  THEN '1–2 — Normal frequency'
            WHEN num_prev_transactions_24h <= 5  THEN '3–5 — Active user'
            WHEN num_prev_transactions_24h <= 9  THEN '6–9 — Very active'
            ELSE                                      '10+ — High velocity'
        END AS velocity_bucket,
        ROUND(AVG(is_fraud) * 100, 2)   AS fraud_rate_pct
    FROM transactions
    GROUP BY velocity_bucket
    ORDER BY MIN(num_prev_transactions_24h);
    """
    df = run_query(sql)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Highlight the high velocity bucket
    pal = [COLOR_FRAUD if '10+' in label else COLOR_LEGIT for label in df['velocity_bucket']]
    sns.barplot(x='velocity_bucket', y='fraud_rate_pct', data=df, palette=pal, ax=ax)
    
    set_professional_style(ax, 'Fraud Rate by 24h Transaction Velocity (SQL)', 'Transactions in past 24h', 'Fraud Rate (%)')
    plt.xticks(rotation=45, ha='right')
    
    for i, p in enumerate(ax.patches):
        ax.annotate(f'{p.get_height():.1f}%', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='bottom', fontweight='bold')
        
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '02_sql_velocity_fraud.png'), dpi=300)
    plt.close()


def print_executive_summary():
    print("  → Running SQL: Executive Dashboard Summary...")
    sql = """
    SELECT 'Total Transactions Processed'  AS metric, CAST(COUNT(*) AS TEXT) AS value FROM transactions
    UNION ALL
    SELECT 'Fraudulent Transactions', CAST(SUM(is_fraud) AS TEXT) FROM transactions
    UNION ALL
    SELECT 'Overall Fraud Rate', ROUND(AVG(is_fraud) * 100, 2) || '%' FROM transactions
    UNION ALL
    SELECT 'Total Fraud Value (₹)', '₹' || CAST(ROUND(SUM(CASE WHEN is_fraud=1 THEN amount ELSE 0 END)) AS TEXT) FROM transactions
    """
    df = run_query(sql)
    
    print("\n  " + "="*45)
    print("  📈 EXECUTIVE SQL SUMMARY")
    print("  " + "="*45)
    for index, row in df.iterrows():
        print(f"  {row['metric']:<30} : {row['value']:>10}")
    print("  " + "="*45 + "\n")


def main():
    print("\n" + "="*50)
    print("  🗄️  Running SQL Analytics (02_sql_analysis.py)")
    print("="*50)
    
    plot_sql_fraud_by_distance()
    plot_sql_fraud_by_velocity()
    print_executive_summary()
    
    print("  ✅ SQL Analytics Complete! Charts saved to outputs/figures/sql/\n")

if __name__ == "__main__":
    main()
