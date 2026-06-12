"""
app.py — Streamlit Web Dashboard
================================

PURPOSE:
    The user-facing web interface for FraudShield. Provides:
    1. 🏠 Project Overview & Platform Metrics
    2. 📊 Interactive Exploratory Data Analysis (EDA) using Plotly
    3. 🗄️ Live SQL Query Console executing SQLite queries on 100k transactions
    4. 🤖 Model Diagnostics comparing NumPy Scratch vs. Scikit-learn models
    5. 🔍 Real-Time Simulator scoring transactions and explaining risk via SHAP

AUTHOR: Sisirath Chaloor Kuppadan | github.com/sisirathck
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Page settings
st.set_page_config(
    page_title="FraudShield — Payment Fraud Detection Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global Styling (Premium Dark/Slate Theme Elements) ──────────
st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Inter:wght@300;400;500;600;700&display=swap');

        /* Global Font Overrides */
        html, body, [class*="css"], .stMarkdown {
            font-family: 'Inter', sans-serif !important;
        }

        h1, h2, h3, h4, h5, h6, .main-header, .gradient-text {
            font-family: 'Outfit', sans-serif !important;
        }

        /* Scrollbar styling */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        ::-webkit-scrollbar-track {
            background: #0B0F19;
        }
        ::-webkit-scrollbar-thumb {
            background: #1E293B;
            border-radius: 4px;
        }
        ::-webkit-scrollbar-thumb:hover {
            background: #334155;
        }

        /* CSS Variables for theme colors */
        :root {
            --primary-color: #06B6D4;     /* Electric Cyan */
            --accent-color: #F59E0B;      /* Warm Amber */
            --fraud-color: #EF4444;       /* Neon Crimson */
            --legit-color: #10B981;       /* Emerald Safe */
            --bg-card: rgba(15, 23, 42, 0.65);
            --border-card: rgba(255, 255, 255, 0.05);
        }

        /* Premium Main Header styling */
        .main-header {
            font-size: 2.8rem;
            font-weight: 800;
            background: linear-gradient(135deg, #3B82F6 0%, #06B6D4 50%, #10B981 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            margin-bottom: 0.1rem;
            letter-spacing: -0.02em;
        }

        .sub-header {
            font-size: 1.15rem;
            color: #94A3B8;
            margin-bottom: 2rem;
            font-weight: 400;
        }

        /* Custom premium container card */
        .glass-card {
            background: var(--bg-card);
            backdrop-filter: blur(16px);
            -webkit-backdrop-filter: blur(16px);
            border: 1px solid var(--border-card);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.4);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }
        .glass-card:hover {
            transform: translateY(-2px);
            border-color: rgba(6, 182, 212, 0.2);
            box-shadow: 0 10px 30px rgba(0, 0, 0, 0.35);
        }

        /* Custom Metric Cards */
        .custom-metric-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 1.25rem;
            margin-bottom: 2rem;
        }

        .custom-metric {
            background: rgba(30, 41, 59, 0.5);
            backdrop-filter: blur(8px);
            -webkit-backdrop-filter: blur(8px);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 14px;
            padding: 1.25rem 1.5rem;
            box-shadow: 0 4px 25px rgba(0, 0, 0, 0.15);
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
            position: relative;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }

        .custom-metric::before {
            content: '';
            position: absolute;
            top: 0;
            left: 0;
            width: 4px;
            height: 100%;
            background: var(--metric-color, #3B82F6);
        }

        .custom-metric:hover {
            transform: translateY(-3px);
            box-shadow: 0 8px 30px rgba(0, 0, 0, 0.25);
            border-color: rgba(255, 255, 255, 0.12);
        }

        .custom-metric-value {
            font-size: 2.1rem;
            font-weight: 800;
            color: #F8FAFC;
            line-height: 1.2;
            margin-top: 0.25rem;
            font-family: 'Outfit', sans-serif !important;
        }

        .custom-metric-label {
            font-size: 0.85rem;
            font-weight: 600;
            color: #94A3B8;
            text-transform: uppercase;
            letter-spacing: 0.07em;
            display: flex;
            align-items: center;
            gap: 0.5rem;
        }

        .custom-metric-label span.icon {
            font-size: 1.1rem;
        }

        /* Profile Card Styling */
        .profile-card {
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.8) 0%, rgba(30, 41, 59, 0.5) 100%);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
            padding: 2rem;
            box-shadow: 0 10px 40px rgba(0, 0, 0, 0.3);
            transition: all 0.3s ease;
            margin-bottom: 2rem;
        }
        .profile-card:hover {
            border-color: rgba(6, 182, 212, 0.3);
            box-shadow: 0 15px 50px rgba(6, 182, 212, 0.1);
        }

        /* Badge list tags */
        .badge-pill {
            display: inline-block;
            padding: 0.25rem 0.75rem;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            background: rgba(6, 182, 212, 0.12);
            color: #06B6D4;
            border: 1px solid rgba(6, 182, 212, 0.2);
            margin-right: 0.5rem;
            margin-bottom: 0.5rem;
            transition: all 0.2s ease;
        }
        .badge-pill:hover {
            background: rgba(6, 182, 212, 0.2);
            border-color: rgba(6, 182, 212, 0.4);
        }

        /* Social link buttons */
        .social-links {
            display: flex;
            gap: 0.75rem;
            margin-top: 1.25rem;
            flex-wrap: wrap;
        }
        .social-btn {
            display: inline-flex;
            align-items: center;
            gap: 0.5rem;
            padding: 0.5rem 1rem;
            border-radius: 8px;
            font-size: 0.85rem;
            font-weight: 500;
            text-decoration: none !important;
            transition: all 0.2s ease;
            border: 1px solid rgba(255, 255, 255, 0.08);
            color: #E2E8F0 !important;
            background: rgba(30, 41, 59, 0.6);
        }
        .social-btn:hover {
            transform: translateY(-2px);
            background: rgba(51, 65, 85, 0.8);
            border-color: rgba(255, 255, 255, 0.15);
        }
        .social-btn.linkedin:hover {
            border-color: #0A66C2;
            box-shadow: 0 0 10px rgba(10, 102, 194, 0.2);
        }
        .social-btn.github:hover {
            border-color: #E2E8F0;
            box-shadow: 0 0 10px rgba(255, 255, 255, 0.1);
        }
        .social-btn.email:hover {
            border-color: #EA4335;
            box-shadow: 0 0 10px rgba(234, 67, 53, 0.2);
        }

        /* Database schema layout styling */
        .db-table-badge {
            font-family: monospace;
            font-weight: bold;
            color: #F59E0B;
            background: rgba(245, 158, 11, 0.1);
            border: 1px solid rgba(245, 158, 11, 0.2);
            padding: 2px 6px;
            border-radius: 4px;
        }

        /* Math Explanation block styling */
        .math-card {
            background: rgba(15, 23, 42, 0.6);
            border: 1px solid rgba(255, 255, 255, 0.06);
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1rem;
        }

        /* Streamlit Button overrides */
        .stButton > button {
            background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%) !important;
            color: #E2E8F0 !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 8px !important;
            padding: 0.5rem 1.5rem !important;
            font-weight: 600 !important;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
            box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;
        }

        .stButton > button:hover {
            transform: translateY(-2px) !important;
            border-color: var(--primary-color) !important;
            box-shadow: 0 6px 15px rgba(6, 182, 212, 0.2) !important;
            color: #FFFFFF !important;
        }

        .stButton > button:active {
            transform: translateY(0px) !important;
        }

        /* Active tab styling overrides */
        button[data-baseweb="tab"] {
            background: transparent !important;
            color: #94A3B8 !important;
            border-bottom: 2px solid transparent !important;
            font-size: 1rem !important;
            font-weight: 600 !important;
            padding: 12px 24px !important;
            transition: all 0.25s ease !important;
        }

        button[data-baseweb="tab"]:hover {
            color: #F8FAFC !important;
        }

        button[data-baseweb="tab"][aria-selected="true"] {
            color: #06B6D4 !important;
            border-bottom-color: #06B6D4 !important;
            font-size: 1.05rem !important;
        }

        /* Custom Alert / Status containers */
        .status-approved {
            background-color: rgba(16, 185, 129, 0.08);
            border: 1px solid rgba(16, 185, 129, 0.2);
            box-shadow: 0 0 20px rgba(16, 185, 129, 0.05);
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
            transition: all 0.3s ease;
        }
        .status-approved:hover {
            border-color: rgba(16, 185, 129, 0.4);
            box-shadow: 0 0 30px rgba(16, 185, 129, 0.15);
        }

        .status-declined {
            background-color: rgba(239, 68, 68, 0.08);
            border: 1px solid rgba(239, 68, 68, 0.2);
            box-shadow: 0 0 20px rgba(239, 68, 68, 0.05);
            padding: 1.5rem;
            border-radius: 12px;
            text-align: center;
            transition: all 0.3s ease;
        }
        .status-declined:hover {
            border-color: rgba(239, 68, 68, 0.4);
            box-shadow: 0 0 30px rgba(239, 68, 68, 0.15);
        }
    </style>
""", unsafe_allow_html=True)

# ── Import local project modules ─────────────────────────────────
import sys
sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.data_loader import load_joined, run_query
from src.feature_engineering import prepare_ml_dataset, create_temporal_features, create_interaction_features
from src.model import train_sklearn_model, handle_class_imbalance
from src.visualizations import COLOR_LEGIT, COLOR_FRAUD, COLOR_ACCENT

# ── Plotly Custom Theme Application ─────────────────────────────
def apply_plotly_theme(fig, title=None):
    fig.update_layout(
        template="plotly_dark",
        title={
            'text': f"<b>{title}</b>" if title else None,
            'y': 0.95,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': {'family': 'Outfit, sans-serif', 'size': 16}
        } if title else None,
        font=dict(
            family="Inter, sans-serif",
            size=11,
            color="#CBD5E1"
        ),
        paper_bgcolor="rgba(0, 0, 0, 0)",
        plot_bgcolor="rgba(30, 41, 59, 0.2)",
        margin=dict(l=40, r=40, t=60, b=40),
        xaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.05)",
            zerolinecolor="rgba(255, 255, 255, 0.1)",
            linecolor="rgba(255, 255, 255, 0.1)"
        ),
        yaxis=dict(
            gridcolor="rgba(255, 255, 255, 0.05)",
            zerolinecolor="rgba(255, 255, 255, 0.1)",
            linecolor="rgba(255, 255, 255, 0.1)"
        )
    )
    return fig

# ── CACHED MACHINE LEARNING PIPELINE ─────────────────────────────
@st.cache_resource
def load_and_train_pipeline():
    """
    Loads data from SQLite and trains the Scikit-learn Logistic Regression model.
    Caches the results so the web app starts instantly after the first load.
    
    Why cache?
    Streamlit re-runs the entire file on every user interaction (slider change, click).
    If we retrained the model on 100,000 rows every time, the app would lag terribly.
    @st.cache_resource keeps the trained model in memory.
    """
    # 1. Load the merged transaction + customer data
    df = load_joined()
    
    # 2. Run feature engineering and scale columns
    # We pass a copy to avoid mutating the original DataFrame in memory
    X, y = prepare_ml_dataset(df.copy())
    
    # 3. Train/Test Split (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # 4. Handle class imbalance using SMOTE
    X_train_bal, y_train_bal = handle_class_imbalance(X_train, y_train)
    
    # 5. Train Scikit-learn Logistic Regression model
    model = train_sklearn_model(X_train_bal, y_train_bal)
    
    # 6. Fit a separate StandardScaler on the raw continuous columns
    # We do this because we need a standalone scaler to process raw user inputs
    # in the simulator (Tab 5) before sending them to the model.
    continuous_cols = ['amount', 'distance_from_home_km', 'num_prev_transactions_24h', 'velocity_amount_interaction']
    raw_df = df.copy()
    raw_df = create_temporal_features(raw_df)
    raw_df = create_interaction_features(raw_df)
    
    scaler = StandardScaler()
    scaler.fit(raw_df[continuous_cols])
    
    # 7. Precompute feature means on the balanced training data
    # Needed for our mathematically exact local SHAP values
    feature_means = X_train_bal.mean()
    
    return {
        'model': model,
        'scaler': scaler,
        'feature_means': feature_means,
        'X_train_bal': X_train_bal,
        'X_test': X_test,
        'y_test': y_test,
        'df': df,
        'features': list(X.columns)
    }

# Auto-generate database if it's missing (essential for hosted environments like Streamlit Cloud where fraudshield.db is gitignored)
db_file_path = os.path.join(os.path.dirname(__file__), 'data', 'fraudshield.db')
if not os.path.exists(db_file_path):
    st.info("📦 SQLite database not found. Generating 100,000 transaction records. Please wait...")
    try:
        import importlib
        data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data'))
        if data_dir not in sys.path:
            sys.path.insert(0, data_dir)
        generate_data = importlib.import_module('generate_data')
        importlib.reload(generate_data)  # force re-execution, not cached version
        generate_data.main()
        st.success("✓ Database generated successfully!")
    except Exception as e:
        st.error(f"Failed to auto-generate database: {e}")

# Load pipeline
with st.spinner("🚀 Initializing FraudShield Pipeline & Training Models (this takes ~5 seconds)..."):
    pipeline = load_and_train_pipeline()

df = pipeline['df']
model = pipeline['model']
scaler = pipeline['scaler']
feature_means = pipeline['feature_means']
X_test = pipeline['X_test']
y_test = pipeline['y_test']
X_train_bal = pipeline['X_train_bal']
features = pipeline['features']

# Calculate high-level stats
total_txn = len(df)
total_fraud = df['is_fraud'].sum()
fraud_rate = df['is_fraud'].mean() * 100
total_customers = df['customer_id'].nunique()

# ── SIDEBAR: Portfolio & Professional Bio ───────────────────────
with st.sidebar:
    st.markdown("""
        <div style="text-align: center; margin-bottom: 2rem; margin-top: 1rem;">
            <h1 style="color: #06B6D4; font-family: 'Outfit', sans-serif; font-size: 2rem; font-weight: 800; margin: 0; letter-spacing: -0.02em;">
                FraudShield
            </h1>
            <div style="font-size: 0.72rem; color: #64748B; text-transform: uppercase; letter-spacing: 0.2em; margin-top: 0.35rem; font-weight: 700;">
                Risk Control Platform
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div class="glass-card" style="padding: 1.25rem; margin-bottom: 1.5rem; background: rgba(15, 23, 42, 0.6); border: 1px solid rgba(255,255,255,0.05);">
            <div style="font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1rem; color: #64748B; margin-bottom: 0.75rem;">
                Developer Profile
            </div>
            <div style="font-size: 0.95rem; color: #F8FAFC; font-weight: 700;">Sisirath Chaloor Kuppadan</div>
            <div style="font-size: 0.75rem; color: #06B6D4; margin-bottom: 0.75rem; font-weight: 500;">Risk Specialist & Data Scientist</div>
            <div style="font-size: 0.75rem; color: #94A3B8; line-height: 1.4;">
                3+ years auditing and automating transaction controls at <b>EY</b> and <b>RELX Group</b>.
            </div>
            <div class="social-links" style="margin-top: 0.85rem; gap: 0.4rem;">
                <a class="social-btn linkedin" href="https://linkedin.com/in/sisirath" target="_blank" style="padding: 0.3rem 0.6rem; font-size: 0.7rem; border-radius: 4px;">
                    LinkedIn
                </a>
                <a class="social-btn github" href="https://github.com/sisirathck/fraudshield" target="_blank" style="padding: 0.3rem 0.6rem; font-size: 0.7rem; border-radius: 4px;">
                    GitHub
                </a>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
        <div style="margin-top: 1.5rem; margin-bottom: 0.5rem; font-size: 0.75rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.1rem; color: #64748B;">
            Engine Configuration
        </div>
    """, unsafe_allow_html=True)
    
    decision_threshold = st.slider(
        "Fraud Classification Threshold",
        min_value=0.05, max_value=0.95, value=0.50, step=0.05,
        help="Lowering threshold catches MORE fraud (higher Recall) but increases false alarms (lower Precision)."
    )

# ── MAIN LAYOUT ──────────────────────────────────────────────────
st.markdown('<div class="main-header">FraudShield</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Enterprise risk intelligence portal & real-time transaction auditing engine</div>', unsafe_allow_html=True)

# Tabs setup
tab_overview, tab_eda, tab_sql, tab_diagnostics, tab_simulator = st.tabs([
    "Overview",
    "Exploratory Analysis",
    "SQL Console",
    "Model Diagnostics",
    "Real-time Simulator"
])

# 🚀 TAB 1: PROJECT OVERVIEW
with tab_overview:
    st.markdown("### Platform Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
            <div class="custom-metric" style="--metric-color: #06B6D4;">
                <div class="custom-metric-label">Total Transactions</div>
                <div class="custom-metric-value">{total_txn:,}</div>
            </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
            <div class="custom-metric" style="--metric-color: {COLOR_FRAUD};">
                <div class="custom-metric-label">Fraudulent Cases</div>
                <div class="custom-metric-value">{total_fraud:,}</div>
            </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
            <div class="custom-metric" style="--metric-color: {COLOR_ACCENT};">
                <div class="custom-metric-label">Overall Fraud Rate</div>
                <div class="custom-metric-value">{fraud_rate:.2f}%</div>
            </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
            <div class="custom-metric" style="--metric-color: #A855F7;">
                <div class="custom-metric-label">Unique Customers</div>
                <div class="custom-metric-value">{total_customers:,}</div>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    col_story, col_details = st.columns([3, 2])
    with col_story:
        st.markdown("""
            <div class="glass-card">
                <h4 style="margin-top:0; color:#F8FAFC; font-weight: 700; font-size: 1.1rem; letter-spacing: -0.01em;">
                    Domain Threat Model & Data Strategy
                </h4>
                <p style="color:#CBD5E1; font-size:0.92rem; line-height:1.6; margin-bottom:0.75rem;">
                    In payment ecosystems, fraud datasets are typically highly imbalanced (~98% legitimate, ~2% fraudulent) and heavily masked to protect cardholder PII.
                </p>
                <p style="color:#CBD5E1; font-size:0.92rem; line-height:1.6; margin-bottom:0.75rem;">
                    This platform integrates realistic financial behavior by embedding key domain-driven risk rules directly into the synthetic data generation pipeline (100,000 transactions and 10,000 customer profiles):
                </p>
                <ul style="color:#CBD5E1; font-size:0.92rem; line-height:1.6; padding-left:1.2rem; margin-bottom:0;">
                    <li><b>Geographic Anomaly</b>: Transactions occurring outside the cardholder's home radius or overseas.</li>
                    <li><b>Temporal Anomaly</b>: Midnight transaction spikes (11 PM - 3 AM) when cardholders are asleep and oversight is minimal.</li>
                    <li><b>Merchant Profiles</b>: High resale value targets like electronics, luxury goods, and travel.</li>
                    <li><b>Velocity Signals</b>: Rapidly successive transactions within a tight 24-hour window.</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
            <div class="glass-card" style="border-left: 3px solid #06B6D4; background: rgba(6, 182, 212, 0.03);">
                <h4 style="margin-top:0; color:#06B6D4; font-weight: 700; font-size: 1.05rem; letter-spacing: -0.01em;">
                    Developer's Perspective
                </h4>
                <p style="color:#CBD5E1; font-size:0.92rem; line-height:1.6; font-style:italic; margin-bottom:0;">
                    "Having spent three years auditing and automating transaction controls at EY and RELX Group, I built FraudShield on a key principle: risk models must be as auditable as the records they evaluate. This portal is designed to bridge the gap between machine learning predictions and clear, regulatory-compliant explanations."
                </p>
            </div>
        """, unsafe_allow_html=True)
        
    with col_details:
        st.markdown("""
            <div class="glass-card">
                <h4 style="margin-top:0; color:#F8FAFC; font-weight: 700; font-size: 1.1rem; letter-spacing: -0.01em;">
                    System Architecture
                </h4>
                <pre style="background:rgba(15,23,42,0.6); padding:1rem; border-radius:8px; border:1px solid rgba(255,255,255,0.05); color:#38BDF8; font-size:0.85rem; font-family:monospace; margin-bottom:0;">
[ generate_data.py ] 
         │
         ▼
  fraudshield.db (SQLite)
         │
  ┌──────┴──────┐
  ▼             ▼
[01/02] EDA &   [03] ML Model
SQL Console     (SMOTE Balanced)
  └──────┬──────┘
         ▼
     [ app.py ] Streamlit Portal
                </pre>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("""
            <div class="glass-card">
                <h4 style="margin-top:0; color:#F8FAFC; font-weight: 700; font-size: 1.1rem; letter-spacing: -0.01em;">
                    Core Engine Modules
                </h4>
                <ul style="color:#CBD5E1; font-size:0.88rem; line-height:1.8; list-style-type:none; padding-left:0; margin-bottom:0;">
                    <li style="margin-bottom:0.4rem;">📁 <code style="color:#38BDF8;">data_loader.py</code>: SQL Connectors & Queries</li>
                    <li style="margin-bottom:0.4rem;">📁 <code style="color:#38BDF8;">feature_engineering.py</code>: Preprocessors & Encoders</li>
                    <li style="margin-bottom:0.4rem;">📁 <code style="color:#38BDF8;">model.py</code>: Estimators & SMOTE Class Balancing</li>
                    <li style="margin-bottom:0.0rem;">📁 <code style="color:#38BDF8;">visualizations.py</code>: Diagnostic Plot Generators</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

    # Professional Developer Portfolio Card at bottom of overview
    st.markdown("""
        <div class="profile-card">
            <h3 style="margin:0 0 0.25rem 0; color:#F8FAFC; font-family:'Outfit',sans-serif; font-size:1.6rem;">Sisirath Chaloor Kuppadan</h3>
            <p style="margin:0 0 1rem 0; color:#06B6D4; font-weight:600; font-size:0.95rem;">Financial Data Professional &amp; Data Scientist</p>
            <div style="font-size:0.9rem; color:#CBD5E1; line-height:1.6; margin-bottom:1.25rem;">
                Domain-expert developer with <b>3+ years</b> of financial analytics experience at <b>Ernst &amp; Young (EY)</b> and <b>LexisNexis (RELX Group)</b>. 
                Pursuing an <b>MSc in Data Science</b> at Manipal Academy of Higher Education, focused on deploying secure, transparent, and auditable risk mitigation software.
            </div>
            <div style="margin-bottom:1.5rem;">
                <span class="badge-pill">MSc Data Science (MAHE)</span>
                <span class="badge-pill">Financial Auditing</span>
                <span class="badge-pill">Taxation Analytics</span>
                <span class="badge-pill">Python &amp; SQLite</span>
                <span class="badge-pill">Scikit-Learn</span>
                <span class="badge-pill">Explainable AI (SHAP)</span>
            </div>
            <div class="social-links">
                <a href="https://linkedin.com/in/sisirath" target="_blank" class="social-btn linkedin">
                    <svg style="width:16px;height:16px;fill:currentColor" viewBox="0 0 24 24"><path d="M19 0h-14c-2.761 0-5 2.239-5 5v14c0 2.761 2.239 5 5 5h14c2.762 0 5-2.239 5-5v-14c0-2.761-2.238-5-5-5zm-11 19h-3v-11h3v11zm-1.5-12.268c-.966 0-1.75-.79-1.75-1.764s.784-1.764 1.75-1.764 1.75.79 1.75 1.764-.783 1.764-1.75 1.764zm13.5 12.268h-3v-5.604c0-3.368-4-3.113-4 0v5.604h-3v-11h3v1.765c1.396-2.586 7-2.777 7 2.476v6.759z"/></svg>
                    LinkedIn Profile
                </a>
                <a href="https://github.com/sisirathck/fraudshield" target="_blank" class="social-btn github">
                    <svg style="width:16px;height:16px;fill:currentColor" viewBox="0 0 24 24"><path d="M12 0c-6.626 0-12 5.373-12 12 0 5.302 3.438 9.8 8.207 11.387.599.111.793-.261.793-.577v-2.234c-3.338.726-4.033-1.416-4.033-1.416-.546-1.387-1.333-1.756-1.333-1.756-1.089-.745.083-.729.083-.729 1.205.084 1.839 1.237 1.839 1.237 1.07 1.834 2.807 1.304 3.492.997.107-.775.418-1.305.762-1.604-2.665-.305-5.467-1.334-5.467-5.931 0-1.311.469-2.381 1.236-3.221-.124-.303-.535-1.524.117-3.176 0 0 1.008-.322 3.301 1.23.957-.266 1.983-.399 3.003-.404 1.02.005 2.047.138 3.006.404 2.291-1.552 3.297-1.23 3.297-1.23.653 1.653.242 2.874.118 3.176.77.84 1.235 1.911 1.235 3.221 0 4.609-2.807 5.624-5.479 5.921.43.372.823 1.102.823 2.222v3.293c0 .319.192.694.801.576 4.765-1.589 8.199-6.086 8.199-11.386 0-6.627-5.373-12-12-12z"/></svg>
                    GitHub Project
                </a>
            </div>
        </div>
    """, unsafe_allow_html=True)

# 📊 TAB 2: INTERACTIVE EDA
with tab_eda:
    st.markdown("### Exploratory Data Analysis")
    st.write("Analyze fraud patterns across merchant categories, temporal cycles, and transaction distance.")
    
    # State Selector for Regional Drilling
    states = ['All India'] + list(df['state'].unique())
    selected_state = st.selectbox("Filter Data by Customer State", states)
    
    if selected_state != 'All India':
        filtered_df = df[df['state'] == selected_state]
    else:
        filtered_df = df

    # Plot 1: Fraud Rate by Merchant Category
    st.markdown("#### 1. Fraud Rate by Merchant Category")
    cat_summary = filtered_df.groupby('merchant_category').agg(
        total_txns=('transaction_id', 'count'),
        fraud_txns=('is_fraud', 'sum')
    ).reset_index()
    cat_summary['fraud_rate_%'] = (cat_summary['fraud_txns'] / cat_summary['total_txns'] * 100).round(2)
    cat_summary = cat_summary.sort_values(by='fraud_rate_%', ascending=False)
    
    fig_cat = px.bar(
        cat_summary,
        x='merchant_category',
        y='fraud_rate_%',
        text='fraud_rate_%',
        color='fraud_rate_%',
        color_continuous_scale=[COLOR_LEGIT, COLOR_ACCENT, COLOR_FRAUD],
        labels={'fraud_rate_%': 'Fraud Rate (%)', 'merchant_category': 'Merchant Category'}
    )
    fig_cat.update_traces(textposition='outside', texttemplate='%{text:.1f}%')
    fig_cat.update_layout(xaxis={'categoryorder': 'total descending'})
    apply_plotly_theme(fig_cat, f"Fraud Rate (%) by Merchant Category in {selected_state}")
    st.plotly_chart(fig_cat, use_container_width=True)

    # Split row for two columns
    col_left, col_right = st.columns(2)
    
    with col_left:
        # Plot 2: Fraud Rate by Time of Day
        st.markdown("#### 2. Fraud Rate by Hour of Day")
        hour_summary = filtered_df.groupby('transaction_hour').agg(
            total_txns=('transaction_id', 'count'),
            fraud_txns=('is_fraud', 'sum')
        ).reset_index()
        hour_summary['fraud_rate_%'] = (hour_summary['fraud_txns'] / hour_summary['total_txns'] * 100).round(2)
        
        fig_hour = px.area(
            hour_summary,
            x='transaction_hour',
            y='fraud_rate_%',
            labels={'fraud_rate_%': 'Fraud Rate (%)', 'transaction_hour': 'Hour of Day (0-23)'}
        )
        fig_hour.update_traces(line_color=COLOR_FRAUD, fillcolor="rgba(239, 68, 68, 0.12)", mode="lines+markers")
        apply_plotly_theme(fig_hour, "Hourly Fraud Rate Trend (Peaks Late Night)")
        st.plotly_chart(fig_hour, use_container_width=True)
        
    with col_right:
        # Plot 3: Distance from Home Distribution
        st.markdown("#### 3. Fraud Risk by Transaction Distance from Home")
        fig_dist = px.histogram(
            filtered_df,
            x='distance_from_home_km',
            color='is_fraud',
            color_discrete_map={0: COLOR_LEGIT, 1: COLOR_FRAUD},
            labels={'distance_from_home_km': 'Distance from Home (km)', 'is_fraud': 'Is Fraud?'},
            barmode='overlay',
            nbins=100,
            log_y=True # Log scale because legit transactions heavily dominate low distances
        )
        # Relabel legend
        newnames = {0: 'Legit', 1: 'Fraud'}
        fig_dist.for_each_trace(lambda t: t.update(name = newnames.get(int(t.name), t.name)))
        apply_plotly_theme(fig_dist, "Transaction Distance (Log Scale)")
        st.plotly_chart(fig_dist, use_container_width=True)

# 🗄️ TAB 3: SQL ANALYTICS CONSOLE
with tab_sql:
    st.markdown("### SQL Analytics Console")
    st.write("Execute live SQLite queries on a database populated with 100,000 transaction records and 10,000 customer profiles.")
    
    # Database Schema Guide Expander
    with st.expander("View SQLite Database Schema & Relations"):
        col_schema_1, col_schema_2 = st.columns(2)
        with col_schema_1:
            st.markdown("""
                <div class="glass-card" style="padding: 1rem; margin-bottom: 0; background: rgba(15, 23, 42, 0.35);">
                    <div style="font-weight: 700; color: #F8FAFC; margin-bottom: 0.5rem;">
                        <span class="db-table-badge">transactions</span> Table
                    </div>
                    <ul style="font-family: monospace; font-size: 0.8rem; color: #94A3B8; list-style-type: none; padding-left: 0;">
                        <li>🔑 <b>transaction_id</b> (TEXT) - Primary Key</li>
                        <li>👤 <b>customer_id</b> (TEXT) - Foreign Key</li>
                        <li>🏷️ <b>merchant_category</b> (TEXT) - Retail, fuel, electronics etc.</li>
                        <li>💰 <b>amount</b> (REAL) - Transaction amount in INR</li>
                        <li>🕒 <b>transaction_hour</b> (INTEGER) - Hour of transaction (0-23)</li>
                        <li>📍 <b>distance_from_home_km</b> (REAL) - Customer home distance</li>
                        <li>🌏 <b>is_foreign</b> (INTEGER) - 1 if international, 0 if local</li>
                        <li>📅 <b>is_weekend</b> (INTEGER) - 1 if Sat/Sun, 0 otherwise</li>
                        <li>🚨 <b>is_fraud</b> (INTEGER) - Target Label (1=fraud, 0=legit)</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
            
        with col_schema_2:
            st.markdown("""
                <div class="glass-card" style="padding: 1rem; margin-bottom: 0; background: rgba(15, 23, 42, 0.35);">
                    <div style="font-weight: 700; color: #F8FAFC; margin-bottom: 0.5rem;">
                        <span class="db-table-badge">customers</span> Table
                    </div>
                    <ul style="font-family: monospace; font-size: 0.8rem; color: #94A3B8; list-style-type: none; padding-left: 0;">
                        <li>🔑 <b>customer_id</b> (TEXT) - Primary Key</li>
                        <li>👤 <b>full_name</b> (TEXT) - Customer name</li>
                        <li>🎂 <b>age</b> (INTEGER) - Age of customer</li>
                        <li>🏙️ <b>city</b> (TEXT) - Customer registered city</li>
                        <li>🏛️ <b>state</b> (TEXT) - Customer registered state</li>
                        <li>💯 <b>credit_score</b> (INTEGER) - Credit score (300-850)</li>
                        <li>💵 <b>monthly_income_inr</b> (REAL) - Monthly income</li>
                        <li>💳 <b>account_type</b> (TEXT) - Gold, Platinum, Signature</li>
                    </ul>
                </div>
            """, unsafe_allow_html=True)
            
    st.markdown("<br>", unsafe_allow_html=True)

    sql_questions = {
        "1. Executive Board Dashboard Metrics": """-- Executive board metrics using UNION ALL
SELECT 'Total Transactions Processed'  AS metric, CAST(COUNT(*) AS TEXT) AS value FROM transactions
UNION ALL
SELECT 'Fraudulent Transactions', CAST(SUM(is_fraud) AS TEXT) FROM transactions
UNION ALL
SELECT 'Overall Fraud Rate', ROUND(AVG(is_fraud) * 100, 2) || '%' FROM transactions
UNION ALL
SELECT 'Total Fraud Value (₹)', '₹' || CAST(ROUND(SUM(CASE WHEN is_fraud=1 THEN amount ELSE 0 END)) AS TEXT) FROM transactions
UNION ALL
SELECT 'Highest Risk Merchant Category', merchant_category FROM (
    SELECT merchant_category, AVG(is_fraud) AS r FROM transactions GROUP BY merchant_category ORDER BY r DESC LIMIT 1
)
UNION ALL
SELECT 'Peak Fraud Hour', CAST(transaction_hour AS TEXT) || ':00' FROM (
    SELECT transaction_hour, AVG(is_fraud) AS r FROM transactions GROUP BY transaction_hour ORDER BY r DESC LIMIT 1
)
UNION ALL
SELECT 'Foreign Transaction Fraud Rate', ROUND(AVG(CASE WHEN is_foreign=1 THEN is_fraud END) * 100, 2) || '%' FROM transactions;""",
 
        "2. Fraud Rate by Merchant Category": """-- Aggregating merchant category fraud rates
SELECT
    merchant_category,
    COUNT(*) AS total_transactions,
    SUM(is_fraud) AS fraud_cases,
    ROUND(AVG(is_fraud) * 100, 2) AS fraud_rate_pct,
    ROUND(AVG(amount), 2) AS avg_amount,
    CASE
        WHEN AVG(is_fraud) > 0.08  THEN '🔴 HIGH RISK'
        WHEN AVG(is_fraud) > 0.03  THEN '🟡 MEDIUM RISK'
        ELSE                            '🟢 LOW RISK'
    END AS risk_label
FROM transactions
GROUP BY merchant_category
ORDER BY fraud_rate_pct DESC;""",
 
        "3. Fraud Rate by Hour of Day": """-- Analysis of fraud density by hour (essential for staffing)
SELECT
    transaction_hour,
    COUNT(*) AS total_transactions,
    SUM(is_fraud) AS fraud_cases,
    ROUND(AVG(is_fraud) * 100, 2) AS fraud_rate_pct,
    CASE
        WHEN transaction_hour BETWEEN 0 AND 3   THEN '🔴 Night (12AM–3AM)'
        WHEN transaction_hour BETWEEN 4 AND 8   THEN '🟡 Early Morning'
        WHEN transaction_hour BETWEEN 9 AND 18  THEN '🟢 Business Hours'
        WHEN transaction_hour = 23              THEN '🔴 Night (11PM)'
        ELSE                                         '🟡 Evening'
    END AS risk_period
FROM transactions
GROUP BY transaction_hour
ORDER BY transaction_hour;""",
 
        "4. Fraud by Distance Bucket": """-- Bucketing transaction distances
SELECT
    CASE
        WHEN distance_from_home_km < 5      THEN '0–5km   (Home area)'
        WHEN distance_from_home_km < 25     THEN '5–25km  (Local)'
        WHEN distance_from_home_km < 100    THEN '25–100km (City/Region)'
        WHEN distance_from_home_km < 500    THEN '100–500km (State)'
        ELSE                                     '500km+  (Far / International)'
    END AS distance_bucket,
    COUNT(*) AS total_transactions,
    SUM(is_fraud) AS fraud_cases,
    ROUND(AVG(is_fraud) * 100, 2) AS fraud_rate_pct
FROM transactions
GROUP BY distance_bucket
ORDER BY fraud_rate_pct DESC;""",
 
        "5. Fraud Risk by Credit Score Band": """-- Joining transactions with customers and bucketing credit scores
WITH credit_bands AS (
    SELECT
        t.is_fraud,
        CASE
            WHEN c.credit_score < 500 THEN '1. Poor (< 500)'
            WHEN c.credit_score < 600 THEN '2. Fair (500–599)'
            WHEN c.credit_score < 700 THEN '3. Good (600–699)'
            WHEN c.credit_score < 750 THEN '4. Very Good (700–749)'
            ELSE                          '5. Excellent (750+)'
        END AS credit_band
    FROM transactions t
    INNER JOIN customers c ON t.customer_id = c.customer_id
)
SELECT
    credit_band,
    COUNT(*) AS total_transactions,
    SUM(is_fraud) AS fraud_cases,
    ROUND(AVG(is_fraud) * 100, 2) AS fraud_rate_pct
FROM credit_bands
GROUP BY credit_band
ORDER BY credit_band;""",
 
        "6. Customers with Multiple Fraud Incidents": """-- Finding high-incident customers using a subquery
SELECT
    c.full_name,
    c.city,
    c.credit_score,
    c.account_type,
    fraud_stats.fraud_count,
    ROUND(fraud_stats.total_fraud_amount, 0) AS total_fraud_amount
FROM customers c
INNER JOIN (
    SELECT
        customer_id,
        COUNT(*) AS fraud_count,
        SUM(amount) AS total_fraud_amount
    FROM transactions
    WHERE is_fraud = 1
    GROUP BY customer_id
    HAVING COUNT(*) >= 3
) AS fraud_stats ON c.customer_id = fraud_stats.customer_id
ORDER BY fraud_stats.fraud_count DESC
LIMIT 10;"""
    }
    
    selected_query_label = st.selectbox("Select SQL Business Question", list(sql_questions.keys()))
    sql_code = sql_questions[selected_query_label]
    
    st.markdown("#### SQL Workbook Editor")
    st.code(sql_code, language="sql")
    
    if st.button("Run Query"):
        with st.spinner("Executing query on database..."):
            try:
                # Execute the query on fraudshield.db
                result_df = run_query(sql_code)
                st.markdown(f"#### Query Results ({len(result_df)} rows returned)")
                st.dataframe(result_df, use_container_width=True)
                
                # Dynamic plotting for selected queries to make SQL visual
                if "Category" in selected_query_label:
                    fig = px.bar(result_df, x='merchant_category', y='fraud_rate_pct', text='fraud_rate_pct',
                                 labels={'merchant_category': 'Category', 'fraud_rate_pct': 'Fraud Rate (%)'},
                                 color_discrete_sequence=[COLOR_FRAUD])
                    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                    apply_plotly_theme(fig, "SQL Output: Category Fraud Rates")
                    st.plotly_chart(fig, use_container_width=True)
                elif "Hour" in selected_query_label:
                    fig = px.line(result_df, x='transaction_hour', y='fraud_rate_pct', markers=True,
                                  labels={'transaction_hour': 'Hour of Day (0-23)', 'fraud_rate_pct': 'Fraud Rate (%)'},
                                  color_discrete_sequence=[COLOR_FRAUD])
                    apply_plotly_theme(fig, "SQL Output: Hourly Fraud Rates")
                    st.plotly_chart(fig, use_container_width=True)
                elif "Distance" in selected_query_label:
                    fig = px.bar(result_df, x='distance_bucket', y='fraud_rate_pct', text='fraud_rate_pct',
                                 labels={'distance_bucket': 'Distance Bucket', 'fraud_rate_pct': 'Fraud Rate (%)'},
                                 color_discrete_sequence=[COLOR_ACCENT])
                    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                    apply_plotly_theme(fig, "SQL Output: Distance Fraud Rates")
                    st.plotly_chart(fig, use_container_width=True)
                elif "Credit Score" in selected_query_label:
                    fig = px.bar(result_df, x='credit_band', y='fraud_rate_pct', text='fraud_rate_pct',
                                 labels={'credit_band': 'Credit Score Band', 'fraud_rate_pct': 'Fraud Rate (%)'},
                                 color_discrete_sequence=[COLOR_LEGIT])
                    fig.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                    apply_plotly_theme(fig, "SQL Output: Credit Score Risk Bands")
                    st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Error running query: {e}")

# 🤖 TAB 4: MODEL DIAGNOSTICS
with tab_diagnostics:
    st.markdown("### Model Diagnostics & Validation")
    st.write("Validation metrics and statistical comparisons between the custom NumPy scratch model and the production Scikit-learn model.")
    
    col_explain_left, col_explain_right = st.columns(2)
    with col_explain_left:
        st.markdown("""
            <div class="glass-card" style="height: 100%;">
                <h4 style="margin-top:0; color:#F59E0B; font-weight: 700; font-size: 1.1rem; letter-spacing: -0.01em;">
                    Mathematical Formulation & Sigmoid Scaling
                </h4>
                <p style="color:#CBD5E1; font-size:0.9rem; line-height:1.5; margin-bottom:0.75rem;">
                    Logistic Regression evaluates a linear combination of scaled transaction features (<i>z</i>) and maps the output to a probability space [0, 1] using the logistic sigmoid function:
                </p>
                <div class="math-card" style="text-align:center; padding: 0.75rem; margin: 0.75rem 0;">
                    <span style="color:#06B6D4; font-family:serif; font-size:1.15rem;">P(Fraud) = &sigma;(z) = 1 / (1 + e<sup>-z</sup>)</span>
                </div>
                <p style="color:#CBD5E1; font-size:0.9rem; line-height:1.5; margin-bottom:0.75rem;">
                    Here, <i>z</i> represents the raw decision boundary (log-odds unit), calculated as the dot product of feature weights and transaction vectors.
                </p>
                <p style="color:#CBD5E1; font-size:0.9rem; line-height:1.5; margin-bottom:0;">
                    <b>Class Imbalance Mitigation</b>: Because fraud transactions represent less than 2% of the dataset, training directly on raw historical data yields a high-accuracy, zero-recall model. We apply SMOTE (Synthetic Minority Over-sampling Technique) to align class representation during model training.
                </p>
            </div>
        """, unsafe_allow_html=True)
        
    with col_explain_right:
        st.markdown("""
            <div class="glass-card" style="height: 100%;">
                <h4 style="margin-top:0; color:#EF4444; font-weight: 700; font-size: 1.1rem; letter-spacing: -0.01em;">
                    Regulatory Compliance & Auditing (XAI)
                </h4>
                <p style="color:#CBD5E1; font-size:0.9rem; line-height:1.5; margin-bottom:0.75rem;">
                    Under frameworks like the European Union AI Act and Reserve Bank of India (RBI) risk mitigation guidelines, automated transaction declines must support an audit trail of clear, mathematical evidence.
                </p>
                <p style="color:#CBD5E1; font-size:0.9rem; line-height:1.5; margin-bottom:0.75rem;">
                    While black-box models (e.g., Deep Networks, XGBoost) offer complex decision spaces, they struggle to provide exact local feature attributions at runtime.
                </p>
                <p style="color:#CBD5E1; font-size:0.9rem; line-height:1.5; margin-bottom:0;">
                    By pairing a linear model with SHAP (SHapley Additive exPlanations), we calculate exactly how each transaction signal (e.g., high velocity, extreme distance) pushes the probability away from the baseline, satisfying regulatory transparency rules.
                </p>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Evaluator comparison
    st.markdown("#### Model Performance Metrics")
    
    # Sklearn model metrics
    y_prob_sklearn = model.predict_proba(X_test)[:, 1]
    y_pred_sklearn = (y_prob_sklearn >= decision_threshold).astype(int)
    
    acc_sk = accuracy_score(y_test, y_pred_sklearn)
    prec_sk = precision_score(y_test, y_pred_sklearn)
    rec_sk = recall_score(y_test, y_pred_sklearn)
    f1_sk = f1_score(y_test, y_pred_sklearn)
    auc_sk = roc_auc_score(y_test, y_prob_sklearn)
    
    # Hardcoded Custom NumPy metrics from our static runs to avoid re-training scratch model on fly (saving time)
    acc_sc = 0.8245
    prec_sc = 0.0880
    rec_sc = 0.7780
    f1_sc = 0.1581
    auc_sc = 0.8653
    
    # Render customized comparison table in HTML
    st.markdown(f"""
        <div class="glass-card" style="padding: 0; overflow-x: auto; border: 1px solid rgba(255, 255, 255, 0.08);">
            <table style="width: 100%; border-collapse: collapse; text-align: left; font-size: 0.9rem;">
                <thead>
                    <tr style="background: rgba(15, 23, 42, 0.6); border-bottom: 2px solid rgba(255, 255, 255, 0.1);">
                        <th style="padding: 1rem 1.25rem; font-weight: 700; color: #F8FAFC; font-family: 'Outfit', sans-serif;">Metric</th>
                        <th style="padding: 1rem 1.25rem; font-weight: 700; color: #06B6D4; font-family: 'Outfit', sans-serif;">NumPy Scratch Model</th>
                        <th style="padding: 1rem 1.25rem; font-weight: 700; color: #3B82F6; font-family: 'Outfit', sans-serif;">Scikit-Learn (Selected)</th>
                        <th style="padding: 1rem 1.25rem; font-weight: 700; color: #94A3B8; font-family: 'Outfit', sans-serif;">Business Metric Goal</th>
                    </tr>
                </thead>
                <tbody>
                    <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.05); background: rgba(30, 41, 59, 0.1);">
                        <td style="padding: 0.85rem 1.25rem; font-weight: 600; color: #E2E8F0;">Accuracy (Overall)</td>
                        <td style="padding: 0.85rem 1.25rem; color: #CBD5E1; font-family: monospace;">82.45%</td>
                        <td style="padding: 0.85rem 1.25rem; color: #CBD5E1; font-family: monospace;">{acc_sk*100:.2f}%</td>
                        <td style="padding: 0.85rem 1.25rem; color: #94A3B8; font-size: 0.8rem;">Not primary focus due to class imbalance</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.05); background: rgba(30, 41, 59, 0.25);">
                        <td style="padding: 0.85rem 1.25rem; font-weight: 600; color: #E2E8F0;">Precision (Correct flags)</td>
                        <td style="padding: 0.85rem 1.25rem; color: #CBD5E1; font-family: monospace;">8.80%</td>
                        <td style="padding: 0.85rem 1.25rem; color: #CBD5E1; font-family: monospace;">{prec_sk*100:.2f}%</td>
                        <td style="padding: 0.85rem 1.25rem; color: #94A3B8; font-size: 0.8rem;">High = fewer angry legitimate customers blocked</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.05); background: rgba(30, 41, 59, 0.1);">
                        <td style="padding: 0.85rem 1.25rem; font-weight: 600; color: #E2E8F0;">Recall (Caught Fraud %)</td>
                        <td style="padding: 0.85rem 1.25rem; color: #10B981; font-weight: bold; font-family: monospace;">77.80%</td>
                        <td style="padding: 0.85rem 1.25rem; color: #10B981; font-weight: bold; font-family: monospace;">{rec_sk*100:.2f}%</td>
                        <td style="padding: 0.85rem 1.25rem; color: #10B981; font-size: 0.8rem; font-weight: 500;">🟢 MUST BE HIGH (Catch the thieves!)</td>
                    </tr>
                    <tr style="border-bottom: 1px solid rgba(255, 255, 255, 0.05); background: rgba(30, 41, 59, 0.25);">
                        <td style="padding: 0.85rem 1.25rem; font-weight: 600; color: #E2E8F0;">F1 Score (Balanced)</td>
                        <td style="padding: 0.85rem 1.25rem; color: #CBD5E1; font-family: monospace;">15.81%</td>
                        <td style="padding: 0.85rem 1.25rem; color: #CBD5E1; font-family: monospace;">{f1_sk*100:.2f}%</td>
                        <td style="padding: 0.85rem 1.25rem; color: #94A3B8; font-size: 0.8rem;">Balanced performance metric</td>
                    </tr>
                    <tr style="background: rgba(30, 41, 59, 0.1);">
                        <td style="padding: 0.85rem 1.25rem; font-weight: 600; color: #E2E8F0;">ROC-AUC Score</td>
                        <td style="padding: 0.85rem 1.25rem; color: #06B6D4; font-weight: bold; font-family: monospace;">0.8653</td>
                        <td style="padding: 0.85rem 1.25rem; color: #06B6D4; font-weight: bold; font-family: monospace;">{auc_sk:.4f}</td>
                        <td style="padding: 0.85rem 1.25rem; color: #94A3B8; font-size: 0.8rem;">Closer to 1.0 means perfect class separation</td>
                    </tr>
                </tbody>
            </table>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)

    # Show pre-generated charts for validation
    st.markdown("#### Model Visualizations")
    col_chart_left, col_chart_right = st.columns(2)
    
    fig_dir = os.path.join(os.path.dirname(__file__), 'outputs', 'figures', 'ml')
    
    with col_chart_left:
        st.markdown("**ROC & Precision-Recall Curves**")
        roc_path = os.path.join(fig_dir, '01_roc_curve.png')
        if os.path.exists(roc_path):
            st.image(roc_path, caption="ROC-AUC Curve (Measures general ability to separate classes)", use_column_width=True)
        else:
            st.warning("ROC Curve chart not found. Run analysis/03_ml_modeling.py first.")
            
    with col_chart_right:
        st.markdown("**Confusion Matrix & SHAP Summary**")
        cm_path = os.path.join(fig_dir, '03_confusion_matrix.png')
        if os.path.exists(cm_path):
            st.image(cm_path, caption="Confusion Matrix (Shows actual numbers of hits vs. misses)", use_column_width=True)
        else:
            st.warning("Confusion Matrix chart not found. Run analysis/03_ml_modeling.py first.")
 
    st.markdown("#### Global Model Interpretability (SHAP Summary)")
    shap_path = os.path.join(fig_dir, '04_shap_summary.png')
    if os.path.exists(shap_path):
        st.image(shap_path, caption="SHAP Global Feature Importance (Which features drive overall model decisions)", width=800)
    else:
        st.warning("SHAP Summary plot not found. Run analysis/03_ml_modeling.py first.")

# TAB 5: REAL-TIME TRANSACTION SIMULATOR
with tab_simulator:

    st.markdown("""
        <div style="margin-bottom: 0.25rem;">
            <h3 style="margin:0; color:#F8FAFC; font-family:'Outfit',sans-serif; font-weight: 700; font-size: 1.4rem; letter-spacing: -0.01em;">
                Real-Time Risk Simulator
            </h3>
            <p style="color:#94A3B8; font-size:0.9rem; margin-top:0.25rem; margin-bottom:0;">
                Simulate a transaction payload, score it through the pipeline, and view mathematically exact feature attribution values.
            </p>
        </div>
        <hr style="border-color: rgba(255,255,255,0.06); margin: 1rem 0 1.5rem 0;">
    """, unsafe_allow_html=True)

    # ── Step 1: Customer Profile ─────────────────────────────────
    st.markdown("""
        <div style="font-size:0.72rem; text-transform:uppercase; letter-spacing:0.12em; font-weight:700; color:#64748B; margin-bottom:0.4rem;">Phase 1</div>
        <div style="font-size:1.1rem; font-weight:700; color:#F1F5F9; margin-bottom:0.75rem;">Select Customer Profile</div>
    """, unsafe_allow_html=True)

    # Query 10 sample customers to populate dropdown
    sample_customers = run_query("""
        SELECT customer_id, full_name, age, city, credit_score, account_type, monthly_income_inr
        FROM customers
        ORDER BY customer_id
        LIMIT 50
    """)

    cust_options = []
    for _, row in sample_customers.iterrows():
        cust_options.append(f"{row['customer_id']} — {row['full_name']} · {row['city']} · Credit {row['credit_score']}")

    selected_cust_str = st.selectbox("Customer", cust_options, label_visibility="collapsed")
    selected_cust_id = selected_cust_str.split(" — ")[0]
    cust_row = sample_customers[sample_customers['customer_id'] == selected_cust_id].iloc[0]

    # Styled customer context card
    # Compute credit score tier
    cs = int(cust_row['credit_score'])
    if cs >= 750:
        tier_label, tier_color = "Excellent", "#10B981"
    elif cs >= 700:
        tier_label, tier_color = "Very Good", "#3B82F6"
    elif cs >= 600:
        tier_label, tier_color = "Good", "#F59E0B"
    else:
        tier_label, tier_color = "Fair", "#EF4444"

    st.markdown(f"""
        <div class="glass-card" style="padding:1.25rem 1.5rem; margin-bottom:1.75rem; border-left: 3px solid {tier_color};">
            <div style="display:flex; align-items:center; gap:1rem; flex-wrap:wrap;">
                <div style="width:48px; height:48px; border-radius:50%; background:linear-gradient(135deg,#1E293B,#334155); display:flex; align-items:center; justify-content:center; font-size:1.5rem; border:2px solid rgba(255,255,255,0.08);">👤</div>
                <div style="flex:1;">
                    <div style="font-weight:700; font-size:1rem; color:#F8FAFC;">{cust_row['full_name']}</div>
                    <div style="font-size:0.78rem; color:#94A3B8;">📍 {cust_row['city']} &nbsp;·&nbsp; 🎂 Age {cust_row['age']} &nbsp;·&nbsp; 💳 {cust_row['account_type']} Account</div>
                </div>
                <div style="text-align:right;">
                    <div style="font-size:1.4rem; font-weight:800; color:{tier_color}; font-family:'Outfit',sans-serif;">{cs}</div>
                    <div style="font-size:0.7rem; color:{tier_color}; font-weight:600; text-transform:uppercase; letter-spacing:0.08em;">{tier_label} Credit</div>
                </div>
                <div style="text-align:right; padding-left:1.5rem; border-left:1px solid rgba(255,255,255,0.07);">
                    <div style="font-size:1.1rem; font-weight:700; color:#F8FAFC;">₹{int(cust_row['monthly_income_inr']):,}</div>
                    <div style="font-size:0.7rem; color:#94A3B8;">Monthly Income</div>
                </div>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # ── Step 2: Transaction Configuration ───────────────────────
    st.markdown("""
        <div style="font-size:0.72rem; text-transform:uppercase; letter-spacing:0.12em; font-weight:700; color:#64748B; margin-bottom:0.4rem;">Phase 2</div>
        <div style="font-size:1.1rem; font-weight:700; color:#F1F5F9; margin-bottom:0.75rem;">Configure Transaction Parameters</div>
    """, unsafe_allow_html=True)

    col_sim_1, col_sim_2, col_sim_3 = st.columns(3)

    with col_sim_1:
        st.markdown('<div style="font-size:0.8rem; font-weight:600; color:#94A3B8; text-transform:uppercase; letter-spacing:0.07em; margin-bottom:0.5rem;">Merchant Profile</div>', unsafe_allow_html=True)
        category = st.selectbox(
            "Merchant Category",
            ['grocery', 'restaurant', 'fuel', 'healthcare', 'education', 'entertainment', 'online_retail', 'electronics', 'travel', 'luxury_goods'],
            index=0
        )
        typical_amounts = {
            "grocery": 800, "restaurant": 1200, "fuel": 2500, "healthcare": 4000,
            "education": 8000, "entertainment": 2000, "online_retail": 3500,
            "electronics": 18000, "travel": 15000, "luxury_goods": 30000
        }
        typical_amt = typical_amounts.get(category, 1000)
        amount = st.number_input(
            f"Transaction Amount (₹) — Typical: ₹{typical_amt:,}",
            min_value=10.0, max_value=500000.0, value=float(typical_amt), step=100.0
        )
        txn_type = st.selectbox(
            "Payment Channel",
            ['POS', 'online', 'ATM', 'bank_transfer', 'contactless'],
            index=1
        )

    with col_sim_2:
        st.markdown('<div style="font-size:0.8rem; font-weight:600; color:#94A3B8; text-transform:uppercase; letter-spacing:0.07em; margin-bottom:0.5rem;">Risk Signals</div>', unsafe_allow_html=True)
        is_foreign = st.checkbox("Foreign / Overseas Merchant", value=False)
        distance = st.slider(
            "Distance from Home (km)",
            min_value=0.0, max_value=15000.0, value=15.0, step=1.0,
            help="Stolen cards are often used far from the cardholder's home address."
        )
        prev_24h = st.slider(
            "Transactions in Last 24 Hours",
            min_value=0, max_value=20, value=2,
            help="High velocity (>5 txns/24h) is a strong indicator of card compromise."
        )

    with col_sim_3:
        st.markdown('<div style="font-size:0.8rem; font-weight:600; color:#94A3B8; text-transform:uppercase; letter-spacing:0.07em; margin-bottom:0.5rem;">Temporal Context</div>', unsafe_allow_html=True)
        hour = st.slider(
            "Transaction Hour (0–23)",
            min_value=0, max_value=23, value=12,
            help="Fraud peaks between 11PM and 3AM when bank oversight is minimal."
        )
        is_weekend_bool = st.checkbox("Weekend Transaction", value=False)
        is_weekend = 1 if is_weekend_bool else 0

        # Show a live risk signal helper
        is_late_night_preview = hour in [23, 0, 1, 2, 3]
        hour_label = "Night Cycle (Elevated Risk)" if is_late_night_preview else "Day Cycle (Standard Risk)"
        st.markdown(f"""
            <div style="margin-top:0.75rem; padding:0.6rem 0.85rem; border-radius:6px; background:rgba(15,23,42,0.4); border:1px solid rgba(255,255,255,0.05); font-size:0.8rem; color:#CBD5E1;">
                Hour Parameter: <b style="color:#F8FAFC;">{hour}:00</b> &mdash; <span style="color:{'#EF4444' if is_late_night_preview else '#10B981'}; font-weight:600;">{hour_label}</span>
            </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Score button — more prominent
    col_btn_l, col_btn_m, col_btn_r = st.columns([1, 2, 1])
    with col_btn_m:
        score_clicked = st.button("⚡ Run Fraud Analysis", use_container_width=True)

    # ── Step 3: Results ──────────────────────────────────────────
    if score_clicked:

        # ── PROCESS INPUT FEATURES ─────────────────────────────────
        is_late_night = 1 if hour in [23, 0, 1, 2, 3] else 0
        velocity_amount_interaction = prev_24h * amount

        raw_inputs = {
            'amount': amount,
            'is_foreign': 1 if is_foreign else 0,
            'distance_from_home_km': distance,
            'num_prev_transactions_24h': prev_24h,
            'is_weekend': is_weekend,
            'is_late_night': is_late_night,
            'velocity_amount_interaction': velocity_amount_interaction
        }
        for cat in ['electronics', 'entertainment', 'fuel', 'grocery', 'healthcare', 'luxury_goods', 'online_retail', 'restaurant', 'travel']:
            raw_inputs[f'category_{cat}'] = 1 if category == cat else 0
        for t in ['POS', 'bank_transfer', 'contactless', 'online']:
            raw_inputs[f'type_{t}'] = 1 if txn_type == t else 0

        X_new_raw = pd.DataFrame([raw_inputs])
        X_new_scaled = X_new_raw.copy()
        continuous_cols = ['amount', 'distance_from_home_km', 'num_prev_transactions_24h', 'velocity_amount_interaction']
        X_new_scaled[continuous_cols] = scaler.transform(X_new_raw[continuous_cols])
        X_new_scaled = X_new_scaled[features]

        # ── MODEL PREDICTION ────────────────────────────────────────
        prob_fraud = model.predict_proba(X_new_scaled.values)[0][1]
        is_flagged = prob_fraud >= decision_threshold

        # ── VERDICT BANNER ──────────────────────────────────────────
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("""
            <div style="font-size:0.72rem; text-transform:uppercase; letter-spacing:0.12em; font-weight:700; color:#64748B; margin-bottom:0.4rem;">Phase 3</div>
            <div style="font-size:1.1rem; font-weight:700; color:#F1F5F9; margin-bottom:0.75rem;">Transaction Audit & Decision Summary</div>
        """, unsafe_allow_html=True)

        risk_pct = prob_fraud * 100

        if is_flagged:
            st.markdown(f"""
                <div class="status-declined" style="text-align: left; padding: 1.25rem 1.5rem; border-left: 4px solid #EF4444; background: rgba(239, 68, 68, 0.05); border-radius: 6px;">
                    <div style="font-size: 0.75rem; font-weight: 700; text-transform: uppercase; color: #EF4444; letter-spacing: 0.1em; margin-bottom: 0.25rem;">
                        System Verdict: Transaction Declined
                    </div>
                    <div style="font-size: 1.5rem; font-weight: 800; color: #F8FAFC; font-family: 'Outfit', sans-serif;">
                        Risk Score: {risk_pct:.1f}%
                    </div>
                    <div style="font-size: 0.85rem; color: #94A3B8; margin-top: 0.5rem; line-height: 1.5;">
                        The computed risk exceeds the decision threshold of <b>{decision_threshold:.0%}</b>. 
                        The engine has generated a declination payload and routed the transaction to risk queue for audit.
                    </div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="status-approved" style="text-align: left; padding: 1.25rem 1.5rem; border-left: 4px solid #10B981; background: rgba(16, 185, 129, 0.05); border-radius: 6px;">
                    <div style="font-size: 0.75rem; font-weight: 700; text-transform: uppercase; color: #10B981; letter-spacing: 0.1em; margin-bottom: 0.25rem;">
                        System Verdict: Transaction Approved
                    </div>
                    <div style="font-size: 1.5rem; font-weight: 800; color: #F8FAFC; font-family: 'Outfit', sans-serif;">
                        Risk Score: {risk_pct:.1f}%
                    </div>
                    <div style="font-size: 0.85rem; color: #94A3B8; margin-top: 0.5rem; line-height: 1.5;">
                        The computed risk remains within acceptable limits below the threshold of <b>{decision_threshold:.0%}</b>. 
                        The payment clearance pipeline executed successfully.
                    </div>
                </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # ── GAUGE + SHAP SIDE BY SIDE ───────────────────────────────
        col_res_left, col_res_right = st.columns([2, 3])

        with col_res_left:
            # Styled gauge
            gauge_color = COLOR_FRAUD if is_flagged else COLOR_LEGIT
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=risk_pct,
                delta={'reference': decision_threshold * 100, 'increasing': {'color': COLOR_FRAUD}, 'decreasing': {'color': COLOR_LEGIT}},
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Risk Probability", 'font': {'size': 14, 'family': 'Outfit, sans-serif', 'color': '#CBD5E1'}},
                number={'suffix': '%', 'font': {'size': 40, 'family': 'Outfit, sans-serif', 'color': gauge_color}},
                gauge={
                    'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': '#334155', 'tickfont': {'color': '#94A3B8'}},
                    'bar': {'color': gauge_color, 'thickness': 0.25},
                    'bgcolor': 'rgba(15,23,42,0.4)',
                    'borderwidth': 0,
                    'steps': [
                        {'range': [0, 30], 'color': 'rgba(16, 185, 129, 0.12)'},
                        {'range': [30, 70], 'color': 'rgba(245, 158, 11, 0.10)'},
                        {'range': [70, 100], 'color': 'rgba(239, 68, 68, 0.12)'}
                    ],
                    'threshold': {
                        'line': {'color': '#F59E0B', 'width': 2},
                        'thickness': 0.8,
                        'value': decision_threshold * 100
                    }
                }
            ))
            fig_gauge.update_layout(
                height=280,
                margin=dict(l=20, r=20, t=50, b=10),
                paper_bgcolor='rgba(0,0,0,0)',
                font=dict(family='Inter, sans-serif', color='#CBD5E1')
            )
            st.plotly_chart(fig_gauge, use_container_width=True)

            # Quick signal summary pills
            signals = []
            if is_flagged: signals.append(("", "Declined", "#EF4444"))
            else: signals.append(("", "Approved", "#10B981"))
            if is_late_night: signals.append(("", "Late-Night Cycle", "#F59E0B"))
            if is_foreign: signals.append(("", "International Merchant", "#F59E0B"))
            if distance > 500: signals.append(("", f"Out-of-Radius ({distance:.0f}km)", "#EF4444"))
            if prev_24h > 5: signals.append(("", f"Velocity Alert ({prev_24h} txns/24h)", "#EF4444"))

            pills_html = "".join([
                f'<span style="display:inline-flex; align-items:center; gap:0.3rem; padding:0.3rem 0.7rem; border-radius:4px; font-size:0.75rem; font-weight:600; background:rgba(30,41,59,0.7); border:1px solid {c}30; color:{c}; margin:0.2rem;">{label}</span>'
                for ico, label, c in signals
            ])
            st.markdown(f"""
                <div style="margin-top:0.5rem;">{pills_html}</div>
            """, unsafe_allow_html=True)

        with col_res_right:
            # ── CALCULATE LOCAL SHAP VALUES ──────────────────────────
            # SHAP_i = coefficient_i × (scaled_value_i − mean_value_i)
            coefs = model.coef_[0]

            shap_values = []
            for idx, col_name in enumerate(features):
                x_val = X_new_scaled.iloc[0][col_name]
                mu_val = feature_means[col_name]
                w = coefs[idx]
                shap_values.append(w * (x_val - mu_val))

            explain_df = pd.DataFrame({'Feature': features, 'SHAP_Value': shap_values})

            readable_mapping = {
                'amount': 'Transaction Amount',
                'is_foreign': 'Foreign Transaction Flag',
                'distance_from_home_km': 'Distance from Home (km)',
                'num_prev_transactions_24h': 'Velocity (Txns/24h)',
                'is_weekend': 'Weekend Transaction',
                'is_late_night': 'Late-Night Hour (11PM–3AM)',
                'velocity_amount_interaction': 'Velocity × Amount Interaction',
                'category_electronics': 'Category: Electronics',
                'category_entertainment': 'Category: Entertainment',
                'category_fuel': 'Category: Fuel',
                'category_grocery': 'Category: Grocery Store',
                'category_healthcare': 'Category: Healthcare',
                'category_luxury_goods': 'Category: Luxury Goods',
                'category_online_retail': 'Category: Online Retail',
                'category_restaurant': 'Category: Dining/Restaurants',
                'category_travel': 'Category: Travel/Flights',
                'type_POS': 'Channel: POS Terminal',
                'type_bank_transfer': 'Channel: Bank Transfer',
                'type_contactless': 'Channel: Tap-to-Pay',
                'type_online': 'Channel: Online/CNP'
            }
            explain_df['Display_Feature'] = explain_df['Feature'].map(readable_mapping)
            explain_df['Abs_SHAP'] = explain_df['SHAP_Value'].abs()
            explain_df = explain_df.sort_values('Abs_SHAP', ascending=False)
            explain_df = explain_df[explain_df['Abs_SHAP'] > 0.01].head(8)
            explain_df = explain_df.iloc[::-1]  # flip for horizontal bar
            explain_df['Color'] = explain_df['SHAP_Value'].apply(
                lambda x: '#EF4444' if x > 0 else '#10B981'
            )

            # SHAP bar chart
            fig_shap = go.Figure()
            fig_shap.add_trace(go.Bar(
                y=explain_df['Display_Feature'],
                x=explain_df['SHAP_Value'],
                orientation='h',
                marker=dict(
                    color=explain_df['Color'],
                    line=dict(width=0)
                ),
                text=explain_df['SHAP_Value'].apply(lambda x: f"+{x:.3f}" if x > 0 else f"{x:.3f}"),
                textposition='outside',
                textfont=dict(size=11, color='#CBD5E1'),
                hovertemplate="<b>%{y}</b><br>Risk Contribution: %{x:.4f}<extra></extra>"
            ))
            fig_shap.add_vline(x=0, line_color='rgba(255,255,255,0.15)', line_width=1)
            fig_shap.update_layout(
                title={
                    'text': '<b>Risk Factor Contributions (SHAP Values)</b>',
                    'font': {'size': 13, 'family': 'Outfit, sans-serif', 'color': '#F8FAFC'}
                },
                xaxis=dict(
                    title=dict(
                        text='Impact on Fraud Score (Log-Odds Units)',
                        font=dict(size=11, color='#94A3B8')
                    ),
                    gridcolor='rgba(255,255,255,0.05)',
                    zerolinecolor='rgba(255,255,255,0.1)',
                    tickfont=dict(color='#94A3B8')
                ),
                yaxis=dict(
                    tickfont=dict(size=11, color='#CBD5E1')
                ),
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(30,41,59,0.15)',
                margin=dict(l=10, r=50, t=45, b=10),
                height=300,
                showlegend=False,
                font=dict(family='Inter, sans-serif')
            )
            st.plotly_chart(fig_shap, use_container_width=True)

        # ── PLAIN ENGLISH EXPLANATION ───────────────────────────────
        top_risk = explain_df[explain_df['SHAP_Value'] > 0].sort_values('SHAP_Value', ascending=False)
        top_safety = explain_df[explain_df['SHAP_Value'] < 0].sort_values('SHAP_Value', ascending=True)

        risk_phrases = [f"<b>{r['Display_Feature']}</b> <code style='color:#F87171;'>+{r['SHAP_Value']:.3f}</code>" for _, r in top_risk.iterrows()]
        safe_phrases = [f"<b>{s['Display_Feature']}</b> <code style='color:#34D399;'>{s['SHAP_Value']:.3f}</code>" for _, s in top_safety.iterrows()]

        if risk_phrases or safe_phrases:
            risk_html = ""
            if risk_phrases:
                risk_html += f"""
                <div style="margin-bottom:0.75rem;">
                    <div style="font-size:0.75rem; text-transform:uppercase; letter-spacing:0.15em; font-weight:700; color:#EF4444; margin-bottom:0.4rem;">Risk Drivers (Positive SHAP)</div>
                    <div style="font-size:0.85rem; color:#CBD5E1; line-height:1.8;">{' &nbsp;·&nbsp; '.join(risk_phrases)}</div>
                </div>
                """
            if safe_phrases:
                risk_html += f"""
                <div>
                    <div style="font-size:0.75rem; text-transform:uppercase; letter-spacing:0.15em; font-weight:700; color:#10B981; margin-bottom:0.4rem;">Safety Factors (Negative SHAP)</div>
                    <div style="font-size:0.85rem; color:#CBD5E1; line-height:1.8;">{' &nbsp;·&nbsp; '.join(safe_phrases)}</div>
                </div>
                """

            st.markdown(f"""
                <div class="glass-card" style="border-top: 3px solid {'#EF4444' if is_flagged else '#10B981'}; padding:1.25rem 1.5rem;">
                    <div style="font-weight:700; color:#F8FAFC; margin-bottom:0.85rem; display:flex; align-items:center; gap:0.5rem; font-size: 0.95rem; font-family:'Outfit', sans-serif;">
                        Feature Attribution Summary
                        <span style="font-size:0.7rem; padding:0.2rem 0.6rem; border-radius:4px; background:rgba(6,182,212,0.1); color:#06B6D4; border:1px solid rgba(6,182,212,0.2); font-weight:600; text-transform: uppercase; letter-spacing: 0.05em;">Audit Trail Enabled</span>
                    </div>
                    {risk_html}
                </div>
            """, unsafe_allow_html=True)
