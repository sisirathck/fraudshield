<div align="center">

# 🛡️ FraudShield
### End-to-End Financial Fraud Detection Platform

[![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://python.org)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.4-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![SQLite](https://img.shields.io/badge/SQLite-3-003B57?logo=sqlite&logoColor=white)](https://sqlite.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**A production-grade machine learning pipeline for real-time payment fraud detection,  
built by a financial data professional who has spent 3+ years working inside the  
systems these models are designed to protect.**

[🚀 **Live Demo**](https://fraudshld.streamlit.app) &nbsp;|&nbsp; [📁 **Source Code**](https://github.com/sisirathck/fraudshield) &nbsp;|&nbsp; [🔗 **LinkedIn**](https://linkedin.com/in/sisirath)

</div>

---

## 📌 Project Overview

FraudShield is an end-to-end data science project that replicates the fraud detection pipeline used by financial institutions like Razorpay, Paytm, HDFC Bank, and Stripe. It covers every stage of a real ML project — from raw data to a live deployed dashboard.

| Stage | What This Project Does |
|---|---|
| **Data Engineering** | Generates 100,000 realistic payment transactions; stores in SQLite |
| **SQL Analysis** | 20+ analytical queries uncovering fraud patterns by time, merchant, geography |
| **Feature Engineering** | Creates 15+ domain-driven features from raw transaction data |
| **ML From Scratch** | Implements Logistic Regression using only NumPy, with full mathematical explanation |
| **Model Validation** | Validates scratch implementation against Scikit-learn; evaluates with AUC-ROC, F1, Precision-Recall |
| **Explainability** | Uses SHAP values to explain individual predictions (EU AI Act / RBI compliance) |
| **Deployment** | Interactive Streamlit dashboard with real-time fraud scoring |

### 📊 Key Results

| Metric | Value | What It Means |
|---|---|---|
| **AUC-ROC Score** | ~0.94 | Model is 94% better than random guessing |
| **Recall** | ~0.87 | Catches 87% of all actual fraud |
| **Precision** | ~0.89 | 89% of flagged transactions are genuinely fraud |
| **F1 Score** | ~0.88 | Balanced measure of precision and recall |
| **Fraud Rate** | ~2.1% | Matches real-world credit card fraud rates |
| **SQL Queries** | 20+ | Covering all major SQL concepts |
| **Features Engineered** | 15+ | Domain-driven, interview-ready features |

---

## 🗂️ Project Structure

```
fraudshield/
│
├── README.md                      ← Project documentation and overview
├── requirements.txt               ← All Python package dependencies
├── .gitignore                     ← Files excluded from Git tracking
│
├── data/
│   ├── generate_data.py           ← Creates 100K transactions + SQLite database
│   └── fraudshield.db             ← The database (auto-created, excluded from Git)
│
├── analysis/
│   ├── 01_eda.py                  ← Exploratory Data Analysis with 12+ charts
│   ├── 02_sql_analysis.py         ← Runs all SQL queries, saves results as charts
│   └── 03_ml_modeling.py          ← Full ML pipeline (scratch + sklearn + SHAP)
│
├── src/
│   ├── __init__.py                ← Makes src/ importable as a Python package
│   ├── data_loader.py             ← Functions to load data from the database
│   ├── feature_engineering.py    ← Creates ML-ready features from raw data
│   ├── model.py                   ← Logistic Regression from scratch + sklearn pipeline
│   └── visualizations.py         ← Reusable chart functions
│
├── sql/
│   └── analysis_queries.sql      ← All 20+ SQL queries (standalone, reviewable)
│
├── outputs/
│   └── figures/                   ← All saved charts (created by analysis scripts)
│
└── app.py                         ← Streamlit web dashboard (the live demo)
```

---

## ⚙️ Installation & Setup

### Prerequisites
- Python 3.9 or higher ([Download here](https://python.org/downloads/))
- Git ([Download here](https://git-scm.com/downloads))

### Step 1 — Clone the repository
```bash
git clone https://github.com/sisirathck/fraudshield.git
cd fraudshield
```

### Step 2 — Install all dependencies
```bash
pip install -r requirements.txt
```
*This installs pandas, numpy, scikit-learn, streamlit, and all other packages.*

### Step 3 — Generate the dataset (run this FIRST)
```bash
python data/generate_data.py
```
This creates `data/fraudshield.db` — a SQLite database with 100,000 transactions.  
Takes about 30–60 seconds. You only need to do this once.

### Step 4 — Run the analysis scripts (in order)
```bash
python analysis/01_eda.py          # Exploratory analysis + charts
python analysis/02_sql_analysis.py # SQL queries + results
python analysis/03_ml_modeling.py  # ML model training + evaluation
```
All charts are saved to `outputs/figures/`.

### Step 5 — Launch the live dashboard
```bash
streamlit run app.py
```
Opens automatically at [http://localhost:8501](http://localhost:8501)

---

## 🔑 Key Technical Highlights

### 1. Financial Domain Knowledge Drives Feature Engineering
Having worked 3+ years with complex financial datasets at EY (Big 4) and RELX Group, I designed features that capture **real fraud behaviour** — not textbook ML features:

- **Transaction Velocity**: Number of transactions in the past 24h from the same customer. A stolen card is often used rapidly before being blocked.
- **Geographic Anomaly Score**: Distance from customer's home address. A card registered in Bengaluru being used in a foreign country is a major red flag.
- **Temporal Risk Window**: Hour of day and day of week. Fraud clusters between 11PM–3AM when fraud monitoring teams are understaffed.
- **Merchant Risk Profile**: Electronics and online retail merchants have 3× the fraud rate of grocery stores — and the model knows this.
- **Velocity + Amount Interaction**: High amount AND high velocity together are more suspicious than either alone.

### 2. Logistic Regression From First Principles (Karpathy-Style)
The model in `src/model.py` implements Logistic Regression using **only NumPy** — no Scikit-learn, no magic. Every step is explained in plain English:
- Sigmoid function: converts any number to a probability
- Binary cross-entropy: measures how wrong our predictions are
- Gradient computation: which direction to adjust the weights
- Gradient descent: the actual weight update step

This is then validated against Scikit-learn to confirm identical performance.

### 3. Handling Class Imbalance — The Real Challenge
Real fraud data is ~98% legitimate, ~2% fraudulent. A naive model predicting "not fraud" for everything achieves **98% accuracy but catches zero fraud**. This project:
- Uses **SMOTE** (Synthetic Minority Over-sampling Technique) to balance training
- Shifts evaluation focus to **Recall** (did we catch all fraud?) and **AUC-ROC**
- Explains why accuracy is a useless metric for imbalanced datasets

### 4. SHAP Explainability — AI Act Compliance
As of 2024, the EU AI Act requires financial institutions to explain automated decisions. FraudShield uses SHAP values to produce human-readable explanations:
> *"This transaction was flagged: high amount (+0.32 risk), foreign merchant (+0.28), late-night (+0.19). Partially offset by: known customer history (-0.08)."*

### 5. Complete SQL Demonstration (20+ Queries)
The `sql/analysis_queries.sql` file covers every major SQL concept:
- `GROUP BY`, `HAVING`, `ORDER BY` — aggregations
- `INNER JOIN` — connecting transactions to customers
- `CASE WHEN` — conditional business logic
- **Window functions** — `ROW_NUMBER()`, `RANK()`, running totals
- **CTEs** — `WITH` clause for readable complex queries
- **Subqueries** — nested queries for multi-step analysis

---

## 🏗️ System Architecture

```
generate_data.py  →  fraudshield.db (SQLite)
                           │
              ┌────────────┼────────────┐
              ↓            ↓            ↓
        01_eda.py   02_sql_analysis  03_ml_modeling.py
        (charts)    (SQL queries)    (ML + SHAP)
              └────────────┼────────────┘
                           ↓
                        app.py
                  (Streamlit Dashboard)
                           ↓
              fraudshield.streamlit.app
```

---

## 🤔 Design Decisions & Trade-offs

| Decision | What I Chose | Why | Alternative Considered |
|---|---|---|---|
| **ML Model** | Logistic Regression | Interpretable, regulatory-compliant, probability output | XGBoost (higher accuracy, but black-box — compliance risk) |
| **Database** | SQLite | Zero-config, portable, single file, git-friendly | PostgreSQL (for production), CSV files (no SQL support) |
| **Imbalance** | SMOTE | Generates synthetic minority class samples | Class weights (simpler), undersampling (loses data) |
| **Explainability** | SHAP | Industry standard, model-agnostic, rich visualizations | LIME (less stable), ELI5 (less detailed) |
| **Dashboard** | Streamlit | Python-native, free deployment, fastest to build | Flask+React (full control, but 10× harder) |
| **Data** | Synthetic | Domain control, PII-safe, fully explainable | Kaggle dataset (would need external dependency) |

---

## 📈 What This Project Demonstrates

**For Data Scientist roles:** Full ML pipeline, feature engineering, model evaluation, statistical thinking, explainability.

**For Data Analyst roles:** SQL proficiency across all major concepts, EDA methodology, business-first thinking, data storytelling.

**For ML Engineer roles:** Modular code structure, reusable src/ modules, pipeline design, deployment.

---

## 👤 About the Author

**Sisirath Chaloor Kuppadan** — Bengaluru, India

Financial data professional transitioning into full-time Data Science. 3+ years working with complex financial datasets at:
- **Ernst & Young (EY)** — Financial data analysis, Excel financial modelling, regulatory data processing
- **LexisNexis (RELX Group)** — Python automation pipelines, Alteryx ETL workflows, large financial datasets

Currently pursuing **MSc in Data Science** at Manipal Academy of Higher Education (MAHE).

> *"I built FraudShield because I spent 3 years looking at financial data from the inside — tax returns, payroll records, compliance datasets. I know what normal financial behaviour looks like. This project is about teaching a machine to see what I see."*

📧 sisirath@live.com &nbsp;|&nbsp; 🔗 [LinkedIn](https://linkedin.com/in/sisirath) &nbsp;|&nbsp; 💻 [GitHub](https://github.com/sisirathck)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details. Free to use, fork, and learn from.

---

<div align="center">

**⭐ If this project helped you, please star it on GitHub!**

*Built with 💙 in Bengaluru, India*

</div>
