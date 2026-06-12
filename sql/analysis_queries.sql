-- ================================================================
-- analysis_queries.sql — FraudShield SQL Analysis
-- ================================================================
--
-- PURPOSE:
--   20+ SQL queries that answer real business questions about
--   payment fraud. These are the same query TYPES written by
--   analysts at Razorpay, Paytm, HDFC, and Stripe.
--
-- HOW TO RUN:
--   Option 1: python analysis/02_sql_analysis.py  (runs them all)
--   Option 2: DB Browser for SQLite (free GUI app — open fraudshield.db)
--
-- SQL CONCEPTS COVERED:
--   ✓ SELECT, WHERE, GROUP BY, ORDER BY, LIMIT, HAVING
--   ✓ Aggregate functions: COUNT, SUM, AVG, MIN, MAX
--   ✓ ROUND, CAST, CASE WHEN
--   ✓ String functions: strftime (date parsing)
--   ✓ INNER JOIN (linking transactions to customers)
--   ✓ Subqueries (query inside a query)
--   ✓ CTEs — WITH clause (named temporary queries)
--   ✓ Window Functions: ROW_NUMBER, RANK, SUM OVER
--
-- DATABASE SCHEMA (what the tables look like):
-- -----------------------------------------------
-- customers:    customer_id | full_name | age | gender | city | state |
--               account_type | credit_score | account_opened_date | monthly_income_inr
--
-- transactions: transaction_id | customer_id | timestamp | transaction_hour |
--               day_of_week | is_weekend | amount | merchant_name | merchant_category |
--               transaction_type | is_foreign | distance_from_home_km |
--               num_prev_transactions_24h | fraud_probability | is_fraud
-- ================================================================


-- ================================================================
-- SECTION 1: THE BIG PICTURE
-- Business Question: "What is the overall state of fraud in our system?"
-- ================================================================

-- QUERY 1: Master Summary — The First Query Any Analyst Runs
-- This gives you a one-line summary of the entire dataset.
-- COUNT(*) = count all rows, SUM(is_fraud) = count only fraud rows (1=fraud, 0=legit)
SELECT
    COUNT(*)                                                           AS total_transactions,
    SUM(is_fraud)                                                      AS total_fraud_cases,
    ROUND(AVG(is_fraud) * 100, 2)                                     AS fraud_rate_percent,
    ROUND(SUM(amount), 2)                                             AS total_value_inr,
    ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END), 2)     AS total_fraud_value_inr,
    ROUND(AVG(amount), 2)                                             AS avg_transaction_amount,
    ROUND(MAX(amount), 2)                                             AS largest_transaction,
    ROUND(MIN(amount), 2)                                             AS smallest_transaction
FROM transactions;


-- QUERY 2: Fraud vs Legitimate — Side-by-Side Comparison
-- CASE WHEN is SQL's version of an if-else statement:
--   WHEN condition THEN value_if_true ELSE value_if_false END
SELECT
    CASE WHEN is_fraud = 1 THEN 'Fraudulent' ELSE 'Legitimate' END   AS transaction_status,
    COUNT(*)                                                          AS count,
    ROUND(COUNT(*) * 100.0 / SUM(COUNT(*)) OVER (), 2)              AS pct_of_total,
    ROUND(AVG(amount), 2)                                            AS avg_amount,
    ROUND(MAX(amount), 2)                                            AS max_amount,
    ROUND(AVG(distance_from_home_km), 1)                            AS avg_distance_km,
    ROUND(AVG(is_foreign) * 100, 1)                                 AS pct_foreign,
    ROUND(AVG(transaction_hour), 1)                                  AS avg_hour
FROM transactions
GROUP BY is_fraud;


-- QUERY 3: Monthly Transaction Trend
-- strftime('%Y-%m', timestamp) extracts "2023-01" from "2023-01-15 14:22:00"
-- This tracks how fraud evolved month by month
SELECT
    strftime('%Y', timestamp)                                        AS year,
    strftime('%m', timestamp)                                        AS month,
    COUNT(*)                                                         AS total_transactions,
    SUM(is_fraud)                                                    AS fraud_count,
    ROUND(AVG(is_fraud) * 100, 2)                                   AS fraud_rate_pct,
    ROUND(SUM(amount), 0)                                           AS total_value_inr,
    ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END), 0)   AS fraud_value_inr
FROM transactions
GROUP BY year, month
ORDER BY year, month;


-- ================================================================
-- SECTION 2: MERCHANT ANALYSIS
-- Business Question: "Which merchants / categories carry the most fraud risk?"
-- ================================================================

-- QUERY 4: Fraud Rate by Merchant Category
-- ORDER BY fraud_rate_pct DESC → highest risk categories appear first
SELECT
    merchant_category,
    COUNT(*)                                                          AS total_transactions,
    SUM(is_fraud)                                                     AS fraud_cases,
    ROUND(AVG(is_fraud) * 100, 2)                                    AS fraud_rate_pct,
    ROUND(AVG(amount), 2)                                            AS avg_amount,
    ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END), 0)    AS total_fraud_value,
    -- Label each category's risk level using CASE WHEN
    CASE
        WHEN AVG(is_fraud) > 0.08  THEN '🔴 HIGH RISK'
        WHEN AVG(is_fraud) > 0.03  THEN '🟡 MEDIUM RISK'
        ELSE                            '🟢 LOW RISK'
    END AS risk_label
FROM transactions
GROUP BY merchant_category
ORDER BY fraud_rate_pct DESC;


-- QUERY 5: Top 15 Individual Merchants by Fraud Volume
-- HAVING filters AFTER grouping (like WHERE, but for aggregated results)
-- We require at least 30 transactions to get a reliable fraud rate estimate
SELECT
    merchant_name,
    merchant_category,
    COUNT(*)                         AS total_transactions,
    SUM(is_fraud)                    AS fraud_cases,
    ROUND(AVG(is_fraud) * 100, 2)   AS fraud_rate_pct,
    ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END), 0) AS total_fraud_value
FROM transactions
GROUP BY merchant_name, merchant_category
HAVING total_transactions >= 30         -- Only merchants with enough data to be statistically reliable
ORDER BY fraud_rate_pct DESC
LIMIT 15;


-- ================================================================
-- SECTION 3: TIME ANALYSIS
-- Business Question: "WHEN does fraud happen? When should we scale up monitoring?"
-- ================================================================

-- QUERY 6: Fraud Rate by Hour of Day
-- This directly informs fraud team staffing decisions.
-- A spike at 2AM means you need more staff working nights.
SELECT
    transaction_hour,
    COUNT(*)                         AS total_transactions,
    SUM(is_fraud)                    AS fraud_cases,
    ROUND(AVG(is_fraud) * 100, 2)   AS fraud_rate_pct,
    -- Classify each hour into a risk period
    CASE
        WHEN transaction_hour BETWEEN 0 AND 3   THEN '🔴 Night (12AM–3AM)'
        WHEN transaction_hour BETWEEN 4 AND 8   THEN '🟡 Early Morning (4AM–8AM)'
        WHEN transaction_hour BETWEEN 9 AND 18  THEN '🟢 Business Hours (9AM–6PM)'
        WHEN transaction_hour = 23              THEN '🔴 Night (11PM)'
        ELSE                                         '🟡 Evening (7PM–10PM)'
    END AS risk_period
FROM transactions
GROUP BY transaction_hour
ORDER BY transaction_hour;


-- QUERY 7: Fraud Rate by Day of Week
-- 0=Monday, 6=Sunday. Do weekends have more fraud?
SELECT
    day_of_week,
    CASE day_of_week
        WHEN 0 THEN 'Monday'    WHEN 1 THEN 'Tuesday'
        WHEN 2 THEN 'Wednesday' WHEN 3 THEN 'Thursday'
        WHEN 4 THEN 'Friday'    WHEN 5 THEN 'Saturday'
        WHEN 6 THEN 'Sunday'
    END AS day_name,
    is_weekend,
    COUNT(*)                         AS total_transactions,
    SUM(is_fraud)                    AS fraud_cases,
    ROUND(AVG(is_fraud) * 100, 2)   AS fraud_rate_pct
FROM transactions
GROUP BY day_of_week
ORDER BY day_of_week;


-- QUERY 8: Weekend vs Weekday Summary
SELECT
    CASE WHEN is_weekend = 1 THEN 'Weekend' ELSE 'Weekday' END AS period,
    COUNT(*)                         AS total_transactions,
    SUM(is_fraud)                    AS fraud_cases,
    ROUND(AVG(is_fraud) * 100, 2)   AS fraud_rate_pct,
    ROUND(AVG(amount), 2)           AS avg_amount
FROM transactions
GROUP BY is_weekend
ORDER BY fraud_rate_pct DESC;


-- ================================================================
-- SECTION 4: GEOGRAPHIC ANALYSIS
-- Business Question: "Does location matter for fraud risk?"
-- ================================================================

-- QUERY 9: Foreign vs Domestic Transaction Fraud Rate
SELECT
    CASE WHEN is_foreign = 1 THEN 'Foreign Transaction' ELSE 'Domestic Transaction' END  AS location_type,
    COUNT(*)                         AS total_transactions,
    SUM(is_fraud)                    AS fraud_cases,
    ROUND(AVG(is_fraud) * 100, 2)   AS fraud_rate_pct,
    ROUND(AVG(amount), 2)           AS avg_amount,
    ROUND(AVG(distance_from_home_km), 1) AS avg_distance_from_home_km
FROM transactions
GROUP BY is_foreign
ORDER BY fraud_rate_pct DESC;


-- QUERY 10: Fraud Rate by Distance from Home (Bucketed)
-- "Bucketing" = grouping a continuous number into ranges
-- This is a very common analyst technique
SELECT
    CASE
        WHEN distance_from_home_km < 5      THEN '0–5km   (Home area)'
        WHEN distance_from_home_km < 25     THEN '5–25km  (Local)'
        WHEN distance_from_home_km < 100    THEN '25–100km (City/Region)'
        WHEN distance_from_home_km < 500    THEN '100–500km (State/Neighbour)'
        ELSE                                     '500km+  (Far / International)'
    END AS distance_bucket,
    COUNT(*)                         AS total_transactions,
    SUM(is_fraud)                    AS fraud_cases,
    ROUND(AVG(is_fraud) * 100, 2)   AS fraud_rate_pct,
    ROUND(AVG(amount), 2)           AS avg_amount
FROM transactions
GROUP BY distance_bucket
ORDER BY fraud_rate_pct DESC;


-- ================================================================
-- SECTION 5: TRANSACTION TYPE & AMOUNT ANALYSIS
-- ================================================================

-- QUERY 11: Fraud Rate by Transaction Type
SELECT
    transaction_type,
    COUNT(*)                         AS total_transactions,
    SUM(is_fraud)                    AS fraud_cases,
    ROUND(AVG(is_fraud) * 100, 2)   AS fraud_rate_pct,
    ROUND(AVG(amount), 2)           AS avg_amount
FROM transactions
GROUP BY transaction_type
ORDER BY fraud_rate_pct DESC;


-- QUERY 12: Fraud Rate by Amount Range (Bucketed)
-- This helps set thresholds: "flag all transactions above ₹10,000 for manual review"
SELECT
    CASE
        WHEN amount < 500     THEN '₹0–₹500'
        WHEN amount < 2000    THEN '₹500–₹2,000'
        WHEN amount < 5000    THEN '₹2,000–₹5,000'
        WHEN amount < 10000   THEN '₹5,000–₹10,000'
        WHEN amount < 25000   THEN '₹10,000–₹25,000'
        ELSE                       '₹25,000+'
    END AS amount_bucket,
    COUNT(*)                         AS total_transactions,
    SUM(is_fraud)                    AS fraud_cases,
    ROUND(AVG(is_fraud) * 100, 2)   AS fraud_rate_pct,
    ROUND(AVG(amount), 0)           AS avg_amount_in_bucket
FROM transactions
GROUP BY amount_bucket
ORDER BY fraud_rate_pct DESC;


-- ================================================================
-- SECTION 6: JOINS — Linking Transactions to Customer Profiles
-- Business Question: "Which types of customers are most targeted?"
-- ================================================================

-- QUERY 13: Fraud by Customer Age Group (INNER JOIN)
-- INNER JOIN: For each transaction, look up the customer in the customers table
--             and bring in their details (age, gender, credit score, etc.)
-- t = alias for transactions, c = alias for customers
SELECT
    CASE
        WHEN c.age < 25  THEN 'Under 25'
        WHEN c.age < 35  THEN '25–34'
        WHEN c.age < 45  THEN '35–44'
        WHEN c.age < 55  THEN '45–54'
        ELSE                  '55+'
    END AS age_group,
    c.gender,
    COUNT(t.transaction_id)          AS total_transactions,
    SUM(t.is_fraud)                  AS fraud_cases,
    ROUND(AVG(t.is_fraud) * 100, 2) AS fraud_rate_pct,
    ROUND(AVG(t.amount), 2)         AS avg_amount
FROM transactions t
INNER JOIN customers c
    ON t.customer_id = c.customer_id   -- The "bridge" between the two tables
GROUP BY age_group, c.gender
ORDER BY fraud_rate_pct DESC
LIMIT 20;


-- QUERY 14: Fraud by City (JOIN)
-- Which cities have the highest fraud rates?
SELECT
    c.city,
    c.state,
    COUNT(DISTINCT t.customer_id)     AS unique_customers,
    COUNT(t.transaction_id)           AS total_transactions,
    SUM(t.is_fraud)                   AS fraud_cases,
    ROUND(AVG(t.is_fraud) * 100, 2)  AS fraud_rate_pct,
    ROUND(SUM(CASE WHEN t.is_fraud = 1 THEN t.amount ELSE 0 END), 0) AS total_fraud_value
FROM transactions t
INNER JOIN customers c ON t.customer_id = c.customer_id
GROUP BY c.city, c.state
ORDER BY fraud_rate_pct DESC;


-- QUERY 15: Fraud by Account Type (JOIN)
SELECT
    c.account_type,
    COUNT(t.transaction_id)           AS total_transactions,
    SUM(t.is_fraud)                   AS fraud_cases,
    ROUND(AVG(t.is_fraud) * 100, 2)  AS fraud_rate_pct,
    ROUND(AVG(c.credit_score), 0)    AS avg_credit_score,
    ROUND(AVG(t.amount), 2)          AS avg_transaction_amount
FROM transactions t
INNER JOIN customers c ON t.customer_id = c.customer_id
GROUP BY c.account_type
ORDER BY fraud_rate_pct DESC;


-- ================================================================
-- SECTION 7: SUBQUERIES
-- A query INSIDE another query — like a formula inside a formula in Excel
-- ================================================================

-- QUERY 16: Customers with Multiple Fraud Incidents
-- STEP 1 (inner query): Find customers who had 3+ fraud incidents
-- STEP 2 (outer query): Join with customers table to get their details
SELECT
    c.full_name,
    c.city,
    c.credit_score,
    c.account_type,
    fraud_stats.fraud_count,
    ROUND(fraud_stats.total_fraud_amount, 0) AS total_fraud_amount
FROM customers c
INNER JOIN (
    -- This inner query runs FIRST and produces a temporary result
    SELECT
        customer_id,
        COUNT(*)        AS fraud_count,
        SUM(amount)    AS total_fraud_amount
    FROM transactions
    WHERE is_fraud = 1          -- Only look at fraudulent transactions
    GROUP BY customer_id
    HAVING COUNT(*) >= 3        -- Only customers with 3 or more fraud cases
) AS fraud_stats
    ON c.customer_id = fraud_stats.customer_id
ORDER BY fraud_stats.fraud_count DESC
LIMIT 20;


-- ================================================================
-- SECTION 8: CTEs (Common Table Expressions) — WITH clause
-- Like creating a named temporary table inside your query
-- Much more readable than nested subqueries
-- ================================================================

-- QUERY 17: Monthly Fraud Trend with Cumulative Running Total
-- CTE 'monthly': calculates per-month statistics
-- Main query: adds a running total using a window function
WITH monthly AS (
    SELECT
        strftime('%Y-%m', timestamp)                                AS month,
        COUNT(*)                                                    AS total_txn,
        SUM(is_fraud)                                              AS fraud_count,
        ROUND(AVG(is_fraud) * 100, 2)                             AS fraud_rate_pct,
        ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END), 0) AS fraud_value
    FROM transactions
    GROUP BY strftime('%Y-%m', timestamp)
)
SELECT
    month,
    total_txn,
    fraud_count,
    fraud_rate_pct,
    fraud_value,
    -- WINDOW FUNCTION: SUM() OVER (ORDER BY month) = running cumulative total
    -- For each row, sum all fraud_count values in rows up to and including this one
    SUM(fraud_count)  OVER (ORDER BY month) AS cumulative_fraud_count,
    SUM(fraud_value)  OVER (ORDER BY month) AS cumulative_fraud_value
FROM monthly
ORDER BY month;


-- QUERY 18: Category Risk Ranking with RANK Window Function
-- RANK() assigns a rank number: 1 = highest fraud rate, 2 = second highest, etc.
WITH category_stats AS (
    SELECT
        merchant_category,
        COUNT(*)                         AS total_transactions,
        SUM(is_fraud)                    AS fraud_count,
        ROUND(AVG(is_fraud) * 100, 3)   AS fraud_rate_pct,
        ROUND(SUM(CASE WHEN is_fraud = 1 THEN amount ELSE 0 END), 0) AS fraud_value
    FROM transactions
    GROUP BY merchant_category
)
SELECT
    merchant_category,
    total_transactions,
    fraud_count,
    fraud_rate_pct,
    fraud_value,
    -- RANK() is a window function: no OVER partition means rank globally
    RANK() OVER (ORDER BY fraud_rate_pct DESC) AS fraud_risk_rank
FROM category_stats
ORDER BY fraud_risk_rank;


-- ================================================================
-- SECTION 9: VELOCITY ANALYSIS
-- Business Question: "Do customers who transact more frequently have more fraud?"
-- ================================================================

-- QUERY 19: Fraud Rate by Transaction Velocity (Previous 24h)
SELECT
    CASE
        WHEN num_prev_transactions_24h = 0   THEN '0 — First transaction today'
        WHEN num_prev_transactions_24h <= 2  THEN '1–2 — Normal frequency'
        WHEN num_prev_transactions_24h <= 5  THEN '3–5 — Active user'
        WHEN num_prev_transactions_24h <= 9  THEN '6–9 — Very active'
        ELSE                                      '10+ — Unusually high velocity'
    END AS velocity_bucket,
    COUNT(*)                         AS total_transactions,
    SUM(is_fraud)                    AS fraud_cases,
    ROUND(AVG(is_fraud) * 100, 2)   AS fraud_rate_pct
FROM transactions
GROUP BY velocity_bucket
ORDER BY fraud_rate_pct DESC;


-- ================================================================
-- SECTION 10: CREDIT SCORE ANALYSIS (JOIN)
-- Business Question: "Do customers with lower credit scores face more fraud?"
-- ================================================================

-- QUERY 20: Fraud by Credit Score Band
WITH credit_bands AS (
    SELECT
        t.transaction_id,
        t.amount,
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
    COUNT(*)                         AS total_transactions,
    SUM(is_fraud)                    AS fraud_cases,
    ROUND(AVG(is_fraud) * 100, 2)   AS fraud_rate_pct,
    ROUND(AVG(amount), 2)           AS avg_transaction_amount
FROM credit_bands
GROUP BY credit_band
ORDER BY credit_band;


-- QUERY 21: Executive Dashboard — Board-Level Summary
-- UNION ALL stacks multiple single-row queries into one result table
-- This is how you build an "executive summary" in SQL
SELECT 'Total Transactions Processed'  AS metric,
       CAST(COUNT(*) AS TEXT)           AS value
FROM transactions

UNION ALL

SELECT 'Fraudulent Transactions',
       CAST(SUM(is_fraud) AS TEXT)
FROM transactions

UNION ALL

SELECT 'Overall Fraud Rate',
       ROUND(AVG(is_fraud) * 100, 2) || '%'
FROM transactions

UNION ALL

SELECT 'Total Fraud Value (₹)',
       '₹' || CAST(ROUND(SUM(CASE WHEN is_fraud=1 THEN amount ELSE 0 END)) AS TEXT)
FROM transactions

UNION ALL

SELECT 'Highest Risk Merchant Category',
       merchant_category
FROM (
    SELECT merchant_category, AVG(is_fraud) AS r
    FROM transactions GROUP BY merchant_category ORDER BY r DESC LIMIT 1
)

UNION ALL

SELECT 'Peak Fraud Hour',
       CAST(transaction_hour AS TEXT) || ':00'
FROM (
    SELECT transaction_hour, AVG(is_fraud) AS r
    FROM transactions GROUP BY transaction_hour ORDER BY r DESC LIMIT 1
)

UNION ALL

SELECT 'Foreign Transaction Fraud Rate',
       ROUND(AVG(CASE WHEN is_foreign=1 THEN is_fraud END) * 100, 2) || '%'
FROM transactions;
