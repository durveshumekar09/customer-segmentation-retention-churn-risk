-- Cohort analysis query
-- This calculates cohort month, order month, cohort index, and retention counts

WITH base AS (
    SELECT
        customer_id,
        transaction_id,
        DATE(transaction_date) AS transaction_date,
        strftime('%Y-%m', transaction_date) AS order_month
    FROM transactions
),

first_purchase AS (
    SELECT
        customer_id,
        MIN(DATE(transaction_date)) AS first_purchase_date,
        strftime('%Y-%m', MIN(transaction_date)) AS cohort_month
    FROM transactions
    GROUP BY customer_id
),

cohort_base AS (
    SELECT
        b.customer_id,
        b.transaction_id,
        b.transaction_date,
        b.order_month,
        f.cohort_month,
        (
            (CAST(substr(b.order_month, 1, 4) AS INTEGER) - CAST(substr(f.cohort_month, 1, 4) AS INTEGER)) * 12
            + (CAST(substr(b.order_month, 6, 2) AS INTEGER) - CAST(substr(f.cohort_month, 6, 2) AS INTEGER))
            + 1
        ) AS cohort_index
    FROM base b
    LEFT JOIN first_purchase f
        ON b.customer_id = f.customer_id
),

cohort_counts AS (
    SELECT
        cohort_month,
        cohort_index,
        COUNT(DISTINCT customer_id) AS customers
    FROM cohort_base
    GROUP BY cohort_month, cohort_index
),

cohort_size AS (
    SELECT
        cohort_month,
        customers AS cohort_size
    FROM cohort_counts
    WHERE cohort_index = 1
)

SELECT
    c.cohort_month,
    c.cohort_index,
    c.customers,
    s.cohort_size,
    ROUND(1.0 * c.customers / s.cohort_size, 4) AS retention_rate
FROM cohort_counts c
LEFT JOIN cohort_size s
    ON c.cohort_month = s.cohort_month
ORDER BY c.cohort_month, c.cohort_index;