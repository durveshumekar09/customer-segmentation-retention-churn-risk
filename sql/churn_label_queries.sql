-- Customer-level churn label and supporting features
-- Churn rule: no purchase in the last 90 days from dataset end date

WITH dataset_end AS (
    SELECT MAX(DATE(transaction_date)) AS max_transaction_date
    FROM transactions
),

customer_base AS (
    SELECT
        customer_id,
        MIN(DATE(transaction_date)) AS first_purchase_date,
        MAX(DATE(transaction_date)) AS last_purchase_date,
        COUNT(transaction_id) AS total_transactions,
        ROUND(SUM(order_value), 2) AS total_spend,
        ROUND(AVG(order_value), 2) AS avg_order_value,
        SUM(quantity) AS quantity_total,
        ROUND(AVG(discount_used_pct), 4) AS discount_usage_rate,
        COUNT(DISTINCT strftime('%Y-%m', transaction_date)) AS active_months
    FROM transactions
    GROUP BY customer_id
),

recent_activity AS (
    SELECT
        t.customer_id,
        COUNT(t.transaction_id) AS transactions_last_90_days,
        ROUND(SUM(t.order_value), 2) AS spend_last_90_days
    FROM transactions t
    CROSS JOIN dataset_end d
    WHERE DATE(t.transaction_date) >= DATE(d.max_transaction_date, '-90 day')
    GROUP BY t.customer_id
)

SELECT
    c.customer_id,
    c.first_purchase_date,
    c.last_purchase_date,
    c.total_transactions,
    c.total_spend,
    c.avg_order_value,
    c.quantity_total,
    c.discount_usage_rate,
    c.active_months,
    CAST(julianday(c.last_purchase_date) - julianday(c.first_purchase_date) AS INTEGER) AS customer_lifespan_days,
    CAST(julianday(d.max_transaction_date) - julianday(c.last_purchase_date) AS INTEGER) AS days_since_last_purchase,
    CASE
        WHEN CAST(julianday(d.max_transaction_date) - julianday(c.last_purchase_date) AS INTEGER) > 90 THEN 1
        ELSE 0
    END AS churn_label,
    ROUND(
        c.total_transactions * c.avg_order_value * (1.0 * c.active_months / 12),
        2
    ) AS clv_proxy,
    COALESCE(r.transactions_last_90_days, 0) AS transactions_last_90_days,
    COALESCE(r.spend_last_90_days, 0) AS spend_last_90_days
FROM customer_base c
CROSS JOIN dataset_end d
LEFT JOIN recent_activity r
    ON c.customer_id = r.customer_id
ORDER BY c.customer_id;