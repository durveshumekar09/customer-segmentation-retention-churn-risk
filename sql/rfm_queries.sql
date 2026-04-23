-- Calculate customer-level RFM metrics from the transactions table

SELECT
    customer_id,
    CAST(julianday('2025-12-31') - julianday(MAX(transaction_date)) AS INTEGER) AS recency,
    COUNT(transaction_id) AS frequency,
    ROUND(SUM(order_value), 2) AS monetary
FROM transactions
GROUP BY customer_id
ORDER BY monetary DESC;


