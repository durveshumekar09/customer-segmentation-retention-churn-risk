-- Validation checks for the transactions table and derived customer logic

-- 1. Total transaction rows
SELECT COUNT(*) AS total_transactions
FROM transactions;

-- 2. Total unique customers
SELECT COUNT(DISTINCT customer_id) AS total_customers
FROM transactions;

-- 3. Date range check
SELECT
    MIN(DATE(transaction_date)) AS min_transaction_date,
    MAX(DATE(transaction_date)) AS max_transaction_date
FROM transactions;

-- 4. Null check for key columns
SELECT
    SUM(CASE WHEN transaction_id IS NULL THEN 1 ELSE 0 END) AS null_transaction_id,
    SUM(CASE WHEN customer_id IS NULL THEN 1 ELSE 0 END) AS null_customer_id,
    SUM(CASE WHEN transaction_date IS NULL THEN 1 ELSE 0 END) AS null_transaction_date,
    SUM(CASE WHEN order_value IS NULL THEN 1 ELSE 0 END) AS null_order_value,
    SUM(CASE WHEN quantity IS NULL THEN 1 ELSE 0 END) AS null_quantity
FROM transactions;

-- 5. Duplicate transaction_id check
SELECT
    transaction_id,
    COUNT(*) AS duplicate_count
FROM transactions
GROUP BY transaction_id
HAVING COUNT(*) > 1;

-- 6. Negative or zero order value check
SELECT COUNT(*) AS non_positive_order_value_rows
FROM transactions
WHERE order_value <= 0;

-- 7. Negative or zero quantity check
SELECT COUNT(*) AS non_positive_quantity_rows
FROM transactions
WHERE quantity <= 0;

-- 8. Customers with only one transaction
SELECT COUNT(*) AS one_time_customers
FROM (
    SELECT customer_id
    FROM transactions
    GROUP BY customer_id
    HAVING COUNT(transaction_id) = 1
) t;

-- 9. Customers with more than one transaction
SELECT COUNT(*) AS repeat_customers
FROM (
    SELECT customer_id
    FROM transactions
    GROUP BY customer_id
    HAVING COUNT(transaction_id) > 1
) t;

-- 10. Channel distribution
SELECT
    channel,
    COUNT(*) AS transactions
FROM transactions
GROUP BY channel
ORDER BY transactions DESC;

-- 11. Product category distribution
SELECT
    product_category,
    COUNT(*) AS transactions
FROM transactions
GROUP BY product_category
ORDER BY transactions DESC;

-- 12. Customer state distribution
SELECT
    customer_state,
    COUNT(DISTINCT customer_id) AS customers
FROM transactions
GROUP BY customer_state
ORDER BY customers DESC;