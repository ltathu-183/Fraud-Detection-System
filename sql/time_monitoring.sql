-- TransactionDT is seconds from an undisclosed origin, so this is relative-week monitoring.
SELECT
    CAST(timestamp_seconds / 604800 AS INTEGER) AS relative_week,
    COUNT(*) AS transaction_count,
    SUM(transaction_amount) AS transaction_amount,
    AVG(actual_label * 1.0) AS fraud_rate,
    AVG(fraud_score) AS average_score,
    AVG(CASE WHEN decision = 'REVIEW' THEN 1.0 ELSE 0.0 END) AS review_rate,
    AVG(CASE WHEN decision = 'DECLINE' THEN 1.0 ELSE 0.0 END) AS decline_rate
FROM transaction_decisions
GROUP BY CAST(timestamp_seconds / 604800 AS INTEGER)
ORDER BY relative_week;
