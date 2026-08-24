-- Product segment is emitted only when available in the decision export.
SELECT
    COALESCE(productcd, 'UNKNOWN') AS product_segment,
    COUNT(*) AS transaction_count,
    AVG(actual_label * 1.0) AS fraud_rate,
    AVG(CASE WHEN decision = 'REVIEW' THEN 1.0 ELSE 0.0 END) AS review_rate,
    AVG(CASE WHEN decision = 'DECLINE' THEN 1.0 ELSE 0.0 END) AS decline_rate,
    SUM(CASE WHEN actual_label = 1 AND decision IN ('REVIEW', 'DECLINE') THEN 1 ELSE 0 END) * 1.0
        / NULLIF(SUM(CASE WHEN actual_label = 1 THEN 1 ELSE 0 END), 0) AS fraud_triage_coverage
FROM transaction_decisions
GROUP BY COALESCE(productcd, 'UNKNOWN')
ORDER BY transaction_count DESC;
