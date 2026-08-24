-- All denominators are explicit. REVIEW means routing, not confirmed detection.
SELECT
    COUNT(*) AS total_transaction_volume,
    AVG(actual_label * 1.0) AS fraud_rate,
    AVG(CASE WHEN decision = 'APPROVE' THEN 1.0 ELSE 0.0 END) AS approve_rate,
    AVG(CASE WHEN decision = 'REVIEW' THEN 1.0 ELSE 0.0 END) AS review_rate,
    AVG(CASE WHEN decision = 'DECLINE' THEN 1.0 ELSE 0.0 END) AS decline_rate,
    SUM(CASE WHEN actual_label = 1 AND decision IN ('REVIEW', 'DECLINE') THEN 1 ELSE 0 END) * 1.0
        / NULLIF(SUM(CASE WHEN actual_label = 1 THEN 1 ELSE 0 END), 0) AS fraud_triage_coverage,
    SUM(CASE WHEN actual_label = 0 AND decision = 'DECLINE' THEN 1 ELSE 0 END) * 1.0
        / NULLIF(SUM(CASE WHEN actual_label = 0 THEN 1 ELSE 0 END), 0) AS legitimate_false_decline_rate,
    SUM(CASE WHEN actual_label = 1 AND decision = 'APPROVE' THEN transaction_amount ELSE 0 END) AS missed_fraud_amount_exposure
FROM transaction_decisions;
