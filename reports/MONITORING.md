# Monitoring

The generated report compares policy to final test as consecutive chronological windows.

- Data quality: missing rates for major numeric features.
- Drift: reference-quantile PSI for those features and model scores.
- With mature labels: ROC-AUC, PR-AUC, and APPROVE/REVIEW/DECLINE outcomes.
- Without labels: drift and decision-rate changes can trigger investigation but cannot prove performance degradation.

No universal PSI or performance alert threshold is asserted. Illustrative thresholds without portfolio backtesting create false precision. Investigation should consider data-pipeline changes, segment mix, score drift, capacity breaches, customer harm, and—once labels mature—ranking and policy outcomes. Data drift, score drift, and model-performance deterioration are related but not interchangeable.

Artifact: `artifacts/monitoring/monitoring_report.json`.
