# Dashboard-ready schema

No `.pbix` is generated. `artifacts/dashboard/transaction_decisions.csv` is the row-level fact table with transaction ID, relative timestamp, amount, actual evaluation label, score, decision, and available segments. SQL-generated aggregates are also CSV files.

Suggested Power BI pages:

1. Transaction overview: volume and amount.
2. Fraud risk: score distribution, fraud rate, auto-decline recall, triage coverage.
3. Review workload: review rate, volume, and review precision (evaluation only).
4. Customer impact: legitimate auto-decline count/rate.
5. Time monitoring: relative-week volume, fraud, score, and decision trends.
6. Segments: product-level volume and policy outcomes.

IEEE `TransactionDT` is seconds from an undisclosed origin, so time outputs use relative week, not fabricated calendar dates.
