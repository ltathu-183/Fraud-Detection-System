# CV claim ledger

## SAFE FOR CV

1. Built a leakage-aware fraud decision pipeline using LightGBM with separate chronological train, validation, policy, and final-test stages, achieving ROC-AUC 0.901 and PR-AUC 0.518 on IEEE-CIS.

2. Designed an APPROVE / REVIEW / DECLINE policy balancing fraud coverage, manual-review capacity, and legitimate-customer impact; final evaluation auto-declined 48.5% of fraud and routed 30.4% to review while limiting legitimate auto-decline to 1.77%.

3. Evaluated policy robustness across rolling temporal windows and retained the simpler frozen design after adaptive retraining and thresholding failed to improve temporal generalization.

Evidence: `artifacts/evaluation/canonical_final_frozen.json`, `artifacts/evaluation/rolling_temporal_robustness.json`, `artifacts/v2/system_ablation.json`, and the linked robustness reports.

## SAFE FOR INTERVIEW

- Validation controls model training; a separate policy split selects thresholds; the final test evaluates both frozen choices once.
- The policy met 80% triage on its policy split but achieved 78.88% on final test; thresholds were not retrospectively retuned.
- Rolling PR-AUC ranged 0.4240–0.5496 and triage ranged 72.95%–83.83%; zero of four static-policy windows met every constraint.
- Small marginal PSI values did not identify Window 3 degradation; PSI is descriptive, not proof of stability.
- V2 recent-window and recency-weighted training worsened mean/worst PR-AUC; adaptive thresholds modestly stabilized triage but did not justify replacing V1.
- Historical-only Platt calibration improved Brier score, log loss, and ECE while leaving ranking unchanged.
- Hypothetical costs demonstrate methodology and sensitivity only; they are not bank economics.

## DO NOT CLAIM

- “Met the 80% triage constraint on final test.”
- “Guaranteed ≥80% fraud coverage” or any guaranteed fraud-coverage claim.
- “Production-ready fraud system” or “bank-grade fraud system.”
- “Real bank cost optimization.”
- “Temporal policy is stable” or “stable under drift.”
- “PSI proves no drift” or “PSI proved no shift.”
- “V2 improved final performance.”
- “Reviewed fraud was necessarily detected or recovered.”
- “Results represent Techcombank data or performance.”
- “Statistically significant temporal degradation.”
