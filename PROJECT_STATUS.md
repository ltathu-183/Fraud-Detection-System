# Project status — version 1.0

**Status: FROZEN FOR PORTFOLIO USE**

- **CANONICAL / FROZEN:** V1 expanding-history LightGBM and APPROVE / REVIEW / DECLINE policy.
- **Canonical entry point:** `python -m src.pipeline.ieee_cis_pipeline`.
- **Final test:** consumed once after threshold freeze; unavailable for any further selection or tuning.
- **Frozen evidence:** `artifacts/evaluation/canonical_final_frozen.json`.
- **Canonical metrics:** ROC-AUC 0.9014; PR-AUC 0.5181; triage 78.88%; review workload 13.31%; legitimate auto-decline 1.77%.
- **Scientific conclusion:** material temporal instability; development constraints did not reliably transfer.
- **EXPLORATORY / NOT PROMOTED:** V2 recent training, recency weighting, adaptive policy, and calibration. V2 used zero final-test rows; **STATIC V1 REMAINS PREFERRED**.
- **LEGACY / NON-CANONICAL:** tiered rules, feature-store, calibration, adversarial-validation, and submission modules outside the supported path.

Further model or policy development requires genuinely later untouched labeled data or a new dataset.
