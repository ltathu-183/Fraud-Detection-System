# Audit summary

## What was wrong

The corrected model path existed, but evidence stopped at a single JSON file. There was no cost scenario, executable SQL layer, BI schema, generated monitoring report, or explainability output. A temporal ablation had been evaluated on final test, consuming that holdout for the comparison. Legacy modules remained beside canonical code and contained unsafe target aggregates, but were only documented—not structurally marked—as experimental. Some README claims no longer matched the current expanded temporal feature set.

## What changed

The canonical entry point now freezes four policy variants on policy, generates final-test decision/SQLite/dashboard artifacts, reports hypothetical costs, emits TreeSHAP and gain evidence, and creates label-aware monitoring. The ablation is validation-only. Tests protect cost semantics and SQL execution in addition to the existing identity/leakage contracts. Legacy modules remain preserved and explicitly non-canonical.

## What remains imperfect

The final test has been consumed; model changes need a new later holdout. Four pre-final rolling windows show material temporal instability: PR-AUC and policy transfer vary, and safety margins do not reliably satisfy all constraints. There is no label-maturity model, observed review outcome, real cost input, calibrated-probability claim, production serving system, or bank-specific validation.

## Commands

`python -m pytest -q`

`python -m src.pipeline.ieee_cis_pipeline`

`python scripts/run_sql_analytics.py`

`python -m scripts.run_rolling_robustness`
