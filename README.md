# Fraud Detection & Operational Decisioning

> Leakage-aware fraud modelling and operational decision policy under review-capacity and legitimate-customer constraints.

Portfolio/reference implementation on **590,540 public IEEE-CIS transactions**. It does not represent Techcombank data, economics, systems, or expected performance.

## Problem

A fraud model does not directly make a banking decision. A useful decision system must balance:

- fraud risk and coverage;
- finite manual-review workload;
- legitimate-customer impact.

This project therefore converts a LightGBM fraud score into explicit **APPROVE / REVIEW / DECLINE** actions.

## System

```text
IEEE-CIS transactions
        ↓
Leakage-safe temporal feature engineering
        ↓
LightGBM fraud risk score
        ↓
Policy selection on a dedicated policy split
        ↓
APPROVE / REVIEW / DECLINE
        ↓
Frozen final-test evaluation
        ↓
Monitoring / TreeSHAP / SQL analytics
```

## Evaluation discipline

```text
train → validation → policy → threshold freeze → untouched final test
```

- **Train:** fits preprocessing statistics, class weight, and model parameters.
- **Validation:** controls LightGBM early stopping only.
- **Policy:** selects operational thresholds under triage, capacity, and customer-impact constraints.
- **Final test:** evaluates the already-frozen model and policy once.

Rows are stably ordered by `TransactionDT, TransactionID`. Point-in-time windows use `[t-W, t)`, excluding the current event, same-time peers, and future rows. The final test is now consumed and cannot be reused for development.

## Canonical results — V1 frozen

| Final-test metric | Result |
|---|---:|
| ROC-AUC | **0.9014** |
| PR-AUC | **0.5181** |
| Fraud auto-decline recall | **48.47%** |
| Fraud routed to review | **30.41%** |
| Fraud triage coverage | **78.88%** |
| Review workload | **13.31%** |
| Legitimate auto-decline | **1.77%** |

PR-AUC is important because final-test fraud prevalence is only 3.91%. Reviewed fraud means routed to investigators; it is not assumed detected or recovered.

> The policy satisfied the 80% target on the policy split but achieved **78.88% on final test**. It did not meet the final-test 80% target, and thresholds were not retrospectively retuned.

## Key finding

> **Good predictive discrimination did not guarantee stable operational constraint transfer over time.**

Four pre-final pseudo-future evaluations produced PR-AUC of **0.4240–0.5496** and triage coverage of **72.95%–83.83%**; zero of four satisfied every canonical constraint. Small marginal feature/score PSI values did not identify the largest degradation, so simple marginal drift monitoring is not proof of policy stability.

## What I tested after observing instability — V2 exploratory

- recent-window LightGBM retraining;
- recency-weighted expanding training;
- capacity-aware adaptive thresholds with a Wilson customer-risk bound;
- historical-only Platt and isotonic calibration.

Expanding-history LightGBM remained best for mean and worst PR-AUC. Adaptive thresholds modestly stabilized triage but did not justify replacement, while Platt scaling improved calibration metrics without improving ranking. **None provided enough evidence to replace frozen V1.** V2 used zero final-test rows and remains exploratory/not promoted.

## Engineering and governance

- identity, chronology, same-timestamp, train-only preprocessing, and threshold-provenance tests;
- immutable final manifest with configuration, boundaries, metrics, and SHA-256 hashes;
- TreeSHAP/global gain and local model-behavior explanations;
- data-quality, marginal drift, score, performance, and operational monitoring;
- executable SQLite analytics and dashboard-ready aggregate exports.

Technical reviewer path:

1. [Audit summary](reports/AUDIT_SUMMARY.md)
2. [Frozen temporal robustness](reports/TEMPORAL_ROBUSTNESS.md)
3. [Policy robustness](reports/POLICY_ROBUSTNESS.md)
4. [V2 adaptive comparison](reports/v2/ADAPTIVE_POLICY.md)
5. [V2 calibration](reports/v2/CALIBRATION.md)
6. [CV-safe claim ledger](reports/CV_CLAIMS.md)

Canonical entry point: `src.pipeline.ieee_cis_pipeline`. V1 is **CANONICAL / FROZEN**; `src/v2/` is **EXPLORATORY / NOT PROMOTED**; tiered rules, feature store, calibration, adversarial-validation, and submission modules are **LEGACY / NON-CANONICAL**.

## Limitations

Public IEEE-CIS data; no bank-specific validation or observed review outcomes; hypothetical costs only; final test consumed; four rolling pseudo-future windows; no new future holdout; no production serving, security, scalability, or governance validation. This is not production-ready.

Repository code is released under the [MIT License](LICENSE). IEEE-CIS data has separate availability and usage terms; raw competition CSVs are ignored and are not distributed here. See [public release notes](reports/PUBLIC_RELEASE.md).

## Reproduce

Supported Python: **3.11**. Place `train_transaction.csv` and `train_identity.csv` in `data/raw/`; the committed DVC pointers do not provide a usable public remote.

For the exact locked environment, use `uv sync --frozen`; `requirements.txt` is the concise pip-compatible dependency list. Dependency versions were not upgraded during release polish.

```bash
python -m pip install -r requirements.txt
python -m pytest -q
python -m src.pipeline.ieee_cis_pipeline
python scripts/run_sql_analytics.py
```

The full pipeline command reproduces evidence; it must not be used to select new changes on the consumed test. Any future model or policy redesign requires genuinely later untouched labeled data or a new dataset. See [project status](PROJECT_STATUS.md).
