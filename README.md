# IEEE-CIS Fraud Detection — corrected portfolio pipeline

This repository supports one execution path:

```text
train_transaction.csv + train_identity.csv
  -> stable chronological order and immutable row/split IDs
  -> 70% train / 15% model validation / 7.5% policy / 7.5% final test
  -> train-fitted frequency encoding
  -> point-in-time behavioural features
  -> weighted LightGBM with validation-only early stopping
  -> hard-constrained policy selection on the policy split
  -> one frozen final-test evaluation
```

The active entry point is `src.pipeline.ieee_cis_pipeline`. The tiered pipeline,
rules engine, feature store, calibration module, adversarial validation, and
submission script are legacy/experimental and are not used by this path.

## Correctness contract

- `_internal_row_id` and `_split_id` are assigned before feature work. Splits
  are recovered only by `_split_id`, never by position after a transformation.
- The active raw transaction schema is `TransactionAmt`, `ProductCD`, card,
  address, distance, email-domain, and C/D/M feature families, plus IEEE identity
  fields. The high-dimensional `V*` transaction family is intentionally excluded
  so the documented in-memory path fits on a developer machine.
- Categorical frequencies and all preprocessing statistics are fitted on train.
- Behavioural windows contain `[t-W, t)`: the current transaction, same-time
  peers, and future transactions are excluded. Missing entities have no history.
- No fraud-label-derived feature is active because label maturity is not modeled.
- LightGBM receives native `NaN`; only infinities are converted to `NaN`.
- `scale_pos_weight = N_negative_train / N_positive_train`.
- Validation is used only for model selection/early stopping. Policy thresholds
  are selected only on the policy split. The test split is opened only after a
  feasible policy has been frozen.
- Policy constraints have one source in `src/pipeline/config.py`: fraud triage
  coverage at least 80%, overall review rate at most 20%, and legitimate
  auto-decline rate at most 2%. Infeasibility stops evaluation; constraints are
  never silently relaxed.

`fraud_review_coverage` reports routing, not successful fraud detection.
`fraud_triage_coverage` is auto-declined fraud plus reviewed fraud and likewise
does not assume review effectiveness. Ranking ROC-AUC and average precision are
reported separately from operational policy metrics.

## Data and canonical run

The tracked `.dvc` files are pointers only. The previous `.dvc/config` referenced
the placeholder `/path/to/dvc/remote/storage`; no reconstructable remote or
credentials are available. Obtain the IEEE-CIS data under its applicable terms
and place these files locally:

```text
data/raw/train_transaction.csv
data/raw/train_identity.csv
```

Then run:

```bash
python -m pytest
python -m src.pipeline.ieee_cis_pipeline
```

Corrected results are written to `artifacts/corrected_results.json`. This checkout
contains the required raw CSVs. Run the command above to reproduce the committed
post-audit report. All performance numbers previously shown in this README were
removed as invalid; only numbers emitted by this command are valid.

## Corrected frozen-test results

Generated on the supplied raw data by the canonical command above:

| Model | ROC-AUC | PR-AUC | Notes |
|---|---:|---:|---|
| Dummy prior | 0.5000 | 0.0391 | Same frozen test |
| Linear logistic-loss SGD | 0.7972 | 0.2121 | Train-only imputation/scaling |
| LightGBM, no temporal features | 0.8977 | 0.5117 | Ablation; same split/protocol |
| LightGBM, active feature set | 0.9010 | 0.5051 | Corrected final model |

The active policy met its pre-specified constraints on the final test: 80.32%
fraud triage coverage, 15.25% overall review rate, and 1.80% legitimate
auto-decline rate. It auto-declined 47.84% of fraud and routed 32.49% of fraud
to review. Review routing is not claimed as confirmed fraud detection.

The ablation improves PR-AUC slightly while reducing ROC-AUC. This is not used
to revise the selected model after testing; it is reported as a post-audit
diagnostic, so the active configuration remains the pre-specified feature set.
Because that diagnostic compared model variants on the test split, this test
split must not be used for any future model-selection decision. A newly reserved
chronological holdout is required before promoting a research change.

## Research track — not a new performance claim

The next experiments are validation-only until a new holdout is reserved:

1. Point-in-time behavioural features: card/device/email/address novelty and
   entity age. They use only strictly earlier transactions and preserve raw
   entity identifiers until feature generation is complete.
2. Capacity evaluation: report `Recall@5%`, `Recall@10%`, `Recall@15%` review
   capacity and their precision counterparts, alongside PR-AUC. These measure
   ranking under realistic review limits without selecting a threshold on test.
3. Temporal stability: use rolling chronological validation before considering
   calibration, cost assumptions, or more complex relational models.

No cost-sensitive policy is claimed yet because the data contains no real bank
loss, customer-friction, or review-cost inputs. No target/fraud-history feature
is active because label-maturity timing is unknown.

## Limitations

This is an in-memory pandas portfolio pipeline, not a serving system. It has no
real bank cost data, measured reviewer effectiveness, production API, operational
monitoring evidence, or scalability validation. Fraud labels may be delayed, but
label maturity is not modeled. IEEE-CIS results may not generalize to another
merchant, geography, time period, or bank. Weighted LightGBM scores are not
assumed to be calibrated; calibration is optional/unused in the supported path.
