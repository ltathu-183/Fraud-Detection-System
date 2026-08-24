# Model evaluation

The canonical model is a class-weighted LightGBM trained on the first 70% of chronological rows with validation-only early stopping. Frequency encodings and linear-baseline preprocessing are fit on train only.

| Model (final test) | ROC-AUC | PR-AUC |
|---|---:|---:|
| Dummy prior | 0.5000 | 0.0391 |
| Linear logistic-loss SGD | 0.7947 | 0.1421 |
| Canonical LightGBM | 0.9014 | 0.5181 |

PR-AUC is more diagnostic for the rare positive class because it measures the precision/recall trade-off rather than averaging ranking over the abundant negatives.

The no-card1-temporal-feature ablation is a validation-only research result: ROC-AUC 0.9154, PR-AUC 0.5457. The canonical validation ROC-AUC was 0.9160. This mixed result does not justify promotion, and no significance claim is made. Bootstrap intervals were not added because naive row bootstrap would ignore temporal dependence; rolling later-period evaluation is the higher-value next step.

Source: `artifacts/corrected_results.json`. Reproduce with `python -m src.pipeline.ieee_cis_pipeline`.
