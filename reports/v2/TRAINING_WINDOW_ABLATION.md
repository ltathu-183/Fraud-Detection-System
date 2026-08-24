# V2 training-window ablation

## Scope and protocol

This is exploratory development evidence on the same four pre-final windows already inspected in V1. It is not a new holdout, does not alter frozen V1, and uses zero V1 final-test rows.

Three treatments were predeclared with the same LightGBM architecture and core features:

1. expanding history, matching V1 walk-forward training;
2. the most recent 177,160 training rows (four 44,290-row blocks);
3. expanding history with exponential weights and an 88,580-row half-life.

No training length, half-life, feature, or hyperparameter was selected using pseudo-future results.

## Predictive results

| Training treatment | Mean PR-AUC | Worst PR-AUC | PR-AUC standard deviation | Mean ROC-AUC | Worst ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Expanding history | **0.5053** | **0.4240** | **0.0512** | **0.8990** | **0.8823** |
| Recent 177,160 rows | 0.4936 | 0.3985 | 0.0568 | 0.8946 | 0.8787 |
| Recency-weighted expanding | 0.4911 | 0.3958 | 0.0582 | 0.8922 | 0.8727 |

| Window | Expanding PR-AUC | Recent PR-AUC | Weighted PR-AUC |
|---|---:|---:|---:|
| 1 | 0.5485 | 0.5346 | 0.5357 |
| 2 | 0.5496 | 0.5399 | 0.5409 |
| 3 | **0.4240** | 0.3985 | 0.3958 |
| 4 | 0.4989 | **0.5015** | 0.4921 |

Recent-window retraining improved only Window 4 by 0.0026 PR-AUC and worsened the drift-heavy Window 3 by 0.0255. Recency weighting was also worse in Window 3. The hypothesis that discarding or down-weighting older training data fixes the observed instability is not supported by this predeclared experiment.

## Interpretation

Expanding history remains the preferred development benchmark for predictive ranking. This does not prove that all recency schemes are inferior; it shows that these two simple, leakage-safe choices did not improve temporal robustness. Searching more lengths or half-lives on these already-inspected periods would be retrospective tuning.

Machine-readable evidence: `artifacts/v2/walk_forward_results.json`.
