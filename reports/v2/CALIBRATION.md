# V2 probability calibration

## Protocol

For every model/window, Platt logistic calibration and isotonic calibration were fit only on the most recent labelled policy block. Calibrators were frozen before the next pseudo-future period. The evaluation period supplied no calibration inputs.

## Mean pseudo-future diagnostics

| Model | Calibration | Brier | Log loss | 10-bin ECE | PR-AUC |
|---|---|---:|---:|---:|---:|
| Expanding | Uncalibrated | 0.0623 | 0.2285 | 0.1401 | 0.5053 |
| Expanding | Platt | 0.0244 | 0.0991 | 0.0085 | 0.5053 |
| Expanding | Isotonic | **0.0238** | **0.0981** | **0.0046** | 0.4889 |
| Recent | Uncalibrated | 0.0536 | 0.2039 | 0.1209 | 0.4936 |
| Recent | Platt | 0.0248 | 0.1010 | 0.0082 | 0.4936 |
| Recent | Isotonic | **0.0243** | **0.1007** | **0.0052** | 0.4752 |
| Weighted | Uncalibrated | 0.0596 | 0.2222 | 0.1356 | 0.4911 |
| Weighted | Platt | 0.0249 | 0.1015 | 0.0081 | 0.4911 |
| Weighted | Isotonic | **0.0243** | **0.1004** | **0.0050** | 0.4754 |

The weighted LightGBM raw scores are not calibrated probabilities. Both methods substantially improved Brier score, log loss, and ECE in these windows. Platt scaling is monotonic and preserved ROC-AUC/PR-AUC exactly. Isotonic created tied predictions and reduced average PR-AUC, so its lower calibration errors must not be described as improved ranking.

These comparisons are exploratory on four known periods. No calibrator is promoted for V1 or validated on a new holdout. Per-window calibration curves are stored in `artifacts/v2/walk_forward_results.json`.
