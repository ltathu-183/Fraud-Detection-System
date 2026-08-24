# V2 adaptive capacity-aware policy

## Design

Before each pseudo-future period, V2 recalculates thresholds using only the preceding labelled policy block:

- review target: 15% of transactions, derived from the recent score distribution;
- decline threshold: maximize historical fraud triage while the one-sided 95% Wilson upper bound for legitimate auto-decline remains at or below 2%;
- thresholds freeze before pseudo-future scoring.

The 15% review target is an operating target, not a guarantee: future review rates can move when score distributions shift. Common system constraint counts below use the canonical region of ≥80% triage, ≤20% review, and ≤2% legitimate auto-decline.

For the “recent model only” cell, static means carrying forward the V1 expanding-model raw-score thresholds. This deliberately tests threshold portability; because model score scales can change, the adaptive recent-model cell is the fairer operational deployment design.

## 2×2 system ablation

| System | Mean PR-AUC | Worst PR-AUC | Mean triage | Worst triage | Mean review | Max legit decline | Constraints satisfied |
|---|---:|---:|---:|---:|---:|---:|---:|
| V1 static | **0.5053** | **0.4240** | 79.09% | 72.95% | 14.34% | 2.21% | 0/4 |
| Adaptive policy only | **0.5053** | **0.4240** | **80.10%** | **75.43%** | 15.12% | 2.21% | 0/4 |
| Recent model only | 0.4936 | 0.3985 | 74.29% | 68.74% | 11.63% | **1.54%** | 1/4 |
| Full V2 adaptive | 0.4936 | 0.3985 | 78.96% | 74.88% | 14.97% | 2.21% | 1/4 |

Policy adaptation cannot change predictive ranking: expanding-model PR-AUC is identical with static and adaptive thresholds. It raised mean triage by 1.00 percentage point, raised worst triage by 2.48 points, and reduced triage standard deviation from 4.64 to 2.99 points, but increased mean review by 0.78 points and still achieved 0/4 full-constraint successes. In short, policy adaptation modestly improved operational robustness without improving predictive ranking, but not enough to justify replacing V1.

Recent-window training reduced ranking quality and did not rescue the weak Window 3. The full adaptive system met all common constraints in one window but had worse mean/worst PR-AUC and worse mean triage than adaptive policy with the expanding model.

## Monitoring redesign

Immediate, unlabeled monitoring should cover schema and missingness, unseen categories, feature and score distributions, decision rates, review-capacity usage, PSI, and KS. These signals can trigger investigation but cannot establish ranking or policy performance.

After labels mature, monitoring should add prevalence, PR-AUC/ROC-AUC, Brier/log loss/ECE, review fraud yield, legitimate auto-decline, realised triage coverage, and segment/time-slice outcomes. Retraining or recalibration should occur only after labels mature and a governed trigger is confirmed; the candidate must then be frozen and evaluated on genuinely newer untouched labels.

## Decision

**STATIC V1 REMAINS PREFERRED**

V2 is not selected merely for smoother average policy behavior. Neither recent training nor adaptive policy produced a sufficiently strong robustness improvement under customer and operations constraints. This conclusion is development-stage evidence, not a new final validation.

Machine-readable evidence: `artifacts/v2/system_ablation.json` and `artifacts/v2/walk_forward_results.json`.
