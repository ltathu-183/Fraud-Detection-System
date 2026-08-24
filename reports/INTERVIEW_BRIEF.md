# Interview brief

Private preparation notes for discussing the frozen portfolio project.

## Core technical questions

### Why PR-AUC instead of accuracy?

Final-test fraud prevalence is 3.91%, so a model can achieve high accuracy by predicting almost everything as legitimate. PR-AUC focuses on precision and recall for the fraud class and is more informative under this imbalance.

### Why is ROC-AUC not enough?

ROC-AUC measures ranking across all false-positive rates and can remain strong when precision at realistic operational capacity is weak. I report both: final ROC-AUC was 0.9014 and PR-AUC 0.5181.

### Why separate validation, policy, and final test?

Validation controls model training and early stopping. Policy selects APPROVE/REVIEW/DECLINE thresholds under operational constraints. Final test then evaluates both frozen choices once; combining these stages would leak future evidence into selection.

### Why not use threshold 0.5?

A 0.5 cutoff is not tied to fraud prevalence, review capacity, or customer-impact limits, and weighted LightGBM scores are not claimed calibrated. Thresholds should express the operational policy, selected outside the final test.

### What do the policy metrics mean?

- **Fraud recall:** share of fraud assigned to a chosen positive action; its meaning depends on that action.
- **Auto-decline recall:** share of all fraud stopped automatically.
- **Review-routed fraud:** share of all fraud sent to investigators; routing is not confirmed recovery.
- **Triage coverage:** auto-declined plus review-routed fraud, without assuming review success.

### Why did final test miss the 80% target?

The target was satisfied on the policy period, but score/label relationships and the operating population changed over time. The final 78.88% result is evidence that development constraints do not automatically transfer.

### Why not retune after seeing 78.88%?

That would turn the final test into another policy-selection set and produce optimistic evidence. I preserved the miss and required a genuinely later holdout for any redesigned policy.

### What did the rolling experiment reveal?

Across four pre-final pseudo-future periods, PR-AUC ranged 0.4240–0.5496 and triage coverage 72.95%–83.83%. None of the 80%-target windows met all constraints, and higher development margins were not consistently feasible or transferable.

### Why did PSI not flag Window 3?

PSI here measures marginal distribution changes. It can miss conditional shifts, feature interactions, prevalence changes, and changes in score/label separation. Low marginal PSI therefore cannot guarantee stable PR-AUC or policy performance.

### What monitoring would you use in production?

Before labels mature: schema/data-quality checks, missingness, unseen categories, feature and score distributions, and decision/workload rates. After labels mature: PR-AUC, ROC-AUC, calibration diagnostics, triage coverage, review outcomes, false declines, and segment/time-slice performance, with governed investigation thresholds.

### What would you do with newer labeled data?

Freeze one candidate methodology before access, evaluate it on a genuinely later untouched holdout, and compare ranking, calibration, operational constraints, customer impact, and temporal segments without retrospective tuning.

### What business inputs are missing?

Observed fraud loss and recovery, reviewer cost/capacity/effectiveness, customer-friction and attrition effects, amount exposure semantics, label delays, segment risk appetite, and governance-approved decision limits.

### Why is this not production-ready?

It is an offline in-memory public-data study without bank validation, serving/security/scalability evidence, reviewer outcomes, mature cost inputs, calibrated-probability assurance, or production model governance.

## STAR stories

### Story 1 — Correctness over headline metrics

- **Situation:** The repository contained impressive historical components and results, but the evidence path and final-test use required verification.
- **Task:** Establish one scientifically defensible canonical pipeline.
- **Action:** Traced row identity, temporal features, train-only preprocessing, class weights, and policy provenance; separated validation, policy, and final test; froze thresholds before test access; added correctness tests and machine-readable evidence.
- **Result:** Produced a reproducible final result—ROC-AUC 0.9014 and PR-AUC 0.5181—while removing unsupported claims. The lesson was that a higher-looking result is not useful if its evaluation is invalid.

### Story 2 — Policy generalization failure

- **Situation:** The policy achieved the required 80% triage coverage on its selection period but only 78.88% on final test.
- **Task:** Determine whether this was an isolated miss without retuning on consumed evidence.
- **Action:** Preserved the final result and ran four strictly pre-final rolling policy-to-pseudo-future evaluations with predeclared 80%, 82%, and 85% targets.
- **Result:** Triage ranged 72.95%–83.83%, and no margin reliably transferred all constraints. I rejected retrospective threshold repair and concluded that new future data is required.

### Story 3 — Monitoring limitation

- **Situation:** Window 3 showed the weakest PR-AUC, 0.4240, and only 72.95% triage coverage.
- **Task:** Check whether existing drift indicators explained the deterioration.
- **Action:** Compared policy and pseudo-future score PSI plus major-feature marginal PSI while keeping the analysis descriptive.
- **Result:** PSI values stayed small and did not identify the degradation. The lesson was that marginal drift monitoring must be complemented by label-aware performance, operational metrics, and repeated temporal validation.
