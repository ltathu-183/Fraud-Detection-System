# Policy evaluation

All policy thresholds are selected on the chronological policy split and frozen before the final test. REVIEW means routed for investigation; it is not counted as confirmed fraud detection.

| Frozen policy (final-test evaluation) | Fraud triage | Review rate | Legitimate decline | Fraud auto-decline |
|---|---:|---:|---:|---:|
| Default 0.5 | 66.88% | 0.00% | 6.50% | 66.88% |
| Policy-split F1 threshold | 42.35% | 0.00% | 0.92% | 42.35% |
| Operational constraints | 78.88% | 13.31% | 1.77% | 48.47% |
| Hypothetical cost-sensitive | 82.17% | 17.31% | 0.92% | 42.35% |

The operational policy met all selection constraints on policy (80.00% triage, 13.00% review, 1.63% legitimate decline), but final-test triage fell to 78.88%. This is evidence of imperfect temporal stability. It would require investigation or a governed buffer in constraints—not retrospective test tuning.

The cost-sensitive policy is not recommended as a bank policy. It is only the optimizer result under documented fictional assumptions.
