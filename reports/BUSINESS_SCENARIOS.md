# Business scenario analysis

> Scenario assumptions for methodological evaluation only; not representative of any bank's actual economics.

The example assigns 500 units to missed fraud, 100 to falsely declining a legitimate transaction, 5 per manual review, and assumes reviewers recover 50% of reviewed fraud. Transaction amount is exported for analysis but is not treated as verified loss.

| Frozen policy | Illustrative cost / final-test transaction |
|---|---:|
| Default 0.5 | 12.72 |
| Policy-split F1 threshold | 12.16 |
| Operational constraints | 9.47 |
| Cost-sensitive | 9.13 |

The cost-sensitive policy is lowest only within this scenario. Changing missed-fraud severity, customer-friction cost, review cost, or reviewer effectiveness can change the ordering. Actual use would require finance, fraud-operations, risk, conduct, and model-governance inputs.
