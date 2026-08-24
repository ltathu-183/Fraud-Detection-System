# Policy robustness and safety margins

## Historical canonical final result

The consumed final-test result remains unchanged: 78.88% triage coverage, 13.31% review workload, and 1.77% legitimate auto-decline. No rolling outcome was used to retrofit it.

## Canonical 80% development target

| Window | Review threshold | Decline threshold | Future triage | Future review | Future legitimate decline | All constraints |
|---|---:|---:|---:|---:|---:|---|
| 1 | 0.2716 | 0.7013 | 83.83% | 16.76% | 2.21% | No |
| 2 | 0.2820 | 0.6578 | 76.27% | 11.21% | 1.21% | No |
| 3 | 0.4042 | 0.8147 | 72.95% | 13.31% | 1.40% | No |
| 4 | 0.2756 | 0.7218 | 83.33% | 16.07% | 2.00% | No |

Threshold movement is material, especially in Window 3. Zero of four windows satisfied all constraints. Two missed triage; two exceeded the legitimate-decline ceiling, although Window 4 exceeded it by only 0.003 percentage points. No window breached the 20% review limit under the 80% target. The worst triage shortfall was 7.05 percentage points and the worst legitimate-decline excess was 0.21 percentage points.

## Predeclared safety-margin experiment

All thresholds were selected on policy windows before pseudo-future scoring. Success below always means the original canonical constraints (≥80% future triage, ≤20% review, ≤2% legitimate decline).

| Development target | Policy-feasible windows | Mean future triage | Future triage range | Mean review | Mean legitimate decline | All-constraint success |
|---:|---:|---:|---:|---:|---:|---:|
| 80% | 4/4 | 79.09% | 72.95–83.83% | 14.34% | 1.71% | 0/4 |
| 82% | 4/4 | 81.60% | 76.60–85.40% | 16.88% | 1.71% | 0/4 |
| 85% | 3/4 | 82.82% | 79.16–86.37% | 19.25% | 1.54% | 1/4 |

An 82% target increased average future triage by 2.51 percentage points but also increased review workload by 2.54 points and still never satisfied every constraint. The 85% target was infeasible on one policy window, breached review capacity on another, and succeeded fully in only one of four windows. Therefore no margin is selected as a new policy.

`artifacts/evaluation/policy_frontier.csv` is the development-only policy frontier. It shows why maximizing coverage alone is operationally unacceptable: higher coverage consumes review capacity and can still fail customer-impact limits.

Rolling pre-final experiments suggest that higher development coverage can increase average next-period triage, but no tested margin demonstrated reliable constraint generalization. This strategy has not been validated on a new untouched future holdout.
