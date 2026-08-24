# Temporal robustness

## Historical canonical final result

The frozen canonical final evidence remains ROC-AUC 0.9014, PR-AUC 0.5181, fraud triage coverage 78.88%, review workload 13.31%, and legitimate auto-decline 1.77%. It is preserved in `artifacts/evaluation/canonical_final_frozen.json`.

The final test was consumed under the previously frozen policy and is not reused for subsequent development.

## Subsequent pre-final rolling robustness analysis

Four expanding-history experiments use only rows 0–546,248. Each has a 44,290-row validation block, a disjoint 44,290-row policy block, and the next 44,290 rows as pseudo-future evaluation. The last evaluation ends exactly where the consumed final test begins.

| Window | Pseudo-future rows | TransactionDT range | Fraud cases | Prevalence | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| 1 | 369,089–413,378 | 9,169,152–10,438,003 | 2,028 | 4.58% | 0.9074 | 0.5485 |
| 2 | 413,379–457,668 | 10,438,017–11,725,712 | 1,593 | 3.60% | 0.9046 | 0.5496 |
| 3 | 457,669–501,958 | 11,725,727–13,151,840 | 1,449 | 3.27% | 0.8823 | 0.4240 |
| 4 | 501,959–546,248 | 13,151,880–14,417,365 | 1,350 | 3.05% | 0.9018 | 0.4989 |

Across four windows, ROC-AUC ranged 0.8823–0.9074 (mean 0.8990; standard deviation 0.0099). PR-AUC ranged 0.4240–0.5496 (mean 0.5053; standard deviation 0.0512). Prevalence ranged 3.05%–4.58%. The sharp PR-AUC fall in Window 3 means predictive quality is not uniformly stable even though ROC-AUC varies less.

## Failure analysis

Score PSI from each policy period to its pseudo-future period was only 0.0007–0.0036. The monitored feature PSIs were also small; the largest was 0.0175 for `D1` in Window 2. These selected marginal drift measures did not flag the large Window 3 PR-AUC and triage deterioration. This does not prove absence of shift: PSI can miss conditional, interaction, label-prevalence, and score/label-separation changes.

With only four temporal observations, correlations are descriptive and too unstable for inference. No causal claim is made. The most relevant observed association is that Window 3 combined the lowest PR-AUC with the lowest triage coverage, while small marginal PSI values did not explain the failure.

## Temporal uncertainty

The empirical window distributions are the uncertainty evidence: PR-AUC 0.4240–0.5496, triage coverage 72.95%–83.83%, review workload 11.21%–16.76%, and legitimate auto-decline 1.21%–2.21%. Four periods are enough to reveal instability but too few for formal confidence or significance claims. Random row bootstrap intervals would not address this temporal variation.

## Decision

**MATERIAL TEMPORAL INSTABILITY**

Predictive ranking and operational constraint transfer vary enough that a safety margin cannot be treated as a reliable fix.
