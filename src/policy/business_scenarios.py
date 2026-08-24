"""Scenario-only economic evaluation for three-way fraud decisions."""

from dataclasses import asdict

import numpy as np

from src.evaluation.metrics import evaluate_policy


DISCLAIMER = (
    "Scenario assumptions for methodological evaluation only; "
    "not representative of any bank's actual economics."
)


def expected_cost(y_true, probabilities, low_threshold, high_threshold, scenario):
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    approve = p < low_threshold
    decline = p >= high_threshold
    review = ~(approve | decline)
    fraud = y == 1
    legitimate = ~fraud
    missed = int((fraud & approve).sum())
    reviewed_fraud = int((fraud & review).sum())
    false_declines = int((legitimate & decline).sum())
    reviews = int(review.sum())
    unrecovered_reviewed_fraud = reviewed_fraud * (1 - scenario.reviewed_fraud_recovery_rate)
    total = (
        scenario.missed_fraud_cost * (missed + unrecovered_reviewed_fraud)
        + scenario.false_decline_cost * false_declines
        + scenario.manual_review_cost * reviews
    )
    return {
        "expected_cost_total": float(total),
        "expected_cost_per_transaction": float(total / len(y)),
        "missed_fraud_count": missed,
        "reviewed_fraud_count": reviewed_fraud,
        "false_decline_count": false_declines,
        "manual_review_count": reviews,
    }


def optimize_cost_policy(y_true, probabilities, scenario, grid_size=101):
    """Optimize illustrative cost on the policy split only."""
    p = np.asarray(probabilities, dtype=float)
    grid = np.unique(np.r_[0.0, np.quantile(p, np.linspace(0, 1, grid_size)), 1.0])
    candidates = []
    for low in grid:
        for high in grid[grid >= low]:
            cost = expected_cost(y_true, p, float(low), float(high), scenario)
            candidates.append((cost["expected_cost_total"], float(high), float(low), cost))
    best = min(candidates, key=lambda item: item[:3])
    return {"low_threshold": best[2], "high_threshold": best[1], **best[3]}


def compare_policies(y_true, probabilities, policies, scenario):
    output = {}
    for name, thresholds in policies.items():
        low, high = thresholds
        output[name] = {
            "thresholds": {"low": float(low), "high": float(high)},
            "operational_metrics": evaluate_policy(y_true, probabilities, low, high),
            "scenario_cost": expected_cost(y_true, probabilities, low, high, scenario),
        }
    return {"disclaimer": DISCLAIMER, "assumptions": asdict(scenario), "policies": output}
