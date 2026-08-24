"""Capacity-aware adaptive policy selected from historical labelled scores only."""

from dataclasses import dataclass
from statistics import NormalDist

import numpy as np

from src.evaluation.metrics import evaluate_policy


@dataclass(frozen=True)
class AdaptivePolicyConfig:
    review_capacity: float = 0.15
    max_legitimate_decline_rate: float = 0.02
    confidence_level: float = 0.95
    threshold_grid_size: int = 101


def wilson_upper_bound(successes, trials, confidence_level=0.95):
    """One-sided Wilson upper confidence bound for a binomial proportion."""
    if trials <= 0 or not 0 <= successes <= trials:
        raise ValueError("Wilson bound requires 0 <= successes <= positive trials")
    if not 0.5 < confidence_level < 1:
        raise ValueError("confidence_level must be in (0.5, 1)")
    z = NormalDist().inv_cdf(confidence_level)
    rate = successes / trials
    denominator = 1 + z * z / trials
    centre = rate + z * z / (2 * trials)
    radius = z * np.sqrt(rate * (1 - rate) / trials + z * z / (4 * trials * trials))
    return float((centre + radius) / denominator)


def _capacity_low_threshold(scores, high_threshold, review_capacity):
    """Choose a score quantile so REVIEW uses no more than the declared capacity."""
    scores = np.asarray(scores, dtype=float)
    non_declined = scores[scores < high_threshold]
    review_slots = int(np.floor(len(scores) * review_capacity))
    if review_slots <= 0 or len(non_declined) == 0:
        return float(high_threshold)
    if review_slots >= len(non_declined):
        return float(np.nextafter(non_declined.min(), -np.inf))
    return float(np.sort(non_declined, kind="mergesort")[-review_slots])


def select_adaptive_policy(y_policy, policy_scores, config=AdaptivePolicyConfig()):
    """Maximize historical fraud triage under capacity and Wilson customer-risk limits."""
    y = np.asarray(y_policy, dtype=int)
    scores = np.asarray(policy_scores, dtype=float)
    if len(y) != len(scores) or set(np.unique(y)) != {0, 1}:
        raise ValueError("Aligned binary policy labels and scores are required")
    legitimate = y == 0
    high_grid = np.unique(np.r_[np.quantile(scores, np.linspace(0, 1, config.threshold_grid_size)), 1.0])
    candidates = []
    for high in high_grid:
        decline = scores >= high
        legitimate_declines = int((legitimate & decline).sum())
        upper = wilson_upper_bound(legitimate_declines, int(legitimate.sum()), config.confidence_level)
        if upper > config.max_legitimate_decline_rate:
            continue
        low = _capacity_low_threshold(scores, float(high), config.review_capacity)
        metrics = evaluate_policy(y, scores, low, float(high))
        if metrics["overall_review_rate"] > config.review_capacity + 1 / len(scores):
            continue
        candidates.append((metrics["fraud_triage_coverage"], metrics["fraud_auto_decline_recall"],
                           -metrics["overall_review_rate"], float(high), low, upper, metrics))
    if not candidates:
        return {"feasible": False, "low_threshold": None, "high_threshold": None,
                "review_capacity": config.review_capacity,
                "legitimate_decline_wilson_upper": None}
    best = max(candidates, key=lambda item: item[:5])
    return {"feasible": True, "low_threshold": best[4], "high_threshold": best[3],
            "review_capacity": config.review_capacity,
            "confidence_level": config.confidence_level,
            "legitimate_decline_wilson_upper": best[5],
            "historical_policy_metrics": best[6],
            "thresholds_frozen_before_evaluation": True}


def apply_policy(scores, low_threshold, high_threshold):
    scores = np.asarray(scores, dtype=float)
    return np.where(scores < low_threshold, "APPROVE",
                    np.where(scores >= high_threshold, "DECLINE", "REVIEW"))
