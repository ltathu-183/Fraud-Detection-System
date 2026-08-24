"""Small, deterministic monitoring primitives suitable for batch reports."""

import numpy as np
import pandas as pd

from src.evaluation.metrics import calculate_ranking_metrics, evaluate_policy


def population_stability_index(reference, current, bins=10):
    """PSI using reference-derived quantile bins; interpretation thresholds are contextual."""
    ref = pd.Series(reference).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    cur = pd.Series(current).replace([np.inf, -np.inf], np.nan).dropna().to_numpy()
    if len(ref) == 0 or len(cur) == 0:
        return None
    edges = np.unique(np.quantile(ref, np.linspace(0, 1, bins + 1)))
    if len(edges) < 2:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref_pct = np.histogram(ref, bins=edges)[0] / len(ref)
    cur_pct = np.histogram(cur, bins=edges)[0] / len(cur)
    ref_pct, cur_pct = np.clip(ref_pct, 1e-6, None), np.clip(cur_pct, 1e-6, None)
    return float(np.sum((cur_pct - ref_pct) * np.log(cur_pct / ref_pct)))


def build_monitoring_report(reference, current, reference_scores, current_scores,
                            y_current=None, thresholds=None, feature_columns=()):
    quality = {}
    drift = {}
    for col in feature_columns:
        if col not in reference or col not in current:
            continue
        quality[col] = {
            "reference_missing_rate": float(reference[col].isna().mean()),
            "current_missing_rate": float(current[col].isna().mean()),
        }
        if pd.api.types.is_numeric_dtype(reference[col]):
            drift[col] = {"psi": population_stability_index(reference[col], current[col])}
    report = {
        "data_quality": quality,
        "drift": drift,
        "score_drift": {"psi": population_stability_index(reference_scores, current_scores)},
        "interpretation": {
            "without_labels": "Drift and decision rates can trigger investigation but cannot establish performance degradation.",
            "with_labels": "Ranking and operational metrics should be reviewed after labels mature.",
            "thresholds": "No universal alert threshold is asserted; limits require portfolio-specific backtesting and governance approval.",
        },
    }
    if y_current is not None:
        report["performance"] = calculate_ranking_metrics(np.asarray(y_current), np.asarray(current_scores))
        if thresholds is not None:
            report["operational"] = evaluate_policy(y_current, current_scores, *thresholds)
    return report
