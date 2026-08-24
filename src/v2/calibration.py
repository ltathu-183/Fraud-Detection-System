"""Historical-only score calibration diagnostics."""

import numpy as np
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, average_precision_score


def expected_calibration_error(y_true, probabilities, bins=10):
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    edges = np.linspace(0, 1, bins + 1)
    bin_ids = np.minimum(np.digitize(p, edges[1:-1], right=True), bins - 1)
    error = 0.0
    curve = []
    for index in range(bins):
        mask = bin_ids == index
        if not mask.any():
            continue
        mean_score, observed = float(p[mask].mean()), float(y[mask].mean())
        error += mask.mean() * abs(mean_score - observed)
        curve.append({"bin": index, "rows": int(mask.sum()), "mean_score": mean_score, "observed_rate": observed})
    return float(error), curve


def calibration_metrics(y_true, probabilities):
    p = np.clip(np.asarray(probabilities, dtype=float), 1e-8, 1 - 1e-8)
    ece, curve = expected_calibration_error(y_true, p)
    return {"brier_score": float(brier_score_loss(y_true, p)),
            "log_loss": float(log_loss(y_true, p, labels=[0, 1])), "ece_10_bin": ece,
            "roc_auc": float(roc_auc_score(y_true, p)),
            "pr_auc": float(average_precision_score(y_true, p)), "calibration_curve": curve}


def fit_calibrators(y_calibration, raw_calibration_scores):
    y = np.asarray(y_calibration, dtype=int)
    scores = np.asarray(raw_calibration_scores, dtype=float).reshape(-1, 1)
    platt = LogisticRegression(random_state=42, max_iter=1000).fit(scores, y)
    isotonic = IsotonicRegression(out_of_bounds="clip").fit(scores.ravel(), y)
    return {"platt": platt, "isotonic": isotonic}


def evaluate_calibration(y_evaluation, raw_evaluation_scores, calibrators):
    raw = np.asarray(raw_evaluation_scores, dtype=float)
    variants = {"uncalibrated": raw,
                "platt": calibrators["platt"].predict_proba(raw.reshape(-1, 1))[:, 1],
                "isotonic": calibrators["isotonic"].predict(raw)}
    return {name: calibration_metrics(y_evaluation, values) for name, values in variants.items()}
