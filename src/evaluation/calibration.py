"""
Model Calibration for Fraud Detection
=====================================

Following IEEE-CIS blueprint: STEP 16

Calibration is critical in fraud systems.
Reliable probabilities are required for:
- threshold optimization
- expected loss estimation
- risk ranking
- business decision systems

Methods:
- Platt scaling (sigmoid): stable for smaller datasets
- Isotonic regression: more flexible but can overfit
"""

from __future__ import annotations

import logging
from typing import Dict

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.calibration import (
    calibration_curve,
    CalibratedClassifierCV
)
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Native LightGBM Booster Wrapper
# -----------------------------------------------------------------------------

class LightGBMBoosterWrapper(
    ClassifierMixin,
    BaseEstimator
):
    """
    sklearn-compatible wrapper
    for native LightGBM Booster.
    """

    def __init__(self, booster):

        self.booster = booster

        self.classes_ = np.array([0, 1])

        self._estimator_type = "classifier"

        # sklearn compatibility
        self.n_features_in_ = None

    def fit(self, X, y=None):

        self.n_features_in_ = X.shape[1]

        return self

    def predict_proba(self, X):

        X = np.asarray(X)

        raw_preds = self.booster.predict(X)

        raw_preds = np.asarray(
            raw_preds,
            dtype=np.float32
        )

        proba = np.column_stack([
            1.0 - raw_preds,
            raw_preds
        ])

        return proba

    def predict(self, X):

        proba = self.predict_proba(X)

        return (
            proba[:, 1] >= 0.5
        ).astype(np.int32)

    # NEW sklearn API compatibility
    def __sklearn_tags__(self):

        tags = super().__sklearn_tags__()

        tags.estimator_type = "classifier"

        return tags


# -----------------------------------------------------------------------------
# Main Calibration Class
# -----------------------------------------------------------------------------

class ModelCalibrator:
    """
    Calibrate model predictions for reliable probabilities.
    """

    def __init__(
        self,
        method: str = "isotonic"
    ):
        """
        Parameters
        ----------
        method : str
            'platt'     -> sigmoid calibration
            'sigmoid'   -> sigmoid calibration
            'isotonic'  -> isotonic regression
        """

        if method == "platt":
            method = "sigmoid"

        valid_methods = [
            "sigmoid",
            "isotonic"
        ]

        if method not in valid_methods:
            raise ValueError(
                f"Invalid method: {method}"
            )

        self.method = method
        self.calibrator = None
        self.model = None

    # -------------------------------------------------------------------------
    # Main calibration
    # -------------------------------------------------------------------------

    def calibrate_model(
        self,
        model,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
        X_eval: np.ndarray = None,
        y_eval: np.ndarray = None
    ):
        """
        Calibrate predictions using a dedicated calibration set.

        IMPORTANT:
        Calibration set should be TEMPORALLY AFTER training data.

        Example:
            train -> calibration -> validation -> final test

        NOT:
            random split
        """

        logger.info("\n" + "=" * 70)
        logger.info(
            f"CALIBRATING MODEL "
            f"({self.method.upper()})"
        )
        logger.info("=" * 70)

        # ---------------------------------------------------------------------
        # Validation
        # ---------------------------------------------------------------------

        y_cal = np.asarray(y_cal)

        unique_labels = np.unique(y_cal)

        if not np.array_equal(
            unique_labels,
            np.array([0, 1])
        ):
            raise ValueError(
                "Calibration labels must be binary {0,1}"
            )

        if len(X_cal) != len(y_cal):
            raise ValueError(
                "X_cal and y_cal size mismatch"
            )

        self.model = model

        # ---------------------------------------------------------------------
        # Get predictions from model
        # (handle both native LightGBM and sklearn estimators)
        # ---------------------------------------------------------------------

        import lightgbm as lgb

        if isinstance(model, lgb.Booster):
            X_cal_array = np.asarray(X_cal)
            raw_preds = model.predict(X_cal_array)
            y_cal_proba = np.asarray(
                raw_preds,
                dtype=np.float64
            )

        else:
            # sklearn estimator
            y_cal_proba = model.predict_proba(X_cal)[:, 1]

        y_cal_proba = np.asarray(
            y_cal_proba,
            dtype=np.float64
        )

        y_cal_proba = np.clip(
            y_cal_proba,
            1e-15,
            1.0 - 1e-15
        )

        # ---------------------------------------------------------------------
        # Fit calibration method
        # ---------------------------------------------------------------------

        if self.method == "isotonic":
            self.calibrator = IsotonicRegression(
                out_of_bounds='clip'
            )
            self.calibrator.fit(
                y_cal_proba,
                y_cal
            )

        elif self.method == "sigmoid":
            self._fit_sigmoid(y_cal_proba, y_cal)

        logger.info(
            f"Calibration method: {self.method}"
        )

        logger.info(
            f"Calibration samples: "
            f"{len(X_cal):,}"
        )

        if X_eval is not None and y_eval is not None:
            eval_raw_proba = self._predict_raw_proba(X_eval)
            eval_y = np.asarray(y_eval)
            calibrated_proba = self.predict_proba(eval_raw_proba)[:, 1]
            evaluation_name = "validation"
        else:
            eval_raw_proba = y_cal_proba
            eval_y = y_cal
            calibrated_proba = self.predict_proba(y_cal_proba)[:, 1]
            evaluation_name = "calibration"

        raw_brier = float(
            brier_score_loss(
                eval_y,
                eval_raw_proba
            )
        )

        raw_metrics = self.evaluate_calibration(
            eval_y,
            eval_raw_proba
        )

        calibrated_brier = float(
            brier_score_loss(
                eval_y,
                calibrated_proba
            )
        )

        calibrated_metrics = self.evaluate_calibration(
            eval_y,
            calibrated_proba
        )

        logger.info(
            f"Raw {evaluation_name} Brier score: "
            f"{raw_brier:.6f}"
        )
        logger.info(
            f"Raw {evaluation_name} ECE: "
            f"{raw_metrics['ece']:.6f}"
        )
        logger.info(
            f"Calibrated {evaluation_name} Brier score: "
            f"{calibrated_brier:.6f}"
        )
        logger.info(
            f"Calibrated {evaluation_name} ECE: "
            f"{calibrated_metrics['ece']:.6f}"
        )

        raw_percentiles = np.percentile(
            eval_raw_proba,
            [1, 50, 99]
        )
        calibrated_percentiles = np.percentile(
            calibrated_proba,
            [1, 50, 99]
        )

        logger.info(
            f"Raw probs percentile [1,50,99]: "
            f"{raw_percentiles}"
        )
        logger.info(
            f"Calibrated probs percentile [1,50,99]: "
            f"{calibrated_percentiles}"
        )

        if self.method == "isotonic":
            collapse_rate = np.mean(
                (calibrated_proba <= 1e-8) |
                (calibrated_proba >= 1.0 - 1e-8)
            )
            if collapse_rate > 0.95:
                logger.warning(
                    "Isotonic calibration returned extreme probability values "
                    "for more than 95% of examples. Falling back to sigmoid."
                )
                self.method = "sigmoid"
                self._fit_sigmoid(y_cal_proba, y_cal)
                calibrated_proba = self.predict_proba(eval_raw_proba)[:, 1]
                calibrated_brier = float(
                    brier_score_loss(
                        eval_y,
                        calibrated_proba
                    )
                )
                calibrated_metrics = self.evaluate_calibration(
                    eval_y,
                    calibrated_proba
                )
                logger.info(
                    "Fallback sigmoid calibrated ECE: "
                    f"{calibrated_metrics['ece']:.6f}"
                )

        logger.info("=" * 70)

        return self.calibrator

    # -------------------------------------------------------------------------
    # Predict calibrated probabilities
    # -------------------------------------------------------------------------

    def predict_proba(
        self,
        X: np.ndarray
    ) -> np.ndarray:
        """
        Get calibrated probabilities.

        Accepts either:
        - raw probability scores for isotonic/logistic transforms
        - raw feature matrix if self.model is available
        """

        if self.calibrator is None:
            raise ValueError(
                "Model not calibrated yet. "
                "Call calibrate_model() first."
            )

        X_arr = np.asarray(X)
        is_raw_proba = (
            X_arr.ndim == 1 or
            (X_arr.ndim == 2 and X_arr.shape[1] == 1)
        )

        if self.method == "isotonic":
            if not is_raw_proba:
                if self.model is None:
                    raise ValueError(
                        "Cannot transform feature matrix without a fitted model. "
                        "Provide raw probabilities or fit the calibrator model first."
                    )

                X_arr = self._predict_raw_proba(X_arr)

            X_arr = np.asarray(X_arr).ravel()
            if not np.all(np.isfinite(X_arr)):
                raise ValueError("Input contains NaN or infinite values.")

            calibrated = self.calibrator.predict(X_arr)
            calibrated = np.asarray(calibrated, dtype=np.float64)
            calibrated = np.clip(calibrated, 1e-15, 1.0 - 1e-15)
            return np.vstack([1.0 - calibrated, calibrated]).T

        if self.method == "sigmoid":
            if not is_raw_proba:
                if self.model is None:
                    raise ValueError(
                        "Cannot transform feature matrix without a fitted model. "
                        "Provide raw probabilities or fit the calibrator model first."
                    )

                X_arr = self._predict_raw_proba(X_arr)

            X_arr = np.asarray(X_arr).ravel()
            if not np.all(np.isfinite(X_arr)):
                raise ValueError("Input contains NaN or infinite values.")

            logit_proba = np.log(X_arr / (1.0 - X_arr))
            calibrated = self.calibrator.predict_proba(
                logit_proba.reshape(-1, 1)
            )[:, 1]
            calibrated = np.asarray(calibrated, dtype=np.float64)
            calibrated = np.clip(calibrated, 1e-15, 1.0 - 1e-15)
            return np.vstack([1.0 - calibrated, calibrated]).T

        raise ValueError(
            f"Unsupported calibration method: {self.method}"
        )

    def _predict_raw_proba(
        self,
        X: np.ndarray
    ) -> np.ndarray:
        """
        Convert raw feature inputs into probability scores using the saved model.
        """
        if self.model is None:
            raise ValueError(
                "No base model is available to compute raw probabilities."
            )

        import lightgbm as lgb

        if isinstance(self.model, lgb.Booster):
            raw_preds = self.model.predict(np.asarray(X))
        else:
            raw_preds = self.model.predict_proba(X)[:, 1]

        raw_preds = np.asarray(raw_preds, dtype=np.float64)
        raw_preds = np.clip(raw_preds, 1e-15, 1.0 - 1e-15)

        if not np.all(np.isfinite(raw_preds)):
            raise ValueError("Base model produced NaN or infinite probabilities.")

        return raw_preds

    def _fit_sigmoid(
        self,
        y_proba: np.ndarray,
        y_true: np.ndarray
    ) -> None:
        """
        Fit a sigmoid calibration model on raw probability scores.
        """
        from sklearn.linear_model import LogisticRegression

        logit_proba = np.log(
            y_proba / (1.0 - y_proba)
        )

        lr = LogisticRegression(
            penalty=None,
            solver='lbfgs',
            max_iter=1000
        )
        lr.fit(
            logit_proba.reshape(-1, 1),
            y_true
        )
        self.calibrator = lr

    def predict(
        self,
        X: np.ndarray
    ) -> np.ndarray:
        """
        Predict the positive-class probability vector for fallback usage.
        """

        proba = self.predict_proba(X)
        return proba[:, 1]

    # -------------------------------------------------------------------------
    # Calibration evaluation
    # -------------------------------------------------------------------------

    def evaluate_calibration(
        self,
        y_true: np.ndarray,
        y_pred_proba: np.ndarray,
        n_bins: int = 10
    ) -> Dict[str, float]:
        """
        Evaluate calibration quality.

        Metrics:
        - ECE: Expected Calibration Error
        - MCE: Maximum Calibration Error

        Lower is better.
        """

        y_true = np.asarray(y_true)

        y_pred_proba = np.asarray(
            y_pred_proba
        )

        y_pred_proba = np.clip(
            y_pred_proba,
            1e-8,
            1 - 1e-8
        )

        # ---------------------------------------------------------------------
        # Calibration curve
        # ---------------------------------------------------------------------

        prob_true, prob_pred = (
            calibration_curve(
                y_true,
                y_pred_proba,
                n_bins=n_bins,
                strategy="quantile"
            )
        )

        # ---------------------------------------------------------------------
        # Weighted Expected Calibration Error
        # ---------------------------------------------------------------------

        bin_edges = np.quantile(
            y_pred_proba,
            np.linspace(
                0,
                1,
                n_bins + 1
            )
        )

        ece = 0.0
        mce = 0.0

        for i in range(n_bins):

            lower = bin_edges[i]
            upper = bin_edges[i + 1]

            if i == n_bins - 1:
                mask = (
                    (y_pred_proba >= lower) &
                    (y_pred_proba <= upper)
                )
            else:
                mask = (
                    (y_pred_proba >= lower) &
                    (y_pred_proba < upper)
                )

            if np.sum(mask) == 0:
                continue

            acc = y_true[mask].mean()

            conf = y_pred_proba[mask].mean()

            error = abs(acc - conf)

            weight = np.mean(mask)

            ece += weight * error

            mce = max(mce, error)

        metrics = {
            "ece": float(ece),
            "mce": float(mce),
            "prob_true": prob_true.tolist(),
            "prob_pred": prob_pred.tolist(),
        }

        logger.info("\nCalibration Metrics:")

        logger.info(
            f"  Expected Calibration Error (ECE): "
            f"{ece:.6f}"
        )

        logger.info(
            f"  Maximum Calibration Error (MCE): "
            f"{mce:.6f}"
        )

        logger.info(
            "  Interpretation: lower is better"
        )

        return metrics


# -----------------------------------------------------------------------------
# Compare calibration methods
# -----------------------------------------------------------------------------

def compare_calibration_methods(
    model,
    X_cal: np.ndarray,
    y_cal: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, Dict]:
    """
    Compare:
    - Platt scaling
    - Isotonic regression
    """

    results = {}

    for method in [
        "platt",
        "isotonic"
    ]:

        logger.info(
            f"\nTesting "
            f"{method.upper()} calibration..."
        )

        calibrator = ModelCalibrator(
            method=method
        )

        calibrator.calibrate_model(
            model=model,
            X_cal=X_cal,
            y_cal=y_cal
        )

        y_pred_proba = (
            calibrator
            .predict_proba(X_test)[:, 1]
        )

        metrics = (
            calibrator
            .evaluate_calibration(
                y_test,
                y_pred_proba
            )
        )

        results[method] = metrics

    logger.info("\n" + "=" * 70)
    logger.info("CALIBRATION COMPARISON")
    logger.info("=" * 70)

    logger.info(
        f"Platt ECE: "
        f"{results['platt']['ece']:.6f}"
    )

    logger.info(
        f"Isotonic ECE: "
        f"{results['isotonic']['ece']:.6f}"
    )

    best_method = (
        "isotonic"
        if (
            results["isotonic"]["ece"]
            <
            results["platt"]["ece"]
        )
        else "platt"
    )

    logger.info(
        f"Recommended: {best_method}"
    )

    logger.info("=" * 70)

    return results