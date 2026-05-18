"""
Tier 2: LightGBM Model for Fraud Detection
==========================================

Tree-based fraud detection model.

Design goals:
- low latency inference
- temporal-safe training pipeline
- calibration-ready probabilities
- memory-efficient processing
"""

from __future__ import annotations

import warnings
from typing import Dict, List, Tuple, Optional

import lightgbm as lgb
import numpy as np
import pandas as pd

warnings.filterwarnings(
    "ignore",
    category=UserWarning
)


class LightGBMFraudModel:
    """
    LightGBM model for fraud detection.

    This tier handles transactions that:
    - pass rule-based filtering
    - still require ML scoring
    """

    # -------------------------------------------------------------------------
    # Initialization
    # -------------------------------------------------------------------------

    def __init__(
        self,
        params: Dict = None,
        num_rounds: int = 1000,
        early_stopping_rounds: int = 50
    ):

        self.model = None

        self.calibrator = None

        self.params = (
            params or
            self._get_default_params()
        )

        self.num_rounds = num_rounds

        self.early_stopping_rounds = (
            early_stopping_rounds
        )

        self.feature_names = None

        self.scale_pos_weight = None

    # -------------------------------------------------------------------------
    # Default parameters
    # -------------------------------------------------------------------------

    def _get_default_params(self) -> Dict:
        """
        Default LightGBM parameters optimized
        for fraud detection.
        """

        return {
            "objective": "binary",
            "metric": "auc",
            "boosting_type": "gbdt",

            "learning_rate": 0.03,

            "num_leaves": 31,
            "max_depth": 6,

            "feature_fraction": 0.7,

            "bagging_fraction": 0.7,
            "bagging_freq": 1,

            "min_child_samples": 100,
            "min_data_in_leaf": 100,

            "reg_alpha": 1.0,
            "reg_lambda": 2.0,
            "lambda_l1": 1.0,
            "lambda_l2": 5.0,

            # -----------------------------------------------------------------
            # Imbalance handling
            # -----------------------------------------------------------------

            "scale_pos_weight": 27.5,

            # -----------------------------------------------------------------
            # Reproducibility
            # -----------------------------------------------------------------

            "random_state": 42,

            "feature_fraction_seed": 42,
            "bagging_seed": 42,
            "data_random_seed": 42,

            "deterministic": True,

            # -----------------------------------------------------------------
            # Performance
            # -----------------------------------------------------------------

            "force_row_wise": True,

            "verbose": -1
        }

    # -------------------------------------------------------------------------
    # Class imbalance handling
    # -------------------------------------------------------------------------

    def calculate_scale_pos_weight(
        self,
        y_train: np.ndarray
    ) -> float:
        """
        Calculate imbalance ratio.
        """

        y_train = np.asarray(y_train)

        neg_count = np.sum(y_train == 0)

        pos_count = np.sum(y_train == 1)

        if pos_count == 0:
            self.scale_pos_weight = 1.0
        else:
            self.scale_pos_weight = (
                neg_count / pos_count
            )

        print("\nClass imbalance:")
        print(
            f"  Negative: {neg_count:,}"
        )

        print(
            f"  Positive: {pos_count:,}"
        )

        print(
            f"  scale_pos_weight: "
            f"{self.scale_pos_weight:.2f}"
        )

        return self.scale_pos_weight

    # -------------------------------------------------------------------------
    # Data preparation
    # -------------------------------------------------------------------------

    def prepare_data(
        self,
        train_data: pd.DataFrame,
        val_data: pd.DataFrame,
        cal_data: pd.DataFrame,
        test_data: pd.DataFrame,
        label_col: str = "isFraud",
        drop_cols: List[str] = None
    ) -> Tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        List[str]
    ]:
        """
        Prepare NumPy arrays for training with calibration set.
        """

        if drop_cols is None:
            drop_cols = [
                "TransactionID",
                "TransactionDT",
                "DeviceInfo",
                "addr1",
                "card1"
            ]

        if label_col in drop_cols:
            drop_cols.remove(label_col)

        # Select only numeric columns (exclude object dtype)
        feature_cols = [
            col
            for col in train_data.columns
            if col not in (drop_cols + [label_col])
            and train_data[col].dtype in [
                np.float32,
                np.float64,
                np.int8,
                np.int16,
                np.int32,
                np.int64,
                np.uint8,
                np.uint16,
                np.uint32,
                np.uint64
            ]
        ]

        # Cache feature names
        self.feature_names = feature_cols

        # Memory-efficient NumPy conversion
        X_train = (
            train_data[feature_cols]
            .to_numpy(
                dtype=np.float32,
                copy=False
            )
        )

        y_train = (
            train_data[label_col]
            .to_numpy(
                dtype=np.int8,
                copy=False
            )
        )

        X_val = (
            val_data[feature_cols]
            .to_numpy(
                dtype=np.float32,
                copy=False
            )
        )

        y_val = (
            val_data[label_col]
            .to_numpy(
                dtype=np.int8,
                copy=False
            )
        )

        X_cal = (
            cal_data[feature_cols]
            .to_numpy(
                dtype=np.float32,
                copy=False
            )
        )

        y_cal = (
            cal_data[label_col]
            .to_numpy(
                dtype=np.int8,
                copy=False
            )
        )

        X_test = (
            test_data[feature_cols]
            .to_numpy(
                dtype=np.float32,
                copy=False
            )
        )

        y_test = (
            test_data[label_col]
            .to_numpy(
                dtype=np.int8,
                copy=False
            )
        )

        print("\nDataset summary:")
        print(f"  Train samples: {len(X_train):,}")
        print(f"  Validation samples: {len(X_val):,}")
        print(f"  Calibration samples: {len(X_cal):,}")
        print(f"  Test samples: {len(X_test):,}")
        print(f"  Features: {len(feature_cols):,}")

        return (
            X_train,
            y_train,
            X_val,
            y_val,
            X_cal,
            y_cal,
            X_test,
            y_test,
            feature_cols
        )

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray,
        y_val: np.ndarray,
        feature_names: List[str] = None
    ) -> lgb.Booster:
        """
        Train LightGBM model.
        """

        if feature_names is not None:
            self.feature_names = feature_names

        if self.feature_names is None:
            raise ValueError(
                "feature_names missing"
            )

        # ---------------------------------------------------------------------
        # Calculate imbalance ratio
        # ---------------------------------------------------------------------

        if self.scale_pos_weight is None:
            self.calculate_scale_pos_weight(
                y_train
            )

        params = self.params.copy()

        params["scale_pos_weight"] = (
            self.scale_pos_weight
        )

        # ---------------------------------------------------------------------
        # LightGBM datasets
        # ---------------------------------------------------------------------

        train_dataset = lgb.Dataset(
            X_train,
            label=y_train,
            feature_name=self.feature_names,
            free_raw_data=True
        )

        val_dataset = lgb.Dataset(
            X_val,
            label=y_val,
            feature_name=self.feature_names,
            reference=train_dataset,
            free_raw_data=True
        )

        print("\n" + "=" * 60)
        print("TRAINING LIGHTGBM MODEL")
        print("=" * 60)

        # ---------------------------------------------------------------------
        # Training
        # ---------------------------------------------------------------------

        self.model = lgb.train(
            params=params,

            train_set=train_dataset,

            num_boost_round=self.num_rounds,

            valid_sets=[
                train_dataset,
                val_dataset
            ],

            valid_names=[
                "train",
                "valid"
            ],

            callbacks=[

                lgb.early_stopping(
                    stopping_rounds=(
                        self.early_stopping_rounds
                    ),
                    verbose=True
                ),

                lgb.log_evaluation(
                    period=100
                )
            ]
        )

        print("\nTraining complete.")

        print(
            f"Best iteration: "
            f"{self.model.best_iteration}"
        )

        valid_auc = (
            self.model.best_score
            ["valid"]["auc"]
        )

        print(
            f"Best validation AUC: "
            f"{valid_auc:.6f}"
        )

        return self.model

    # -------------------------------------------------------------------------
    # Raw probability prediction
    # -------------------------------------------------------------------------

    def predict_raw_proba(
        self,
        X: np.ndarray
    ) -> np.ndarray:
        """
        Raw model probabilities without calibration.
        """

        if self.model is None:
            raise ValueError(
                "Model not trained."
            )

        raw_preds = self.model.predict(
            X,
            num_iteration=(
                self.model.best_iteration
            )
        )

        raw_preds = np.asarray(raw_preds)

        raw_preds = np.clip(
            raw_preds,
            1e-8,
            1 - 1e-8
        )

        return raw_preds

    # -------------------------------------------------------------------------
    # Calibrated probability prediction
    # -------------------------------------------------------------------------

    def predict_proba(
        self,
        X: np.ndarray
    ) -> np.ndarray:
        """
        Standard sklearn-compatible probability output:

            [[p0, p1],
             [p0, p1],
             ...]
        """

        if self.model is None:
            raise ValueError(
                "Model not trained."
            )

        # Get raw predictions from model
        raw_preds = self.predict_raw_proba(X)

        # Apply calibration if available
        if self.calibrator is not None:
            from sklearn.isotonic import IsotonicRegression
            from sklearn.linear_model import LogisticRegression

            if isinstance(
                self.calibrator,
                IsotonicRegression
            ):
                # IsotonicRegression.predict()
                # transforms probabilities
                calibrated_preds = (
                    self.calibrator.predict(
                        raw_preds
                    )
                )

            elif isinstance(
                self.calibrator,
                LogisticRegression
            ):
                # Sigmoid calibration via
                # LogisticRegression.predict_proba()
                logit_raw = np.log(
                    raw_preds / (1.0 - raw_preds)
                )

                calibrated_preds = (
                    self.calibrator
                    .predict_proba(
                        logit_raw.reshape(-1, 1)
                    )[:, 1]
                )

            else:
                # Fallback: handle our ModelCalibrator wrapper
                from src.evaluation.calibration import ModelCalibrator

                if isinstance(self.calibrator, ModelCalibrator):
                    calibrated_preds = self.calibrator.predict(raw_preds)
                else:
                    calibrated_preds = self.calibrator.predict(raw_preds)

            return np.vstack([
                1.0 - calibrated_preds,
                calibrated_preds
            ]).T

        # No calibration: use raw predictions
        return np.vstack([
            1.0 - raw_preds,
            raw_preds
        ]).T

    # -------------------------------------------------------------------------
    # Binary prediction
    # -------------------------------------------------------------------------

    def predict(
        self,
        X: np.ndarray,
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        Binary predictions.

        NOTE:
        Fraud systems often use:
            threshold << 0.5
        """

        proba = (
            self.predict_proba(X)[:, 1]
        )

        return (
            proba >= threshold
        ).astype(np.int32)

    # -------------------------------------------------------------------------
    # Set calibrator
    # -------------------------------------------------------------------------

    def set_calibrator(
        self,
        calibrator
    ) -> None:
        """
        Attach probability calibrator.
        """

        self.calibrator = calibrator

        print(
            "Calibration model attached."
        )

    # -------------------------------------------------------------------------
    # Feature importance
    # -------------------------------------------------------------------------

    def get_feature_importance(
        self,
        importance_type: str = "gain",
        top_n: int = 20
    ) -> pd.DataFrame:
        """
        Feature importance ranking.
        """

        if self.model is None:
            raise ValueError(
                "Model not trained."
            )

        importance = (
            self.model.feature_importance(
                importance_type=(
                    importance_type
                )
            )
        )

        importance_df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importance
        })

        importance_df = (
            importance_df
            .sort_values(
                "importance",
                ascending=False
            )
        )

        print("\n" + "=" * 60)

        print(
            f"TOP {top_n} FEATURES "
            f"({importance_type.upper()})"
        )

        print("=" * 60)

        print(
            importance_df
            .head(top_n)
            .to_string(index=False)
        )

        return importance_df

    # -------------------------------------------------------------------------
    # Save model
    # -------------------------------------------------------------------------

    def save_model(
        self,
        filepath: str
    ) -> None:
        """
        Save LightGBM model.
        """

        if self.model is None:
            raise ValueError(
                "Model not trained."
            )

        self.model.save_model(filepath)

        print(
            f"Model saved to: {filepath}"
        )

    # -------------------------------------------------------------------------
    # Load model
    # -------------------------------------------------------------------------

    def load_model(
        self,
        filepath: str
    ) -> lgb.Booster:
        """
        Load LightGBM model.
        """

        self.model = lgb.Booster(
            model_file=filepath
        )

        print(
            f"Model loaded from: {filepath}"
        )

        return self.model


# -----------------------------------------------------------------------------
# Factory function
# -----------------------------------------------------------------------------

def create_lightgbm_model(
    params: Optional[Dict] = None
) -> LightGBMFraudModel:
    """
    Factory helper.
    """

    return LightGBMFraudModel(
        params=params
    )