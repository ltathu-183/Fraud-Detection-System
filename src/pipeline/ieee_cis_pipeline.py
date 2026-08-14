"""Canonical, methodologically isolated IEEE-CIS training/evaluation path."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.data.categorical_encoding import CategoricalEncoder
from src.evaluation.metrics import calculate_ranking_metrics, evaluate_policy, optimize_policy_thresholds
from src.features.temporal_features import TemporalFeatureEngineer
from src.models.lightgbm_model import LightGBMFraudModel
from src.pipeline.config import PipelineConfig

IDENTITY_COLUMNS = {"_internal_row_id", "_split_id", "TransactionID", "TransactionDT", "isFraud"}
TRANSACTION_PREFIXES = ("card", "addr", "dist", "P_emaildomain", "R_emaildomain", "C", "D", "M")
REQUIRED_TRANSACTION_COLUMNS = {"TransactionID", "TransactionDT", "TransactionAmt", "ProductCD", "isFraud"}


class FraudDetectionPipeline:
    """One supported path: train -> validation -> policy -> untouched test."""

    def __init__(self, config=None):
        self.config = config or PipelineConfig()
        self.encoder = CategoricalEncoder()
        self.feature_engineer = TemporalFeatureEngineer()
        self.model = LightGBMFraudModel()
        self.splits = {}
        self.feature_cols = None

    def load_data(self, transaction_path, identity_path=None):
        header = pd.read_csv(transaction_path, nrows=0).columns.tolist()
        usecols = [
            col for col in header
            if col in REQUIRED_TRANSACTION_COLUMNS or col.startswith(TRANSACTION_PREFIXES)
        ]
        # Float32/int32 reduce the full training data from an impractical all-column
        # allocation to a reproducible in-memory feature set.
        categorical = {"ProductCD", "card4", "card6", "P_emaildomain", "R_emaildomain"}
        dtypes = {
            col: "float32"
            for col in usecols
            if col not in categorical and not col.startswith("M") and col not in {"TransactionID", "TransactionDT", "isFraud"}
        }
        dtypes["TransactionID"] = "int32"
        dtypes["TransactionDT"] = "int32"
        dtypes["isFraud"] = "int8"
        tx = pd.read_csv(transaction_path, usecols=usecols, dtype=dtypes)
        if tx["TransactionID"].duplicated().any():
            raise ValueError("TransactionID must be unique")
        if identity_path:
            identity_header = pd.read_csv(identity_path, nrows=0).columns.tolist()
            identity_dtypes = {
                col: "float32"
                for col in identity_header
                if col.startswith("id_") and col[3:].isdigit() and int(col[3:]) <= 11
            }
            identity_dtypes["TransactionID"] = "int32"
            identity = pd.read_csv(identity_path, usecols=identity_header, dtype=identity_dtypes)
            if identity["TransactionID"].duplicated().any():
                raise ValueError("Identity data has duplicate TransactionID values")
            tx = tx.merge(identity, on="TransactionID", how="left", validate="one_to_one")
        return tx.sort_values(["TransactionDT", "TransactionID"], kind="mergesort").reset_index(drop=True)

    def temporal_split(self, df):
        data = df.copy()
        data["_internal_row_id"] = np.arange(len(data), dtype=np.int64)
        fractions = self.config.split
        ends = np.cumsum([fractions.train, fractions.validation, fractions.policy])
        boundaries = [int(len(data) * value) for value in ends]
        names = np.select(
            [data.index < boundaries[0], data.index < boundaries[1], data.index < boundaries[2]],
            ["train", "validation", "policy"], default="test"
        )
        data["_split_id"] = names
        for earlier, later in zip(("train", "validation", "policy"), ("validation", "policy", "test")):
            if data.loc[data._split_id == earlier, "TransactionDT"].max() > data.loc[data._split_id == later, "TransactionDT"].min():
                raise AssertionError("Chronological split boundary violated")
        self._expected_membership = data.set_index("_internal_row_id")["_split_id"].to_dict()
        self._expected_targets = data.set_index("_internal_row_id")[["TransactionID", "isFraud"]].to_dict("index")
        self.splits = {name: part.copy() for name, part in data.groupby("_split_id", sort=False)}

    def engineer_features(self):
        combined = pd.concat(self.splits.values(), ignore_index=True)
        cat_cols = self.splits["train"].select_dtypes(include=["object", "string"]).columns.difference(["_split_id"]).tolist()
        # Temporal features need raw entity IDs (device/email), while the model
        # still receives train-fitted encoded versions afterwards.
        featured = self.feature_engineer.engineer_all_features(combined)
        self.encoder.fit(self.splits["train"], cat_cols)
        encoded = self.encoder.transform(featured, cat_cols)
        self._assert_integrity(encoded)
        self.splits = {name: encoded.loc[encoded._split_id == name].copy() for name in ("train", "validation", "policy", "test")}

    def _assert_integrity(self, df):
        if len(df) != len(self._expected_membership) or not df._internal_row_id.is_unique:
            raise AssertionError("Row count/identity changed")
        for row in df[["_internal_row_id", "_split_id", "TransactionID", "isFraud"]].itertuples(index=False):
            row_id, split_id, transaction_id, target = row
            if self._expected_membership[row_id] != split_id:
                raise AssertionError("Split membership changed")
            expected = self._expected_targets[row_id]
            if expected["TransactionID"] != transaction_id or expected["isFraud"] != target:
                raise AssertionError("Target/TransactionID alignment changed")

    def _matrices(self):
        self.feature_cols = [c for c in self.splits["train"] if c not in IDENTITY_COLUMNS and pd.api.types.is_numeric_dtype(self.splits["train"][c])]
        if not self.feature_cols:
            raise ValueError("No numeric model features")
        return {name: (part[self.feature_cols], part.isFraud) for name, part in self.splits.items()}

    def _baseline_results(self, x_train, y_train, x_test, y_test):
        """Pre-specified, train-only reference models on the same schema/splits."""
        dummy = DummyClassifier(strategy="prior")
        dummy.fit(x_train, y_train)
        results = {
            "dummy_prior": calculate_ranking_metrics(y_test.to_numpy(), dummy.predict_proba(x_test)[:, 1])
        }
        # This is a logistic-loss linear baseline. Imputation/scaling is fit on
        # train only; no test/validation information enters it.
        linear = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            SGDClassifier(loss="log_loss", class_weight="balanced", alpha=1e-4,
                          max_iter=1000, tol=1e-3, random_state=self.config.random_state),
        )
        linear.fit(x_train, y_train)
        results["linear_logistic_sgd"] = calculate_ranking_metrics(
            y_test.to_numpy(), linear.predict_proba(x_test)[:, 1]
        )
        return results

    def run(self, transaction_path, identity_path=None, output_path="artifacts/corrected_results.json"):
        self.temporal_split(self.load_data(transaction_path, identity_path))
        self.engineer_features()
        matrices = self._matrices()
        x_train, y_train = matrices["train"]
        x_val, y_val = matrices["validation"]
        self.model.train(x_train, y_train, x_val, y_val, feature_names=self.feature_cols)
        x_policy, y_policy = matrices["policy"]
        policy_probs = self.model.predict_proba(x_policy)[:, 1]
        selected = optimize_policy_thresholds(y_policy.to_numpy(), policy_probs, self.config.policy)
        if not selected["feasible"]:
            raise RuntimeError("Policy constraints are INFEASIBLE; final test was not opened")
        # This is the policy-freeze point. Nothing from the test split has been
        # passed to training, threshold selection, or the selected policy.
        threshold_provenance = {
            "threshold_selection_split": "policy",
            "threshold_selection_rows": int(len(y_policy)),
            "final_evaluation_split": "test",
            "final_evaluation_rows": int(len(self.splits["test"])),
            "thresholds_frozen_before_test": True,
            "test_excluded_from_model_training": True,
            "test_excluded_from_threshold_selection": True,
        }
        x_test, y_test = matrices["test"]
        test_probs = self.model.predict_proba(x_test)[:, 1]
        temporal_cols = [col for col in self.feature_cols if col.startswith("card1_")]
        ablation_features = [col for col in self.feature_cols if col not in temporal_cols]
        ablation = LightGBMFraudModel()
        ablation.train(
            self.splits["train"][ablation_features], y_train,
            self.splits["validation"][ablation_features], y_val,
            feature_names=ablation_features,
        )
        ablation_probs = ablation.predict_proba(self.splits["test"][ablation_features])[:, 1]
        results = {
            "status": "CORRECTED / POST-AUDIT RESULTS",
            "ranking": calculate_ranking_metrics(y_test.to_numpy(), test_probs),
            "policy": evaluate_policy(y_test.to_numpy(), test_probs, selected["low_threshold"], selected["high_threshold"]),
            "threshold_selection": selected,
            "evaluation_provenance": threshold_provenance,
            "feature_schema": self.feature_cols,
            "scale_pos_weight": self.model.scale_pos_weight,
            "baselines": self._baseline_results(x_train, y_train, x_test, y_test),
            "temporal_feature_ablation": {
                "excluded_features": temporal_cols,
                "ranking": calculate_ranking_metrics(y_test.to_numpy(), ablation_probs),
            },
        }
        target = Path(output_path); target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(results, indent=2), encoding="utf-8")
        return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--transactions", default="data/raw/train_transaction.csv")
    parser.add_argument("--identity", default="data/raw/train_identity.csv")
    parser.add_argument("--output", default="artifacts/corrected_results.json")
    args = parser.parse_args()
    print(json.dumps(FraudDetectionPipeline().run(args.transactions, args.identity, args.output), indent=2))


if __name__ == "__main__":
    main()
