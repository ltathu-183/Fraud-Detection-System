"""Exploratory V2 walk-forward model, calibration, and policy experiment."""

import gc
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from src.data.categorical_encoding import CategoricalEncoder
from src.evaluation.metrics import calculate_ranking_metrics, evaluate_policy
from src.evaluation.rolling_robustness import construct_rolling_windows, validate_windows, summarize
from src.features.temporal_features import TemporalFeatureEngineer
from src.models.lightgbm_model import LightGBMFraudModel
from src.pipeline.ieee_cis_pipeline import FraudDetectionPipeline, IDENTITY_COLUMNS
from src.v2.adaptive_policy import AdaptivePolicyConfig, select_adaptive_policy
from src.v2.calibration import evaluate_calibration, fit_calibrators


FINAL_TEST_START = 546249
RECENT_HISTORY_ROWS = 177160
RECENCY_HALF_LIFE_ROWS = 88580
SYSTEM_NAMES = ("v1_static", "adaptive_policy_only", "recent_model_only", "full_v2_adaptive")
CANONICAL_REVIEW_LIMIT = 0.20


def engineer_temporal_memory_safe(frame):
    """Compute the unchanged canonical temporal features on their minimal inputs."""
    required = ["_internal_row_id", "TransactionDT", "TransactionAmt", "card1",
                "DeviceInfo", "P_emaildomain", "addr1"]
    slim = frame[[column for column in required if column in frame]].copy()
    engineered = TemporalFeatureEngineer().engineer_all_features(slim)
    new_columns = [column for column in engineered if column not in slim]
    output = frame.join(engineered[new_columns])
    if not output["_internal_row_id"].equals(frame["_internal_row_id"]):
        raise AssertionError("Memory-safe temporal engineering changed row identity")
    return output


def _encode(featured, train_start, train_end, evaluation_end):
    train = featured.iloc[train_start:train_end]
    categorical = train.select_dtypes(include=["object", "string"]).columns.difference(["_split_id"]).tolist()
    encoder = CategoricalEncoder(); encoder.fit(train, categorical)
    encoded = encoder.transform(featured.iloc[:evaluation_end].copy(), categorical)
    features = [col for col in encoded if col not in IDENTITY_COLUMNS and pd.api.types.is_numeric_dtype(encoded[col])]
    return encoded, features


def _train_model(encoded, features, train_bounds, validation_bounds, weighted=False):
    train = encoded.iloc[slice(*train_bounds)]
    validation = encoded.iloc[slice(*validation_bounds)]
    if not weighted:
        model = LightGBMFraudModel()
        model.train(train[features], train.isFraud, validation[features], validation.isFraud, feature_names=features)
        return model
    base = LightGBMFraudModel()
    params = base.params.copy()
    params["scale_pos_weight"] = base.calculate_scale_pos_weight(train.isFraud.to_numpy())
    age = train_bounds[1] - 1 - np.arange(train_bounds[0], train_bounds[1])
    weights = np.power(0.5, age / RECENCY_HALF_LIFE_ROWS)
    train_set = lgb.Dataset(train[features], label=train.isFraud, weight=weights, feature_name=features)
    validation_set = lgb.Dataset(validation[features], label=validation.isFraud,
                                 feature_name=features, reference=train_set)
    booster = lgb.train(params, train_set, num_boost_round=base.num_rounds,
                        valid_sets=[validation_set], valid_names=["valid"],
                        callbacks=[lgb.early_stopping(base.early_stopping_rounds, verbose=False),
                                   lgb.log_evaluation(period=0)])
    base.model, base.feature_names = booster, features
    return base


def _summarize_system(entries, policy_config):
    metrics = [entry["operational_metrics"] for entry in entries]
    ranking = [entry["ranking"] for entry in entries]
    constraints = [metric["fraud_triage_coverage"] >= .80
                   and metric["overall_review_rate"] <= CANONICAL_REVIEW_LIMIT
                   and metric["legitimate_auto_decline_rate"] <= policy_config.max_legitimate_decline_rate
                   for metric in metrics]
    return {"mean_pr_auc": float(np.mean([item["pr_auc"] for item in ranking])),
            "worst_pr_auc": float(min(item["pr_auc"] for item in ranking)),
            "mean_roc_auc": float(np.mean([item["roc_auc"] for item in ranking])),
            "mean_triage": float(np.mean([item["fraud_triage_coverage"] for item in metrics])),
            "worst_triage": float(min(item["fraud_triage_coverage"] for item in metrics)),
            "mean_review": float(np.mean([item["overall_review_rate"] for item in metrics])),
            "max_legitimate_decline": float(max(item["legitimate_auto_decline_rate"] for item in metrics)),
            "constraints_satisfied": int(sum(constraints)), "windows": len(entries),
            "constraint_definition": {"min_triage": .80, "max_review": CANONICAL_REVIEW_LIMIT,
                                      "max_legitimate_decline": policy_config.max_legitimate_decline_rate}}


def run_v2(frozen_manifest="artifacts/evaluation/canonical_final_frozen.json",
           v1_rolling="artifacts/evaluation/rolling_temporal_robustness.json",
           output="artifacts/v2/walk_forward_results.json",
           ablation_output="artifacts/v2/system_ablation.json"):
    frozen = json.loads(Path(frozen_manifest).read_text(encoding="utf-8"))
    if frozen["split_boundaries"]["test"]["start_row_inclusive"] != FINAL_TEST_START:
        raise AssertionError("Frozen V1 final-test boundary changed")
    raw = FraudDetectionPipeline().load_data("data/raw/train_transaction.csv", "data/raw/train_identity.csv")
    pre_final = raw.iloc[:FINAL_TEST_START].copy().reset_index(drop=True)
    del raw
    pre_final["_internal_row_id"] = np.arange(len(pre_final), dtype=np.int64)
    pre_final["_split_id"] = "v2_exploratory_pre_final"
    windows = construct_rolling_windows(len(pre_final)); validate_windows(windows, FINAL_TEST_START)
    featured = engineer_temporal_memory_safe(pre_final)
    v1 = json.loads(Path(v1_rolling).read_text(encoding="utf-8"))
    static_thresholds = {item["window"]: (item["canonical_policy"]["low_threshold"],
                                           item["canonical_policy"]["high_threshold"]) for item in v1["windows"]}
    policy_config = AdaptivePolicyConfig()
    results = []
    system_entries = {name: [] for name in SYSTEM_NAMES}
    for bounds in windows:
        window_id = bounds["window"]
        model_specs = {
            "expanding": (bounds["train"], False),
            "recent_177160": ((max(0, bounds["train"][1] - RECENT_HISTORY_ROWS), bounds["train"][1]), False),
            "recency_weighted": (bounds["train"], True),
        }
        window_models = {}
        for name, (train_bounds, weighted) in model_specs.items():
            encoded, features = _encode(featured, train_bounds[0], train_bounds[1], bounds["evaluation"][1])
            model = _train_model(encoded, features, train_bounds, bounds["validation"], weighted=weighted)
            policy = encoded.iloc[slice(*bounds["policy"])]
            evaluation = encoded.iloc[slice(*bounds["evaluation"])]
            policy_scores = model.predict_proba(policy[features])[:, 1]
            adaptive = select_adaptive_policy(policy.isFraud.to_numpy(), policy_scores, policy_config)
            calibrators = fit_calibrators(policy.isFraud.to_numpy(), policy_scores)
            # Thresholds and calibrators are frozen before pseudo-future scores.
            evaluation_scores = model.predict_proba(evaluation[features])[:, 1]
            y_eval = evaluation.isFraud.to_numpy()
            ranking = calculate_ranking_metrics(y_eval, evaluation_scores)
            adaptive_metrics = evaluate_policy(y_eval, evaluation_scores,
                                               adaptive["low_threshold"], adaptive["high_threshold"])
            calibration = evaluate_calibration(y_eval, evaluation_scores, calibrators)
            window_models[name] = {"training_bounds": {"start": train_bounds[0], "end_exclusive": train_bounds[1]},
                                   "weighted": weighted, "ranking": ranking,
                                   "adaptive_policy": adaptive, "adaptive_evaluation_metrics": adaptive_metrics,
                                   "calibration": calibration,
                                   "calibration_fit_split": "policy", "evaluation_used_for_fit_or_selection": False}
            if name in ("expanding", "recent_177160"):
                low, high = static_thresholds[window_id]
                static_metrics = evaluate_policy(y_eval, evaluation_scores, low, high)
                key_static = "v1_static" if name == "expanding" else "recent_model_only"
                key_adaptive = "adaptive_policy_only" if name == "expanding" else "full_v2_adaptive"
                system_entries[key_static].append({"window": window_id, "ranking": ranking,
                                                   "operational_metrics": static_metrics,
                                                   "thresholds": {"low": low, "high": high}})
                system_entries[key_adaptive].append({"window": window_id, "ranking": ranking,
                                                     "operational_metrics": adaptive_metrics,
                                                     "thresholds": {"low": adaptive["low_threshold"],
                                                                    "high": adaptive["high_threshold"]}})
            del encoded, model
            gc.collect()
        eval_block = pre_final.iloc[slice(*bounds["evaluation"])]
        results.append({"window": window_id, "evaluation_bounds": {"start": bounds["evaluation"][0],
                        "end_exclusive": bounds["evaluation"][1], "rows": len(eval_block),
                        "fraud_cases": int(eval_block.isFraud.sum()),
                        "fraud_prevalence": float(eval_block.isFraud.mean())},
                        "models": window_models, "final_test_rows_used": 0})
    predictive_summary = {}
    calibration_summary = {}
    for model_name in ("expanding", "recent_177160", "recency_weighted"):
        predictive_summary[model_name] = {
            "pr_auc": summarize([window["models"][model_name]["ranking"]["pr_auc"] for window in results]),
            "roc_auc": summarize([window["models"][model_name]["ranking"]["roc_auc"] for window in results])}
        calibration_summary[model_name] = {}
        for method in ("uncalibrated", "platt", "isotonic"):
            calibration_summary[model_name][method] = {
                metric: summarize([window["models"][model_name]["calibration"][method][metric] for window in results])
                for metric in ("brier_score", "log_loss", "ece_10_bin", "roc_auc", "pr_auc")}
    artifact = {"status": "V2 EXPLORATORY DEVELOPMENT EVIDENCE; NOT A NEW HOLDOUT",
                "predeclared_design": {"recent_history_rows": RECENT_HISTORY_ROWS,
                    "recency_half_life_rows": RECENCY_HALF_LIFE_ROWS,
                    "review_capacity": policy_config.review_capacity,
                    "legitimate_decline_limit": policy_config.max_legitimate_decline_rate,
                    "wilson_one_sided_confidence": policy_config.confidence_level},
                "windows": results, "predictive_summary": predictive_summary,
                "calibration_summary": calibration_summary, "final_test_rows_used": 0}
    ablation = {"status": "V2 2x2 EXPLORATORY SYSTEM ABLATION",
                "systems": {name: {"windows": entries, "summary": _summarize_system(entries, policy_config)}
                            for name, entries in system_entries.items()}, "final_test_rows_used": 0}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    Path(ablation_output).write_text(json.dumps(ablation, indent=2), encoding="utf-8")
    return artifact, ablation
