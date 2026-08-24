"""Development-only rolling temporal policy robustness analysis."""

from dataclasses import replace
from pathlib import Path
import json

import numpy as np
import pandas as pd

from src.data.categorical_encoding import CategoricalEncoder
from src.evaluation.metrics import calculate_ranking_metrics, evaluate_policy, optimize_policy_thresholds
from src.features.temporal_features import TemporalFeatureEngineer
from src.models.lightgbm_model import LightGBMFraudModel
from src.monitoring.monitoring import population_stability_index
from src.pipeline.config import PipelineConfig
from src.pipeline.ieee_cis_pipeline import FraudDetectionPipeline, IDENTITY_COLUMNS


SAFETY_TARGETS = (0.80, 0.82, 0.85)


def construct_rolling_windows(pre_final_rows, n_windows=4, block_rows=44290):
    """Deterministic expanding-train windows wholly before the frozen final test."""
    required = (n_windows + 2) * block_rows
    initial_train = pre_final_rows - required
    if initial_train < block_rows:
        raise ValueError("Insufficient pre-final history for meaningful rolling windows")
    windows = []
    for index in range(n_windows):
        train_end = initial_train + index * block_rows
        validation_end = train_end + block_rows
        policy_end = validation_end + block_rows
        evaluation_end = policy_end + block_rows
        windows.append({
            "window": index + 1,
            "train": (0, train_end),
            "validation": (train_end, validation_end),
            "policy": (validation_end, policy_end),
            "evaluation": (policy_end, evaluation_end),
        })
    return windows


def validate_windows(windows, frozen_test_start):
    for window in windows:
        train, validation, policy, evaluation = (window[name] for name in ("train", "validation", "policy", "evaluation"))
        if not (train[0] == 0 and train[1] <= validation[0] < validation[1]
                <= policy[0] < policy[1] <= evaluation[0] < evaluation[1]
                <= frozen_test_start):
            raise AssertionError("Rolling chronology, disjointness, or final-test exclusion violated")
    return True


def summarize(values):
    array = np.asarray(values, dtype=float)
    return {"mean": float(array.mean()), "median": float(np.median(array)),
            "min": float(array.min()), "max": float(array.max()),
            "std": float(array.std(ddof=0)),
            "iqr": float(np.quantile(array, .75) - np.quantile(array, .25))}


def constraint_outcome(metrics, config):
    deviations = {
        "triage_shortfall": max(0.0, config.min_recall - metrics["fraud_triage_coverage"]),
        "review_excess": max(0.0, metrics["overall_review_rate"] - config.max_review_rate),
        "legitimate_decline_excess": max(0.0, metrics["legitimate_auto_decline_rate"] - config.max_false_decline_rate),
    }
    passes = {name: value == 0 for name, value in deviations.items()}
    return {"all_constraints_satisfied": all(passes.values()), "passes": passes, "violation_magnitude": deviations}


def summarize_safety_targets(safety):
    summaries = {}
    for target, entries in safety.items():
        feasible = [entry for entry in entries if entry.get("feasible_on_policy")]
        metrics = [entry["evaluation_metrics"] for entry in feasible]
        summaries[target] = {
            "windows": len(entries), "policy_feasible_windows": len(feasible),
            "all_constraints_satisfied_count": sum(
                entry["canonical_constraint_outcome"]["all_constraints_satisfied"] for entry in feasible
            ),
            "all_constraints_satisfied_fraction_all_windows": sum(
                entry["canonical_constraint_outcome"]["all_constraints_satisfied"] for entry in feasible
            ) / len(entries),
            "future_triage_coverage": summarize([metric["fraud_triage_coverage"] for metric in metrics]),
            "future_review_rate": summarize([metric["overall_review_rate"] for metric in metrics]),
            "future_legitimate_auto_decline_rate": summarize([
                metric["legitimate_auto_decline_rate"] for metric in metrics
            ]),
        }
    return summaries


def _boundary(frame, start, end):
    block = frame.iloc[start:end]
    return {"start_row_inclusive": start, "end_row_exclusive": end, "rows": len(block),
            "fraud_cases": int(block.isFraud.sum()), "fraud_prevalence": float(block.isFraud.mean()),
            "start_transaction_dt": int(block.TransactionDT.iloc[0]),
            "end_transaction_dt": int(block.TransactionDT.iloc[-1]),
            "start_transaction_id": int(block.TransactionID.iloc[0]),
            "end_transaction_id": int(block.TransactionID.iloc[-1])}


def _encode_window(featured, bounds):
    train_start, train_end = bounds["train"]
    encoder = CategoricalEncoder()
    categorical = featured.iloc[train_start:train_end].select_dtypes(include=["object", "string"]).columns.difference(["_split_id"]).tolist()
    encoder.fit(featured.iloc[train_start:train_end], categorical)
    encoded = encoder.transform(featured.iloc[:bounds["evaluation"][1]].copy(), categorical)
    feature_cols = [c for c in encoded.columns if c not in IDENTITY_COLUMNS and pd.api.types.is_numeric_dtype(encoded[c])]
    return encoded, feature_cols, categorical


def run_rolling_experiment(transactions="data/raw/train_transaction.csv",
                           identity="data/raw/train_identity.csv",
                           frozen_manifest="artifacts/evaluation/canonical_final_frozen.json",
                           output="artifacts/evaluation/rolling_temporal_robustness.json",
                           margin_output="artifacts/evaluation/policy_safety_margin.json"):
    frozen = json.loads(Path(frozen_manifest).read_text(encoding="utf-8"))
    frozen_test_start = frozen["split_boundaries"]["test"]["start_row_inclusive"]
    # Loading is schema-identical to canonical; every row at/after the frozen
    # boundary is immediately excluded and never enters any statistic or fit.
    raw_all = FraudDetectionPipeline().load_data(transactions, identity)
    pre_final = raw_all.iloc[:frozen_test_start].copy().reset_index(drop=True)
    pre_final["_internal_row_id"] = np.arange(len(pre_final), dtype=np.int64)
    pre_final["_split_id"] = "rolling_development"
    windows = construct_rolling_windows(len(pre_final))
    validate_windows(windows, frozen_test_start)
    # Point-in-time features are label-free and depend only on earlier rows.
    featured = TemporalFeatureEngineer().engineer_all_features(pre_final)
    config = PipelineConfig()
    window_results = []
    safety = {str(int(target * 100)): [] for target in SAFETY_TARGETS}
    for bounds in windows:
        encoded, feature_cols, categorical = _encode_window(featured, bounds)
        slices = {name: encoded.iloc[bounds[name][0]:bounds[name][1]]
                  for name in ("train", "validation", "policy", "evaluation")}
        model = LightGBMFraudModel()
        model.train(slices["train"][feature_cols], slices["train"].isFraud,
                    slices["validation"][feature_cols], slices["validation"].isFraud,
                    feature_names=feature_cols)
        policy_scores = model.predict_proba(slices["policy"][feature_cols])[:, 1]
        # All target-specific thresholds freeze before pseudo-future scoring.
        frozen_selections = {}
        for target in SAFETY_TARGETS:
            target_config = replace(config.policy, min_recall=target)
            frozen_selections[str(int(target * 100))] = optimize_policy_thresholds(
                slices["policy"].isFraud.to_numpy(), policy_scores, target_config
            )
        evaluation_scores = model.predict_proba(slices["evaluation"][feature_cols])[:, 1]
        y_evaluation = slices["evaluation"].isFraud.to_numpy()
        ranking = calculate_ranking_metrics(y_evaluation, evaluation_scores)
        margin_metrics = {}
        for target_name, selection in frozen_selections.items():
            if not selection["feasible"]:
                margin_metrics[target_name] = {"feasible_on_policy": False}
                safety[target_name].append({"window": bounds["window"], "feasible_on_policy": False})
                continue
            metrics = evaluate_policy(y_evaluation, evaluation_scores, selection["low_threshold"], selection["high_threshold"])
            outcome = constraint_outcome(metrics, replace(config.policy, min_recall=.80))
            entry = {"window": bounds["window"], "feasible_on_policy": True,
                     "selection_target": int(target_name) / 100,
                     "low_threshold": selection["low_threshold"], "high_threshold": selection["high_threshold"],
                     "policy_window_metrics": {k: selection[k] for k in (
                         "fraud_triage_coverage", "overall_review_rate", "legitimate_auto_decline_rate")},
                     "evaluation_metrics": metrics, "canonical_constraint_outcome": outcome,
                     "thresholds_frozen_before_evaluation": True}
            margin_metrics[target_name] = entry
            safety[target_name].append(entry)
        canonical = margin_metrics["80"]
        drift_features = {}
        for col in ("TransactionAmt", "card1", "dist1", "C1", "D1"):
            if col in feature_cols:
                drift_features[col] = population_stability_index(slices["policy"][col], slices["evaluation"][col])
        window_results.append({
            "window": bounds["window"],
            "boundaries": {name: _boundary(pre_final, *bounds[name]) for name in ("train", "validation", "policy", "evaluation")},
            "preprocessing_fit_end_exclusive": bounds["train"][1],
            "categorical_columns_fit_on_train": categorical,
            "model_metrics": ranking,
            "score_distribution": {"mean": float(evaluation_scores.mean()), "std": float(evaluation_scores.std()),
                                   "min": float(evaluation_scores.min()), "max": float(evaluation_scores.max())},
            "canonical_policy": canonical,
            "safety_margins": margin_metrics,
            "drift": {"score_psi_policy_to_evaluation": population_stability_index(policy_scores, evaluation_scores),
                      "feature_psi_policy_to_evaluation": drift_features},
            "final_test_rows_used": 0,
        })
    aggregates = {
        "roc_auc": summarize([w["model_metrics"]["roc_auc"] for w in window_results]),
        "pr_auc": summarize([w["model_metrics"]["pr_auc"] for w in window_results]),
        "fraud_prevalence": summarize([w["boundaries"]["evaluation"]["fraud_prevalence"] for w in window_results]),
    }
    for metric in ("fraud_triage_coverage", "overall_review_rate", "legitimate_auto_decline_rate"):
        aggregates[metric] = summarize([w["canonical_policy"]["evaluation_metrics"][metric] for w in window_results])
    constraints = {}
    canonical_outcomes = [w["canonical_policy"]["canonical_constraint_outcome"] for w in window_results]
    constraints["all_satisfied_count"] = sum(o["all_constraints_satisfied"] for o in canonical_outcomes)
    constraints["all_satisfied_fraction"] = constraints["all_satisfied_count"] / len(canonical_outcomes)
    for key in ("triage_shortfall", "review_excess", "legitimate_decline_excess"):
        values = [o["violation_magnitude"][key] for o in canonical_outcomes]
        constraints[key] = {"violating_windows": sum(v > 0 for v in values),
                            "average_violation_all_windows": float(np.mean(values)), "worst_violation": float(max(values))}
    result = {
        "status": "PRE-FINAL DEVELOPMENT-ONLY ROLLING ROBUSTNESS",
        "canonical_frozen_manifest": frozen_manifest,
        "frozen_test_start_row": frozen_test_start,
        "final_test_rows_used": 0,
        "design": {"windows": len(windows), "safety_targets_predeclared": SAFETY_TARGETS,
                   "model_design": "fixed canonical LightGBM; no new features/models/hyperparameter search"},
        "windows": window_results, "aggregate_temporal_distribution": aggregates,
        "canonical_constraint_generalization": constraints,
    }
    margin_result = {"status": "PRE-FINAL DEVELOPMENT-ONLY SAFETY-MARGIN ANALYSIS",
                     "targets": safety, "summary": summarize_safety_targets(safety), "final_test_rows_used": 0,
                     "selection_rule": "Targets were predeclared; pseudo-future outcomes were not used to select thresholds."}
    Path(output).write_text(json.dumps(result, indent=2), encoding="utf-8")
    Path(margin_output).write_text(json.dumps(margin_result, indent=2), encoding="utf-8")
    return result, margin_result
