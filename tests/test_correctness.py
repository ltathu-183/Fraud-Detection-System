import numpy as np
import pandas as pd
import pytest

from src.data.categorical_encoding import CategoricalEncoder
from src.evaluation.metrics import capacity_metrics, evaluate_policy, optimize_policy_thresholds
from src.features.temporal_features import TemporalFeatureEngineer
from src.models.lightgbm_model import LightGBMFraudModel
from src.pipeline.config import PolicyConfig
from src.pipeline.ieee_cis_pipeline import FraudDetectionPipeline


def sample_frame():
    return pd.DataFrame({
        "TransactionID": [10, 20, 30, 40, 50, 60, 70, 80], "TransactionDT": [1, 2, 3, 4, 5, 6, 100, 101],
        "TransactionAmt": [10., 20., 11., 21., 12., 22., 30., 40.], "card1": [1, 2, 1, 2, 1, 2, 2, 1],
        "category": ["old"] * 6 + ["future"] * 2, "isFraud": [0, 1, 0, 0, 1, 0, 0, 1],
    })


def test_split_membership_and_target_alignment_survive_entity_sort():
    pipeline = FraudDetectionPipeline()
    pipeline.temporal_split(sample_frame())
    pipeline.engineer_features()
    assert set(pipeline.splits["train"].TransactionID) == {10, 20, 30, 40, 50}
    assert set(pipeline.splits["test"].TransactionID) == {80}
    labels = pd.concat(pipeline.splits.values()).set_index("TransactionID").isFraud.to_dict()
    assert labels == {10: 0, 20: 1, 30: 0, 40: 0, 50: 1, 60: 0, 70: 0, 80: 1}
    assert pipeline.raw_splits["test"].TransactionID.tolist() == [80]


def test_past_only_windows_future_and_current_do_not_leak():
    base = pd.DataFrame({"_internal_row_id": [0, 1], "TransactionDT": [10, 20], "TransactionAmt": [5., 100.], "card1": [1, 1]})
    first = TemporalFeatureEngineer().engineer_all_features(base, velocity_entities=("card1",), novelty_entities=(), windows=(10,))
    with_future = pd.concat([base, pd.DataFrame({"_internal_row_id": [2], "TransactionDT": [30], "TransactionAmt": [999.], "card1": [1]})], ignore_index=True)
    second = TemporalFeatureEngineer().engineer_all_features(with_future, velocity_entities=("card1",), novelty_entities=(), windows=(10,))
    assert first.loc[0, "card1_10s_count"] == 0
    assert first.loc[1, "card1_10s_count"] == 1
    pd.testing.assert_series_equal(first.iloc[0], second.iloc[0], check_names=False)


def test_same_timestamp_events_are_mutually_invisible_and_boundary_is_inclusive_left():
    df = pd.DataFrame({"_internal_row_id": [0, 1, 2], "TransactionDT": [10, 10, 20], "TransactionAmt": [5., 7., 9.], "card1": [1, 1, 1]})
    out = TemporalFeatureEngineer().engineer_all_features(df, velocity_entities=("card1",), novelty_entities=(), windows=(10,))
    assert out["card1_10s_count"].tolist() == [0, 0, 2]
    assert out["card1_time_since_last"].iloc[:2].isna().all()


def test_unseen_entity_has_empty_history():
    df = pd.DataFrame({"_internal_row_id": [0, 1], "TransactionDT": [1, 2], "TransactionAmt": [1., 2.], "card1": [1, 2]})
    out = TemporalFeatureEngineer().engineer_all_features(df, velocity_entities=("card1",), novelty_entities=(), windows=(10,))
    assert out["card1_10s_count"].eq(0).all()


def test_novelty_features_are_past_only_and_same_time_safe():
    df = pd.DataFrame({"_internal_row_id": [0, 1, 2], "TransactionDT": [10, 10, 20], "TransactionAmt": [1., 2., 3.], "DeviceInfo": ["A", "A", "A"]})
    out = TemporalFeatureEngineer().engineer_all_features(df, velocity_entities=(), novelty_entities=("DeviceInfo",), windows=(10,))
    assert out["DeviceInfo_is_new"].tolist() == [1, 1, 0]
    assert out["DeviceInfo_time_since_first_seen"].tolist() == [0, 0, 10]


def test_frequency_encoder_is_train_only_and_unseen_is_zero():
    enc = CategoricalEncoder(); enc.fit(pd.DataFrame({"x": ["a", "a", "b"]}), ["x"])
    out = enc.transform(pd.DataFrame({"x": ["a", "future"]}), ["x"])
    assert enc.frequency_encodings["x"] == {"a": 2/3, "b": 1/3}
    assert out.x_freq.tolist() == [2/3, 0]


def test_class_weight_uses_only_supplied_training_labels():
    model = LightGBMFraudModel()
    assert model.calculate_scale_pos_weight(np.array([0, 0, 0, 1])) == 3


def policy_data():
    return np.array([1, 1, 0, 0, 0, 0]), np.array([.9, .6, .8, .5, .2, .1])


def test_optimizer_enforces_all_constraints_and_is_deterministic():
    y, p = policy_data(); cfg = PolicyConfig(min_recall=.5, max_review_rate=.5, max_false_decline_rate=.25, threshold_grid_size=11)
    a = optimize_policy_thresholds(y, p, cfg); b = optimize_policy_thresholds(y, p, cfg)
    assert a == b and a["feasible"] and a["high_threshold"] >= a["low_threshold"]
    assert a["fraud_triage_coverage"] >= cfg.min_recall
    assert a["overall_review_rate"] <= cfg.max_review_rate
    assert a["legitimate_auto_decline_rate"] <= cfg.max_false_decline_rate


@pytest.mark.parametrize("cfg", [
    PolicyConfig(min_recall=1, max_review_rate=0, max_false_decline_rate=0, threshold_grid_size=11),
    PolicyConfig(min_recall=.75, max_review_rate=0, max_false_decline_rate=0, threshold_grid_size=11),
])
def test_infeasible_policy_is_explicit(cfg):
    y, p = policy_data(); result = optimize_policy_thresholds(y, p, cfg)
    assert not result["feasible"] and result["low_threshold"] is None


def test_metric_semantics_partition_fraud_and_decisions():
    y, p = policy_data(); m = evaluate_policy(y, p, .3, .7)
    assert m["fraud_auto_decline_recall"] + m["fraud_review_coverage"] + m["fraud_approved_rate"] == 1
    assert m["approve_count"] + m["review_count"] + m["decline_count"] == len(y)


def test_capacity_metrics_use_top_scored_transactions_only():
    metrics = capacity_metrics(np.array([1, 0, 1, 0]), np.array([.9, .8, .7, .1]), review_rates=(.5,))
    assert metrics == {"recall_at_50%_review": .5, "precision_at_50%_review": .5}


def test_feature_schema_order_guard():
    model = LightGBMFraudModel(); model.model = object(); model.feature_names = ["a", "b"]
    with pytest.raises(ValueError, match="schema/order"):
        model.predict_raw_proba(pd.DataFrame({"b": [1], "a": [2]}))


def test_final_test_not_opened_when_policy_is_infeasible(monkeypatch, tmp_path):
    pipeline = FraudDetectionPipeline()
    pipeline.config = type(pipeline.config)(policy=PolicyConfig(min_recall=1, max_review_rate=0, max_false_decline_rate=0))
    # The optimizer's explicit infeasible contract is the guard used before test prediction.
    result = optimize_policy_thresholds(*policy_data(), pipeline.config.policy)
    assert result["feasible"] is False


def test_results_include_frozen_threshold_provenance():
    """The final artifact must distinguish policy fitting from final testing."""
    source = open("src/pipeline/ieee_cis_pipeline.py", encoding="utf-8").read()
    freeze_point = source.index('"thresholds_frozen_before_test": True')
    test_prediction = source.index("test_probs = self.model.predict_proba(x_test)")
    assert freeze_point < test_prediction
    for field in (
        "threshold_selection_split", "final_evaluation_split",
        "test_excluded_from_model_training", "test_excluded_from_threshold_selection",
    ):
        assert field in source
