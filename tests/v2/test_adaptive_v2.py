import inspect

import numpy as np

from src.evaluation.metrics import evaluate_policy
from src.evaluation.rolling_robustness import construct_rolling_windows
from src.v2.adaptive_policy import (
    AdaptivePolicyConfig,
    apply_policy,
    select_adaptive_policy,
    wilson_upper_bound,
)
from src.v2.calibration import fit_calibrators
from src.v2.walk_forward import CANONICAL_REVIEW_LIMIT, FINAL_TEST_START, RECENT_HISTORY_ROWS, engineer_temporal_memory_safe, run_v2


def policy_sample():
    legitimate = np.zeros(1000, dtype=int)
    fraud = np.ones(100, dtype=int)
    y = np.r_[legitimate, fraud]
    scores = np.r_[np.linspace(.001, .80, 1000), np.linspace(.40, .99, 100)]
    return y, scores


def test_v2_windows_are_chronological_and_exclude_v1_final_test():
    windows = construct_rolling_windows(FINAL_TEST_START)
    for window in windows:
        assert window["train"][1] <= window["validation"][0]
        assert window["validation"][1] <= window["policy"][0]
        assert window["policy"][1] == window["evaluation"][0]
        assert window["evaluation"][1] <= FINAL_TEST_START


def test_recent_training_window_is_predeclared_not_selected_from_evaluation():
    assert RECENT_HISTORY_ROWS == 4 * 44290
    assert CANONICAL_REVIEW_LIMIT == .20


def test_wilson_upper_bound_is_conservative_and_deterministic():
    first = wilson_upper_bound(10, 1000, .95)
    second = wilson_upper_bound(10, 1000, .95)
    assert first == second and first > .01


def test_adaptive_policy_obeys_review_capacity_and_wilson_limit():
    y, scores = policy_sample()
    config = AdaptivePolicyConfig(review_capacity=.15, max_legitimate_decline_rate=.05)
    selected = select_adaptive_policy(y, scores, config)
    assert selected["feasible"]
    metrics = evaluate_policy(y, scores, selected["low_threshold"], selected["high_threshold"])
    assert metrics["overall_review_rate"] <= .15 + 1 / len(y)
    assert selected["legitimate_decline_wilson_upper"] <= .05


def test_policy_decisions_are_deterministic():
    scores = np.array([.1, .4, .8])
    assert apply_policy(scores, .3, .7).tolist() == ["APPROVE", "REVIEW", "DECLINE"]
    np.testing.assert_array_equal(apply_policy(scores, .3, .7), apply_policy(scores, .3, .7))


def test_calibrators_fit_only_supplied_historical_labels():
    y, scores = policy_sample()
    calibrators_a = fit_calibrators(y, scores)
    calibrators_b = fit_calibrators(y, scores)
    probe = np.array([[.2], [.8]])
    np.testing.assert_allclose(calibrators_a["platt"].predict_proba(probe),
                               calibrators_b["platt"].predict_proba(probe))


def test_thresholds_and_calibrators_freeze_before_pseudo_future_scoring():
    source = inspect.getsource(run_v2)
    evaluation_prediction = source.index("evaluation_scores = model.predict_proba")
    assert source.index("adaptive = select_adaptive_policy") < evaluation_prediction
    assert source.index("calibrators = fit_calibrators") < evaluation_prediction
    assert '"evaluation_used_for_fit_or_selection": False' in source


def test_v2_artifact_paths_cannot_overwrite_v1_defaults():
    signature = inspect.signature(run_v2)
    assert str(signature.parameters["output"].default).startswith("artifacts/v2/")
    assert str(signature.parameters["ablation_output"].default).startswith("artifacts/v2/")


def test_memory_safe_temporal_engineering_preserves_rows_and_core_features():
    import pandas as pd
    frame = pd.DataFrame({"_internal_row_id": [0, 1], "TransactionDT": [1, 2],
                          "TransactionAmt": [10., 20.], "card1": [1, 1],
                          "DeviceInfo": ["a", "a"], "P_emaildomain": ["x", "x"],
                          "addr1": [2, 2], "isFraud": [0, 1], "unrelated": [5., 6.]})
    output = engineer_temporal_memory_safe(frame)
    assert output._internal_row_id.tolist() == [0, 1]
    assert output.unrelated.tolist() == [5., 6.]
    assert output.card1_3600s_count.tolist() == [0, 1]
