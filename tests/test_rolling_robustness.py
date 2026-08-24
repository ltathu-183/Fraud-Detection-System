import inspect

import pandas as pd
import pytest

from scripts.freeze_canonical_final import create_manifest
from src.evaluation.rolling_robustness import (
    SAFETY_TARGETS,
    _encode_window,
    construct_rolling_windows,
    validate_windows,
)


def test_rolling_window_construction_is_deterministic():
    assert construct_rolling_windows(546249) == construct_rolling_windows(546249)


def test_frozen_canonical_manifest_refuses_overwrite(tmp_path):
    target = tmp_path / "already_frozen.json"
    target.write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        create_manifest(target)


def test_rolling_windows_are_chronological_and_disjoint_at_policy_evaluation_boundary():
    windows = construct_rolling_windows(546249)
    assert validate_windows(windows, 546249)
    for window in windows:
        assert window["policy"][1] == window["evaluation"][0]
        assert set(range(*window["policy"])).isdisjoint(range(*window["evaluation"]))


def test_consumed_final_test_is_excluded_from_every_development_window():
    windows = construct_rolling_windows(546249)
    assert max(window["evaluation"][1] for window in windows) == 546249
    assert all(window["evaluation"][1] <= 546249 for window in windows)


def test_preprocessing_fits_historical_train_only():
    frame = pd.DataFrame({
        "_internal_row_id": range(8), "_split_id": ["dev"] * 8,
        "TransactionID": range(8), "TransactionDT": range(8), "isFraud": [0, 1] * 4,
        "TransactionAmt": range(8), "category": ["old"] * 4 + ["future"] * 4,
    })
    bounds = {"window": 1, "train": (0, 4), "validation": (4, 5), "policy": (5, 6), "evaluation": (6, 8)}
    encoded, _, _ = _encode_window(frame, bounds)
    assert encoded.loc[:3, "category_freq"].eq(1).all()
    assert encoded.loc[4:, "category_freq"].eq(0).all()


def test_safety_targets_are_small_and_predeclared():
    assert SAFETY_TARGETS == (0.80, 0.82, 0.85)


def test_window_bounds_can_slice_without_treating_window_id_as_a_range():
    bounds = construct_rolling_windows(546249)[0]
    frame = pd.DataFrame({"x": range(bounds["evaluation"][1])})
    slices = {name: frame.iloc[bounds[name][0]:bounds[name][1]]
              for name in ("train", "validation", "policy", "evaluation")}
    assert len(slices["evaluation"]) == 44290


def test_thresholds_freeze_before_pseudo_future_prediction():
    source = inspect.getsource(__import__(
        "src.evaluation.rolling_robustness", fromlist=["run_rolling_experiment"]
    ).run_rolling_experiment)
    assert source.index("frozen_selections = {}") < source.index("evaluation_scores = model.predict_proba")
    assert '"thresholds_frozen_before_evaluation": True' in source


def test_safety_margin_selection_does_not_reference_evaluation_outcomes():
    source = inspect.getsource(__import__(
        "src.evaluation.rolling_robustness", fromlist=["run_rolling_experiment"]
    ).run_rolling_experiment)
    selection_block = source[source.index("frozen_selections = {}"):source.index("evaluation_scores = model.predict_proba")]
    assert "evaluation_scores" not in selection_block
    assert "y_evaluation" not in selection_block
