import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def load(name):
    return json.loads((ROOT / "artifacts" / "v2" / name).read_text(encoding="utf-8"))


def test_v2_artifacts_are_exploratory_and_exclude_consumed_final_rows():
    walk = load("walk_forward_results.json")
    ablation = load("system_ablation.json")
    assert "EXPLORATORY" in walk["status"] and "NOT A NEW HOLDOUT" in walk["status"]
    assert walk["final_test_rows_used"] == ablation["final_test_rows_used"] == 0
    assert all(window["final_test_rows_used"] == 0 for window in walk["windows"])
    assert max(window["evaluation_bounds"]["end_exclusive"] for window in walk["windows"]) == 546249


def test_every_v2_calibrator_and_policy_uses_historical_policy_split_only():
    walk = load("walk_forward_results.json")
    for window in walk["windows"]:
        for model in window["models"].values():
            assert model["calibration_fit_split"] == "policy"
            assert model["evaluation_used_for_fit_or_selection"] is False
            assert model["adaptive_policy"]["thresholds_frozen_before_evaluation"] is True
            assert model["adaptive_policy"]["legitimate_decline_wilson_upper"] <= .02


def test_system_ablation_contains_exact_predeclared_2x2_and_common_constraints():
    ablation = load("system_ablation.json")
    assert set(ablation["systems"]) == {
        "v1_static", "adaptive_policy_only", "recent_model_only", "full_v2_adaptive"
    }
    for system in ablation["systems"].values():
        definition = system["summary"]["constraint_definition"]
        assert definition == {"min_triage": .8, "max_review": .2, "max_legitimate_decline": .02}
