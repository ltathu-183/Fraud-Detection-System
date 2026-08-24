import sqlite3

import numpy as np
import pandas as pd

from scripts.run_sql_analytics import run_queries
from src.analytics.exports import build_decision_table, write_analytics_artifacts
from src.monitoring.monitoring import population_stability_index
from src.pipeline.config import CostScenarioConfig
from src.policy.business_scenarios import DISCLAIMER, compare_policies, expected_cost


def test_reviewed_fraud_cost_is_not_treated_as_confirmed_detection():
    scenario = CostScenarioConfig(missed_fraud_cost=100, false_decline_cost=10,
                                  manual_review_cost=1, reviewed_fraud_recovery_rate=.5)
    # Fraud rows: one approved (100), one reviewed (50 expected residual); one review costs 1.
    result = expected_cost([1, 1, 0], [.1, .5, .1], .2, .8, scenario)
    assert result["expected_cost_total"] == 151
    assert "not representative" in DISCLAIMER


def test_policy_comparison_keeps_detection_and_routing_semantics_separate():
    result = compare_policies([1, 1, 0, 0], [.9, .5, .4, .1], {"p": (.3, .8)}, CostScenarioConfig())
    metrics = result["policies"]["p"]["operational_metrics"]
    assert metrics["fraud_auto_decline_recall"] == .5
    assert metrics["fraud_review_coverage"] == .5
    assert metrics["fraud_triage_coverage"] == 1


def test_psi_is_zero_for_identical_populations():
    values = np.arange(100)
    assert population_stability_index(values, values) == 0


def test_sql_queries_execute_and_return_metrics(tmp_path):
    frame = pd.DataFrame({"TransactionID": [1, 2, 3, 4], "TransactionDT": [1, 2, 3, 4],
                          "TransactionAmt": [10., 20., 30., 40.], "isFraud": [0, 1, 0, 1],
                          "ProductCD": ["W", "W", "C", "C"]})
    table = build_decision_table(frame, [.1, .5, .9, .7], .3, .8)
    artifacts = write_analytics_artifacts(table, tmp_path / "dashboard", tmp_path / "analytics.sqlite")
    outputs = run_queries(artifacts["database"], "sql", tmp_path / "query_outputs")
    assert len(outputs) == 3
    with sqlite3.connect(artifacts["database"]) as connection:
        assert connection.execute("SELECT COUNT(*) FROM transaction_decisions").fetchone()[0] == 4
