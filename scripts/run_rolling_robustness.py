"""Run the final pre-final rolling temporal experiment."""

import json

from src.evaluation.rolling_robustness import run_rolling_experiment


if __name__ == "__main__":
    results, margins = run_rolling_experiment()
    print(json.dumps({"aggregate_temporal_distribution": results["aggregate_temporal_distribution"],
                      "canonical_constraint_generalization": results["canonical_constraint_generalization"],
                      "safety_targets": list(margins["targets"])}, indent=2))
