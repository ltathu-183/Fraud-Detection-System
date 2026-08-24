"""Run the isolated exploratory adaptive temporal V2."""

import json

from src.v2.walk_forward import run_v2


if __name__ == "__main__":
    results, ablation = run_v2()
    print(json.dumps({"predictive_summary": results["predictive_summary"],
                      "systems": {name: value["summary"] for name, value in ablation["systems"].items()},
                      "final_test_rows_used": results["final_test_rows_used"]}, indent=2))
