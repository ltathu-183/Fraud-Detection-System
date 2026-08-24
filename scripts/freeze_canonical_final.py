"""Create the one-time, non-overwriting canonical final-evaluation manifest."""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.models.lightgbm_model import LightGBMFraudModel
from src.pipeline.config import PipelineConfig


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def create_manifest(output="artifacts/evaluation/canonical_final_frozen.json",
                    transactions="data/raw/train_transaction.csv"):
    target = Path(output)
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite frozen artifact: {target}")
    canonical_path = Path("artifacts/corrected_results.json")
    results = json.loads(canonical_path.read_text(encoding="utf-8"))
    times = pd.read_csv(transactions, usecols=["TransactionID", "TransactionDT"]).sort_values(
        ["TransactionDT", "TransactionID"], kind="mergesort"
    ).reset_index(drop=True)
    fractions = PipelineConfig().split
    ends = [int(len(times) * value) for value in (
        fractions.train,
        fractions.train + fractions.validation,
        fractions.train + fractions.validation + fractions.policy,
        1.0,
    )]
    starts = [0, *ends[:-1]]
    names = ["train", "validation", "policy", "test"]
    boundaries = {}
    for name, start, end in zip(names, starts, ends):
        block = times.iloc[start:end]
        boundaries[name] = {
            "start_row_inclusive": start, "end_row_exclusive": end, "rows": len(block),
            "start_transaction_dt": int(block.TransactionDT.iloc[0]),
            "end_transaction_dt": int(block.TransactionDT.iloc[-1]),
            "start_transaction_id": int(block.TransactionID.iloc[0]),
            "end_transaction_id": int(block.TransactionID.iloc[-1]),
        }
    files = [canonical_path, Path("src/pipeline/ieee_cis_pipeline.py"),
             Path("src/pipeline/config.py"), Path("src/features/temporal_features.py"),
             Path("src/models/lightgbm_model.py"), Path("requirements.txt")]
    manifest = {
        "status": "IMMUTABLE HISTORICAL CANONICAL FINAL EVIDENCE",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "repository_version": "working-tree snapshot; hashes below are authoritative",
        "model_configuration": LightGBMFraudModel().params,
        "feature_set": results["feature_schema"],
        "split_boundaries": boundaries,
        "policy_thresholds": {
            "low": results["threshold_selection"]["low_threshold"],
            "high": results["threshold_selection"]["high_threshold"],
        },
        "policy_constraints": results["threshold_selection"]["constraints"],
        "final_test_metrics": {"ranking": results["ranking"], "policy": results["policy"]},
        "evaluation_provenance": results["evaluation_provenance"],
        "artifact_hashes_sha256": {str(path).replace("\\", "/"): sha256(path) for path in files},
        "governance_note": (
            "The final test was consumed under the previously frozen policy and is not reused "
            "for subsequent development. This file must not be overwritten."
        ),
    }
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/evaluation/canonical_final_frozen.json")
    parser.add_argument("--transactions", default="data/raw/train_transaction.csv")
    args = parser.parse_args()
    print(json.dumps(create_manifest(args.output, args.transactions), indent=2))
