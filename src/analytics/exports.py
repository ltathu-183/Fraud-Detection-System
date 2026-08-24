"""Create reproducible decision-level, database, and dashboard artifacts."""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


def decision_labels(scores, low_threshold, high_threshold):
    scores = np.asarray(scores, dtype=float)
    return np.where(scores < low_threshold, "APPROVE", np.where(scores >= high_threshold, "DECLINE", "REVIEW"))


def build_decision_table(frame, scores, low_threshold, high_threshold):
    columns = ["TransactionID", "TransactionDT", "TransactionAmt", "isFraud"]
    optional = ["ProductCD", "card4", "card6", "P_emaildomain"]
    table = frame[[c for c in columns + optional if c in frame]].copy()
    table.columns = [c.lower() for c in table.columns]
    table = table.rename(columns={"transactionid": "transaction_id", "transactiondt": "timestamp_seconds",
                                  "transactionamt": "transaction_amount", "isfraud": "actual_label"})
    table["fraud_score"] = np.asarray(scores, dtype=float)
    table["decision"] = decision_labels(scores, low_threshold, high_threshold)
    return table


def write_analytics_artifacts(table, output_dir="artifacts/dashboard", database_path="artifacts/evaluation/fraud_analytics.sqlite"):
    out = Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    db = Path(database_path); db.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "transaction_decisions.csv", index=False)
    aggregate = table.groupby("decision", observed=True).agg(
        transaction_count=("transaction_id", "count"),
        fraud_count=("actual_label", "sum"),
        total_amount=("transaction_amount", "sum"),
        average_score=("fraud_score", "mean"),
    ).reset_index()
    aggregate.to_csv(out / "decision_summary.csv", index=False)
    with sqlite3.connect(db) as connection:
        table.to_sql("transaction_decisions", connection, if_exists="replace", index=False)
    return {"decision_table": str(out / "transaction_decisions.csv"),
            "decision_summary": str(out / "decision_summary.csv"), "database": str(db)}
