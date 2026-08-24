"""Execute every committed analytics query against the generated SQLite database."""

import argparse
import sqlite3
from pathlib import Path

import pandas as pd


def run_queries(database, sql_dir, output_dir):
    output = Path(output_dir); output.mkdir(parents=True, exist_ok=True)
    generated = []
    with sqlite3.connect(database) as connection:
        for query_path in sorted(Path(sql_dir).glob("*.sql")):
            result = pd.read_sql_query(query_path.read_text(encoding="utf-8"), connection)
            target = output / f"{query_path.stem}.csv"
            result.to_csv(target, index=False)
            generated.append(str(target))
    if not generated:
        raise RuntimeError("No SQL queries found")
    return generated


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", default="artifacts/evaluation/fraud_analytics.sqlite")
    parser.add_argument("--sql-dir", default="sql")
    parser.add_argument("--output-dir", default="artifacts/dashboard")
    args = parser.parse_args()
    for path in run_queries(args.database, args.sql_dir, args.output_dir):
        print(path)


if __name__ == "__main__":
    main()
