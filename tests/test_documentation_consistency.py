import hashlib
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_json(relative_path):
    return json.loads((ROOT / relative_path).read_text(encoding="utf-8"))


def test_readme_headlines_match_frozen_canonical_artifact():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    frozen = load_json("artifacts/evaluation/canonical_final_frozen.json")
    ranking = frozen["final_test_metrics"]["ranking"]
    policy = frozen["final_test_metrics"]["policy"]
    expected = {
        f"{ranking['roc_auc']:.4f}", f"{ranking['pr_auc']:.4f}",
        f"{policy['fraud_auto_decline_recall']:.2%}",
        f"{policy['fraud_review_coverage']:.2%}",
        f"{policy['fraud_triage_coverage']:.2%}",
        f"{policy['overall_review_rate']:.2%}",
        f"{policy['legitimate_auto_decline_rate']:.2%}",
    }
    assert expected.issubset(set(re.findall(r"\d+\.\d+%?|\d+\.\d+", readme)))
    assert "did not meet the final-test 80% target" in readme


def test_readme_rolling_ranges_match_robustness_artifact():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    rolling = load_json("artifacts/evaluation/rolling_temporal_robustness.json")
    aggregate = rolling["aggregate_temporal_distribution"]
    separator = "\u2013"
    expected_ranges = (
        f"{aggregate['pr_auc']['min']:.4f}{separator}{aggregate['pr_auc']['max']:.4f}",
        f"{aggregate['fraud_triage_coverage']['min']:.2%}{separator}{aggregate['fraud_triage_coverage']['max']:.2%}",
    )
    assert all(value in readme for value in expected_ranges)
    assert rolling["final_test_rows_used"] == 0


def test_policy_report_matches_safety_margin_artifact():
    report = (ROOT / "reports/POLICY_ROBUSTNESS.md").read_text(encoding="utf-8")
    margins = load_json("artifacts/evaluation/policy_safety_margin.json")
    for target, summary in margins["summary"].items():
        assert f"| {target}% |" in report
        assert f"{summary['future_triage_coverage']['mean']:.2%}" in report
        assert f"{summary['future_review_rate']['mean']:.2%}" in report
    assert margins["final_test_rows_used"] == 0


def test_frozen_artifact_hashes_still_match():
    frozen = load_json("artifacts/evaluation/canonical_final_frozen.json")
    for relative_path, expected_hash in frozen["artifact_hashes_sha256"].items():
        actual_hash = hashlib.sha256((ROOT / relative_path).read_bytes()).hexdigest()
        assert actual_hash == expected_hash, relative_path


def test_public_markdown_relative_links_resolve_and_has_no_absolute_windows_paths():
    markdown_files = [ROOT / "README.md", ROOT / "PROJECT_STATUS.md", *sorted((ROOT / "reports").glob("*.md"))]
    link_pattern = re.compile(r"\[[^]]+\]\(([^)]+)\)")
    for document in markdown_files:
        text = document.read_text(encoding="utf-8")
        assert not re.search(r"[A-Za-z]:\\", text), document
        for target in link_pattern.findall(text):
            if "://" in target or target.startswith("#"):
                continue
            assert (document.parent / target).resolve().exists(), f"{document}: {target}"


def test_cv_ledger_has_exact_required_categories_and_prohibitions():
    ledger = (ROOT / "reports/CV_CLAIMS.md").read_text(encoding="utf-8")
    headings = re.findall(r"^## (.+)$", ledger, flags=re.MULTILINE)
    assert headings == ["SAFE FOR CV", "SAFE FOR INTERVIEW", "DO NOT CLAIM"]
    for prohibited in (
        "Met the 80% triage constraint on final test",
        "Guaranteed ≥80% fraud coverage",
        "Production-ready fraud system",
        "PSI proves no drift",
        "Temporal policy is stable",
        "Real bank cost optimization",
    ):
        assert prohibited in ledger


def test_public_status_separates_frozen_v1_from_exploratory_v2():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    status = (ROOT / "PROJECT_STATUS.md").read_text(encoding="utf-8")
    v2 = load_json("artifacts/v2/system_ablation.json")
    assert "V1 is **CANONICAL / FROZEN**" in readme
    assert "V2 used zero final-test rows" in readme
    assert "STATIC V1 REMAINS PREFERRED" in status.upper()
    assert v2["final_test_rows_used"] == 0
