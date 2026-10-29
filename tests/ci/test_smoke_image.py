# SPDX-License-Identifier: Apache-2.0
"""Guard the smoke harness's own acceptance checks; actual image tests run Docker."""

import importlib.util
import json
from pathlib import Path

import pytest

SPEC = importlib.util.spec_from_file_location("ci_smoke_image", Path(__file__).parents[2] / "ci/smoke_image.py")
smoke = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(smoke)


def _no_change(directory):
    summary = {
        "overall_status": "skipped",
        "overall_passed": False,
        "total": 0,
        "passed": 0,
        "failed": 0,
        "incomplete": 0,
        "total_validators": 0,
        "skills": [],
        "execution": {"llm_enabled": False},
        "selection": {"mode": "changed", "discovered": 2, "selected": []},
    }
    for name in ("summary.md", "details.md", "index.html", "pr-comment.md"):
        (directory / name).write_text("No skills changed. Validation did not run.")
    (directory / "summary.json").write_text(json.dumps(summary))
    return summary


def test_neutral_no_change_is_accepted_without_inventing_pass(tmp_path):
    _no_change(tmp_path)
    result = smoke.verify_case(tmp_path, "no-change", 0)
    assert result["verified"] and result["overall_status"] == "skipped"


@pytest.mark.parametrize("change", ["exit", "green", "selected", "validators", "scope", "missing"])
def test_no_change_rejects_wrong_exit_or_evidence(tmp_path, change):
    summary = _no_change(tmp_path)
    if change == "green":
        summary.update(overall_status="passed", overall_passed=True)
    elif change == "selected":
        summary["selection"]["selected"] = ["skills/glossary-builder"]
    elif change == "validators":
        summary["total_validators"] = 12
    elif change == "scope":
        summary["execution"]["llm_enabled"] = True
    elif change == "missing":
        (tmp_path / "details.md").unlink()
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(smoke.SmokeError):
        smoke.verify_case(tmp_path, "no-change", 1 if change == "exit" else 0)


def test_missing_llm_key_requires_precise_configuration_failure(tmp_path):
    summary = _no_change(tmp_path)
    summary.update(overall_status="error", overall_passed=False, error="LLM_API_KEY is required")
    summary["execution"]["llm_enabled"] = True
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    (tmp_path / "summary.md").write_text("ERROR: LLM_API_KEY is required")
    assert smoke.verify_case(tmp_path, "missing-llm-key", 1)["overall_status"] == "error"
    summary["error"] = "Generic failure"
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(smoke.SmokeError):
        smoke.verify_case(tmp_path, "missing-llm-key", 1)


@pytest.mark.parametrize("case", ["no-change", "missing-llm-key"])
def test_real_github_rendered_markdown_is_accepted(tmp_path, case):
    """Replay exact summary Markdown downloaded from the real image-verification run."""
    summary = _no_change(tmp_path)
    if case == "missing-llm-key":
        summary.update(overall_status="error", error="LLM_API_KEY is required when LLM_ENABLED is true.")
        summary["execution"]["llm_enabled"] = True
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    captured = Path(__file__).with_name("fixtures") / f"smoke-{case}-summary.md"
    (tmp_path / "summary.md").write_bytes(captured.read_bytes())
    assert smoke.verify_case(tmp_path, case, 1 if case == "missing-llm-key" else 0)["verified"]


def _scan_case(directory, case):
    failing = case == "invalid-name"
    paths = (
        ["skills/glossary-builder"] if case == "changed-one" else ["skills/evidence-table", "skills/glossary-builder"]
    )
    rows = []
    for index, path in enumerate(paths, 1):
        passed = not (failing and path.endswith("glossary-builder"))
        row = {
            "path": path,
            "report_dir": f"skill-{index:03d}",
            "passed": passed,
            "exit_code": int(not passed),
            "overall_status": "passed" if passed else "failed",
            "incomplete_scans": [],
        }
        rows.append(row)
        child = directory / row["report_dir"]
        child.mkdir()
        results = [{"status": "passed", "incomplete_scans": [], "findings": []} for _ in range(10)]
        if not passed:
            results[0].update(status="failed", findings=[{"category": "SCHEMA", "message": "INVALID NAME is invalid"}])
        payload = {
            "overall_passed": passed,
            "overall_status": row["overall_status"],
            "incomplete_scans": [],
            "results": results,
        }
        (child / "skillevaluator-tier1.json").write_text(json.dumps(payload))
        for filename in (
            "skillevaluator-tier1.html",
            "skillevaluator-tier1.md",
            "skillevaluator-tier1.sarif.json",
            "stdout.log",
            "stderr.log",
        ):
            (child / filename).write_text("test fixture")
    summary = {
        "overall_status": "failed" if failing else "passed",
        "overall_passed": not failing,
        "total": len(rows),
        "passed": len(rows) - int(failing),
        "failed": int(failing),
        "incomplete": 0,
        "skills": rows,
        "execution": {"llm_enabled": False},
        "selection": {"mode": "changed" if case == "changed-one" else "all", "discovered": 2, "selected": paths},
    }
    (directory / "summary.json").write_text(json.dumps(summary))
    for filename in ("summary.md", "details.md", "index.html", "pr-comment.md"):
        (directory / filename).write_text("Test summary")
    return summary


@pytest.mark.parametrize("case", ["all-pass", "invalid-name", "changed-one"])
def test_acceptance_checks_cover_each_real_scan_scenario(tmp_path, case):
    _scan_case(tmp_path, case)
    assert smoke.verify_case(tmp_path, case, int(case == "invalid-name"))["verified"]


@pytest.mark.parametrize("case", ["all-pass", "invalid-name", "changed-one"])
@pytest.mark.parametrize(
    "damage",
    [
        "missing-sarif",
        "incomplete-scan",
        "missing-validators",
        "wrong-selection",
        "wrong-verdict",
        "wrong-exit",
        "process-exit",
        "summary-verdict",
        "summary-passed",
        "summary-failed",
        "summary-incomplete",
        "selection-discovered",
        "report-dir",
        "row-incomplete",
        "report-incomplete",
    ],
)
def test_acceptance_rejects_incomplete_or_contradictory_scan_evidence(tmp_path, case, damage):
    summary = _scan_case(tmp_path, case)
    child = tmp_path / "skill-001"
    path = child / "skillevaluator-tier1.json"
    data = json.loads(path.read_text())
    process_exit = int(case == "invalid-name")
    if damage == "missing-sarif":
        (child / "skillevaluator-tier1.sarif.json").unlink()
    elif damage == "incomplete-scan":
        data["results"][0]["incomplete_scans"] = ["gitleaks"]
    elif damage == "missing-validators":
        data["results"] = data["results"][:9]
    elif damage == "wrong-selection":
        summary["skills"][0]["path"] = "skills/other"
    elif damage == "wrong-verdict":
        data["overall_passed"] = not data["overall_passed"]
    elif damage == "wrong-exit":
        summary["skills"][0]["exit_code"] = 9
    elif damage == "process-exit":
        process_exit = 1 - process_exit
    elif damage == "summary-verdict":
        summary["overall_status"] = "passed" if case == "invalid-name" else "failed"
    elif damage.startswith("summary-"):
        summary[damage.removeprefix("summary-")] += 1
    elif damage == "selection-discovered":
        if case != "changed-one":
            pytest.skip("Only changed-one checks discovered selection metadata")
        summary["selection"]["discovered"] = 3
    elif damage == "report-dir":
        child.rename(tmp_path / "skill-999")
        summary["skills"][0]["report_dir"] = "skill-999"
        path = tmp_path / "skill-999/skillevaluator-tier1.json"
    elif damage == "row-incomplete":
        summary["skills"][0]["incomplete_scans"] = ["semgrep"]
    elif damage == "report-incomplete":
        data["incomplete_scans"] = ["semgrep"]
    path.write_text(json.dumps(data))
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    with pytest.raises(smoke.SmokeError):
        smoke.verify_case(tmp_path, case, process_exit)


def test_failure_fixture_must_retain_expected_invalid_name_finding(tmp_path):
    _scan_case(tmp_path, "invalid-name")
    path = tmp_path / "skill-002/skillevaluator-tier1.json"
    data = json.loads(path.read_text())
    data["results"][0]["findings"] = []
    path.write_text(json.dumps(data))
    with pytest.raises(smoke.SmokeError, match="invalid-name"):
        smoke.verify_case(tmp_path, "invalid-name", 1)
