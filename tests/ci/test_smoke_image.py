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
