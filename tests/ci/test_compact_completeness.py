# SPDX-License-Identifier: Apache-2.0
"""Compact feedback must preserve canonical Tier 3 execution uncertainty."""

import io
import json
import zipfile

from ci.feedback import read_results
from ci.summary_metrics import summarize_results
from ci.tier_manifest import tier3_execution_complete

from skillevaluator.models import ValidationResult
from skillevaluator.reporting.json_reporter import JSONReporter
from skillevaluator.reporting.retained import retained_results
from skillevaluator.validators.hygiene import HygieneValidator


def test_failed_unpaired_tier3_evidence_does_not_claim_a_complete_issue_total():
    result = ValidationResult(
        validator_name="AGENT_EVAL",
        metadata={
            "gating": {"tier": 3, "blocking": False},
            "agent_eval": {
                "mode": "with_skill",
                "execution_status": "succeeded",
                "verdict": "fail",
                "summary": {
                    "expected_attempts": 2,
                    "scored_attempts": 2,
                    "execution_status": "succeeded",
                    "verdict": "fail",
                    "overall_score": 0.2,
                    "overall_lift": 0.1,
                },
            },
        },
    )
    result.fail_gate("Score is below the required minimum")
    native = json.loads(JSONReporter().render_all([result]))
    assert tier3_execution_complete(native) is False
    json_path = "tier3/skill-001/skillevaluator-tier3.json"
    html_path = json_path.removesuffix(".json") + ".html"
    manifest = {
        "schema_version": 1,
        "report_contract": "skillevaluator-ci-v1",
        "commit": "a" * 40,
        "repository": "acme/skills",
        "platform": "github",
        "run_id": "123",
        "run_attempt": "1",
        "environment": "docker",
        "expected_tiers": [3],
        "tiers": [
            {
                "tier": 3,
                "required": False,
                "status": "failed",
                "skills": [
                    {
                        "path": "skills/demo",
                        "status": "failed",
                        "report_json": json_path,
                        "report_html": html_path,
                    }
                ],
            }
        ],
    }
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as retained:
        retained.writestr("ci-results.json", json.dumps(manifest))
        retained.writestr(json_path, json.dumps(native))
        retained.writestr(html_path, "<p>Retained native evidence</p>")

    validated = read_results(archive.getvalue())
    metrics = validated["tiers"][0]["skills"][0]["metrics"]

    assert metrics["issue_count"] == 1
    assert metrics["issue_count_known"] is False
    assert "score" not in metrics
    assert "skill_lift" not in metrics


def test_current_tier1_legacy_error_and_warning_count_as_issues(tmp_path):
    (tmp_path / "SKILL.md").write_text("# Demo\n\nRead the [guide](missing.md).\n", encoding="utf-8")
    validator = HygieneValidator()
    result = validator.validate(tmp_path)
    result.validator_name = validator.name
    result.metadata["gating"] = {"tier": 1, "blocking": True}
    assert any("Dead link" in error for error in result.errors)
    assert any("test-file candidates" in warning for warning in result.warnings)
    assert not result.findings
    assert not result.is_incomplete
    native = json.loads(JSONReporter().render_all([result]))
    validated = retained_results(native, allowed_tiers={1})

    metrics = summarize_results(validated, tier=1, native=native)

    assert metrics["issue_count"] == 2
    assert metrics["issue_count_known"] is True
