# SPDX-License-Identifier: Apache-2.0
"""The aggregate gate and its display depend on typed retained outcomes."""

import copy
import json

import pytest

from ci import tier_manifest


def manifest():
    return {
        "schema_version": 1,
        "report_contract": "skillevaluator-ci-v1",
        "commit": "a" * 40,
        "base_commit": None,
        "repository": "acme/skills",
        "platform": "github",
        "run_id": "123",
        "run_attempt": "1",
        "environment": "docker",
        "producer": {"workflow": "trusted"},
        "tiers": [
            {
                "tier": tier,
                "status": "passed",
                "required": tier == 1,
                "skills": [
                    {
                        "path": "skills/demo",
                        "status": "passed",
                        "report_json": f"tier{tier}/skill-001/skillevaluator-tier{tier}.json",
                        "report_html": f"tier{tier}/skill-001/skillevaluator-tier{tier}.html",
                    }
                ],
            }
            for tier in (1, 2, 3)
        ],
    }


def test_advisory_failure_keeps_own_status_without_failing_required_gate():
    payload = manifest()
    payload["tiers"][2].update(status="failed", metrics={"score": 0.8, "skill_lift": -0.1, "coverage": "1/1"})
    payload["tiers"][2]["skills"][0]["status"] = "failed"
    validated = tier_manifest.validate_manifest(payload)
    assert validated["overall_passed"] is True
    assert validated["tiers"][2]["status"] == "failed"
    assert validated["producer"] == payload["producer"]
    validated["producer"]["workflow"] = "changed"
    assert payload["producer"]["workflow"] == "trusted"
    markdown = tier_manifest.render_markdown(payload, "https://example.test/artifacts/123")
    assert "Tier 1 | passed | Required" in markdown
    assert "Tier 2 | passed | Advisory" in markdown
    assert "Tier 3 | failed | Advisory" in markdown
    assert "score: 0.8" in markdown and "skill lift: -0.1" in markdown and "coverage: 1/1" in markdown
    assert "https://example.test/artifacts/123/tier3/skill-001/skillevaluator-tier3.html" in markdown


@pytest.mark.parametrize("tier,status", [(1, "failed"), (2, "incomplete"), (2, "skipped"), (3, "needs_action")])
def test_required_tier_nonpass_blocks_overall(tier, status):
    payload = manifest()
    record = payload["tiers"][tier - 1]
    record.update(required=True, status=status)
    record["skills"][0]["status"] = status
    assert tier_manifest.validate_manifest(payload)["overall_passed"] is False


def test_missing_required_tier_and_empty_pass_cannot_claim_success():
    payload = manifest()
    payload["tiers"] = payload["tiers"][1:]
    assert tier_manifest.validate_manifest(payload)["overall_passed"] is False
    markdown = tier_manifest.render_markdown(payload)
    assert "Required gate: **Not passed**" in markdown
    assert "Tier 1 | not_run | Required" in markdown
    payload = manifest()
    payload["tiers"][0]["skills"] = []
    with pytest.raises(ValueError, match=r"empty|nonempty"):
        tier_manifest.validate_manifest(payload)


@pytest.mark.parametrize("tier", [2, 3])
@pytest.mark.parametrize("status", ["failed", "incomplete", "needs_action", "not_run", "neutral", "skipped"])
def test_paid_advisory_outcomes_do_not_change_required_gate(tier, status):
    payload = manifest()
    payload["tiers"][tier - 1]["status"] = status
    payload["tiers"][tier - 1]["skills"][0]["status"] = status
    result = tier_manifest.validate_manifest(payload)
    assert result["overall_passed"] is True
    assert result["overall_status"] == "passed"
    assert result["tiers"][tier - 1]["status"] == status


def test_missing_paid_advisory_results_do_not_block_tier1():
    payload = manifest()
    payload["tiers"] = payload["tiers"][:1]
    result = tier_manifest.validate_manifest(payload)
    assert result["overall_passed"] is True
    assert result["overall_status"] == "passed"
    markdown = tier_manifest.render_markdown(result)
    assert "Tier 2 | not_run | Advisory" in markdown
    assert "Tier 3 | not_run | Advisory" in markdown


@pytest.mark.parametrize("required_paid", [[], [2], [3], [2, 3]])
def test_report_policy_text_matches_current_or_historical_declared_flags(tmp_path, required_paid):
    payload = manifest()
    for tier in required_paid:
        payload["tiers"][tier - 1]["required"] = True
    required = "Required tiers: " + ", ".join(f"Tier {tier}" for tier in [1, *required_paid]) + "."
    advisory = "Advisory tiers: " + ", ".join(f"Tier {tier}" for tier in (2, 3) if tier not in required_paid) + "."
    if len(required_paid) == 2:
        advisory = "Advisory tiers: none."
    tier_manifest.write_reports(tmp_path, payload)
    for name in ("summary.md", "index.html"):
        rendered = (tmp_path / name).read_text()
        assert required in rendered and advisory in rendered


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.update(overall_passed=False),
        lambda p: p.update(run_id="0"),
        lambda p: p.update(commit="branch"),
        lambda p: p["tiers"].append(copy.deepcopy(p["tiers"][0])),
        lambda p: p["tiers"][0].update(tier=True),
        lambda p: p["tiers"][0].update(required=False),
        lambda p: p["tiers"][0]["skills"][0].update(status="failed"),
        lambda p: p["tiers"][0]["skills"][0].update(report_html="../secret.html"),
        lambda p: p["tiers"][0]["skills"][0].update(report_json="https://evil.test/x"),
        lambda p: p["tiers"][0]["skills"][0].update(path="skills/demo\nforged"),
        lambda p: p.update(producer={"number": float("nan")}),
    ],
)
def test_rejects_contradictory_or_unsafe_manifest(mutate):
    payload = manifest()
    mutate(payload)
    with pytest.raises(ValueError):
        tier_manifest.validate_manifest(payload)


def test_write_reports_escapes_content_and_leaves_native_evidence_untouched(tmp_path):
    payload = manifest()
    payload["tiers"][1]["metrics"] = {"findings": "<img src=x onerror=alert(1)> | fake"}
    path = tmp_path / "tier2/skill-001"
    path.mkdir(parents=True)
    (path / "skillevaluator-tier2.html").write_text("native evidence")
    tier_manifest.write_reports(tmp_path, payload)
    assert json.loads((tmp_path / "ci-results.json").read_text())["overall_passed"] is True
    html = (tmp_path / "index.html").read_text()
    assert "Content-Security-Policy" in html
    assert "&lt;img src=x" in html and "<img src=x" not in html
    assert 'href="tier2/skill-001/skillevaluator-tier2.html"' in html
    assert "Tier 3" in html and "Advisory" in html
    assert "\\| fake" in (tmp_path / "summary.md").read_text()
    assert (path / "skillevaluator-tier2.html").read_text() == "native evidence"


def test_write_reports_rejects_symlink_output(tmp_path):
    external = tmp_path / "external"
    external.write_text("keep")
    output = tmp_path / "reports"
    output.mkdir()
    (output / "index.html").symlink_to(external)
    with pytest.raises(ValueError):
        tier_manifest.write_reports(output, manifest())
    assert external.read_text() == "keep"


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p.update(platform=[]),
        lambda p: p.update(environment={}),
        lambda p: p["tiers"][0].update(status=[]),
        lambda p: p["tiers"][0]["skills"][0].update(status={}),
    ],
)
def test_untyped_discriminators_raise_validation_errors(mutate):
    payload = manifest()
    mutate(payload)
    with pytest.raises(ValueError):
        tier_manifest.validate_manifest(payload)


def test_explicit_expected_tiers_supports_a_single_tier_producer():
    payload = manifest()
    payload["expected_tiers"] = [1]
    payload["tiers"] = payload["tiers"][:1]
    result = tier_manifest.validate_manifest(payload)
    assert result["overall_passed"] is True
    assert result["overall_status"] == "passed"
    assert "Tier 2 | not_run" in tier_manifest.render_markdown(result)


def test_explicit_empty_selection_satisfies_gate_with_skipped_headline():
    payload = manifest()
    payload["expected_tiers"] = [1]
    payload["tiers"] = [
        {
            "tier": 1,
            "required": True,
            "status": "skipped",
            "skills": [],
            "reason": "No selected skills changed; evaluation did not run.",
        }
    ]
    result = tier_manifest.validate_manifest(payload)
    assert result["overall_passed"] is True
    assert result["overall_status"] == "skipped"
    markdown = tier_manifest.render_markdown(result)
    assert "**Skipped**" in markdown and "Required gate: **Passed**" not in markdown


def test_neutral_advisory_result_keeps_its_outcome():
    payload = manifest()
    payload["tiers"][2]["status"] = "neutral"
    payload["tiers"][2]["skills"][0]["status"] = "neutral"
    result = tier_manifest.validate_manifest(payload)
    assert result["overall_passed"] is True
    assert "Tier 3 | neutral | Advisory" in tier_manifest.render_markdown(result)
