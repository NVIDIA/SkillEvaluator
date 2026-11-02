# SPDX-License-Identifier: Apache-2.0
"""Roundtrip and trust-boundary tests for retained native HTML rendering."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from skillevaluator.models import Finding, SuccessDetail, ValidationResult
from skillevaluator.reporting.html import HTMLReporter
from skillevaluator.reporting.json_reporter import JSONReporter
from skillevaluator.reporting.retained import RetainedReportError, render_retained_report, retained_results


@pytest.fixture
def payload():
    result = ValidationResult(validator_name="Schema & Repository Governance", validator_description="Schema checks")
    result.add_finding(
        Finding(
            category="SCHEMA",
            severity="high",
            check_name="name_consistency",
            message="Directory name differs from 'name'",
            file_path="/untrusted/skill/SKILL.md",
            suggestion="Match the directory and frontmatter names",
        )
    )
    result.success_details.append(SuccessDetail(check_name="manifest_exists", message="Found SKILL.md"))
    result.metadata.update(
        {
            "gating": {"tier": 1, "blocking": True},
            "workflow": {"tier": 1, "stages": {"llm": {"status": "skipped", "reason": "Disabled"}}},
        }
    )
    quality = ValidationResult(validator_name="QUALITY", validator_description="Quality dimensions")
    quality.metadata["quality_scores"] = {
        "skill_name": "skill",
        "overall_score": 71.8,
        "grade": "C",
        "dimensions": {"correctness": {"score": 71.8, "weight": 1.0, "issues_count": 0, "issues": []}},
        "metrics": {},
    }
    return json.loads(JSONReporter(include_timestamp=False).render_all([result, quality]))


def test_roundtrip_preserves_verdict_findings_quality_and_unknown_line(payload):
    rows = retained_results(payload)
    exported = json.loads(JSONReporter(include_timestamp=False).render_all(rows))
    assert exported["overall_passed"] is False
    assert exported["results"] == payload["results"]
    assert exported["quality_summary"] == payload["quality_summary"]
    assert rows[0].findings[0].line_number is None
    html = render_retained_report(payload, skill_name="skill")
    assert "71.8" in html and "name_consistency" in html
    assert "Match the directory and frontmatter names" in html


def test_roundtrip_preserves_upstream_quality_configuration_provenance(payload):
    provenance = {
        "source": "file",
        "file": "/workspace/quality.yaml",
        "digest": "sha256:" + "a" * 64,
        "min_score": 80,
    }
    payload["quality_config"] = provenance
    exported = json.loads(JSONReporter(include_timestamp=False).render_all(retained_results(payload)))
    assert exported["quality_config"] == provenance
    assert exported["results"] == payload["results"]


def test_render_explicit_identity_never_inspects_artifact_paths(payload):
    payload["results"][0]["findings"][0]["file_path"] = "/private/secret/../../etc/SKILL.md"
    # Package template loading remains permitted, but artifact path discovery must not occur.
    with patch.object(Path, "exists", side_effect=AssertionError("Artifact path IO")):
        rendered = render_retained_report(payload, skill_name="skill")
    assert "name_consistency" in rendered


def test_incomplete_remains_incomplete():
    result = ValidationResult(validator_name="Security", metadata={"incomplete_scans": ["semgrep"]})
    result.add_error("Scanner execution did not complete")
    payload = json.loads(JSONReporter(include_timestamp=False).render_all([result]))
    rows = retained_results(payload)
    assert rows[0].is_incomplete
    assert not json.loads(JSONReporter(include_timestamp=False).render_all(rows))["overall_passed"]
    assert "incomplete" in render_retained_report(payload, skill_name="skill").lower()


@pytest.mark.parametrize("same_validator", [False, True])
def test_confirmed_failure_precedes_incompleteness_in_retained_roundtrip(payload, same_validator):
    rows = retained_results(payload)
    result = rows[0] if same_validator else rows[1]
    result.metadata["incomplete_scans"] = ["semgrep"]
    result.add_error("Scanner execution did not complete")
    emitted = json.loads(JSONReporter(include_timestamp=False).render_all(rows))
    assert emitted["overall_status"] == "failed"
    assert emitted["incomplete_scans"] == ["semgrep"]
    reconstructed = retained_results(emitted)
    assert json.loads(JSONReporter(include_timestamp=False).render_all(reconstructed))["results"] == emitted["results"]


@pytest.mark.parametrize("status", ["passed", "incomplete"])
def test_retained_evidence_rejects_status_that_hides_confirmed_failure(payload, status):
    payload["results"][0]["incomplete_scans"] = ["semgrep"]
    payload["results"][0]["status"] = status
    with pytest.raises(RetainedReportError, match="Result status contradicts evidence"):
        retained_results(payload)


def test_confirmed_legacy_failure_survives_retained_roundtrip():
    result = ValidationResult(validator_name="Code Integrity & Hygiene")
    result.add_error("Banned dependency found in one audited member")
    result.metadata["confirmed_failure"] = True
    result.mark_scan_incomplete("another-member")
    emitted = json.loads(JSONReporter(include_timestamp=False).render_all([result]))
    assert emitted["results"][0]["confirmed_failure"] is True
    reconstructed = retained_results(emitted)
    assert reconstructed[0].metadata["confirmed_failure"] is True
    assert json.loads(JSONReporter(include_timestamp=False).render_all(reconstructed))["overall_status"] == "failed"


@pytest.mark.parametrize("value", [1, "true", [], None])
def test_retained_confirmed_failure_marker_must_be_boolean(payload, value):
    payload["results"][0]["confirmed_failure"] = value
    with pytest.raises(RetainedReportError, match="confirmed_failure must be boolean"):
        retained_results(payload)


def test_retained_confirmed_failure_marker_requires_failing_diagnostics():
    result = ValidationResult(validator_name="Security", passed=False)
    emitted = json.loads(JSONReporter(include_timestamp=False).render_all([result]))
    emitted["results"][0]["confirmed_failure"] = True
    with pytest.raises(RetainedReportError, match="Confirmed failure requires failing diagnostics"):
        retained_results(emitted)


def test_retained_finding_failure_cannot_claim_a_passing_result(payload):
    payload["results"][0]["passed"] = True
    payload["overall_passed"] = True
    payload["overall_status"] = "passed"
    with pytest.raises(RetainedReportError, match="Result verdict contradicts evidence"):
        retained_results(payload)


def test_advisory_skip_keeps_required_gate_but_is_outside_retained_tier1_contract():
    from skillevaluator.evaluation.tier3_report import advisory_skip_result
    from skillevaluator.reporting.base import is_cleanly_skipped, passes_required_gate

    result = advisory_skip_result("Live evaluation runtime unavailable", skill_name="skill")
    assert result.passed is False
    assert is_cleanly_skipped(result) and passes_required_gate(result)
    emitted = json.loads(JSONReporter(include_timestamp=False).render_all([result]))
    assert emitted["overall_passed"] is True
    assert emitted["results"][0]["status"] == "skipped"
    with pytest.raises(RetainedReportError, match="Only retained Tier 1 reports are supported"):
        retained_results(emitted)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda p: p["results"][0].update(passed="false"),
        lambda p: p["results"][0]["findings"][0].update(severity="invented"),
        lambda p: p["results"][0]["summary"].update(errors=True),
        lambda p: p["results"][0].update(incomplete_scans="semgrep"),
        lambda p: p["results"][0].update(workflow=[]),
        lambda p: p["results"][0].update(gating={"tier": "1", "blocking": True}),
        lambda p: p["results"][1]["quality"].update(overall_score="71.8"),
        lambda p: p.update(overall_passed=True),
        lambda p: p.update(results=[]),
    ],
)
def test_malformed_evidence_rejected(payload, mutate):
    mutate(payload)
    with pytest.raises(RetainedReportError):
        retained_results(payload)


def test_hostile_text_safe_and_context_urls_rejected(payload):
    payload["results"][0]["findings"][0]["message"] = '</script><script>alert("x")</script>'
    rendered = render_retained_report(payload, skill_name="skill", context={"commit": "<img src=x onerror=alert(1)>"})
    assert '</script><script>alert("x")</script>' not in rendered
    assert "\\u003c/script\\u003e" in rendered
    for url in (
        "javascript:alert(1)",
        "http://example.com",
        "https://user:pass@example.com",
        "https://example.com\\evil",
    ):
        with pytest.raises(ValueError):
            render_retained_report(payload, skill_name="skill", context={"run_url": url})


def test_no_quality_identity_is_explicit(payload):
    payload["results"] = payload["results"][:1]
    payload["total_validators"] = 1
    html = render_retained_report(payload, skill_name="explicit-name")
    assert "explicit-name" in html
    assert "manifest_exists" in html


def test_ci_context_allowlist_and_https():
    assert HTMLReporter(ci_context={"run_url": "", "report_url": ""}).ci_context == {}
    reporter = HTMLReporter(
        ci_context={"run_url": "https://github.com/a/b/actions/runs/1", "commit": "a" * 40, "unrecognized": "ignored"}
    )
    assert reporter.ci_context == {"run_url": "https://github.com/a/b/actions/runs/1", "commit": "a" * 40}


def test_fixed_relative_ci_links_only():
    reporter = HTMLReporter(
        ci_context={"report_index_url": "../index.html", "canonical_json_url": "skillevaluator-tier1.json"}
    )
    assert reporter.ci_context["report_index_url"] == "../index.html"
    for key, value in (("run_url", "../index.html"), ("canonical_json_url", "../../secret")):
        with pytest.raises(ValueError):
            HTMLReporter(ci_context={key: value})


def test_validation_without_copies_matches_the_copying_rehydration(payload):
    payload["results"][0]["findings"][0]["metadata"] = {"rule": {"nested": ["evidence"]}}
    copied = retained_results(payload)
    shared = retained_results(payload, copy_metadata=False)
    assert shared == copied
    finding_metadata = payload["results"][0]["findings"][0]["metadata"]
    assert copied[0].findings[0].metadata is not finding_metadata
    assert shared[0].findings[0].metadata is finding_metadata
    payload["results"][0]["passed"] = "true"
    with pytest.raises(RetainedReportError, match="passed must be boolean"):
        retained_results(payload, copy_metadata=False)


def test_reporters_share_one_environment_and_compile_each_template_once(payload, monkeypatch):
    from skillevaluator.reporting import html as html_module

    loads = []
    original = html_module.PackageLoader.get_source

    def counting(self, environment, template):
        loads.append(template)
        return original(self, environment, template)

    monkeypatch.setattr(html_module.PackageLoader, "get_source", counting)
    HTMLReporter._create_environment.cache_clear()
    results = retained_results(payload)
    plain = HTMLReporter(include_timestamp=False)
    ci = HTMLReporter(include_timestamp=False, title="Other", ci_context={"report_index_url": "../index.html"})
    assert plain._env is ci._env
    first = plain.render_all(results)
    ci.render_all(results)
    assert plain.render_all(results) == first
    # Pages built outside report.html.j2, such as the CI index, reuse the same compiled logo.
    assert html_module.render_logo("one") != html_module.render_logo("two")
    assert sorted(loads) == ["logo.svg.j2", "report.html.j2", "report_ui.css.j2", "report_ui.js.j2"]


def test_shared_environment_renders_identically_from_many_threads(payload):
    from concurrent.futures import ThreadPoolExecutor

    HTMLReporter._create_environment.cache_clear()

    def render(index):
        context = {"report_index_url": "../index.html"} if index % 2 else None
        return render_retained_report(payload, skill_name="skill", context=context)

    expected = [render(0), render(1)]
    HTMLReporter._create_environment.cache_clear()
    with ThreadPoolExecutor(max_workers=8) as pool:
        documents = list(pool.map(render, range(32)))
    assert all(document == expected[index % 2] for index, document in enumerate(documents))


def test_only_standalone_reports_embed_the_export_json(payload):
    standalone = render_retained_report(payload, skill_name="skill")
    embedded = standalone.split('<script id="report-data" type="application/json">', 1)[1].split("</script>", 1)[0]
    assert json.loads(embedded)["results"][0]["validator_name"] == payload["results"][0]["validator"]
    assert 'onclick="exportJSON()"' in standalone
    hosted = render_retained_report(
        payload, skill_name="skill", context={"canonical_json_url": "skillevaluator-tier1.json"}
    )
    assert 'id="report-data"' not in hosted
    assert 'href="skillevaluator-tier1.json" download' in hosted
    assert 'onclick="exportJSON()"' not in hosted


def test_not_applicable_and_incomplete_presentation_preserves_gate():
    lint = ValidationResult(validator_name="SCRIPT_LINT", validator_description="Scripts")
    lint.success_details.append(SuccessDetail(check_name="lint", message="No scripts/ or tools/ directory found"))
    reporter = HTMLReporter(explicit_skill_name="skill")
    data = reporter._reorganize_single_skill([lint], "skill")
    assert data["skill"]["validators"]["SCRIPT_LINT"]["status"] == "not_applicable"
    assert lint.passed is True
    assert reporter._env.filters["execution_status"](lint) == "not_applicable"
    lint.metadata["incomplete_scans"] = ["scanner"]
    lint.errors.append("Scanner unavailable")
    lint.warnings.append("Scanner timeout after 30 seconds")
    data = reporter._reorganize_single_skill([lint], "skill")
    assert data["skill"]["status"] == "incomplete"
    assert data["skill"]["validators"]["SCRIPT_LINT"]["errors"] == ["Scanner unavailable"]
    assert data["skill"]["validators"]["SCRIPT_LINT"]["warnings"] == ["Scanner timeout after 30 seconds"]
    assert data["skill"]["validators"]["SCRIPT_LINT"]["incomplete_scans"] == ["scanner"]
    assert reporter._env.filters["execution_status"](lint) == "incomplete"
    assert data["skill"]["validators"]["SCRIPT_LINT"]["status"] == "incomplete"


@pytest.mark.parametrize(
    "check,message",
    [
        ("code_file_discovery", "No code files found - code risk analysis is not applicable"),
        ("dependency_file_discovery", "No dependency files found - vulnerability audit is not applicable"),
        ("version_optional", "Version is optional"),
    ],
)
def test_explicit_discovery_nonexecution_is_not_presented_as_pass(check, message):
    result = ValidationResult(validator_name="Discovery")
    result.success_details.append(SuccessDetail(check_name=check, message=message))
    reporter = HTMLReporter(explicit_skill_name="skill")
    assert reporter._env.filters["execution_status"](result) == "not_applicable"
    data = reporter._reorganize_single_skill([result], "skill")
    assert data["skill"]["validators"]["Discovery"]["details"][0]["status"] == "not_applicable"
    assert result.passed is True


def test_explicit_retained_paths_preserve_canonical_finding_location(payload):
    results = retained_results(payload)
    data = HTMLReporter(explicit_skill_name="skill")._reorganize_single_skill(results, "skill")
    finding = data["skill"]["validators"]["Schema & Repository Governance"]["findings"][0]
    assert finding["file_path"] == "/untrusted/skill/SKILL.md"
    assert finding["line_number"] is None


def test_display_path_preserves_canonical_publication_identity_and_bound_waivers():
    from skillevaluator.publication_evidence import stamp_publication_evidence, tier1_required_checks

    results = []
    for check in tier1_required_checks(skill_like=True):
        result = ValidationResult(validator_name=check)
        result.add_success(check, "Scanned")
        result.metadata.update(
            publication_target={
                "skill_name": "demo",
                "skill_digest": "sha256:" + "c" * 64,
                "skill_digest_algorithm": "skill-evaluator-source-tree/2",
            },
            benchmark_policy={"tier2_required": False, "tier3_required": False},
        )
        stamp_publication_evidence([result], tier=1, check_id=check)
        results.append(result)
    payload = json.loads(JSONReporter(include_timestamp=False, expected_skill_name="demo").render_all(results))
    assert payload["publication"]["status"] == "pass"
    rendered = render_retained_report(payload, skill_name="skills/demo", expected_skill_name="demo")
    assert 'data-publication-status="pass"' in rendered
    assert "skills/demo" in rendered
    mismatched = render_retained_report(payload, skill_name="skills/demo", expected_skill_name="different-skill")
    assert 'data-publication-status="incomplete"' in mismatched


def test_skillspector_version_survives_json_and_retained_html_roundtrip():
    result = ValidationResult(validator_name="Security")
    result.metadata["skillspector_version"] = "2.12.0"
    payload = json.loads(JSONReporter(include_timestamp=False).render_all([result]))
    assert payload["results"][0]["skillspector_version"] == "2.12.0"
    assert retained_results(payload)[0].metadata["skillspector_version"] == "2.12.0"
    assert "SkillSpector <strong>v2.12.0" in render_retained_report(payload, skill_name="demo")


@pytest.mark.parametrize(
    "check,message,metadata,expected",
    [
        ("code_file_discovery", "No code files found - code risk analysis is not applicable", {}, "NOT APPLICABLE"),
        ("lint", "No scripts/ or tools/ directory found", {}, "NOT APPLICABLE"),
        ("random-check", "Some rule is not applicable, other rules ran", {}, "PASS"),
        ("new-check", "No applicable target", {"not_applicable": True}, "NOT APPLICABLE"),
        ("lint", "No scripts/ or tools/ directory found", {"not_applicable": False}, "PASS"),
    ],
)
def test_html_and_ci_use_the_same_not_applicable_evidence(check, message, metadata, expected):
    from ci.report_views import _status

    result = ValidationResult(validator_name="Discovery")
    result.success_details.append(SuccessDetail(check_name=check, message=message, metadata=metadata))
    recorded = json.loads(JSONReporter(include_timestamp=False).render(result))
    html_status = HTMLReporter()._env.filters["execution_status"](result)
    assert html_status.upper().replace("_", " ") == expected
    assert _status(recorded) == expected
