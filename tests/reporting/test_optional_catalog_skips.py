# SPDX-License-Identifier: Apache-2.0
"""Optional catalog absence cannot fabricate or block executed semantic evidence."""

import json

import pytest

from skillevaluator.models import Finding, Severity, ValidationResult
from skillevaluator.publication_evidence import stamp_publication_evidence, tier1_required_checks
from skillevaluator.reporting.base import assess_publication
from skillevaluator.reporting.benchmark import BenchmarkReporter
from skillevaluator.reporting.json_reporter import JSONReporter
from skillevaluator.reporting.retained import RetainedReportError, retained_results


def _results(*, context="complete", check_id="context-optimization"):
    results = []
    required = tier1_required_checks(skill_like=True)
    for check in required:
        result = ValidationResult(validator_name=check.upper())
        result.add_success(check, "Validation completed")
        stamp_publication_evidence([result], tier=1, check_id=check)
        result.metadata["tier1_required_checks"] = list(required)
        results.append(result)

    dependency = ValidationResult(validator_name="Plugin Dependency Deduplication")
    dependency.add_success("plugin_dep_dedup", "No duplicate dependency references found")
    semantic = []
    if context != "absent":
        result = ValidationResult(validator_name="Context Deduplication")
        if context == "complete":
            result.add_success("context_dedup", "Analyzed local skill content")
        elif context == "skipped":
            result.metadata.update(skipped=True, optional_skip=True)
            result.add_warning("Context comparison disabled")
        elif context == "incomplete":
            result.add_success("files_collected", "Read local files")
            result.mark_scan_incomplete("context-analysis")
        semantic.append(result)

    catalogs = []
    for name in ("Inter-Skill Deduplication", "Inter-Plugin Deduplication"):
        result = ValidationResult(validator_name=name)
        result.metadata.update(skipped=True, optional_skip=True)
        result.add_warning("No local catalog supplied")
        catalogs.append(result)
    tier2 = [dependency, *semantic, *catalogs]
    stamp_publication_evidence(tier2, tier=2, check_id=check_id)
    results.extend(tier2)
    for result in results:
        result.metadata.update(
            publication_target={
                "skill_name": "demo",
                "skill_digest": "sha256:" + "a" * 64,
                "skill_digest_algorithm": "skill-evaluator-source-tree/2",
            },
            benchmark_policy={"tier2_required": True, "tier3_required": False},
        )
    return results


def _emit(results):
    return json.loads(JSONReporter(include_timestamp=False, expected_skill_name="demo").render_all(results))


def test_optional_catalog_skips_allow_completed_context_and_survive_retention():
    results = _results()
    assessment = assess_publication(results, expected_skill_name="demo")
    assert assessment.status == "pass"
    payload = _emit(results)
    assert payload["publication_status"] == "pass"
    assert [row["optional_skip"] for row in payload["results"][-2:]] == [True, True]
    assert assess_publication(retained_results(payload), expected_skill_name="demo") == assessment


@pytest.mark.parametrize(
    ("context", "overall", "tier_status"),
    [
        ("complete", "PASS", "PASSED WITH OBSERVATIONS"),
        ("absent", "INCOMPLETE", "INCOMPLETE"),
        ("empty", "INCOMPLETE", "INCOMPLETE"),
        ("skipped", "INCOMPLETE", "INCOMPLETE"),
        ("incomplete", "INCOMPLETE", "INCOMPLETE"),
    ],
)
def test_benchmark_tier2_table_agrees_with_optional_catalog_publication(context, overall, tier_status):
    rendered = BenchmarkReporter(skill_name="demo", include_timestamp=False).render_all(_results(context=context))

    assert f"Overall verdict: {overall}" in rendered
    assert f"- Decision: **{overall}**" in rendered
    assert f"| Tier 2 | Semantic deduplication | **{tier_status}** |" in rendered
    if context == "complete":
        assert "No local catalog supplied" in rendered


def test_benchmark_tier2_table_keeps_required_catalog_skip_incomplete():
    results = _results()
    results[-1].metadata["optional_skip"] = False
    rendered = BenchmarkReporter(skill_name="demo", include_timestamp=False).render_all(results)

    assert "Overall verdict: INCOMPLETE" in rendered
    assert "| Tier 2 | Semantic deduplication | **INCOMPLETE** |" in rendered


@pytest.mark.parametrize("context", ["absent", "empty", "skipped", "incomplete"])
def test_optional_catalog_skips_cannot_replace_context_analysis(context):
    results = _results(context=context)
    assert assess_publication(results, expected_skill_name="demo").status == "incomplete"


def test_all_optional_tier2_skips_cannot_certify_required_tier2():
    results = _results(context="absent")
    results = [result for result in results if result.validator_name != "Plugin Dependency Deduplication"]
    assert assess_publication(results, expected_skill_name="demo").status == "incomplete"
    rendered = BenchmarkReporter(skill_name="demo", include_timestamp=False).render_all(results)
    assert "| Tier 2 | Semantic deduplication | **INCOMPLETE** |" in rendered


@pytest.mark.parametrize("value", [False, None, 1, "true"])
def test_catalog_skip_requires_explicit_optional_boolean(value):
    results = _results()
    results[-1].metadata["optional_skip"] = value
    assert assess_publication(results, expected_skill_name="demo").status == "incomplete"


@pytest.mark.parametrize(
    "marker", [None, {"schema_version": 1, "producer": "external", "tier": 2, "check_id": "context-optimization"}]
)
def test_optional_catalog_skip_requires_recognized_producer(marker):
    results = _results()
    results[-1].metadata["publication_evidence"] = marker
    assert assess_publication(results, expected_skill_name="demo").status == "incomplete"


def test_optional_catalog_skip_requires_matching_semantic_check():
    results = _results()
    stamp_publication_evidence([results[-1]], tier=2, check_id="similarity")
    assert assess_publication(results, expected_skill_name="demo").status == "incomplete"


@pytest.mark.parametrize("tier", [1, 3])
def test_optional_catalog_skip_cannot_override_contradictory_gate_tier(tier):
    results = _results()
    results[-1].metadata["gating"] = {"tier": tier, "blocking": False}
    assert assess_publication(results, expected_skill_name="demo").status == "incomplete"


def test_optional_catalog_skip_cannot_waive_required_tier1():
    results = _results()
    results[0].success_details.clear()
    results[0].metadata.update(skipped=True, optional_skip=True)
    assessment = assess_publication(results, expected_skill_name="demo")
    assert assessment.status == "incomplete"
    assert "A publication-required validator was skipped." in assessment.reasons


def test_optional_catalog_skip_cannot_waive_required_tier3():
    results = _results()
    for result in results:
        result.metadata["benchmark_policy"]["tier3_required"] = True
    assert assess_publication(results, expected_skill_name="demo").status == "incomplete"


def test_optional_catalog_skip_cannot_relabel_agent_eval():
    results = _results()
    results[-1].metadata["agent_eval"] = {"execution_status": "skipped", "verdict": "skipped"}
    assert assess_publication(results, expected_skill_name="demo").status == "incomplete"


@pytest.mark.parametrize("subject", ["catalog", "context"])
@pytest.mark.parametrize("diagnostic", ["error", "high-finding", "confirmed-failure", "security-failure"])
def test_optional_catalog_skip_cannot_hide_contradictory_failure(subject, diagnostic):
    results = _results()
    result = results[-1] if subject == "catalog" else results[-3]
    if diagnostic == "error":
        result.add_error("Semantic analysis failed")
    elif diagnostic == "high-finding":
        result.findings.append(Finding("CONTENT_DEDUP", Severity.HIGH, "unsafe", "Unsafe input", "SKILL.md"))
    else:
        result.metadata[diagnostic.replace("-", "_")] = True
    result.passed = True
    assert assess_publication(results, expected_skill_name="demo").status == "incomplete"
    rendered = BenchmarkReporter(skill_name="demo", include_timestamp=False).render_all(results)
    assert "Overall verdict: INCOMPLETE" in rendered
    assert "| Tier 2 | Semantic deduplication | **INCOMPLETE** |" in rendered


@pytest.mark.parametrize("value", [None, 0, 1, "true", [], {}])
def test_retained_optional_skip_rejects_non_boolean_values(value):
    payload = _emit(_results())
    payload["results"][-1]["optional_skip"] = value
    with pytest.raises(RetainedReportError, match="optional_skip"):
        retained_results(payload)


def test_executed_catalog_comparison_can_cover_other_inapplicable_catalog_type():
    results = _results(context="absent", check_id="similarity")
    compared = results[-2]
    compared.metadata.pop("skipped")
    compared.metadata.pop("optional_skip")
    compared.add_success("skill_catalog_comparison", "Compared local catalog skills")
    assert assess_publication(results, expected_skill_name="demo").status == "pass"
