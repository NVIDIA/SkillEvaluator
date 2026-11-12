# SPDX-License-Identifier: Apache-2.0
"""Retained Tier 1 scope cannot turn incomplete publication evidence into PASS."""

import json

import pytest

from skillevaluator.models import ValidationResult
from skillevaluator.publication_evidence import stamp_publication_evidence, tier1_required_checks
from skillevaluator.reporting.base import assess_publication
from skillevaluator.reporting.json_reporter import JSONReporter
from skillevaluator.reporting.retained import RetainedReportError, retained_results


def _results(*, skill_like=True, omitted=None, unscanned=None):
    required = tier1_required_checks(skill_like=skill_like)
    results = []
    for check in required:
        if check == omitted:
            continue
        result = ValidationResult(validator_name=check.upper())
        result.add_success(check, "Validation completed")
        stamp_publication_evidence([result], tier=1, check_id=check)
        result.metadata.update(
            publication_target={
                "skill_name": "demo",
                "skill_digest": "sha256:" + "a" * 64,
                "skill_digest_algorithm": "skill-evaluator-source-tree/2",
            },
            benchmark_policy={"tier2_required": False, "tier3_required": False},
            tier1_required_checks=list(required),
            tier1_unscanned_paths=list(unscanned or []),
        )
        results.append(result)
    return results


def _emit(results):
    return json.loads(JSONReporter(include_timestamp=False, expected_skill_name="demo").render_all(results))


@pytest.mark.parametrize(
    ("options", "status"),
    [
        ({}, "pass"),
        ({"skill_like": False}, "pass"),
        ({"omitted": "security"}, "incomplete"),
        ({"unscanned": ["references/results"]}, "incomplete"),
    ],
    ids=["complete-skill", "complete-core", "omitted-required-check", "unscanned-content"],
)
def test_scope_roundtrip_preserves_publication_verdict(options, status):
    results = _results(**options)
    assessment = assess_publication(results, expected_skill_name="demo")
    assert assessment.status == status

    payload = _emit(results)
    restored = retained_results(payload)

    assert assess_publication(restored, expected_skill_name="demo") == assessment
    assert payload["publication_status"] == status
    assert _emit(restored)["publication"] == payload["publication"]
    for original, row, retained in zip(results, payload["results"], restored, strict=True):
        for key in ("tier1_required_checks", "tier1_unscanned_paths"):
            assert row[key] == original.metadata[key]
            assert retained.metadata[key] == original.metadata[key]
            assert retained.metadata[key] is not row[key]


def test_contradictory_required_lineups_remain_incomplete_after_roundtrip():
    results = _results(skill_like=False)
    results[0].metadata["tier1_required_checks"] = tier1_required_checks(skill_like=True)
    assessment = assess_publication(results, expected_skill_name="demo")
    assert assessment.status == "incomplete"
    assert "Tier 1 did not run every required check: lint, quality, version." in assessment.reasons
    assert assess_publication(retained_results(_emit(results)), expected_skill_name="demo") == assessment


@pytest.mark.parametrize("key", ["tier1_required_checks", "tier1_unscanned_paths"])
@pytest.mark.parametrize(
    "value",
    [None, "schema", {"schema": True}, [True], [1], [None], [""], ["x" * 4097], ["x"] * 20_001],
    ids=["null", "string", "object", "boolean-item", "number-item", "null-item", "empty-item", "long-item", "many"],
)
def test_retained_scope_rejects_malformed_or_unbounded_lists(key, value):
    payload = _emit(_results())
    payload["results"][0][key] = value
    with pytest.raises(RetainedReportError, match=f"{key}|Metadata list too large"):
        retained_results(payload)


@pytest.mark.parametrize("key", ["tier1_required_checks", "tier1_unscanned_paths"])
@pytest.mark.parametrize("value", [None, "schema", [1], [""], ["x" * 4097], ["x"] * 20_001])
def test_json_scope_rejects_malformed_declarations_instead_of_dropping_them(key, value):
    results = _results()
    results[0].metadata[key] = value
    with pytest.raises(ValueError, match=key):
        _emit(results)


def test_legacy_reports_without_scope_fields_remain_readable():
    results = _results()
    for result in results:
        result.metadata.pop("tier1_required_checks")
        result.metadata.pop("tier1_unscanned_paths")
    payload = _emit(results)
    restored = retained_results(payload)
    assert assess_publication(restored, expected_skill_name="demo").status == "pass"
    assert _emit(restored)["results"] == payload["results"]
