# SPDX-License-Identifier: Apache-2.0
"""Public Tier 2 and 3 JSON can be re-rendered without inventing evidence."""

import json

import pytest

from skillevaluator.evaluation.tier3_report import advisory_skip_result, build_agent_eval_payload
from skillevaluator.models import Finding, ValidationResult
from skillevaluator.reporting.json_reporter import JSONReporter
from skillevaluator.reporting.retained import RetainedReportError, retained_results
from skillevaluator.tier3.harbor.metrics import DEFAULT_METRICS


def emit(result):
    return json.loads(JSONReporter(include_timestamp=False).render_all([result]))


def tier3_result():
    conditions = {
        arm: {"execution_status": "succeeded", "execution_errors": [], "expected_attempts": 1, "scored_attempts": 1}
        for arm in ("with_skill", "without_skill")
    }
    payload = build_agent_eval_payload(
        "demo",
        {
            "opencode": {
                "with_skill": dict.fromkeys(DEFAULT_METRICS, 0.8),
                "without_skill": dict.fromkeys(DEFAULT_METRICS, 0.4),
                "execution_status": "succeeded",
                "execution_errors": [],
                "expected_attempts": 2,
                "scored_attempts": 2,
                "conditions": conditions,
                "num_trials": 1,
                "num_trials_baseline": 1,
            }
        },
        dataset=[{"id": "case-a"}],
        attempt_policy={"max_attempts": 1, "stop_on_pass": False, "pass_threshold": 0.5},
        use_llm_judge=False,
    )
    return ValidationResult(
        validator_name="AGENT_EVAL", metadata={"agent_eval": payload, "gating": {"tier": 3, "blocking": False}}
    )


def test_tier2_retains_llm_and_work_limit_evidence():
    result = ValidationResult(
        validator_name="Context Optimization",
        metadata={
            "gating": {"tier": 2, "blocking": True},
            "llm_analysis": {"completed_clusters": 2, "failed_clusters": 0},
            "work_limit": {"reason": "cluster budget", "limit": 2},
        },
    )
    result.add_finding(
        Finding(
            category="CONTENT_DEDUP",
            severity="medium",
            check_name="redundancy",
            message="Repeated guidance",
            file_path="SKILL.md",
        )
    )
    payload = emit(result)
    restored = retained_results(payload)
    assert restored[0].metadata["work_limit"] == result.metadata["work_limit"]
    assert emit(restored[0])["results"] == payload["results"]


def test_tier3_roundtrip_keeps_canonical_agent_payload():
    payload = emit(tier3_result())
    restored = retained_results(payload)
    assert restored[0].metadata["agent_eval"] == payload["results"][0]["tier3"]
    assert emit(restored[0])["tier3"] == payload["tier3"]


def test_completed_native_tier3_quality_failure_retains_execution_success():
    from skillevaluator.evaluation.tier3_report import _validation_result_from_payload

    payload = tier3_result().metadata["agent_eval"]
    payload["verdict"] = payload["summary"]["verdict"] = "fail"
    native = _validation_result_from_payload(payload)
    native.metadata["gating"] = {"tier": 3, "blocking": False}
    assert native.passed is True
    document = emit(native)
    assert document["results"][0]["status"] == "failed"
    restored = retained_results(document)
    assert restored[0].passed is True
    assert emit(restored[0])["results"][0]["status"] == "failed"
    assert document["overall_passed"] is True


def test_advisory_incomplete_retains_gate_pass_and_result_failure():
    result = tier3_result()
    result.passed = False
    result.mark_scan_incomplete("live-agent")
    payload = emit(result)
    assert payload["overall_passed"] is True
    assert payload["results"][0]["status"] == "incomplete"
    assert retained_results(payload)[0].is_incomplete


def test_explicit_tier1_boundary_rejects_other_tiers():
    payload = emit(advisory_skip_result("Unavailable", skill_name="demo"))
    with pytest.raises(RetainedReportError, match=r"Tier 1|allowed"):
        retained_results(payload, allowed_tiers={1})


def test_gating_cannot_disguise_tier3_as_tier1():
    payload = emit(tier3_result())
    payload["results"][0]["gating"]["tier"] = 1
    with pytest.raises(RetainedReportError, match="gating"):
        retained_results(payload)


def test_native_tier3_uses_trusted_render_for_dynamic_csp():
    from ci.native_reports import render_native_retained_report

    result = tier3_result()
    result.metadata["agent_eval"]["suggestions"] = ["</script><img src=x onerror=alert(1)>"]
    document = render_native_retained_report(emit(result), skill_name="skills/demo", expected_skill_name="demo")
    assert "Content-Security-Policy" in document and "sha256-" in document
    assert "showTier3SubTab" in document
    assert "<img src=x onerror=alert(1)>" not in document
    assert "<script src=" not in document
    assert "unsafe-eval" not in document


@pytest.mark.parametrize(
    "key,value", [("summary", []), ("agents", []), ("dataset", {}), ("dimensions", {}), ("execution_errors", {})]
)
def test_tier3_rejects_malformed_native_containers(key, value):
    payload = emit(tier3_result())
    payload["results"][0]["tier3"][key] = value
    with pytest.raises(RetainedReportError, match="tier3"):
        retained_results(payload)


def test_native_tier3_links_reports_root_and_its_actual_json_filename():
    from ci.native_reports import render_native_retained_report

    document = render_native_retained_report(
        emit(tier3_result()),
        skill_name="skills/demo",
        context={"report_index_url": "../../index.html", "canonical_json_url": "skillevaluator-tier3.json"},
    )
    assert 'href="../../index.html"' in document
    assert 'href="skillevaluator-tier3.json"' in document
    assert 'href="skillevaluator-tier1.json"' not in document


def test_public_tier3_applicability_survives_retention():
    result = tier3_result()
    result.metadata["tier3_applicability"] = {"status": "not_applicable", "reason": "No live cases"}
    payload = emit(result)
    assert payload["tier3_applicability"] == result.metadata["tier3_applicability"]
    assert retained_results(payload)[0].metadata["tier3_applicability"] == payload["tier3_applicability"]


def test_advisory_failed_result_keeps_overall_required_gate_pass():
    result = tier3_result()
    result.passed = False
    result.metadata["agent_eval"]["verdict"] = "fail"
    result.metadata["agent_eval"]["summary"]["verdict"] = "fail"
    payload = emit(result)
    assert payload["overall_passed"] is True and payload["overall_status"] == "passed"
    assert payload["results"][0]["status"] == "failed"
    assert retained_results(payload)[0].passed is False
