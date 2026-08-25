# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Advisory Tier 3 skips stay visible without failing required validation."""

from __future__ import annotations

import base64
import hashlib
import json
import re
from copy import deepcopy
from html import unescape
from pathlib import Path

import pytest
from scripts.ci import check_public_benchmarks as benchmark_gate

from skillevaluator.evaluation.tier3_report import _validation_result_from_payload, advisory_skip_result
from skillevaluator.models import Finding, Severity, ValidationResult
from skillevaluator.reporting import BenchmarkReporter, CLIReporter, HTMLReporter, JSONReporter, MarkdownReporter
from skillevaluator.reporting.html import _sanitize_tier3_display_payload


def _plain_cli(output: str) -> str:
    """Strip ANSI styling that CI may force into Rich output."""
    return re.sub(r"\x1b\[[0-9;]*m", "", output)


def _html_report_data(output: str) -> dict:
    match = re.search(r'<script id="report-data" type="application/json">(.*?)</script>', output, re.DOTALL)
    assert match is not None
    return json.loads(match.group(1))


def _html_tier3_payload(output: str) -> dict:
    match = re.search(
        r'<script type="application/json" id="tier3-full"(?P<attrs>[^>]*)>(?P<body>.*?)</script>',
        output,
        re.DOTALL,
    )
    assert match is not None
    body = match.group("body").strip()
    if 'data-encoding="base64"' in match.group("attrs"):
        body = base64.b64decode(body).decode("utf-8")
    return json.loads(body)


def _html_tab(output: str, tab_id: str) -> str:
    marker = f'id="tab-{tab_id}"'
    start = output.index(marker)
    next_tab = output.find('id="tab-', start + len(marker))
    report_data = output.find('id="report-data"', start + len(marker))
    candidates = [position for position in (next_tab, report_data) if position >= 0]
    end = min(candidates) if candidates else len(output)
    return output[start:end]


def _publication_target(skill_name: str) -> dict[str, str]:
    """Return a deterministic valid source identity for reporter fixtures."""
    digest = hashlib.sha256(f"fixture:{skill_name}".encode()).hexdigest()
    return {
        "skill_name": skill_name,
        "skill_digest": f"sha256:{digest}",
        "skill_digest_algorithm": "skill-evaluator-source-tree/2",
    }


def _bind_publication_target(
    results: list[ValidationResult],
    target: dict[str, str],
) -> None:
    """Bind fixture results and duplicated Tier 3 claims to one target."""
    fixture_producers = {
        "SCHEMA": (1, "schema"),
        "Code Integrity & Hygiene": (1, "code-integrity"),
        "Similarity Check": (2, "similarity"),
        "Context Deduplication": (2, "context-optimization"),
    }
    for result in results:
        result.metadata["publication_target"] = dict(target)
        if "publication_evidence" not in result.metadata and result.validator_name in fixture_producers:
            tier, check_id = fixture_producers[result.validator_name]
            _stamp_publication_evidence(result, tier=tier, check_id=check_id)
        payload = result.metadata.get("agent_eval")
        if not isinstance(payload, dict):
            continue
        payload["publication_target"] = dict(target)
        summary = payload.get("summary")
        if isinstance(summary, dict):
            summary["publication_target"] = dict(target)


def _stamp_publication_evidence(result: ValidationResult, *, tier: int, check_id: str) -> None:
    """Stamp the producer contract expected from built-in tier wrappers."""
    result.metadata["publication_evidence"] = {
        "schema_version": 1,
        "producer": f"skillevaluator.tier{tier}",
        "tier": tier,
        "check_id": check_id,
    }


def _complete_tier3_result(
    marker: str,
    *,
    score: float,
    runtime_seconds: float,
    evaluated_at: str = "2026-08-24T12:00:00+00:00",
) -> ValidationResult:
    skill_name = f"candidate-{marker.lower()}"
    publication_target = _publication_target(skill_name)
    run_id = f"fixture-{hashlib.sha256(marker.encode()).hexdigest()[:16]}"
    result = ValidationResult(validator_name="AGENT_EVAL")
    result.add_success("agent_eval", "Live evaluation completed")
    result.metadata["agent_eval"] = {
        "marker": marker,
        "skill_name": skill_name,
        "run_id": run_id,
        "publication_target": dict(publication_target),
        "verdict": "pass",
        "execution_status": "succeeded",
        "evaluated_at": evaluated_at,
        "evaluator_version": "0.9.0",
        "expected_attempts": 1,
        "scored_attempts": 1,
        "dataset_summary": {"total_tasks": 1},
        "dataset_digest": "sha256:" + "a" * 64,
        "dataset_digest_algorithm": "skill-evaluator-dataset-snapshot/1",
        "attempt_policy": {"max_attempts": 1, "pass_threshold": 0.5},
        "summary": {
            "verdict": "pass",
            "execution_status": "succeeded",
            "environment": "docker",
            "expected_attempts": 1,
            "scored_attempts": 1,
            "run_id": run_id,
            "publication_target": dict(publication_target),
        },
        "runtime_seconds": runtime_seconds,
        "agents": {
            "codex": {
                "model": "gpt-codex",
                "execution_status": "succeeded",
                "expected_attempts": 1,
                "scored_attempts": 1,
                "with_skill": score,
                "dimensions": [
                    {"id": dimension, "with_skill": score}
                    for dimension in ("security", "correctness", "discoverability", "effectiveness", "efficiency")
                ],
            }
        },
        "evaluators": {
            "accuracy": {
                "with_skill": score,
                "baseline": 0.4,
                "lift": score - 0.4,
            }
        },
        "trials": [],
    }
    result.metadata["publication_target"] = dict(publication_target)
    return result


def test_advisory_skip_is_non_blocking_in_json() -> None:
    payload = json.loads(
        JSONReporter(include_timestamp=False).render_all(
            [advisory_skip_result("No public provider key", skill_name="demo")]
        )
    )

    assert payload["overall_status"] == "passed"
    assert payload["overall_passed"] is True
    assert payload["total_advisory_skipped"] == 1
    assert payload["results"][0]["passed"] is False
    assert payload["results"][0]["status"] == "skipped"


def test_clean_tier2_skip_is_explicit_in_combined_outputs() -> None:
    tier1 = ValidationResult(validator_name="SCHEMA")
    tier1.add_success("schema", "Schema passed")
    tier1.metadata["benchmark_policy"] = {"tier3_required": False}
    tier2 = ValidationResult(validator_name="Tier 2 Deduplication")
    tier2.add_warning("Skipped: configure a public embedding provider")
    tier2.metadata.update(
        {
            "skipped": True,
            "gating": {"tier": 2, "blocking": True},
        }
    )
    results = [tier1, tier2]

    payload = json.loads(JSONReporter(include_timestamp=False).render_all(results))
    html_data = _html_report_data(HTMLReporter(include_timestamp=False).render_all(results))
    benchmark = BenchmarkReporter(include_timestamp=False).render_all(results)

    assert payload["overall_status"] == "passed"
    assert payload["total_skipped"] == 1
    assert payload["total_advisory_skipped"] == 0
    assert payload["results"][1]["passed"] is True
    assert payload["results"][1]["status"] == "skipped"
    assert payload["results"][1]["skipped"] is True
    assert payload["results"][1]["skip_reason"] == "Skipped: configure a public embedding provider"

    assert html_data["summary"]["status"] == "passed"
    assert html_data["summary"]["passed_count"] == 1
    assert html_data["summary"]["skipped_count"] == 1
    assert html_data["summary"]["advisory_skipped_count"] == 0
    assert html_data["results"][1]["status"] == "skipped"
    assert html_data["results"][1]["skipped"] is True
    assert html_data["results"][1]["skip_reason"] == "Skipped: configure a public embedding provider"

    assert "Overall verdict: INCOMPLETE" in benchmark
    assert "## Publication Recommendation" not in benchmark


def test_legacy_content_dedup_category_is_tier2_for_every_reporter() -> None:
    legacy_tier2 = ValidationResult(validator_name="Legacy Validator")
    legacy_tier2.add_finding(
        Finding(
            category="CONTENT_DEDUP",
            severity=Severity.LOW,
            check_name="legacy_dedup",
            message="Legacy deduplication completed",
            file_path="",
        )
    )
    legacy_tier2.metadata["benchmark_policy"] = {
        "tier2_required": False,
        "tier3_required": False,
    }

    benchmark = BenchmarkReporter(include_timestamp=False).render_all([legacy_tier2])
    json_payload = json.loads(JSONReporter(include_timestamp=False).render_all([legacy_tier2]))
    html_data = _html_report_data(HTMLReporter(include_timestamp=False).render_all([legacy_tier2]))
    markdown = MarkdownReporter(include_timestamp=False).render_all([legacy_tier2])

    assert "Overall verdict: INCOMPLETE" in benchmark
    assert "| Tier 1 | Static validation | **NOT RUN** |" in benchmark
    assert "| Tier 2 | Semantic deduplication | **INCOMPLETE** |" in benchmark
    assert json_payload["publication_status"] == "incomplete"
    assert html_data["publication"]["status"] == "incomplete"
    assert "**Publication status:** ⚠️ INCOMPLETE" in markdown


def test_html_tier2_failure_stays_failed_when_cli_gate_is_nonblocking() -> None:
    tier2 = ValidationResult(validator_name="Similarity Check")
    tier2.add_error("Similarity provider failed")
    tier2.metadata["gating"] = {"tier": 2, "blocking": False}

    html = HTMLReporter(include_timestamp=False).render_all([tier2])
    html_data = _html_report_data(html)

    assert html_data["summary"]["status"] == "passed"
    assert re.search(
        r'tier-card fail.*?tier-card-label">Tier 2</span>.*?tier-card-verdict">\s*FAIL\s*</span>',
        html,
        re.DOTALL,
    )


def test_html_tier2_incomplete_evidence_is_not_rendered_as_success() -> None:
    tier2 = ValidationResult(validator_name="Similarity Check")
    tier2.add_success("similarity", "No semantic overlaps found")
    tier2.metadata["incomplete_scans"] = ["embedding-provider"]

    html = HTMLReporter(include_timestamp=False).render_all([tier2])
    tier2_tab = _html_tab(html, "tier2")

    assert re.search(
        r'tier-card warn.*?tier-card-label">Tier 2</span>.*?tier-card-verdict">\s*INCOMPLETE\s*</span>',
        html,
        re.DOTALL,
    )
    assert "No semantic overlaps detected" not in tier2_tab
    assert "Incomplete scanner evidence" in tier2_tab
    assert "embedding-provider" in tier2_tab


def test_html_malformed_tier3_result_is_incomplete_like_benchmark() -> None:
    tier1 = ValidationResult(validator_name="SCHEMA")
    tier1.add_success("schema", "Schema passed")
    tier2 = ValidationResult(validator_name="Similarity Check")
    tier2.add_success("similarity", "No semantic overlaps found")
    malformed_tier3 = ValidationResult(validator_name="AGENT_EVAL")
    results = [tier1, tier2, malformed_tier3]

    benchmark = BenchmarkReporter(include_timestamp=False).render_all(results)
    reporter = HTMLReporter(include_timestamp=False)
    html = reporter.render_all(results)
    fallback = reporter._tier3_report_data([malformed_tier3])

    assert "| Tier 3 | Live agent evaluation | **INCOMPLETE** |" in benchmark
    assert re.search(
        r'tier-card warn.*?tier-card-label">Tier 3</span>.*?tier-card-verdict">\s*INCOMPLETE\s*</span>',
        html,
        re.DOTALL,
    )
    assert not re.search(
        r'tier-card pass.*?tier-card-label">Tier 3</span>.*?tier-card-verdict">\s*PASS\s*</span>',
        html,
        re.DOTALL,
    )
    assert fallback is not None
    assert fallback["verdict"] == "incomplete"
    assert fallback["execution_status"] == "incomplete"
    assert fallback["summary"]["verdict"] == "incomplete"
    assert fallback["summary"]["execution_status"] == "incomplete"


def test_html_tier3_scan_incomplete_is_not_rendered_as_failure() -> None:
    incomplete_tier3 = ValidationResult(validator_name="AGENT_EVAL")
    incomplete_tier3.mark_scan_incomplete("harbor")

    benchmark = BenchmarkReporter(include_timestamp=False).render_all([incomplete_tier3])
    reporter = HTMLReporter(include_timestamp=False)
    html = reporter.render_all([incomplete_tier3])
    html_data = _html_report_data(html)
    fallback = reporter._tier3_report_data([incomplete_tier3])

    assert "| Tier 3 | Live agent evaluation | **INCOMPLETE** |" in benchmark
    assert html_data["summary"]["status"] == "incomplete"
    assert re.search(
        r'tier-card warn.*?tier-card-label">Tier 3</span>.*?tier-card-verdict">\s*INCOMPLETE\s*</span>',
        html,
        re.DOTALL,
    )
    assert not re.search(
        r'tier-card fail.*?tier-card-label">Tier 3</span>.*?tier-card-verdict">\s*FAIL\s*</span>',
        html,
        re.DOTALL,
    )
    assert fallback is not None
    assert fallback["verdict"] == "incomplete"
    assert fallback["execution_status"] == "incomplete"


def test_html_payloadless_generic_tier3_skip_is_rendered_as_skipped() -> None:
    skipped_tier3 = ValidationResult(validator_name="AGENT_EVAL")
    skipped_tier3.add_warning("Skipped: Harbor runtime unavailable")
    skipped_tier3.metadata.update(
        {
            "skipped": True,
            "benchmark_policy": {"tier3_required": False},
        }
    )

    benchmark = BenchmarkReporter(include_timestamp=False).render_all([skipped_tier3])
    reporter = HTMLReporter(include_timestamp=False)
    html = reporter.render_all([skipped_tier3])
    html_data = _html_report_data(html)
    fallback = reporter._tier3_report_data([skipped_tier3])

    # A legacy unbound ``tier3_required: false`` claim remains readable but
    # cannot waive publication evidence for an unknown target.
    assert "| Tier 3 | Live agent evaluation | **INCOMPLETE** |" in benchmark
    assert html_data["summary"]["status"] == "passed"
    assert re.search(
        r'tier-card skipped.*?tier-card-label">Tier 3</span>.*?tier-card-verdict">\s*SKIPPED\s*</span>',
        html,
        re.DOTALL,
    )
    assert "Live evaluation skipped:</strong> Skipped: Harbor runtime unavailable" in html
    assert fallback is not None
    assert fallback["verdict"] == "skipped"
    assert fallback["execution_status"] == "skipped"


@pytest.mark.parametrize(
    ("payload_verdict", "summary_verdict", "execution_status", "expected_status"),
    [
        ("fail", "fail", "failed", "FAIL"),
        ("pass", "neutral", "succeeded", "INCOMPLETE"),
        ("pass", "pass", "succeeded", "INCOMPLETE"),
    ],
)
def test_generic_tier3_skip_metadata_cannot_hide_payload_truth(
    payload_verdict: str,
    summary_verdict: str,
    execution_status: str,
    expected_status: str,
) -> None:
    tier1 = ValidationResult(validator_name="SCHEMA")
    tier1.add_success("schema", "Schema passed")
    tier2 = ValidationResult(validator_name="Similarity Check")
    tier2.add_success("similarity_check", "Similarity scan completed")
    tier3 = _complete_tier3_result("SKIP-TRUTH", score=0.9, runtime_seconds=1.0)
    payload = tier3.metadata["agent_eval"]
    payload["verdict"] = payload_verdict
    payload["summary"]["verdict"] = summary_verdict
    payload["execution_status"] = execution_status
    payload["summary"]["execution_status"] = execution_status
    tier3.metadata.update(
        {
            "skipped": True,
            "benchmark_policy": {"tier3_required": False},
        }
    )

    results = [tier1, tier2, tier3]
    benchmark = BenchmarkReporter(include_timestamp=False).render_all(results)
    json_payload = json.loads(JSONReporter(include_timestamp=False).render_all(results))

    assert f"Overall verdict: {expected_status}" in benchmark
    assert "## Publication Recommendation" not in benchmark
    assert f"| Tier 3 | Live agent evaluation | **{expected_status}** |" in benchmark
    assert json_payload["publication_status"] == expected_status.casefold()


@pytest.mark.parametrize("tier", ["tier1", "tier2", "tier3"])
def test_generic_optional_metadata_cannot_waive_partial_required_tier(tier: str) -> None:
    tier1 = ValidationResult(validator_name="SCHEMA")
    tier1.add_success("schema", "Schema passed")
    if tier != "tier3":
        tier1.metadata["benchmark_policy"] = {"tier3_required": False}
    tier2 = ValidationResult(validator_name="Similarity Check")
    tier2.add_success("similarity_check", "Similarity scan completed")
    skipped = ValidationResult(
        validator_name=(
            "Code Integrity & Hygiene"
            if tier == "tier1"
            else "Context Deduplication"
            if tier == "tier2"
            else "AGENT_EVAL"
        )
    )
    skipped.add_warning("Skipped: validator runtime unavailable")
    skipped.metadata.update({"skipped": True, "optional": True})
    results = [tier1, tier2, skipped]

    benchmark = BenchmarkReporter(include_timestamp=False).render_all(results)
    json_payload = json.loads(JSONReporter(include_timestamp=False).render_all(results))
    html_data = _html_report_data(HTMLReporter(include_timestamp=False).render_all(results))
    markdown = MarkdownReporter(include_timestamp=False).render_all(results)
    tier_label = {"tier1": "Tier 1", "tier2": "Tier 2", "tier3": "Tier 3"}[tier]

    assert "> ⚠️ **Overall verdict: INCOMPLETE — Required evidence is missing**" in benchmark
    assert "## Publication Recommendation" not in benchmark
    assert f"| {tier_label} |" in benchmark
    assert "**INCOMPLETE**" in next(line for line in benchmark.splitlines() if line.startswith(f"| {tier_label} "))
    assert json_payload["publication_status"] == "incomplete"
    assert html_data["publication"]["status"] == "incomplete"
    assert "**Publication status:** ⚠️ INCOMPLETE" in markdown


def test_html_partial_tier3_payload_is_process_pass_but_publication_incomplete() -> None:
    partial_payloads = [
        {"verdict": "pass"},
        {
            "verdict": "pass",
            "execution_status": "succeeded",
            "overall_score": 0.9,
            "summary": {},
            "agents": {},
            "dimensions": [],
            "evaluators": {},
            "dataset": [],
        },
    ]

    for payload in partial_payloads:
        partial_tier3 = ValidationResult(validator_name="AGENT_EVAL")
        partial_tier3.metadata["agent_eval"] = payload

        benchmark = BenchmarkReporter(include_timestamp=False).render_all([partial_tier3])
        reporter = HTMLReporter(include_timestamp=False)
        html = reporter.render_all([partial_tier3])
        html_data = _html_report_data(html)
        fallback = reporter._tier3_report_data([partial_tier3])

        assert "| Tier 3 | Live agent evaluation | **INCOMPLETE** |" in benchmark
        assert html_data["summary"]["status"] == "passed"
        assert html_data["publication"]["status"] == "incomplete"
        assert re.search(
            r'tier-card warn.*?tier-card-label">Tier 3</span>.*?tier-card-verdict">\s*INCOMPLETE\s*</span>',
            html,
            re.DOTALL,
        )
        assert not re.search(
            r'tier-card pass.*?tier-card-label">Tier 3</span>.*?tier-card-verdict">\s*PASS\s*</span>',
            html,
            re.DOTALL,
        )
        assert fallback is not None
        assert fallback["verdict"] == "incomplete"
        assert fallback["execution_status"] == "incomplete"


def test_html_production_wrapped_partial_tier3_payload_is_incomplete() -> None:
    partial_tier3 = _validation_result_from_payload(
        {
            "verdict": "pass",
            "execution_status": "succeeded",
            "overall_score": 0.9,
            "summary": {},
            "agents": {},
            "dimensions": [],
            "evaluators": {},
            "dataset": [],
        }
    )
    assert partial_tier3 is not None
    assert partial_tier3.passed is True
    assert partial_tier3.summary.checks_performed == 1

    benchmark = BenchmarkReporter(include_timestamp=False).render_all([partial_tier3])
    reporter = HTMLReporter(include_timestamp=False)
    html = reporter.render_all([partial_tier3])
    json_payload = json.loads(JSONReporter(include_timestamp=False).render_all([partial_tier3]))
    markdown = MarkdownReporter(include_timestamp=False).render_all([partial_tier3])
    fallback = reporter._tier3_report_data([partial_tier3])

    assert "| Tier 3 | Live agent evaluation | **INCOMPLETE** |" in benchmark
    assert re.search(
        r'tier-card warn.*?tier-card-label">Tier 3</span>.*?tier-card-verdict">\s*INCOMPLETE\s*</span>',
        html,
        re.DOTALL,
    )
    assert fallback is not None
    assert fallback["verdict"] == "incomplete"
    assert fallback["execution_status"] == "incomplete"
    assert fallback["overall_score"] is None
    assert fallback["agents"] == {}
    assert "0.90" not in html
    assert json_payload["overall_status"] == "passed"
    assert json_payload["publication_status"] == "incomplete"
    assert json_payload["publication"]["tier3"]["status"] == "incomplete"
    assert "**Status:** ✅ PASSED" in markdown
    assert "**Publication status:** ⚠️ INCOMPLETE" in markdown
    assert "**Verdict:** INCOMPLETE" in markdown


def test_html_forced_tier3_failure_preserves_result_diagnostics() -> None:
    tier3 = _validation_result_from_payload(
        {
            "verdict": "pass",
            "execution_status": "succeeded",
            "overall_score": 0.9,
            "execution_errors": [],
            "suggestions": [],
            "summary": {"execution_errors": []},
            "agents": {},
            "dimensions": [],
            "evaluators": {},
            "dataset": [],
        }
    )
    assert tier3 is not None
    tier3.add_error("Engine diagnostic must remain visible")

    reporter = HTMLReporter(include_timestamp=False)
    html = reporter.render_all([tier3])
    fallback = reporter._tier3_report_data([tier3])

    assert fallback is not None
    assert fallback["verdict"] == "fail"
    assert fallback["execution_status"] == "failed"
    assert fallback["execution_errors"] == ["Engine diagnostic must remain visible"]
    assert fallback["summary"]["execution_errors"] == ["Engine diagnostic must remain visible"]
    assert "Engine diagnostic must remain visible" in fallback["suggestions"]
    assert "Engine diagnostic must remain visible" in html


def test_html_selects_executed_tier3_payload_independent_of_result_order() -> None:
    partial_tier3 = ValidationResult(validator_name="AGENT_EVAL")
    partial_tier3.metadata["agent_eval"] = {"verdict": "pass"}
    publication_target = _publication_target("demo")
    run_id = "fixture-executed-selection"
    executed_tier3 = _validation_result_from_payload(
        {
            "skill_name": "demo",
            "verdict": "pass",
            "execution_status": "succeeded",
            "run_id": run_id,
            "publication_target": dict(publication_target),
            "overall_score": 0.9,
            "expected_attempts": 1,
            "scored_attempts": 1,
            "summary": {
                "skill_name": "demo",
                "verdict": "pass",
                "execution_status": "succeeded",
                "run_id": run_id,
                "publication_target": dict(publication_target),
            },
            "agents": {
                "codex": {
                    "execution_status": "succeeded",
                    "with_skill": 0.9,
                    "expected_attempts": 1,
                    "scored_attempts": 1,
                }
            },
            "dimensions": [],
            "evaluators": {},
            "dataset": [],
        }
    )
    assert executed_tier3 is not None

    reporter = HTMLReporter(include_timestamp=False)
    for results in ([partial_tier3, executed_tier3], [executed_tier3, partial_tier3]):
        html = reporter.render_all(results)
        payload = reporter._tier3_report_data(results)
        json_payload = json.loads(JSONReporter(include_timestamp=False).render_all(results))
        markdown = MarkdownReporter(include_timestamp=False).render_all(results)

        assert re.search(
            r'tier-card pass.*?tier-card-label">Tier 3</span>.*?tier-card-verdict">\s*PASS\s*</span>',
            html,
            re.DOTALL,
        )
        assert payload is not None
        assert payload["verdict"] == "pass"
        assert payload["execution_status"] == "succeeded"
        assert payload["overall_score"] == 0.9
        assert json_payload["tier3"]["overall_score"] == 0.9
        assert "**Verdict:** PASS" in markdown


def test_all_reporters_select_same_complete_tier3_payload_independent_of_order() -> None:
    candidate_a = _complete_tier3_result("A", score=0.81, runtime_seconds=8.0)
    candidate_b = _complete_tier3_result("B", score=0.91, runtime_seconds=9.0)

    reporter = HTMLReporter(include_timestamp=False)
    for results in ([candidate_a, candidate_b], [candidate_b, candidate_a]):
        benchmark = BenchmarkReporter(include_timestamp=False).render_all(results)
        html_payload = reporter._tier3_report_data(results)
        json_payload = json.loads(JSONReporter(include_timestamp=False).render_all(results))
        markdown = MarkdownReporter(include_timestamp=False).render_all(results)

        assert "# Skill Benchmark: candidate-b" in benchmark
        assert html_payload is not None and html_payload["marker"] == "B"
        assert json_payload["tier3"]["marker"] == "B"
        assert "**Runtime:** 9.0s" in markdown


def test_all_reporters_select_newest_complete_tier3_payload_before_score() -> None:
    older = _complete_tier3_result(
        "OLDER",
        score=0.95,
        runtime_seconds=5.0,
        evaluated_at="2026-08-23T12:00:00+00:00",
    )
    newer = _complete_tier3_result(
        "NEWER",
        score=0.60,
        runtime_seconds=10.0,
        evaluated_at="2026-08-24T12:00:00+00:00",
    )

    reporter = HTMLReporter(include_timestamp=False)
    for results in ([older, newer], [newer, older]):
        benchmark = BenchmarkReporter(include_timestamp=False).render_all(results)
        html_payload = reporter._tier3_report_data(results)
        json_payload = json.loads(JSONReporter(include_timestamp=False).render_all(results))
        markdown = MarkdownReporter(include_timestamp=False).render_all(results)

        assert "# Skill Benchmark: candidate-newer" in benchmark
        assert html_payload is not None and html_payload["marker"] == "NEWER"
        assert json_payload["tier3"]["marker"] == "NEWER"
        assert "**Runtime:** 10.0s" in markdown


def test_future_dated_tier3_payload_cannot_dominate_candidate_selection() -> None:
    valid = _complete_tier3_result(
        "VALID",
        score=0.60,
        runtime_seconds=10.0,
        evaluated_at="2026-08-24T12:00:00+00:00",
    )
    future = _complete_tier3_result(
        "FUTURE",
        score=0.99,
        runtime_seconds=99.0,
        evaluated_at="9999-01-01T00:00:00+00:00",
    )
    for result in (valid, future):
        result.metadata["agent_eval"]["skill_name"] = "same-target"
        result.metadata["agent_eval"]["summary"]["skill_name"] = "same-target"

    reporter = HTMLReporter(include_timestamp=False)
    for results in ([valid, future], [future, valid]):
        html_payload = reporter._tier3_report_data(results)
        json_payload = json.loads(JSONReporter(include_timestamp=False).render_all(results))
        markdown = MarkdownReporter(include_timestamp=False).render_all(results)

        assert html_payload is not None and html_payload["marker"] == "VALID"
        assert json_payload["tier3"]["marker"] == "VALID"
        assert "**Runtime:** 10.0s" in markdown


@pytest.mark.parametrize("reverse", [False, True])
def test_conflicting_tier3_payload_policies_fail_closed_independent_of_order(reverse: bool) -> None:
    tier1 = ValidationResult(validator_name="SCHEMA")
    tier1.add_success("schema", "Schema passed")
    optional = _complete_tier3_result("OPTIONAL", score=0.91, runtime_seconds=5.0)
    optional.metadata["agent_eval"]["benchmark_policy"] = {"tier2_required": False}
    required = _complete_tier3_result("REQUIRED", score=0.81, runtime_seconds=6.0)
    required.metadata["agent_eval"]["benchmark_policy"] = {"tier2_required": True}
    for result in (optional, required):
        result.metadata["agent_eval"]["skill_name"] = "same-target"
        result.metadata["agent_eval"]["summary"]["skill_name"] = "same-target"
    tier3_results = [optional, required]
    if reverse:
        tier3_results.reverse()
    results = [tier1, *tier3_results]

    benchmark = BenchmarkReporter(include_timestamp=False).render_all(results)
    json_payload = json.loads(JSONReporter(include_timestamp=False).render_all(results))

    assert "Overall verdict: INCOMPLETE" in benchmark
    assert "- Tier 2 evidence: required for publication" in benchmark
    assert json_payload["benchmark_policy"]["tier2_required"] is True
    assert json_payload["publication_status"] == "incomplete"


@pytest.mark.parametrize("key", ["tier2_required", "tier3_required"])
@pytest.mark.parametrize("top_value", [False, True], ids=["top-optional", "top-required"])
def test_conflicting_policy_inside_one_payload_fails_closed(
    key: str,
    top_value: bool,
) -> None:
    tier1 = ValidationResult(validator_name="SCHEMA")
    tier1.add_success("schema", "Schema passed")
    skipped = advisory_skip_result("Tier 3 intentionally omitted", skill_name="demo")
    payload = skipped.metadata["agent_eval"]
    payload["benchmark_policy"] = {"tier2_required": False, "tier3_required": False, key: top_value}
    payload["summary"]["benchmark_policy"] = {
        "tier2_required": False,
        "tier3_required": False,
        key: not top_value,
    }
    results = [tier1, skipped]

    benchmark = BenchmarkReporter(include_timestamp=False).render_all(results)
    json_payload = json.loads(JSONReporter(include_timestamp=False).render_all(results))

    assert "Overall verdict: INCOMPLETE" in benchmark
    assert json_payload["benchmark_policy"][key] is True
    assert json_payload["publication_status"] == "incomplete"


def test_weak_tier3_policy_decoy_cannot_waive_canonical_required_evidence() -> None:
    tier1 = ValidationResult(validator_name="SCHEMA")
    tier1.add_success("schema", "Schema passed")
    tier1.metadata["benchmark_policy"] = {"tier2_required": True}
    complete = _complete_tier3_result("COMPLETE", score=0.9, runtime_seconds=1.0)
    complete.metadata["agent_eval"]["summary"]["benchmark_policy"] = {"tier2_required": True}
    decoy = ValidationResult(validator_name="AGENT_EVAL")
    decoy.metadata["agent_eval"] = {"benchmark_policy": {"tier2_required": False, "tier3_required": True}}

    results = [tier1, complete, decoy]
    benchmark = BenchmarkReporter(include_timestamp=False).render_all(results)
    json_payload = json.loads(JSONReporter(include_timestamp=False).render_all(results))

    assert "Overall verdict: INCOMPLETE" in benchmark
    assert "- Tier 2 evidence: required for publication" in benchmark
    assert json_payload["benchmark_policy"]["tier2_required"] is True


@pytest.mark.parametrize("reverse", [False, True])
def test_explicit_failed_tier3_peer_overrides_complete_pass_independent_of_order(reverse: bool) -> None:
    tier1 = ValidationResult(validator_name="SCHEMA")
    tier1.add_success("schema", "Schema passed")
    tier2 = ValidationResult(validator_name="Similarity Check")
    tier2.add_success("similarity_check", "Similarity scan completed")
    passing = _complete_tier3_result("PASSING", score=0.91, runtime_seconds=5.0)
    failed = ValidationResult(validator_name="AGENT_EVAL")
    failed.metadata["agent_eval"] = {
        "marker": "FAILED",
        "verdict": "fail",
        "execution_status": "failed",
        "execution_errors": ["boom"],
    }
    tier3_results = [passing, failed]
    if reverse:
        tier3_results.reverse()
    results = [tier1, tier2, *tier3_results]

    benchmark = BenchmarkReporter(include_timestamp=False).render_all(results)
    html_payload = HTMLReporter(include_timestamp=False)._tier3_report_data(tier3_results)
    json_payload = json.loads(JSONReporter(include_timestamp=False).render_all(results))
    markdown = MarkdownReporter(include_timestamp=False).render_all(results)

    assert "Overall verdict: FAIL" in benchmark
    assert html_payload is not None and html_payload["verdict"] == "fail"
    assert json_payload["tier3"]["marker"] == "FAILED"
    assert json_payload["publication_status"] == "fail"
    assert "**Verdict:** FAIL" in markdown


def test_failed_tier3_execution_status_overrides_pass_verdict_across_reporters() -> None:
    tier1 = ValidationResult(validator_name="SCHEMA")
    tier1.add_success("schema", "Schema passed")
    tier2 = ValidationResult(validator_name="Similarity Check")
    tier2.add_success("similarity_check", "Similarity scan completed")
    tier3 = _complete_tier3_result("FAILED-EXECUTION", score=0.9, runtime_seconds=1.0)
    tier3.metadata["agent_eval"]["execution_status"] = "failed"
    tier3.metadata["agent_eval"]["summary"]["execution_status"] = "failed"
    results = [tier1, tier2, tier3]

    benchmark = BenchmarkReporter(include_timestamp=False).render_all(results)
    html_output = HTMLReporter(include_timestamp=False).render_all(results)
    html_payload = HTMLReporter(include_timestamp=False)._tier3_report_data([tier3])
    json_payload = json.loads(JSONReporter(include_timestamp=False).render_all(results))
    markdown = MarkdownReporter(include_timestamp=False).render_all(results)

    assert "Overall verdict: FAIL" in benchmark
    assert "| Tier 3 | Live agent evaluation | **FAIL** |" in benchmark
    assert html_payload is not None and html_payload["verdict"] == "fail"
    assert 'data-publication-status="fail"' in html_output
    assert re.search(r'tier-card-verdict">\s*FAIL\s*</span>', html_output)
    assert json_payload["publication_status"] == "fail"
    assert "**Publication status:** ❌ FAIL" in markdown
    assert "**Verdict:** FAIL" in markdown


def test_missing_tier3_execution_status_stays_incomplete_across_reporters() -> None:
    tier1 = ValidationResult(validator_name="SCHEMA")
    tier1.add_success("schema", "Schema passed")
    tier2 = ValidationResult(validator_name="Similarity Check")
    tier2.add_success("similarity_check", "Similarity scan completed")
    tier3 = _complete_tier3_result("MISSING-STATUS", score=0.9, runtime_seconds=1.0)
    tier3.metadata["agent_eval"].pop("execution_status")
    tier3.metadata["agent_eval"]["summary"].pop("execution_status")
    results = [tier1, tier2, tier3]

    benchmark = BenchmarkReporter(include_timestamp=False).render_all(results)
    json_payload = json.loads(JSONReporter(include_timestamp=False).render_all(results))
    html_output = HTMLReporter(include_timestamp=False).render_all(results)
    html_payload = HTMLReporter(include_timestamp=False)._tier3_report_data([tier3])
    html_data = _html_report_data(html_output)
    markdown = MarkdownReporter(include_timestamp=False).render_all(results)

    assert "Overall verdict: INCOMPLETE" in benchmark
    assert "| Tier 3 | Live agent evaluation | **INCOMPLETE** |" in benchmark
    assert json_payload["publication_status"] == "incomplete"
    assert html_payload is not None and html_payload["execution_status"] == "incomplete"
    assert html_data["publication"]["status"] == "incomplete"
    assert 'data-publication-status="incomplete"' in html_output
    assert "**Publication status:** ⚠️ INCOMPLETE" in markdown


def test_advisory_skip_payload_policy_is_preserved_across_reporters(tmp_path: Path) -> None:
    tier1 = ValidationResult(validator_name="SCHEMA")
    tier1.add_success("schema", "Schema passed")
    tier3 = advisory_skip_result("Tier 3 intentionally omitted", skill_name="demo")
    tier3.metadata["agent_eval"]["benchmark_policy"] = {
        "tier2_required": False,
        "tier3_required": False,
    }
    results = [tier1, tier3]
    _bind_publication_target(results, _publication_target("demo"))

    benchmark = BenchmarkReporter(include_timestamp=False).render_all(results)
    json_payload = json.loads(JSONReporter(include_timestamp=False).render_all(results))
    html_output = HTMLReporter(include_timestamp=False).render_all(results)
    html_data = _html_report_data(html_output)
    markdown = MarkdownReporter(include_timestamp=False).render_all(results)
    benchmark_path = tmp_path / "BENCHMARK.md"
    benchmark_path.write_text(benchmark, encoding="utf-8")
    _files, offenders = benchmark_gate.find_offenders([benchmark_path])

    assert "Overall verdict: PASS" in benchmark
    assert json_payload["benchmark_policy"] == {"tier2_required": False, "tier3_required": False}
    assert json_payload["publication_status"] == "pass"
    assert html_data["publication"]["status"] == "pass"
    assert 'data-publication-status="pass"' in html_output
    assert "**Publication status:** ✅ PASS" in markdown
    assert offenders == []


def test_completed_tier3_peer_suppresses_stale_skip_narrative() -> None:
    tier1 = ValidationResult(validator_name="SCHEMA")
    tier1.add_success("schema", "Schema passed")
    tier2 = ValidationResult(validator_name="Similarity Check")
    tier2.add_success("similarity_check", "Similarity scan completed")
    completed = _complete_tier3_result("COMPLETED-PEER", score=0.9, runtime_seconds=1.0)
    completed.metadata["agent_eval"]["skill_name"] = "demo"
    completed.metadata["agent_eval"]["summary"]["skill_name"] = "demo"
    skipped = advisory_skip_result("Earlier live run skipped", skill_name="demo")
    results = [tier1, tier2, skipped, completed]
    _bind_publication_target(results, _publication_target("demo"))

    benchmark = BenchmarkReporter(include_timestamp=False).render_all(results)

    assert "> ✅ **Overall verdict: PASS — Recommended for publication**" in benchmark
    assert "## Publication Recommendation" in benchmark
    assert "Tier 3 live evaluation was skipped" not in benchmark
    assert "Tier 3 live evaluation: SKIPPED" not in benchmark
    assert "rerun Tier 3" not in benchmark


def test_stale_advisory_skip_provenance_cannot_waive_tier3_failure(tmp_path: Path) -> None:
    tier1 = ValidationResult(validator_name="SCHEMA")
    tier1.add_success("schema", "Schema passed")
    tier2 = ValidationResult(validator_name="Similarity Check")
    tier2.add_success("similarity_check", "Similarity scan completed")
    tier3 = advisory_skip_result("stale skip", skill_name="demo")
    stale_provenance = tier3.metadata["agent_eval"]["provenance"]
    payload = _complete_tier3_result("CONTRADICTORY-SKIP", score=0.1, runtime_seconds=1.0).metadata["agent_eval"]
    payload["provenance"] = stale_provenance
    payload["benchmark_policy"] = {"tier3_required": False}
    payload["verdict"] = "fail"
    payload["execution_status"] = "failed"
    payload["summary"]["verdict"] = "fail"
    payload["summary"]["execution_status"] = "failed"
    tier3.metadata["agent_eval"] = payload
    results = [tier1, tier2, tier3]

    benchmark = BenchmarkReporter(include_timestamp=False).render_all(results)
    json_payload = json.loads(JSONReporter(include_timestamp=False).render_all(results))
    html_output = HTMLReporter(include_timestamp=False).render_all(results)
    html_data = _html_report_data(html_output)
    markdown = MarkdownReporter(include_timestamp=False).render_all(results)
    cli = CLIReporter().render_all(results)
    benchmark_path = tmp_path / "BENCHMARK.md"
    benchmark_path.write_text(benchmark, encoding="utf-8")
    _files, offenders = benchmark_gate.find_offenders([benchmark_path])

    assert "Overall verdict: FAIL" in benchmark
    assert "## Publication Recommendation" not in benchmark
    assert json_payload["publication_status"] == "fail"
    assert html_data["publication"]["status"] == "fail"
    assert 'data-publication-status="fail"' in html_output
    assert "**Publication status:** ❌ FAIL" in markdown
    assert "Live evaluation did not run" not in cli
    assert "[FAIL] Validation failed" in _plain_cli(cli)
    assert offenders == []


@pytest.mark.parametrize(
    ("field", "top_level", "summary_value"),
    [
        ("verdict", "pass", "fail"),
        ("verdict", "fail", "pass"),
        ("execution_status", "succeeded", "failed"),
        ("execution_status", "failed", "succeeded"),
    ],
    ids=["summary-verdict-fail", "top-verdict-fail", "summary-status-failed", "top-status-failed"],
)
def test_contradictory_tier3_truth_fields_fail_closed_across_reporters(
    tmp_path: Path,
    field: str,
    top_level: str,
    summary_value: str,
) -> None:
    tier1 = ValidationResult(validator_name="SCHEMA")
    tier1.add_success("schema", "Schema passed")
    tier2 = ValidationResult(validator_name="Similarity Check")
    tier2.add_success("similarity_check", "Similarity scan completed")
    tier3 = _complete_tier3_result("CONTRADICTORY-TRUTH", score=0.9, runtime_seconds=1.0)
    tier3.metadata["agent_eval"][field] = top_level
    tier3.metadata["agent_eval"]["summary"][field] = summary_value
    results = [tier1, tier2, tier3]

    benchmark = BenchmarkReporter(include_timestamp=False).render_all(results)
    json_payload = json.loads(JSONReporter(include_timestamp=False).render_all(results))
    html_data = _html_report_data(HTMLReporter(include_timestamp=False).render_all(results))
    markdown = MarkdownReporter(include_timestamp=False).render_all(results)
    benchmark_path = tmp_path / "BENCHMARK.md"
    benchmark_path.write_text(benchmark, encoding="utf-8")
    _files, offenders = benchmark_gate.find_offenders([benchmark_path])

    assert "Overall verdict: FAIL" in benchmark
    assert "## Publication Recommendation" not in benchmark
    assert json_payload["publication_status"] == "fail"
    assert html_data["publication"]["status"] == "fail"
    assert "**Publication status:** ❌ FAIL" in markdown
    assert offenders == []


@pytest.mark.parametrize(
    ("dimension_score", "expected_status"),
    [(0.45, "neutral"), (0.10, "fail")],
)
def test_html_displays_effective_dimension_verdict(
    dimension_score: float,
    expected_status: str,
) -> None:
    tier1 = ValidationResult(validator_name="SCHEMA")
    tier1.add_success("schema", "Schema passed")
    tier2 = ValidationResult(validator_name="Similarity Check")
    tier2.add_success("similarity_check", "Similarity scan completed")
    tier3 = _complete_tier3_result("DIMENSIONS", score=0.9, runtime_seconds=1.0)
    for dimension in tier3.metadata["agent_eval"]["agents"]["codex"]["dimensions"]:
        dimension["with_skill"] = dimension_score
    results = [tier1, tier2, tier3]
    _bind_publication_target(results, tier3.metadata["publication_target"])

    benchmark = BenchmarkReporter(include_timestamp=False).render_all(results)
    html_output = HTMLReporter(include_timestamp=False).render_all(results)
    json_payload = json.loads(JSONReporter(include_timestamp=False).render_all(results))
    markdown = MarkdownReporter(include_timestamp=False).render_all(results)

    expected_upper = expected_status.upper()
    assert f"Overall verdict: {expected_upper}" in benchmark
    assert f"| Tier 3 | Live agent evaluation | **{expected_upper}** |" in benchmark
    assert f'data-publication-status="{expected_status}"' in html_output
    assert re.search(rf'tier-card-verdict">\s*{expected_upper}\s*</span>', html_output)
    assert f"Verdict at a glance:</strong>\n                {expected_upper}" in html_output
    assert json_payload["publication_status"] == expected_status
    assert f"**Verdict:** {expected_upper}" in markdown


@pytest.mark.parametrize(
    ("verdict", "score"),
    [
        ("neutral", 0.45),
        ("fail", 0.10),
    ],
)
def test_all_reporters_fail_closed_across_conflicting_complete_tier3_payloads(
    verdict: str,
    score: float,
) -> None:
    tier1 = ValidationResult(validator_name="SCHEMA")
    tier1.add_success("schema", "Schema passed")
    tier2 = ValidationResult(validator_name="Context Deduplication")
    tier2.add_success("similarity_check", "Similarity scan completed")
    passing = _complete_tier3_result("PASSING", score=0.91, runtime_seconds=9.0)
    conservative = _complete_tier3_result("CONSERVATIVE", score=score, runtime_seconds=10.0)
    for result in (passing, conservative):
        result.metadata["agent_eval"]["skill_name"] = "same-target"
        result.metadata["agent_eval"]["summary"]["skill_name"] = "same-target"
    payload = conservative.metadata["agent_eval"]
    payload["verdict"] = verdict
    payload["summary"]["verdict"] = verdict
    for agent in payload["agents"].values():
        agent["with_skill"] = score
        for dimension in agent["dimensions"]:
            dimension["with_skill"] = score

    results = [tier1, tier2, passing, conservative]
    _bind_publication_target(results, _publication_target("same-target"))
    benchmark = BenchmarkReporter(include_timestamp=False).render_all(results)
    html_payload = HTMLReporter(include_timestamp=False)._tier3_report_data([passing, conservative])
    json_payload = json.loads(JSONReporter(include_timestamp=False).render_all(results))
    markdown = MarkdownReporter(include_timestamp=False).render_all(results)

    assert f"Overall verdict: {verdict.upper()}" in benchmark
    assert f"| Tier 3 | Live agent evaluation | **{verdict.upper()}** |" in benchmark
    assert "## Publication Recommendation" not in benchmark
    assert html_payload is not None and html_payload["marker"] == "CONSERVATIVE"
    assert html_payload["verdict"] == verdict
    assert json_payload["tier3"]["marker"] == "CONSERVATIVE"
    assert f"**Verdict:** {verdict.upper()}" in markdown


def test_all_reporters_prefer_publication_complete_payload_over_scored_decoy() -> None:
    complete = _complete_tier3_result("COMPLETE", score=0.81, runtime_seconds=8.0)
    decoy = ValidationResult(validator_name="AGENT_EVAL")
    decoy.metadata["agent_eval"] = {
        "marker": "DECOY",
        "skill_name": "decoy",
        "verdict": "pass",
        "execution_status": "succeeded",
        "scored_attempts": 1,
        "runtime_seconds": 99.0,
        "agents": {
            "decoy": {
                "execution_status": "succeeded",
                "overall_score": 0.99,
                "scored_attempts": 1,
            }
        },
    }

    reporter = HTMLReporter(include_timestamp=False)
    for results in ([complete, decoy], [decoy, complete]):
        benchmark = BenchmarkReporter(include_timestamp=False).render_all(results)
        html_payload = reporter._tier3_report_data(results)
        json_payload = json.loads(JSONReporter(include_timestamp=False).render_all(results))
        markdown = MarkdownReporter(include_timestamp=False).render_all(results)

        assert "# Skill Benchmark: candidate-complete" in benchmark
        assert html_payload is not None and html_payload["marker"] == "COMPLETE"
        assert json_payload["tier3"]["marker"] == "COMPLETE"
        assert "**Runtime:** 8.0s" in markdown


def test_all_reporters_tolerate_non_mapping_agent_payload() -> None:
    malformed = ValidationResult(validator_name="AGENT_EVAL")
    malformed.metadata["agent_eval"] = {
        "verdict": "pass",
        "execution_status": "succeeded",
        "agents": {"malformed": []},
    }

    benchmark = BenchmarkReporter(include_timestamp=False).render_all([malformed])
    html = HTMLReporter(include_timestamp=False).render_all([malformed])
    json_payload = json.loads(JSONReporter(include_timestamp=False).render_all([malformed]))
    markdown = MarkdownReporter(include_timestamp=False).render_all([malformed])

    assert "| Tier 3 | Live agent evaluation | **INCOMPLETE** |" in benchmark
    assert re.search(r'tier-card-verdict">\s*INCOMPLETE\s*</span>', html)
    assert json_payload["tier3"]["agents"] == {"malformed": []}
    assert "## Tier 3: Agent Evaluation" in markdown


@pytest.mark.parametrize(
    ("field", "malformed"),
    [
        ("verdict", 1),
        ("insights", None),
        ("insights", []),
        ("insights", {"security": 1}),
        ("suggestions_v2", 1),
        ("suggestions_v2", [1]),
        ("suggestions_v2", {"unexpected": "mapping"}),
        ("suggestions_v2", [{"recommendation": "Review", "evidence_refs": [1]}]),
        ("recommendations", 1),
    ],
)
def test_markdown_tolerates_malformed_tier3_display_fields(field: str, malformed: object) -> None:
    tier3 = _complete_tier3_result("MALFORMED", score=0.9, runtime_seconds=1.0)
    tier3.metadata["agent_eval"][field] = malformed

    markdown = MarkdownReporter(include_timestamp=False).render_all([tier3])

    assert "## Tier 3: Agent Evaluation" in markdown
    assert "## Results" in markdown


def test_markdown_flattens_untrusted_tier3_suggestion_markup() -> None:
    tier3 = _complete_tier3_result("MARKUP", score=0.9, runtime_seconds=1.0)
    tier3.metadata["agent_eval"]["suggestions_v2"] = [
        {
            "metric": "security\n## Forged metric section",
            "recommendation": "Review this\n## Forged recommendation section <script>alert(1)</script>",
            "evidence_refs": [
                {
                    "kind": "trace`\n## Forged evidence section",
                    "json_pointer": "/x`\n## Forged pointer section",
                    "excerpt": "excerpt\n## Forged excerpt section",
                }
            ],
        }
    ]

    markdown = MarkdownReporter(include_timestamp=False).render_all([tier3])

    assert "\n## Forged" not in markdown
    assert "<script>" not in markdown


def test_markdown_escapes_untrusted_tier3_evaluator_name() -> None:
    tier3 = _complete_tier3_result("EVALUATOR", score=0.9, runtime_seconds=1.0)
    injected_name = "accuracy | X\n## FORGED <script>alert(1)</script> @reviewer"
    tier3.metadata["agent_eval"]["evaluators"] = {injected_name: {"with_skill": 0.9, "baseline": 0.4, "lift": 0.5}}

    markdown = MarkdownReporter(include_timestamp=False).render_all([tier3])

    assert "\n## FORGED" not in markdown
    assert "<script>" not in markdown
    assert "&#124;" in markdown
    assert "&#64;Reviewer" in markdown


@pytest.mark.parametrize(
    "unsafe_url",
    [
        "https://good.example/) <script>alert(1)</script>",
        "https://good.example/\n## FORGED",
    ],
    ids=["closing-paren-script", "newline-heading"],
)
def test_markdown_rejects_unsafe_link_destinations(unsafe_url: str) -> None:
    tier3 = _complete_tier3_result("UNSAFE-URL", score=0.9, runtime_seconds=1.0)
    tier3.metadata["agent_eval"]["harbor_viewer"] = {"job_url": unsafe_url}
    tier3.metadata["agent_eval"]["suggestions_v2"] = [
        {
            "metric": "security",
            "recommendation": "Inspect the retained evidence",
            "harbor_evidence": {"url": unsafe_url, "label": "trace"},
        }
    ]

    markdown = MarkdownReporter(include_timestamp=False).render_all([tier3])

    assert "**Harbor logs:**" not in markdown
    assert "Evidence: [" not in markdown
    assert "<script>" not in markdown
    assert "\n## FORGED" not in markdown


def test_markdown_renders_canonical_missing_baseline_as_not_available() -> None:
    tier3 = _complete_tier3_result("A", score=0.9, runtime_seconds=3.0)
    tier3.metadata["agent_eval"]["evaluators"]["accuracy"].update({"baseline": None, "lift": None})

    markdown = MarkdownReporter(include_timestamp=False).render_all([tier3])

    assert "| Accuracy | 0.90 | N/A | N/A |" in markdown


@pytest.mark.parametrize(
    "pass_threshold",
    [[], {}, 10**10000, float("nan"), True, -1],
    ids=["list", "mapping", "huge-integer", "nan", "boolean", "negative"],
)
def test_html_normalizes_invalid_attempt_threshold_with_trials(pass_threshold: object) -> None:
    tier3 = _complete_tier3_result("A", score=0.9, runtime_seconds=3.0)
    tier3.metadata["agent_eval"]["attempt_policy"]["pass_threshold"] = pass_threshold
    tier3.metadata["agent_eval"]["trials"] = [
        {
            "agent": "codex",
            "entry_id": "case-001",
            "overall": 0.9,
            "scores": {"accuracy": 0.9},
        }
    ]

    reporter = HTMLReporter(include_timestamp=False)
    html = reporter.render_all([tier3])
    payload = reporter._tier3_report_data([tier3])

    assert payload is not None
    assert payload["attempt_policy"]["pass_threshold"] is None
    assert "Tier 3" in html


def test_html_partial_tier3_payload_wrong_types_fail_closed_without_crashing() -> None:
    malformed_fields = [
        ("agents", "not-a-mapping"),
        ("dimensions", [{"id": "accuracy", "score": "not-a-number"}]),
        ("recommendations", {}),
        ("metric_ids", 1),
        ("provenance", []),
        ("dimension_hints", []),
        ("pass_at_k", []),
        ("report_truncation", {"omitted": []}),
        ("timing", [{"name": "collect", "seconds": "not-a-number"}]),
        ("best_agent", []),
        ("trials", {}),
        ("overall_score", float("nan")),
        ("overall_score", float("inf")),
        ("overall_score", 10**10000),
    ]

    for field, value in malformed_fields:
        partial_tier3 = ValidationResult(validator_name="AGENT_EVAL")
        partial_tier3.metadata["agent_eval"] = {"verdict": "pass", field: value}

        reporter = HTMLReporter(include_timestamp=False)
        html = reporter.render_all([partial_tier3])
        payload = reporter._tier3_report_data([partial_tier3])

        assert re.search(
            r'tier-card warn.*?tier-card-label">Tier 3</span>.*?tier-card-verdict">\s*INCOMPLETE\s*</span>',
            html,
            re.DOTALL,
        )
        assert payload is not None
        assert payload["verdict"] == "incomplete"
        assert payload["execution_status"] == "incomplete"


def test_malformed_root_tier3_payload_fails_closed_without_crashing_reporters() -> None:
    tier1 = ValidationResult(validator_name="SCHEMA")
    tier1.add_success("schema", "Schema passed")
    tier1.metadata["benchmark_policy"] = {"tier2_required": False, "tier3_required": True}
    tier3 = ValidationResult(validator_name="AGENT_EVAL")
    tier3.metadata["agent_eval"] = ["malformed"]
    results = [tier1, tier3]

    benchmark = BenchmarkReporter(include_timestamp=False).render_all(results)
    json_payload = json.loads(JSONReporter(include_timestamp=False).render_all(results))
    html_data = _html_report_data(HTMLReporter(include_timestamp=False).render_all(results))
    markdown = MarkdownReporter(include_timestamp=False).render_all(results)
    cli = CLIReporter().render_all(results)

    assert "Overall verdict: INCOMPLETE" in benchmark
    assert json_payload["publication_status"] == "incomplete"
    assert json_payload["results"][1]["tier3"]["_serialization_truncated"] is True
    assert html_data["publication"]["status"] == "incomplete"
    assert "**Publication status:** ⚠️ INCOMPLETE" in markdown
    assert "Live evaluation did not run" not in cli


@pytest.mark.parametrize(
    "diagnostic",
    ["line\u202econtrol", 10**1000],
    ids=["control-normalization", "oversized-integer"],
)
def test_json_lossy_tier3_normalization_is_disclosed_and_fails_closed(diagnostic: object) -> None:
    tier1 = ValidationResult(validator_name="SCHEMA")
    tier1.add_success("schema", "Schema passed")
    tier2 = ValidationResult(validator_name="Similarity Check")
    tier2.add_success("similarity_check", "Similarity scan completed")
    tier3 = _complete_tier3_result("JSON-LOSS", score=0.9, runtime_seconds=1.0)
    tier3.metadata["agent_eval"]["diagnostic"] = diagnostic

    payload = json.loads(JSONReporter(include_timestamp=False).render_all([tier1, tier2, tier3]))

    assert payload["tier3"]["_serialization_truncated"] is True
    assert payload["publication_status"] == "incomplete"
    assert payload["publication"]["eligible"] is False


@pytest.mark.parametrize(
    "diagnostic",
    [{("tuple", "key"): "value"}, float("nan"), "x" * (3 * 1024 * 1024)],
    ids=["non-string-key", "non-finite-number", "oversized-text"],
)
def test_lossy_payload_is_bounded_and_consistent_across_reporters(
    tmp_path: Path,
    diagnostic: object,
) -> None:
    tier1 = ValidationResult(validator_name="SCHEMA")
    tier1.add_success("schema", "Schema passed")
    tier2 = ValidationResult(validator_name="Similarity Check")
    tier2.add_success("similarity_check", "Similarity scan completed")
    tier3 = _complete_tier3_result("LOSSY-KEY", score=0.9, runtime_seconds=1.0)
    tier3.metadata["agent_eval"]["diagnostic"] = diagnostic
    results = [tier1, tier2, tier3]
    _bind_publication_target(results, tier3.metadata["publication_target"])

    benchmark = BenchmarkReporter(include_timestamp=False).render_all(results)
    json_output = JSONReporter(include_timestamp=False).render_all(results)
    json_payload = json.loads(json_output)
    html_output = HTMLReporter(include_timestamp=False).render_all(results)
    html_data = _html_report_data(html_output)
    markdown = MarkdownReporter(include_timestamp=False).render_all(results)
    benchmark_path = tmp_path / "BENCHMARK.md"
    benchmark_path.write_text(benchmark, encoding="utf-8")
    _files, offenders = benchmark_gate.find_offenders([benchmark_path])

    assert "Overall verdict: INCOMPLETE" in benchmark
    assert json_payload["tier3"]["_serialization_truncated"] is True
    assert json_payload["publication_status"] == "incomplete"
    assert html_data["publication"]["status"] == "incomplete"
    assert "**Publication status:** ⚠️ INCOMPLETE" in markdown
    assert len(json_output.encode("utf-8")) < 2 * 1024 * 1024
    assert len(html_output.encode("utf-8")) < 2 * 1024 * 1024
    assert offenders == []


@pytest.mark.parametrize("shape", ["cycle", "deep"])
def test_html_tier3_sanitizer_bounds_recursive_metadata(shape: str) -> None:
    tier3 = _complete_tier3_result("RECURSIVE", score=0.9, runtime_seconds=1.0)
    payload = tier3.metadata["agent_eval"]
    if shape == "cycle":
        payload["diagnostic"] = payload
    else:
        nested: dict[str, object] = {}
        payload["diagnostic"] = nested
        for _ in range(20_000):
            child: dict[str, object] = {}
            nested["child"] = child
            nested = child

    output = HTMLReporter(include_timestamp=False).render_all([tier3])
    benchmark = BenchmarkReporter(include_timestamp=False).render_all([tier3])
    json_output = JSONReporter(include_timestamp=False).render_all([tier3])
    markdown = MarkdownReporter(include_timestamp=False).render_all([tier3])

    assert "Tier 3" in output
    assert '<script id="report-data" type="application/json">' in output
    assert "Overall verdict:" in benchmark
    assert json.loads(json_output)["publication_status"] == "incomplete"
    assert "## Tier 3: Agent Evaluation" in markdown


def test_html_agent_name_cannot_escape_into_inline_javascript() -> None:
    injected_name = "x');globalThis.__pwned=1;//"
    tier3 = _complete_tier3_result("XSS", score=0.9, runtime_seconds=1.0)
    agent = tier3.metadata["agent_eval"]["agents"].pop("codex")
    tier3.metadata["agent_eval"]["agents"][injected_name] = agent

    output = HTMLReporter(include_timestamp=False).render_all([tier3])
    handlers = [unescape(value) for value in re.findall(r'onclick="([^"]*)"', output)]

    assert injected_name in unescape(output)
    assert handlers
    assert all("__pwned" not in handler for handler in handlers)


def test_tier3_serialization_budget_fails_publication_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("skillevaluator.reporting.base.AGENT_EVAL_REPORT_MAX_NODES", 500)
    tier1 = ValidationResult(validator_name="SCHEMA")
    tier1.add_success("schema", "Schema passed")
    tier2 = ValidationResult(validator_name="Similarity Check")
    tier2.add_success("similarity_check", "Similarity scan completed")
    tier3 = _complete_tier3_result("BOUNDED", score=0.9, runtime_seconds=1.0)
    payload = tier3.metadata["agent_eval"]
    tier3.metadata["agent_eval"] = {
        "diagnostic": {f"item-{index}": index for index in range(1_000)},
        **payload,
    }
    results = [tier1, tier2, tier3]

    json_payload = json.loads(JSONReporter(include_timestamp=False).render_all(results))
    html_output = HTMLReporter(include_timestamp=False).render_all(results)
    html_payload = HTMLReporter(include_timestamp=False)._tier3_report_data([tier3])
    benchmark = BenchmarkReporter(include_timestamp=False).render_all(results)

    assert json_payload["publication_status"] == "incomplete"
    assert json_payload["publication"]["eligible"] is False
    assert json_payload["publication"]["tier3"]["evidence_complete"] is False
    assert json_payload["tier3"]["verdict"] == "pass"
    assert json_payload["tier3"]["execution_status"] == "succeeded"
    assert json_payload["tier3"]["agents"]
    assert json_payload["tier3"]["_serialization_truncated"] is True
    assert html_payload is not None and html_payload["verdict"] == "pass"
    assert html_payload["agents"]
    assert html_payload["_serialization_truncated"] is True
    assert 'data-publication-status="incomplete"' in html_output
    assert "| Tier 3 | Live agent evaluation | **INCOMPLETE** |" in benchmark


def test_tier3_serialization_rejects_huge_text_before_normalizing_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import skillevaluator.reporting.base as reporting_base
    import skillevaluator.reporting.html as html_reporter

    huge_text = "x" * (reporting_base.AGENT_EVAL_REPORT_MAX_TEXT_BYTES + 1)
    original_semantic_text = reporting_base.publication_semantic_text
    original_json_safe_text = html_reporter._json_safe_tier3_text

    def semantic_text_spy(value: object, *, strip_marks: bool = False) -> str:
        assert value is not huge_text, "oversized text was normalized after its length already proved rejection"
        return original_semantic_text(value, strip_marks=strip_marks)

    def json_safe_text_spy(value: str) -> str:
        assert value is not huge_text, "oversized text was encoded by the bounded reporter"
        return original_json_safe_text(value)

    monkeypatch.setattr(reporting_base, "publication_semantic_text", semantic_text_spy)
    monkeypatch.setattr(html_reporter, "_json_safe_tier3_text", json_safe_text_spy)

    tier3 = _complete_tier3_result("HUGE-TEXT", score=0.9, runtime_seconds=1.0)
    tier3.metadata["agent_eval"]["diagnostic"] = huge_text

    issue = reporting_base.agent_eval_report_serialization_issue(tier3.metadata["agent_eval"])
    json_payload = json.loads(JSONReporter(include_timestamp=False).render_all([tier3]))

    assert issue == (
        f"The Tier 3 payload exceeds the {reporting_base.AGENT_EVAL_REPORT_MAX_TEXT_BYTES:,}-byte report text limit."
    )
    assert json_payload["publication_status"] == "incomplete"
    assert json_payload["publication"]["tier3"]["evidence_complete"] is False
    assert json_payload["tier3"]["_serialization_truncated"] is True
    assert json_payload["tier3"].get("diagnostic") is None


def test_tier3_bounded_copy_stops_before_iterating_or_loading_wide_map_tail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import skillevaluator.reporting.base as reporting_base
    from skillevaluator.reporting.html import _json_safe_tier3_payload

    class TailGuardDict(dict[str, object]):
        def __init__(self) -> None:
            super().__init__((f"diagnostic-{index}", index) for index in range(1_000))
            self.iterated: list[str] = []
            self.loaded: list[str] = []

        def __iter__(self):
            for key in super().__iter__():
                if len(self.iterated) >= 3:
                    raise AssertionError("wide mapping tail was eagerly materialized")
                self.iterated.append(key)
                yield key

        def __getitem__(self, key: str) -> object:
            if len(self.loaded) >= 3:
                raise AssertionError("wide mapping tail value was loaded")
            self.loaded.append(key)
            return super().__getitem__(key)

    monkeypatch.setattr(reporting_base, "AGENT_EVAL_REPORT_MAX_NODES", 4)
    wide_payload = TailGuardDict()

    issue = reporting_base.agent_eval_report_serialization_issue(wide_payload)

    assert issue == "The Tier 3 payload exceeds the 4-node report limit."
    assert wide_payload.iterated == []
    assert wide_payload.loaded == []

    emitted = _json_safe_tier3_payload(wide_payload)

    assert emitted["_serialization_truncated"] is True
    assert [key for key in emitted if key != "_serialization_truncated"] == [
        "diagnostic-0",
        "diagnostic-1",
        "diagnostic-2",
    ]
    assert wide_payload.iterated == ["diagnostic-0", "diagnostic-1", "diagnostic-2"]
    assert wide_payload.loaded == ["diagnostic-0", "diagnostic-1", "diagnostic-2"]


def test_invalid_tier3_fingerprint_uses_bounded_fallback_without_full_json_dump(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import skillevaluator.reporting.base as reporting_base

    huge_text = "x" * (reporting_base.AGENT_EVAL_REPORT_MAX_TEXT_BYTES + 1)
    payload = {"diagnostic": huge_text}
    original_dumps = reporting_base.json.dumps

    def dumps_spy(value: object, *args: object, **kwargs: object) -> str:
        assert value is not payload, "invalid raw payload was passed to an unbounded canonical JSON dump"
        return original_dumps(value, *args, **kwargs)

    monkeypatch.setattr(reporting_base.json, "dumps", dumps_spy)

    first = reporting_base._agent_eval_payload_fingerprint(payload)
    second = reporting_base._agent_eval_payload_fingerprint(payload)

    assert re.fullmatch(r"[0-9a-f]{64}", first)
    assert second == first


@pytest.mark.parametrize(
    ("explicit_failure", "expected_tier3_status"),
    [(False, "incomplete"), (True, "fail")],
    ids=["rejected-pass-claim", "rejected-explicit-failure"],
)
def test_reporters_bound_wide_agents_before_selection_and_assessment(
    monkeypatch: pytest.MonkeyPatch,
    *,
    explicit_failure: bool,
    expected_tier3_status: str,
) -> None:
    import skillevaluator.reporting.base as reporting_base

    tail_key = "tail-agent-must-not-be-loaded"

    class GuardedAgents(dict[str, object]):
        def __init__(self) -> None:
            super().__init__(
                {
                    **{
                        f"agent-{index:04d}": {"model": "guarded-model", "execution_status": "succeeded"}
                        for index in range(1_000)
                    },
                    tail_key: object(),
                }
            )
            self.loaded: list[str] = []

        def __iter__(self):
            for index, key in enumerate(super().__iter__()):
                if index >= 64:
                    raise AssertionError("wide agents tail was iterated beyond the bounded fallback sample")
                yield key

        def __getitem__(self, key: str) -> object:
            assert key != tail_key, "wide agents tail value was loaded"
            self.loaded.append(key)
            return super().__getitem__(key)

        def items(self):
            raise AssertionError("wide agents mapping was materialized through items()")

        def values(self):
            raise AssertionError("wide agents mapping was traversed through values()")

    monkeypatch.setattr(reporting_base, "AGENT_EVAL_REPORT_MAX_NODES", 64)
    tier3 = _complete_tier3_result("WIDE-AGENTS", score=0.9, runtime_seconds=1.0)
    payload = tier3.metadata["agent_eval"]
    payload["agents"] = GuardedAgents()
    if explicit_failure:
        payload["verdict"] = "fail"
        payload["execution_status"] = "failed"
        payload["summary"]["verdict"] = "fail"
        payload["summary"]["execution_status"] = "failed"

    json_payload = json.loads(JSONReporter(include_timestamp=False).render_all([tier3]))
    html_output = HTMLReporter(include_timestamp=False).render_all([tier3])
    html_report_data = _html_report_data(html_output)
    html_tier3 = _html_tier3_payload(html_output)

    assert json_payload["publication"]["tier3"]["status"] == expected_tier3_status
    assert json_payload["publication"]["tier3"]["evidence_complete"] is False
    assert json_payload["tier3"]["_serialization_truncated"] is True
    assert html_report_data["publication"]["tier3"]["status"] == expected_tier3_status
    assert html_report_data["publication"]["tier3"]["evidence_complete"] is False
    assert html_tier3["verdict"] == expected_tier3_status
    assert html_tier3["_serialization_truncated"] is True
    assert tail_key not in payload["agents"].loaded


def test_tier3_serialization_cannot_omit_proof_while_reporting_publication_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("skillevaluator.reporting.base.AGENT_EVAL_REPORT_MAX_NODES", 250)
    tier1 = ValidationResult(validator_name="SCHEMA")
    tier1.add_success("schema", "Schema passed")
    tier2 = ValidationResult(validator_name="Similarity Check")
    tier2.add_success("similarity_check", "Similarity scan completed")
    tier3 = _complete_tier3_result("PROOF-LAST", score=0.9, runtime_seconds=1.0)
    canonical = tier3.metadata["agent_eval"]["agents"]["codex"]
    tier3.metadata["agent_eval"]["agents"] = {
        **{f"decoy-{index:03d}": {"model": "decoy"} for index in range(300)},
        "codex": canonical,
    }
    results = [tier1, tier2, tier3]

    json_payload = json.loads(JSONReporter(include_timestamp=False).render_all(results))
    html_output = HTMLReporter(include_timestamp=False).render_all(results)
    html_data = _html_report_data(html_output)
    benchmark = BenchmarkReporter(include_timestamp=False).render_all(results)
    markdown = MarkdownReporter(include_timestamp=False).render_all(results)

    assert json_payload["tier3"]["_serialization_truncated"] is True
    assert "codex" not in json_payload["tier3"]["agents"]
    assert json_payload["publication_status"] == "incomplete"
    assert json_payload["publication"]["tier3"]["evidence_complete"] is False
    assert html_data["publication"]["status"] == "incomplete"
    assert html_data["publication"]["tier3"]["evidence_complete"] is False
    assert 'data-publication-status="incomplete"' in html_output
    assert "Overall verdict: INCOMPLETE" in benchmark
    assert "| Tier 3 | Live agent evaluation | **INCOMPLETE** |" in benchmark
    assert "**Publication status:** ⚠️ INCOMPLETE" in markdown


@pytest.mark.parametrize(
    ("verdict", "score", "expected_status", "eligible"),
    [
        ("pass", 0.90, "pass", True),
        ("neutral", 0.45, "neutral", False),
        ("fail", 0.10, "fail", False),
    ],
)
def test_publication_status_matrix_is_consistent_across_reporters(
    verdict: str,
    score: float,
    expected_status: str,
    eligible: bool,
) -> None:
    tier1 = ValidationResult(validator_name="SCHEMA")
    tier1.add_success("schema", "Schema passed")
    tier2 = ValidationResult(validator_name="Similarity Check")
    tier2.add_success("similarity_check", "Similarity scan completed")
    tier3 = _complete_tier3_result("MATRIX", score=score, runtime_seconds=1.0)
    payload = tier3.metadata["agent_eval"]
    payload["verdict"] = verdict
    payload["summary"]["verdict"] = verdict
    for agent in payload["agents"].values():
        agent["with_skill"] = score
        for dimension in agent["dimensions"]:
            dimension["with_skill"] = score
    results = [tier1, tier2, tier3]
    _bind_publication_target(results, tier3.metadata["publication_target"])

    benchmark = BenchmarkReporter(include_timestamp=False).render_all(results)
    json_payload = json.loads(JSONReporter(include_timestamp=False).render_all(results))
    html_output = HTMLReporter(include_timestamp=False).render_all(results)
    html_data = _html_report_data(html_output)
    markdown = MarkdownReporter(include_timestamp=False).render_all(results)

    assert f"Overall verdict: {expected_status.upper()}" in benchmark
    assert json_payload["publication_status"] == expected_status
    assert json_payload["publication"]["eligible"] is eligible
    assert html_data["publication"]["status"] == expected_status
    assert html_data["publication"]["eligible"] is eligible
    assert f'data-publication-status="{expected_status}"' in html_output
    assert f"Publication: {expected_status.title()}" in html_output
    assert "**Publication status:**" in markdown


@pytest.mark.parametrize(
    ("field", "value", "metadata_label"),
    [
        ("evaluated_at", "banana", "Evaluation date"),
        ("evaluated_at", "2026-08-24", "Evaluation date"),
        ("evaluated_at", "2026-08-24T12:00:00", "Evaluation date"),
        ("evaluated_at", "9999-01-01T00:00:00+00:00", "Evaluation date"),
        ("dataset_digest", "x", "Dataset digest"),
        ("dataset_digest_algorithm", "sha256", "Dataset digest"),
    ],
)
def test_malformed_tier3_provenance_cannot_certify_publication(
    tmp_path: Path,
    field: str,
    value: str,
    metadata_label: str,
) -> None:
    tier1 = ValidationResult(validator_name="SCHEMA")
    tier1.add_success("schema", "Schema passed")
    tier2 = ValidationResult(validator_name="Similarity Check")
    tier2.add_success("similarity_check", "Similarity scan completed")
    tier3 = _complete_tier3_result("PROVENANCE", score=0.9, runtime_seconds=1.0)
    tier3.metadata["agent_eval"][field] = value
    results = [tier1, tier2, tier3]

    benchmark = BenchmarkReporter(include_timestamp=False).render_all(results)
    json_payload = json.loads(JSONReporter(include_timestamp=False).render_all(results))
    html_data = _html_report_data(HTMLReporter(include_timestamp=False).render_all(results))
    markdown = MarkdownReporter(include_timestamp=False).render_all(results)
    benchmark_path = tmp_path / "BENCHMARK.md"
    benchmark_path.write_text(benchmark, encoding="utf-8")
    _files, offenders = benchmark_gate.find_offenders([benchmark_path])

    assert "Overall verdict: INCOMPLETE" in benchmark
    assert "## Publication Recommendation" not in benchmark
    assert "| Tier 3 | Live agent evaluation | **INCOMPLETE** |" in benchmark
    assert f"- {metadata_label}: not recorded (legacy or non-live result)" in benchmark
    assert json_payload["publication_status"] == "incomplete"
    assert html_data["publication"]["status"] == "incomplete"
    assert "**Publication status:** ⚠️ INCOMPLETE" in markdown
    assert offenders == []


@pytest.mark.parametrize(
    "mutation",
    [
        "dataset-digest-value",
        "evaluated-at-value",
        "algorithm-value",
        "verdict-value",
        "execution-status-value",
        "dimension-id-value",
        "attempt-policy-key",
        "agents-key",
        "model-key",
        "task-count-key",
        "environment-key",
    ],
)
def test_json_normalization_cannot_invent_publication_evidence(mutation: str) -> None:
    tier1 = ValidationResult(validator_name="SCHEMA")
    tier1.add_success("schema", "Schema passed")
    tier2 = ValidationResult(validator_name="Similarity Check")
    tier2.add_success("similarity_check", "Similarity scan completed")
    tier3 = _complete_tier3_result("JSON-RAW-EVIDENCE", score=0.9, runtime_seconds=1.0)
    payload = tier3.metadata["agent_eval"]
    control = "\u200b"

    if mutation == "dataset-digest-value":
        payload["dataset_digest"] += control
    elif mutation == "evaluated-at-value":
        payload["evaluated_at"] += control
    elif mutation == "algorithm-value":
        payload["dataset_digest_algorithm"] += control
    elif mutation == "verdict-value":
        payload["verdict"] = f"pa{control}ss"
    elif mutation == "execution-status-value":
        payload["execution_status"] = f"suc{control}ceeded"
    elif mutation == "dimension-id-value":
        payload["agents"]["codex"]["dimensions"][0]["id"] = f"secu{control}rity"
    elif mutation == "attempt-policy-key":
        payload["attempt_policy"][f"max{control}_attempts"] = payload["attempt_policy"].pop("max_attempts")
    elif mutation == "agents-key":
        payload[f"age{control}nts"] = payload.pop("agents")
    elif mutation == "model-key":
        agent = payload["agents"]["codex"]
        agent[f"mo{control}del"] = agent.pop("model")
    elif mutation == "task-count-key":
        summary = payload["dataset_summary"]
        summary[f"total{control}_tasks"] = summary.pop("total_tasks")
    else:
        summary = payload["summary"]
        summary[f"environ{control}ment"] = summary.pop("environment")

    json_payload = json.loads(JSONReporter(include_timestamp=False).render_all([tier1, tier2, tier3]))

    assert json_payload["publication_status"] == "incomplete"
    assert json_payload["publication"]["eligible"] is False


def test_json_normalization_cannot_invent_optional_tier_policy() -> None:
    tier1 = ValidationResult(validator_name="SCHEMA")
    tier1.add_success("schema", "Schema passed")
    tier3 = _complete_tier3_result("JSON-RAW-POLICY", score=0.9, runtime_seconds=1.0)
    tier3.metadata["agent_eval"]["benchmark_policy"] = {
        "tier2\u200b_required": False,
        "tier3_required": True,
    }

    json_payload = json.loads(JSONReporter(include_timestamp=False).render_all([tier1, tier3]))

    assert json_payload["publication_status"] == "incomplete"
    assert json_payload["benchmark_policy"]["tier2_required"] is True


@pytest.mark.parametrize(
    ("identity", "identity_kind"),
    [
        ("\u200b", "format-control"),
        ("\x00", "control"),
        ("\ufe0f", "variation-selector"),
        ("\u034f", "combining-grapheme-joiner"),
        ("\u20dd", "enclosing-mark"),
        ("\u115f", "hangul-filler"),
        ("\u2800", "braille-blank"),
        ("\U00013441", "hieroglyph-full-blank"),
        ("\U00013442", "hieroglyph-half-blank"),
        ("\U0001d159", "musical-null-notehead"),
        ("not recorded", "reserved-placeholder"),
        ("model not recorded", "reserved-model-placeholder"),
        ("not recor\u0301ded", "reserved-combining-placeholder"),
        ("not recor\ufe0fded", "reserved-variation-placeholder"),
        ("not recor\u034fded", "reserved-joiner-placeholder"),
        ("not recor\u20ddded", "reserved-enclosing-placeholder"),
        ("not recorded.", "reserved-punctuated-placeholder"),
        ("(not recorded)", "reserved-parenthesized-placeholder"),
        ("model not recorded!", "reserved-punctuated-model-placeholder"),
        ("unkn\u043ewn", "reserved-cyrillic-confusable-placeholder"),
        ("unkn\u03bfwn", "reserved-greek-confusable-placeholder"),
        ("m\u043edel not recorded", "reserved-confusable-model-placeholder"),
        ("modeI not recorded", "reserved-ascii-confusable-model-placeholder"),
        ("mode\u0406 not recorded", "reserved-cyrillic-i-confusable-model-placeholder"),
        ("unkn0wn", "reserved-digit-confusable-placeholder"),
        ("\u039codel not recorded", "reserved-greek-mu-confusable-model-placeholder"),
        ("model not re\u03f2orded", "reserved-greek-lunate-sigma-placeholder"),
        ("unkn\u0c02wn", "reserved-telugu-mark-confusable-placeholder"),
        ("unkn\U0001cce4wn", "reserved-unicode17-confusable-placeholder"),
        ("mode\u0140not recorded", "reserved-trailing-separator-confusable-placeholder"),
        ("model\u0149ot recorded", "reserved-leading-separator-confusable-placeholder"),
        ("unknow\u145amodel", "reserved-folded-separator-confusable-placeholder"),
        ("u\u0295nknown", "reserved-vanishing-letter-confusable-placeholder"),
        ("u\U0001f40dnknown", "reserved-vanishing-symbol-confusable-placeholder"),
        ("u\u6138nknown", "reserved-vanishing-cjk-confusable-placeholder"),
        (123, "non-string-identity"),
    ],
    ids=[
        "format-control",
        "control",
        "variation-selector",
        "combining-grapheme-joiner",
        "enclosing-mark",
        "hangul-filler",
        "braille-blank",
        "hieroglyph-full-blank",
        "hieroglyph-half-blank",
        "musical-null-notehead",
        "reserved-placeholder",
        "reserved-model-placeholder",
        "reserved-combining-placeholder",
        "reserved-variation-placeholder",
        "reserved-joiner-placeholder",
        "reserved-enclosing-placeholder",
        "reserved-punctuated-placeholder",
        "reserved-parenthesized-placeholder",
        "reserved-punctuated-model-placeholder",
        "reserved-cyrillic-confusable-placeholder",
        "reserved-greek-confusable-placeholder",
        "reserved-confusable-model-placeholder",
        "reserved-ascii-confusable-model-placeholder",
        "reserved-cyrillic-i-confusable-model-placeholder",
        "reserved-digit-confusable-placeholder",
        "reserved-greek-mu-confusable-model-placeholder",
        "reserved-greek-lunate-sigma-placeholder",
        "reserved-telugu-mark-confusable-placeholder",
        "reserved-unicode17-confusable-placeholder",
        "reserved-trailing-separator-confusable-placeholder",
        "reserved-leading-separator-confusable-placeholder",
        "reserved-folded-separator-confusable-placeholder",
        "reserved-vanishing-letter-confusable-placeholder",
        "reserved-vanishing-symbol-confusable-placeholder",
        "reserved-vanishing-cjk-confusable-placeholder",
        "non-string-identity",
    ],
)
@pytest.mark.parametrize("field", ["agent_key", "model", "evaluator_version", "environment"])
def test_semantically_empty_identity_cannot_certify_publication_across_reporters(
    tmp_path: Path,
    identity: object,
    identity_kind: str,
    field: str,
) -> None:
    assert identity_kind
    tier1 = ValidationResult(validator_name="SCHEMA")
    tier1.add_success("schema", "Schema passed")
    _stamp_publication_evidence(tier1, tier=1, check_id="schema")
    tier2 = ValidationResult(validator_name="Similarity Check")
    tier2.add_success("similarity_check", "Similarity scan completed")
    _stamp_publication_evidence(tier2, tier=2, check_id="similarity")
    tier3 = _complete_tier3_result("IDENTITY", score=0.9, runtime_seconds=1.0)
    payload = tier3.metadata["agent_eval"]
    if field == "agent_key":
        agent = payload["agents"].pop("codex")
        payload["agents"][identity] = agent
    elif field == "model":
        payload["agents"]["codex"]["model"] = identity
    elif field == "evaluator_version":
        payload["evaluator_version"] = identity
    else:
        payload["summary"]["environment"] = identity
    results = [tier1, tier2, tier3]

    benchmark = BenchmarkReporter(include_timestamp=False).render_all(results)
    json_payload = json.loads(JSONReporter(include_timestamp=False).render_all(results))
    html_data = _html_report_data(HTMLReporter(include_timestamp=False).render_all(results))
    markdown = MarkdownReporter(include_timestamp=False).render_all(results)
    benchmark_path = tmp_path / "BENCHMARK.md"
    benchmark_path.write_text(benchmark, encoding="utf-8")
    _files, offenders = benchmark_gate.find_offenders([benchmark_path])

    assert "Overall verdict: INCOMPLETE" in benchmark
    assert "## Publication Recommendation" not in benchmark
    assert "| Tier 3 | Live agent evaluation | **INCOMPLETE** |" in benchmark
    assert json_payload["publication_status"] == "incomplete"
    assert html_data["publication"]["status"] == "incomplete"
    assert "**Publication status:** ⚠️ INCOMPLETE" in markdown
    assert offenders == []


def test_compatibility_distinct_publication_names_cannot_mix_across_tiers(tmp_path: Path) -> None:
    ascii_target = _publication_target("demo")
    fullwidth_target = {**ascii_target, "skill_name": "\uff44\uff45\uff4d\uff4f"}
    tier1 = ValidationResult(validator_name="SCHEMA")
    tier1.add_success("schema", "Schema passed")
    _stamp_publication_evidence(tier1, tier=1, check_id="schema")
    tier2 = ValidationResult(validator_name="Similarity Check")
    tier2.add_success("similarity_check", "Similarity scan completed")
    _stamp_publication_evidence(tier2, tier=2, check_id="similarity")
    tier3 = _complete_tier3_result("EXACT-NAME", score=0.9, runtime_seconds=1.0)
    payload = tier3.metadata["agent_eval"]
    payload["skill_name"] = "demo"
    payload["summary"]["skill_name"] = "demo"
    _bind_publication_target([tier1], fullwidth_target)
    _bind_publication_target([tier2, tier3], ascii_target)
    results = [tier1, tier2, tier3]

    benchmark = BenchmarkReporter(skill_name="demo", include_timestamp=False).render_all(results)
    json_payload = json.loads(JSONReporter(include_timestamp=False, expected_skill_name="demo").render_all(results))
    html_data = _html_report_data(HTMLReporter(include_timestamp=False, expected_skill_name="demo").render_all(results))
    markdown = MarkdownReporter(include_timestamp=False, expected_skill_name="demo").render_all(results)
    benchmark_path = tmp_path / "BENCHMARK.md"
    benchmark_path.write_text(benchmark, encoding="utf-8")
    _files, offenders = benchmark_gate.find_offenders([benchmark_path])

    assert "Overall verdict: INCOMPLETE" in benchmark
    assert "## Publication Recommendation" not in benchmark
    assert json_payload["publication_status"] == "incomplete"
    assert json_payload["results"][0]["publication_target"]["skill_name"] == "\uff44\uff45\uff4d\uff4f"
    assert html_data["publication"]["status"] == "incomplete"
    assert "**Publication status:** \u26a0\ufe0f INCOMPLETE" in markdown
    assert offenders == []


def test_exact_fullwidth_publication_name_passes_when_every_tier_agrees(tmp_path: Path) -> None:
    skill_name = "\uff44\uff45\uff4d\uff4f"
    target = _publication_target(skill_name)
    tier1 = ValidationResult(validator_name="SCHEMA")
    tier1.add_success("schema", "Schema passed")
    tier2 = ValidationResult(validator_name="Similarity Check")
    tier2.add_success("similarity_check", "Similarity scan completed")
    tier3 = _complete_tier3_result("FULLWIDTH", score=0.9, runtime_seconds=1.0)
    payload = tier3.metadata["agent_eval"]
    payload["skill_name"] = skill_name
    payload["summary"]["skill_name"] = skill_name
    results = [tier1, tier2, tier3]
    _bind_publication_target(results, target)

    benchmark = BenchmarkReporter(skill_name=skill_name, include_timestamp=False).render_all(results)
    json_payload = json.loads(JSONReporter(include_timestamp=False, expected_skill_name=skill_name).render_all(results))
    html_data = _html_report_data(
        HTMLReporter(include_timestamp=False, expected_skill_name=skill_name).render_all(results)
    )
    mismatched_expected = json.loads(
        JSONReporter(include_timestamp=False, expected_skill_name="demo").render_all(results)
    )
    benchmark_path = tmp_path / "BENCHMARK.md"
    benchmark_path.write_text(benchmark, encoding="utf-8")
    _files, offenders = benchmark_gate.find_offenders([benchmark_path])

    assert "Overall verdict: PASS" in benchmark
    assert f"# Skill Benchmark: {skill_name}" in benchmark
    assert json_payload["publication_status"] == "pass"
    assert html_data["publication"]["status"] == "pass"
    assert {item["publication_target"]["skill_name"] for item in json_payload["results"]} == {skill_name}
    assert mismatched_expected["publication_status"] == "incomplete"
    assert offenders == []


def test_case_distinct_expected_publication_name_cannot_match_source_evidence(tmp_path: Path) -> None:
    source_name = "Demo"
    target = _publication_target(source_name)
    tier1 = ValidationResult(validator_name="SCHEMA")
    tier1.add_success("schema", "Schema passed")
    tier2 = ValidationResult(validator_name="Similarity Check")
    tier2.add_success("similarity_check", "Similarity scan completed")
    tier3 = _complete_tier3_result("CASE-NAME", score=0.9, runtime_seconds=1.0)
    payload = tier3.metadata["agent_eval"]
    payload["skill_name"] = source_name
    payload["summary"]["skill_name"] = source_name
    results = [tier1, tier2, tier3]
    _bind_publication_target(results, target)

    benchmark = BenchmarkReporter(skill_name="demo", include_timestamp=False).render_all(results)
    json_payload = json.loads(JSONReporter(include_timestamp=False, expected_skill_name="demo").render_all(results))
    html_data = _html_report_data(HTMLReporter(include_timestamp=False, expected_skill_name="demo").render_all(results))
    markdown = MarkdownReporter(include_timestamp=False, expected_skill_name="demo").render_all(results)
    benchmark_path = tmp_path / "BENCHMARK.md"
    benchmark_path.write_text(benchmark, encoding="utf-8")
    _files, offenders = benchmark_gate.find_offenders([benchmark_path])

    assert "Overall verdict: INCOMPLETE" in benchmark
    assert "## Publication Recommendation" not in benchmark
    assert json_payload["publication_status"] == "incomplete"
    assert html_data["publication"]["status"] == "incomplete"
    assert "**Publication status:** ⚠️ INCOMPLETE" in markdown
    assert offenders == []


def test_tier3_payload_name_must_exactly_match_its_publication_target() -> None:
    target = _publication_target("\uff44\uff45\uff4d\uff4f")
    tier1 = ValidationResult(validator_name="SCHEMA")
    tier1.add_success("schema", "Schema passed")
    tier2 = ValidationResult(validator_name="Similarity Check")
    tier2.add_success("similarity_check", "Similarity scan completed")
    tier3 = _complete_tier3_result("NAME-SPOOF", score=0.9, runtime_seconds=1.0)
    payload = tier3.metadata["agent_eval"]
    payload["skill_name"] = "demo"
    payload["summary"]["skill_name"] = "demo"
    results = [tier1, tier2, tier3]
    _bind_publication_target(results, target)

    json_payload = json.loads(JSONReporter(include_timestamp=False).render_all(results))

    assert json_payload["publication_status"] == "incomplete"
    assert "different target skill" in " ".join(json_payload["publication"]["reasons"])


@pytest.mark.parametrize("dual_tier", [1, 2])
def test_tier3_result_cannot_double_as_required_tier1_or_tier2_evidence(
    tmp_path: Path,
    dual_tier: int,
) -> None:
    target = _publication_target("demo")
    tier3 = _complete_tier3_result("DUAL-TIER", score=0.9, runtime_seconds=1.0)
    payload = tier3.metadata["agent_eval"]
    payload["skill_name"] = "demo"
    payload["summary"]["skill_name"] = "demo"
    if dual_tier == 1:
        peer = ValidationResult(validator_name="Similarity Check")
        peer.add_success("similarity_check", "Similarity scan completed")
        _stamp_publication_evidence(tier3, tier=1, check_id="schema")
        missing_row = "| Tier 1 | Static validation | **NOT RUN** |"
    else:
        peer = ValidationResult(validator_name="SCHEMA")
        peer.add_success("schema", "Schema passed")
        _stamp_publication_evidence(tier3, tier=2, check_id="similarity")
        missing_row = "| Tier 2 | Semantic deduplication | **NOT RUN** |"
    results = [peer, tier3]
    _bind_publication_target(results, target)

    benchmark = BenchmarkReporter(skill_name="demo", include_timestamp=False).render_all(results)
    json_payload = json.loads(JSONReporter(include_timestamp=False, expected_skill_name="demo").render_all(results))
    benchmark_path = tmp_path / "BENCHMARK.md"
    benchmark_path.write_text(benchmark, encoding="utf-8")
    _files, offenders = benchmark_gate.find_offenders([benchmark_path])

    assert "Overall verdict: INCOMPLETE" in benchmark
    assert missing_row in benchmark
    assert json_payload["publication_status"] == "incomplete"
    assert offenders == []


@pytest.mark.parametrize(
    "attack",
    ["unknown-tier1", "unknown-tier2", "unknown-policy-waiver", "unknown-policy-requirement"],
)
def test_unrecognized_producers_cannot_certify_or_waive_publication(tmp_path: Path, attack: str) -> None:
    target = _publication_target("demo")
    tier1 = ValidationResult(validator_name="SCHEMA")
    tier1.add_success("schema", "Schema passed")
    _stamp_publication_evidence(tier1, tier=1, check_id="schema")
    tier2 = ValidationResult(validator_name="Similarity Check")
    tier2.add_success("similarity_check", "Similarity scan completed")
    _stamp_publication_evidence(tier2, tier=2, check_id="similarity")
    tier3 = _complete_tier3_result("PRODUCER", score=0.9, runtime_seconds=1.0)
    payload = tier3.metadata["agent_eval"]
    payload["skill_name"] = "demo"
    payload["summary"]["skill_name"] = "demo"

    if attack == "unknown-tier1":
        tier1 = ValidationResult(validator_name="UNRELATED POLICY CARRIER")
        tier1.add_success("unrelated", "Unrelated check completed")
        results = [tier1, tier2, tier3]
    elif attack == "unknown-tier2":
        tier2 = ValidationResult(validator_name="UNRELATED SIMILARITY POLICY CARRIER")
        tier2.add_success("unrelated", "Unrelated check completed")
        results = [tier1, tier2, tier3]
    elif attack == "unknown-policy-waiver":
        carrier = ValidationResult(validator_name="UNRELATED POLICY CARRIER")
        carrier.add_success("unrelated", "Unrelated check completed")
        carrier.metadata["benchmark_policy"] = {"tier2_required": False, "tier3_required": True}
        results = [tier1, carrier, tier3]
    else:
        carrier = ValidationResult(validator_name="UNRELATED POLICY CARRIER")
        carrier.add_success("unrelated", "Unrelated check completed")
        carrier.metadata["benchmark_policy"] = {"tier2_required": True, "tier3_required": True}
        payload["benchmark_policy"] = {"tier2_required": False, "tier3_required": True}
        payload["summary"]["benchmark_policy"] = {"tier2_required": False, "tier3_required": True}
        results = [tier1, carrier, tier3]
    _bind_publication_target(results, target)

    benchmark = BenchmarkReporter(skill_name="demo", include_timestamp=False).render_all(results)
    json_payload = json.loads(JSONReporter(include_timestamp=False, expected_skill_name="demo").render_all(results))
    html_data = _html_report_data(HTMLReporter(include_timestamp=False, expected_skill_name="demo").render_all(results))
    markdown = MarkdownReporter(include_timestamp=False, expected_skill_name="demo").render_all(results)
    benchmark_path = tmp_path / "BENCHMARK.md"
    benchmark_path.write_text(benchmark, encoding="utf-8")
    _files, offenders = benchmark_gate.find_offenders([benchmark_path])

    assert "Overall verdict: INCOMPLETE" in benchmark
    assert "## Publication Recommendation" not in benchmark
    assert json_payload["publication_status"] == "incomplete"
    assert json_payload["benchmark_policy"]["tier2_required"] is True
    assert html_data["publication"]["status"] == "incomplete"
    assert "**Publication status:** \u26a0\ufe0f INCOMPLETE" in markdown
    assert offenders == []


@pytest.mark.parametrize(
    ("identity", "expected_status"),
    [("Cafe\u0301", "pass"), ("エージェント", "pass"), ("🤖\ufe0f", "incomplete")],
    ids=["decomposed-latin", "multilingual", "emoji"],
)
def test_unicode_identity_requires_recorded_letters_or_numbers(
    tmp_path: Path,
    identity: str,
    expected_status: str,
) -> None:
    tier1 = ValidationResult(validator_name="SCHEMA")
    tier1.add_success("schema", "Schema passed")
    tier2 = ValidationResult(validator_name="Similarity Check")
    tier2.add_success("similarity_check", "Similarity scan completed")
    tier3 = _complete_tier3_result("UNICODE", score=0.9, runtime_seconds=1.0)
    payload = tier3.metadata["agent_eval"]
    agent = payload["agents"].pop("codex")
    agent["model"] = identity
    payload["agents"][identity] = agent
    payload["evaluator_version"] = identity
    results = [tier1, tier2, tier3]
    _bind_publication_target(results, tier3.metadata["publication_target"])

    benchmark = BenchmarkReporter(include_timestamp=False).render_all(results)
    json_payload = json.loads(JSONReporter(include_timestamp=False).render_all(results))
    html_data = _html_report_data(HTMLReporter(include_timestamp=False).render_all(results))
    markdown = MarkdownReporter(include_timestamp=False).render_all(results)
    benchmark_path = tmp_path / "BENCHMARK.md"
    benchmark_path.write_text(benchmark, encoding="utf-8")
    _files, offenders = benchmark_gate.find_offenders([benchmark_path])

    expected_label = "PASS" if expected_status == "pass" else "INCOMPLETE"
    assert f"Overall verdict: {expected_label}" in benchmark
    assert json_payload["publication_status"] == expected_status
    assert html_data["publication"]["status"] == expected_status
    assert f"**Publication status:** {'✅' if expected_status == 'pass' else '⚠️'} {expected_label}" in markdown
    assert offenders == []


@pytest.mark.parametrize("reverse_order", [False, True], ids=["canonical-first", "alias-first"])
def test_normalized_agent_key_collision_fails_publication_closed_across_reporters(
    tmp_path: Path,
    reverse_order: bool,
) -> None:
    tier1 = ValidationResult(validator_name="SCHEMA")
    tier1.add_success("schema", "Schema passed")
    tier2 = ValidationResult(validator_name="Similarity Check")
    tier2.add_success("similarity_check", "Similarity scan completed")
    tier3 = _complete_tier3_result("AGENT-COLLISION", score=0.9, runtime_seconds=1.0)
    canonical_agent = tier3.metadata["agent_eval"]["agents"].pop("codex")
    alias_agent = deepcopy(canonical_agent)
    alias_agent["dimensions"] = []
    agent_items = [("codex", canonical_agent), ("co\u200bdex", alias_agent)]
    if reverse_order:
        agent_items.reverse()
    tier3.metadata["agent_eval"]["agents"] = dict(agent_items)
    results = [tier1, tier2, tier3]

    benchmark = BenchmarkReporter(include_timestamp=False).render_all(results)
    json_payload = json.loads(JSONReporter(include_timestamp=False).render_all(results))
    html_output = HTMLReporter(include_timestamp=False).render_all(results)
    html_data = _html_report_data(html_output)
    html_tier3 = HTMLReporter(include_timestamp=False)._tier3_report_data(results)
    markdown = MarkdownReporter(include_timestamp=False).render_all(results)
    benchmark_path = tmp_path / "BENCHMARK.md"
    benchmark_path.write_text(benchmark, encoding="utf-8")
    _files, offenders = benchmark_gate.find_offenders([benchmark_path])

    assert "Overall verdict: INCOMPLETE" in benchmark
    assert "## Publication Recommendation" not in benchmark
    assert json_payload["publication_status"] == "incomplete"
    assert json_payload["tier3"]["agents"] == {}
    assert html_data["publication"]["status"] == "incomplete"
    assert html_tier3 is not None and html_tier3["agents"] == {}
    assert "**Publication status:** ⚠️ INCOMPLETE" in markdown
    assert offenders == []


def test_html_normalizes_mixed_control_identity_fields() -> None:
    control = "\u202e"
    tier3 = _complete_tier3_result("CONTROL", score=0.9, runtime_seconds=1.0)
    payload = tier3.metadata["agent_eval"]
    agent = payload["agents"].pop("codex")
    agent.update(
        {
            "model": f"gpt{control}FAIL",
            "display_name": f"Co{control}dex",
        }
    )
    payload["agents"][f"co{control}dex"] = agent
    payload["evaluator_version"] = f"1.0{control}PASS"
    payload["summary"]["environment"] = f"dock{control}er"

    html_payload = HTMLReporter(include_timestamp=False)._tier3_report_data([tier3])

    assert html_payload is not None
    assert control not in json.dumps(html_payload, ensure_ascii=False)
    assert html_payload["agents"]["codex"]["display_name"] == "Codex"
    assert html_payload["agents"]["codex"]["model"] == "gptFAIL"
    assert html_payload["evaluator_version"] == "1.0PASS"
    assert html_payload["summary"]["environment"] == "docker"


def test_html_normalizes_nested_template_identifiers_and_collisions() -> None:
    control = "\u202e"
    payload = {
        "agents": {
            f"co{control}dex": {
                "model": f"gpt{control}model",
                "with_skill": 0.9,
                "dimensions": [
                    {
                        "id": f"secu{control}rity",
                        "with_skill": 0.9,
                        "evaluators": [f"accu{control}racy"],
                    }
                ],
                "evaluator_cards": [
                    {"label": f"Accu{control}racy", "with_skill": 0.9},
                ],
            }
        },
        "evaluators": {f"accu{control}racy": {"with_skill": 0.9}},
        "metric_ids": [f"accu{control}racy"],
        "agents_run": [f"co{control}dex"],
        "metric_labels": {f"accu{control}racy": f"Accu{control}racy"},
        "dimension_hints": {f"secu{control}rity": f"Secu{control}rity hint"},
        "pass_at_k": {
            "with_skill": {"cases": {f"case{control}-1": {"passed": True}}},
        },
        "dataset": [{"id": f"case{control}-1"}],
        "trials": [
            {
                "entry_id": f"case{control}-1",
                "lift_scores": {f"accu{control}racy": 0.5},
            }
        ],
        "evaluator_cards": [{"label": f"Accu{control}racy", "with_skill": 0.9}],
    }

    sanitized = _sanitize_tier3_display_payload(payload)

    assert control not in json.dumps(sanitized, ensure_ascii=False)
    assert set(sanitized["agents"]) == {"codex"}
    assert set(sanitized["evaluators"]) == {"accuracy"}
    assert sanitized["metric_ids"] == ["accuracy"]
    assert sanitized["agents_run"] == ["codex"]
    assert set(sanitized["pass_at_k"]["with_skill"]["cases"]) == {"case-1"}
    assert sanitized["dataset"][0]["id"] == "case-1"
    assert set(sanitized["trials"][0]["lift_scores"]) == {"accuracy"}
    assert sanitized["evaluator_cards"][0]["label"] == "Accuracy"


@pytest.mark.parametrize("reverse_order", [False, True])
def test_html_drops_normalized_score_key_collisions_order_independently(reverse_order: bool) -> None:
    items = [("accuracy", 0.9), ("accu\u200bracy", 0.1)]
    if reverse_order:
        items.reverse()
    payload = _sanitize_tier3_display_payload({"trials": [{"entry_id": "case-1", "scores": dict(items)}]})

    assert payload["trials"][0]["scores"] == {}
    assert payload["_serialization_truncated"] is True


def test_html_uses_collision_free_index_ids_for_numeric_agent_and_dataset_names() -> None:
    tier3 = _complete_tier3_result("DOM-IDS", score=0.9, runtime_seconds=1.0)
    payload = tier3.metadata["agent_eval"]
    agent = payload["agents"].pop("codex")
    payload["agents"] = {"0": agent}
    payload["dataset"] = [{"id": "0"}, {"id": "tier3-agent-panel-0"}]

    html = HTMLReporter(include_timestamp=False).render_all([tier3])
    generated_ids = re.findall(r'\sid="(tier3-(?:agent|dataset)-[^"]+)"', html)

    assert len(generated_ids) == len(set(generated_ids))
    assert 'id="tier3-agent-panel-0"' in html
    assert 'id="tier3-agent-anchor-0"' in html
    assert 'data-tier3-agent-panel-id="tier3-agent-panel-0"' in html
    assert 'id="tier3-dataset-case-0"' in html
    assert 'id="tier3-dataset-case-1"' in html


@pytest.mark.parametrize("max_attempts", ["1", 1.5, 1.0], ids=["string", "fractional-float", "integral-float"])
def test_noninteger_attempt_count_cannot_certify_publication(
    tmp_path: Path,
    max_attempts: object,
) -> None:
    tier1 = ValidationResult(validator_name="SCHEMA")
    tier1.add_success("schema", "Schema passed")
    tier2 = ValidationResult(validator_name="Similarity Check")
    tier2.add_success("similarity_check", "Similarity scan completed")
    tier3 = _complete_tier3_result("ATTEMPTS", score=0.9, runtime_seconds=1.0)
    tier3.metadata["agent_eval"]["attempt_policy"]["max_attempts"] = max_attempts
    results = [tier1, tier2, tier3]

    benchmark = BenchmarkReporter(include_timestamp=False).render_all(results)
    json_payload = json.loads(JSONReporter(include_timestamp=False).render_all(results))
    html_data = _html_report_data(HTMLReporter(include_timestamp=False).render_all(results))
    markdown = MarkdownReporter(include_timestamp=False).render_all(results)
    benchmark_path = tmp_path / "BENCHMARK.md"
    benchmark_path.write_text(benchmark, encoding="utf-8")
    _files, offenders = benchmark_gate.find_offenders([benchmark_path])

    assert "Overall verdict: INCOMPLETE" in benchmark
    assert "- Attempts per task: not recorded (legacy or non-live result)" in benchmark
    assert json_payload["publication_status"] == "incomplete"
    assert html_data["publication"]["status"] == "incomplete"
    assert "**Publication status:** ⚠️ INCOMPLETE" in markdown
    assert offenders == []


@pytest.mark.parametrize(
    ("expected_attempts", "scored_attempts"),
    [(0, 0), (1, 0), (1, 2), (None, None)],
    ids=["zero", "unscored", "over-scored", "missing"],
)
def test_missing_or_inconsistent_scored_attempts_cannot_certify_publication(
    tmp_path: Path,
    expected_attempts: int | None,
    scored_attempts: int | None,
) -> None:
    tier1 = ValidationResult(validator_name="SCHEMA")
    tier1.add_success("schema", "Schema passed")
    tier2 = ValidationResult(validator_name="Similarity Check")
    tier2.add_success("similarity_check", "Similarity scan completed")
    tier3 = _complete_tier3_result("SCORED-ATTEMPTS", score=0.9, runtime_seconds=1.0)
    payload = tier3.metadata["agent_eval"]
    for container in (payload, payload["summary"], *payload["agents"].values()):
        if expected_attempts is None:
            container.pop("expected_attempts", None)
            container.pop("scored_attempts", None)
        else:
            container["expected_attempts"] = expected_attempts
            container["scored_attempts"] = scored_attempts
    results = [tier1, tier2, tier3]

    benchmark = BenchmarkReporter(include_timestamp=False).render_all(results)
    json_payload = json.loads(JSONReporter(include_timestamp=False).render_all(results))
    html_data = _html_report_data(HTMLReporter(include_timestamp=False).render_all(results))
    markdown = MarkdownReporter(include_timestamp=False).render_all(results)
    benchmark_path = tmp_path / "BENCHMARK.md"
    benchmark_path.write_text(benchmark, encoding="utf-8")
    _files, offenders = benchmark_gate.find_offenders([benchmark_path])

    assert "Overall verdict: INCOMPLETE" in benchmark
    assert json_payload["publication_status"] == "incomplete"
    assert html_data["publication"]["status"] == "incomplete"
    assert "**Publication status:** ⚠️ INCOMPLETE" in markdown
    assert offenders == []


def test_aggregate_attempt_counts_must_match_agent_evidence() -> None:
    tier1 = ValidationResult(validator_name="SCHEMA")
    tier1.add_success("schema", "Schema passed")
    tier2 = ValidationResult(validator_name="Similarity Check")
    tier2.add_success("similarity_check", "Similarity scan completed")
    tier3 = _complete_tier3_result("COUNT-MISMATCH", score=0.9, runtime_seconds=1.0)
    agent = next(iter(tier3.metadata["agent_eval"]["agents"].values()))
    agent["expected_attempts"] = 100
    agent["scored_attempts"] = 100

    payload = json.loads(JSONReporter(include_timestamp=False).render_all([tier1, tier2, tier3]))

    assert payload["publication_status"] == "incomplete"
    assert payload["publication"]["eligible"] is False


def test_benchmark_cannot_publish_tier3_evidence_from_another_skill(tmp_path: Path) -> None:
    tier1 = ValidationResult(validator_name="SCHEMA")
    tier1.add_success("schema", "Schema passed")
    tier1.metadata["quality_scores"] = {"skill_name": "skill-b"}
    tier2 = ValidationResult(validator_name="Similarity Check")
    tier2.add_success("similarity_check", "Similarity scan completed")
    tier3 = _complete_tier3_result("SKILL-A", score=0.9, runtime_seconds=1.0)
    tier3.metadata["agent_eval"]["skill_name"] = "skill-a"
    tier3.metadata["agent_eval"]["summary"]["skill_name"] = "skill-a"

    benchmark = BenchmarkReporter(skill_name="skill-b", include_timestamp=False).render_all([tier1, tier2, tier3])
    json_payload = json.loads(
        JSONReporter(include_timestamp=False, expected_skill_name="skill-b").render_all([tier1, tier2, tier3])
    )
    html_data = _html_report_data(
        HTMLReporter(include_timestamp=False, expected_skill_name="skill-b").render_all([tier1, tier2, tier3])
    )
    markdown = MarkdownReporter(include_timestamp=False, expected_skill_name="skill-b").render_all(
        [tier1, tier2, tier3]
    )
    benchmark_path = tmp_path / "BENCHMARK.md"
    benchmark_path.write_text(benchmark, encoding="utf-8")
    _files, offenders = benchmark_gate.find_offenders([benchmark_path])

    assert "# Skill Benchmark: skill-b" in benchmark
    assert "Overall verdict: INCOMPLETE" in benchmark
    assert "| Tier 3 | Live agent evaluation | **INCOMPLETE** |" in benchmark
    assert "Tier 3 evidence belongs to a different target skill." in benchmark
    assert "## Publication Recommendation" not in benchmark
    assert json_payload["publication_status"] == "incomplete"
    assert json_payload["publication"]["tier3"]["status"] == "incomplete"
    assert json_payload["publication"]["tier3"]["reason"] == ("Tier 3 evidence belongs to a different target skill.")
    assert html_data["publication"]["status"] == "incomplete"
    assert html_data["publication"]["tier3"]["status"] == "incomplete"
    assert html_data["publication"]["tier3"]["reason"] == ("Tier 3 evidence belongs to a different target skill.")
    assert "**Publication status:** ⚠️ INCOMPLETE" in markdown
    assert offenders == []


def test_anonymous_tier1_and_tier2_evidence_cannot_certify_named_tier3(tmp_path: Path) -> None:
    tier1 = ValidationResult(validator_name="SCHEMA")
    tier1.add_success("schema", "Schema passed")
    tier2 = ValidationResult(validator_name="Similarity Check")
    tier2.add_success("similarity_check", "Similarity scan completed")
    tier3 = _complete_tier3_result("SKILL-B", score=0.9, runtime_seconds=1.0)
    tier3.metadata["agent_eval"]["skill_name"] = "skill-b"
    tier3.metadata["agent_eval"]["summary"]["skill_name"] = "skill-b"
    _bind_publication_target([tier3], _publication_target("skill-b"))
    results = [tier1, tier2, tier3]

    benchmark = BenchmarkReporter(skill_name="skill-b", include_timestamp=False).render_all(results)
    json_payload = json.loads(JSONReporter(include_timestamp=False, expected_skill_name="skill-b").render_all(results))
    html_data = _html_report_data(
        HTMLReporter(include_timestamp=False, expected_skill_name="skill-b").render_all(results)
    )
    markdown = MarkdownReporter(include_timestamp=False, expected_skill_name="skill-b").render_all(results)
    benchmark_path = tmp_path / "BENCHMARK.md"
    benchmark_path.write_text(benchmark, encoding="utf-8")
    _files, offenders = benchmark_gate.find_offenders([benchmark_path])

    assert "Overall verdict: INCOMPLETE" in benchmark
    assert "## Publication Recommendation" not in benchmark
    assert json_payload["publication_status"] == "incomplete"
    assert html_data["publication"]["status"] == "incomplete"
    assert "**Publication status:** ⚠️ INCOMPLETE" in markdown
    assert offenders == []


def test_anonymous_policy_carrier_cannot_waive_target_tier2(tmp_path: Path) -> None:
    tier1 = ValidationResult(validator_name="Unrelated Policy Carrier")
    tier1.add_success("unrelated", "Unrelated check completed")
    tier1.metadata["benchmark_policy"] = {"tier2_required": False, "tier3_required": True}
    tier3 = _complete_tier3_result("SKILL-B-POLICY", score=0.9, runtime_seconds=1.0)
    tier3.metadata["agent_eval"]["skill_name"] = "skill-b"
    tier3.metadata["agent_eval"]["summary"]["skill_name"] = "skill-b"
    _bind_publication_target([tier3], _publication_target("skill-b"))
    results = [tier1, tier3]

    benchmark = BenchmarkReporter(skill_name="skill-b", include_timestamp=False).render_all(results)
    json_payload = json.loads(JSONReporter(include_timestamp=False, expected_skill_name="skill-b").render_all(results))
    benchmark_path = tmp_path / "BENCHMARK.md"
    benchmark_path.write_text(benchmark, encoding="utf-8")
    _files, offenders = benchmark_gate.find_offenders([benchmark_path])

    assert "Overall verdict: INCOMPLETE" in benchmark
    assert "## Publication Recommendation" not in benchmark
    assert "- Tier 2 evidence: required for publication" in benchmark
    assert json_payload["benchmark_policy"]["tier2_required"] is True
    assert json_payload["publication_status"] == "incomplete"
    assert offenders == []


@pytest.mark.parametrize(
    ("policy_location", "expected_required"),
    [("foreign_payload", True), ("target_tier1", False)],
)
def test_foreign_skipped_tier3_payload_cannot_waive_or_certify_target_policy(
    tmp_path: Path,
    policy_location: str,
    expected_required: bool,
) -> None:
    tier1 = ValidationResult(validator_name="SCHEMA")
    tier1.add_success("schema", "Schema passed")
    tier1.metadata["quality_scores"] = {"skill_name": "skill-b"}
    tier2 = ValidationResult(validator_name="Similarity Check")
    tier2.add_success("similarity_check", "Similarity scan completed")
    tier3 = advisory_skip_result("Live evaluation runtime unavailable", skill_name="skill-a")
    if policy_location == "foreign_payload":
        tier3.metadata["agent_eval"]["benchmark_policy"] = {"tier3_required": False}
    else:
        tier1.metadata["benchmark_policy"] = {"tier3_required": False}
        _bind_publication_target([tier1, tier2], _publication_target("skill-b"))
    results = [tier1, tier2, tier3]

    benchmark = BenchmarkReporter(skill_name="skill-b", include_timestamp=False).render_all(results)
    json_payload = json.loads(JSONReporter(include_timestamp=False, expected_skill_name="skill-b").render_all(results))
    html_data = _html_report_data(
        HTMLReporter(include_timestamp=False, expected_skill_name="skill-b").render_all(results)
    )
    markdown = MarkdownReporter(include_timestamp=False, expected_skill_name="skill-b").render_all(results)
    benchmark_path = tmp_path / "BENCHMARK.md"
    benchmark_path.write_text(benchmark, encoding="utf-8")
    _files, offenders = benchmark_gate.find_offenders([benchmark_path])

    assert "Overall verdict: INCOMPLETE" in benchmark
    assert "## Publication Recommendation" not in benchmark
    expected_policy_label = "required for publication" if expected_required else "optional by policy"
    assert f"- Tier 3 evidence: {expected_policy_label}" in benchmark
    assert "| Tier 3 | Live agent evaluation | **INCOMPLETE** |" in benchmark
    assert "Tier 3 evidence belongs to a different target skill." in benchmark
    assert json_payload["benchmark_policy"]["tier3_required"] is expected_required
    assert json_payload["publication_status"] == "incomplete"
    assert json_payload["publication"]["tier3"]["status"] == "incomplete"
    assert html_data["publication"]["status"] == "incomplete"
    assert html_data["publication"]["tier3"]["status"] == "incomplete"
    assert "**Publication status:** ⚠️ INCOMPLETE" in markdown
    assert offenders == []


@pytest.mark.parametrize("identity_mutation", ["conflicting", "missing"])
def test_default_reporters_reject_skipped_payload_with_invalid_target_identity(
    identity_mutation: str,
) -> None:
    tier1 = ValidationResult(validator_name="SCHEMA")
    tier1.add_success("schema", "Schema passed")
    tier3 = advisory_skip_result("Live evaluation runtime unavailable", skill_name="skill-a")
    payload = tier3.metadata["agent_eval"]
    payload["benchmark_policy"] = {"tier2_required": False, "tier3_required": False}
    if identity_mutation == "conflicting":
        payload["summary"]["skill_name"] = "skill-b"
    else:
        payload.pop("skill_name")
        payload["summary"].pop("skill_name")
    results = [tier1, tier3]

    benchmark = BenchmarkReporter(include_timestamp=False).render_all(results)
    json_payload = json.loads(JSONReporter(include_timestamp=False).render_all(results))
    html_data = _html_report_data(HTMLReporter(include_timestamp=False).render_all(results))
    markdown = MarkdownReporter(include_timestamp=False).render_all(results)

    assert "Overall verdict: INCOMPLETE" in benchmark
    assert "## Publication Recommendation" not in benchmark
    assert "| Tier 3 | Live agent evaluation | **INCOMPLETE** |" in benchmark
    assert "Tier 3 evidence lacks a consistent target skill identity." in benchmark
    assert json_payload["benchmark_policy"] == {"tier2_required": True, "tier3_required": True}
    assert json_payload["publication_status"] == "incomplete"
    assert json_payload["publication"]["tier3"]["status"] == "incomplete"
    assert html_data["publication"]["status"] == "incomplete"
    assert html_data["publication"]["tier3"]["status"] == "incomplete"
    assert "**Publication status:** ⚠️ INCOMPLETE" in markdown


def test_html_executed_tier3_payload_normalizes_malformed_display_fields() -> None:
    malformed_fields = [
        ("dimensions", [{"id": "accuracy", "score": "not-a-number"}]),
        ("recommendations", {}),
        ("metric_ids", 1),
        ("provenance", {"evaluator_paths": []}),
        ("dimension_hints", []),
        ("pass_at_k", []),
        ("report_truncation", {"omitted": []}),
        ("timing", [{"name": "collect", "seconds": 10**10000}]),
        ("best_agent", []),
        ("skill_name", "\ud800"),
        ("attempt_policy", {"max_attempts": 0}),
        ("dataset", [{"id": "case-001", "assertions": 1}]),
        ("trials", {}),
        ("evaluator_cards", [{"with_skill": "not-a-number"}]),
        ("overall_score", float("nan")),
    ]

    for field, value in malformed_fields:
        canonical = {
            "skill_name": "fixture-tier3",
            "verdict": "pass",
            "execution_status": "succeeded",
            "run_id": "fixture-malformed-display",
            "publication_target": _publication_target("fixture-tier3"),
            "overall_score": 0.9,
            "expected_attempts": 1,
            "scored_attempts": 1,
            "summary": {
                "skill_name": "fixture-tier3",
                "verdict": "pass",
                "execution_status": "succeeded",
                "run_id": "fixture-malformed-display",
                "publication_target": _publication_target("fixture-tier3"),
            },
            "agents": {
                "codex": {
                    "execution_status": "succeeded",
                    "with_skill": 0.9,
                    "expected_attempts": 1,
                    "scored_attempts": 1,
                    "evaluator_cards": [
                        {
                            "with_skill": 0.9,
                            "evidence": [
                                {
                                    "entry_id": ["case-001"],
                                    "occurrences": "not-a-count",
                                    "score": "not-a-number",
                                    "notes": {},
                                },
                                {
                                    "entry_id": {"case": "case-002"},
                                    "occurrences": 1,
                                    "score": 0.9,
                                    "notes": "valid score with malformed identifier",
                                },
                            ],
                        }
                    ],
                }
            },
            "dimensions": [],
            "evaluators": {},
            "dataset": [],
            "trials": [
                {
                    "agent": ["codex"],
                    "entry_id": ["case-001"],
                    "overall": 0.9,
                    "scores": {"accuracy": 0.9},
                    "baseline_scores": {},
                    "lift_scores": {},
                    "tokens": {},
                    "warnings": [],
                }
            ],
            field: value,
        }
        tier3 = ValidationResult(validator_name="AGENT_EVAL")
        tier3.add_success("agent_eval", "Tier 3 evaluation complete")
        tier3.metadata["agent_eval"] = canonical

        reporter = HTMLReporter(include_timestamp=False)
        html = reporter.render_all([tier3])
        payload = reporter._tier3_report_data([tier3])

        lossy = field in {"timing", "skill_name", "overall_score"}
        expected_class = "warn" if lossy else "pass"
        expected_verdict = "INCOMPLETE" if lossy else "PASS"
        assert re.search(
            rf'tier-card {expected_class}.*?tier-card-label">Tier 3</span>.*?tier-card-verdict">\s*'
            rf"{expected_verdict}\s*</span>",
            html,
            re.DOTALL,
        )
        assert payload is not None
        assert payload["verdict"] == "pass"
        assert payload["execution_status"] == "succeeded"
        assert isinstance(payload["agents"], dict)
        assert isinstance(payload["dimensions"], list)
        assert isinstance(payload["trials"], list)
        assert isinstance(payload["provenance"]["evaluator_paths"], dict)
        if lossy:
            assert payload["_serialization_truncated"] is True
            assert 'data-publication-status="incomplete"' in html
        else:
            assert "Embedded report details were bounded" not in html
        json.dumps(payload, allow_nan=False).encode("utf-8")
        if field == "attempt_policy":
            assert "max_attempts" not in payload["attempt_policy"]


def test_html_invalid_tier3_agent_scores_and_verdicts_fail_closed() -> None:
    malformed_payloads = [
        {
            "verdict": "pass",
            "execution_status": "succeeded",
            "overall_score": 0.9,
            "scored_attempts": 1,
            "agents": {
                "codex": {
                    "execution_status": "succeeded",
                    "with_skill": True,
                    "scored_attempts": 1,
                }
            },
        },
        {
            "verdict": "pass",
            "execution_status": "succeeded",
            "overall_score": 0.9,
            "scored_attempts": 1,
            "agents": {
                "codex": {
                    "execution_status": "succeeded",
                    "with_skill": 999,
                    "scored_attempts": 1,
                }
            },
        },
        {
            "verdict": "banana",
            "execution_status": "succeeded",
            "overall_score": 0.9,
            "scored_attempts": 1,
            "agents": {
                "codex": {
                    "execution_status": "succeeded",
                    "with_skill": 0.9,
                    "scored_attempts": 1,
                }
            },
        },
    ]

    for malformed_payload in malformed_payloads:
        tier3 = _validation_result_from_payload(malformed_payload)
        assert tier3 is not None

        benchmark = BenchmarkReporter(include_timestamp=False).render_all([tier3])
        reporter = HTMLReporter(include_timestamp=False)
        html = reporter.render_all([tier3])
        payload = reporter._tier3_report_data([tier3])

        assert "| Tier 3 | Live agent evaluation | **INCOMPLETE** |" in benchmark
        assert re.search(
            r'tier-card warn.*?tier-card-label">Tier 3</span>.*?tier-card-verdict">\s*INCOMPLETE\s*</span>',
            html,
            re.DOTALL,
        )
        assert payload is not None
        assert payload["verdict"] == "incomplete"
        assert payload["execution_status"] == "incomplete"


def test_html_partial_tier3_payload_removes_unsafe_only_harbor_links() -> None:
    partial_tier3 = ValidationResult(validator_name="AGENT_EVAL")
    partial_tier3.metadata["agent_eval"] = {
        "verdict": "pass",
        "harbor_viewer": {
            "job_url": "javascript:alert(1)",
            "analysis_url": "file:///tmp/private-report.html",
        },
        "summary": {
            "harbor_viewer": {
                "job_url": "javascript:alert(2)",
                "analysis_url": "file:///tmp/private-summary.html",
            }
        },
    }

    reporter = HTMLReporter(include_timestamp=False)
    html = reporter.render_all([partial_tier3])
    fallback = reporter._tier3_report_data([partial_tier3])

    assert "javascript:alert" not in html
    assert "file:///tmp" not in html
    assert fallback is not None
    assert "harbor_viewer" not in fallback
    assert "harbor_viewer" not in fallback["summary"]


def test_html_payloadless_tier3_failure_is_order_independent() -> None:
    bare_tier3 = ValidationResult(validator_name="AGENT_EVAL")
    failed_tier3 = ValidationResult(validator_name="AGENT_EVAL")
    failed_tier3.add_error("Harbor execution failed")

    reporter = HTMLReporter(include_timestamp=False)
    for results in ([bare_tier3, failed_tier3], [failed_tier3, bare_tier3]):
        html = reporter.render_all(results)
        fallback = reporter._tier3_report_data(results)

        assert re.search(
            r'tier-card fail.*?tier-card-label">Tier 3</span>.*?tier-card-verdict">\s*FAIL\s*</span>',
            html,
            re.DOTALL,
        )
        assert fallback is not None
        assert fallback["verdict"] == "fail"
        assert fallback["execution_status"] == "failed"


def test_html_result_failure_scrubs_untrusted_pass_shaped_scores() -> None:
    failed_tier3 = _validation_result_from_payload(
        {
            "verdict": "pass",
            "execution_status": "succeeded",
            "overall_score": 0.99,
            "agents": {
                "codex": {
                    "execution_status": "succeeded",
                    "with_skill": 0.99,
                }
            },
        }
    )
    assert failed_tier3 is not None
    failed_tier3.add_error("Engine forced failure")

    reporter = HTMLReporter(include_timestamp=False)
    html = reporter.render_all([failed_tier3])
    payload = reporter._tier3_report_data([failed_tier3])

    assert payload is not None
    assert payload["verdict"] == "fail"
    assert payload["execution_status"] == "failed"
    assert payload["overall_score"] is None
    assert payload["agents"]["codex"]["with_skill"] is None
    assert payload["execution_errors"] == ["Engine forced failure"]
    assert "0.99" not in html
    assert "Engine forced failure" in html


def test_html_tier1_dashboard_and_details_exclude_clean_skips() -> None:
    completed = ValidationResult(validator_name="SCHEMA")
    completed.add_success("schema", "Schema passed")
    skipped = ValidationResult(validator_name="Code Integrity & Hygiene")
    skipped.add_warning("Skipped: repository checkout unavailable")
    skipped.metadata["skipped"] = True

    html = HTMLReporter(include_timestamp=False).render_all([completed, skipped])
    tier1 = _html_tab(html, "tier1")

    assert re.search(r"Validators Run</h3>\s*<p[^>]*>1</p>", tier1)
    assert re.search(r"Passed</h3>\s*<p[^>]*>1</p>", tier1)
    assert 'data-status="skipped"' in tier1
    assert re.search(r'status-badge warn">\s*Skipped\s*</span>', tier1)
    assert "Skipped: repository checkout unavailable" in tier1
    assert "<strong>Skip reason:</strong> Skipped: repository checkout unavailable" in tier1
    assert "Skipped:</strong> Skipped:" not in tier1


def test_markdown_generic_clean_skip_matches_json_and_html_counts() -> None:
    completed = ValidationResult(validator_name="SCHEMA")
    completed.add_success("schema", "Schema passed")
    skipped = ValidationResult(validator_name="Similarity Check")
    skipped.add_warning("Skipped: embedding provider unavailable")
    skipped.metadata["skipped"] = True
    results = [completed, skipped]

    markdown = MarkdownReporter(include_timestamp=False).render_all(results)
    json_data = json.loads(JSONReporter(include_timestamp=False).render_all(results))
    html_data = _html_report_data(HTMLReporter(include_timestamp=False).render_all(results))

    assert json_data["total_skipped"] == html_data["summary"]["skipped_count"] == 1
    assert html_data["summary"]["executed_count"] == 1
    assert html_data["summary"]["passed_count"] == 1
    assert "| Validator Results | 2 |" in markdown
    assert "| Validators Run | 1 |" in markdown
    assert "| ✅ Passed | 1 |" in markdown
    assert "| ⏭️ Skipped | 1 |" in markdown
    assert "### ⏭️ SKIPPED Similarity Check" in markdown
    assert "- Skip reason: Skipped: embedding provider unavailable" in markdown
    assert "### ✅ Similarity Check" not in markdown


def test_advisory_skip_is_non_blocking_in_markdown() -> None:
    output = MarkdownReporter(include_timestamp=False).render_all(
        [advisory_skip_result("No public provider key", skill_name="demo")]
    )

    assert "**Status:** ✅ PASSED" in output
    assert "| Validator Results | 1 |" in output
    assert "| Validators Run | 0 |" in output
    assert "| ✅ Passed | 0 |" in output
    assert "| ⏭️ Skipped | 1 |" in output
    assert "| ⏭️ Advisory skips | 1 |" in output
    assert "### ⏭️ SKIPPED AGENT_EVAL" in output
    assert "- Skip reason: No public provider key" in output
    assert "❌ FAILED" not in output


def test_advisory_skip_is_non_blocking_in_html_but_incomplete_for_benchmark_publication() -> None:
    result = advisory_skip_result("No public provider key", skill_name="demo")
    html_data = _html_report_data(HTMLReporter(include_timestamp=False).render_all([result]))
    benchmark = BenchmarkReporter(include_timestamp=False).render_all([result])

    assert html_data["summary"]["status"] == "passed"
    assert html_data["summary"]["all_passed"] is True
    assert html_data["summary"]["passed_count"] == 0
    assert html_data["summary"]["failed_count"] == 0
    assert html_data["summary"]["advisory_skipped_count"] == 1
    assert html_data["results"][0]["status"] == "skipped"
    assert html_data["gating"]["would_block"] is False
    html = HTMLReporter(include_timestamp=False).render_all([result])
    assert re.search(r'tier-card-verdict">\s*SKIPPED\s*</span>', html)
    assert "Live evaluation skipped:</strong> No public provider key" in html
    assert not re.search(r'tier-card-verdict">\s*NEUTRAL\s*</span>', html)
    assert "Overall verdict: INCOMPLETE" in benchmark
    assert "Overall verdict: FAIL" not in benchmark
    assert "Tier 3 live evaluation: SKIPPED — No public provider key" in benchmark
    assert "benchmark is not publication-complete" in benchmark


def test_benchmark_redacts_advisory_skip_path() -> None:
    result = advisory_skip_result(
        "Runtime unavailable under /Users/alice/private/tier3",
        skill_name="demo",
    )

    benchmark = BenchmarkReporter(include_timestamp=False).render_all([result])

    assert "/Users/alice" not in benchmark
    assert "Runtime unavailable under tier3" in benchmark


def test_advisory_skip_does_not_appear_as_tier1_failure_in_html() -> None:
    schema = ValidationResult(validator_name="SCHEMA", validator_description="Schema validation")
    skipped = advisory_skip_result("No public provider key", skill_name="demo")

    output = HTMLReporter(include_timestamp=False).render_all([schema, skipped])
    html_data = _html_report_data(output)
    tier1 = _html_tab(output, "tier1")

    assert html_data["summary"]["passed_count"] == 1
    assert html_data["summary"]["failed_count"] == 0
    assert html_data["summary"]["advisory_skipped_count"] == 1
    assert re.search(r"Validators Run</h3>\s*<p[^>]*>1</p>", tier1)
    assert re.search(r"Failed</h3>\s*<p[^>]*>0</p>", tier1)
    assert "AGENT_EVAL" not in tier1


def test_real_agent_eval_failure_still_fails_required_validation() -> None:
    result = ValidationResult(validator_name="AGENT_EVAL")
    result.add_error("Harbor execution failed")

    payload = json.loads(JSONReporter(include_timestamp=False).render_all([result]))

    assert payload["overall_status"] == "failed"
    assert payload["overall_passed"] is False
    assert payload["results"][0]["status"] == "failed"

    html = HTMLReporter(include_timestamp=False).render_all([result])
    tier3 = _html_tab(html, "tier3")
    assert re.search(r'tier-card-verdict">\s*FAIL\s*</span>', html)
    assert "Harbor execution failed" in tier3


def test_explicit_advisory_gating_keeps_failed_result_non_blocking() -> None:
    result = ValidationResult(validator_name="AGENT_EVAL")
    result.add_error("Harbor execution failed")
    result.metadata["gating"] = {"tier": 3, "blocking": False}

    payload = json.loads(JSONReporter(include_timestamp=False).render_all([result]))
    html_data = _html_report_data(HTMLReporter(include_timestamp=False).render_all([result]))

    assert payload["overall_status"] == "passed"
    assert payload["overall_passed"] is True
    assert payload["gating"]["tiers"]["3"]["blocking"] is False
    assert html_data["summary"]["status"] == "passed"
    assert html_data["gating"]["would_block"] is False


def test_explicit_blocking_gating_promotes_tier3_failure() -> None:
    result = ValidationResult(validator_name="AGENT_EVAL")
    result.add_error("Harbor execution failed")
    result.metadata["gating"] = {"tier": 3, "blocking": True}

    payload = json.loads(JSONReporter(include_timestamp=False).render_all([result]))
    html_data = _html_report_data(HTMLReporter(include_timestamp=False).render_all([result]))

    assert payload["overall_status"] == "failed"
    assert payload["gating"]["tiers"]["3"]["blocking"] is True
    assert html_data["summary"]["status"] == "failed"
    assert html_data["gating"]["would_block"] is True


@pytest.mark.parametrize(
    ("verdict", "score", "expected_passed"),
    [("pass", 0.9, True), ("neutral", 0.45, False), ("fail", 0.1, False)],
)
def test_blocking_gate_uses_complete_tier3_verdict(
    verdict: str,
    score: float,
    expected_passed: bool,
) -> None:
    result = _complete_tier3_result("BLOCKING-VERDICT", score=score, runtime_seconds=1.0)
    result.metadata["agent_eval"]["verdict"] = verdict
    result.metadata["agent_eval"]["summary"]["verdict"] = verdict
    result.metadata["gating"] = {"tier": 3, "blocking": True}

    payload = json.loads(JSONReporter(include_timestamp=False).render_all([result]))
    html_data = _html_report_data(HTMLReporter(include_timestamp=False).render_all([result]))

    assert payload["overall_passed"] is expected_passed
    assert payload["overall_status"] == ("passed" if expected_passed else "failed")
    assert html_data["gating"]["would_block"] is (not expected_passed)


def test_blocking_tier3_process_pass_stays_separate_from_publication_provenance() -> None:
    result = _complete_tier3_result("PROCESS-VS-PUBLICATION", score=0.9, runtime_seconds=1.0)
    payload = result.metadata["agent_eval"]
    payload.pop("dataset_digest")
    payload.pop("dataset_digest_algorithm")
    payload.pop("evaluator_version")
    payload.pop("skill_name")
    payload["summary"].pop("environment")
    result.metadata["gating"] = {"tier": 3, "blocking": True}

    json_payload = json.loads(JSONReporter(include_timestamp=False).render_all([result]))
    html_data = _html_report_data(HTMLReporter(include_timestamp=False).render_all([result]))

    assert json_payload["overall_status"] == "passed"
    assert json_payload["overall_passed"] is True
    assert json_payload["publication_status"] == "incomplete"
    assert html_data["summary"]["status"] == "passed"
    assert html_data["publication"]["status"] == "incomplete"
    assert html_data["gating"]["would_block"] is False


def test_explicit_blocking_gating_overrides_legacy_advisory_skip_shape() -> None:
    result = advisory_skip_result("Harbor runtime unavailable", skill_name="demo")
    result.metadata["gating"] = {"tier": 3, "blocking": True}

    payload = json.loads(JSONReporter(include_timestamp=False).render_all([result]))
    html_data = _html_report_data(HTMLReporter(include_timestamp=False).render_all([result]))
    markdown = MarkdownReporter(include_timestamp=False).render_all([result])
    benchmark = BenchmarkReporter(include_timestamp=False).render_all([result])

    assert payload["overall_status"] == "failed"
    assert payload["overall_passed"] is False
    assert payload["total_advisory_skipped"] == 0
    assert payload["results"][0]["status"] == "failed"
    assert payload["gating"]["tiers"]["3"]["blocking"] is True
    assert html_data["summary"]["status"] == "failed"
    assert html_data["summary"]["advisory_skipped_count"] == 0
    assert html_data["results"][0]["status"] == "failed"
    assert html_data["gating"]["would_block"] is True
    assert "**Status:** ❌ FAILED" in markdown
    assert "### ❌ AGENT_EVAL" in markdown
    assert "SKIPPED AGENT_EVAL" not in markdown
    assert "Overall verdict: FAIL" in benchmark


@pytest.mark.parametrize("reporter_name", ["json", "html"])
def test_reporters_project_publication_target_to_validated_canonical_fields(
    reporter_name: str,
) -> None:
    result = ValidationResult(validator_name="SCHEMA")
    result.add_success("schema", "Schema passed")
    expected = _publication_target("demo")
    untrusted_target: dict[str, object] = dict(expected)
    untrusted_target["recursive"] = untrusted_target
    untrusted_target["non_finite"] = float("nan")
    untrusted_target["oversized_private_field"] = "do-not-leak-" * 100_000
    result.metadata["publication_target"] = untrusted_target

    if reporter_name == "json":
        report_data = json.loads(JSONReporter(include_timestamp=False).render_all([result]))
    else:
        report_data = _html_report_data(HTMLReporter(include_timestamp=False).render_all([result]))

    emitted = report_data["results"][0]["publication_target"]
    assert emitted == expected
    assert set(emitted) == {"skill_name", "skill_digest", "skill_digest_algorithm"}


@pytest.mark.parametrize("reporter_name", ["json", "html"])
@pytest.mark.parametrize("malformation", ["cyclic", "non_finite", "numeric_name", "oversized"])
def test_reporters_omit_malformed_or_unbounded_publication_target(
    reporter_name: str,
    malformation: str,
) -> None:
    result = ValidationResult(validator_name="SCHEMA")
    result.add_success("schema", "Schema passed")
    untrusted_target: dict[str, object] = _publication_target("demo")
    if malformation == "cyclic":
        untrusted_target["skill_name"] = untrusted_target
    elif malformation == "non_finite":
        untrusted_target["skill_digest"] = float("nan")
    elif malformation == "numeric_name":
        untrusted_target["skill_name"] = 123
    else:
        untrusted_target["skill_name"] = "do-not-leak-" * 10_000
    result.metadata["publication_target"] = untrusted_target

    if reporter_name == "json":
        rendered = JSONReporter(include_timestamp=False).render_all([result])
        report_data = json.loads(rendered)
    else:
        rendered = HTMLReporter(include_timestamp=False).render_all([result])
        report_data = _html_report_data(rendered)

    assert "publication_target" not in report_data["results"][0]
    assert "do-not-leak" not in rendered


def test_publication_metadata_rejects_huge_text_before_semantic_normalization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from skillevaluator.reporting import base as reporting_base

    original_safe_text = reporting_base._agent_eval_safe_text
    original_semantic_text = reporting_base.publication_semantic_text
    huge = "x" * 10_000

    def bounded_safe_text(value: object) -> str:
        if value is huge:
            raise AssertionError("oversized metadata reached UTF-8 normalization")
        return original_safe_text(value)

    def bounded_semantic_text(value: object, *, strip_marks: bool = False) -> str:
        if value is huge:
            raise AssertionError("oversized metadata reached semantic normalization")
        return original_semantic_text(value, strip_marks=strip_marks)

    monkeypatch.setattr(reporting_base, "_agent_eval_safe_text", bounded_safe_text)
    monkeypatch.setattr(reporting_base, "publication_semantic_text", bounded_semantic_text)

    canonical = {
        "skill_name": "demo",
        "skill_digest": "sha256:" + "a" * 64,
        "skill_digest_algorithm": "skill-evaluator-source-tree/2",
    }
    for field in ("skill_name", "skill_digest", "skill_digest_algorithm"):
        assert (
            reporting_base.publication_target_dict(
                {
                    **canonical,
                    field: huge,
                }
            )
            is None
        )
    assert reporting_base.publication_target_conflict_marker(huge) == "publication target identity conflict"

    def unexpected_consistent_text(*_args: object, **_kwargs: object) -> str:
        raise AssertionError("oversized run identity reached duplicate-field normalization")

    monkeypatch.setattr(reporting_base, "_agent_eval_consistent_text_field", unexpected_consistent_text)
    assert reporting_base.agent_eval_publication_run_id({"run_id": huge, "summary": {"run_id": huge}}) is None


@pytest.mark.parametrize("reporter_name", ["json", "html"])
def test_reporters_bound_and_flatten_untrusted_publication_conflict_metadata(
    reporter_name: str,
) -> None:
    result = ValidationResult(validator_name="SCHEMA")
    result.add_success("schema", "Schema passed")
    result.metadata["publication_target"] = _publication_target("demo")
    conflict: dict[str, object] = {
        "non_finite": float("nan"),
        "oversized_private_field": "do-not-leak-" * 100_000,
    }
    conflict["recursive"] = conflict
    result.metadata["publication_target_conflict"] = conflict

    if reporter_name == "json":
        report_data = json.loads(JSONReporter(include_timestamp=False).render_all([result]))
    else:
        report_data = _html_report_data(HTMLReporter(include_timestamp=False).render_all([result]))

    emitted = report_data["results"][0]
    assert "publication_target" not in emitted
    assert emitted["publication_target_conflict"] == "publication target identity conflict"


@pytest.mark.parametrize("reporter_name", ["json", "html"])
def test_reporters_normalize_bounded_plain_text_publication_conflict_marker(
    reporter_name: str,
) -> None:
    result = ValidationResult(validator_name="SCHEMA")
    result.add_success("schema", "Schema passed")
    result.metadata["publication_target_conflict"] = "  source\nchanged\x00 during validation  "

    if reporter_name == "json":
        report_data = json.loads(JSONReporter(include_timestamp=False).render_all([result]))
    else:
        report_data = _html_report_data(HTMLReporter(include_timestamp=False).render_all([result]))

    assert report_data["results"][0]["publication_target_conflict"] == "source changed during validation"


@pytest.mark.parametrize("reporter_name", ["json", "html"])
def test_reporters_project_nested_tier3_publication_identity_metadata(
    reporter_name: str,
) -> None:
    result = _complete_tier3_result("NESTED-IDENTITY", score=0.9, runtime_seconds=1.0)
    payload = result.metadata["agent_eval"]
    expected = dict(payload["publication_target"])
    for container in (payload, payload["summary"]):
        target = container["publication_target"]
        target["recursive"] = target
        target["non_finite"] = float("nan")
        target["oversized_private_field"] = "do-not-leak-" * 1_000
        conflict: dict[str, object] = {"oversized_private_field": "do-not-leak-" * 1_000}
        conflict["recursive"] = conflict
        container["publication_target_conflict"] = conflict

    if reporter_name == "json":
        rendered = JSONReporter(include_timestamp=False).render_all([result])
        report_data = json.loads(rendered)
        emitted = report_data["tier3"]
    else:
        rendered = HTMLReporter(include_timestamp=False).render_all([result])
        emitted = _html_tier3_payload(rendered)

    assert emitted["publication_target"] == expected
    assert emitted["summary"]["publication_target"] == expected
    assert emitted["publication_target_conflict"] == "publication target identity conflict"
    assert emitted["summary"]["publication_target_conflict"] == "publication target identity conflict"
    assert "do-not-leak" not in rendered


@pytest.mark.parametrize("conflict_location", ["outer", "payload", "summary"])
def test_conflict_marked_skipped_tier3_cannot_waive_publication_tiers(
    conflict_location: str,
) -> None:
    target = _publication_target("demo")
    tier1 = ValidationResult(validator_name="SCHEMA")
    tier1.add_success("schema", "Schema passed")
    tier3 = advisory_skip_result("Live evaluation runtime unavailable", skill_name="demo")
    tier3.metadata["agent_eval"]["benchmark_policy"] = {
        "tier2_required": False,
        "tier3_required": False,
    }
    tier3.metadata["agent_eval"]["summary"]["benchmark_policy"] = {
        "tier2_required": False,
        "tier3_required": False,
    }
    _bind_publication_target([tier1, tier3], target)
    if conflict_location == "outer":
        tier3.metadata["publication_target_conflict"] = "source changed during validation"
    else:
        payload_container = tier3.metadata["agent_eval"]
        if conflict_location == "summary":
            payload_container = payload_container["summary"]
        payload_container["publication_target_conflict"] = "source changed during validation"
    results = [tier1, tier3]

    json_payload = json.loads(JSONReporter(include_timestamp=False).render_all(results))
    html_data = _html_report_data(HTMLReporter(include_timestamp=False).render_all(results))
    benchmark = BenchmarkReporter(skill_name="demo", include_timestamp=False).render_all(results)

    assert json_payload["benchmark_policy"] == {"tier2_required": True, "tier3_required": True}
    assert json_payload["publication_status"] == "incomplete"
    assert json_payload["publication"]["tier3"]["status"] == "incomplete"
    assert html_data["publication"]["status"] == "incomplete"
    assert html_data["publication"]["tier3"]["status"] == "incomplete"
    assert "Overall verdict: INCOMPLETE" in benchmark
    assert "## Publication Recommendation" not in benchmark


def test_contradictory_outer_tier3_run_id_fails_closed_but_missing_outer_copy_is_compatible() -> None:
    tier3 = _complete_tier3_result("OUTER-RUN-ID", score=0.9, runtime_seconds=1.0)
    target = tier3.metadata["publication_target"]
    tier1 = ValidationResult(validator_name="SCHEMA")
    tier1.add_success("schema", "Schema passed")
    _bind_publication_target([tier1, tier3], target)
    payload = tier3.metadata["agent_eval"]
    payload["benchmark_policy"] = {"tier2_required": False, "tier3_required": True}
    payload["summary"]["benchmark_policy"] = {"tier2_required": False, "tier3_required": True}
    results = [tier1, tier3]

    compatible = json.loads(JSONReporter(include_timestamp=False).render_all(results))
    assert compatible["publication_status"] == "pass"

    tier3.metadata["run_id"] = "different-run"
    contradictory = json.loads(JSONReporter(include_timestamp=False).render_all(results))

    assert contradictory["publication_status"] == "incomplete"
    assert any("contradictory run identities" in reason for reason in contradictory["publication"]["reasons"])


@pytest.mark.parametrize("missing_location", ["payload", "summary"])
def test_completed_tier3_requires_run_id_in_payload_and_summary(missing_location: str) -> None:
    tier3 = _complete_tier3_result("REQUIRED-RUN-ID", score=0.9, runtime_seconds=1.0)
    target = tier3.metadata["publication_target"]
    tier1 = ValidationResult(validator_name="SCHEMA")
    tier1.add_success("schema", "Schema passed")
    _bind_publication_target([tier1, tier3], target)
    payload = tier3.metadata["agent_eval"]
    payload["benchmark_policy"] = {"tier2_required": False, "tier3_required": True}
    payload["summary"]["benchmark_policy"] = {"tier2_required": False, "tier3_required": True}
    container = payload if missing_location == "payload" else payload["summary"]
    container.pop("run_id")
    results = [tier1, tier3]

    json_payload = json.loads(JSONReporter(include_timestamp=False).render_all(results))
    html_data = _html_report_data(HTMLReporter(include_timestamp=False).render_all(results))
    benchmark = BenchmarkReporter(skill_name=target["skill_name"], include_timestamp=False).render_all(results)

    assert json_payload["publication_status"] == "incomplete"
    assert json_payload["publication"]["tier3"]["evidence_complete"] is False
    assert html_data["publication"]["status"] == "incomplete"
    assert "Overall verdict: INCOMPLETE" in benchmark


@pytest.mark.parametrize(
    ("identity_field", "invalid_identity"),
    [
        ("evaluator_version", "-"),
        ("evaluator_version", "N/A"),
        ("agent_name", "unknown"),
        ("model", "TBD"),
    ],
)
def test_completed_tier3_rejects_punctuation_and_placeholder_identities(
    identity_field: str,
    invalid_identity: str,
) -> None:
    tier3 = _complete_tier3_result("INVALID-IDENTITY", score=0.9, runtime_seconds=1.0)
    target = tier3.metadata["publication_target"]
    tier1 = ValidationResult(validator_name="SCHEMA")
    tier1.add_success("schema", "Schema passed")
    tier2 = ValidationResult(validator_name="Similarity Check")
    tier2.add_success("similarity_check", "Similarity scan completed")
    _bind_publication_target([tier1, tier2, tier3], target)
    payload = tier3.metadata["agent_eval"]
    if identity_field == "agent_name":
        agent = payload["agents"].pop("codex")
        payload["agents"][invalid_identity] = agent
    elif identity_field == "model":
        payload["agents"]["codex"]["model"] = invalid_identity
    else:
        payload[identity_field] = invalid_identity
    results = [tier1, tier2, tier3]

    json_payload = json.loads(JSONReporter(include_timestamp=False).render_all(results))
    benchmark = BenchmarkReporter(skill_name=target["skill_name"], include_timestamp=False).render_all(results)

    assert json_payload["publication_status"] == "incomplete"
    assert json_payload["publication"]["tier3"]["evidence_complete"] is False
    assert "Overall verdict: INCOMPLETE" in benchmark
