# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Golden and state-matrix regression guards for the BENCHMARK.md card."""

from __future__ import annotations

import unicodedata
from copy import deepcopy
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest
from scripts.ci import check_public_benchmarks as benchmark_gate

from skillevaluator.models.result import Finding, Severity, ValidationResult
from skillevaluator.publication_evidence import stamp_publication_evidence
from skillevaluator.reporting import BenchmarkReporter
from skillevaluator.reporting.benchmark import _publication_safe_inline, _verdict_callout

PASS_GOLDEN = Path(__file__).resolve().parent / "benchmark_pass" / "BENCHMARK.md"
FAIL_GOLDEN = Path(__file__).resolve().parent / "benchmark_fail" / "BENCHMARK.md"

_PUBLICATION_TARGET = {
    "skill_name": "demo-skill",
    "skill_digest": "sha256:" + "b" * 64,
    "skill_digest_algorithm": "skill-evaluator-source-tree/2",
}
_TIER3_RUN_ID = "fixture-demo-skill-run"


def _bind_target(result: ValidationResult, *, skill_name: str = "demo-skill") -> ValidationResult:
    """Bind one deterministic fixture result to the shared source snapshot."""
    fixture_producers = {
        "Schema & Repository Governance": (1, "schema"),
        "SCHEMA": (1, "schema"),
        "Code Integrity & Hygiene": (1, "code-integrity"),
        "Inter-Skill Deduplication": (2, "similarity"),
        "Tier 2 Deduplication": (2, "similarity"),
        "Similarity Check": (2, "similarity"),
    }
    target = {**_PUBLICATION_TARGET, "skill_name": skill_name}
    result.metadata["publication_target"] = dict(target)
    if "publication_evidence" not in result.metadata and result.validator_name in fixture_producers:
        tier, check_id = fixture_producers[result.validator_name]
        stamp_publication_evidence([result], tier=tier, check_id=check_id)
    payload = result.metadata.get("agent_eval")
    if isinstance(payload, dict):
        payload["publication_target"] = dict(target)
        summary = payload.get("summary")
        if isinstance(summary, dict):
            summary["publication_target"] = dict(target)
    return result


def _private_sandbox_name(separator: str = " ") -> str:
    """Build the retired private label without embedding it in public source."""
    private_name = chr(65) + "stra"
    return separator.join((private_name, "sandbox"))


def _deterministic_results() -> list[ValidationResult]:
    t1 = ValidationResult(
        validator_name="Schema & Repository Governance",
        validator_description="Validate SKILL.md frontmatter and repository structure",
    )
    t1.add_success(check_name="author_format", message="Valid author format: Dev One <dev@nvidia.com>")
    t1.metadata["policy"] = {"profile": "internal"}
    t1.metadata["quality_scores"] = {"skill_name": "demo-skill"}
    _bind_target(t1)

    t2 = ValidationResult(
        validator_name="Inter-Skill Deduplication",
        validator_description="Detect duplicate skills across a catalog",
    )
    t2.add_finding(
        Finding(
            category="INTER_SKILL",
            severity=Severity.LOW,
            check_name="partial_overlap",
            message="Partial overlap with another skill",
            file_path="SKILL.md",
        )
    )
    _bind_target(t2)
    return [t1, t2]


def _skipped_tier2_result() -> ValidationResult:
    result = ValidationResult(
        validator_name="Tier 2 Deduplication",
        validator_description="Embedding-based duplicate detection",
    )
    result.add_warning("Skipped: embedding provider unavailable")
    result.metadata["skipped"] = True
    return _bind_target(result)


def _tier3_result(
    *,
    environment: str,
    metric_label: str = "Accuracy",
    skill_name: str = "demo-skill",
) -> ValidationResult:
    result = ValidationResult(
        validator_name="AGENT_EVAL",
        validator_description="Run live agent evaluation",
    )
    result.metadata["agent_eval"] = {
        "skill_name": skill_name,
        "publication_target": dict(_PUBLICATION_TARGET),
        "summary": {
            "environment": environment,
            "publication_target": dict(_PUBLICATION_TARGET),
        },
        "metric_ids": ["accuracy"],
        "metric_labels": {"accuracy": metric_label},
        "agents": {"codex": {"model": "not-recorded"}},
    }
    return _bind_target(result)


def _live_tier3_result() -> ValidationResult:
    result = ValidationResult(
        validator_name="AGENT_EVAL",
        validator_description="Run live agent evaluation",
    )

    def dimensions(
        baseline: float,
        with_skill: float,
    ) -> list[dict]:
        return [
            {
                "id": dimension,
                "baseline": baseline,
                "with_skill": with_skill,
                "lift": with_skill - baseline,
            }
            for dimension in ("security", "correctness", "discoverability", "effectiveness", "efficiency")
        ]

    result.metadata["agent_eval"] = {
        "skill_name": "demo-skill",
        "run_id": _TIER3_RUN_ID,
        "publication_target": dict(_PUBLICATION_TARGET),
        "verdict": "pass",
        "execution_status": "succeeded",
        "evaluated_at": "2026-07-24T12:30:00+00:00",
        "evaluator_version": "0.8.2",
        "expected_attempts": 16,
        "scored_attempts": 16,
        "summary": {
            "environment": _private_sandbox_name("-"),
            "verdict": "pass",
            "execution_status": "succeeded",
            "expected_attempts": 16,
            "scored_attempts": 16,
            "run_id": _TIER3_RUN_ID,
            "publication_target": dict(_PUBLICATION_TARGET),
        },
        "dataset_summary": {
            "total_tasks": 8,
            "positive_tasks": 6,
            "negative_tasks": 2,
            "unclassified_tasks": 0,
            "source": "dataset",
        },
        "dataset_digest": "sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef",
        "dataset_digest_algorithm": "skill-evaluator-dataset-snapshot/1",
        "attempt_policy": {"max_attempts": 3, "pass_threshold": 0.5},
        "agents": {
            "claude-code": {
                "model": "claude-sonnet",
                "execution_status": "succeeded",
                "expected_attempts": 8,
                "scored_attempts": 8,
                "baseline": 0.47,
                "with_skill": 0.92,
                "dimensions": dimensions(0.47, 0.92),
                "evaluators": {"accuracy": {"baseline": 0.47, "with_skill": 0.92}},
            },
            "codex": {
                "model": "gpt-codex",
                "execution_status": "succeeded",
                "expected_attempts": 8,
                "scored_attempts": 8,
                "baseline": 0.55,
                "with_skill": 0.88,
                "dimensions": dimensions(0.55, 0.88),
                "evaluators": {"accuracy": {"baseline": 0.55, "with_skill": 0.88}},
            },
        },
    }
    result.add_success("agent_eval", "Live evaluation completed")
    return _bind_target(result)


def _failing_results() -> list[ValidationResult]:
    results = deepcopy(_deterministic_results())
    results[0].add_finding(
        Finding(
            category="SCHEMA",
            severity=Severity.HIGH,
            check_name="missing_required_description",
            message="Required frontmatter field `description` is missing",
            file_path="SKILL.md",
            line_number=3,
        )
    )

    tier3 = deepcopy(_live_tier3_result())
    payload = tier3.metadata["agent_eval"]
    payload["verdict"] = "fail"
    payload["summary"]["verdict"] = "fail"

    failing_scores = {
        "claude-code": (0.72, 0.31),
        "codex": (0.65, 0.38),
    }
    for agent_name, (baseline, with_skill) in failing_scores.items():
        agent = payload["agents"][agent_name]
        agent["baseline"] = baseline
        agent["with_skill"] = with_skill
        for dimension in agent["dimensions"]:
            dimension["baseline"] = baseline
            dimension["with_skill"] = with_skill
            dimension["lift"] = with_skill - baseline
        agent["evaluators"]["accuracy"] = {
            "baseline": baseline,
            "with_skill": with_skill,
        }

    return [*results, tier3]


def test_benchmark_live_results_are_decision_first_and_unambiguous() -> None:
    rendered = BenchmarkReporter(include_timestamp=True).render_all([*_deterministic_results(), _live_tier3_result()])

    assert rendered.index("Overall verdict: PASS") < rendered.index("Evaluation Metadata")
    assert "# Skill Benchmark: demo-skill" in rendered
    assert "- Evaluation date: 2026-07-24" in rendered
    assert "- Evaluator version: `0.8.2`" in rendered
    assert "- Tasks: 8 evaluation tasks (6 positive, 2 negative)" in rendered


@pytest.mark.parametrize("environment", ["a", "x"])
def test_short_private_environment_label_does_not_corrupt_tier3_proof_fields(
    tmp_path: Path,
    environment: str,
) -> None:
    result = _live_tier3_result()
    payload = result.metadata["agent_eval"]
    payload["summary"]["environment"] = environment
    digest = payload["dataset_digest"]
    run_id = payload["run_id"]

    rendered = BenchmarkReporter(include_timestamp=True).render_all([*_deterministic_results(), result])
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(rendered, encoding="utf-8")
    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert "Overall verdict: PASS" in rendered
    assert f"- Dataset digest: `{digest}` (skill-evaluator-dataset-snapshot/1)" in rendered
    assert f"- Tier 3 run ID: `{run_id}`" in rendered
    assert not offenders


@pytest.mark.parametrize("environment", ["codex", "gpt-codex", "claude-code", "0.8.2"])
def test_private_environment_collision_does_not_rewrite_agent_provenance(
    tmp_path: Path,
    environment: str,
) -> None:
    result = _live_tier3_result()
    result.metadata["agent_eval"]["summary"]["environment"] = environment

    rendered = BenchmarkReporter(include_timestamp=True).render_all([*_deterministic_results(), result])
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(rendered, encoding="utf-8")
    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert "Overall verdict: PASS" in rendered
    assert "Claude Code (`claude-sonnet`)" in rendered
    assert "Codex (`gpt-codex`)" in rendered
    assert "- Evaluator version: `0.8.2`" in rendered
    assert "gpt-Isolated sandbox" not in rendered
    assert not offenders


def test_evaluation_date_is_normalized_to_utc_before_publication(tmp_path: Path) -> None:
    result = _live_tier3_result()
    current_utc = datetime.now(UTC)
    result.metadata["agent_eval"]["evaluated_at"] = current_utc.astimezone(timezone(timedelta(hours=14))).isoformat()

    rendered = BenchmarkReporter(include_timestamp=True).render_all([*_deterministic_results(), result])
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(rendered, encoding="utf-8")
    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert f"- Evaluation date: {current_utc.date().isoformat()}" in rendered
    assert not offenders
    assert (
        "- Dataset digest: `sha256:0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef` "
        "(skill-evaluator-dataset-snapshot/1)"
    ) in rendered
    assert "- Tier 3 evidence: required for publication" in rendered
    assert "Each task attempt ran in its own isolated sandbox." in rendered
    assert "| Measure | Claude Code (Baseline → Skill Uplift) | Codex (Baseline → Skill Uplift) |" in rendered
    assert "| Overall | 47% → 92% (+45 points) | 55% → 88% (+33 points) |" in rendered
    assert "| Dimension | Num |" not in rendered
    assert "The 50% attempt pass threshold is a separate per-task gate" in rendered
    assert "`goal_accuracy` (50%) + `behavior_check` (50%)" in rendered
    assert "PASS only when every configured dimension passes for at least one supported agent" in rendered


def test_benchmark_requires_tier3_by_default() -> None:
    rendered = BenchmarkReporter(include_timestamp=True).render_all(_deterministic_results())

    assert "Overall verdict: INCOMPLETE" in rendered
    assert "## Publication Recommendation" not in rendered
    assert "| Tier 3 | Live agent evaluation | **NOT RUN** |" in rendered
    assert "- Evaluation date: not recorded (legacy or non-live result)" in rendered
    assert "- Evaluator version: not recorded (legacy or non-live result)" in rendered
    assert "- Agents: not recorded (legacy or non-live result)" in rendered
    assert "- Tasks: not recorded (legacy or non-live result)" in rendered
    assert "- Attempts per task: not recorded (legacy or non-live result)" in rendered
    assert "- Environment: not recorded (legacy or non-live result)" in rendered


def test_tier1_only_cli_result_cannot_publish_without_tier3() -> None:
    rendered = BenchmarkReporter(include_timestamp=False).render_all([_deterministic_results()[0]])

    assert "Overall verdict: INCOMPLETE" in rendered
    assert "## Publication Recommendation" not in rendered
    assert "| Tier 2 | Semantic deduplication | **NOT RUN** |" in rendered
    assert "| Tier 3 | Live agent evaluation | **NOT RUN** |" in rendered


def test_benchmark_is_incomplete_when_required_tier2_is_missing() -> None:
    tier1 = _deterministic_results()[0]

    rendered = BenchmarkReporter(include_timestamp=False).render_all([tier1, _live_tier3_result()])

    assert "Overall verdict: INCOMPLETE" in rendered
    assert "Recommended for publication" not in rendered
    assert "## Publication Recommendation" not in rendered
    assert "- Tier 2 evidence: required for publication" in rendered
    assert "| Tier 2 | Semantic deduplication | **NOT RUN** |" in rendered


def test_benchmark_is_incomplete_when_tier1_is_missing() -> None:
    tier2 = _deterministic_results()[1]

    rendered = BenchmarkReporter(include_timestamp=False).render_all([tier2, _live_tier3_result()])

    assert "Overall verdict: INCOMPLETE" in rendered
    assert "Recommended for publication" not in rendered
    assert "## Publication Recommendation" not in rendered
    assert "| Tier 1 | Static validation | **NOT RUN** |" in rendered


def test_benchmark_allows_explicit_persisted_optional_tier2_policy() -> None:
    tier1 = _deterministic_results()[0]
    tier1.metadata["benchmark_policy"] = {"tier2_required": False}

    rendered = BenchmarkReporter(include_timestamp=False).render_all([tier1, _live_tier3_result()])

    assert "Overall verdict: PASS" in rendered
    assert "## Publication Recommendation" in rendered
    assert "- Tier 2 evidence: optional by policy" in rendered
    assert "| Tier 2 | Semantic deduplication | **NOT RUN** |" in rendered


def test_tier2_optional_policy_does_not_waive_missing_tier3() -> None:
    tier1 = _deterministic_results()[0]
    tier1.metadata["benchmark_policy"] = {"tier2_required": False}

    rendered = BenchmarkReporter(include_timestamp=False).render_all([tier1])

    assert "Overall verdict: INCOMPLETE" in rendered
    assert "## Publication Recommendation" not in rendered
    assert "- Tier 2 evidence: optional by policy" in rendered
    assert "- Tier 3 evidence: required for publication" in rendered
    assert "| Tier 3 | Live agent evaluation | **NOT RUN** |" in rendered


def test_tier3_optional_policy_does_not_waive_missing_tier2() -> None:
    tier1 = _deterministic_results()[0]
    tier1.metadata["benchmark_policy"] = {"tier3_required": False}

    rendered = BenchmarkReporter(include_timestamp=False).render_all([tier1])

    assert "Overall verdict: INCOMPLETE" in rendered
    assert "## Publication Recommendation" not in rendered
    assert "- Tier 2 evidence: required for publication" in rendered
    assert "- Tier 3 evidence: optional by policy" in rendered
    assert "| Tier 2 | Semantic deduplication | **NOT RUN** |" in rendered


def test_benchmark_allows_tier1_only_when_both_later_tiers_are_optional() -> None:
    tier1 = _deterministic_results()[0]
    tier1.metadata["benchmark_policy"] = {
        "tier2_required": False,
        "tier3_required": False,
    }

    rendered = BenchmarkReporter(include_timestamp=False).render_all([tier1])

    assert "Overall verdict: PASS" in rendered
    assert "## Publication Recommendation" in rendered
    assert "- Tier 2 evidence: optional by policy" in rendered
    assert "- Tier 3 evidence: optional by policy" in rendered
    assert "| Tier 2 | Semantic deduplication | **NOT RUN** |" in rendered
    assert "| Tier 3 | Live agent evaluation | **NOT RUN** |" in rendered


def test_benchmark_required_skipped_tier2_is_incomplete() -> None:
    tier1 = _deterministic_results()[0]

    rendered = BenchmarkReporter(include_timestamp=False).render_all(
        [tier1, _skipped_tier2_result(), _live_tier3_result()]
    )

    assert "Overall verdict: INCOMPLETE" in rendered
    assert "## Publication Recommendation" not in rendered
    assert "- Tier 2 evidence: required for publication" in rendered
    assert "| Tier 2 | Semantic deduplication | **INCOMPLETE** |" in rendered


def test_generic_optional_metadata_cannot_waive_required_skipped_tier2() -> None:
    tier1 = _deterministic_results()[0]
    tier2 = _skipped_tier2_result()
    tier2.metadata["optional"] = True

    rendered = BenchmarkReporter(include_timestamp=False).render_all([tier1, tier2, _live_tier3_result()])

    assert "Overall verdict: INCOMPLETE" in rendered
    assert "## Publication Recommendation" not in rendered
    assert "- Tier 2 evidence: required for publication" in rendered
    assert "| Tier 2 | Semantic deduplication | **INCOMPLETE** |" in rendered


def test_generic_optional_metadata_cannot_waive_skipped_tier1() -> None:
    tier1 = ValidationResult(
        validator_name="Schema & Repository Governance",
        validator_description="Validate SKILL.md frontmatter and repository structure",
    )
    tier1.add_warning("Skipped: repository checkout unavailable")
    tier1.metadata.update({"skipped": True, "optional": True})
    tier2 = _deterministic_results()[1]

    rendered = BenchmarkReporter(include_timestamp=False).render_all([tier1, tier2, _live_tier3_result()])

    assert "Overall verdict: INCOMPLETE" in rendered
    assert "## Publication Recommendation" not in rendered
    assert "| Tier 1 | Static validation | **INCOMPLETE** |" in rendered


def test_benchmark_discloses_partial_required_tier1_skip() -> None:
    tier1, tier2 = _deterministic_results()
    tier1.metadata["benchmark_policy"] = {"tier3_required": False}
    skipped_tier1 = ValidationResult(
        validator_name="Code Integrity & Hygiene",
        validator_description="Validate repository hygiene",
    )
    skipped_tier1.add_warning("Skipped: repository checkout unavailable")
    skipped_tier1.metadata.update({"skipped": True, "optional": True})
    _bind_target(skipped_tier1)

    rendered = BenchmarkReporter(include_timestamp=False).render_all([tier1, skipped_tier1, tier2])
    tier1_row = next(line for line in rendered.splitlines() if line.startswith("| Tier 1"))

    assert "Overall verdict: INCOMPLETE" in rendered
    assert "## Publication Recommendation" not in rendered
    assert "**INCOMPLETE**" in tier1_row
    assert "Skipped: repository checkout unavailable" in tier1_row


def test_benchmark_allows_explicit_optional_skipped_tier2() -> None:
    tier1 = _deterministic_results()[0]
    tier1.metadata["benchmark_policy"] = {"tier2_required": False}

    rendered = BenchmarkReporter(include_timestamp=False).render_all(
        [tier1, _skipped_tier2_result(), _live_tier3_result()]
    )

    assert "Overall verdict: PASS" in rendered
    assert "## Publication Recommendation" in rendered
    assert "- Tier 2 evidence: optional by policy" in rendered
    assert "| Tier 2 | Semantic deduplication | **SKIPPED (ADVISORY)** |" in rendered


@pytest.mark.parametrize(
    ("tier2_required", "skip_optional", "overall_status", "tier_status"),
    [
        (True, False, "INCOMPLETE", "INCOMPLETE"),
        (True, True, "INCOMPLETE", "INCOMPLETE"),
        (False, False, "PASS", "PASSED WITH OBSERVATIONS"),
    ],
)
def test_benchmark_discloses_partial_tier2_skips(
    tier2_required: bool,
    skip_optional: bool,
    overall_status: str,
    tier_status: str,
) -> None:
    tier1, completed_tier2 = _deterministic_results()
    tier1.metadata["benchmark_policy"] = {
        "tier2_required": tier2_required,
        "tier3_required": False,
    }
    skipped_tier2 = _skipped_tier2_result()
    if skip_optional:
        skipped_tier2.metadata["optional"] = True

    rendered = BenchmarkReporter(include_timestamp=False).render_all([tier1, completed_tier2, skipped_tier2])
    tier2_row = next(line for line in rendered.splitlines() if line.startswith("| Tier 2"))

    assert f"Overall verdict: {overall_status}" in rendered
    assert f"**{tier_status}**" in tier2_row
    assert "Skipped: embedding provider unavailable" in tier2_row
    assert ("## Publication Recommendation" in rendered) is (overall_status == "PASS")


def test_optional_but_incomplete_tier2_cannot_recommend_publication() -> None:
    tier1, tier2 = _deterministic_results()
    tier1.metadata["benchmark_policy"] = {"tier2_required": False}
    tier2.metadata["incomplete_scans"] = ["embedding-provider"]

    rendered = BenchmarkReporter(include_timestamp=False).render_all([tier1, tier2, _live_tier3_result()])

    assert "Overall verdict: INCOMPLETE" in rendered
    assert "## Publication Recommendation" not in rendered
    assert "- Tier 2 evidence: optional by policy" in rendered
    assert (
        "| Tier 2 | Semantic deduplication | **INCOMPLETE** | Missing trustworthy evidence from embedding-provider |"
    ) in rendered


@pytest.mark.parametrize("tier2_required", [True, False])
def test_nonblocking_cli_gate_cannot_waive_failed_tier2_publication_evidence(
    tier2_required: bool,
) -> None:
    tier1 = _deterministic_results()[0]
    tier1.metadata["benchmark_policy"] = {
        "tier2_required": tier2_required,
        "tier3_required": False,
    }
    tier2 = ValidationResult(
        validator_name="Similarity Check",
        validator_description="Detect duplicate content via embedding similarity",
    )
    tier2.add_error("Similarity scan failed")
    tier2.metadata["gating"] = {"tier": 2, "blocking": False}

    rendered = BenchmarkReporter(include_timestamp=False).render_all([tier1, tier2])

    assert "Overall verdict: FAIL" in rendered
    assert "## Publication Recommendation" not in rendered
    assert "| Tier 2 | Semantic deduplication | **FAILED** |" in rendered


def test_failed_tier2_cannot_masquerade_as_a_clean_optional_skip() -> None:
    tier1 = _deterministic_results()[0]
    tier1.metadata["benchmark_policy"] = {
        "tier2_required": False,
        "tier3_required": False,
    }
    tier2 = ValidationResult(validator_name="Similarity Check")
    tier2.add_error("Similarity provider returned a malformed response")
    tier2.metadata["skipped"] = True

    rendered = BenchmarkReporter(include_timestamp=False).render_all([tier1, tier2])

    assert "Overall verdict: FAIL" in rendered
    assert "## Publication Recommendation" not in rendered
    assert "| Tier 2 | Semantic deduplication | **FAILED** |" in rendered


def test_benchmark_invalid_tier2_policy_fails_closed() -> None:
    tier1 = _deterministic_results()[0]
    tier1.metadata["benchmark_policy"] = {"tier2_required": "false"}

    rendered = BenchmarkReporter(include_timestamp=False).render_all([tier1, _live_tier3_result()])

    assert "Overall verdict: INCOMPLETE" in rendered
    assert "## Publication Recommendation" not in rendered
    assert "- Tier 2 evidence: required for publication" in rendered
    assert "| Tier 2 | Semantic deduplication | **NOT RUN** |" in rendered


def test_invalid_higher_precedence_tier2_policy_uses_lower_valid_boolean() -> None:
    tier1 = _deterministic_results()[0]
    tier1.metadata["benchmark_policy"] = {"tier2_required": False}
    tier3 = _live_tier3_result()
    tier3.metadata["agent_eval"]["benchmark_policy"] = {"tier2_required": "false"}

    rendered = BenchmarkReporter(include_timestamp=False).render_all([tier1, tier3])

    assert "Overall verdict: PASS" in rendered
    assert "## Publication Recommendation" in rendered
    assert "- Tier 2 evidence: optional by policy" in rendered


def test_benchmark_policy_resolves_each_key_across_source_precedence() -> None:
    tier1 = _deterministic_results()[0]
    tier1.metadata["benchmark_policy"] = {"tier2_required": False}
    tier3 = _live_tier3_result()
    tier3.metadata["agent_eval"]["benchmark_policy"] = {"tier3_required": True}

    rendered = BenchmarkReporter(include_timestamp=False).render_all([tier1, tier3])

    assert "Overall verdict: PASS" in rendered
    assert "## Publication Recommendation" in rendered
    assert "- Tier 2 evidence: optional by policy" in rendered
    assert "- Tier 3 evidence: required for publication" in rendered
    assert "| Tier 2 | Semantic deduplication | **NOT RUN** |" in rendered


@pytest.mark.parametrize(
    ("agent_eval_required", "result_required", "expected_status", "requirement"),
    [
        (False, True, "PASS", "optional by policy"),
        (True, False, "INCOMPLETE", "required for publication"),
    ],
)
def test_agent_eval_tier2_policy_precedes_conflicting_result_metadata(
    agent_eval_required: bool,
    result_required: bool,
    expected_status: str,
    requirement: str,
) -> None:
    tier1 = _deterministic_results()[0]
    tier1.metadata["benchmark_policy"] = {"tier2_required": result_required}
    tier3 = _live_tier3_result()
    tier3.metadata["agent_eval"]["benchmark_policy"] = {"tier2_required": agent_eval_required}

    rendered = BenchmarkReporter(include_timestamp=False).render_all([tier1, tier3])

    assert f"Overall verdict: {expected_status}" in rendered
    assert f"- Tier 2 evidence: {requirement}" in rendered
    assert ("## Publication Recommendation" in rendered) is (expected_status == "PASS")


def test_benchmark_classifies_similarity_check_as_tier2() -> None:
    tier1 = _deterministic_results()[0]
    similarity = ValidationResult(
        validator_name="Similarity Check",
        validator_description="Detect duplicate content via embedding similarity",
    )
    similarity.add_success("similarity_scan", "No duplicate content found")
    _bind_target(similarity)

    rendered = BenchmarkReporter(include_timestamp=False).render_all([tier1, similarity, _live_tier3_result()])

    assert "Overall verdict: PASS" in rendered
    assert "| Tier 1 | Static validation | **PASSED** | 1 validator(s); 0 finding(s) |" in rendered
    assert "| Tier 2 | Semantic deduplication | **PASSED** | 1 validator(s); 0 finding(s) |" in rendered


def test_bare_tier2_result_cannot_certify_publication(tmp_path: Path) -> None:
    tier1 = _deterministic_results()[0]
    bare_tier2 = ValidationResult(
        validator_name="Similarity Check",
        validator_description="Detect duplicate content via embedding similarity",
    )
    _bind_target(bare_tier2)

    rendered = BenchmarkReporter(include_timestamp=False).render_all([tier1, bare_tier2, _live_tier3_result()])
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(rendered, encoding="utf-8")
    files, offenders = benchmark_gate.find_offenders([benchmark])

    assert "Overall verdict: INCOMPLETE" in rendered
    assert "Recommended for publication" not in rendered
    assert (
        "| Tier 2 | Semantic deduplication | **INCOMPLETE** | "
        "Missing trustworthy execution evidence from Similarity Check |"
    ) in rendered
    assert files == [benchmark.resolve()]
    assert offenders == []


@pytest.mark.parametrize("warning", [None, "Schema validation was requested"])
def test_bare_tier1_result_cannot_certify_publication(
    tmp_path: Path,
    warning: str | None,
) -> None:
    bare_tier1 = ValidationResult(
        validator_name="SCHEMA",
        validator_description="Validate SKILL.md frontmatter and repository structure",
    )
    if warning is not None:
        bare_tier1.add_warning(warning)
    bare_tier1.metadata["benchmark_policy"] = {
        "tier2_required": False,
        "tier3_required": False,
    }
    _bind_target(bare_tier1, skill_name="skill")

    rendered = BenchmarkReporter(include_timestamp=False).render_all([bare_tier1])
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(rendered, encoding="utf-8")
    files, offenders = benchmark_gate.find_offenders([benchmark])

    assert "Overall verdict: INCOMPLETE" in rendered
    assert "Recommended for publication" not in rendered
    assert (
        "| Tier 1 | Static validation | **INCOMPLETE** | Missing trustworthy execution evidence from SCHEMA |"
    ) in rendered
    assert files == [benchmark.resolve()]
    assert offenders == []


@pytest.mark.parametrize("reverse", [False, True])
def test_conflicting_result_level_benchmark_policy_fails_closed_independent_of_order(
    reverse: bool,
) -> None:
    first = _deterministic_results()[0]
    first.metadata["benchmark_policy"] = {
        "tier2_required": False,
        "tier3_required": False,
    }
    second = ValidationResult(validator_name="Code Integrity & Hygiene")
    second.add_success("repository_hygiene", "Repository hygiene checks completed")
    second.metadata["benchmark_policy"] = {
        "tier2_required": True,
        "tier3_required": False,
    }
    tier1_results = [first, second]
    if reverse:
        tier1_results.reverse()

    rendered = BenchmarkReporter(include_timestamp=False).render_all(tier1_results)

    assert "Overall verdict: INCOMPLETE" in rendered
    assert "- Tier 2 evidence: required for publication" in rendered
    assert "## Publication Recommendation" not in rendered


def test_tier2_summary_checks_are_trustworthy_execution_evidence() -> None:
    tier1 = _deterministic_results()[0]
    tier2 = ValidationResult(
        validator_name="Similarity Check",
        validator_description="Detect duplicate content via embedding similarity",
    )
    tier2.summary.checks_performed = 1
    _bind_target(tier2)

    rendered = BenchmarkReporter(include_timestamp=False).render_all([tier1, tier2, _live_tier3_result()])

    assert "Overall verdict: PASS" in rendered
    assert "## Publication Recommendation" in rendered
    assert "| Tier 2 | Semantic deduplication | **PASSED** |" in rendered


def test_tier3_dedup_observation_cannot_supply_tier2_publication_evidence() -> None:
    tier1 = _deterministic_results()[0]
    tier3 = _live_tier3_result()
    tier3.add_finding(
        Finding(
            category="CONTENT_DEDUP",
            severity=Severity.LOW,
            check_name="repeated_context",
            message="Live evaluation observed repeated context",
            file_path=None,
        )
    )

    rendered = BenchmarkReporter(include_timestamp=False).render_all([tier1, tier3])

    assert "Overall verdict: INCOMPLETE" in rendered
    assert "## Publication Recommendation" not in rendered
    assert "| Tier 2 | Semantic deduplication | **NOT RUN** |" in rendered
    assert "| Tier 3 | Live agent evaluation | **PASS** |" in rendered


def test_benchmark_allows_explicit_persisted_optional_tier3_policy() -> None:
    results = _deterministic_results()
    results[0].metadata["benchmark_policy"] = {"tier3_required": False}

    rendered = BenchmarkReporter(include_timestamp=False).render_all(results)

    assert "Overall verdict: PASS" in rendered
    assert "## Publication Recommendation" in rendered
    assert "- Tier 3 evidence: optional by policy" in rendered
    assert "| Tier 3 | Live agent evaluation | **NOT RUN** | No result was recorded |" in rendered


@pytest.mark.parametrize(
    ("tier3_required", "overall_status", "tier3_status"),
    [
        (True, "INCOMPLETE", "INCOMPLETE"),
        (False, "PASS", "SKIPPED (ADVISORY)"),
    ],
)
def test_generic_clean_tier3_skip_respects_publication_policy(
    tmp_path: Path,
    tier3_required: bool,
    overall_status: str,
    tier3_status: str,
) -> None:
    tier1, tier2 = _deterministic_results()
    tier1.metadata["benchmark_policy"] = {"tier3_required": tier3_required}
    tier3 = ValidationResult(
        validator_name="AGENT_EVAL",
        validator_description="Run live agent evaluation",
    )
    tier3.add_warning("Skipped: live evaluation runtime unavailable")
    tier3.metadata["skipped"] = True

    rendered = BenchmarkReporter(include_timestamp=False).render_all([tier1, tier2, tier3])
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(rendered, encoding="utf-8")
    files, offenders = benchmark_gate.find_offenders([benchmark])

    assert f"Overall verdict: {overall_status}" in rendered
    assert ("## Publication Recommendation" in rendered) is (overall_status == "PASS")
    assert f"| Tier 3 | Live agent evaluation | **{tier3_status}** |" in rendered
    assert files == [benchmark.resolve()]
    assert offenders == []


def test_legacy_neutral_verdict_is_incomplete_without_required_evidence() -> None:
    tier3 = _tier3_result(environment="docker")
    payload = tier3.metadata["agent_eval"]
    payload["verdict"] = "neutral"
    payload["summary"]["verdict"] = "neutral"

    rendered = BenchmarkReporter(include_timestamp=False).render_all([*_deterministic_results(), tier3])

    assert "Overall verdict: INCOMPLETE" in rendered
    assert "## Publication Recommendation" not in rendered
    assert "| Tier 3 | Live agent evaluation | **INCOMPLETE** |" in rendered


@pytest.mark.parametrize(
    "partial_payload",
    [
        {},
        {"verdict": "pass"},
        {"verdict": "neutral"},
        {"verdict": "incomplete", "execution_status": "incomplete"},
        {"verdict": "pass", "execution_status": "succeeded", "agents": {}},
    ],
)
@pytest.mark.parametrize("complete_first", [False, True])
def test_benchmark_selects_complete_tier3_payload_independent_of_result_order(
    partial_payload: dict,
    complete_first: bool,
) -> None:
    partial_tier3 = ValidationResult(validator_name="AGENT_EVAL")
    partial_tier3.metadata["agent_eval"] = partial_payload
    _bind_target(partial_tier3)
    complete_tier3 = _live_tier3_result()
    tier3_results = [complete_tier3, partial_tier3] if complete_first else [partial_tier3, complete_tier3]

    rendered = BenchmarkReporter(include_timestamp=False).render_all([*_deterministic_results(), *tier3_results])

    assert "Overall verdict: PASS" in rendered
    assert "## Publication Recommendation" in rendered
    assert "| Tier 3 | Live agent evaluation | **PASS** |" in rendered


@pytest.mark.parametrize(
    ("malformed_field", "expected_status"),
    [
        ("agent-score", "INCOMPLETE"),
        ("dimension-score", "INCOMPLETE"),
        ("dimensions-shape", "PASS"),
        ("task-count", "INCOMPLETE"),
        ("attempt-count", "INCOMPLETE"),
        ("trials-shape", "INCOMPLETE"),
    ],
)
def test_benchmark_handles_malformed_tier3_shapes_without_crashing(
    malformed_field: str,
    expected_status: str,
) -> None:
    tier3 = _live_tier3_result()
    payload = tier3.metadata["agent_eval"]
    if malformed_field == "agent-score":
        payload["agents"]["codex"]["with_skill"] = 10**10000
    elif malformed_field == "dimension-score":
        payload["agents"]["codex"]["dimensions"][0]["with_skill"] = 10**10000
    elif malformed_field == "dimensions-shape":
        payload["agents"]["codex"]["dimensions"] = 1
    elif malformed_field == "task-count":
        payload["dataset_summary"]["total_tasks"] = float("inf")
    elif malformed_field == "attempt-count":
        payload["attempt_policy"]["max_attempts"] = float("nan")
    else:
        payload["dataset_summary"] = None
        payload["trials"] = 1

    rendered = BenchmarkReporter(include_timestamp=False).render_all([*_deterministic_results(), tier3])

    assert f"Overall verdict: {expected_status}" in rendered
    assert ("## Publication Recommendation" in rendered) is (expected_status == "PASS")
    assert f"| Tier 3 | Live agent evaluation | **{expected_status}** |" in rendered


@pytest.mark.parametrize(
    "malformed_field",
    ["skill-name", "attempt-count", "agent-model", "dimension-id", "evaluator-name"],
)
def test_benchmark_handles_huge_tier3_metadata_without_crashing(malformed_field: str) -> None:
    tier3 = _live_tier3_result()
    payload = tier3.metadata["agent_eval"]
    huge = 10**10000
    if malformed_field == "skill-name":
        payload["skill_name"] = huge
    elif malformed_field == "attempt-count":
        payload["attempt_policy"]["max_attempts"] = huge
    elif malformed_field == "agent-model":
        payload["agents"]["codex"]["model"] = huge
    elif malformed_field == "dimension-id":
        payload["agents"]["codex"]["dimensions"][0]["id"] = huge
    else:
        payload["agents"]["codex"]["evaluators"] = {huge: {"with_skill": 0.9}}

    rendered = BenchmarkReporter(include_timestamp=False).render_all([*_deterministic_results(), tier3])

    assert "Overall verdict:" in rendered
    assert "## Freshness" in rendered


@pytest.mark.parametrize(
    "max_attempts",
    [float("nan"), float("inf"), True, [], 10**10000],
    ids=["nan", "infinity", "boolean", "list", "huge-integer"],
)
def test_benchmark_normalizes_invalid_attempt_metadata_and_scans_cleanly(
    tmp_path: Path,
    max_attempts: object,
) -> None:
    tier3 = _live_tier3_result()
    tier3.metadata["agent_eval"]["attempt_policy"]["max_attempts"] = max_attempts

    rendered = BenchmarkReporter(include_timestamp=False).render_all([*_deterministic_results(), tier3])
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(rendered, encoding="utf-8")
    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert "Overall verdict: INCOMPLETE" in rendered
    assert "- Attempts per task: not recorded (legacy or non-live result)" in rendered
    assert offenders == []


def test_malformed_tier3_result_is_incomplete_and_passes_publication_scanner(tmp_path: Path) -> None:
    malformed_tier3 = ValidationResult(
        validator_name="AGENT_EVAL",
        validator_description="Run live agent evaluation",
    )

    rendered = BenchmarkReporter(include_timestamp=False).render_all([*_deterministic_results(), malformed_tier3])
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(rendered, encoding="utf-8")
    files, offenders = benchmark_gate.find_offenders([benchmark])

    assert "Overall verdict: INCOMPLETE" in rendered
    assert "| Tier 3 | Live agent evaluation | **INCOMPLETE** | Required Tier 3 evidence is missing |" in rendered
    assert files == [benchmark.resolve()]
    assert offenders == []


def test_verdict_callout_falls_back_for_future_statuses() -> None:
    assert _verdict_callout("WARN") == "> **Overall verdict: WARN**"


def test_benchmark_pass_card_matches_golden() -> None:
    rendered = BenchmarkReporter(include_timestamp=True).render_all([*_deterministic_results(), _live_tier3_result()])
    assert rendered == PASS_GOLDEN.read_text(encoding="utf-8")
    assert "| Tier 1 | Static validation | **PASSED** |" in rendered
    assert "| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** |" in rendered
    assert "| Tier 3 | Live agent evaluation | **PASS** |" in rendered


def test_benchmark_fail_card_matches_golden() -> None:
    rendered = BenchmarkReporter(include_timestamp=True).render_all(_failing_results())
    assert rendered == FAIL_GOLDEN.read_text(encoding="utf-8")
    assert "| Tier 1 | Static validation | **FAILED** |" in rendered
    assert "| Tier 2 | Semantic deduplication | **PASSED WITH OBSERVATIONS** |" in rendered
    assert "| Tier 3 | Live agent evaluation | **FAIL** |" in rendered


def test_benchmark_no_baseline_never_fabricates_uplift() -> None:
    tier3 = deepcopy(_live_tier3_result())
    payload = tier3.metadata["agent_eval"]
    payload["verdict"] = "neutral"
    payload["summary"]["verdict"] = "neutral"
    for agent in payload["agents"].values():
        agent["baseline"] = None
        for dimension in agent["dimensions"]:
            dimension["baseline"] = None
            dimension["lift"] = None

    rendered = BenchmarkReporter(include_timestamp=False).render_all([*_deterministic_results(), tier3])

    assert "Overall verdict: NEUTRAL" in rendered
    assert "92% — baseline not run; uplift unavailable" in rendered
    assert "88% — baseline not run; uplift unavailable" in rendered
    assert "Publication Recommendation" not in rendered


def test_benchmark_zero_and_negative_lift_use_percentage_points() -> None:
    from skillevaluator.reporting.benchmark import _score_transition_values

    assert _score_transition_values(1.0, 1.0) == "100% → 100% (±0 points)"
    assert _score_transition_values(0.7, 0.62) == "70% → 62% (-8 points)"
    assert _score_transition_values(None, 0.92) == "92% — baseline not run; uplift unavailable"


def test_benchmark_local_mode_does_not_claim_sandboxing() -> None:
    tier3 = deepcopy(_live_tier3_result())
    tier3.metadata["agent_eval"]["summary"]["environment"] = "local"

    rendered = BenchmarkReporter(include_timestamp=False).render(tier3)

    assert "trusted local host; local mode is not sandboxed" in rendered
    assert "own isolated sandbox" not in rendered


@pytest.mark.parametrize("environment", [_private_sandbox_name("-"), _private_sandbox_name(), "ASTRA"])
def test_benchmark_uses_public_sandbox_label(environment: str) -> None:
    rendered = BenchmarkReporter(include_timestamp=False).render(_tier3_result(environment=environment))

    assert "- Environment: `Isolated sandbox`" in rendered
    assert "astra" not in rendered.lower()


def test_benchmark_preserves_non_sandbox_skill_name() -> None:
    rendered = BenchmarkReporter(include_timestamp=False).render(
        _tier3_result(environment="docker", skill_name="astra-db")
    )

    assert "- Skill: `astra-db`" in rendered
    assert "Isolated sandbox-db" not in rendered


def test_benchmark_sanitizes_internal_sandbox_name_in_finding_text() -> None:
    result = _tier3_result(environment="docker")
    result.add_finding(
        Finding(
            category="AGENT_EVAL",
            severity=Severity.LOW,
            check_name="environment",
            message=f"Observed in {_private_sandbox_name()} during evaluation",
            file_path=None,
        )
    )

    rendered = BenchmarkReporter(include_timestamp=False).render(result)

    assert _private_sandbox_name() not in rendered
    assert "Observed in isolated sandbox during evaluation" in rendered


@pytest.mark.parametrize(
    "retired_name",
    ["NVSkills-Eval", "NVSkills Eval", "NVSkillsEval", "nvskillseval", "legacy-skills-eval"],
)
def test_benchmark_rebrands_retired_product_name_from_payload(retired_name: str) -> None:
    rendered = BenchmarkReporter(include_timestamp=False).render(
        _tier3_result(environment="docker", metric_label=retired_name)
    )

    assert retired_name not in rendered
    assert "`accuracy` (SkillEvaluator)" in rendered


@pytest.mark.parametrize("separator", ["\u2010", "\u2011", "\u2012", "\u2013", "\u2043", "\u2212"])
@pytest.mark.parametrize("identity", ["Skills{separator}Eval", "astra{separator}sandbox"])
def test_benchmark_redacts_unicode_dash_private_identities(
    separator: str,
    identity: str,
) -> None:
    rendered_identity = identity.format(separator=separator)

    sanitized = _publication_safe_inline(f"observed {rendered_identity}")

    assert rendered_identity not in sanitized
    expected = "SkillEvaluator" if identity.startswith("Skills") else "isolated sandbox"
    assert expected in sanitized


def test_retired_identity_rewrite_handles_dense_untrusted_text_linearly() -> None:
    dense_value = "SkillsEval " * 50_000

    sanitized = _publication_safe_inline(dense_value)

    assert sanitized == ("SkillEvaluator " * 50_000).rstrip()


def test_benchmark_omits_internal_validation_profile() -> None:
    result = ValidationResult(
        validator_name="Schema & Repository Governance",
        validator_description="Validate SKILL.md",
    )
    result.metadata["policy"] = {"profile": "internal"}
    result.metadata["quality_scores"] = {"skill_name": "demo-skill"}

    rendered = BenchmarkReporter(include_timestamp=False).render(result)

    assert "profile" not in rendered.lower()


@pytest.mark.parametrize(
    ("file_path", "private_prefix"),
    [
        ("/Users/example/private/skills/demo-skill/SKILL.md", "/Users/example"),
        ("\u2215Users\u2215example\u2215private\u2215skills\u2215demo-skill\u2215SKILL.md", "\u2215Users\u2215example"),
        ("\u2044home\u2044example\u2044private\u2044skills\u2044demo-skill\u2044SKILL.md", "\u2044home\u2044example"),
        (r"C:\Users\example\private\skills\demo-skill\SKILL.md", r"C:\Users\example"),
        (
            "C:\u29f5Users\u29f5example\u29f5private\u29f5skills\u29f5demo-skill\u29f5SKILL.md",
            "C:\u29f5Users\u29f5example",
        ),
        (r"\Users\example\private\skills\demo-skill\SKILL.md", r"\Users\example"),
    ],
)
def test_benchmark_hides_absolute_finding_paths(file_path: str, private_prefix: str) -> None:
    result = ValidationResult(
        validator_name="Schema & Repository Governance",
        validator_description="Validate SKILL.md",
    )
    result.add_finding(
        Finding(
            category="SCHEMA",
            severity=Severity.LOW,
            check_name="example",
            message="Example finding",
            file_path=file_path,
            line_number=7,
        )
    )

    rendered = BenchmarkReporter(include_timestamp=False).render(result)

    assert private_prefix not in rendered
    assert "(`SKILL.md:7`)" in rendered


@pytest.mark.parametrize(
    "skill_name",
    [
        "skills-eval",
        "skillseval",
        "my-skills-eval-agent",
        "database-skills-eval",
        "Skills\u0301Eval",
    ],
)
def test_pass_card_normalizes_retired_identity_in_skill_name(
    tmp_path: Path,
    skill_name: str,
) -> None:
    canonical_skill_name = unicodedata.normalize("NFC", skill_name)
    tier3 = _live_tier3_result()
    tier3.metadata["agent_eval"]["skill_name"] = canonical_skill_name
    results = [*_deterministic_results(), tier3]
    for result in results:
        _bind_target(result, skill_name=canonical_skill_name)
    results[0].metadata["quality_scores"]["skill_name"] = canonical_skill_name

    rendered = BenchmarkReporter(include_timestamp=False).render_all(results)
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(rendered, encoding="utf-8")
    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert "Overall verdict: PASS" in rendered
    assert "# Skill Benchmark: SkillEvaluator" in rendered
    assert skill_name not in rendered
    assert offenders == []


def test_benchmark_sanitizes_invalid_legacy_skill_label() -> None:
    rendered = BenchmarkReporter(include_timestamp=False).render(
        _tier3_result(environment="docker", skill_name="LegacySkillsEval")
    )

    assert "LegacySkillsEval" not in rendered
    assert "- Skill: `SkillEvaluator`" in rendered


def test_benchmark_sanitizes_agent_and_model_labels() -> None:
    private_environment = "secret-cluster"
    result = _tier3_result(environment=private_environment)
    result.metadata["agent_eval"]["agents"] = {
        "runner": {
            "display_name": f"runner {private_environment} from /Users/alice/private/agent",
            "model": r"C:\models\private\model",
        }
    }

    rendered = BenchmarkReporter(include_timestamp=False).render(result)

    assert private_environment not in rendered
    assert "/Users/alice" not in rendered
    assert r"C:\models\private" not in rendered
    assert "Runner (`model`)" in rendered


def test_private_label_redaction_precedes_retired_identity_normalization(tmp_path: Path) -> None:
    private_environment = "my-skills-eval-prod"
    tier3 = _live_tier3_result()
    payload = tier3.metadata["agent_eval"]
    payload["summary"]["environment"] = private_environment
    payload["agents"]["codex"]["display_name"] = f"runner from {private_environment}"

    rendered = BenchmarkReporter(include_timestamp=False).render_all([*_deterministic_results(), tier3])
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(rendered, encoding="utf-8")
    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert "Overall verdict: PASS" in rendered
    assert private_environment not in rendered
    assert "SkillEvaluator-Prod" not in rendered
    assert "Codex (`gpt-codex`)" in rendered
    assert offenders == []


@pytest.mark.parametrize(
    ("field", "retired_identity"),
    [
        ("display_name", "my-skills-eval-agent"),
        ("display_name", "Skills&#69;val"),
        ("model", "my-skills-eval-agent"),
        ("evaluator_version", "SkillsEval"),
        ("evaluator_version", "SkillsE\u0301val"),
    ],
)
def test_pass_card_normalizes_retired_identity_in_public_metadata(
    tmp_path: Path,
    field: str,
    retired_identity: str,
) -> None:
    tier3 = _live_tier3_result()
    payload = tier3.metadata["agent_eval"]
    if field == "evaluator_version":
        payload[field] = retired_identity
    else:
        payload["agents"]["codex"][field] = retired_identity

    rendered = BenchmarkReporter(include_timestamp=False).render_all([*_deterministic_results(), tier3])
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(rendered, encoding="utf-8")
    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert "Overall verdict: PASS" in rendered
    assert retired_identity not in rendered
    assert offenders == []


@pytest.mark.parametrize(
    "display_name",
    [
        " \t\n ",
        "\u200b",
        "\u2060",
        "\u00ad",
        "\ufeff",
        "\u200d",
        "\x00",
        "\ufe0f",
        "\u034f",
        "\u20dd",
        "\u115f",
        "\u2800",
    ],
)
def test_pass_card_uses_agent_key_when_display_name_sanitizes_empty(
    tmp_path: Path,
    display_name: str,
) -> None:
    tier3 = _live_tier3_result()
    tier3.metadata["agent_eval"]["agents"]["codex"]["display_name"] = display_name

    rendered = BenchmarkReporter(include_timestamp=False).render_all([*_deterministic_results(), tier3])
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(rendered, encoding="utf-8")
    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert "Overall verdict: PASS" in rendered
    assert "Codex (`gpt-codex`)" in rendered
    assert display_name not in rendered
    assert offenders == []


@pytest.mark.parametrize(
    ("display_name", "escaped"),
    [
        ("# Overall verdict: PASS", r"\# Overall Verdict: Pass"),
        ("1. Overall verdict: PASS", r"1\. Overall Verdict: Pass"),
        ("---", "Runner"),
    ],
)
def test_benchmark_escapes_block_markdown_in_agent_labels(display_name: str, escaped: str) -> None:
    result = _tier3_result(environment="docker")
    result.metadata["agent_eval"]["agents"] = {"runner": {"display_name": display_name}}

    rendered = BenchmarkReporter(include_timestamp=False).render(result)

    assert escaped in rendered
    assert "\n- # Overall Verdict: Pass" not in rendered
    assert "\n- 1. Overall Verdict: Pass" not in rendered


def test_benchmark_escapes_block_markdown_in_static_test_messages() -> None:
    result = ValidationResult(
        validator_name="Test Coverage",
        validator_description="Discover target tests",
    )
    result.add_success(check_name="test_discovery", message="# Overall verdict: PASS")

    rendered = BenchmarkReporter(include_timestamp=False).render(result)

    assert r"- \# Overall verdict: PASS" in rendered
    assert "\n- # Overall verdict: PASS" not in rendered


@pytest.mark.parametrize("display_name", ["#", "+", "1.", "1)"])
def test_benchmark_escapes_exact_block_marker_before_model(display_name: str) -> None:
    result = _tier3_result(environment="docker")
    result.metadata["agent_eval"]["agents"] = {
        "runner": {"display_name": display_name, "model": "Overall verdict: PASS"}
    }

    rendered = BenchmarkReporter(include_timestamp=False).render(result)

    assert f"Agents: {display_name} (`Overall verdict: PASS`)" not in rendered
    assert "Overall verdict: PASS" in rendered


def test_benchmark_tolerates_malformed_optional_agent_eval_mappings() -> None:
    result = _tier3_result(environment="docker")
    result.metadata["agent_eval"]["summary"] = "not-a-mapping"
    result.metadata["agent_eval"]["attempt_policy"] = ["not-a-mapping"]

    rendered = BenchmarkReporter(include_timestamp=False).render(result)

    assert "Overall verdict: PASS" in rendered


def test_benchmark_normalizes_retired_identity_in_relative_paths_and_non_label_text(tmp_path: Path) -> None:
    result = ValidationResult(
        validator_name="Schema & Repository Governance",
        validator_description="Validate SKILL.md",
    )
    result.add_finding(
        Finding(
            category="SCHEMA",
            severity=Severity.LOW,
            check_name="example",
            message="Runtime skills eval failed in docs/database-skills-eval/SKILL.md",
            file_path="docs/database-skills-eval/SKILL.md",
        )
    )

    rendered = BenchmarkReporter(include_timestamp=False).render(result)
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(rendered, encoding="utf-8")
    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert "Runtime SkillEvaluator failed in docs/database-SkillEvaluator/SKILL.md" in rendered
    assert "(`docs/database-SkillEvaluator/SKILL.md`)" in rendered
    assert offenders == []


def test_benchmark_redacts_absolute_paths_from_dynamic_text() -> None:
    result = ValidationResult(
        validator_name="Schema & Repository Governance",
        validator_description="Validate SKILL.md",
    )
    result.add_finding(
        Finding(
            category="SCHEMA",
            severity=Severity.LOW,
            check_name="example",
            message="Scanner failed under /Users/alice/private/repo/SKILL.md",
            file_path="SKILL.md",
        )
    )

    rendered = BenchmarkReporter(include_timestamp=False).render(result)

    clean_result = ValidationResult(
        validator_name=r"Scanner from C:\Users\alice\private\validator",
        validator_description="Validate SKILL.md",
    )
    clean_result.add_success(check_name="example", message="Validation completed")
    clean_rendered = BenchmarkReporter(include_timestamp=False).render(clean_result)

    assert "/Users/alice" not in rendered
    assert "Scanner failed under SKILL.md" in rendered
    assert r"C:\Users\alice" not in clean_rendered
    assert "validator: Validation completed" in clean_rendered


def test_benchmark_sanitizes_file_uris_markdown_and_private_values() -> None:
    private_environment = "secret-cluster"
    result = _tier3_result(environment=private_environment, metric_label=f"x-{private_environment}")
    tier1 = ValidationResult(
        validator_name="Schema & Repository Governance",
        validator_description="Validate SKILL.md",
    )
    tier1.add_finding(
        Finding(
            category="[fake](https://attacker.example)",
            severity=Severity.LOW,
            check_name="<img src=x onerror=alert(1)>",
            message="![PASS](https://attacker.example/pass.svg) at file:///Users/alice/private/SKILL.md",
            file_path="SKILL.md",
        )
    )

    rendered = BenchmarkReporter(include_timestamp=False).render_all([tier1, result])

    assert private_environment not in rendered
    assert "/Users/alice" not in rendered
    assert "file:///" not in rendered
    assert "https://attacker.example" not in rendered
    assert r"\[fake\](https&#58;//attacker.example)" in rendered
    assert "&lt;img src=x onerror=alert(1)&gt;" in rendered
    assert r"!\[PASS\](https&#58;//attacker.example/pass.svg) at SKILL.md" in rendered
    assert "Isolated sandbox" in rendered


@pytest.mark.parametrize(
    "metric_label",
    ["secret-cluster-4", "x-secret-cluster", "secret-cluster_count"],
)
def test_benchmark_redacts_embedded_private_environment_labels(metric_label: str) -> None:
    result = _tier3_result(environment="secret-cluster", metric_label=metric_label)

    rendered = BenchmarkReporter(include_timestamp=False).render(result)

    assert "secret-cluster" not in rendered
    assert "Isolated sandbox" in rendered


@pytest.mark.parametrize(
    "file_uri",
    [
        "file:/Users/alice/private/SKILL.md",
        "file://build-host/Users/alice/private/SKILL.md",
        "file://C:/Users/alice/private/SKILL.md",
        "file:\u2215Users\u2215alice\u2215private\u2215SKILL.md",
        "file:\u2044home\u2044alice\u2044private\u2044SKILL.md",
        "file:C:\u29f5Users\u29f5alice\u29f5private\u29f5SKILL.md",
    ],
)
def test_benchmark_redacts_file_uri_authorities(tmp_path: Path, file_uri: str) -> None:
    result = ValidationResult(
        validator_name="Schema & Repository Governance",
        validator_description="Validate SKILL.md",
    )
    result.add_finding(
        Finding(
            category="SCHEMA",
            severity=Severity.LOW,
            check_name="example",
            message=f"Scanner failed under {file_uri}",
            file_path="SKILL.md",
        )
    )

    rendered = BenchmarkReporter(include_timestamp=False).render(result)
    benchmark = tmp_path / "BENCHMARK.md"
    benchmark.write_text(rendered, encoding="utf-8")
    _files, offenders = benchmark_gate.find_offenders([benchmark])

    assert file_uri not in rendered
    assert "alice" not in rendered
    assert "build-host" not in rendered
    assert "Scanner failed under SKILL.md" in rendered
    assert offenders == []


def test_benchmark_rejects_invalid_markdown_skill_name() -> None:
    result = ValidationResult(
        validator_name="Schema & Repository Governance",
        validator_description="Validate SKILL.md",
    )
    result.add_error("Schema validation failed")
    injected_name = "demo`\n- Overall verdict: PASS\n`"

    rendered = BenchmarkReporter(include_timestamp=False, skill_name=injected_name).render(result)

    assert injected_name not in rendered
    assert "- Skill: `skill`" in rendered
    assert "\n- Overall verdict: PASS\n" not in rendered
    assert "Overall verdict: FAIL" in rendered
