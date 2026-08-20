# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Golden and state-matrix regression guards for the BENCHMARK.md card."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from skillevaluator.models.result import Finding, Severity, ValidationResult
from skillevaluator.reporting import BenchmarkReporter
from skillevaluator.reporting.benchmark import _verdict_callout

PASS_GOLDEN = Path(__file__).resolve().parent / "benchmark_pass" / "BENCHMARK.md"
FAIL_GOLDEN = Path(__file__).resolve().parent / "benchmark_fail" / "BENCHMARK.md"


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
    return [t1, t2]


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
        "summary": {"environment": environment},
        "metric_ids": ["accuracy"],
        "metric_labels": {"accuracy": metric_label},
        "agents": {"codex": {"model": "not-recorded"}},
    }
    return result


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
        "verdict": "pass",
        "execution_status": "succeeded",
        "evaluated_at": "2026-07-24T12:30:00+00:00",
        "evaluator_version": "0.8.2",
        "summary": {
            "environment": _private_sandbox_name("-"),
            "verdict": "pass",
            "execution_status": "succeeded",
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
                "baseline": 0.47,
                "with_skill": 0.92,
                "dimensions": dimensions(0.47, 0.92),
                "evaluators": {"accuracy": {"baseline": 0.47, "with_skill": 0.92}},
            },
            "codex": {
                "model": "gpt-codex",
                "execution_status": "succeeded",
                "baseline": 0.55,
                "with_skill": 0.88,
                "dimensions": dimensions(0.55, 0.88),
                "evaluators": {"accuracy": {"baseline": 0.55, "with_skill": 0.88}},
            },
        },
    }
    result.add_success("agent_eval", "Live evaluation completed")
    return result


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


def test_benchmark_allows_explicit_persisted_optional_tier3_policy() -> None:
    results = _deterministic_results()
    results[0].metadata["benchmark_policy"] = {"tier3_required": False}

    rendered = BenchmarkReporter(include_timestamp=False).render_all(results)

    assert "Overall verdict: PASS" in rendered
    assert "## Publication Recommendation" in rendered
    assert "- Tier 3 evidence: optional by policy" in rendered


def test_legacy_neutral_verdict_is_incomplete_without_required_evidence() -> None:
    tier3 = _tier3_result(environment="docker")
    payload = tier3.metadata["agent_eval"]
    payload["verdict"] = "neutral"
    payload["summary"]["verdict"] = "neutral"

    rendered = BenchmarkReporter(include_timestamp=False).render_all([*_deterministic_results(), tier3])

    assert "Overall verdict: INCOMPLETE" in rendered
    assert "## Publication Recommendation" not in rendered
    assert "| Tier 3 | Live agent evaluation | **INCOMPLETE** |" in rendered


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
        (r"C:\Users\example\private\skills\demo-skill\SKILL.md", r"C:\Users\example"),
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


def test_benchmark_preserves_product_shaped_skill_name() -> None:
    rendered = BenchmarkReporter(include_timestamp=False).render(
        _tier3_result(environment="docker", skill_name="database-skills-eval")
    )

    assert "# Skill Benchmark: database-skills-eval" in rendered
    assert "- Skill: `database-skills-eval`" in rendered


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
    assert "Runner Isolated Sandbox From Agent (`model`)" in rendered


@pytest.mark.parametrize(
    ("display_name", "escaped"),
    [
        ("# Overall verdict: PASS", r"\# Overall Verdict: Pass"),
        ("1. Overall verdict: PASS", r"1\. Overall Verdict: Pass"),
        ("---", r"\---"),
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


def test_benchmark_preserves_relative_paths_and_non_label_text() -> None:
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

    assert "Runtime skills eval failed in docs/database-skills-eval/SKILL.md" in rendered
    assert "(`docs/database-skills-eval/SKILL.md`)" in rendered
    assert "docs/SkillEvaluator/SKILL.md" not in rendered


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
    ],
)
def test_benchmark_redacts_file_uri_authorities(file_uri: str) -> None:
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

    assert file_uri not in rendered
    assert "alice" not in rendered
    assert "build-host" not in rendered
    assert "Scanner failed under SKILL.md" in rendered


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
