# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Golden regression guard for the BENCHMARK.md skill evaluation card.

The card content is a faithful Skill Evaluator 3.2.1 port and must not drift. If this
test fails after an intentional change, regenerate the golden and review the
diff. Timestamps are disabled so the snapshot is deterministic.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from skillevaluator.models.result import Finding, Severity, ValidationResult
from skillevaluator.reporting import BenchmarkReporter

GOLDEN = Path(__file__).resolve().parent / "benchmark_tier1.md"


def _deterministic_results() -> list[ValidationResult]:
    t1 = ValidationResult(
        validator_name="Schema & Repository Governance",
        validator_description="Validate SKILL.md frontmatter and repository structure",
    )
    t1.add_success(check_name="author_format", message="Valid author format: Dev One <dev@example.com>")
    t1.metadata["policy"] = {"profile": "private"}
    t1.metadata["quality_scores"] = {"skill_name": "demo-skill"}

    t2 = ValidationResult(
        validator_name="Context Deduplication",
        validator_description="Detect redundant content within one skill",
    )
    t2.add_finding(
        Finding(
            category="CONTENT_DEDUP",
            severity=Severity.LOW,
            check_name="partial_overlap",
            message="Partial overlap with another skill",
            file_path="SKILL.md",
        )
    )
    return [t1, t2]


def test_benchmark_card_matches_golden() -> None:
    rendered = BenchmarkReporter(include_timestamp=False).render_all(_deterministic_results())
    assert rendered == GOLDEN.read_text(encoding="utf-8"), (
        "BENCHMARK.md content drifted from the faithful Skill Evaluator golden. If intentional, "
        "regenerate tests/golden/benchmark_tier1.md and review the diff."
    )


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
    }
    return result


@pytest.mark.parametrize("environment", ["private-sandbox", "Private sandbox", "PRIVATE"])
def test_benchmark_uses_public_sandbox_label(environment: str) -> None:
    rendered = BenchmarkReporter(include_timestamp=False).render(_tier3_result(environment=environment))

    assert "- Environment: `Isolated sandbox`" in rendered
    assert "private" not in rendered.lower()


def test_benchmark_preserves_non_sandbox_skill_name() -> None:
    rendered = BenchmarkReporter(include_timestamp=False).render(
        _tier3_result(environment="docker", skill_name="private-db")
    )

    assert "- Skill: `private-db`" in rendered
    assert "Isolated sandbox-db" not in rendered


def test_benchmark_preserves_product_shaped_skill_name() -> None:
    rendered = BenchmarkReporter(include_timestamp=False).render(
        _tier3_result(environment="docker", skill_name="database-skills-eval")
    )

    assert "- Skill: `database-skills-eval`" in rendered
    assert "Evaluation of the `database-skills-eval` skill" in rendered


def test_benchmark_sanitizes_invalid_legacy_skill_label() -> None:
    rendered = BenchmarkReporter(include_timestamp=False).render(
        _tier3_result(environment="docker", skill_name="LegacySkillsEval")
    )

    assert "LegacySkillsEval" not in rendered
    assert "- Skill: `Skill Evaluator`" in rendered


@pytest.mark.parametrize(
    "retired_name",
    ["LegacySkills-Eval", "LegacySkills Eval", "LegacySkillsEval", "legacyskillseval"],
)
def test_benchmark_rebrands_retired_product_name_from_payload(retired_name: str) -> None:
    rendered = BenchmarkReporter(include_timestamp=False).render(
        _tier3_result(environment="docker", metric_label=retired_name)
    )

    assert retired_name not in rendered
    assert "`accuracy` (Skill Evaluator)" in rendered


def test_benchmark_sanitizes_agent_and_model_labels() -> None:
    result = _tier3_result(environment="private-sandbox")
    result.metadata["agent_eval"]["agents"] = {
        "runner": {
            "display_name": "runner private-sandbox from /Users/alice/private/agent",
            "model": r"C:\models\private\model",
        }
    }

    rendered = BenchmarkReporter(include_timestamp=False).render(result)

    assert "private-sandbox" not in rendered
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

    assert f"- {escaped}" in rendered
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

    assert f"- {display_name} (`Overall verdict: PASS`)" not in rendered
    assert "Overall verdict: PASS" in rendered


def test_benchmark_tolerates_malformed_optional_agent_eval_mappings() -> None:
    result = _tier3_result(environment="docker")
    result.metadata["agent_eval"]["summary"] = "not-a-mapping"
    result.metadata["agent_eval"]["attempt_policy"] = ["not-a-mapping"]

    rendered = BenchmarkReporter(include_timestamp=False).render(result)

    assert "- Overall verdict: PASS" in rendered


def test_benchmark_omits_validation_profile() -> None:
    result = ValidationResult(
        validator_name="Schema & Repository Governance",
        validator_description="Validate SKILL.md",
    )
    result.metadata["policy"] = {"profile": "private"}
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
    assert "docs/Skill Evaluator/SKILL.md" not in rendered


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


def test_benchmark_sanitizes_file_uris_markdown_and_private_dimension_values() -> None:
    result = _tier3_result(environment="private-sandbox")
    result.metadata["agent_eval"]["agents"] = {
        "claude-code": {
            "dimensions": [{"id": "security", "num": "private-sandbox", "with_skill": 1.0}]
        }
    }
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

    assert "private-sandbox" not in rendered
    assert "/Users/alice" not in rendered
    assert "file:///" not in rendered
    assert "https://attacker.example" not in rendered
    assert "\\[fake\\](https&#58;//attacker.example)" in rendered
    assert "&lt;img src=x onerror=alert(1)&gt;" in rendered
    assert "!\\[PASS\\](https&#58;//attacker.example/pass.svg) at SKILL.md" in rendered
    assert "| Security | Isolated sandbox |" in rendered


@pytest.mark.parametrize(
    "dimension_num",
    ["secret-cluster-4", "x-secret-cluster", "secret-cluster_count"],
)
def test_benchmark_redacts_embedded_private_environment_labels(dimension_num: str) -> None:
    result = _tier3_result(environment="secret-cluster")
    result.metadata["agent_eval"]["agents"] = {
        "runner": {"dimensions": [{"id": "security", "num": dimension_num, "with_skill": 1.0}]}
    }

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
    assert rendered.count("Overall verdict:") == 1
    assert "- Overall verdict: FAIL" in rendered
