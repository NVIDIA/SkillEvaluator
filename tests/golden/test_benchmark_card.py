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
