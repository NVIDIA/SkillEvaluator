# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Malformed YAML scalar constructors use the existing frontmatter error paths."""

from pathlib import Path

import pytest

from skillevaluator.validators.frontmatter_parser import parse_frontmatter
from skillevaluator.validators.markdown import markdown_link_targets
from skillevaluator.validators.quality_score import QualityScoreValidator
from skillevaluator.validators.schema import SchemaValidator
from skillevaluator.validators.version import VersionValidator


@pytest.fixture(params=["2026-02-30", '!!int ""', "!!bool maybe", "!!float invalid", "!!timestamp invalid"])
def malformed_skill(tmp_path: Path, request: pytest.FixtureRequest) -> Path:
    skill = tmp_path / "malformed-skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text(
        "---\nname: malformed-skill\n"
        "description: A skill for checking malformed frontmatter. Use when testing validation.\n"
        f"extra: {request.param}\n---\n\n"
        "# Example\n\n## Instructions\n\nRead [the guide](guide.md).\n\n## Examples\n\nExample usage.\n",
        encoding="utf-8",
    )
    return skill


def test_shared_parser_returns_a_frontmatter_error(malformed_skill: Path) -> None:
    parsed, result = parse_frontmatter(malformed_skill / "SKILL.md")

    assert parsed is None
    assert not result.passed
    assert any(message.startswith("Invalid YAML in frontmatter:") for message in result.errors)


def test_schema_reports_a_yaml_finding(malformed_skill: Path) -> None:
    result = SchemaValidator().validate(malformed_skill)

    assert not result.passed
    assert any(finding.check_name == "yaml_syntax" for finding in result.findings)


def test_version_reports_the_parse_error(malformed_skill: Path) -> None:
    result = VersionValidator().validate(malformed_skill)

    assert not result.passed
    assert any(message.startswith("Invalid YAML in frontmatter:") for message in result.errors)


def test_quality_finishes_without_hiding_schema_validation(malformed_skill: Path) -> None:
    result = QualityScoreValidator().validate(malformed_skill)

    assert "quality_scores" in result.metadata


def test_markdown_analysis_keeps_body_links(malformed_skill: Path) -> None:
    content = (malformed_skill / "SKILL.md").read_text(encoding="utf-8")

    assert "guide.md" in markdown_link_targets(content)
