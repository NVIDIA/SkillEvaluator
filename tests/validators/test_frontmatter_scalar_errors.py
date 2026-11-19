# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native YAML loading failures use the existing frontmatter error paths."""

from pathlib import Path

import pytest
import yaml

from skillevaluator.harness.overlay import _tab_indent_breaks_yaml
from skillevaluator.tier1.commands import run_validation
from skillevaluator.validators.frontmatter_parser import load_frontmatter_yaml, parse_frontmatter
from skillevaluator.validators.harness import HARNESS_METADATA_KEY
from skillevaluator.validators.markdown import markdown_link_targets
from skillevaluator.validators.quality_score import QualityScoreValidator
from skillevaluator.validators.schema import SchemaValidator
from skillevaluator.validators.version import VersionValidator


@pytest.fixture(
    params=[
        "2026-02-30",
        '!!int ""',
        "!!bool maybe",
        "!!float invalid",
        "!!timestamp invalid",
        r'"\U00110000"',
        r'"\UFFFFFFFF"',
        pytest.param("[" * 2000 + "leaf" + "]" * 2000, id="excessive-nesting"),
    ]
)
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


@pytest.mark.parametrize(
    ("raw_value", "kind", "value"),
    [
        ("2026-02-30", "timestamp", "2026-02-30"),
        ('!!int ""', "int", ""),
        ("!!bool maybe", "bool", "maybe"),
        ("!!float invalid", "float", "invalid"),
        ("!!timestamp invalid", "timestamp", "invalid"),
    ],
)
def test_native_scalar_error_keeps_value_and_actual_source_location(tmp_path, raw_value, kind, value):
    skill = tmp_path / "invalid-location"
    skill.mkdir()
    raw = f"name: invalid-location\ndescription: Test a source location\nextra: {raw_value}"
    (skill / "SKILL.md").write_text(f"---\n{raw}\n---\n# Example\n", encoding="utf-8")

    with pytest.raises(yaml.constructor.ConstructorError) as error:
        load_frontmatter_yaml(raw)
    assert error.value.problem_mark.line == 2
    assert error.value.problem_mark.column == 7
    assert error.value.problem == f"invalid {kind} value {value!r}"

    result = SchemaValidator().validate(skill)
    finding = next(finding for finding in result.findings if finding.check_name == "yaml_syntax")
    assert finding.line_number == 4
    assert f"invalid {kind} value {value!r}" in finding.message


@pytest.mark.parametrize("raw_value", [r'"\U00110000"', r'"\UFFFFFFFF"'])
def test_native_scanner_error_keeps_actual_source_location(tmp_path, raw_value):
    skill = tmp_path / "invalid-escape"
    skill.mkdir()
    raw = f"name: invalid-escape\ndescription: Test a source location\nextra: {raw_value}"
    (skill / "SKILL.md").write_text(f"---\n{raw}\n---\n# Example\n", encoding="utf-8")

    with pytest.raises(yaml.scanner.ScannerError) as error:
        load_frontmatter_yaml(raw)
    assert error.value.problem_mark.line == 2
    assert error.value.problem_mark.column == 10
    assert "invalid escape sequence" in error.value.problem

    result = SchemaValidator().validate(skill)
    finding = next(finding for finding in result.findings if finding.check_name == "yaml_syntax")
    assert finding.line_number == 4
    assert "invalid escape sequence" in finding.message


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


def test_schema_and_harness_finish_with_a_yaml_finding(malformed_skill: Path) -> None:
    results = run_validation(malformed_skill, checks="schema,harness")

    assert any(finding.check_name == "yaml_syntax" for result in results for finding in result.findings)
    harness = next(result for result in results if HARNESS_METADATA_KEY in result.metadata)
    assert len(harness.metadata[HARNESS_METADATA_KEY][str(malformed_skill)]) == 6


def test_harness_tab_check_handles_scalar_errors(malformed_skill: Path) -> None:
    content = (malformed_skill / "SKILL.md").read_text(encoding="utf-8")
    raw_yaml = content.split("---", 2)[1]
    # A tab inside a block scalar is valid YAML, so construction reaches the
    # invalid explicitly tagged value rather than failing during scanning.
    assert _tab_indent_breaks_yaml("note: |\n  \ttext\n" + raw_yaml)


def test_ordinary_nested_yaml_keeps_values_and_body_links(tmp_path: Path) -> None:
    skill = tmp_path / "ordinary-skill"
    skill.mkdir()
    manifest = skill / "SKILL.md"
    content = (
        "---\nname: ordinary-skill\n"
        "description: A skill for checking ordinary frontmatter. Use when testing validation.\n"
        "extra:\n  enabled: true\n  values: [1, [two, null]]\n---\n\n"
        "# Example\n\nRead [the guide](guide.md).\n"
    )
    manifest.write_text(content, encoding="utf-8")

    parsed, result = parse_frontmatter(manifest)

    assert result.passed
    assert parsed is not None
    assert parsed.yaml_data["extra"] == {"enabled": True, "values": [1, ["two", None]]}
    assert not any(finding.check_name == "yaml_syntax" for finding in SchemaValidator().validate(skill).findings)
    assert "guide.md" in markdown_link_targets(content)
