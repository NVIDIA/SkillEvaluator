# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tier 3 readers of SKILL.md frontmatter use the bounded YAML loader.

Each reader stringifies ``name`` or ``description``. With plain ``yaml.safe_load`` a field that
aliases a nested graph expands without limit, so a few hundred bytes of SKILL.md exhaust memory.
"""

from pathlib import Path

import pytest
from tests.yaml_alias_fixtures import (
    OVER_LIMIT_LEVELS,
    YAML_DIRECTIVE_VALUE_ERROR,
    alias_bomb_skill_md,
    deeply_nested_flow,
    merge_bomb_yaml,
    scalar_alias_fan_yaml,
)

from skillevaluator.tier3 import generate_dataset
from skillevaluator.tier3.case_summary import _skill_description


def _skill(tmp_path: Path, manifest: str) -> Path:
    skill = tmp_path / "demo-skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text(manifest, encoding="utf-8")
    return skill


@pytest.mark.parametrize("field", ["name", "description"])
def test_dataset_generation_rejects_an_aliased_field_without_expanding_it(tmp_path: Path, field: str):
    skill = _skill(tmp_path, alias_bomb_skill_md(OVER_LIMIT_LEVELS, alias_fields=(field,)))

    with pytest.raises(ValueError, match="complexity limit exceeded"):
        generate_dataset._parse_skill(skill)


def test_dataset_generation_still_reads_a_plain_frontmatter(tmp_path: Path):
    skill = _skill(tmp_path, "---\nname: demo-skill\nname: renamed\ndescription: Use when demoing.\n---\n# Demo\n")

    parsed = generate_dataset._parse_skill(skill)

    assert parsed["name"] == "renamed"
    assert parsed["description"] == "Use when demoing."


def test_dataset_generation_falls_back_to_the_directory_name_for_invalid_yaml(tmp_path: Path):
    skill = _skill(tmp_path, "---\nname: [unclosed\n---\n# Demo\n")

    assert generate_dataset._parse_skill(skill)["name"] == "demo-skill"


def test_case_summary_reads_no_description_from_an_aliased_field(tmp_path: Path):
    skill = _skill(tmp_path, alias_bomb_skill_md(OVER_LIMIT_LEVELS, alias_fields=("description",)))

    assert _skill_description(skill) == ""


def test_case_summary_still_reads_a_plain_description(tmp_path: Path):
    skill = _skill(tmp_path, "---\nname: demo-skill\ndescription: Use when demoing.\n---\n# Demo\n")

    assert _skill_description(skill) == "Use when demoing."


@pytest.mark.usefixtures("default_int_digit_limit")
def test_case_summary_reads_no_description_from_a_yaml_directive_past_the_digit_limit(tmp_path: Path):
    # PyYAML's scanner raises a plain ValueError here; the reader caught every exception before.
    skill = _skill(tmp_path, f"---\n{YAML_DIRECTIVE_VALUE_ERROR}\n---\n# Demo\n")

    assert _skill_description(skill) == ""


@pytest.mark.usefixtures("default_int_digit_limit")
def test_dataset_generation_falls_back_to_the_directory_name_for_a_yaml_directive_past_the_digit_limit(tmp_path: Path):
    skill = _skill(tmp_path, f"---\n{YAML_DIRECTIVE_VALUE_ERROR}\n---\n# Demo\n")

    assert generate_dataset._parse_skill(skill)["name"] == "demo-skill"


@pytest.mark.usefixtures("default_int_digit_limit")
@pytest.mark.parametrize(
    "frontmatter",
    [
        merge_bomb_yaml(),
        scalar_alias_fan_yaml(),
        "name: *missing",
        "name: [unclosed",
        # These raise from plain yaml.safe_load, so reverting the reader to it fails the test.
        "name: renamed\nreleased: 2026-13-45",
        YAML_DIRECTIVE_VALUE_ERROR,
        deeply_nested_flow(),
    ],
    ids=[
        "merge-key-bomb",
        "scalar-alias-fan",
        "undefined-alias",
        "invalid-yaml",
        "invalid-date",
        "yaml-directive-digits",
        "deep-nesting",
    ],
)
def test_workspace_skill_name_falls_back_to_the_directory_name(tmp_path: Path, frontmatter: str):
    from skillevaluator.tier3.harbor.runner import _workspace_skill_name

    skill = _skill(tmp_path, f"---\n{frontmatter}\n---\n# Demo\n")

    assert _workspace_skill_name(skill) == "demo-skill"


def test_workspace_skill_name_reads_a_plain_name(tmp_path: Path):
    from skillevaluator.tier3.harbor.runner import _workspace_skill_name

    skill = _skill(tmp_path, "---\nname: renamed\n---\n# Demo\n")

    assert _workspace_skill_name(skill) == "renamed"


def test_create_eval_dataset_reports_over_limit_frontmatter_as_one_error_line(tmp_path: Path):
    from click.testing import CliRunner

    from skillevaluator.cli import cli

    skill = _skill(tmp_path, alias_bomb_skill_md(OVER_LIMIT_LEVELS))

    result = CliRunner().invoke(cli, ["tier3", "create-eval-dataset", str(skill), "--no-llm"])

    assert result.exit_code != 0
    assert "complexity limit exceeded" in result.output
    # A traceback would leave the ValueError as the exception; a clean error exits through SystemExit.
    assert isinstance(result.exception, SystemExit)
    assert "Traceback" not in result.output


def test_workspace_skill_name_keeps_the_last_of_a_repeated_key(tmp_path: Path):
    from skillevaluator.tier3.harbor.runner import _workspace_skill_name

    skill = _skill(tmp_path, "---\nname: first\nname: second\n---\n# Demo\n")

    assert _workspace_skill_name(skill) == "second"


def test_case_summary_keeps_the_last_of_a_repeated_description(tmp_path: Path):
    skill = _skill(tmp_path, "---\nname: demo-skill\ndescription: first\ndescription: second\n---\n# Demo\n")

    assert _skill_description(skill) == "second"
