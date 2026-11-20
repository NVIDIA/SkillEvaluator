# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Readers of files that ship with a skill or sit in the repository under test use the bounded YAML loader.

Plain ``yaml.safe_load`` expands chained merge keys inside the parser, and shares aliases that a later
``str()`` or ``json.dumps`` expands without limit. Each input here is cheap to parse with the bounded
loader and rejected by it; the assertions are on small values so a regression cannot expand the graph
inside a failing assertion.
"""

import json
from pathlib import Path

import pytest
from tests.yaml_alias_fixtures import merge_bomb_yaml, scalar_alias_fan_yaml

from skillevaluator.quality_config import QualityConfigError, load_quality_config
from skillevaluator.tier3.dataset_utils import load_dataset_entries, parse_dataset_text
from skillevaluator.tier3.harbor import adapter, report_data, runner
from skillevaluator.validators.policy import load_policy_file

HOSTILE = pytest.mark.parametrize(
    "text",
    [merge_bomb_yaml(), scalar_alias_fan_yaml()],
    ids=["merge-key-bomb", "scalar-alias-fan"],
)


def _skill_with_dataset(tmp_path: Path, text: str, name: str = "evals.yaml") -> Path:
    skill = tmp_path / "demo-skill"
    (skill / "evals").mkdir(parents=True)
    (skill / "evals" / name).write_text(text, encoding="utf-8")
    return skill


@pytest.mark.parametrize("suffix", [".yaml", ".yml"])
@HOSTILE
def test_dataset_text_over_the_structure_limits_is_a_value_error(text: str, suffix: str):
    with pytest.raises(ValueError, match="complexity limit"):
        parse_dataset_text(text, suffix)


@pytest.mark.parametrize("suffix", [".json", ".jsonl"])
def test_dataset_json_nested_past_the_parser_recursion_limit_is_a_value_error(suffix: str):
    with pytest.raises(ValueError, match="nested too deeply"):
        parse_dataset_text("[" * 200_000 + "]" * 200_000, suffix)


def test_dataset_text_still_parses_each_supported_format():
    assert parse_dataset_text("- question: q\n  ground_truth: g\n", ".YAML") == [{"question": "q", "ground_truth": "g"}]
    assert parse_dataset_text('[{"question": "q"}]', ".json") == [{"question": "q"}]
    assert parse_dataset_text('{"a": 1}\n\n{"a": 2}\n', ".jsonl") == [{"a": 1}, {"a": 2}]
    with pytest.raises(ValueError, match="Unsupported dataset format"):
        parse_dataset_text("x", ".toml")


@HOSTILE
def test_dataset_loader_rejects_a_hostile_yaml_dataset(tmp_path: Path, text: str):
    dataset = _skill_with_dataset(tmp_path, text) / "evals" / "evals.yaml"

    with pytest.raises(ValueError, match="complexity limit"):
        load_dataset_entries(dataset)


@HOSTILE
def test_harbor_adapter_rejects_a_hostile_yaml_dataset(tmp_path: Path, text: str):
    dataset = _skill_with_dataset(tmp_path, text) / "evals" / "evals.yaml"

    with pytest.raises(ValueError, match="complexity limit"):
        adapter._load_evals(dataset)


@HOSTILE
def test_projection_config_that_exceeds_the_limits_selects_nothing(text: str):
    assert adapter._decode_projection_config(text.encode("utf-8")) == ({}, False)


@HOSTILE
def test_dataset_generation_provenance_ignores_a_hostile_dataset(tmp_path: Path, text: str):
    assert runner._dataset_generation_provenance(_skill_with_dataset(tmp_path, text)) is None


@HOSTILE
def test_report_dataset_loader_returns_no_entries_for_a_hostile_dataset(tmp_path: Path, text: str):
    assert report_data.load_dataset(_skill_with_dataset(tmp_path, text)) == []


@HOSTILE
def test_custom_compose_build_args_reject_a_hostile_compose_file(tmp_path: Path, text: str):
    env_dir = tmp_path / "environment"
    env_dir.mkdir()
    (env_dir / "docker-compose.yaml").write_text(text, encoding="utf-8")
    (env_dir / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Cannot safely resolve custom Docker Compose build args"):
        adapter._compose_build_arg_overrides(env_dir, env_dir / "Dockerfile")


@HOSTILE
def test_custom_compose_validation_rejects_a_hostile_compose_file(tmp_path: Path, text: str):
    compose = tmp_path / "docker-compose.yaml"
    compose.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="cannot be read safely"):
        adapter._validate_and_sanitize_custom_compose(compose, allowed_env=set())


@HOSTILE
def test_kata_sidecar_check_reports_a_hostile_compose_file_as_unreadable(tmp_path: Path, text: str):
    task_dir = tmp_path / "task"
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "environment" / "docker-compose.yaml").write_text(text, encoding="utf-8")

    error = runner._staged_task_environment_error("kata", [task_dir])

    assert error is not None and error.startswith("Could not read the Docker Compose model")


@HOSTILE
def test_policy_file_that_exceeds_the_limits_is_an_invalid_policy(tmp_path: Path, text: str):
    policy = tmp_path / "policy.yaml"
    policy.write_text(text, encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid policy YAML"):
        load_policy_file(policy)


@HOSTILE
def test_quality_config_that_exceeds_the_limits_is_an_invalid_config(tmp_path: Path, text: str):
    config = tmp_path / "quality.yaml"
    config.write_text(text, encoding="utf-8")

    with pytest.raises(QualityConfigError, match="invalid YAML"):
        load_quality_config(config)


_LINE_SEPARATORS = {"U+2028": "\u2028", "U+2029": "\u2029", "U+0085": "\u0085"}


@pytest.mark.parametrize("newline", ["\n", "\r\n", "\r"], ids=["lf", "crlf", "cr"])
@pytest.mark.parametrize("separator", _LINE_SEPARATORS.values(), ids=_LINE_SEPARATORS.keys())
def test_jsonl_dataset_keeps_a_unicode_line_separator_inside_a_string(tmp_path: Path, separator: str, newline: str):
    # JSON allows these characters unescaped inside a string; only LF, CRLF and CR end a record.
    records = [
        {"question": f"first{separator}prompt", "ground_truth": "g", "expected_behavior": ["b"]},
        {"question": "second prompt", "ground_truth": "g", "expected_behavior": ["b"]},
    ]
    payload = newline.join(json.dumps(record, ensure_ascii=False) for record in records) + newline
    dataset = tmp_path / "evals.jsonl"
    dataset.write_bytes(payload.encode("utf-8"))

    assert parse_dataset_text(payload, ".jsonl") == records
    assert [entry["question"] for entry in load_dataset_entries(dataset)] == [r["question"] for r in records]
    assert [entry["question"] for entry in adapter._load_evals(dataset)] == [r["question"] for r in records]


def test_jsonl_dataset_still_skips_blank_lines_and_surrounding_whitespace():
    assert parse_dataset_text('  {"a": 1}  \n\n\t\n{"a": 2}\r\n', ".jsonl") == [{"a": 1}, {"a": 2}]


def test_yaml_dataset_syntax_error_keeps_what_is_wrong_and_where_without_the_source(tmp_path: Path):
    text = "- question: hunter2-secret-prompt\n  expected_behavior: [one, two\n"
    dataset = tmp_path / "evals.yaml"
    dataset.write_text(text, encoding="utf-8")

    for parse in (lambda: parse_dataset_text(text, ".yaml"), lambda: load_dataset_entries(dataset)):
        with pytest.raises(ValueError) as caught:
            parse()
        message = str(caught.value)
        assert "line 3, column 1" in message
        assert "expected ',' or ']'" in message
        assert "hunter2" not in message


def test_yaml_dataset_value_error_without_a_position_stays_generic():
    with pytest.raises(ValueError, match=r"^Input is not valid YAML$"):
        parse_dataset_text("- !!int abc\n", ".yaml")
