# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``tier3 validate`` rejects dataset fields that scoring would silently misread."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from click.testing import CliRunner

from skillevaluator.cli import cli
from skillevaluator.tier3.evals_spec import validate_skillevaluators

_LEAKED = "private-marker-value"


def _agentskills_case(**fields: Any) -> dict[str, Any]:
    return {"id": "case-1", "prompt": "Summarize notes.txt", "expected_output": "A three line summary", **fields}


def _legacy_case(**fields: Any) -> dict[str, Any]:
    return {"id": "case-1", "question": "Summarize notes.txt", "ground_truth": "A three line summary", **fields}


def _write_skill(tmp_path: Path, shape: str, **fields: Any) -> Path:
    skill = tmp_path / "summarizer"
    (skill / "evals").mkdir(parents=True)
    (skill / "SKILL.md").write_text("---\nname: summarizer\ndescription: Summarize notes\n---\n", encoding="utf-8")
    if shape == "agentskills":
        payload: Any = {"skill_name": "summarizer", "evals": [_agentskills_case(**fields)]}
    else:
        payload = [_legacy_case(**fields)]
    (skill / "evals" / "evals.json").write_text(json.dumps(payload), encoding="utf-8")
    return skill


def _errors(skill: Path) -> list[str]:
    return [item.message for item in validate_skillevaluators(skill) if item.status == "error"]


SHAPES = pytest.mark.parametrize("shape", ["agentskills", "legacy"])


@SHAPES
@pytest.mark.parametrize(
    ("value", "type_name"),
    [(7, "int"), (["summarizer"], "list"), ({"name": "summarizer"}, "dict"), (True, "bool"), (1.5, "float")],
)
def test_validate_rejects_an_expected_skill_that_is_not_a_string_or_null(
    tmp_path: Path, shape: str, value: object, type_name: str
) -> None:
    errors = _errors(_write_skill(tmp_path, shape, expected_skill=value))

    assert len(errors) == 1
    assert errors[0].startswith(f"expected_skill must be a skill name string or null, got {type_name}")


@SHAPES
@pytest.mark.parametrize("value", ["  ", "\t\n"])
def test_validate_rejects_a_whitespace_only_expected_skill(tmp_path: Path, shape: str, value: str) -> None:
    errors = _errors(_write_skill(tmp_path, shape, expected_skill=value))

    assert len(errors) == 1
    assert errors[0].startswith("expected_skill contains only whitespace")


@SHAPES
@pytest.mark.parametrize("field", ["assertions", "expected_behavior"])
@pytest.mark.parametrize(
    ("value", "message"),
    [
        (_LEAKED, "must be a list of strings, got str."),
        ({"step": _LEAKED}, "must be a list of strings, got dict."),
        (5, "must be a list of strings, got int."),
        ([3], "must contain only strings; found int."),
        ([["x"], 3, {"k": _LEAKED}], "must contain only strings; found dict, int, list."),
        (["", "n/a", 0, False, None, {}, []], "must contain only strings; found NoneType, bool, dict, int, list."),
    ],
)
def test_validate_rejects_behaviors_that_are_not_a_list_of_strings(
    tmp_path: Path, shape: str, field: str, value: object, message: str
) -> None:
    errors = _errors(_write_skill(tmp_path, shape, **{field: value}))

    assert errors == [f"{field} {message}"]


@SHAPES
@pytest.mark.parametrize(
    "fields",
    [
        {},
        {"expected_skill": None},
        {"expected_skill": ""},
        {"expected_behavior": []},
        {"assertions": []},
        {"assertions": None, "expected_behavior": None},
        {"assertions": ["Lists the three key points"], "expected_skill": "summarizer"},
        {"expected_behavior": ["Lists the three key points", "", "n/a"]},
    ],
    ids=str,
)
def test_validate_accepts_omitted_null_empty_and_well_formed_fields(
    tmp_path: Path, shape: str, fields: dict[str, Any]
) -> None:
    assert _errors(_write_skill(tmp_path, shape, **fields)) == []


def test_validate_reports_both_behavior_fields_as_authored(tmp_path: Path) -> None:
    skill = _write_skill(tmp_path, "agentskills", assertions=[0], expected_behavior=[False])

    assert _errors(skill) == [
        "assertions must contain only strings; found int.",
        "expected_behavior must contain only strings; found bool.",
    ]


def test_validate_error_names_the_entry_and_never_echoes_the_value(tmp_path: Path) -> None:
    skill = _write_skill(tmp_path, "agentskills", assertions=_LEAKED, expected_skill={"name": _LEAKED})

    findings = [item for item in validate_skillevaluators(skill) if item.status == "error"]

    assert [item.message.split()[0] for item in findings] == ["expected_skill", "assertions"]
    assert {item.path for item in findings} == {"entry case-1"}
    assert all(_LEAKED not in item.message for item in findings)


def test_tier3_validate_cli_exits_nonzero_for_a_malformed_case(tmp_path: Path) -> None:
    skill = _write_skill(tmp_path, "agentskills", expected_skill=7)

    result = CliRunner().invoke(cli, ["tier3", "validate", str(skill), "--json"])

    assert result.exit_code == 1, result.output
    payload = json.loads(result.output)
    assert [(item["path"], item["status"]) for item in payload if item["status"] == "error"] == [
        ("entry case-1", "error")
    ]
