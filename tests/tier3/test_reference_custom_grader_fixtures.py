# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import pytest

from skillevaluator.tier3.harbor.adapter import _stage_task_inputs

SKILL = Path(__file__).parents[2] / "src/skillevaluator/tier3/reference_skills/create-custom-grader"
EVALS = SKILL / "evals"
ENTRIES = {entry["id"]: entry for entry in json.loads((EVALS / "evals.json").read_text(encoding="utf-8"))["evals"]}


def _stage(tmp_path: Path, case_id: str) -> list[str]:
    _stage_task_inputs(
        tmp_path, input_files_dir=EVALS / "files", entry=ENTRIES[case_id], source_skill_path=SKILL, evals_dir=EVALS
    )
    return sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*") if path.is_file())


@pytest.mark.parametrize(
    ("case_id", "staged_dir", "required"),
    [
        (
            "create-custom-grader-001",
            "input/create-custom-grader-001/benchmark",
            ["task.yaml", "rubric.md", "code/totals.py", "code/fixtures/orders.csv", "prompts/direct.md"],
        ),
        (
            "create-custom-grader-002",
            "input/create-custom-grader-002/csv-summary",
            ["evals/evals.json", "evals/files/sales.csv", "scripts/summarize.py"],
        ),
    ],
)
def test_positive_cases_stage_only_their_fixture(
    tmp_path: Path, case_id: str, staged_dir: str, required: list[str]
) -> None:
    staged = _stage(tmp_path, case_id)

    assert all(path.startswith(f"input/{case_id}/") for path in staged)
    assert {f"{staged_dir}/{name}" for name in required} <= set(staged)
    assert staged_dir.removeprefix("input/") in ENTRIES[case_id]["prompt"]


def test_negative_case_stages_no_input(tmp_path: Path) -> None:
    assert _stage(tmp_path, "create-custom-grader-003-negative") == []


def test_fixtures_add_no_nested_skill_manifest() -> None:
    assert list((EVALS / "files").rglob("SKILL.md")) == []
