# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The bundled create-custom-grader skill names a validate command that exists."""

from __future__ import annotations

from pathlib import Path

SKILL = Path(__file__).parents[2] / "src/skillevaluator/tier3/reference_skills/create-custom-grader"
SKILL_TEXT = (SKILL / "SKILL.md").read_text(encoding="utf-8")
EVALS_TEXT = (SKILL / "evals" / "evals.json").read_text(encoding="utf-8")


def test_every_command_that_names_the_contract_flag_is_a_tier3_validate_command() -> None:
    # Only `tier3 validate` has --harbor-contract; `validate` rejects it as an unknown option.
    lines = [line for line in (SKILL_TEXT + "\n" + EVALS_TEXT).splitlines() if "--harbor-contract" in line]

    assert lines
    assert all("tier3 validate" in line for line in lines), lines
