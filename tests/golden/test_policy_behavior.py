# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Golden behavior tests for the default public validation policy."""

from __future__ import annotations

from pathlib import Path

from click.testing import CliRunner

from skillevaluator.cli import cli

_BODY = """
# {name}

## Instructions

Do the task and report the result clearly.

## Examples

User: "do it"

Assistant: "done"
"""


def _write_skill(root: Path, name: str, author_line: str | None) -> Path:
    skill = root / name
    skill.mkdir(parents=True)
    meta = f"metadata:\n  author: {author_line}\n" if author_line else ""
    (skill / "SKILL.md").write_text(
        f"---\nname: {name}\ndescription: A test skill for policy behavior checks.\n{meta}---\n"
        + _BODY.format(name=name),
        encoding="utf-8",
    )
    return skill


def _validate(skill: Path, out: Path, *args: str) -> int:
    result = CliRunner().invoke(
        cli,
        ["validate", str(skill), "--no-llm", "--checks", "schema", "-o", str(out), *args],
    )
    return result.exit_code


def test_default_policy_accepts_public_author_metadata(tmp_path: Path) -> None:
    out = tmp_path / "reports"
    nvidia = _write_skill(tmp_path, "nvidia-skill", "Dev One <dev@nvidia.com>")
    external = _write_skill(tmp_path, "ext-skill", "Jane Doe <jane@example.com>")
    missing = _write_skill(tmp_path, "no-author-skill", None)

    assert _validate(nvidia, out) == 0
    assert _validate(external, out) == 0
    assert _validate(missing, out) == 1


def test_external_alias_matches_default_public_policy(tmp_path: Path) -> None:
    out = tmp_path / "reports"
    external = _write_skill(tmp_path, "ext-skill", "Jane Doe <jane@example.com>")
    missing = _write_skill(tmp_path, "no-author-skill", None)

    assert _validate(external, out, "--external") == 0  # domain check disabled
    assert _validate(missing, out, "--external") == 1


def test_benchmark_md_generated_for_skill(tmp_path: Path) -> None:
    out = tmp_path / "reports"
    nvidia = _write_skill(tmp_path, "nvidia-skill", "Dev One <dev@nvidia.com>")
    assert _validate(nvidia, out) == 0
    assert (out / "BENCHMARK.md").exists()
