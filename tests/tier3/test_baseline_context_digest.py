# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Only the generated repository-context metadata is excluded from task identity."""

from pathlib import Path

import pytest

from skillevaluator.tier3.baseline_store import baseline_task_tree_digest


@pytest.mark.parametrize(
    "relative",
    ["repo-context.json", "repo/repo-context.json", "environment/files/repo-context.json", "tests/repo-context.json"],
)
def test_authored_context_filename_changes_baseline_task_identity(tmp_path: Path, relative: str) -> None:
    task = tmp_path / "case-001"
    fixture = task / relative
    fixture.parent.mkdir(parents=True)
    fixture.write_text('{"expected": "first"}', encoding="utf-8")
    original = baseline_task_tree_digest({"codex": [task]})

    fixture.write_text('{"expected": "second"}', encoding="utf-8")

    assert baseline_task_tree_digest({"codex": [task]}) != original


def test_generated_context_metadata_remains_outside_baseline_task_identity(tmp_path: Path) -> None:
    task = tmp_path / "case-001"
    metadata = task / "environment" / "repo-context.json"
    metadata.parent.mkdir(parents=True)
    metadata.write_text('{"staged_path": "first-run"}', encoding="utf-8")
    original = baseline_task_tree_digest({"codex": [task]})

    metadata.write_text('{"staged_path": "second-run"}', encoding="utf-8")

    assert baseline_task_tree_digest({"codex": [task]}) == original
