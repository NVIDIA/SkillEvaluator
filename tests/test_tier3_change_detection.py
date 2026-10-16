# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from pathlib import Path

import pytest

from skillevaluator.tier3.change_detection import tier3_run_decision


def _write_skill(
    root: Path,
    *,
    owner: str = "platform",
    description: str = "Use the demo skill.",
    body: str = "# Demo\n\nFollow the workflow.\n",
    artifact: str | None = None,
) -> None:
    root.mkdir()
    (root / "SKILL.md").write_text(
        "---\n"
        "name: demo\n"
        f"description: {description}\n"
        "metadata:\n"
        f"  owner: {owner}\n"
        "---\n"
        f"{body}",
        encoding="utf-8",
    )
    if artifact is not None:
        (root / artifact).write_text("# Prior Tier 3 result\n", encoding="utf-8")


@pytest.mark.parametrize("artifact", ["skill-card.md", "BENCHMARK.md"])
def test_metadata_only_change_skips_tier3_with_prior_evidence(tmp_path: Path, artifact: str) -> None:
    previous = tmp_path / "previous"
    current = tmp_path / "current"
    _write_skill(previous, artifact=artifact)
    _write_skill(current, owner="release-engineering")

    decision = tier3_run_decision(current / "SKILL.md", previous)

    assert decision.should_skip is True
    assert decision.reason_code == "metadata_only_change"
    assert decision.evidence_file == artifact
    assert decision.to_dict()["should_run"] is False


def test_metadata_only_change_requires_prior_tier3_evidence(tmp_path: Path) -> None:
    previous = tmp_path / "previous"
    current = tmp_path / "current"
    _write_skill(previous)
    _write_skill(current, owner="release-engineering", artifact="BENCHMARK.md")

    decision = tier3_run_decision(current, previous)

    assert decision.should_run is True
    assert decision.reason_code == "previous_tier3_evidence_missing"


@pytest.mark.parametrize(
    ("current_kwargs", "reason_code"),
    [
        ({"body": "# Demo\n\nChanged workflow.\n"}, "skill_body_changed"),
        ({"description": "Different behavior."}, "skill_frontmatter_changed"),
    ],
)
def test_behavioral_skill_changes_require_tier3(
    tmp_path: Path,
    current_kwargs: dict[str, str],
    reason_code: str,
) -> None:
    previous = tmp_path / "previous"
    current = tmp_path / "current"
    _write_skill(previous, artifact="BENCHMARK.md")
    _write_skill(current, **current_kwargs)

    decision = tier3_run_decision(current, previous)

    assert decision.should_run is True
    assert decision.reason_code == reason_code


def test_invalid_frontmatter_requires_tier3(tmp_path: Path) -> None:
    previous = tmp_path / "previous"
    current = tmp_path / "current"
    _write_skill(previous, artifact="skill-card.md")
    current.mkdir()
    (current / "SKILL.md").write_text(
        "---\nmetadata: [\n---\n# Demo\n\nFollow the workflow.\n",
        encoding="utf-8",
    )

    decision = tier3_run_decision(current, previous)

    assert decision.should_run is True
    assert decision.reason_code == "skill_frontmatter_invalid"
