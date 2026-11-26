# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""One unreadable manifest must not discard quality results for other skills."""

from __future__ import annotations

from pathlib import Path

import pytest

from skillevaluator.validators.quality_score import QualityScoreValidator


def _skill(root, name, *, readable=True):
    skill = root / name
    skill.mkdir()
    content = f"---\nname: {name}\ndescription: Use when checking a valid skill.\n---\n# Instructions\n"
    (skill / "SKILL.md").write_bytes(content.encode("utf-8") + (b"" if readable else b"\xff"))
    return skill


def test_invalid_utf8_manifest_fails_without_a_quality_grade(tmp_path):
    skill = _skill(tmp_path, "broken", readable=False)

    result = QualityScoreValidator().validate(skill)

    assert result.passed is False
    assert any(str(skill / "SKILL.md") in reason and "read" in reason for reason in result.gate_failures)
    assert "quality_scores" not in result.metadata
    assert "quality_config" in result.metadata
    assert not any(detail.check_name == "quality_score" for detail in result.success_details)


@pytest.mark.parametrize("failure", ["encoding", "permission", "disappeared"])
def test_unreadable_member_keeps_healthy_quality_results(tmp_path, monkeypatch, failure):
    broken = _skill(tmp_path, "broken", readable=failure != "encoding")
    healthy = _skill(tmp_path, "healthy")
    validator = QualityScoreValidator()
    expected = validator.validate(healthy).metadata["quality_scores"]
    manifest = broken / "SKILL.md"

    if failure != "encoding":
        original = Path.read_text

        def fail_read(path, *args, **kwargs):
            if path == manifest:
                if failure == "permission":
                    raise PermissionError("manifest is not readable")
                raise FileNotFoundError("manifest disappeared after discovery")
            return original(path, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", fail_read)

    result = validator.validate(tmp_path)

    assert result.passed is False
    assert any("[broken]" in reason and str(manifest) in reason for reason in result.gate_failures)
    assert result.metadata["quality_scores_all"] == [expected]
    assert result.metadata["quality_scores"]["skill_count"] == 1
    assert result.metadata["quality_scores"]["overall_score"] == expected["overall_score"]


def test_all_unreadable_members_do_not_produce_an_aggregate_grade(tmp_path):
    _skill(tmp_path, "first", readable=False)
    _skill(tmp_path, "second", readable=False)

    result = QualityScoreValidator().validate(tmp_path)

    assert result.passed is False
    assert len(result.gate_failures) == 2
    assert "quality_scores" not in result.metadata
    assert "quality_scores_all" not in result.metadata
