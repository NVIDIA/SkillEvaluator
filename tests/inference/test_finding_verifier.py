# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Containment checks for FindingVerifier file-context reads."""

from pathlib import Path

from skillevaluator.inference.finding_verifier import FindingVerifier


def _skill_with_outside(tmp_path: Path) -> tuple[Path, Path]:
    outside = tmp_path / "outside.txt"
    outside.write_text("hello-from-outside\n", encoding="utf-8")
    skill = tmp_path / "skill"
    skill.mkdir()
    (skill / "SKILL.md").write_text(
        "---\nname: fv\ndescription: Containment check.\n---\n# inside\n",
        encoding="utf-8",
    )
    return skill, outside


def test_read_file_context_allows_in_skill_relative_path(tmp_path):
    skill, _ = _skill_with_outside(tmp_path)
    context = FindingVerifier._read_file_context(skill, "SKILL.md")
    assert context is not None
    assert "inside" in context
    assert "hello-from-outside" not in context


def test_read_file_context_rejects_parent_relative_path(tmp_path):
    skill, _ = _skill_with_outside(tmp_path)
    assert FindingVerifier._read_file_context(skill, "../outside.txt") is None


def test_read_file_context_rejects_absolute_path_outside_skill(tmp_path):
    skill, outside = _skill_with_outside(tmp_path)
    assert FindingVerifier._read_file_context(skill, str(outside)) is None


def test_read_file_context_rejects_symlink_pointing_outside_skill(tmp_path):
    skill, outside = _skill_with_outside(tmp_path)
    link = skill / "notes.md"
    link.symlink_to(outside)
    assert FindingVerifier._read_file_context(skill, "notes.md") is None


def test_build_prompt_redacts_symlink_finding_payload(tmp_path):
    from skillevaluator.models import Finding, Severity

    skill, outside = _skill_with_outside(tmp_path)
    outside.write_text("secret-pii-data@example.com\n", encoding="utf-8")
    link = skill / "notes.md"
    link.symlink_to(outside)

    finding = Finding(
        category="PII",
        severity=Severity.HIGH,
        check_name="email",
        message="Found email secret-pii-data@example.com",
        file_path="notes.md",
        line_number=1,
        line_content="secret-pii-data@example.com",
    )
    prompt = FindingVerifier()._build_prompt([finding], skill)
    assert "secret-pii-data" not in prompt
    assert "(redacted: file outside skill boundary)" in prompt
