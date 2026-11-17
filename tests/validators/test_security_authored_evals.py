# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Security staging preserves authored evaluation inputs and their findings."""

import os
from pathlib import Path

import pytest

from skillevaluator.utils.tool_runner import Tools
from skillevaluator.validators.base import ValidationResult
from skillevaluator.validators.security import SecurityValidator


@pytest.mark.parametrize("eval_dir", ["evals", ".evals"])
@pytest.mark.parametrize("use_llm", [False, True])
def test_staged_security_scan_keeps_authored_eval_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, eval_dir: str, use_llm: bool
) -> None:
    skill = tmp_path / "skill"
    authored = {
        "SKILL.md": f"Read [{eval_dir}/config.yml]({eval_dir}/config.yml).\n".encode(),
        f"{eval_dir}/config.yml": b"dependencies:\n  - helper-skill\n",
        f"{eval_dir}/EVAL.md": b"# Evaluate the skill\n",
        f"{eval_dir}/evals.json": b'{"evals": []}\n',
        f"{eval_dir}/scripts/check.py": b"print('authored evaluation helper')\n",
        f"{eval_dir}/__pycache__/check.pyc": b"shipped bytecode",
    }
    for relative, content in authored.items():
        target = skill / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    generated = [skill / "results", skill / eval_dir / "results", skill / eval_dir / ".results"]
    for directory in generated:
        directory.mkdir()
        (directory / "generated.log").write_text("generated output must not be traversed\n")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "private.txt").write_text("outside the skill\n")
    link = skill / eval_dir / "external"
    link.symlink_to(outside, target_is_directory=True)
    broken_link = skill / eval_dir / "unavailable"
    broken_link.symlink_to("missing-target", target_is_directory=True)

    real_scandir = os.scandir

    def bounded_scandir(path="."):
        if not isinstance(path, int):
            assert Path(path) not in generated, "Generated output directories must not be traversed"
            assert Path(path) != outside, "Staging must not follow directory symlinks"
        return real_scandir(path)

    monkeypatch.setattr(os, "scandir", bounded_scandir)
    monkeypatch.setattr(Tools.skillspector, "_path", "/unused/skillspector")
    validator = SecurityValidator(use_llm=use_llm)
    seen: list[tuple[Path, bool]] = []

    def inspect_input(scan_root: Path, *, original_root, use_llm, baseline):
        assert scan_root != skill.resolve()
        assert original_root == skill.resolve()
        for relative, content in authored.items():
            target = scan_root / relative
            assert target.is_file(), f"Authored input was removed: {relative}"
            assert target.read_bytes() == content
        for directory in generated:
            assert not (scan_root / directory.relative_to(skill)).exists()
        assert (scan_root / eval_dir / "external").is_symlink()
        assert (scan_root / eval_dir / "external").readlink() == outside
        assert (scan_root / eval_dir / "unavailable").is_symlink()
        seen.append((scan_root, use_llm))
        return ValidationResult()

    monkeypatch.setattr(validator, "_run_skillspector_once", inspect_input)
    validator._run_skillspector(skill)

    assert [llm for _root, llm in seen] == ([False, True] if use_llm else [False])
    assert all(not root.exists() for root, _llm in seen)


@pytest.mark.parametrize("eval_dir", ["evals", ".evals"])
def test_authored_evals_alone_do_not_require_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, eval_dir: str
) -> None:
    skill = tmp_path / "skill"
    config = skill / eval_dir / "config.yml"
    config.parent.mkdir(parents=True)
    config.write_text("dependencies: []\n")
    monkeypatch.setattr(Tools.skillspector, "_path", "/unused/skillspector")
    validator = SecurityValidator()
    seen = []

    def inspect_input(scan_root: Path, *, original_root, use_llm, baseline):
        seen.append(scan_root)
        assert scan_root == skill.resolve()
        assert original_root is None
        assert (scan_root / eval_dir / "config.yml").read_text() == "dependencies: []\n"
        return ValidationResult()

    monkeypatch.setattr(validator, "_run_skillspector_once", inspect_input)
    validator._run_skillspector(skill)

    assert seen == [skill.resolve()]


@pytest.mark.parametrize("eval_dir", ["evals", ".evals"])
@pytest.mark.parametrize("absolute", [False, True])
@pytest.mark.parametrize("with_source_root", [False, True])
def test_authored_eval_finding_is_not_classified_as_generated(
    tmp_path: Path, eval_dir: str, absolute: bool, with_source_root: bool
) -> None:
    skill = tmp_path / "skill"
    target = skill / eval_dir / "scripts" / "check.py"
    target.parent.mkdir(parents=True)
    target.write_text("authored script\n")
    reported_path = target if absolute else target.relative_to(skill)
    issue = {"location": {"file": str(reported_path), "start_line": 1}}

    assert not SecurityValidator._is_generated_artifact_issue(issue, source_root=skill if with_source_root else None)


@pytest.mark.parametrize("eval_dir", ["evals", ".evals"])
@pytest.mark.parametrize("output_dir", ["results", ".results"])
def test_generated_eval_output_is_still_classified_as_generated(tmp_path: Path, eval_dir: str, output_dir: str) -> None:
    skill = tmp_path / "skill"
    target = skill / eval_dir / output_dir / "output.md"
    target.parent.mkdir(parents=True)
    target.write_text("generated output\n")
    issue = {"location": {"file": str(target.relative_to(skill)), "start_line": 1}}

    assert SecurityValidator._is_generated_artifact_issue(issue, source_root=skill)
