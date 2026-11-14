# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``validate`` treats a SKILL.md or workflow-rules.mdc argument as the folder that holds it."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from click.testing import CliRunner

from skillevaluator import cli as cli_module

FIXTURE = Path(__file__).parents[1] / "fixtures" / "skills" / "simple"


@pytest.fixture(autouse=True)
def _default_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("SKILLEVALUATOR_PROFILE", raising=False)
    monkeypatch.delenv("SKILLEVALUATOR_PREVIOUS_VERSION", raising=False)


def _validate(target: Path, output: Path, *args: str) -> tuple[int, dict]:
    result = CliRunner().invoke(
        cli_module.cli,
        ["validate", str(target), "--tiers", "1", "--no-llm", "-r", "json", "-o", str(output), *args],
    )
    report = json.loads(next(output.glob("*.json")).read_text(encoding="utf-8"))
    report.pop("generated_at")
    return result.exit_code, report


def _skill_names(report: dict) -> list[str]:
    return [skill["name"] for skill in report["skills"]]


def test_skill_manifest_file_validates_like_its_folder(tmp_path: Path) -> None:
    skill = tmp_path / "simple"
    shutil.copytree(FIXTURE, skill)
    checks = ("--checks", "schema,version,quality")

    folder = _validate(skill, tmp_path / "folder-reports", *checks)
    manifest = _validate(skill / "SKILL.md", tmp_path / "manifest-reports", *checks)

    assert folder[0] == 0
    assert manifest == folder
    assert _skill_names(manifest[1]) == ["simple"]
    assert {result["publication_target"]["skill_name"] for result in manifest[1]["results"]} == {"simple"}


def test_skill_manifest_file_scans_the_whole_skill(tmp_path: Path) -> None:
    # Only the bundled script carries hidden characters, so a scan of SKILL.md alone would pass.
    skill = tmp_path / "simple"
    shutil.copytree(FIXTURE, skill)
    script = skill / "scripts" / "summarize.py"
    hidden = "".join(chr(0xE0000 + ord(char)) for char in "hidden")
    script.write_text(script.read_text(encoding="utf-8") + f"# note{hidden}\n", encoding="utf-8")

    folder = _validate(skill, tmp_path / "folder-reports", "--checks", "unicode")
    manifest = _validate(skill / "SKILL.md", tmp_path / "manifest-reports", "--checks", "unicode")

    assert folder[0] == 1
    assert manifest == folder


def test_workflow_manifest_file_validates_like_its_folder(tmp_path: Path, sample_workflow_dir: Path) -> None:
    folder = _validate(sample_workflow_dir, tmp_path / "folder-reports", "--checks", "schema")
    manifest = _validate(
        sample_workflow_dir / "workflow-rules.mdc", tmp_path / "manifest-reports", "--checks", "schema"
    )

    assert folder[0] == 0
    assert manifest == folder


def test_linked_skill_manifest_validates_the_folder_holding_the_link(tmp_path: Path) -> None:
    skill = tmp_path / "simple"
    shutil.copytree(FIXTURE, skill)
    linked = tmp_path / "linked"
    linked.mkdir()
    try:
        (linked / "SKILL.md").symlink_to(skill / "SKILL.md")
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"symlinks are unavailable: {exc}")
    # The Tier 2 guard refuses a linked argument before any mapping, so leave Tier 2 out.
    args = ("--no-dedup", "--checks", "schema")

    folder = _validate(linked, tmp_path / "folder-reports", *args)
    manifest = _validate(linked / "SKILL.md", tmp_path / "manifest-reports", *args)

    assert manifest == folder
    assert _skill_names(manifest[1]) == ["linked"]


@pytest.mark.parametrize("tiers", [(), ("--tiers", "1")], ids=["default-tiers", "tier1-only"])
def test_manifest_in_linked_skill_folder_is_refused_like_the_folder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, tiers: tuple[str, ...]
) -> None:
    target = tmp_path / "real-skill"
    target.mkdir()
    (target / "SKILL.md").write_text("---\nname: sample\ndescription: Sample skill\n---\n", encoding="utf-8")
    linked_target = tmp_path / "linked-skill"
    try:
        linked_target.symlink_to(target, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"Directory symlinks are unavailable: {exc}")

    def must_not_run(*_args, **_kwargs):
        raise AssertionError("Validation must not receive a linked root unless --no-dedup is passed")

    monkeypatch.setattr(cli_module, "run_validation", must_not_run)

    # The guard runs before --tiers narrows the run, so --tiers 1 alone does not skip it.
    for argument in (linked_target, linked_target / "SKILL.md"):
        result = CliRunner().invoke(cli_module.cli, ["validate", str(argument), *tiers])

        assert result.exit_code == 2, result.output
        assert "symlink or reparse point: linked-skill" in result.output
        assert str(target) not in result.output


def test_manifest_in_linked_skill_folder_validates_like_the_folder_with_no_dedup(tmp_path: Path) -> None:
    skill = tmp_path / "simple"
    shutil.copytree(FIXTURE, skill)
    linked = tmp_path / "linked"
    try:
        linked.symlink_to(skill, target_is_directory=True)
    except (OSError, NotImplementedError) as exc:
        pytest.skip(f"Directory symlinks are unavailable: {exc}")
    args = ("--no-dedup", "--checks", "schema")

    folder = _validate(linked, tmp_path / "folder-reports", *args)
    manifest = _validate(linked / "SKILL.md", tmp_path / "manifest-reports", *args)

    assert folder[0] == 0
    assert manifest == folder


def test_skill_manifest_file_keeps_default_reports_outside_the_skill(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    skill = tmp_path / "simple"
    shutil.copytree(FIXTURE, skill)
    monkeypatch.chdir(skill)

    result = CliRunner().invoke(
        cli_module.cli, ["validate", "SKILL.md", "--tiers", "1", "--no-llm", "--checks", "schema", "-r", "json"]
    )

    assert result.exit_code == 0, result.output
    assert (tmp_path / "simple-reports" / "BENCHMARK.md").is_file()
    assert not (skill / "reports").exists()


def test_skill_manifest_file_refuses_an_output_dir_inside_the_skill(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    skill = tmp_path / "simple"
    shutil.copytree(FIXTURE, skill)
    monkeypatch.chdir(skill)
    args = ("--tiers", "1", "--no-llm", "--checks", "schema", "-r", "json", "-o", "out")
    message = "Report output must be outside the publication target; use an external --output-dir: out"

    folder = CliRunner().invoke(cli_module.cli, ["validate", ".", *args])
    manifest = CliRunner().invoke(cli_module.cli, ["validate", "SKILL.md", *args])

    assert folder.exit_code == 1, folder.output
    assert message in folder.output
    assert manifest.exit_code == folder.exit_code, manifest.output
    assert message in manifest.output
    assert not (skill / "out").exists()


def test_other_file_in_a_skill_is_validated_as_given(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    skill = tmp_path / "skills" / "demo"
    script = skill / "scripts" / "run.py"
    script.parent.mkdir(parents=True)
    script.write_text("print('ok')\n", encoding="utf-8")
    (skill / "SKILL.md").write_text("---\nname: demo\ndescription: Demo skill\n---\n", encoding="utf-8")
    seen = []
    monkeypatch.setattr(
        cli_module, "run_validation", lambda path, **kwargs: seen.append((path, kwargs["content_type"])) or []
    )

    result = CliRunner().invoke(
        cli_module.cli, ["validate", str(script), "--tiers", "1", "-o", str(tmp_path / "reports")]
    )

    assert result.exit_code == 0, result.output
    assert seen == [(script.resolve(), "skill")]
