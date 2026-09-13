# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Filesystem safety regressions for Harbor task adaptation."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
from pathlib import Path

import pytest

from skillevaluator.tier3.harbor.adapter import (
    _copy_custom_grader,
    _rebase_custom_dockerfile_content,
    _stage_repo_context,
    generate_harbor_tasks,
    stage_native_harbor_tasks,
)

_GRADER_CANDIDATES = (
    Path("grader.py"),
    Path("grader.sh"),
    Path("tests/grader.py"),
    Path("tests/grader.sh"),
)

_UNSAFE_EVAL_IDS = (
    "",
    ".",
    "..",
    "case/001",
    r"case\001",
    "C:case",
    r"C:\case",
    r"\\server\share",
    "CON",
    "com1",
)

_SENSITIVE_REPO_FILES = (
    ".env.production",
    ".env.staging.local",
    ".git-credentials",
    ".docker/config.json",
    ".kube/config",
    ".aws/credentials",
    ".aws/config",
    ".config/gcloud/application_default_credentials.json",
    ".azure/accessTokens.json",
    ".netrc",
    "_netrc",
    ".npmrc",
    ".pypirc",
    ".terraform.d/credentials.tfrc.json",
    ".ssh/id_ed25519",
    "keys/deploy.pem",
    "keys/client.KEY",
)

_SAFE_REPO_FILES = (
    ".env.example",
    ".env.template",
    "docs/environment.production.md",
    "docs/git-credentials.md",
    "examples/docker/config.json",
    "examples/kube/config",
    "examples/aws/config",
    ".docker/config.example.json",
    ".kube/config.example",
    "keys/id_ed25519.pub",
    "certs/public.crt",
    "credentials.example.json",
    "package.json",
)


def _write_generated_skill(tmp_path: Path, entry_ids: list[str]) -> Path:
    skill = tmp_path / "skill"
    evals_dir = skill / "evals"
    evals_dir.mkdir(parents=True)
    (skill / "SKILL.md").write_text("# Test skill\n", encoding="utf-8")
    entries = [
        {
            "id": entry_id,
            "question": f"Run case {index}",
            "expected_skill": skill.name,
        }
        for index, entry_id in enumerate(entry_ids)
    ]
    (evals_dir / "evals.json").write_text(json.dumps(entries), encoding="utf-8")
    return skill


def _write_native_skill(tmp_path: Path) -> Path:
    skill = tmp_path / "skill"
    task = skill / "evals" / "harbor" / "case-001"
    task.mkdir(parents=True)
    (skill / "SKILL.md").write_text("# Test skill\n", encoding="utf-8")
    (task / "task.toml").write_text(
        'schema_version = "1.3"\n\n[task]\nname = "nvidia/case-001"\n\n[environment]\n',
        encoding="utf-8",
    )
    return skill


def _write_repo_context_fixture(repo: Path) -> Path:
    skill = repo / "skill"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("# Test skill\n", encoding="utf-8")
    for rel in (*_SENSITIVE_REPO_FILES, *_SAFE_REPO_FILES):
        path = repo / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"fixture: {rel}\n", encoding="utf-8")
    return skill


def _assert_sensitive_repo_files_are_excluded(repo: Path, skill: Path, env_dir: Path) -> None:
    env_dir.mkdir(parents=True)

    metadata = _stage_repo_context(env_dir, source_skill_path=skill, mode="full")

    staged_root = env_dir / "repo"
    for rel in _SENSITIVE_REPO_FILES:
        assert not (staged_root / rel).exists(), f"sensitive file was staged: {rel}"
    for rel in _SAFE_REPO_FILES:
        assert (staged_root / rel).read_text(encoding="utf-8") == f"fixture: {rel}\n"
    staged_sources = {Path(item["source"]).relative_to(repo).as_posix() for item in metadata["files"]}
    assert staged_sources.isdisjoint(_SENSITIVE_REPO_FILES)


def test_copy_repo_excludes_tracked_sensitive_files(tmp_path: Path) -> None:
    repo = tmp_path / "tracked-repo"
    skill = _write_repo_context_fixture(repo)
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "add", "-f", "--", "."], check=True)

    _assert_sensitive_repo_files_are_excluded(repo, skill, tmp_path / "tracked-task" / "environment")


def test_copy_repo_excludes_sensitive_files_without_git(tmp_path: Path) -> None:
    repo = tmp_path / "plain-repo"
    skill = _write_repo_context_fixture(repo)

    _assert_sensitive_repo_files_are_excluded(repo, skill, tmp_path / "plain-task" / "environment")


@pytest.mark.parametrize("tracked", [False, True], ids=["plain", "git"])
def test_copy_repo_prunes_excluded_output_before_any_file_inspection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tracked: bool,
) -> None:
    repo = tmp_path / "repo"
    skill = repo / "skill"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("# Test skill\n", encoding="utf-8")
    (repo / "README.md").write_text("safe\n", encoding="utf-8")
    excluded = repo / "custom-results"
    excluded.mkdir()
    (excluded / "secret.json").write_text('{"ground_truth": "private"}', encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "canary.txt").write_text("host canary", encoding="utf-8")
    _symlink_or_skip(excluded / "latest", outside, target_is_directory=True)
    _symlink_or_skip(excluded / "dangling", tmp_path / "missing")

    if tracked:
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        subprocess.run(["git", "-C", str(repo), "add", "-f", "--", "."], check=True)

    if hasattr(os, "mkfifo"):
        os.mkfifo(excluded / "blocked.fifo")
    unreadable = excluded / "unreadable"
    unreadable.mkdir()
    (unreadable / "private.txt").write_text("private", encoding="utf-8")
    unreadable.chmod(0)

    original_is_file = Path.is_file

    def guarded_is_file(path: Path) -> bool:
        try:
            path.absolute().relative_to(excluded.absolute())
        except ValueError:
            return original_is_file(path)
        raise AssertionError(f"excluded output was inspected: {path}")

    monkeypatch.setattr(Path, "is_file", guarded_is_file)
    env_dir = tmp_path / "task" / "environment"
    env_dir.mkdir(parents=True)
    try:
        metadata = _stage_repo_context(
            env_dir,
            source_skill_path=skill,
            mode="full",
            excluded_roots=(excluded,),
        )
    finally:
        unreadable.chmod(0o700)

    assert (env_dir / "repo" / "README.md").read_text(encoding="utf-8") == "safe\n"
    assert not (env_dir / "repo" / excluded.relative_to(repo)).exists()
    assert all(not Path(item["source"]).is_relative_to(excluded) for item in metadata["files"])


def test_copy_repo_still_rejects_nonexcluded_file_symlink(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    skill = repo / "skill"
    skill.mkdir(parents=True)
    (skill / "SKILL.md").write_text("# Test skill\n", encoding="utf-8")
    source = repo / "source.txt"
    source.write_text("canary", encoding="utf-8")
    _symlink_or_skip(repo / "linked-source.txt", source)
    env_dir = tmp_path / "task" / "environment"
    env_dir.mkdir(parents=True)

    with pytest.raises(ValueError, match=r"symlink|reparse"):
        _stage_repo_context(env_dir, source_skill_path=skill, mode="full")


def _symlink_or_skip(link: Path, target: Path, *, target_is_directory: bool = False) -> None:
    try:
        link.symlink_to(target, target_is_directory=target_is_directory)
    except OSError as exc:  # pragma: no cover - host policy, primarily native Windows
        pytest.skip(f"symlinks unavailable on this host: {exc}")


@pytest.mark.parametrize("unsafe_id", _UNSAFE_EVAL_IDS)
def test_generated_tasks_reject_unsafe_ids_before_removing_existing_tasks(
    tmp_path: Path,
    unsafe_id: str,
) -> None:
    skill = _write_generated_skill(tmp_path, ["existing", unsafe_id])
    output_dir = tmp_path / "run" / "tasks"
    marker = output_dir / "existing" / "keep.txt"
    marker.parent.mkdir(parents=True)
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="case id"):
        generate_harbor_tasks(skill, output_dir)

    assert marker.read_text(encoding="utf-8") == "keep"


def test_generated_tasks_reject_portable_normalization_collisions_before_removal(tmp_path: Path) -> None:
    skill = _write_generated_skill(tmp_path, ["Case-001", "case-001"])
    output_dir = tmp_path / "run" / "tasks"
    marker = output_dir / "Case-001" / "keep.txt"
    marker.parent.mkdir(parents=True)
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(ValueError, match="colliding case id"):
        generate_harbor_tasks(skill, output_dir)

    assert marker.read_text(encoding="utf-8") == "keep"


def test_generated_tasks_reject_resolved_path_escape_before_removal(tmp_path: Path) -> None:
    skill = _write_generated_skill(tmp_path, ["existing", "case-001"])
    output_dir = tmp_path / "run" / "tasks"
    marker = output_dir / "existing" / "keep.txt"
    marker.parent.mkdir(parents=True)
    marker.write_text("keep", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    escaped_task = output_dir / "case-001"
    _symlink_or_skip(escaped_task, outside, target_is_directory=True)

    with pytest.raises(ValueError, match=r"symlink|reparse|outside"):
        generate_harbor_tasks(skill, output_dir)

    assert marker.read_text(encoding="utf-8") == "keep"
    assert not any(outside.iterdir())


def test_generated_tasks_accept_distinct_portable_ids(tmp_path: Path) -> None:
    skill = _write_generated_skill(tmp_path, ["case-001", "Case_002"])
    output_dir = tmp_path / "tasks"

    task_paths = generate_harbor_tasks(skill, output_dir)

    assert [path.name for path in task_paths] == ["case-001", "Case_002"]


def test_generated_task_rebases_custom_dockerfile_content(tmp_path: Path) -> None:
    skill = _write_generated_skill(tmp_path, ["case-001"])
    custom_environment = skill / "evals" / "environment"
    custom_environment.mkdir()
    (custom_environment / "Dockerfile").write_text(
        "FROM python:3.11-slim\nRUN echo generated-custom-layer\n",
        encoding="utf-8",
    )

    task = generate_harbor_tasks(
        skill,
        tmp_path / "tasks",
        base_image="registry.example/eval-base:verified",
    )[0]

    dockerfile = (task / "environment" / "Dockerfile").read_text(encoding="utf-8")
    assert dockerfile.startswith("FROM registry.example/eval-base:verified\n")
    assert dockerfile.index("RUN echo generated-custom-layer\n") < dockerfile.index(
        "# SkillEvaluator: original base was FROM python:3.11-slim\n"
    )


def test_native_task_rebases_custom_dockerfile_content(tmp_path: Path) -> None:
    skill = _write_native_skill(tmp_path)
    native_task = skill / "evals" / "harbor" / "case-001"
    environment = native_task / "environment"
    environment.mkdir()
    (environment / "Dockerfile").write_text(
        "FROM ubuntu:24.04\nRUN echo native-custom-layer\n",
        encoding="utf-8",
    )
    tests_dir = native_task / "tests"
    tests_dir.mkdir()
    (tests_dir / "test.sh").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")

    task = stage_native_harbor_tasks(
        skill,
        tmp_path / "native-tasks",
        grading_mode="custom_only",
        base_image="registry.example/eval-base:verified",
    )[0]

    dockerfile = (task / "environment" / "Dockerfile").read_text(encoding="utf-8")
    assert dockerfile.startswith("FROM registry.example/eval-base:verified\n")
    assert dockerfile.index("RUN echo native-custom-layer\n") < dockerfile.index(
        "# SkillEvaluator: original base was FROM ubuntu:24.04\n"
    )


def test_rebase_custom_dockerfile_content_without_from_fails_closed() -> None:
    content = "# comment only\nRUN echo unchanged\n"

    with pytest.raises(ValueError, match="exactly one FROM instruction"):
        _rebase_custom_dockerfile_content(
            content,
            "registry.example/eval-base:verified",
            agent_config_lines=[],
            include_input=False,
        )


@pytest.mark.parametrize("candidate", _GRADER_CANDIDATES, ids=str)
def test_custom_grader_candidates_reject_external_symlinks(tmp_path: Path, candidate: Path) -> None:
    skill = tmp_path / "skill"
    grader = skill / "evals" / candidate
    grader.parent.mkdir(parents=True)
    outside = tmp_path / f"outside-{candidate.name}"
    outside.write_text("host secret", encoding="utf-8")
    _symlink_or_skip(grader, outside)
    task_dir = tmp_path / "task"

    with pytest.raises(ValueError, match=r"non-symlinked regular file contained under evals/|symlink|reparse"):
        _copy_custom_grader(task_dir, skill, "custom_only")

    assert not (task_dir / "tests" / candidate.name).exists()


def test_custom_grader_rejects_parent_directory_symlink_escape(tmp_path: Path) -> None:
    skill = tmp_path / "skill"
    evals_dir = skill / "evals"
    evals_dir.mkdir(parents=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "grader.py").write_text("host secret", encoding="utf-8")
    _symlink_or_skip(evals_dir / "tests", outside, target_is_directory=True)
    task_dir = tmp_path / "task"

    with pytest.raises(ValueError, match="non-symlinked regular file contained under evals/"):
        _copy_custom_grader(task_dir, skill, "custom_only")

    assert not (task_dir / "tests" / "grader.py").exists()


def test_custom_grader_rejects_contained_parent_directory_symlink(tmp_path: Path) -> None:
    skill = tmp_path / "skill"
    evals_dir = skill / "evals"
    real_tests = evals_dir / "real-tests"
    real_tests.mkdir(parents=True)
    (real_tests / "grader.py").write_text("safe but linked", encoding="utf-8")
    _symlink_or_skip(evals_dir / "tests", real_tests, target_is_directory=True)
    task_dir = tmp_path / "task"

    with pytest.raises(ValueError, match="non-symlinked regular file contained under evals/"):
        _copy_custom_grader(task_dir, skill, "custom_only")

    assert not (task_dir / "tests" / "grader.py").exists()


def test_custom_grader_rejects_non_regular_candidate(tmp_path: Path) -> None:
    skill = tmp_path / "skill"
    grader = skill / "evals" / "grader.py"
    grader.mkdir(parents=True)

    with pytest.raises(ValueError, match="non-symlinked regular file contained under evals/"):
        _copy_custom_grader(tmp_path / "task", skill, "custom_only")


def test_custom_grader_rejects_hardlinked_candidate(tmp_path: Path) -> None:
    skill = tmp_path / "skill"
    grader = skill / "evals" / "grader.py"
    grader.parent.mkdir(parents=True)
    grader.write_text("safe grader", encoding="utf-8")
    outside_alias = tmp_path / "grader-alias.py"
    try:
        outside_alias.hardlink_to(grader)
    except OSError as exc:  # pragma: no cover - filesystem policy
        pytest.skip(f"hardlinks unavailable on this host: {exc}")
    task_dir = tmp_path / "task"

    with pytest.raises(ValueError, match=r"hard.?link|multiple links"):
        _copy_custom_grader(task_dir, skill, "custom_only")

    assert grader.read_text(encoding="utf-8") == "safe grader"
    assert not (task_dir / "tests" / "grader.py").exists()


def test_custom_grader_rejects_source_replacement_after_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secure_copy = importlib.import_module("skillevaluator.tier3.harbor.secure_copy")
    skill = tmp_path / "skill"
    grader = skill / "evals" / "grader.py"
    grader.parent.mkdir(parents=True)
    grader.write_text("validated grader", encoding="utf-8")
    task_dir = tmp_path / "task"
    original = secure_copy._build_file_manifest

    def validate_then_replace(source: Path, allowed_root: Path):
        manifest = original(source, allowed_root)
        if Path(source) == grader.resolve():
            grader.unlink()
            grader.write_text("replacement grader", encoding="utf-8")
        return manifest

    monkeypatch.setattr(secure_copy, "_build_file_manifest", validate_then_replace)

    with pytest.raises(ValueError, match="source changed after validation"):
        _copy_custom_grader(task_dir, skill, "custom_only")

    assert grader.read_text(encoding="utf-8") == "replacement grader"
    assert not (task_dir / "tests" / "grader.py").exists()


@pytest.mark.parametrize("candidate", _GRADER_CANDIDATES, ids=str)
def test_custom_grader_candidates_copy_regular_files(tmp_path: Path, candidate: Path) -> None:
    skill = tmp_path / "skill"
    grader = skill / "evals" / candidate
    grader.parent.mkdir(parents=True)
    grader.write_text("safe grader", encoding="utf-8")
    task_dir = tmp_path / "task"

    assert _copy_custom_grader(task_dir, skill, "custom_only") is True

    copied = task_dir / "tests" / candidate.name
    assert copied.read_text(encoding="utf-8") == "safe grader"
    # Windows does not expose POSIX executable bits; the runner invokes graders via bash.
    if candidate.suffix == ".sh" and os.name != "nt":
        assert copied.stat().st_mode & 0o111


@pytest.mark.parametrize("task_source", ("generated", "native"))
def test_task_modes_reject_custom_grader_symlink_before_copy(tmp_path: Path, task_source: str) -> None:
    skill = (
        _write_generated_skill(tmp_path, ["case-001"]) if task_source == "generated" else _write_native_skill(tmp_path)
    )
    outside = tmp_path / "outside-grader.py"
    outside.write_text("host secret", encoding="utf-8")
    _symlink_or_skip(skill / "evals" / "grader.py", outside)
    output_dir = tmp_path / "tasks"

    with pytest.raises(ValueError, match=r"non-symlinked regular file contained under evals/|symlink|reparse"):
        if task_source == "generated":
            generate_harbor_tasks(skill, output_dir, with_skill=False, grading_mode="custom_only")
        else:
            stage_native_harbor_tasks(skill, output_dir, with_skill=False, grading_mode="custom_only")

    assert not (output_dir / "case-001" / "tests" / "grader.py").exists()


@pytest.mark.parametrize("grading_mode", ("default", "default_plus_custom"))
@pytest.mark.parametrize("task_source", ("generated", "native"))
def test_standard_grading_stages_trusted_evaluated_skill_after_authored_entry(
    tmp_path: Path,
    grading_mode: str,
    task_source: str,
) -> None:
    skill = tmp_path / "trusted-target"
    evals_dir = skill / "evals"
    evals_dir.mkdir(parents=True)
    (skill / "SKILL.md").write_text("# Trusted target\n", encoding="utf-8")
    entry = {
        "id": "case-001",
        "question": "Do not use the target skill",
        "expected_skill": None,
        "evaluated_skill": "authored-spoof",
    }
    (evals_dir / "evals.json").write_text(json.dumps([entry]), encoding="utf-8")

    output_dir = tmp_path / "tasks"
    if task_source == "generated":
        generate_harbor_tasks(skill, output_dir, with_skill=False, grading_mode=grading_mode)
    else:
        native_task = evals_dir / "harbor" / "case-001"
        native_task.mkdir(parents=True)
        (native_task / "task.toml").write_text(
            'schema_version = "1.3"\n\n[task]\nname = "nvidia/case-001"\n\n[environment]\n',
            encoding="utf-8",
        )
        stage_native_harbor_tasks(skill, output_dir, with_skill=False, grading_mode=grading_mode)

    staged = json.loads((output_dir / "case-001" / "tests" / "entry.json").read_text(encoding="utf-8"))

    assert staged["evaluated_skill"] == skill.name


@pytest.mark.parametrize("task_source", ("generated", "native"))
@pytest.mark.parametrize("authored_value", (None, "authored-value"))
def test_custom_only_staging_does_not_inject_or_overwrite_evaluated_skill(
    tmp_path: Path,
    task_source: str,
    authored_value: str | None,
) -> None:
    skill = tmp_path / "trusted-target"
    evals_dir = skill / "evals"
    evals_dir.mkdir(parents=True)
    (skill / "SKILL.md").write_text("# Trusted target\n", encoding="utf-8")
    entry = {"id": "case-001", "question": "Run custom grading"}
    if authored_value is not None:
        entry["evaluated_skill"] = authored_value
    (evals_dir / "evals.json").write_text(json.dumps([entry]), encoding="utf-8")
    (evals_dir / "grader.py").write_text("def grade():\n    return 1\n", encoding="utf-8")

    output_dir = tmp_path / "tasks"
    if task_source == "generated":
        generate_harbor_tasks(skill, output_dir, with_skill=False, grading_mode="custom_only")
    else:
        native_task = evals_dir / "harbor" / "case-001"
        native_task.mkdir(parents=True)
        (native_task / "task.toml").write_text(
            'schema_version = "1.3"\n\n[task]\nname = "nvidia/case-001"\n\n[environment]\n',
            encoding="utf-8",
        )
        stage_native_harbor_tasks(skill, output_dir, with_skill=False, grading_mode="custom_only")
    staged = json.loads((output_dir / "case-001" / "tests" / "entry.json").read_text(encoding="utf-8"))

    if authored_value is None:
        assert "evaluated_skill" not in staged
    else:
        assert staged["evaluated_skill"] == authored_value
