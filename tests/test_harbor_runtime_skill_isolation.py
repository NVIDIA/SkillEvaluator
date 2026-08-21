# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Regression coverage for agent-visible Tier 3 skill projection."""

from __future__ import annotations

import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import tomllib
from pathlib import Path
from types import SimpleNamespace

import pytest

import skillevaluator.tier3.harbor.adapter as adapter_module
from skillevaluator.tier3.harbor.adapter import (
    _collect_all_skill_deps,
    _dockerfile_resolved_build_context_sources,
    _ensure_empty_custom_docker_input_compatibility,
    _path_is_excluded,
    generate_harbor_tasks,
    prebuild_task_environments,
    private_evaluator_skill_snapshot,
    stage_native_harbor_tasks,
    validate_results_root_location,
)
from skillevaluator.tier3.output_provenance import mark_generated_output_root
from skillevaluator.tier3.results_location import publish_latest_results


def _write_runtime_skill(path: Path, package: str) -> Path:
    path.mkdir(parents=True)
    (path / "SKILL.md").write_text(f"# {path.name}\n", encoding="utf-8")
    scripts = path / "scripts"
    scripts.mkdir()
    (scripts / "run.py").write_text("print('runtime')\n", encoding="utf-8")
    (scripts / "requirements.txt").write_text(f"{package}-runtime==1\n", encoding="utf-8")
    (scripts / "apt-packages.txt").write_text(f"{package}-runtime-tool\n", encoding="utf-8")

    nested_evals = path / "references" / "evals"
    nested_evals.mkdir(parents=True)
    (nested_evals / "keep.md").write_text("legitimate runtime reference\n", encoding="utf-8")
    (nested_evals / "requirements.txt").write_text(f"{package}-nested-runtime==1\n", encoding="utf-8")
    (nested_evals / "apt-packages.txt").write_text(f"{package}-nested-runtime-tool\n", encoding="utf-8")

    evals = path / "evals"
    evals.mkdir()
    (evals / "evals.json").write_text("[]\n", encoding="utf-8")
    (evals / "requirements.txt").write_text(f"{package}-oracle==9\n", encoding="utf-8")
    (evals / "apt-packages.txt").write_text(f"{package}-oracle-tool\n", encoding="utf-8")
    return path


def _write_nested_runtime_skill(wrapper: Path, package: str) -> Path:
    nested = wrapper / "references" / "nested-skill"
    nested.mkdir(parents=True)
    (nested / "SKILL.md").write_text(f"# Nested {package}\n", encoding="utf-8")
    scripts = nested / "scripts"
    scripts.mkdir()
    (scripts / "requirements.txt").write_text(f"{package}-nested-skill-runtime==1\n", encoding="utf-8")
    (scripts / "apt-packages.txt").write_text(f"{package}-nested-skill-runtime-tool\n", encoding="utf-8")
    evals = nested / "evals"
    evals.mkdir()
    (evals / "evals.json").write_text('[{"ground_truth": "NESTED-ORACLE"}]\n', encoding="utf-8")
    (evals / "requirements.txt").write_text(f"{package}-nested-skill-oracle==9\n", encoding="utf-8")
    (evals / "apt-packages.txt").write_text(f"{package}-nested-skill-oracle-tool\n", encoding="utf-8")
    preserved = nested / "references" / "evals"
    preserved.mkdir(parents=True)
    (preserved / "keep.md").write_text("nested runtime reference\n", encoding="utf-8")
    return nested


def _write_minimal_native_task(target: Path) -> None:
    native_task = target / "evals" / "harbor" / "case-001"
    native_task.mkdir(parents=True)
    (native_task / "instruction.md").write_text("Run the native case.\n", encoding="utf-8")
    (native_task / "task.toml").write_text(
        'schema_version = "1.3"\n\n[task]\nname = "nvidia/case-001"\n\n'
        '[metadata]\nentry_id = "case-001"\n\n[environment]\n',
        encoding="utf-8",
    )


def _write_projection_fixture(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    repo = tmp_path / "repo"
    target = _write_runtime_skill(repo / "target-skill", "target")
    references_dir = repo / "reference-skills"
    _write_runtime_skill(references_dir / "reference-skill", "reference")
    workspace = _write_runtime_skill(repo / "workspace-skill", "workspace")

    docs_evals = repo / "docs" / "evals"
    docs_evals.mkdir(parents=True)
    (docs_evals / "keep.txt").write_text("not a skill dataset\n", encoding="utf-8")

    evals = target / "evals"
    files = evals / "files"
    files.mkdir()
    (files / "selected.txt").write_text("SELECTED-FIXTURE\n", encoding="utf-8")
    (files / "hidden.txt").write_text("UNDECLARED-ORACLE\n", encoding="utf-8")
    (evals / "evals.json").write_text(
        json.dumps(
            [
                {
                    "id": "case-001",
                    "question": "Use only the selected fixture.",
                    "files": ["evals/files/selected.txt"],
                    "ground_truth": "GROUND-TRUTH-SECRET",
                }
            ]
        ),
        encoding="utf-8",
    )
    (evals / "grader.py").write_text("def grade(*args, **kwargs):\n    return 1\n", encoding="utf-8")
    sidecar = evals / "environment" / "sidecar"
    sidecar.mkdir(parents=True)
    (sidecar / "seed.txt").write_text("custom environment asset\n", encoding="utf-8")

    return repo, target, references_dir, workspace


def _assert_runtime_projection(task: Path, target: Path, workspace: Path) -> None:
    env = task / "environment"
    staged_skills = env / "skills"
    for name in (target.name, "reference-skill", workspace.name):
        staged = staged_skills / name
        assert (staged / "SKILL.md").is_file()
        assert (staged / "scripts" / "run.py").is_file()
        assert not (staged / "evals").exists()
        assert (staged / "references" / "evals" / "keep.md").is_file()

    staged_repo = env / "repo"
    assert not (staged_repo / target.name / "evals").exists()
    assert not (staged_repo / "reference-skills" / "reference-skill" / "evals").exists()
    assert not (staged_repo / workspace.name / "evals").exists()
    assert (staged_repo / target.name / "references" / "evals" / "keep.md").is_file()
    assert (staged_repo / "docs" / "evals" / "keep.txt").is_file()

    readable_environment = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore") for path in env.rglob("*") if path.is_file()
    )
    assert "UNDECLARED-ORACLE" not in readable_environment
    assert "GROUND-TRUTH-SECRET" not in readable_environment
    assert (target / "evals" / "evals.json").is_file(), "source evaluation assets must remain intact"


def test_generated_tasks_project_runtime_skills_without_root_evals(tmp_path: Path) -> None:
    _, target, references_dir, workspace = _write_projection_fixture(tmp_path)

    task = generate_harbor_tasks(
        target,
        tmp_path / "generated",
        with_skill=True,
        reference_skills_dir=references_dir,
        workspace_skill_paths=[workspace],
        copy_repo=True,
    )[0]

    _assert_runtime_projection(task, target, workspace)
    assert (task / "environment" / "input" / "selected.txt").read_text(encoding="utf-8") == "SELECTED-FIXTURE\n"
    assert not (task / "environment" / "input" / "hidden.txt").exists()
    assert (task / "tests" / "grader.py").is_file()
    assert "GROUND-TRUTH-SECRET" in (task / "tests" / "entry.json").read_text(encoding="utf-8")
    assert (task / "environment" / "sidecar" / "seed.txt").is_file()
    dockerfile = (task / "environment" / "Dockerfile").read_text(encoding="utf-8")
    assert "target-runtime==1" in dockerfile
    assert "target-oracle==9" not in dockerfile


def test_native_tasks_project_runtime_skills_without_root_evals(tmp_path: Path) -> None:
    _, target, references_dir, workspace = _write_projection_fixture(tmp_path)
    native_task = target / "evals" / "harbor" / "case-001"
    native_environment = native_task / "environment"
    native_environment.mkdir(parents=True)
    (native_environment / "native.txt").write_text("native environment asset\n", encoding="utf-8")
    prebundled = native_environment / "skills" / "prebundled-skill"
    (prebundled / "evals").mkdir(parents=True)
    (prebundled / "skill.md").write_text("# Prebundled\n", encoding="utf-8")
    (prebundled / "evals" / "secret.txt").write_text("PREBUNDLED-ORACLE\n", encoding="utf-8")
    (prebundled / "references" / "evals").mkdir(parents=True)
    (prebundled / "references" / "evals" / "keep.txt").write_text("keep\n", encoding="utf-8")
    nested_prebundled = _write_nested_runtime_skill(prebundled, "prebundled")
    (native_task / "instruction.md").write_text("Run the native case.\n", encoding="utf-8")
    (native_task / "task.toml").write_text(
        'schema_version = "1.3"\n\n[task]\nname = "nvidia/case-001"\n\n'
        '[metadata]\nentry_id = "case-001"\n\n[environment]\n',
        encoding="utf-8",
    )

    task = stage_native_harbor_tasks(
        target,
        tmp_path / "native",
        with_skill=True,
        reference_skills_dir=references_dir,
        workspace_skill_paths=[workspace],
        copy_repo=True,
        verifier_env={
            "LLM_JUDGE_MODEL": "${LLM_JUDGE_MODEL}",
            "SKILL_EVAL_JUDGE_MODEL": "${SKILL_EVAL_JUDGE_MODEL}",
        },
    )[0]

    _assert_runtime_projection(task, target, workspace)
    assert (task / "environment" / "input" / "selected.txt").read_text(encoding="utf-8") == "SELECTED-FIXTURE\n"
    assert not (task / "environment" / "input" / "hidden.txt").exists()
    assert (task / "instruction.md").read_text(encoding="utf-8") == "Run the native case.\n"
    assert (task / "environment" / "native.txt").is_file()
    assert not (task / "environment" / "skills" / "prebundled-skill" / "evals").exists()
    assert (task / "environment" / "skills" / "prebundled-skill" / "references" / "evals" / "keep.txt").is_file()
    staged_nested_prebundled = (
        task / "environment" / "skills" / "prebundled-skill" / nested_prebundled.relative_to(prebundled)
    )
    assert (staged_nested_prebundled / "SKILL.md").is_file()
    assert not (staged_nested_prebundled / "evals").exists()
    assert (staged_nested_prebundled / "references" / "evals" / "keep.md").is_file()
    assert (task / "tests" / "entry.json").is_file()
    assert (task / "tests" / "grader.py").is_file()
    task_config = tomllib.loads((task / "task.toml").read_text(encoding="utf-8"))
    assert "verifier" not in task_config
    assert "LLM_JUDGE_MODEL" not in task_config["environment"].get("env", {})
    assert "SKILL_EVAL_JUDGE_MODEL" not in task_config["environment"].get("env", {})


@pytest.mark.parametrize("name", ["LLM_JUDGE_MODEL", "SKILL_EVAL_JUDGE_MODEL"])
@pytest.mark.parametrize("grading_mode", ["default", "custom_only"])
def test_native_tasks_preserve_authored_judge_model_in_verifier_env(
    tmp_path: Path,
    name: str,
    grading_mode: str,
) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    _write_minimal_native_task(target)
    task_toml = target / "evals" / "harbor" / "case-001" / "task.toml"
    content = task_toml.read_text(encoding="utf-8").replace(
        "[environment]",
        f'[verifier]\ntimeout_sec = 180.0\n\n[verifier.env]\n{name} = "skill-model"\n\n[environment]',
    )
    task_toml.write_text(content, encoding="utf-8")

    task = stage_native_harbor_tasks(
        target,
        tmp_path / f"native-verifier-{grading_mode}-{name.lower()}",
        grading_mode=grading_mode,
        verifier_env={
            "SKILL_EVAL_LLM_PROVIDER": "${SKILL_EVAL_LLM_PROVIDER}",
            name: f"${{{name}}}",
        },
    )[0]

    task_config = tomllib.loads((task / "task.toml").read_text(encoding="utf-8"))
    expected = {name: "skill-model"}
    if grading_mode == "default":
        expected["SKILL_EVAL_LLM_PROVIDER"] = "${SKILL_EVAL_LLM_PROVIDER}"
    assert task_config["verifier"]["env"] == expected


@pytest.mark.parametrize(
    "assignment",
    [
        'LLM_JUDGE_MODEL = "skill-model"',
        'SKILL_EVAL_JUDGE_MODEL = "skill-model"',
        'JUDGE_HINT = "$LLM_JUDGE_MODEL"',
        'JUDGE_HINT = "${SKILL_EVAL_JUDGE_MODEL}"',
        'JUDGE_HINT = "${LLM_JUDGE_MODEL:-skill-fallback}"',
        'JUDGE_HINT = "${SKILL_EVAL_JUDGE_MODEL:-skill-fallback}"',
        'JUDGE_HINT = "%LLM_JUDGE_MODEL%"',
    ],
)
def test_native_tasks_reject_judge_model_controls_in_agent_environment(tmp_path: Path, assignment: str) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    _write_minimal_native_task(target)
    task_toml = target / "evals" / "harbor" / "case-001" / "task.toml"
    task_toml.write_text(
        task_toml.read_text(encoding="utf-8") + f"\n[environment.env]\n{assignment}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"environment.env.*judge|judge.*environment.env"):
        stage_native_harbor_tasks(target, tmp_path / "native-agent-judge-control")


def test_native_task_reuses_existing_exact_provider_placeholder_when_injecting(tmp_path: Path) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    _write_minimal_native_task(target)
    task_toml = target / "evals" / "harbor" / "case-001" / "task.toml"
    content = task_toml.read_text(encoding="utf-8").replace(
        "[environment]",
        "[verifier]\ntimeout_sec = 180.0\n\n[verifier.env]\n"
        'SKILL_EVAL_LLM_PROVIDER = "${SKILL_EVAL_LLM_PROVIDER}"\n\n[environment]',
    )
    task_toml.write_text(content, encoding="utf-8")

    task = stage_native_harbor_tasks(
        target,
        tmp_path / "native-existing-provider-placeholder",
        verifier_env={
            "SKILL_EVAL_LLM_MODEL": "${SKILL_EVAL_LLM_MODEL}",
            "SKILL_EVAL_LLM_PROVIDER": "${SKILL_EVAL_LLM_PROVIDER}",
        },
    )[0]

    task_config = tomllib.loads((task / "task.toml").read_text(encoding="utf-8"))
    assert task_config["verifier"]["env"] == {
        "SKILL_EVAL_LLM_MODEL": "${SKILL_EVAL_LLM_MODEL}",
        "SKILL_EVAL_LLM_PROVIDER": "${SKILL_EVAL_LLM_PROVIDER}",
    }


@pytest.mark.parametrize("name", ["LLM_JUDGE_MODEL", "SKILL_EVAL_JUDGE_MODEL"])
@pytest.mark.parametrize(
    ("authored_config", "expected_path"),
    [
        (
            '[verifier]\ntimeout_sec = 180.0\n\n[verifier.environment]\nworkdir = "/workspace"\n\n'
            '[verifier.environment.env]\n{name} = "skill-model"\n',
            ("verifier", "environment", "env"),
        ),
        (
            '[[steps]]\nname = "step-one"\n\n[steps.verifier.env]\n{name} = "skill-model"\n',
            ("steps", 0, "verifier", "env"),
        ),
        (
            '[[steps]]\nname = "step-one"\n\n[steps.verifier.environment]\nworkdir = "/workspace"\n\n'
            '[steps.verifier.environment.env]\n{name} = "skill-model"\n',
            ("steps", 0, "verifier", "environment", "env"),
        ),
    ],
)
def test_native_tasks_preserve_judge_model_controls_in_nested_verifier_envs(
    tmp_path: Path,
    name: str,
    authored_config: str,
    expected_path: tuple[str | int, ...],
) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    _write_minimal_native_task(target)
    task_toml = target / "evals" / "harbor" / "case-001" / "task.toml"
    task_toml.write_text(
        task_toml.read_text(encoding="utf-8") + "\n" + authored_config.format(name=name),
        encoding="utf-8",
    )

    task = stage_native_harbor_tasks(
        target,
        tmp_path / "native-nested-verifier-control",
        verifier_env={
            "SKILL_EVAL_LLM_PROVIDER": "${SKILL_EVAL_LLM_PROVIDER}",
            "SKILL_EVAL_JUDGE_MODEL": "${SKILL_EVAL_JUDGE_MODEL}",
        },
    )[0]

    value: object = tomllib.loads((task / "task.toml").read_text(encoding="utf-8"))
    for segment in expected_path:
        value = value[segment]  # type: ignore[index]
    assert value == {name: "skill-model"}
    task_config = tomllib.loads((task / "task.toml").read_text(encoding="utf-8"))
    assert task_config["verifier"]["env"] == {
        "SKILL_EVAL_LLM_PROVIDER": "${SKILL_EVAL_LLM_PROVIDER}",
    }


def test_native_task_staging_uses_one_private_evals_snapshot(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    _write_minimal_native_task(target)
    evals = target / "evals"
    moved = tmp_path / "evals-A"
    real_load = adapter_module._load_entries_by_id
    swapped = False

    def swap_original_evals_after_metadata_read(path: Path) -> dict[str, dict[str, object]]:
        nonlocal swapped
        entries = real_load(path)
        if not swapped:
            swapped = True
            evals.rename(moved)
            shutil.copytree(moved, evals)
            (evals / "evals.json").write_text(
                json.dumps(
                    [
                        {
                            "id": "case-001",
                            "question": "metadata-B",
                            "files": ["evals/files/selected.txt"],
                        }
                    ]
                ),
                encoding="utf-8",
            )
            (evals / "files" / "selected.txt").write_text("fixture-B\n", encoding="utf-8")
            (evals / "harbor" / "case-001" / "instruction.md").write_text(
                "native-B\n",
                encoding="utf-8",
            )
        return entries

    monkeypatch.setattr(adapter_module, "_load_entries_by_id", swap_original_evals_after_metadata_read)

    task = stage_native_harbor_tasks(target, tmp_path / "native-snapshot", with_skill=False)[0]
    entry = json.loads((task / "tests" / "entry.json").read_text(encoding="utf-8"))

    assert swapped
    assert (task / "instruction.md").read_text(encoding="utf-8") == "Run the native case.\n"
    assert entry["question"] == "Use only the selected fixture."
    assert (task / "environment" / "input" / "selected.txt").read_text(encoding="utf-8") == "SELECTED-FIXTURE\n"
    assert (task / "tests" / "grader.py").is_file()


def test_base_image_dependency_collection_ignores_only_root_evals(tmp_path: Path) -> None:
    _, target, references_dir, workspace = _write_projection_fixture(tmp_path)

    pip_reqs, apt_packages = _collect_all_skill_deps(target, references_dir, [workspace])

    assert pip_reqs == [
        "reference-nested-runtime==1",
        "reference-runtime==1",
        "target-nested-runtime==1",
        "target-runtime==1",
        "workspace-nested-runtime==1",
        "workspace-runtime==1",
    ]
    assert apt_packages == [
        "reference-nested-runtime-tool",
        "reference-runtime-tool",
        "target-nested-runtime-tool",
        "target-runtime-tool",
        "workspace-nested-runtime-tool",
        "workspace-runtime-tool",
    ]


def test_runtime_projection_excludes_nested_skill_evals_and_dependencies(tmp_path: Path) -> None:
    _, target, references_dir, workspace = _write_projection_fixture(tmp_path)
    wrappers = [target, references_dir / "reference-skill", workspace]
    nested_skills = [
        _write_nested_runtime_skill(wrapper, package)
        for wrapper, package in zip(wrappers, ("target", "reference", "workspace"), strict=True)
    ]

    task = generate_harbor_tasks(
        target,
        tmp_path / "nested-skill-evals",
        reference_skills_dir=references_dir,
        workspace_skill_paths=[workspace],
    )[0]

    staged_wrappers = [
        task / "environment" / "skills" / target.name,
        task / "environment" / "skills" / "reference-skill",
        task / "environment" / "skills" / workspace.name,
    ]
    for nested, wrapper, staged_wrapper in zip(nested_skills, wrappers, staged_wrappers, strict=True):
        staged_nested = staged_wrapper / nested.relative_to(wrapper)
        assert (staged_nested / "SKILL.md").is_file()
        assert not (staged_nested / "evals").exists()
        assert (staged_nested / "references" / "evals" / "keep.md").is_file()

    pip_reqs, apt_packages = _collect_all_skill_deps(target, references_dir, [workspace])
    for package in ("target", "reference", "workspace"):
        assert f"{package}-nested-skill-runtime==1" in pip_reqs
        assert f"{package}-nested-skill-runtime-tool" in apt_packages
        assert f"{package}-nested-skill-oracle==9" not in pip_reqs
        assert f"{package}-nested-skill-oracle-tool" not in apt_packages


def test_base_image_dependency_collection_does_not_follow_alias_into_root_evals(tmp_path: Path) -> None:
    _, target, references_dir, workspace = _write_projection_fixture(tmp_path)
    alias_dir = target / "scripts" / "eval-alias"
    alias_dir.mkdir()
    try:
        (alias_dir / "requirements.txt").symlink_to("../../evals/requirements.txt")
        (alias_dir / "apt-packages.txt").symlink_to("../../evals/apt-packages.txt")
    except OSError as exc:  # pragma: no cover - host policy, primarily native Windows
        pytest.skip(f"symlinks unavailable on this host: {exc}")

    pip_reqs, apt_packages = _collect_all_skill_deps(target, references_dir, [workspace])

    assert "target-oracle==9" not in pip_reqs
    assert "target-oracle-tool" not in apt_packages


def test_base_image_dependency_collection_matches_ignored_runtime_projection(tmp_path: Path) -> None:
    _, target, references_dir, workspace = _write_projection_fixture(tmp_path)
    archived = target / "results"
    archived.mkdir()
    (archived / "requirements.txt").write_text("archived-oracle==9\n", encoding="utf-8")
    (archived / "apt-packages.txt").write_text("archived-oracle-tool\n", encoding="utf-8")

    pip_reqs, apt_packages = _collect_all_skill_deps(target, references_dir, [workspace])

    assert "archived-oracle==9" not in pip_reqs
    assert "archived-oracle-tool" not in apt_packages


@pytest.mark.parametrize("link_kind", ["symlink", "hardlink"])
def test_runtime_skill_projection_rejects_linked_host_file(tmp_path: Path, link_kind: str) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    outside = tmp_path / "host-secret.txt"
    outside.write_text("HOST-SECRET\n", encoding="utf-8")
    alias = target / "scripts" / "host-secret.txt"
    try:
        if link_kind == "symlink":
            alias.symlink_to(outside)
        else:
            alias.hardlink_to(outside)
    except OSError as exc:  # pragma: no cover - filesystem policy
        pytest.skip(f"{link_kind}s unavailable on this host: {exc}")

    with pytest.raises(ValueError, match=r"symlink|hard.?link"):
        generate_harbor_tasks(target, tmp_path / f"runtime-{link_kind}")


def test_copy_repo_does_not_follow_alias_into_skill_root_evals(tmp_path: Path) -> None:
    repo, target, references_dir, workspace = _write_projection_fixture(tmp_path)
    alias = repo / "oracle-alias.txt"
    try:
        alias.symlink_to(target / "evals" / "files" / "hidden.txt")
    except OSError as exc:  # pragma: no cover - host policy, primarily native Windows
        pytest.skip(f"symlinks unavailable on this host: {exc}")

    task = generate_harbor_tasks(
        target,
        tmp_path / "generated-alias",
        reference_skills_dir=references_dir,
        workspace_skill_paths=[workspace],
        copy_repo=True,
    )[0]

    assert not (task / "environment" / "repo" / alias.name).exists()


def test_copy_repo_ignores_lexical_skill_evals_alias_to_repo_file(tmp_path: Path) -> None:
    repo, target, references_dir, workspace = _write_projection_fixture(tmp_path)
    secret = repo / ".env"
    secret.write_text("TOKEN=repo-secret\n", encoding="utf-8")
    alias = target / "evals" / "repo-secret-alias"
    try:
        alias.symlink_to(secret)
    except OSError as exc:  # pragma: no cover - host policy, primarily native Windows
        pytest.skip(f"symlinks unavailable on this host: {exc}")

    task = generate_harbor_tasks(
        target,
        tmp_path / "generated-lexical-alias",
        reference_skills_dir=references_dir,
        workspace_skill_paths=[workspace],
        copy_repo=True,
    )[0]

    assert not (task / "environment" / "repo" / target.name / "evals" / alias.name).exists()


def test_copy_repo_rejects_hardlink_alias(tmp_path: Path) -> None:
    repo, target, references_dir, workspace = _write_projection_fixture(tmp_path)
    alias = repo / "oracle-hardlink.txt"
    try:
        alias.hardlink_to(target / "evals" / "files" / "hidden.txt")
    except OSError as exc:  # pragma: no cover - filesystem policy
        pytest.skip(f"hardlinks unavailable on this host: {exc}")

    with pytest.raises(ValueError, match=r"hard.?link"):
        generate_harbor_tasks(
            target,
            tmp_path / "generated-hardlink",
            reference_skills_dir=references_dir,
            workspace_skill_paths=[workspace],
            copy_repo=True,
        )


def test_linked_repo_ignores_lexical_skill_evals_alias(tmp_path: Path) -> None:
    repo, target, references_dir, workspace = _write_projection_fixture(tmp_path)
    secret = repo / "linked-secret.txt"
    secret.write_text("LINKED-REPO-SECRET\n", encoding="utf-8")
    alias = target / "evals" / "linked-secret-alias"
    try:
        alias.symlink_to(secret)
    except OSError as exc:  # pragma: no cover - host policy, primarily native Windows
        pytest.skip(f"symlinks unavailable on this host: {exc}")
    manifest = target / "SKILL.md"
    manifest.write_text(
        manifest.read_text(encoding="utf-8") + "\n[hidden](evals/linked-secret-alias)\n",
        encoding="utf-8",
    )

    task = generate_harbor_tasks(
        target,
        tmp_path / "linked-lexical-alias",
        reference_skills_dir=references_dir,
        workspace_skill_paths=[workspace],
        copy_repo=False,
    )[0]

    environment = task / "environment"
    assert "LINKED-REPO-SECRET" not in "\n".join(
        path.read_text(encoding="utf-8", errors="ignore") for path in environment.rglob("*") if path.is_file()
    )


def test_copy_repo_filters_lowercase_skill_manifest_evals(tmp_path: Path) -> None:
    _, target, references_dir, workspace = _write_projection_fixture(tmp_path)
    lowercase_skill = target.parent / "lowercase-skill"
    (lowercase_skill / "evals").mkdir(parents=True)
    (lowercase_skill / "skill.md").write_text("# Lowercase\n", encoding="utf-8")
    (lowercase_skill / "evals" / "secret.txt").write_text("LOWERCASE-ORACLE\n", encoding="utf-8")

    task = generate_harbor_tasks(
        target,
        tmp_path / "lowercase-filter",
        reference_skills_dir=references_dir,
        workspace_skill_paths=[workspace],
        copy_repo=True,
    )[0]

    assert not (task / "environment" / "repo" / lowercase_skill.name / "evals").exists()


def test_copy_repo_excludes_configured_in_repo_output_root(tmp_path: Path) -> None:
    repo, target, references_dir, workspace = _write_projection_fixture(tmp_path)
    output_dir = repo / "custom-results"
    leaked_entry = output_dir / "prior-task" / "tests" / "entry.json"
    leaked_entry.parent.mkdir(parents=True)
    leaked_entry.write_text('{"ground_truth": "PRIOR-RUN-SECRET"}\n', encoding="utf-8")

    task = generate_harbor_tasks(
        target,
        output_dir,
        reference_skills_dir=references_dir,
        workspace_skill_paths=[workspace],
        copy_repo=True,
    )[0]

    assert not (task / "environment" / "repo" / output_dir.name).exists()


@pytest.mark.parametrize("stager", ["generated", "native"])
def test_copy_repo_excludes_private_in_repo_staging_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stager: str,
) -> None:
    repo, target, references_dir, workspace = _write_projection_fixture(tmp_path)
    private_canary = "PRIVATE-STAGING-CANARY"
    entries = json.loads((target / "evals" / "evals.json").read_text(encoding="utf-8"))
    entries[0]["ground_truth"] = private_canary
    (target / "evals" / "evals.json").write_text(json.dumps(entries), encoding="utf-8")
    if stager == "native":
        _write_minimal_native_task(target)
    in_repo_temp_root = repo / "tmp"
    in_repo_temp_root.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", os.fspath(in_repo_temp_root))
    stager_fn = generate_harbor_tasks if stager == "generated" else stage_native_harbor_tasks

    task = stager_fn(
        target,
        tmp_path / f"{stager}-private-root",
        reference_skills_dir=references_dir,
        workspace_skill_paths=[workspace],
        copy_repo=True,
    )[0]

    environment = task / "environment"
    readable_environment = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore") for path in environment.rglob("*") if path.is_file()
    )
    assert private_canary not in readable_environment
    assert not (environment / "repo" / in_repo_temp_root.relative_to(repo)).exists()


def test_runtime_skill_projection_excludes_shared_in_skill_output_root(tmp_path: Path) -> None:
    _, target, references_dir, workspace = _write_projection_fixture(tmp_path)
    output_root = target / "Custom-Results"
    configured_output_root = target / "custom-results"
    first_output = output_root / "agent-a"
    generate_harbor_tasks(
        target,
        first_output,
        reference_skills_dir=references_dir,
        workspace_skill_paths=[workspace],
        repo_context_exclude_paths=(configured_output_root,),
    )

    second = generate_harbor_tasks(
        target,
        output_root / "agent-b",
        reference_skills_dir=references_dir,
        workspace_skill_paths=[workspace],
        repo_context_exclude_paths=(configured_output_root,),
    )[0]

    assert not (second / "environment" / "skills" / target.name / output_root.name).exists()
    pip_reqs, apt_packages = _collect_all_skill_deps(
        target,
        references_dir,
        [workspace],
        excluded_roots=(configured_output_root,),
    )
    assert all("custom-results" not in dep.casefold() for dep in [*pip_reqs, *apt_packages])


def test_copy_repo_excludes_case_only_output_alias(tmp_path: Path) -> None:
    _, target, references_dir, workspace = _write_projection_fixture(tmp_path)
    prior_output = target / "Custom-Results"
    prior_output.mkdir()
    (prior_output / "prior.json").write_text("PRIOR-RESULT-ORACLE\n", encoding="utf-8")

    task = generate_harbor_tasks(
        target,
        tmp_path / "casefold-copy-repo",
        reference_skills_dir=references_dir,
        workspace_skill_paths=[workspace],
        copy_repo=True,
        repo_context_exclude_paths=(target / "custom-results",),
    )[0]

    environment = task / "environment"
    assert not (environment / "skills" / target.name / "Custom-Results").exists()
    assert not (environment / "repo" / target.name / "Custom-Results").exists()


@pytest.mark.parametrize("stager", ["generated", "native"])
def test_declared_in_skill_output_can_be_regenerated(tmp_path: Path, stager: str) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    if stager == "native":
        _write_minimal_native_task(target)
    output_root = target / "custom-results"
    output_dir = output_root / "dataset"
    stager_fn = generate_harbor_tasks if stager == "generated" else stage_native_harbor_tasks

    first = stager_fn(target, output_dir, repo_context_exclude_paths=(output_root,))
    second = stager_fn(target, output_dir, repo_context_exclude_paths=(output_root,))

    assert first and second
    assert (target / "SKILL.md").is_file()
    assert (output_dir / "dataset.toml").is_file()


def test_broad_output_ancestor_does_not_empty_runtime_skill(tmp_path: Path) -> None:
    _, target, references_dir, workspace = _write_projection_fixture(tmp_path)

    task = generate_harbor_tasks(
        target,
        tmp_path / "ancestor-output",
        reference_skills_dir=references_dir,
        workspace_skill_paths=[workspace],
        repo_context_exclude_paths=(tmp_path,),
    )[0]

    staged_target = task / "environment" / "skills" / target.name
    assert (staged_target / "SKILL.md").is_file()
    assert (staged_target / "scripts" / "run.py").is_file()
    pip_reqs, _ = _collect_all_skill_deps(target, references_dir, [workspace], excluded_roots=(tmp_path,))
    assert "target-runtime==1" in pip_reqs


@pytest.mark.parametrize("source_name", ["files", "environment"])
def test_generated_rejects_output_root_inside_evaluator_source(tmp_path: Path, source_name: str) -> None:
    _, target, references_dir, workspace = _write_projection_fixture(tmp_path)
    unsafe_root = target / "evals" / source_name / "configured-results"

    with pytest.raises(ValueError, match="output root must not be inside evaluator source"):
        generate_harbor_tasks(
            target,
            tmp_path / f"unsafe-generated-{source_name}",
            reference_skills_dir=references_dir,
            workspace_skill_paths=[workspace],
            repo_context_exclude_paths=(unsafe_root,),
        )


def test_native_rejects_output_root_inside_evaluator_source(tmp_path: Path) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    _write_minimal_native_task(target)
    unsafe_root = target / "evals" / "harbor" / "configured-results"

    with pytest.raises(ValueError, match="output root must not be inside evaluator source"):
        stage_native_harbor_tasks(
            target,
            tmp_path / "unsafe-native-output",
            repo_context_exclude_paths=(unsafe_root,),
        )


def test_default_results_root_remains_supported(tmp_path: Path) -> None:
    _, target, references_dir, workspace = _write_projection_fixture(tmp_path)

    tasks = generate_harbor_tasks(
        target,
        tmp_path / "safe-default-results",
        reference_skills_dir=references_dir,
        workspace_skill_paths=[workspace],
        repo_context_exclude_paths=(target / "evals" / "results",),
    )

    assert tasks


def test_declared_custom_in_skill_results_root_remains_supported(tmp_path: Path) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)

    validate_results_root_location(target, target / "custom-results")
    with pytest.raises(ValueError, match="runtime skill source"):
        validate_results_root_location(target, target / "scripts" / "results")


@pytest.mark.parametrize("source_alias", ["FILES", "Environment", "HARBOR"])
def test_results_root_rejects_case_alias_of_evaluator_source(tmp_path: Path, source_alias: str) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)

    with pytest.raises(ValueError, match="output root must not be inside evaluator source"):
        validate_results_root_location(target, target / "evals" / source_alias / "prior-results")


def test_excluded_root_matching_is_case_insensitive(tmp_path: Path) -> None:
    actual = tmp_path / "Custom-Results" / "agent-a"
    typed_alias = tmp_path / "custom-results"

    assert _path_is_excluded(actual, (typed_alias,))


def test_prebuild_uses_each_tasks_exact_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset"
    build_calls: list[tuple[str, Path]] = []
    for case_id, fixture in (("case-a", "ALPHA"), ("case-b", "BETA")):
        task = dataset / case_id
        environment = task / "environment"
        (environment / "input").mkdir(parents=True)
        (environment / "Dockerfile").write_text(
            "FROM python:3.12-slim\nCOPY input/ /workspace/input/\n",
            encoding="utf-8",
        )
        (environment / "input" / "fixture.txt").write_text(fixture, encoding="utf-8")
        (task / "task.toml").write_text("[environment]\n", encoding="utf-8")

    class _Result:
        def __init__(self, returncode: int) -> None:
            self.returncode = returncode
            self.stderr = ""
            self.stdout = ""

    def _fake_run(command: list[str], **_kwargs: object) -> _Result:
        if command[:3] == ["docker", "image", "inspect"]:
            return _Result(1)
        assert command[:3] == ["docker", "build", "-t"]
        build_calls.append((command[3], Path(command[4])))
        return _Result(0)

    monkeypatch.setenv("SKILL_EVAL_HARBOR_PREBUILD_TASK_ENVS", "1")
    monkeypatch.setattr("skillevaluator.tier3.harbor.adapter.subprocess.run", _fake_run)

    assert prebuild_task_environments([dataset]) == 2
    assert {path.name for _, path in build_calls} == {"environment"}
    assert {path.parent.name for _, path in build_calls} == {"case-a", "case-b"}
    assert len({tag for tag, _ in build_calls}) == 2
    task_tags = {
        task.name: next(
            line for line in (task / "task.toml").read_text(encoding="utf-8").splitlines() if "docker_image" in line
        )
        for task in sorted(dataset.iterdir())
    }
    assert task_tags["case-a"] != task_tags["case-b"]

    first_build_tags = {path.parent.name: tag for tag, path in build_calls}
    fixture_path = dataset / "case-a" / "environment" / "input" / "fixture.txt"
    original_mode = stat.S_IMODE(fixture_path.stat().st_mode)
    changed_mode = original_mode & ~stat.S_IWRITE if os.name == "nt" else original_mode | stat.S_IXUSR
    try:
        fixture_path.chmod(changed_mode)
        for task in dataset.iterdir():
            (task / "task.toml").write_text("[environment]\n", encoding="utf-8")
        build_calls.clear()

        assert prebuild_task_environments([dataset]) == 2
        second_build_tags = {path.parent.name: tag for tag, path in build_calls}
        assert second_build_tags["case-a"] != first_build_tags["case-a"]
        assert second_build_tags["case-b"] == first_build_tags["case-b"]
    finally:
        fixture_path.chmod(original_mode)


def test_prebuild_skips_task_with_compose_build_override(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("SKILL_EVAL_HARBOR_PREBUILD_TASK_ENVS", "1")
    dataset = tmp_path / "dataset"
    task = dataset / "case-a"
    environment = task / "environment"
    environment.mkdir(parents=True)
    (environment / "Dockerfile").write_text("FROM python:3.12-slim\n", encoding="utf-8")
    (environment / "docker-compose.yaml").write_text(
        "services:\n  main:\n    build:\n      context: ./alternate\n",
        encoding="utf-8",
    )
    (task / "task.toml").write_text("[environment]\n", encoding="utf-8")
    calls: list[list[str]] = []

    def _unexpected_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr("skillevaluator.tier3.harbor.adapter.subprocess.run", _unexpected_run)

    assert prebuild_task_environments([dataset]) == 0
    assert calls == []
    assert "docker_image" not in (task / "task.toml").read_text(encoding="utf-8")


@pytest.mark.parametrize("stager", ["generated", "native"])
@pytest.mark.parametrize("output_kind", ["evals", "skill"])
def test_staging_rejects_output_that_contains_evaluator_sources_before_mutation(
    tmp_path: Path,
    stager: str,
    output_kind: str,
) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    if stager == "native":
        _write_minimal_native_task(target)
    output_dir = target / "evals" if output_kind == "evals" else target

    with pytest.raises(ValueError, match="Staging output directory overlaps evaluator source"):
        if stager == "generated":
            generate_harbor_tasks(target, output_dir)
        else:
            stage_native_harbor_tasks(target, output_dir)

    assert (target / "SKILL.md").is_file()
    assert (target / "evals" / "evals.json").is_file()


@pytest.mark.parametrize("stager", ["generated", "native"])
@pytest.mark.parametrize("eval_subdir", ["data", "tests"])
def test_staging_rejects_output_inside_any_evaluator_source_subtree(
    tmp_path: Path,
    stager: str,
    eval_subdir: str,
) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    if stager == "native":
        _write_minimal_native_task(target)
    output_dir = target / "evals" / eval_subdir
    output_dir.mkdir(exist_ok=True)
    sentinel = output_dir / "keep.txt"
    sentinel.write_text("preserve\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Staging output directory overlaps evaluator source"):
        if stager == "generated":
            generate_harbor_tasks(target, output_dir)
        else:
            stage_native_harbor_tasks(target, output_dir)

    assert sentinel.read_text(encoding="utf-8") == "preserve\n"


@pytest.mark.parametrize("stager", ["generated", "native"])
@pytest.mark.parametrize("alias", ["Results", "RESULTS"])
def test_staging_rejects_case_alias_of_reserved_results_directory(
    tmp_path: Path,
    stager: str,
    alias: str,
) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    if stager == "native":
        _write_minimal_native_task(target)
    output_dir = target / "evals" / alias
    output_dir.mkdir()
    sentinel = output_dir / "keep.txt"
    sentinel.write_text("preserve\n", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=r"Staging output directory overlaps evaluator source|Generated output marker is missing",
    ):
        stager_fn = generate_harbor_tasks if stager == "generated" else stage_native_harbor_tasks
        stager_fn(target, output_dir)

    assert sentinel.read_text(encoding="utf-8") == "preserve\n"


@pytest.mark.parametrize("stager", ["generated", "native", "results-root"])
def test_output_path_rejects_parent_traversal_before_evaluator_source_mutation(tmp_path: Path, stager: str) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    if stager == "native":
        _write_minimal_native_task(target)
    sentinel = target / "evals" / "files" / "selected.txt"
    output_dir = target / "evals" / "results" / ".." / "files"

    with pytest.raises(ValueError, match="parent traversal"):
        if stager == "generated":
            generate_harbor_tasks(target, output_dir)
        elif stager == "native":
            stage_native_harbor_tasks(target, output_dir)
        else:
            validate_results_root_location(target, output_dir)

    assert sentinel.read_text(encoding="utf-8") == "SELECTED-FIXTURE\n"


@pytest.mark.parametrize("stager", ["generated", "native"])
@pytest.mark.parametrize("source_kind", ["reference", "reference-parent", "workspace", "workspace-parent"])
def test_staging_rejects_output_overlapping_supplied_skill_sources(
    tmp_path: Path,
    stager: str,
    source_kind: str,
) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    if stager == "native":
        _write_minimal_native_task(target)
    reference_parent = tmp_path / "external-reference-source"
    references = reference_parent / "references"
    reference = _write_runtime_skill(references / "reference", "reference-external")
    workspace_parent = tmp_path / "external-workspace-source"
    workspace = _write_runtime_skill(workspace_parent / "workspace", "workspace-external")
    output_dir = {
        "reference": reference,
        "reference-parent": reference_parent,
        "workspace": workspace,
        "workspace-parent": workspace_parent,
    }[source_kind]
    marker = output_dir / "source-marker.txt"
    marker.write_text("preserve\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Staging output directory overlaps runtime skill source"):
        stager_fn = generate_harbor_tasks if stager == "generated" else stage_native_harbor_tasks
        stager_fn(
            target,
            output_dir,
            reference_skills_dir=references,
            workspace_skill_paths=[workspace],
        )

    assert marker.read_text(encoding="utf-8") == "preserve\n"


@pytest.mark.parametrize("stager", ["generated", "native"])
def test_staging_rejects_undeclared_output_inside_target_runtime_tree(tmp_path: Path, stager: str) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    runtime_source = target / "src"
    runtime_source.mkdir()
    marker = runtime_source / "runtime.py"
    marker.write_text("print('preserve')\n", encoding="utf-8")
    if stager == "native":
        _write_minimal_native_task(target)

    with pytest.raises(ValueError, match="Staging output directory overlaps runtime skill source"):
        stager_fn = generate_harbor_tasks if stager == "generated" else stage_native_harbor_tasks
        stager_fn(target, runtime_source)

    assert marker.read_text(encoding="utf-8") == "print('preserve')\n"


@pytest.mark.parametrize("stager", ["generated", "native"])
def test_declared_output_cannot_replace_standard_runtime_source(tmp_path: Path, stager: str) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    scripts = target / "scripts"
    marker = scripts / "run.py"
    if stager == "native":
        _write_minimal_native_task(target)

    with pytest.raises(ValueError, match="runtime skill source"):
        stager_fn = generate_harbor_tasks if stager == "generated" else stage_native_harbor_tasks
        stager_fn(target, scripts, repo_context_exclude_paths=(scripts,))

    assert marker.read_text(encoding="utf-8") == "print('runtime')\n"


@pytest.mark.parametrize("stager", ["generated", "native"])
def test_declared_output_root_cannot_contain_authored_nested_skill(tmp_path: Path, stager: str) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    declared_root = target / "addons"
    nested_skill = declared_root / "helper"
    nested_skill.mkdir(parents=True)
    (nested_skill / "SKILL.md").write_text("# Helper\n", encoding="utf-8")
    marker = nested_skill / "runtime.txt"
    marker.write_text("preserve\n", encoding="utf-8")
    if stager == "native":
        _write_minimal_native_task(target)

    with pytest.raises(ValueError, match="output root must not overlap runtime skill source"):
        stager_fn = generate_harbor_tasks if stager == "generated" else stage_native_harbor_tasks
        stager_fn(
            target,
            declared_root / "generated" / "dataset",
            repo_context_exclude_paths=(declared_root,),
        )

    assert marker.read_text(encoding="utf-8") == "preserve\n"


def test_declared_output_does_not_trust_authored_dataset_manifest(tmp_path: Path) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    declared_root = target / "addons"
    nested_skill = declared_root / "helper"
    nested_skill.mkdir(parents=True)
    (declared_root / "dataset.toml").write_text('[dataset]\nname = "authored"\n', encoding="utf-8")
    (nested_skill / "SKILL.md").write_text("# Authored helper\n", encoding="utf-8")
    sentinel = nested_skill / "runtime.txt"
    sentinel.write_text("preserve\n", encoding="utf-8")
    (target / "evals" / "evals.json").write_text(
        json.dumps([{"id": "helper", "question": "Do not replace authored content."}]),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="overlaps runtime skill source"):
        generate_harbor_tasks(target, declared_root, repo_context_exclude_paths=(declared_root,))

    assert sentinel.read_text(encoding="utf-8") == "preserve\n"


def test_marked_prior_run_does_not_block_declared_results_root(tmp_path: Path) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    results_root = target / "custom-results"
    prior_run = results_root / "run-old"
    mark_generated_output_root(prior_run)
    retained_skill = prior_run / "jobs" / "case-001" / "workspace" / "skills" / target.name
    retained_skill.mkdir(parents=True)
    (retained_skill / "SKILL.md").write_text("# Retained generated copy\n", encoding="utf-8")

    validate_results_root_location(target, results_root)


@pytest.mark.parametrize("copy_repo", [False, True])
@pytest.mark.parametrize("stager_name", ["generated", "native"])
def test_rotated_generated_output_is_excluded_from_runtime_skill_and_repo(
    tmp_path: Path,
    copy_repo: bool,
    stager_name: str,
) -> None:
    repo, target, _, _ = _write_projection_fixture(tmp_path)
    (target / "evals" / "files" / "selected.txt").write_text("OLD-CASE-SECRET\n", encoding="utf-8")
    old_output_root = target / "archived-output"
    old_dataset = old_output_root / "dataset"
    generate_harbor_tasks(
        target,
        old_dataset,
        repo_context_exclude_paths=(old_output_root,),
    )
    assert (old_dataset / "case-001" / "environment" / "input" / "selected.txt").is_file()
    older_output_root = target / "older-output"
    older_dataset = older_output_root / "dataset"
    generate_harbor_tasks(
        target,
        older_dataset,
        repo_context_exclude_paths=(older_output_root,),
    )

    (target / "evals" / "evals.json").write_text(
        json.dumps([{"id": "case-001", "question": "Do not use old fixtures.", "files": []}]),
        encoding="utf-8",
    )
    if stager_name == "native":
        _write_minimal_native_task(target)
        stager = stage_native_harbor_tasks
    else:
        stager = generate_harbor_tasks

    task = stager(target, tmp_path / f"current-{stager_name}-{copy_repo}", copy_repo=copy_repo)[0]
    environment = task / "environment"
    for copied_relative in (Path("archived-output") / "dataset", Path("older-output") / "dataset"):
        assert not (environment / "skills" / target.name / copied_relative).exists()
        if copy_repo:
            assert not (environment / "repo" / target.relative_to(repo) / copied_relative).exists()
    readable = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore") for path in environment.rglob("*") if path.is_file()
    )
    assert "OLD-CASE-SECRET" not in readable


@pytest.mark.parametrize("copy_repo", [False, True])
def test_rotated_results_latest_alias_to_authenticated_run_is_excluded(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    copy_repo: bool,
) -> None:
    monkeypatch.setenv(
        "SKILLEVALUATOR_OUTPUT_PROVENANCE_KEY_FILE",
        str(tmp_path / ".skillevaluator-state" / "output-provenance.key"),
    )
    repo, target, _, _ = _write_projection_fixture(tmp_path)
    old_results_root = target / "archived-results"
    run_id = "20260805_120000_123_aaaaaaaaaaaa"
    old_run = old_results_root / run_id
    mark_generated_output_root(old_run)
    (old_run / "run_config.json").write_text("{}\n", encoding="utf-8")
    (old_run / "result.json").write_text(json.dumps({"run_id": run_id}), encoding="utf-8")
    (old_run / "fixture-oracle.txt").write_text("HISTORICAL-RESULT-ORACLE\n", encoding="utf-8")
    assert publish_latest_results(old_results_root, run_id)
    (target / "evals" / "evals.json").write_text(
        json.dumps([{"id": "case-001", "question": "Do not use historical results.", "files": []}]),
        encoding="utf-8",
    )

    task = generate_harbor_tasks(target, tmp_path / f"current-latest-{copy_repo}", copy_repo=copy_repo)[0]

    environment = task / "environment"
    runtime_old_root = environment / "skills" / target.name / old_results_root.relative_to(target)
    assert not (runtime_old_root / run_id).exists()
    assert not (runtime_old_root / "latest").exists()
    if copy_repo:
        repo_old_root = environment / "repo" / target.relative_to(repo) / old_results_root.relative_to(target)
        assert not (repo_old_root / run_id).exists()
        assert not (repo_old_root / "latest").exists()
    readable = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore") for path in environment.rglob("*") if path.is_file()
    )
    assert "HISTORICAL-RESULT-ORACLE" not in readable


@pytest.mark.parametrize("lookalike", ["unmarked-latest", "outside-latest", "non-latest-alias"])
def test_rotated_results_symlink_lookalikes_remain_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    lookalike: str,
) -> None:
    monkeypatch.setenv(
        "SKILLEVALUATOR_OUTPUT_PROVENANCE_KEY_FILE",
        str(tmp_path / ".skillevaluator-state" / "output-provenance.key"),
    )
    _, target, _, _ = _write_projection_fixture(tmp_path)
    old_results_root = target / "archived-results"
    old_results_root.mkdir()
    if lookalike == "non-latest-alias":
        target_dir = old_results_root / "marked-run"
        mark_generated_output_root(target_dir)
        alias = old_results_root / "current"
    elif lookalike == "unmarked-latest":
        target_dir = old_results_root / "unmarked-run"
        target_dir.mkdir()
        alias = old_results_root / "latest"
    else:
        target_dir = tmp_path / "outside-run"
        target_dir.mkdir()
        alias = old_results_root / "latest"
    (target_dir / "payload.txt").write_text("UNAUTHENTICATED-LOOKALIKE\n", encoding="utf-8")
    try:
        alias.symlink_to(target_dir.name if target_dir.parent == old_results_root else target_dir)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")

    output = tmp_path / f"lookalike-{lookalike}"
    with pytest.raises(ValueError, match="symlink"):
        generate_harbor_tasks(target, output)

    assert not output.exists()


@pytest.mark.skipif(sys.platform != "darwin", reason="requires the macOS /var root alias")
def test_rotated_output_is_excluded_through_macos_platform_root_alias() -> None:
    with tempfile.TemporaryDirectory(prefix="issue29-platform-alias-") as temporary:
        root = Path(temporary)
        if root.absolute() == root.resolve():
            pytest.skip("temporary directory does not use a platform root alias")
        _, target, _, _ = _write_projection_fixture(root)
        selected = target / "evals" / "files" / "selected.txt"
        selected.write_text("PLATFORM-ALIAS-SECRET\n", encoding="utf-8")
        old_root = target / "archived-output"
        generate_harbor_tasks(
            target,
            old_root / "dataset",
            repo_context_exclude_paths=(old_root,),
        )
        (target / "evals" / "evals.json").write_text(
            json.dumps([{"id": "case-001", "question": "No old fixtures.", "files": []}]),
            encoding="utf-8",
        )

        task = generate_harbor_tasks(target, root / "current", copy_repo=True)[0]

        readable = "\n".join(
            path.read_text(encoding="utf-8", errors="ignore")
            for path in (task / "environment").rglob("*")
            if path.is_file()
        )
        assert "PLATFORM-ALIAS-SECRET" not in readable


@pytest.mark.parametrize("marker_state", ["copied", "rotated-key"])
def test_unverifiable_historical_generated_output_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    marker_state: str,
) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    old_output_root = target / "archived-output"
    old_dataset = old_output_root / "dataset"
    generate_harbor_tasks(
        target,
        old_dataset,
        repo_context_exclude_paths=(old_output_root,),
    )

    if marker_state == "copied":
        shutil.copytree(old_dataset, target / "copied-output")
    else:
        monkeypatch.setenv(
            "SKILLEVALUATOR_OUTPUT_PROVENANCE_KEY_FILE",
            str(tmp_path / ".rotated-state" / "output-provenance.key"),
        )

    (target / "evals" / "evals.json").write_text(
        json.dumps([{"id": "case-001", "question": "Do not use old fixtures.", "files": []}]),
        encoding="utf-8",
    )
    current = tmp_path / f"current-{marker_state}"
    with pytest.raises(ValueError, match=r"marker.*(?:invalid|authenticated|verified)"):
        generate_harbor_tasks(target, current, copy_repo=True)

    assert not current.exists()


def test_declared_in_skill_output_recovers_after_partial_generation_failure(tmp_path: Path) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    (target / "evals" / "evals.json").write_text(
        json.dumps(
            [
                {"id": "case-a", "question": "Stage existing.", "files": ["evals/files/selected.txt"]},
                {"id": "case-b", "question": "Stage missing.", "files": ["evals/files/later.txt"]},
            ]
        ),
        encoding="utf-8",
    )
    declared_root = target / "custom-results"
    output_dir = declared_root / "dataset"

    with pytest.raises(FileNotFoundError, match="does not exist"):
        generate_harbor_tasks(target, output_dir, repo_context_exclude_paths=(declared_root,))

    (target / "evals" / "files" / "later.txt").write_text("LATER\n", encoding="utf-8")
    tasks = generate_harbor_tasks(target, output_dir, repo_context_exclude_paths=(declared_root,))

    assert {task.name for task in tasks} == {"case-a", "case-b"}
    assert (output_dir / ".skillevaluator-generated-output").is_file()


@pytest.mark.parametrize("stager", ["generated", "native"])
def test_staging_rejects_symlinked_output_parent_without_replacing_target(tmp_path: Path, stager: str) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    if stager == "native":
        _write_minimal_native_task(target)
    victim_parent = tmp_path / "victim-parent"
    victim_output = victim_parent / "dataset"
    victim_output.mkdir(parents=True)
    sentinel = victim_output / "keep.txt"
    sentinel.write_text("preserve\n", encoding="utf-8")
    alias = tmp_path / "output-alias"
    alias.symlink_to(victim_parent, target_is_directory=True)

    with pytest.raises(ValueError, match=r"symlink|reparse|junction"):
        stager_fn = generate_harbor_tasks if stager == "generated" else stage_native_harbor_tasks
        stager_fn(target, alias / "dataset")

    assert sentinel.read_text(encoding="utf-8") == "preserve\n"


def test_legacy_input_rejects_symlinked_evals_ancestor(tmp_path: Path) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    outside_evals = tmp_path / "outside-evals"
    (target / "evals").rename(outside_evals)
    (outside_evals / "evals.json").write_text(
        json.dumps([{"id": "case-001", "question": "Legacy shared inputs."}]),
        encoding="utf-8",
    )
    (target / "evals").symlink_to(outside_evals, target_is_directory=True)

    with pytest.raises(ValueError, match=r"evals directory must be a real directory.*symlink"):
        generate_harbor_tasks(target, tmp_path / "symlinked-evals-output")

    assert not (tmp_path / "symlinked-evals-output").exists()


@pytest.mark.parametrize("invalid_entries", [[{"id": ".."}], [{"id": "case-001"}, {"id": "Case-001"}]])
def test_native_rejects_invalid_or_colliding_case_ids_before_output_mutation(
    tmp_path: Path,
    invalid_entries: list[dict[str, str]],
) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    _write_minimal_native_task(target)
    (target / "evals" / "evals.json").write_text(json.dumps(invalid_entries), encoding="utf-8")
    output_dir = tmp_path / "native-existing-output"
    output_dir.mkdir()
    sentinel = output_dir / "keep.txt"
    sentinel.write_text("preserve\n", encoding="utf-8")

    with pytest.raises(ValueError, match=r"invalid case id|cross-platform colliding case id"):
        stage_native_harbor_tasks(target, output_dir)

    assert sentinel.read_text(encoding="utf-8") == "preserve\n"


def test_native_rejects_unsafe_task_directory_name_before_output_mutation(tmp_path: Path) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    _write_minimal_native_task(target)
    native_root = target / "evals" / "harbor"
    (native_root / "case-001").rename(native_root / "bad name")
    output_dir = tmp_path / "native-invalid-directory"

    with pytest.raises(ValueError, match="invalid case id"):
        stage_native_harbor_tasks(target, output_dir)

    assert not output_dir.exists()


def test_results_root_rejects_supplied_skill_source(tmp_path: Path) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    references = tmp_path / "external-references"
    reference = _write_runtime_skill(references / "reference", "reference-output")
    workspace = _write_runtime_skill(tmp_path / "external-workspace", "workspace-output")

    for unsafe_root in (reference / "evals" / "files" / "results", workspace / "scripts" / "results"):
        with pytest.raises(ValueError, match="output root must not be inside runtime skill source"):
            validate_results_root_location(
                target,
                unsafe_root,
                reference_skills_dir=references,
                workspace_skill_paths=[workspace],
            )


def test_baseline_rejects_target_copied_under_reference_alias(tmp_path: Path) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    references_dir = tmp_path / "aliased-references"
    aliased_target = references_dir / "alias-target"
    shutil.copytree(target, aliased_target)
    (aliased_target / "extra.txt").write_text("tree changed but instructions are identical\n", encoding="utf-8")

    with pytest.raises(
        ValueError,
        match=r"alias of target skill|Baseline environment contains the target skill instructions",
    ):
        generate_harbor_tasks(
            target,
            tmp_path / "baseline-reference-alias",
            with_skill=False,
            reference_skills_dir=references_dir,
        )


@pytest.mark.parametrize("stager", ["generated", "native"])
def test_baseline_reference_parent_skips_real_target_candidate(tmp_path: Path, stager: str) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    if stager == "native":
        _write_minimal_native_task(target)
        tasks = stage_native_harbor_tasks(
            target,
            tmp_path / "native-parent-references",
            with_skill=False,
            reference_skills_dir=target.parent,
        )
    else:
        tasks = generate_harbor_tasks(
            target,
            tmp_path / "generated-parent-references",
            with_skill=False,
            reference_skills_dir=target.parent,
        )

    assert tasks


def test_baseline_alias_scan_ignores_unstaged_candidate_results(tmp_path: Path) -> None:
    _, target, references_dir, workspace = _write_projection_fixture(tmp_path)
    archived = references_dir / "reference-skill" / "results" / "archived.txt"
    archived.parent.mkdir()
    archived.write_bytes((target / "SKILL.md").read_bytes())

    task = generate_harbor_tasks(
        target,
        tmp_path / "ignored-reference-results",
        with_skill=False,
        reference_skills_dir=references_dir,
        workspace_skill_paths=[workspace],
    )[0]

    assert not (task / "environment" / "skills" / "reference-skill" / "results").exists()


@pytest.mark.parametrize("stager", ["generated", "native"])
@pytest.mark.parametrize("candidate_kind", ["reference", "workspace"])
def test_baseline_rejects_target_nested_inside_skill_candidate(
    tmp_path: Path,
    stager: str,
    candidate_kind: str,
) -> None:
    _, target, references_dir, workspace = _write_projection_fixture(tmp_path)
    candidate = references_dir / "reference-skill" if candidate_kind == "reference" else workspace
    nested_alias = candidate / "references" / "target-copy"
    nested_alias.mkdir(parents=True)
    (nested_alias / "SKILL.md").write_bytes((target / "SKILL.md").read_bytes())

    kwargs = {
        "with_skill": False,
        "reference_skills_dir": references_dir,
        "workspace_skill_paths": [workspace],
    }
    with pytest.raises(
        ValueError,
        match=r"alias of target skill|Baseline environment contains the target skill instructions",
    ):
        if stager == "generated":
            generate_harbor_tasks(target, tmp_path / f"nested-alias-{candidate_kind}", **kwargs)
        else:
            _write_minimal_native_task(target)
            stage_native_harbor_tasks(target, tmp_path / f"native-nested-alias-{candidate_kind}", **kwargs)


@pytest.mark.parametrize("stager", ["generated", "native"])
def test_copy_repo_baseline_rejects_renamed_target_payload(tmp_path: Path, stager: str) -> None:
    repo, target, references_dir, workspace = _write_projection_fixture(tmp_path)
    archived = repo / "archive" / "renamed-target.md"
    archived.parent.mkdir()
    archived.write_bytes((target / "SKILL.md").read_bytes())
    kwargs = {
        "with_skill": False,
        "reference_skills_dir": references_dir,
        "workspace_skill_paths": [workspace],
        "copy_repo": True,
    }

    with pytest.raises(ValueError, match="target skill instructions"):
        if stager == "generated":
            generate_harbor_tasks(target, tmp_path / "repo-payload-alias", **kwargs)
        else:
            _write_minimal_native_task(target)
            stage_native_harbor_tasks(target, tmp_path / "native-repo-payload-alias", **kwargs)


def test_baseline_projects_reference_and_workspace_skills_without_root_evals(tmp_path: Path) -> None:
    _, target, references_dir, workspace = _write_projection_fixture(tmp_path)

    task = generate_harbor_tasks(
        target,
        tmp_path / "baseline",
        with_skill=False,
        reference_skills_dir=references_dir,
        workspace_skill_paths=[workspace],
        copy_repo=True,
    )[0]

    environment = task / "environment"
    assert not (environment / "skills" / target.name).exists()
    for name in ("reference-skill", workspace.name):
        assert (environment / "skills" / name / "SKILL.md").is_file()
        assert not (environment / "skills" / name / "evals").exists()
        assert (environment / "skills" / name / "references" / "evals" / "keep.md").is_file()
    assert not (environment / "repo" / target.name).exists()
    assert not (environment / "repo" / "reference-skills" / "reference-skill" / "evals").exists()
    assert not (environment / "repo" / workspace.name / "evals").exists()
    assert (environment / "repo" / "docs" / "evals" / "keep.txt").is_file()
    assert (environment / "input" / "selected.txt").is_file()
    assert "GROUND-TRUTH-SECRET" not in "\n".join(
        path.read_text(encoding="utf-8", errors="ignore") for path in environment.rglob("*") if path.is_file()
    )


def test_linked_repo_context_excludes_skill_evals_but_keeps_unrelated_evals(tmp_path: Path) -> None:
    _, target, references_dir, workspace = _write_projection_fixture(tmp_path)
    (target / "SKILL.md").write_text(
        "# Target\n\n"
        "[Hidden evaluator data](../reference-skills/reference-skill/evals/evals.json)\n"
        "[Runtime documentation](../docs/evals/keep.txt)\n",
        encoding="utf-8",
    )

    task = generate_harbor_tasks(
        target,
        tmp_path / "linked",
        reference_skills_dir=references_dir,
        workspace_skill_paths=[workspace],
        copy_repo=False,
    )[0]

    staged_repo = task / "environment" / "repo"
    assert not (staged_repo / "reference-skills" / "reference-skill" / "evals" / "evals.json").exists()
    assert (staged_repo / "docs" / "evals" / "keep.txt").is_file()


@pytest.mark.parametrize("stager", ["generated", "native"])
def test_linked_repo_context_prunes_excluded_artifacts_before_resolving(
    tmp_path: Path,
    stager: str,
) -> None:
    repo, target, references_dir, workspace = _write_projection_fixture(tmp_path)
    excluded = repo / "custom-results"
    excluded.mkdir()
    try:
        (excluded / "loop").symlink_to("loop")
        (excluded / "dangling").symlink_to("missing")
    except OSError as exc:  # pragma: no cover - host policy, primarily native Windows
        pytest.skip(f"symlinks unavailable on this host: {exc}")
    if hasattr(os, "mkfifo"):
        os.mkfifo(excluded / "artifact.fifo")
    unreadable = excluded / "unreadable.txt"
    unreadable.write_text("private", encoding="utf-8")
    unreadable.chmod(0)
    links = "\n".join(
        f"[{name}](../custom-results/{name})" for name in ("loop", "dangling", "artifact.fifo", "unreadable.txt")
    )
    (target / "SKILL.md").write_text(f"# Target\n\n{links}\n", encoding="utf-8")
    if stager == "native":
        _write_minimal_native_task(target)
        stage = stage_native_harbor_tasks
    else:
        stage = generate_harbor_tasks

    try:
        task = stage(
            target,
            tmp_path / f"{stager}-linked-excluded",
            reference_skills_dir=references_dir,
            workspace_skill_paths=[workspace],
            copy_repo=False,
            repo_context_exclude_paths=(excluded,),
        )[0]
    finally:
        unreadable.chmod(0o600)

    assert not (task / "environment" / "repo" / excluded.relative_to(repo)).exists()


@pytest.mark.parametrize("files_value", [[], ["evals/files/selected.txt"]])
@pytest.mark.parametrize("source", ["INPUT/", "Input/fixture.txt"])
def test_custom_dockerfile_rejects_noncanonical_input_case(
    tmp_path: Path,
    files_value: list[str],
    source: str,
) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    (target / "evals" / "evals.json").write_text(
        json.dumps([{"id": "case-001", "question": "Use fixtures.", "files": files_value}]),
        encoding="utf-8",
    )
    custom_environment = target / "evals" / "environment"
    authored_input = custom_environment / "INPUT"
    authored_input.mkdir()
    (authored_input / "fixture.txt").write_text("UNDECLARED-UPPERCASE-INPUT\n", encoding="utf-8")
    (custom_environment / "Dockerfile").write_text(
        f"FROM python:3.12-slim\nCOPY {source} /workspace/input/\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="canonical lowercase path"):
        generate_harbor_tasks(target, tmp_path / "uppercase-input")


@pytest.mark.parametrize("custom_dockerfile_mode", ["preserve", "rebase"])
@pytest.mark.parametrize(
    "copy_instruction",
    [
        "COPY input/ /workspace/input/",
        "COPY input/. /workspace/input/",
        "COPY ../input/ /workspace/input/",
        "COPY --chown=0:0\tinput/\t/workspace/input/",
        'COPY --chown=0:0 ["input/", "/workspace/input/"]',
        'COPY ["../input/", "/workspace/input/"]',
    ],
)
def test_explicit_empty_input_keeps_authored_copy_input_buildable(
    tmp_path: Path,
    custom_dockerfile_mode: str,
    copy_instruction: str,
) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    (target / "evals" / "evals.json").write_text(
        json.dumps([{"id": "case-001", "question": "No fixture.", "files": []}]),
        encoding="utf-8",
    )
    custom_environment = target / "evals" / "environment"
    authored_input = custom_environment / "input"
    authored_input.mkdir()
    (authored_input / "hidden.txt").write_text("UNDECLARED-CUSTOM-INPUT\n", encoding="utf-8")
    (custom_environment / "Dockerfile").write_text(
        f"FROM python:3.12-slim\n{copy_instruction}\n",
        encoding="utf-8",
    )

    task = generate_harbor_tasks(
        target,
        tmp_path / f"custom-empty-{custom_dockerfile_mode}",
        custom_dockerfile_mode=custom_dockerfile_mode,
        base_image="python:3.12-slim" if custom_dockerfile_mode == "rebase" else "",
    )[0]

    input_dir = task / "environment" / "input"
    assert input_dir.is_dir()
    assert list(input_dir.iterdir()) == []
    assert copy_instruction in (task / "environment" / "Dockerfile").read_text(encoding="utf-8")


def test_readonly_custom_environment_snapshot_stays_outside_docker_context(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if os.name != "posix":
        pytest.skip("POSIX directory modes required")
    _, target, _, _ = _write_projection_fixture(tmp_path)
    (target / "evals" / "evals.json").write_text(
        json.dumps([{"id": "case-001", "question": "No fixture.", "files": []}]),
        encoding="utf-8",
    )
    custom_environment = target / "evals" / "environment"
    snapshot_destinations: list[Path] = []
    original_copytree = adapter_module.copytree_secure

    def record_custom_snapshot(source: Path, destination: Path, *args: object, **kwargs: object) -> None:
        source_path = Path(source)
        if source_path.name == "environment" and source_path.parent.name == "evals":
            snapshot_destinations.append(Path(destination))
        original_copytree(source, destination, *args, **kwargs)

    monkeypatch.setattr(adapter_module, "copytree_secure", record_custom_snapshot)
    authored_input = custom_environment / "input"
    authored_input.mkdir()
    (authored_input / "hidden.txt").write_text("UNDECLARED-CUSTOM-INPUT\n", encoding="utf-8")
    (custom_environment / "Dockerfile").write_text(
        "FROM python:3.12-slim\nCOPY . /context\n",
        encoding="utf-8",
    )
    sidecar = custom_environment / "sidecar"
    sidecar.chmod(0o555)
    custom_environment.chmod(0o555)
    try:
        task = generate_harbor_tasks(
            target,
            tmp_path / "readonly-custom-environment",
            custom_dockerfile_mode="preserve",
            grading_mode="custom_only",
        )[0]
    finally:
        custom_environment.chmod(0o755)
        sidecar.chmod(0o755)

    environment = task / "environment"
    assert len(snapshot_destinations) == 1
    assert not snapshot_destinations[0].absolute().is_relative_to(environment.absolute())
    assert not any(path.name == ".skillevaluator-custom-environment" for path in environment.rglob("*"))
    assert "COPY . /context" in (environment / "Dockerfile").read_text(encoding="utf-8")
    assert (environment / "input").is_dir()
    assert list((environment / "input").iterdir()) == []
    assert not (environment / "input" / "hidden.txt").exists()
    readable_environment = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore") for path in environment.rglob("*") if path.is_file()
    )
    assert "UNDECLARED-CUSTOM-INPUT" not in readable_environment


def test_native_copy_input_without_fixtures_gets_empty_build_context_directory(tmp_path: Path) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    (target / "evals" / "evals.json").write_text(
        json.dumps([{"id": "case-001", "question": "No fixture.", "files": []}]),
        encoding="utf-8",
    )
    native_task = target / "evals" / "harbor" / "case-001"
    environment = native_task / "environment"
    environment.mkdir(parents=True)
    (environment / "Dockerfile").write_text(
        "FROM python:3.12-slim\nCOPY input/ /workspace/input/\n",
        encoding="utf-8",
    )
    (native_task / "instruction.md").write_text("Run the native case.\n", encoding="utf-8")
    (native_task / "task.toml").write_text(
        'schema_version = "1.3"\n\n[task]\nname = "nvidia/case-001"\n\n'
        '[metadata]\nentry_id = "case-001"\n\n[environment]\n',
        encoding="utf-8",
    )

    task = stage_native_harbor_tasks(target, tmp_path / "native-empty-input")[0]

    input_dir = task / "environment" / "input"
    assert input_dir.is_dir()
    assert list(input_dir.iterdir()) == []


def test_native_input_is_preserved_without_eval_fixture_declaration(tmp_path: Path) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    shutil.rmtree(target / "evals" / "files")
    (target / "evals" / "evals.json").write_text(
        json.dumps([{"id": "case-001", "question": "Use native input."}]),
        encoding="utf-8",
    )
    native_task = target / "evals" / "harbor" / "case-001"
    environment = native_task / "environment"
    native_input = environment / "input"
    native_input.mkdir(parents=True)
    (native_input / "native-only.txt").write_text("NATIVE-ONLY\n", encoding="utf-8")
    (environment / "Dockerfile").write_text("FROM python:3.12-slim\n", encoding="utf-8")
    (native_task / "instruction.md").write_text("Run the native case.\n", encoding="utf-8")
    (native_task / "task.toml").write_text(
        'schema_version = "1.3"\n\n[task]\nname = "nvidia/case-001"\n\n'
        '[metadata]\nentry_id = "case-001"\n\n[environment]\n',
        encoding="utf-8",
    )

    task = stage_native_harbor_tasks(target, tmp_path / "native-preserved-input")[0]

    assert (task / "environment" / "input" / "native-only.txt").read_text(encoding="utf-8") == "NATIVE-ONLY\n"


def test_generated_baseline_rejects_aliased_authored_skill(tmp_path: Path) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    custom_environment = target / "evals" / "environment"
    alias = custom_environment / "alias"
    alias.mkdir()
    (alias / "SKILL.md").write_text((target / "SKILL.md").read_text(encoding="utf-8"), encoding="utf-8")
    (custom_environment / "Dockerfile").write_text(
        "FROM python:3.12-slim\nCOPY alias/ /root/.agents/skills/alias/\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unmanaged skill"):
        generate_harbor_tasks(target, tmp_path / "baseline-alias", with_skill=False)


def test_generated_baseline_rejects_renamed_target_manifest_payload(tmp_path: Path) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    custom_environment = target / "evals" / "environment"
    (custom_environment / "payload.txt").write_bytes((target / "SKILL.md").read_bytes())
    (custom_environment / "Dockerfile").write_text(
        "FROM python:3.12-slim\nCOPY payload.txt /root/.agents/skills/alias/SKILL.md\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="target skill instructions"):
        generate_harbor_tasks(target, tmp_path / "baseline-renamed-payload", with_skill=False)


@pytest.mark.parametrize(
    "project_skill_root",
    [
        "/workspace/.agents/skills",
        "/workspace/.claude/skills",
        "/workspace/.cline/skills",
        "/workspace/.codex/skills",
        "/workspace/.config/goose/skills",
        "/workspace/.config/opencode/skills",
        "/workspace/.cursor/skills",
        "/workspace/.gemini/skills",
        "/workspace/.opencode/skills",
        "/workspace/.qwen/skills",
    ],
)
def test_final_projection_clears_authored_project_skill_root(
    tmp_path: Path,
    project_skill_root: str,
) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    environment = target / "evals" / "environment"
    hidden_manifest = f"{project_skill_root}/hidden/SKILL.md"
    (environment / "Dockerfile").write_text(
        "FROM python:3.12-slim\n"
        f"RUN mkdir -p {project_skill_root}/hidden && "
        f"printf 'PROJECT-ROOT-ORACLE\\n' > {hidden_manifest}\n",
        encoding="utf-8",
    )

    task = generate_harbor_tasks(target, tmp_path / "project-skill-reset", with_skill=False)[0]

    dockerfile = (task / "environment" / "Dockerfile").read_text(encoding="utf-8")
    authored_index = dockerfile.index("PROJECT-ROOT-ORACLE")
    reset_line = next(
        line
        for line in dockerfile.splitlines()
        if line.startswith('RUN ["/bin/rm", "-rf"') and f'"{project_skill_root}"' in line
    )
    assert dockerfile.index(reset_line) > authored_index


def test_final_projection_clears_effective_workdir_ancestors_and_home(tmp_path: Path) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    environment = target / "evals" / "environment"
    (environment / "Dockerfile").write_text(
        "FROM python:3.12-slim\n"
        "ENV HOME=/home/agent\n"
        "WORKDIR /app\n"
        "RUN mkdir -p /app/.claude/skills/hidden /home/agent/.claude/skills/hidden && "
        "printf 'AUTHORED-CWD-ORACLE\\n' > /app/.claude/skills/hidden/SKILL.md && "
        "printf 'AUTHORED-HOME-ORACLE\\n' > /home/agent/.claude/skills/hidden/SKILL.md\n"
        "USER 12345\n",
        encoding="utf-8",
    )

    task = generate_harbor_tasks(
        target,
        tmp_path / "dynamic-skill-root-reset",
        with_skill=False,
        grading_mode="custom_only",
        agent_workdir="/opt/task/subdir",
    )[0]

    dockerfile = (task / "environment" / "Dockerfile").read_text(encoding="utf-8")
    final_projection = dockerfile.index("# SkillEvaluator: final runtime projection")
    assert dockerfile.index("AUTHORED-CWD-ORACLE") < final_projection
    assert dockerfile.index("AUTHORED-HOME-ORACLE") < final_projection
    assert dockerfile.count("current=$(/bin/pwd -P)") >= 2
    assert "$home/$rel" in dockerfile
    assert "done < /etc/passwd" in dockerfile
    assert r"[ -d \"$home\" ] || continue" in dockerfile
    assert r"[ -e \"$passwd_home\" ] && [ ! -d \"$passwd_home\" ] && continue" in dockerfile
    assert "${CODEX_HOME:-}" in dockerfile
    assert "${OPENCODE_CONFIG_DIR:-}" in dockerfile
    assert "canonical=$(CDPATH= cd -P" in dockerfile
    assert "ENTRYPOINT []" in dockerfile
    assert "HEALTHCHECK NONE" in dockerfile
    assert 'BASH_ENV=""' in dockerfile
    assert 'CLAUDE_CODE_DISABLE_POLICY_SKILLS="1"' in dockerfile
    assert "/etc/codex/skills" in dockerfile
    assert "/tmp/codex-home/skills" in dockerfile
    assert ".config/goose/skills" in dockerfile
    assert ".cline/skills" in dockerfile
    assert ".qwen/skills" in dockerfile
    assert "WORKDIR /opt/task/subdir" in dockerfile[final_projection:]
    assert dockerfile.rindex("current=$(/bin/pwd -P)") > dockerfile.rindex("COPY input/ /workspace/input/")
    task_toml = (task / "task.toml").read_text(encoding="utf-8")
    assert 'workdir = "/opt/task/subdir"' in task_toml
    assert '"BASH_ENV" = ""' in task_toml
    assert '"CLAUDE_CODE_DISABLE_POLICY_SKILLS" = "1"' in task_toml


@pytest.mark.parametrize("agent_workdir", ["relative", "/opt/../tmp", "/opt/task path"])
def test_generated_tasks_reject_unsafe_agent_workdir(tmp_path: Path, agent_workdir: str) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)

    with pytest.raises(ValueError, match="normalized absolute POSIX path"):
        generate_harbor_tasks(target, tmp_path / "unsafe-agent-workdir", agent_workdir=agent_workdir)


@pytest.mark.parametrize(
    "agent_workdir",
    [
        "/workspace/input",
        "/workspace/input/subdir",
        "/workspace/repo/subdir",
        "/workspace/skills",
        "/workspace/.gemini/extensions/nested",
        "/etc/codex/skills/nested",
        "/logs/agent/sessions/skills/nested",
        "/opt/project/.claude/commands/hidden",
        "/opt/project/.claude/skills/hidden",
        "/opt/project/.cline/skills/hidden",
        "/opt/project/.config/goose/skills/hidden",
        "/opt/project/.qwen/skills/hidden",
    ],
)
def test_generated_tasks_reject_workdir_inside_runtime_projection(tmp_path: Path, agent_workdir: str) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)

    with pytest.raises(ValueError, match=r"runtime projection|skill discovery"):
        generate_harbor_tasks(target, tmp_path / "overlapping-agent-workdir", agent_workdir=agent_workdir)


@pytest.mark.parametrize(
    "name",
    [
        "HOME",
        "USERPROFILE",
        "XDG_CONFIG_HOME",
        "CLAUDE_CONFIG_DIR",
        "CLAUDE_CODE_DISABLE_POLICY_SKILLS",
        "CODEX_HOME",
        "GEMINI_CLI_HOME",
        "OPENCODE_CONFIG_DIR",
    ],
)
def test_generated_tasks_reject_runtime_skill_discovery_environment(tmp_path: Path, name: str) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)

    with pytest.raises(ValueError, match="skill discovery"):
        generate_harbor_tasks(
            target,
            tmp_path / "runtime-discovery-env",
            runtime_env={name: "/runtime/hidden"},
        )


@pytest.mark.parametrize(
    ("environment_toml", "message"),
    [
        ('[environment.env]\nHOME = "/runtime/hidden"\n', "skill discovery"),
        ('[environment.env]\nBASH_ENV = "/runtime/seed.sh"\n', "process loader"),
        ('[environment.env]\n"BASH_FUNC_hidden%%" = "/runtime/seed.sh"\n', "process loader"),
        ('[environment.env]\nCLAUDE_CODE_DISABLE_POLICY_SKILLS = "0"\n', "skill discovery"),
        ('skills_dir = "/workspace/input"\n', "skills_dir"),
        ('docker_image = "prebuilt:latest"\n', "docker_image"),
    ],
)
def test_native_tasks_reject_runtime_projection_bypasses(tmp_path: Path, environment_toml: str, message: str) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    native_task = target / "evals" / "harbor" / "case-001"
    native_task.mkdir(parents=True)
    (native_task / "task.toml").write_text(
        'schema_version = "1.3"\n\n[task]\nname = "nvidia/case-001"\n\n'
        '[metadata]\nentry_id = "case-001"\n\n[environment]\n' + environment_toml,
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=message):
        stage_native_harbor_tasks(target, tmp_path / "native-projection-bypass", grading_mode="custom_only")


def test_native_tasks_pin_skills_dir_to_controlled_projection(tmp_path: Path) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    native_task = target / "evals" / "harbor" / "case-001"
    (native_task / "environment").mkdir(parents=True)
    (native_task / "tests").mkdir()
    (native_task / "task.toml").write_text(
        'schema_version = "1.3"\n\n[task]\nname = "nvidia/case-001"\n\n'
        '[metadata]\nentry_id = "case-001"\n\n[environment]\n',
        encoding="utf-8",
    )
    (native_task / "environment" / "Dockerfile").write_text("FROM python:3.12-slim\n", encoding="utf-8")
    (native_task / "tests" / "test.sh").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")

    staged = stage_native_harbor_tasks(
        target,
        tmp_path / "native-controlled-skills-dir",
        grading_mode="custom_only",
    )[0]

    staged_toml = (staged / "task.toml").read_text(encoding="utf-8")
    assert 'skills_dir = "/workspace/skills"' in staged_toml
    assert '"BASH_ENV" = ""' in staged_toml
    assert '"CLAUDE_CODE_DISABLE_POLICY_SKILLS" = "1"' in staged_toml


@pytest.mark.parametrize("name", ["BASH_ENV", "BASH_FUNC_hidden%%"])
def test_generated_tasks_reject_runtime_process_loader_environment(tmp_path: Path, name: str) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)

    with pytest.raises(ValueError, match="process loader"):
        generate_harbor_tasks(
            target,
            tmp_path / "runtime-loader-env",
            runtime_env={name: "/runtime/seed.sh"},
        )


def test_native_baseline_rejects_aliased_authored_skill(tmp_path: Path) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    native_task = target / "evals" / "harbor" / "case-001"
    environment = native_task / "environment"
    alias = environment / "skills" / "alias"
    alias.mkdir(parents=True)
    (alias / "SKILL.md").write_text((target / "SKILL.md").read_text(encoding="utf-8"), encoding="utf-8")
    (native_task / "instruction.md").write_text("Run the native case.\n", encoding="utf-8")
    (native_task / "task.toml").write_text(
        'schema_version = "1.3"\n\n[task]\nname = "nvidia/case-001"\n\n'
        '[metadata]\nentry_id = "case-001"\n\n[environment]\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="unmanaged skill"):
        stage_native_harbor_tasks(target, tmp_path / "native-baseline-alias", with_skill=False)


def test_native_baseline_rejects_renamed_target_manifest_payload(tmp_path: Path) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    native_task = target / "evals" / "harbor" / "case-001"
    environment = native_task / "environment"
    environment.mkdir(parents=True)
    (environment / "payload.txt").write_bytes((target / "SKILL.md").read_bytes())
    (environment / "Dockerfile").write_text(
        "FROM python:3.12-slim\nCOPY payload.txt /root/.agents/skills/alias/SKILL.md\n",
        encoding="utf-8",
    )
    (native_task / "instruction.md").write_text("Run the native case.\n", encoding="utf-8")
    (native_task / "task.toml").write_text(
        'schema_version = "1.3"\n\n[task]\nname = "nvidia/case-001"\n\n'
        '[metadata]\nentry_id = "case-001"\n\n[environment]\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="target skill instructions"):
        stage_native_harbor_tasks(target, tmp_path / "native-renamed-payload", with_skill=False)


@pytest.mark.parametrize("payload_kind", ["manifest", "renamed"])
def test_native_baseline_ignores_target_payload_in_unstaged_results(tmp_path: Path, payload_kind: str) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    _write_minimal_native_task(target)
    results = target / "evals" / "harbor" / "case-001" / "environment" / "results"
    payload = (target / "SKILL.md").read_bytes()
    if payload_kind == "manifest":
        alias = results / "alias"
        alias.mkdir(parents=True)
        (alias / "SKILL.md").write_bytes(payload)
    else:
        results.mkdir(parents=True)
        (results / "archived.txt").write_bytes(payload)

    task = stage_native_harbor_tasks(
        target,
        tmp_path / f"native-ignored-results-{payload_kind}",
        with_skill=False,
    )[0]

    assert not (task / "environment" / "results").exists()


def test_native_task_discovery_ignores_unstaged_root_results(tmp_path: Path) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    _write_minimal_native_task(target)
    archived_task = target / "evals" / "harbor" / "results"
    archived_task.mkdir()
    (archived_task / "task.toml").write_text(
        'schema_version = "1.3"\n\n[task]\nname = "nvidia/archived"\n\n'
        '[metadata]\nentry_id = "archived"\n\n[environment]\nworkdir = "relative"\n',
        encoding="utf-8",
    )

    tasks = stage_native_harbor_tasks(target, tmp_path / "native-ignored-root-results")

    assert [task.name for task in tasks] == ["case-001"]
    assert not (tmp_path / "native-ignored-root-results" / "results").exists()


def test_native_contamination_scan_uses_shared_ignore_callback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    _write_minimal_native_task(target)
    ignored = target / "evals" / "harbor" / "case-001" / "environment" / "RESULTS"
    ignored.mkdir(parents=True)
    (ignored / "archived.txt").write_bytes((target / "SKILL.md").read_bytes())

    ignored_casefolded = frozenset(name.casefold() for name in adapter_module._NATIVE_SOURCE_IGNORE_NAMES)

    def _casefolding_ignore(_directory: str, names: list[str]) -> set[str]:
        return {name for name in names if name.casefold() in ignored_casefolded}

    monkeypatch.setattr(adapter_module, "_NATIVE_SOURCE_IGNORE", _casefolding_ignore)

    task = stage_native_harbor_tasks(
        target,
        tmp_path / "native-shared-ignore-callback",
        with_skill=False,
    )[0]

    assert not (task / "environment" / "RESULTS").exists()


@pytest.mark.parametrize(
    "copy_instruction",
    [
        "COPY input/missing.txt /workspace/input/missing.txt",
        "COPY ../input/missing.txt /workspace/input/missing.txt",
        'COPY ["../input/missing.txt", "/workspace/input/missing.txt"]',
        "RUN --mount=type=bind,source=input/missing.txt,target=/payload true",
    ],
)
def test_explicit_empty_rejects_specific_custom_docker_input_child(
    tmp_path: Path,
    copy_instruction: str,
) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    (target / "evals" / "evals.json").write_text(
        json.dumps([{"id": "case-001", "question": "No fixture.", "files": []}]),
        encoding="utf-8",
    )
    (target / "evals" / "environment" / "Dockerfile").write_text(
        f"FROM python:3.12-slim\n{copy_instruction}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="specific input path"):
        generate_harbor_tasks(target, tmp_path / "specific-empty-input")


def test_explicit_empty_rejects_dynamic_custom_docker_input_source(tmp_path: Path) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    (target / "evals" / "evals.json").write_text(
        json.dumps([{"id": "case-001", "question": "No fixture.", "files": []}]),
        encoding="utf-8",
    )
    (target / "evals" / "environment" / "Dockerfile").write_text(
        "FROM python:3.12-slim\nCOPY ${SRC}/../input/ /workspace/input/\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="ambiguous input source"):
        generate_harbor_tasks(target, tmp_path / "ambiguous-empty-input")


def test_explicit_empty_rejects_unresolved_docker_parameter_expansion(tmp_path: Path) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    (target / "evals" / "evals.json").write_text(
        json.dumps([{"id": "case-001", "question": "No fixture.", "files": []}]),
        encoding="utf-8",
    )
    (target / "evals" / "environment" / "Dockerfile").write_text(
        "FROM python:3.12-slim\nARG INPUT_DIR\nCOPY ${INPUT_DIR:-input}/ /workspace/input/\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="ambiguous input source"):
        generate_harbor_tasks(target, tmp_path / "parameter-expansion-empty-input")


@pytest.mark.parametrize("declaration", ["ARG INPUT_DIR=input", "ENV INPUT_DIR=input"])
def test_explicit_empty_resolves_static_dynamic_input_directory(tmp_path: Path, declaration: str) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    (target / "evals" / "evals.json").write_text(
        json.dumps([{"id": "case-001", "question": "No fixture.", "files": []}]),
        encoding="utf-8",
    )
    (target / "evals" / "environment" / "Dockerfile").write_text(
        f"FROM python:3.12-slim\n{declaration}\nCOPY ${{INPUT_DIR}}/ /workspace/input/\n",
        encoding="utf-8",
    )

    task = generate_harbor_tasks(target, tmp_path / "static-dynamic-empty-input")[0]

    assert (task / "environment" / "input").is_dir()


def test_explicit_empty_resolves_docker_variables_at_each_instruction(tmp_path: Path) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    (target / "evals" / "evals.json").write_text(
        json.dumps([{"id": "case-001", "question": "No fixture.", "files": []}]),
        encoding="utf-8",
    )
    fixtures = target / "evals" / "environment" / "fixtures"
    fixtures.mkdir()
    (fixtures / "runtime.txt").write_text("runtime\n", encoding="utf-8")
    (target / "evals" / "environment" / "Dockerfile").write_text(
        "FROM python:3.12-slim AS first\n"
        "ARG SRC=input\n"
        "COPY ${SRC}/ /workspace/input/\n"
        "FROM first AS final\n"
        "ARG SRC=fixtures\n"
        "COPY ${SRC}/ /app/\n",
        encoding="utf-8",
    )

    task = generate_harbor_tasks(target, tmp_path / "instruction-scoped-dynamic-input")[0]

    assert (task / "environment" / "input").is_dir()


def test_explicit_empty_does_not_expose_global_arg_without_stage_redeclaration(tmp_path: Path) -> None:
    dockerfile = tmp_path / "Dockerfile"
    dockerfile.write_text(
        "ARG SRC=input\nFROM python:3.12-slim\nCOPY ${SRC}/ /workspace/input/\n",
        encoding="utf-8",
    )

    assert _dockerfile_resolved_build_context_sources(dockerfile) == [("${SRC}/", "${SRC}/", True)]


def test_explicit_empty_inherits_arg_and_env_from_parent_stage(tmp_path: Path) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    (target / "evals" / "evals.json").write_text(
        json.dumps([{"id": "case-001", "question": "No fixture.", "files": []}]),
        encoding="utf-8",
    )
    (target / "evals" / "environment" / "Dockerfile").write_text(
        "FROM python:3.12-slim AS base\n"
        "ARG ROOT=workspace\n"
        "ENV SRC=input\n"
        "FROM base AS final\n"
        "COPY ${SRC}/ /${ROOT}/input/\n",
        encoding="utf-8",
    )

    task = generate_harbor_tasks(target, tmp_path / "parent-stage-scope")

    assert (task[0] / "environment" / "input").is_dir()


def test_explicit_empty_uses_pre_instruction_env_snapshot(tmp_path: Path) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    (target / "evals" / "evals.json").write_text(
        json.dumps([{"id": "case-001", "question": "No fixture.", "files": []}]),
        encoding="utf-8",
    )
    environment = target / "evals" / "environment"
    fixtures = environment / "fixtures"
    fixtures.mkdir()
    (fixtures / "runtime.txt").write_text("runtime\n", encoding="utf-8")
    (environment / "Dockerfile").write_text(
        "FROM python:3.12-slim\nENV SRC=fixtures\nENV SRC=input COPY_ROOT=$SRC\nCOPY ${COPY_ROOT}/ /app/\n",
        encoding="utf-8",
    )

    task = generate_harbor_tasks(target, tmp_path / "env-snapshot")

    assert (task[0] / "environment" / "input").is_dir()
    assert list((task[0] / "environment" / "input").iterdir()) == []


def test_explicit_empty_does_not_retroactively_resolve_unknown_env(tmp_path: Path) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    (target / "evals" / "evals.json").write_text(
        json.dumps([{"id": "case-001", "question": "No fixture.", "files": []}]),
        encoding="utf-8",
    )
    (target / "evals" / "environment" / "Dockerfile").write_text(
        "FROM python:3.12-slim\nENV COPY_ROOT=$SRC\nENV SRC=input\nCOPY ${COPY_ROOT}/ /app/\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="ambiguous input source"):
        generate_harbor_tasks(target, tmp_path / "unknown-env-snapshot")


def test_explicit_empty_ignores_docker_heredoc_body_in_variable_scope(tmp_path: Path) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    (target / "evals" / "evals.json").write_text(
        json.dumps([{"id": "case-001", "question": "No fixture.", "files": []}]),
        encoding="utf-8",
    )
    (target / "evals" / "environment" / "Dockerfile").write_text(
        "FROM python:3.12-slim\n"
        "ARG SRC=input\n"
        "COPY <<EOF /fake-script\n"
        "ARG SRC=fixtures\n"
        "COPY fixtures/ /not-an-instruction/\n"
        "EOF\n"
        "COPY ${SRC}/ /workspace/input/\n",
        encoding="utf-8",
    )

    task = generate_harbor_tasks(target, tmp_path / "heredoc-scope")

    assert (task[0] / "environment" / "input").is_dir()


def test_punctuation_heredoc_body_cannot_change_restored_runtime_user(tmp_path: Path) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    environment = target / "evals" / "environment"
    (environment / "Dockerfile").write_text(
        "FROM python:3.12-slim\nCOPY <<'E!' /payload\nUSER 12345\nE!\n",
        encoding="utf-8",
    )

    task = generate_harbor_tasks(target, tmp_path / "punctuation-heredoc-user")[0]

    dockerfile = (task / "environment" / "Dockerfile").read_text(encoding="utf-8")
    assert not dockerfile.rstrip().endswith("USER 12345")


@pytest.mark.parametrize("source", ["'$SRC'/", r"\$SRC/", "$$SRC/"])
def test_explicit_empty_rejects_literal_or_escaped_docker_dollar_source(tmp_path: Path, source: str) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    (target / "evals" / "evals.json").write_text(
        json.dumps([{"id": "case-001", "question": "No fixture.", "files": []}]),
        encoding="utf-8",
    )
    (target / "evals" / "environment" / "Dockerfile").write_text(
        f"FROM python:3.12-slim\nARG SRC=input\nCOPY {source} /workspace/input/\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"safely parse|ambiguous"):
        generate_harbor_tasks(target, tmp_path / "literal-dollar-source")


@pytest.mark.parametrize("declaration", ["ARG COPY_ROOT='$SRC'", "ENV COPY_ROOT='$SRC'", r"ENV COPY_ROOT=\$SRC"])
def test_explicit_empty_rejects_literal_dollar_in_docker_variable_declaration(
    tmp_path: Path,
    declaration: str,
) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    (target / "evals" / "evals.json").write_text(
        json.dumps([{"id": "case-001", "question": "No fixture.", "files": []}]),
        encoding="utf-8",
    )
    (target / "evals" / "environment" / "Dockerfile").write_text(
        f"FROM python:3.12-slim\nARG SRC=input\n{declaration}\nCOPY ${{COPY_ROOT}}/ /workspace/input/\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match=r"quoted or escaped Dockerfile"):
        generate_harbor_tasks(target, tmp_path / "literal-dollar-declaration")


def test_explicit_empty_honors_deferred_onbuild_copy(tmp_path: Path) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    (target / "evals" / "evals.json").write_text(
        json.dumps([{"id": "case-001", "question": "No fixture.", "files": []}]),
        encoding="utf-8",
    )
    (target / "evals" / "environment" / "Dockerfile").write_text(
        "FROM python:3.12-slim AS base\n"
        "ONBUILD COPY input/ /workspace/input/\n"
        "FROM base AS final\n"
        "RUN test -d /workspace/input\n",
        encoding="utf-8",
    )

    task = generate_harbor_tasks(target, tmp_path / "onbuild-input")

    assert (task[0] / "environment" / "input").is_dir()


def test_quoted_shell_text_is_not_treated_as_docker_heredoc(tmp_path: Path) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    (target / "evals" / "evals.json").write_text(
        json.dumps([{"id": "case-001", "question": "No fixture.", "files": []}]),
        encoding="utf-8",
    )
    (target / "evals" / "environment" / "Dockerfile").write_text(
        "FROM python:3.12-slim\nRUN echo 'x <<EOF y'\nCOPY input/ /workspace/input/\n",
        encoding="utf-8",
    )

    task = generate_harbor_tasks(target, tmp_path / "quoted-heredoc-text")

    assert (task[0] / "environment" / "input").is_dir()


def test_explicit_empty_reads_compose_yml_build_arg_override(tmp_path: Path) -> None:
    environment = tmp_path / "environment"
    environment.mkdir()
    dockerfile = environment / "Dockerfile"
    dockerfile.write_text(
        "FROM python:3.12-slim\nARG SRC=fixtures\nCOPY ${SRC}/ /workspace/input/\n",
        encoding="utf-8",
    )
    (environment / "docker-compose.yml").write_text(
        "services:\n  main:\n    build:\n      context: .\n      args:\n        SRC: input\n",
        encoding="utf-8",
    )

    _ensure_empty_custom_docker_input_compatibility(environment, dockerfile, has_input=False)

    assert (environment / "input").is_dir()


def test_explicit_empty_allows_broad_custom_docker_wildcard(tmp_path: Path) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    (target / "evals" / "evals.json").write_text(
        json.dumps([{"id": "case-001", "question": "No fixture.", "files": []}]),
        encoding="utf-8",
    )
    (target / "evals" / "environment" / "app.txt").write_text("runtime asset\n", encoding="utf-8")
    (target / "evals" / "environment" / "Dockerfile").write_text(
        "FROM python:3.12-slim\nCOPY * /app/\n",
        encoding="utf-8",
    )

    assert generate_harbor_tasks(target, tmp_path / "broad-wildcard")


def test_explicit_empty_allows_unrelated_nested_input_glob(tmp_path: Path) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    (target / "evals" / "evals.json").write_text(
        json.dumps([{"id": "case-001", "question": "No fixture.", "files": []}]),
        encoding="utf-8",
    )
    fixtures = target / "evals" / "environment" / "fixtures"
    fixtures.mkdir()
    (fixtures / "myinput-data.txt").write_text("runtime asset\n", encoding="utf-8")
    (target / "evals" / "environment" / "Dockerfile").write_text(
        "FROM python:3.12-slim\nCOPY fixtures/myinput*.txt /app/\n",
        encoding="utf-8",
    )

    task = generate_harbor_tasks(target, tmp_path / "unrelated-input-glob")
    assert task


@pytest.mark.parametrize(
    ("declaration", "source", "source_dir"),
    [
        ("ARG APP_DIR=fixtures", "${APP_DIR}/", "fixtures"),
        ("ARG ARCH=amd64", "fixtures/${ARCH}/", "fixtures/amd64"),
    ],
)
def test_explicit_empty_allows_unrelated_dynamic_custom_docker_source(
    tmp_path: Path,
    declaration: str,
    source: str,
    source_dir: str,
) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    (target / "evals" / "evals.json").write_text(
        json.dumps([{"id": "case-001", "question": "No fixture.", "files": []}]),
        encoding="utf-8",
    )
    runtime_source = target / "evals" / "environment" / source_dir
    runtime_source.mkdir(parents=True)
    (runtime_source / "runtime.txt").write_text("runtime asset\n", encoding="utf-8")
    (target / "evals" / "environment" / "Dockerfile").write_text(
        f"FROM python:3.12-slim\n{declaration}\nCOPY {source} /app/\n",
        encoding="utf-8",
    )

    assert generate_harbor_tasks(target, tmp_path / "unrelated-dynamic-source")


@pytest.mark.parametrize("escape_char", ["\\", "`"])
def test_explicit_empty_honors_custom_docker_escape_directive(tmp_path: Path, escape_char: str) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    (target / "evals" / "evals.json").write_text(
        json.dumps([{"id": "case-001", "question": "No fixture.", "files": []}]),
        encoding="utf-8",
    )
    (target / "evals" / "environment" / "Dockerfile").write_text(
        f"# escape={escape_char}\n"
        "FROM python:3.12-slim\n"
        f"COPY {escape_char}\n"
        f"  input/ {escape_char}\n"
        "  /workspace/input/\n",
        encoding="utf-8",
    )

    task = generate_harbor_tasks(target, tmp_path / f"escape-{ord(escape_char)}")[0]

    assert (task / "environment" / "input").is_dir()


def _write_explicit_empty_entry(target: Path) -> None:
    (target / "evals" / "evals.json").write_text(
        json.dumps([{"id": "case-001", "question": "No fixture.", "files": []}]),
        encoding="utf-8",
    )


def test_explicit_empty_materializes_compose_additional_input_context(tmp_path: Path) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    _write_explicit_empty_entry(target)
    environment = target / "evals" / "environment"
    (environment / "Dockerfile").write_text("FROM python:3.12-slim\n", encoding="utf-8")
    (environment / "docker-compose.yaml").write_text(
        "services:\n"
        "  helper:\n"
        "    build:\n"
        "      context: .\n"
        "      dockerfile_inline: |\n"
        "        FROM scratch\n"
        "        COPY --from=taskinput . /payload/\n"
        "      additional_contexts:\n"
        "        taskinput: input\n",
        encoding="utf-8",
    )

    task = generate_harbor_tasks(target, tmp_path / "compose-additional-input")[0]

    assert (task / "environment" / "input").is_dir()
    assert list((task / "environment" / "input").iterdir()) == []


def test_explicit_empty_materializes_compose_primary_input_context(tmp_path: Path) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    _write_explicit_empty_entry(target)
    environment = target / "evals" / "environment"
    (environment / "Dockerfile").write_text("FROM python:3.12-slim\n", encoding="utf-8")
    (environment / "docker-compose.yaml").write_text(
        "services:\n"
        "  helper:\n"
        "    build:\n"
        "      context: input\n"
        "      dockerfile_inline: |\n"
        "        FROM scratch\n"
        "        COPY . /payload/\n",
        encoding="utf-8",
    )

    task = generate_harbor_tasks(target, tmp_path / "compose-primary-input")[0]

    assert (task / "environment" / "input").is_dir()
    assert list((task / "environment" / "input").iterdir()) == []


def test_explicit_empty_rejects_specific_source_from_compose_primary_input_context(tmp_path: Path) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    _write_explicit_empty_entry(target)
    environment = target / "evals" / "environment"
    (environment / "Dockerfile").write_text("FROM python:3.12-slim\n", encoding="utf-8")
    (environment / "docker-compose.yaml").write_text(
        "services:\n"
        "  helper:\n"
        "    build:\n"
        "      context: input\n"
        "      dockerfile_inline: |\n"
        "        FROM scratch\n"
        "        COPY missing.txt /payload/\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="specific path from the input build context"):
        generate_harbor_tasks(target, tmp_path / "compose-primary-specific-source")


@pytest.mark.parametrize(
    "instruction",
    [
        "COPY --from=taskinput missing.txt /payload/",
        "RUN --mount=type=bind,from=taskinput,source=missing.txt,target=/payload true",
    ],
)
def test_explicit_empty_rejects_specific_source_from_named_input_context(
    tmp_path: Path,
    instruction: str,
) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    _write_explicit_empty_entry(target)
    environment = target / "evals" / "environment"
    (environment / "Dockerfile").write_text("FROM python:3.12-slim\n", encoding="utf-8")
    (environment / "docker-compose.yaml").write_text(
        "services:\n"
        "  helper:\n"
        "    build:\n"
        "      context: .\n"
        "      dockerfile_inline: |\n"
        "        FROM scratch\n"
        f"        {instruction}\n"
        "      additional_contexts:\n"
        "        taskinput: input\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="specific path from an input context"):
        generate_harbor_tasks(target, tmp_path / "compose-named-specific-source")


def test_explicit_empty_resolves_compose_escaped_inline_arg(tmp_path: Path) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    _write_explicit_empty_entry(target)
    environment = target / "evals" / "environment"
    (environment / "Dockerfile").write_text("FROM python:3.12-slim\n", encoding="utf-8")
    (environment / "docker-compose.yaml").write_text(
        "services:\n"
        "  helper:\n"
        "    build:\n"
        "      context: .\n"
        "      args:\n"
        "        SRC: input\n"
        "      dockerfile_inline: |\n"
        "        FROM scratch\n"
        "        ARG SRC\n"
        "        COPY $${SRC}/ /payload/\n",
        encoding="utf-8",
    )

    task = generate_harbor_tasks(target, tmp_path / "compose-inline-arg")[0]

    assert (task / "environment" / "input").is_dir()


def test_explicit_empty_rejects_specific_compose_input_context(tmp_path: Path) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    _write_explicit_empty_entry(target)
    environment = target / "evals" / "environment"
    (environment / "Dockerfile").write_text("FROM python:3.12-slim\n", encoding="utf-8")
    (environment / "docker-compose.yaml").write_text(
        "services:\n"
        "  helper:\n"
        "    build:\n"
        "      context: .\n"
        "      dockerfile_inline: 'FROM scratch'\n"
        "      additional_contexts:\n"
        "        taskinput: input/missing\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="specific input context"):
        generate_harbor_tasks(target, tmp_path / "compose-specific-input")


def test_compose_rejects_build_with_global_image_tag(tmp_path: Path) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    environment = target / "evals" / "environment"
    (environment / "Dockerfile").write_text("FROM python:3.12-slim\n", encoding="utf-8")
    (environment / "docker-compose.yaml").write_text(
        "services:\n  helper:\n    image: trusted-global-name:latest\n    build:\n      context: sidecar\n",
        encoding="utf-8",
    )
    (environment / "sidecar" / "Dockerfile").write_text("FROM scratch\n", encoding="utf-8")

    with pytest.raises(ValueError, match="image together with build"):
        generate_harbor_tasks(target, tmp_path / "compose-image-poisoning")


def test_compose_rejects_non_string_mapping_keys(tmp_path: Path) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    environment = target / "evals" / "environment"
    (environment / "Dockerfile").write_text("FROM python:3.12-slim\n", encoding="utf-8")
    (environment / "docker-compose.yaml").write_text(
        "1: unsafe\nservices:\n  helper:\n    image: alpine:3.20\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="keys must be strings"):
        generate_harbor_tasks(target, tmp_path / "compose-non-string-key")


def test_rebase_accepts_tab_from_and_preserves_alias_and_runtime_user(tmp_path: Path) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    environment = target / "evals" / "environment"
    (environment / "Dockerfile").write_text(
        "FROM\tpython:3.12-slim AS app\nUSER 12345\n",
        encoding="utf-8",
    )

    task = generate_harbor_tasks(
        target,
        tmp_path / "rebase-tab-from",
        base_image="skillevaluator-base:test",
        custom_dockerfile_mode="rebase",
    )[0]

    dockerfile = (task / "environment" / "Dockerfile").read_text(encoding="utf-8")
    assert dockerfile.startswith("FROM\tskillevaluator-base:test AS app\n")
    assert dockerfile.rstrip().endswith("USER 12345")


def test_rebase_rejects_multi_stage_dockerfile(tmp_path: Path) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    (target / "evals" / "environment" / "Dockerfile").write_text(
        "FROM python:3.12-slim AS build\nFROM build AS final\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="exactly one FROM"):
        generate_harbor_tasks(
            target,
            tmp_path / "rebase-multi-stage",
            base_image="skillevaluator-base:test",
            custom_dockerfile_mode="rebase",
        )


def test_rebase_rejects_continued_from_instruction(tmp_path: Path) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    (target / "evals" / "environment" / "Dockerfile").write_text(
        "FROM \\\n  python:3.12-slim\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="continued FROM"):
        generate_harbor_tasks(
            target,
            tmp_path / "rebase-continued-from",
            base_image="skillevaluator-base:test",
            custom_dockerfile_mode="rebase",
        )


def test_custom_projection_uses_exec_runs_and_ignores_comment_decoys(tmp_path: Path) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    environment = target / "evals" / "environment"
    (environment / "Dockerfile").write_text(
        "FROM python:3.12-slim\n"
        'SHELL ["python", "-c"]\n'
        "# ragas langchain anthropic\n"
        "# COPY skills/ /root/.agents/skills/\n"
        "# RUN rm -rf /workspace/input && mkdir -p /workspace/input\n"
        "USER 12345\n",
        encoding="utf-8",
    )

    task = generate_harbor_tasks(target, tmp_path / "custom-exec-projection")[0]

    dockerfile = (task / "environment" / "Dockerfile").read_text(encoding="utf-8")
    assert 'RUN ["/bin/sh", "-c"' in dockerfile
    assert '"python", "-m", "pip", "install"' in dockerfile
    assert 'RUN ["/bin/rm", "-rf", "/workspace/input"]' in dockerfile
    assert dockerfile.rstrip().endswith("USER 12345")


def test_preserve_projection_restores_user_inherited_from_authored_stage(tmp_path: Path) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    environment = target / "evals" / "environment"
    (environment / "Dockerfile").write_text(
        "FROM python:3.12-slim AS base\nUSER 12345\nFROM base AS final\n",
        encoding="utf-8",
    )

    task = generate_harbor_tasks(target, tmp_path / "inherited-user-projection")[0]

    dockerfile = (task / "environment" / "Dockerfile").read_text(encoding="utf-8")
    assert "USER root\n" in dockerfile
    assert dockerfile.rstrip().endswith("USER 12345")


def test_managed_projection_survives_authored_path_override(tmp_path: Path) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    environment = target / "evals" / "environment"
    (environment / "Dockerfile").write_text(
        "FROM python:3.12-slim\nENV PATH=/custom\n",
        encoding="utf-8",
    )

    task = generate_harbor_tasks(target, tmp_path / "managed-path-projection")[0]

    dockerfile = (task / "environment" / "Dockerfile").read_text(encoding="utf-8")
    assert 'PATH=\\"${PATH:+$PATH:}/usr/local/bin:/usr/bin:/bin\\"' in dockerfile
    assert 'RUN ["/bin/rm", "-rf", "/workspace/input"]' in dockerfile


def test_verifier_constraint_and_skill_requirements_share_resolver_transaction(tmp_path: Path) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    (target / "scripts" / "requirements.txt").write_text(
        "langchain-community>=0.4.2\n",
        encoding="utf-8",
    )

    task = generate_harbor_tasks(target, tmp_path / "verifier-requirement-conflict")[0]

    dockerfile = (task / "environment" / "Dockerfile").read_text(encoding="utf-8")
    install_line = next(line for line in dockerfile.splitlines() if '"pip", "install"' in line)
    assert "langchain-community<0.4.2" in install_line
    assert "langchain-community>=0.4.2" in install_line
    assert "idna>=3.10,<4" in install_line
    assert dockerfile.count('"pip", "install"') == 1
    assert "from ragas.llms.base import llm_factory" in dockerfile


def test_rebase_reasserts_verifier_constraint_after_authored_dependencies(tmp_path: Path) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    environment = target / "evals" / "environment"
    (environment / "Dockerfile").write_text(
        "FROM python:3.12-slim\nRUN python -m pip install 'langchain-community>=0.4.2'\n",
        encoding="utf-8",
    )

    task = generate_harbor_tasks(
        target,
        tmp_path / "rebase-verifier-constraint",
        base_image="skillevaluator-base:test",
        custom_dockerfile_mode="rebase",
    )[0]

    dockerfile = (task / "environment" / "Dockerfile").read_text(encoding="utf-8")
    assert dockerfile.rfind("langchain-community<0.4.2") > dockerfile.find("langchain-community>=0.4.2")
    assert dockerfile.rfind("from ragas.llms.base import llm_factory") > dockerfile.find("langchain-community>=0.4.2")


def test_baseline_rejects_target_manifest_payload_with_appended_comment(tmp_path: Path) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    environment = target / "evals" / "environment"
    payload = (target / "SKILL.md").read_text(encoding="utf-8") + "\n# harmless comment\n"
    (environment / "payload.txt").write_text(payload, encoding="utf-8")
    (environment / "Dockerfile").write_text(
        "FROM python:3.12-slim\nCOPY payload.txt /root/.agents/skills/alias/SKILL.md\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="target skill instructions"):
        generate_harbor_tasks(target, tmp_path / "baseline-appended-payload", with_skill=False)


def test_linked_repo_projection_rejects_reserved_workspace_input(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "SKILLEVALUATOR_OUTPUT_PROVENANCE_KEY_FILE",
        str(tmp_path.parent / f"{tmp_path.name}-state" / "output-provenance.key"),
    )
    repo, target, _, _ = _write_projection_fixture(tmp_path)
    (tmp_path / ".git").mkdir()
    outside_input = tmp_path / "input"
    outside_input.mkdir()
    (outside_input / "hidden.txt").write_text("RESERVED-INPUT-ORACLE\n", encoding="utf-8")
    manifest = target / "SKILL.md"
    manifest.write_text(
        manifest.read_text(encoding="utf-8") + "\n[hidden](../../input/hidden.txt)\n",
        encoding="utf-8",
    )
    assert repo.is_dir()

    with pytest.raises(ValueError, match="reserved runtime path '/workspace/input'"):
        generate_harbor_tasks(target, tmp_path / "reserved-linked-input", copy_repo=False)


@pytest.mark.parametrize(
    ("linked_path", "linked_target"),
    [
        ("skills/target-skill/SKILL.md", "../../skills/target-skill/SKILL.md"),
        (".claude/skills/hidden/SKILL.md", "../../.claude/skills/hidden/SKILL.md"),
        (".cline/skills/hidden/SKILL.md", "../../.cline/skills/hidden/SKILL.md"),
        (".config/goose/skills/hidden/SKILL.md", "../../.config/goose/skills/hidden/SKILL.md"),
        (".qwen/skills/hidden/SKILL.md", "../../.qwen/skills/hidden/SKILL.md"),
    ],
)
def test_linked_repo_cannot_overlay_agent_discovery_roots(
    tmp_path: Path,
    linked_path: str,
    linked_target: str,
) -> None:
    repo, original_target, references, workspace = _write_projection_fixture(tmp_path)
    target = repo / "packages" / original_target.name
    target.parent.mkdir()
    shutil.move(original_target, target)
    (repo / ".git").mkdir()
    hidden_manifest = repo / linked_path
    hidden_manifest.parent.mkdir(parents=True, exist_ok=True)
    hidden_manifest.write_text("HIDDEN-DISCOVERY-ORACLE\n", encoding="utf-8")
    manifest = target / "SKILL.md"
    manifest.write_text(
        manifest.read_text(encoding="utf-8") + f"\n[hidden]({linked_target})\n",
        encoding="utf-8",
    )

    task = generate_harbor_tasks(
        target,
        tmp_path / f"linked-discovery-{hidden_manifest.parent.name}",
        reference_skills_dir=references,
        workspace_skill_paths=[workspace],
        copy_repo=False,
    )[0]

    linked_root = task / "environment" / "repo-linked-root"
    staged_text = (
        "\n".join(
            path.read_text(encoding="utf-8", errors="ignore") for path in linked_root.rglob("*") if path.is_file()
        )
        if linked_root.exists()
        else ""
    )
    assert "HIDDEN-DISCOVERY-ORACLE" not in staged_text


@pytest.mark.skipif(os.name != "nt", reason="requires Windows junctions")
@pytest.mark.parametrize("location", ["runtime", "input"])
def test_windows_junction_sources_are_rejected(tmp_path: Path, location: str) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    outside = tmp_path / "outside-junction"
    outside.mkdir()
    (outside / "hidden.txt").write_text("JUNCTION-ORACLE\n", encoding="utf-8")
    if location == "runtime":
        junction = target / "references" / "junction"
    else:
        junction = target / "evals" / "files" / "junction"
        (target / "evals" / "evals.json").write_text(
            json.dumps([{"id": "case-001", "question": "No explicit files."}]),
            encoding="utf-8",
        )
    subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(junction), str(outside)],
        check=True,
        capture_output=True,
        text=True,
    )

    with pytest.raises(ValueError, match=r"symlink|reparse|junction"):
        generate_harbor_tasks(target, tmp_path / f"windows-junction-{location}")


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "fifo"])
def test_eval_dataset_file_must_be_regular_and_unlinked(tmp_path: Path, kind: str) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    dataset = target / "evals" / "evals.json"
    dataset.unlink()
    outside = tmp_path / "outside-evals.json"
    outside.write_text('[{"id":"case-001","question":"ORACLE"}]\n', encoding="utf-8")
    try:
        if kind == "symlink":
            dataset.symlink_to(outside)
        elif kind == "hardlink":
            dataset.hardlink_to(outside)
        elif hasattr(os, "mkfifo"):
            os.mkfifo(dataset)
        else:
            pytest.skip("FIFO creation is unavailable")
    except OSError as exc:
        pytest.skip(f"{kind} creation is unavailable: {exc}")

    with pytest.raises(ValueError, match="regular non-linked file"):
        generate_harbor_tasks(target, tmp_path / f"unsafe-dataset-{kind}")


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "fifo"])
def test_selective_evaluator_snapshot_does_not_copy_unselected_unsafe_fixture(
    tmp_path: Path,
    kind: str,
) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    files = target / "evals" / "files"
    unselected = files / "unselected-unsafe"
    outside = tmp_path / "outside-unselected.txt"
    outside.write_text("must not be inspected or copied\n", encoding="utf-8")
    try:
        if kind == "symlink":
            unselected.symlink_to(outside)
        elif kind == "hardlink":
            unselected.hardlink_to(outside)
        elif hasattr(os, "mkfifo"):
            os.mkfifo(unselected)
        else:
            pytest.skip("FIFO creation is unavailable")
    except OSError as exc:
        pytest.skip(f"{kind} creation is unavailable: {exc}")

    # A sparse large sibling models the ENOSPC/performance regression without
    # making the test consume the corresponding amount of disk.
    (files / "unselected-large.bin").touch()
    os.truncate(files / "unselected-large.bin", 256 * 1024 * 1024)

    with private_evaluator_skill_snapshot(target, task_source="evals_json") as snapshot:
        projected_files = snapshot / "evals" / "files"
        assert (projected_files / "selected.txt").read_text(encoding="utf-8") == "SELECTED-FIXTURE\n"
        assert not (projected_files / "hidden.txt").exists()
        assert not os.path.lexists(projected_files / "unselected-unsafe")
        assert not (projected_files / "unselected-large.bin").exists()


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "fifo"])
def test_selective_evaluator_snapshot_fails_closed_for_referenced_unsafe_fixture(
    tmp_path: Path,
    kind: str,
) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    files = target / "evals" / "files"
    selected = files / "selected.txt"
    selected.unlink()
    outside = tmp_path / "outside-selected.txt"
    outside.write_text("must never be copied\n", encoding="utf-8")
    try:
        if kind == "symlink":
            selected.symlink_to(outside)
        elif kind == "hardlink":
            selected.hardlink_to(outside)
        elif hasattr(os, "mkfifo"):
            os.mkfifo(selected)
        else:
            pytest.skip("FIFO creation is unavailable")
    except OSError as exc:
        pytest.skip(f"{kind} creation is unavailable: {exc}")

    with (
        pytest.raises((FileNotFoundError, ValueError), match=r"outside evals|symlink|hard.?link|regular file"),
        private_evaluator_skill_snapshot(target, task_source="evals_json"),
    ):
        pass


def test_selective_evaluator_snapshot_preserves_legacy_implicit_files_corpus(tmp_path: Path) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    (target / "evals" / "evals.json").write_text(
        json.dumps([{"id": "case-001", "question": "Use the shared fixtures."}]),
        encoding="utf-8",
    )

    with private_evaluator_skill_snapshot(target, task_source="evals_json") as snapshot:
        projected_files = snapshot / "evals" / "files"
        assert (projected_files / "selected.txt").is_file()
        assert (projected_files / "hidden.txt").is_file()


def test_native_snapshot_ignores_generated_only_entries_when_selecting_fixtures(tmp_path: Path) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    _write_minimal_native_task(target)
    evals = target / "evals"
    (evals / "evals.json").write_text(
        json.dumps(
            [
                {
                    "id": "case-001",
                    "question": "Use only the selected fixture.",
                    "files": ["evals/files/selected.txt"],
                },
                {
                    "id": "generated-only",
                    "question": "This entry has legacy implicit shared files.",
                },
            ]
        ),
        encoding="utf-8",
    )
    unsafe = evals / "files" / "unselected-unsafe"
    outside = tmp_path / "outside-native-fixtures.txt"
    outside.write_text("must not be inspected or copied\n", encoding="utf-8")
    try:
        unsafe.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")

    with private_evaluator_skill_snapshot(target, task_source="native_harbor") as snapshot:
        projected_files = snapshot / "evals" / "files"
        assert (projected_files / "selected.txt").read_text(encoding="utf-8") == "SELECTED-FIXTURE\n"
        assert not (projected_files / "hidden.txt").exists()
        assert not os.path.lexists(projected_files / "unselected-unsafe")

    task = stage_native_harbor_tasks(target, tmp_path / "native-selected-fixtures", with_skill=False)[0]
    staged_input = task / "environment" / "input"
    assert (staged_input / "selected.txt").read_text(encoding="utf-8") == "SELECTED-FIXTURE\n"
    assert not (staged_input / "hidden.txt").exists()
    assert not os.path.lexists(staged_input / "unselected-unsafe")


def test_evidence_snapshot_binds_complete_files_corpus_for_explicit_entries(tmp_path: Path) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)

    with private_evaluator_skill_snapshot(
        target,
        task_source="evals_json",
        bind_full_evidence_sources=True,
    ) as snapshot:
        projected_files = snapshot / "evals" / "files"
        assert (projected_files / "selected.txt").is_file()
        assert (projected_files / "hidden.txt").is_file()


def test_selective_snapshot_ignores_shadowed_grader_but_evidence_binds_it(tmp_path: Path) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    shadowed = target / "evals" / "tests" / "grader.py"
    shadowed.parent.mkdir()
    outside = tmp_path / "outside-shadowed-grader.py"
    outside.write_text("raise RuntimeError('must not run')\n", encoding="utf-8")
    try:
        shadowed.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable: {exc}")

    with private_evaluator_skill_snapshot(target, task_source="evals_json") as snapshot:
        assert (snapshot / "evals" / "grader.py").is_file()
        assert not os.path.lexists(snapshot / "evals" / "tests" / "grader.py")

    with (
        pytest.raises(ValueError, match="symlink"),
        private_evaluator_skill_snapshot(
            target,
            task_source="evals_json",
            bind_full_evidence_sources=True,
        ),
    ):
        pass


def test_evidence_snapshot_rejects_unselected_unsafe_file_it_must_bind(tmp_path: Path) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    hidden = target / "evals" / "files" / "hidden.txt"
    hidden.unlink()
    outside = tmp_path / "evidence-hardlink.txt"
    outside.write_text("bound evidence\n", encoding="utf-8")
    try:
        hidden.hardlink_to(outside)
    except OSError as exc:
        pytest.skip(f"hardlinks unavailable: {exc}")

    with (
        pytest.raises(ValueError, match=r"hard.?link"),
        private_evaluator_skill_snapshot(
            target,
            task_source="evals_json",
            bind_full_evidence_sources=True,
        ),
    ):
        pass


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "fifo"])
def test_selective_evaluator_snapshot_ignores_unsafe_unconsumed_evals_subtree(
    tmp_path: Path,
    kind: str,
) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    archive = target / "evals" / "historical-archive"
    archive.mkdir()
    unsafe = archive / "unsafe"
    outside = tmp_path / "outside-archive.txt"
    outside.write_text("unconsumed\n", encoding="utf-8")
    try:
        if kind == "symlink":
            unsafe.symlink_to(outside)
        elif kind == "hardlink":
            unsafe.hardlink_to(outside)
        elif hasattr(os, "mkfifo"):
            os.mkfifo(unsafe)
        else:
            pytest.skip("FIFO creation is unavailable")
    except OSError as exc:
        pytest.skip(f"{kind} creation is unavailable: {exc}")

    with private_evaluator_skill_snapshot(target, task_source="evals_json") as snapshot:
        assert not (snapshot / "evals" / "historical-archive").exists()


def test_selective_evaluator_snapshot_rejects_fixture_selection_change_during_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    dataset = target / "evals" / "evals.json"
    original_copytree = adapter_module.copytree_secure
    mutated = False

    def mutate_selection_then_copy(source: Path, destination: Path, *args: object, **kwargs: object) -> None:
        nonlocal mutated
        if Path(source) == target / "evals" and not mutated:
            dataset.write_text(
                json.dumps([{"id": "case-001", "question": "Now use every shared fixture."}]),
                encoding="utf-8",
            )
            mutated = True
        original_copytree(source, destination, *args, **kwargs)

    monkeypatch.setattr(adapter_module, "copytree_secure", mutate_selection_then_copy)

    with (
        pytest.raises(ValueError, match="selection changed"),
        private_evaluator_skill_snapshot(target, task_source="evals_json"),
    ):
        pass

    assert mutated


@pytest.mark.parametrize(
    ("task_source", "selected_relative"),
    [
        ("evals_json", Path("files/selected.txt")),
        ("evals_json", Path("grader.py")),
        ("evals_json", Path("environment/sidecar/seed.txt")),
        ("native_harbor", Path("harbor/case-001/instruction.md")),
    ],
)
def test_selective_evaluator_snapshot_rejects_selected_content_change_during_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    task_source: str,
    selected_relative: Path,
) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    if task_source == "native_harbor":
        _write_minimal_native_task(target)
    selected_path = target / "evals" / selected_relative
    original_copytree = adapter_module.copytree_secure
    mutated = False

    def mutate_content_then_copy(source: Path, destination: Path, *args: object, **kwargs: object) -> None:
        nonlocal mutated
        if Path(source) == target / "evals" and not mutated:
            selected_path.write_text(selected_path.read_text(encoding="utf-8") + "MUTATED\n", encoding="utf-8")
            mutated = True
        original_copytree(source, destination, *args, **kwargs)

    monkeypatch.setattr(adapter_module, "copytree_secure", mutate_content_then_copy)

    with (
        pytest.raises(ValueError, match="selection changed"),
        private_evaluator_skill_snapshot(target, task_source=task_source),
    ):
        pass

    assert mutated


@pytest.mark.skipif(os.name != "posix", reason="requires POSIX executable mode semantics")
def test_selective_evaluator_snapshot_rejects_selected_mode_change_during_copy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    script = target / "evals" / "environment" / "run.sh"
    script.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    script.chmod(0o644)
    original_copytree = adapter_module.copytree_secure
    mutated = False

    def mutate_mode_then_copy(source: Path, destination: Path, *args: object, **kwargs: object) -> None:
        nonlocal mutated
        if Path(source) == target / "evals" and not mutated:
            script.chmod(0o755)
            mutated = True
        original_copytree(source, destination, *args, **kwargs)

    monkeypatch.setattr(adapter_module, "copytree_secure", mutate_mode_then_copy)

    with (
        pytest.raises(ValueError, match="selection changed"),
        private_evaluator_skill_snapshot(target, task_source="evals_json"),
    ):
        pass

    assert mutated


def test_evaluator_read_accepts_windows_crt_descriptor_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evals = tmp_path / "evals"
    evals.mkdir()
    dataset = evals / "evals.json"
    payload = b"[]\n"
    dataset.write_bytes(payload)
    original_read_bytes = adapter_module.SecureRoot.read_bytes

    def read_with_windows_crt_identity(
        secure_root: adapter_module.SecureRoot,
        relative_path: Path,
        max_bytes: int,
        *,
        expected: os.stat_result | None = None,
    ) -> tuple[bytes, object]:
        raw, opened = original_read_bytes(secure_root, relative_path, max_bytes, expected=expected)
        return raw, SimpleNamespace(
            st_dev=opened.st_dev + 10_000,
            st_ino=opened.st_ino + 10_000,
            st_mode=opened.st_mode,
            st_nlink=opened.st_nlink,
            st_size=opened.st_size,
            st_mtime_ns=opened.st_mtime_ns,
            st_ctime_ns=opened.st_ctime_ns,
        )

    monkeypatch.setattr(adapter_module, "_PATH_DESCRIPTOR_IDENTITIES_COMPARABLE", False, raising=False)
    monkeypatch.setattr(adapter_module.SecureRoot, "read_bytes", read_with_windows_crt_identity)

    assert adapter_module._read_regular_evals_file(dataset, allowed_root=evals) == payload


@pytest.mark.parametrize("task_source", ["evals_json", "native_harbor"])
def test_baseline_alias_candidates_are_scanned_once_per_multi_case_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    task_source: str,
) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    entries = [
        {
            "id": f"case-{index:03d}",
            "question": f"Run case {index}.",
            "files": ["evals/files/selected.txt"],
        }
        for index in range(1, 4)
    ]
    (target / "evals" / "evals.json").write_text(json.dumps(entries), encoding="utf-8")
    if task_source == "native_harbor":
        for entry in entries:
            native_task = target / "evals" / "harbor" / entry["id"]
            native_task.mkdir(parents=True)
            (native_task / "instruction.md").write_text("Run the native case.\n", encoding="utf-8")
            (native_task / "task.toml").write_text(
                'schema_version = "1.3"\n\n[task]\n'
                f'name = "nvidia/{entry["id"]}"\n\n'
                f'[metadata]\nentry_id = "{entry["id"]}"\n\n[environment]\n',
                encoding="utf-8",
            )

    scans = 0

    def count_scan(*args: object, **kwargs: object) -> None:
        nonlocal scans
        scans += 1

    monkeypatch.setattr(adapter_module, "_check_baseline_skill_candidates_do_not_alias_target", count_scan)
    stager = generate_harbor_tasks if task_source == "evals_json" else stage_native_harbor_tasks

    tasks = stager(target, tmp_path / f"{task_source}-scan-count", with_skill=False)

    assert len(tasks) == 3
    assert scans == 1


def test_run_scoped_baseline_alias_validation_skips_rescan_and_is_path_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, target, references_dir, workspace = _write_projection_fixture(tmp_path)
    excluded_root = target / "evals" / "results"
    validation = adapter_module._prevalidate_baseline_skill_candidates(
        target,
        references_dir,
        [workspace],
        excluded_roots=(excluded_root,),
    )

    def reject_rescan(*args: object, **kwargs: object) -> None:
        raise AssertionError("run-scoped validation should suppress repeated source scans")

    monkeypatch.setattr(adapter_module, "_check_baseline_skill_candidates_do_not_alias_target", reject_rescan)
    tasks = generate_harbor_tasks(
        target,
        tmp_path / "validated-baseline",
        with_skill=False,
        reference_skills_dir=references_dir,
        workspace_skill_paths=[workspace],
        repo_context_exclude_paths=(excluded_root,),
        _baseline_alias_validation=validation,
    )

    assert len(tasks) == 1
    with pytest.raises(ValueError, match="does not match"):
        generate_harbor_tasks(
            target,
            tmp_path / "mismatched-validation",
            with_skill=False,
            reference_skills_dir=references_dir,
            workspace_skill_paths=[],
            repo_context_exclude_paths=(excluded_root,),
            _baseline_alias_validation=validation,
        )


def test_in_repo_temp_evaluator_snapshot_is_excluded_from_copy_repo(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo, target, _, _ = _write_projection_fixture(tmp_path)
    in_repo_temp = repo / "in-repo-temp"
    in_repo_temp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(in_repo_temp))

    task = generate_harbor_tasks(
        target,
        tmp_path / "in-repo-temp-output",
        with_skill=True,
        copy_repo=True,
    )[0]

    staged_repo = task / "environment" / "repo"
    assert staged_repo.is_dir()
    assert not list(staged_repo.rglob("evals.json"))
    readable = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore") for path in staged_repo.rglob("*") if path.is_file()
    )
    assert "GROUND-TRUTH-SECRET" not in readable


def test_temp_snapshot_inside_legacy_files_does_not_recurse_into_itself(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    (target / "evals" / "evals.json").write_text(
        json.dumps([{"id": "case-001", "question": "Use every shared fixture."}]),
        encoding="utf-8",
    )
    nested_temp = target / "evals" / "files" / "nested-temp-root"
    nested_temp.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(nested_temp))

    task = generate_harbor_tasks(target, tmp_path / "nested-temp-output", with_skill=False)[0]

    staged_input = task / "environment" / "input"
    assert (staged_input / "selected.txt").is_file()
    assert (staged_input / "hidden.txt").is_file()
    assert not list(staged_input.rglob("evals.json"))


def test_native_projection_does_not_enumerate_symlinked_harbor_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    outside_skill = tmp_path / "outside-skill"
    _write_minimal_native_task(outside_skill)
    outside = outside_skill / "evals" / "harbor"
    native_root = target / "evals" / "harbor"
    try:
        native_root.symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlink creation is unavailable: {exc}")

    original_iterdir = Path.iterdir
    enumerated_source = False

    def record_iterdir(path: Path):
        nonlocal enumerated_source
        if path.absolute() == native_root.absolute():
            enumerated_source = True
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", record_iterdir)

    with pytest.raises(ValueError, match="symlink"):
        stage_native_harbor_tasks(target, tmp_path / "symlinked-native-root")

    assert not enumerated_source


def test_native_projection_does_not_read_task_directory_on_another_device(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    _write_minimal_native_task(target)
    native_root = target / "evals" / "harbor"
    task_dir = native_root / "case-001"
    original_lstat = Path.lstat
    original_read = adapter_module._read_regular_evals_file
    read_task_config = False

    def cross_device_lstat(path: Path):
        metadata = original_lstat(path)
        if path.absolute() != task_dir.absolute():
            return metadata
        fields = list(metadata)
        fields[2] = metadata.st_dev + 1
        return os.stat_result(fields)

    def record_read(path: Path, **kwargs: object) -> bytes:
        nonlocal read_task_config
        if path.absolute() == (task_dir / "task.toml").absolute():
            read_task_config = True
        return original_read(path, **kwargs)

    monkeypatch.setattr(Path, "lstat", cross_device_lstat)
    monkeypatch.setattr(adapter_module, "_read_regular_evals_file", record_read)

    assert adapter_module._native_projection_entry_ids(native_root) == ()
    assert not read_task_config


def test_link_or_reparse_check_accepts_missing_windows_file_attributes(tmp_path: Path) -> None:
    path = tmp_path / "regular-file"
    path.write_text("regular\n", encoding="utf-8")
    metadata = path.lstat()
    partial_metadata = SimpleNamespace(
        st_mode=metadata.st_mode,
        st_file_attributes=None,
    )

    assert not adapter_module._path_is_link_or_reparse(path, partial_metadata)
