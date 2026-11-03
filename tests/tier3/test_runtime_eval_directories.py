# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Runtime eval helpers survive staging; recognized evaluation datasets do not."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from skillevaluator.plugin_inventory import plugin_inventory_for_root
from skillevaluator.tier3.dataset_utils import DATASET_EXTENSIONS
from skillevaluator.tier3.harbor.adapter import (
    _runtime_skill_copy_ignore,
    _sanitize_staged_runtime_skills,
    generate_harbor_tasks,
)
from skillevaluator.tier3.harbor.secure_copy import UnsafeStagingError, copytree_secure
from skillevaluator.tier3.plugin_execution import NativeRuntimeHooks
from skillevaluator.tier3.plugin_native import resolve_plugin_load
from skillevaluator.tier3.plugin_prepare import prepare_runtime_package, stage_wrapper_skill
from skillevaluator.tier3.workflow import _reject_legacy_datasets


def _skill(root: Path, *, eval_name: str = "eval") -> Path:
    (root / eval_name).mkdir(parents=True)
    (root / "SKILL.md").write_text(
        f"---\nname: {root.name}\ndescription: Runtime helper fixture.\n---\nRun python {eval_name}/run_eval.py.\n"
    )
    (root / eval_name / "run_eval.py").write_text("print('runtime helper executed')\n")
    # An unrelated file with a dataset-like stem is still runtime material.
    (root / eval_name / "dataset.txt").write_text("runtime-only data")
    (root / "evals").mkdir()
    (root / "evals/evals.json").write_text(json.dumps([{"id": "one", "prompt": "Run the helper"}]))
    return root


def _assert_helper_runs(skill: Path, *, eval_name: str = "eval") -> None:
    result = subprocess.run(
        [sys.executable, f"{eval_name}/run_eval.py"], cwd=skill, capture_output=True, text=True, check=True
    )
    assert result.stdout == "runtime helper executed\n"
    assert (skill / eval_name / "dataset.txt").read_text() == "runtime-only data"
    assert not (skill / "evals").exists()


@pytest.mark.parametrize("with_skill", [True, False])
def test_emitted_tasks_keep_target_reference_and_workspace_runtime_helpers(tmp_path: Path, with_skill: bool) -> None:
    target = _skill(tmp_path / "target")
    reference = _skill(tmp_path / "references/decoy")
    workspace = _skill(tmp_path / "workspace")
    _reject_legacy_datasets(target)
    tasks = generate_harbor_tasks(
        target,
        tmp_path / "tasks",
        with_skill=with_skill,
        reference_skills_dir=reference.parent,
        workspace_skill_paths=[workspace],
        workspace_mode="group",
    )
    assert len(tasks) == 1
    task = tasks[0]
    assert "COPY skills/" in (task / "environment/Dockerfile").read_text()
    skills = task / "environment/skills"
    for name in ("decoy", "workspace", *(("target",) if with_skill else ())):
        _assert_helper_runs(skills / name)
    assert (skills / "target").exists() is with_skill


@pytest.mark.parametrize("eval_name", ["eval", "EVAL"])
@pytest.mark.parametrize("extension", [None, *DATASET_EXTENSIONS])
def test_nested_skill_copy_and_sanitizer_agree_on_recognized_datasets(
    tmp_path: Path, eval_name: str, extension: str | None
) -> None:
    source = tmp_path / "source"
    member = _skill(source / "skills/member", eval_name=eval_name)
    if extension is not None:
        (member / eval_name / f"dataset{extension}").write_text("VERIFIER-ONLY-ANSWER")
    copied = tmp_path / "copied"
    copytree_secure(source, copied, ignore=_runtime_skill_copy_ignore(source))
    authored = tmp_path / "authored"
    shutil.copytree(source, authored)
    _sanitize_staged_runtime_skills(authored)
    for root in (copied, authored):
        skill = root / "skills/member"
        assert not (skill / "evals").exists()
        if extension is None:
            _assert_helper_runs(skill, eval_name=eval_name)
        else:
            assert not (skill / eval_name).exists()


@pytest.mark.parametrize("agent", ["claude-code", "codex", "opencode", "hermes"])
def test_wrapper_and_native_tasks_preserve_member_runtime_eval_helpers(tmp_path: Path, agent: str) -> None:
    root = tmp_path / "plugin"
    _skill(root / "skills/member")
    (root / ".claude-plugin").mkdir()
    (root / ".claude-plugin/plugin.json").write_text(json.dumps({"name": "demo"}))
    inventory = plugin_inventory_for_root(root)
    assert inventory is not None
    package = prepare_runtime_package(inventory)
    wrapper = stage_wrapper_skill(package, tmp_path / "wrapper")
    _assert_helper_runs(wrapper.path / "skills/member")
    tasks = generate_harbor_tasks(wrapper.path, tmp_path / "tasks", with_skill=True)
    hooks = NativeRuntimeHooks(
        package.native_source, resolve_plugin_load("native", [agent], env_mode="docker"), package=package
    )
    hooks.stage_with_arm(agent, tasks)
    for task in tasks:
        environment = task / "environment"
        skill = (
            environment / "skilleval/native/claude-code/plugin/skills/member"
            if agent == "claude-code"
            else environment / "skills/member"
        )
        _assert_helper_runs(skill)


def test_legacy_dataset_link_is_excluded_without_reading_the_target(tmp_path: Path) -> None:
    source = _skill(tmp_path / "source")
    (source / "eval/dataset.json").symlink_to(tmp_path / "missing-private-answer")
    copied = tmp_path / "copied"
    copytree_secure(source, copied, ignore=_runtime_skill_copy_ignore(source))
    assert not (copied / "eval").exists()
    assert not (copied / "evals").exists()


def test_runtime_eval_directory_link_is_refused(tmp_path: Path) -> None:
    source = _skill(tmp_path / "source")
    outside = tmp_path / "outside"
    (source / "eval").rename(outside)
    (source / "eval").symlink_to(outside, target_is_directory=True)
    with pytest.raises(UnsafeStagingError):
        copytree_secure(source, tmp_path / "copied", ignore=_runtime_skill_copy_ignore(source))
