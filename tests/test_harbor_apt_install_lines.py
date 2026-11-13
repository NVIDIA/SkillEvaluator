# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generated Dockerfiles pass exactly the declared apt packages to apt-get."""

from __future__ import annotations

import json
import shlex
import subprocess
from pathlib import Path

import pytest

import skillevaluator.tier3.harbor.adapter as adapter_module
from skillevaluator.tier3.harbor.adapter import (
    _write_default_environment_dockerfile,
    build_eval_base_image,
    generate_harbor_tasks,
)

# Derived from the code's own constant so adding a base package does not break
# these tests. The assertions below still compare the whole package list, so a
# blank or stray entry fails them.
_BASE_PACKAGES = adapter_module._BASE_APT_PACKAGES.split()


def _apt_install_packages(dockerfile: str) -> list[str]:
    """Return the words the shell passes to ``apt-get install``.

    Docker joins a line that ends in ``\\`` with the next line before the shell
    sees the command, so join the RUN instruction the same way and split it like
    ``sh`` does. A stray escaped space shows up here as a ``" "`` package.
    """
    lines = dockerfile.splitlines()
    start = next(index for index, line in enumerate(lines) if line.startswith("RUN apt-get "))
    command = ""
    for line in lines[start:]:
        if not line.endswith("\\"):
            command += line
            break
        command += line[:-1]
    words = shlex.split(command.removeprefix("RUN "))
    first = words.index("--no-install-recommends") + 1
    return words[first : words.index("&&", first)]


def _write_skill(root: Path, apt_packages: str | None) -> Path:
    skill = root / "apt-skill"
    scripts = skill / "scripts"
    scripts.mkdir(parents=True)
    (skill / "SKILL.md").write_text("# apt-skill\n", encoding="utf-8")
    (scripts / "run.py").write_text("print('runtime')\n", encoding="utf-8")
    if apt_packages is not None:
        (scripts / "apt-packages.txt").write_text(apt_packages, encoding="utf-8")
    evals = skill / "evals"
    evals.mkdir()
    (evals / "evals.json").write_text(
        json.dumps([{"id": "case-001", "question": "Run the skill.", "ground_truth": "done"}]),
        encoding="utf-8",
    )
    return skill


_CASES = [
    pytest.param(None, _BASE_PACKAGES, id="base-only"),
    pytest.param("tree\n", [*_BASE_PACKAGES, "tree"], id="one-extra"),
    pytest.param("zstd\n# comment\ntree\n", [*_BASE_PACKAGES, "tree", "zstd"], id="two-extras-and-comment"),
    pytest.param("\r\n  \r\n\t\r\n", _BASE_PACKAGES, id="blank-whitespace-crlf"),
    pytest.param(" tree \r\n\t\r\n zstd\r\n", [*_BASE_PACKAGES, "tree", "zstd"], id="packages-crlf"),
]


@pytest.mark.parametrize(("apt_packages", "expected"), _CASES)
def test_generated_task_dockerfile_installs_declared_apt_packages(
    tmp_path: Path, apt_packages: str | None, expected: list[str]
) -> None:
    skill = _write_skill(tmp_path, apt_packages)

    task = generate_harbor_tasks(skill, tmp_path / "tasks")[0]

    assert _apt_install_packages((task / "environment" / "Dockerfile").read_text(encoding="utf-8")) == expected


@pytest.mark.parametrize(("apt_packages", "expected"), _CASES)
def test_shared_eval_base_image_installs_declared_apt_packages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, apt_packages: str | None, expected: list[str]
) -> None:
    skill = _write_skill(tmp_path, apt_packages)
    built: list[str] = []

    def fake_run(cmd: list[str], *args: object, **kwargs: object) -> subprocess.CompletedProcess[str]:
        if cmd[:3] == ["docker", "image", "inspect"]:
            return subprocess.CompletedProcess(cmd, 1, "", "")
        assert cmd[:2] == ["docker", "build"]
        built.append((Path(cmd[-1]) / "Dockerfile").read_text(encoding="utf-8"))
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(adapter_module.subprocess, "run", fake_run)

    assert build_eval_base_image(skill)
    assert len(built) == 1
    assert _apt_install_packages(built[0]) == expected


@pytest.mark.parametrize(("apt_packages", "expected"), _CASES)
def test_native_default_environment_installs_declared_apt_packages(
    tmp_path: Path, apt_packages: str | None, expected: list[str]
) -> None:
    env_dir = tmp_path / "environment"
    _write_skill(env_dir / "skills", apt_packages)

    _write_default_environment_dockerfile(env_dir, base_image="", agent_config_lines=[], include_input=False)

    assert _apt_install_packages((env_dir / "Dockerfile").read_text(encoding="utf-8")) == expected
