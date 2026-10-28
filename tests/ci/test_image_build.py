# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Guard the Tier 1 image layout. Building the image is left to CI."""

import re
import shlex
from pathlib import Path

ROOT = Path(__file__).parents[2]
DOCKERFILE = ROOT / "ci/Dockerfile.tier1"


def _instructions() -> list[tuple[str, str]]:
    """Dockerfile instructions as (keyword, arguments), with continuations joined and comments dropped."""
    text = re.sub(r"\\\n", " ", DOCKERFILE.read_text(encoding="utf-8"))
    lines = [line.strip() for line in text.splitlines()]
    return [tuple(line.split(None, 1)) for line in lines if line and not line.startswith("#")]


def _stages() -> dict[str, list[tuple[str, str]]]:
    """Instructions per stage. The unnamed final stage is the shipped runtime."""
    stages: dict[str, list[tuple[str, str]]] = {}
    for keyword, arguments in _instructions():
        if keyword == "FROM":
            words = arguments.split()
            name = words[2] if len(words) == 3 and words[1].upper() == "AS" else "runtime"
            stages[name] = []
        stages[name].append((keyword, arguments))
    assert list(stages) == ["build", "project", "runtime"]
    return stages


def _arguments(stage: list[tuple[str, str]], keyword: str) -> list[str]:
    return [arguments for name, arguments in stage if name == keyword]


def test_runtime_stage_ships_only_what_scans_run():
    runtime = _stages()["runtime"]
    assert _arguments(runtime, "COPY") == [
        "--from=build /opt/semgrep /opt/semgrep",
        "--from=build /opt/skillspector /opt/skillspector",
        "--from=build /usr/local/bin/gitleaks /usr/local/bin/gitleaks",
        "--from=build /usr/local/share/licenses/gitleaks /usr/local/share/licenses/gitleaks",
        "--from=build /opt/skillevaluator/.venv /opt/skillevaluator/.venv",
        "--from=project /overlay/ /",
        "LICENSE NOTICE THIRD_PARTY_NOTICES.md /opt/skillevaluator/",
        "ci /opt/skillevaluator/ci",
    ]
    commands = {word for command in _arguments(runtime, "RUN") for word in shlex.split(command)}
    assert {"uv", "curl", "pip"}.isdisjoint(commands)
    assert {"git", "ca-certificates"} <= commands
    assert _arguments(runtime, "USER") == ["evaluator"]
    assert _arguments(runtime, "ENTRYPOINT") == ['["python", "/opt/skillevaluator/ci/run_tier1.py"]']


def test_build_and_runtime_stages_share_one_pinned_base():
    bases = {arguments.split()[0] for keyword, arguments in _instructions() if keyword == "FROM"}
    assert bases - {"build"} == {
        "python:3.12-slim-bookworm@sha256:62ed8e07e544e8555c5d93748d5275015136b7f5c52b5e8679b4de339c3779d3"
    }


def test_bytecode_is_compiled_at_build_time_and_never_written_at_runtime():
    stages = _stages()
    build_environment = " ".join(_arguments(stages["build"], "ENV")).split()
    assert {"UV_COMPILE_BYTECODE=1", "PYC_INVALIDATION_MODE=UNCHECKED_HASH"} <= set(build_environment)
    runtime = stages["runtime"]
    assert "PYTHONDONTWRITEBYTECODE=1" in " ".join(_arguments(runtime, "ENV")).split()
    stdlib = [command for command in _arguments(runtime, "RUN") if command.endswith("/usr/local/lib/python3.12")]
    assert len(stdlib) == 1
    assert "-m compileall" in stdlib[0] and "--invalidation-mode unchecked-hash" in stdlib[0]


def test_source_edits_rebuild_only_the_last_layers():
    stages = _stages()
    build = stages["build"]
    first_source = next(
        index
        for index, (keyword, arguments) in enumerate(build)
        if keyword == "COPY" and not arguments.startswith("ci/requirements/")
    )
    scanners = [
        index
        for index, (keyword, arguments) in enumerate(build)
        if keyword == "RUN" and re.search(r"/opt/semgrep|/opt/skillspector|gitleaks", arguments)
    ]
    assert len(scanners) == 3 and max(scanners) < first_source
    assert build[first_source] == ("COPY", "pyproject.toml uv.lock ./")
    assert _arguments(stages["project"], "COPY") == ["README.md LICENSE NOTICE THIRD_PARTY_NOTICES.md ./", "src ./src"]
    runtime = stages["runtime"]
    overlay = runtime.index(("COPY", "--from=project /overlay/ /"))
    checks = next(index for index, (_, arguments) in enumerate(runtime) if "semgrep --version" in arguments)
    shared = [index for index, (_, arguments) in enumerate(runtime) if arguments.startswith("--from=build ")]
    assert max([*shared, checks]) < overlay
