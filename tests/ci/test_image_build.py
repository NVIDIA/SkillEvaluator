# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Guard the Tier 1 image layout and the workflow that builds it. Building the image is left to CI."""

import fnmatch
import re
import shlex
from pathlib import Path

import yaml

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


def _workflow() -> dict:
    path = ROOT / ".github/workflows/publish-tier1-image.yml"
    return yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)


def _build_steps(job: dict) -> list[dict]:
    return [step for step in job["steps"] if step.get("uses", "").startswith("docker/build-push-action@")]


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


def _dockerignored(path: str) -> bool:
    lines = (ROOT / ".dockerignore").read_text(encoding="utf-8").splitlines()
    patterns = [line.strip().rstrip("/") for line in lines if line.strip() and not line.startswith("#")]
    parts = Path(path).parts
    prefixes = ["/".join(parts[:count]) for count in range(1, len(parts) + 1)]
    return any(fnmatch.fnmatchcase(prefix, pattern) for pattern in patterns for prefix in prefixes)


def test_build_context_drops_unused_trees_but_keeps_image_inputs():
    copied = [
        source
        for keyword, arguments in _instructions()
        if keyword == "COPY" and not arguments.startswith("--from=")
        for source in arguments.split()[:-1]
    ]
    # The repository's root Dockerfile also builds the wheel from this context.
    package = ["pyproject.toml", "README.md", "src/skillevaluator/reporting/templates/report.html.j2"]
    for path in copied + package:
        assert (ROOT / path).exists(), path
        assert not _dockerignored(path), path
    unused = ["tests/ci/test_image_build.py", "docs/github-actions.mdx", "scripts", "fern", ".github", "CHANGELOG.md"]
    for path in unused:
        assert (ROOT / path).exists(), path
        assert _dockerignored(path), path


def test_pull_requests_read_but_never_write_the_layer_cache():
    workflow = _workflow()
    assert workflow["on"]["push"]["branches"] == ["main"]
    assert workflow["on"]["push"]["paths"] == workflow["on"]["pull_request"]["paths"]
    (verify,) = _build_steps(workflow["jobs"]["verify"])
    assert verify["with"]["cache-from"] == "type=gha,scope=tier1-image"
    cache_to = verify["with"]["cache-to"]
    assert "github.event_name != 'pull_request'" in cache_to and "github.ref == 'refs/heads/main'" in cache_to
    assert "'type=gha,scope=tier1-image,mode=max,ignore-error=true' || ''" in cache_to
    (release,) = _build_steps(workflow["jobs"]["publish"])
    for step in (verify, release):
        assert step["with"]["context"] == "."
        assert step["with"]["file"] == "ci/Dockerfile.tier1"
        assert step["with"]["load"] == "true" and "push" not in step["with"]
    # These events run in main's cache scope on input that a pull request controls.
    for path in (ROOT / ".github/workflows").glob("*.yml"):
        triggers = yaml.load(path.read_text(encoding="utf-8"), Loader=yaml.BaseLoader)["on"]
        triggers = [triggers] if isinstance(triggers, str) else triggers
        assert {"pull_request_target", "workflow_run", "issue_comment"}.isdisjoint(triggers), path.name


def test_release_builds_reuse_no_cached_layers():
    # A cached apt layer would ship the git and ca-certificates of whichever main build first wrote it.
    workflow = _workflow()
    (verify,) = _build_steps(workflow["jobs"]["verify"])
    (release,) = _build_steps(workflow["jobs"]["publish"])
    assert "cache-from" not in release["with"]
    assert release["with"]["no-cache"] == "true"
    # The release still refreshes the cache that later verify builds read.
    assert release["with"]["cache-to"] == "type=gha,scope=tier1-image,mode=max,ignore-error=true"
    assert verify["with"]["cache-from"] == "type=gha,scope=tier1-image"


def test_release_builds_once_and_pushes_the_image_it_verified():
    jobs = _workflow()["jobs"]
    publish = jobs["publish"]
    # Exactly one of the two building jobs runs for any event.
    assert jobs["verify"]["if"] == "${{ !(" + publish["if"] + ") }}"
    assert "needs" not in publish
    (build,) = _build_steps(publish)
    assert build["with"]["tags"] == "${{ env.IMAGE }}:sha-${{ github.sha }}"
    assert build["with"]["labels"].splitlines() == [
        "org.opencontainers.image.source=https://github.com/NVIDIA/SkillEvaluator",
        "org.opencontainers.image.revision=${{ github.sha }}",
        "org.opencontainers.image.description=SkillEvaluator complete Tier 1 CI runtime",
    ]
    steps = publish["steps"]
    smoke = next(index for index, step in enumerate(steps) if "ci/smoke_image.py" in step.get("run", ""))
    release = next(index for index, step in enumerate(steps) if step.get("id") == "release")
    assert steps.index(build) < smoke < release
    assert '--image "$IMAGE:sha-$GITHUB_SHA"' in steps[smoke]["run"]
    assert 'docker push "$IMAGE:sha-$GITHUB_SHA"' in steps[release]["run"]
    runs = [step.get("run", "") for job in jobs.values() for step in job["steps"]]
    assert not any("docker build" in run for run in runs)
