# SPDX-License-Identifier: Apache-2.0
"""Staged main images remain compatible with Compose 2 health-based waits."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from skillevaluator.tier3.harbor.adapter import generate_harbor_tasks, stage_native_harbor_tasks


def _assert_safe_readiness_check(dockerfile: Path) -> None:
    instructions = [line for line in dockerfile.read_text().splitlines() if line.startswith("HEALTHCHECK ")]
    # Docker keeps only the final HEALTHCHECK. NONE suppresses the authored
    # image check, but Compose 2 needs a real evaluator-owned check afterward.
    assert "HEALTHCHECK NONE" in instructions
    _options, separator, command = instructions[-1].partition(" CMD ")
    assert separator
    assert json.loads(command) == ["/bin/true"]


@pytest.mark.parametrize("image_mode", ["generated", "base", "preserve", "rebase"])
def test_generated_main_image_replaces_authored_healthcheck_with_constant_readiness(
    tmp_path: Path, image_mode: str
) -> None:
    skill = tmp_path / "skill"
    authored = skill / "evals/environment"
    authored.mkdir(parents=True)
    (skill / "SKILL.md").write_text("# Test skill\n")
    (skill / "evals/evals.json").write_text(
        json.dumps([{"id": "case-001", "question": "Run the test", "expected_skill": "skill"}])
    )
    if image_mode in {"preserve", "rebase"}:
        (authored / "Dockerfile").write_text(
            'FROM python:3.12-slim\nHEALTHCHECK CMD ["sh", "-c", "touch /tmp/authored-probe"]\n'
        )
    task = generate_harbor_tasks(
        skill,
        tmp_path / "tasks",
        base_image="fixture-base:latest" if image_mode in {"base", "rebase"} else "",
        custom_dockerfile_mode="rebase" if image_mode == "rebase" else "preserve",
    )[0]
    _assert_safe_readiness_check(task / "environment/Dockerfile")


def test_native_main_image_replaces_authored_healthcheck_with_constant_readiness(tmp_path: Path) -> None:
    skill = tmp_path / "native-skill"
    task = skill / "evals/harbor/case-001"
    environment = task / "environment"
    environment.mkdir(parents=True)
    (task / "tests").mkdir()
    (skill / "SKILL.md").write_text("# Native test skill\n")
    (task / "task.toml").write_text('schema_version = "1.3"\n[task]\nname = "nvidia/case-001"\n[environment]\n')
    (task / "instruction.md").write_text("Run the test.\n")
    (task / "tests/test.sh").write_text("#!/bin/sh\nexit 0\n")
    (environment / "Dockerfile").write_text(
        'FROM python:3.12-slim\nHEALTHCHECK CMD ["sh", "-c", "touch /tmp/authored-probe"]\n'
    )
    staged = stage_native_harbor_tasks(skill, tmp_path / "native-tasks", grading_mode="custom_only")[0]
    _assert_safe_readiness_check(staged / "environment/Dockerfile")


def test_custom_compose_service_healthchecks_keep_their_readiness_policy(tmp_path: Path) -> None:
    skill = tmp_path / "skill"
    authored = skill / "evals/environment"
    authored.mkdir(parents=True)
    (skill / "SKILL.md").write_text("# Test skill\n")
    (skill / "evals/evals.json").write_text(json.dumps([{"id": "case-001", "question": "Run the test"}]))
    services = {
        "main": {
            "depends_on": {"database": {"condition": "service_healthy"}},
        },
        "database": {
            "image": "postgres:16",
            "healthcheck": {"test": ["CMD-SHELL", "pg_isready -U postgres"]},
        },
    }
    (authored / "docker-compose.yaml").write_text(yaml.safe_dump({"services": services}))
    staged = generate_harbor_tasks(skill, tmp_path / "tasks")[0]
    compose = yaml.safe_load((staged / "environment/docker-compose.yaml").read_text())
    assert compose["services"]["main"] == services["main"]
    assert compose["services"]["database"]["healthcheck"] == services["database"]["healthcheck"]
