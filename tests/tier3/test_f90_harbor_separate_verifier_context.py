# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native staging must check the separate-verifier context Harbor 0.24 really uses.

Harbor 0.24 builds a separate verifier from the first tests directory that
defines an image (the step's ``tests/``, then the task's ``tests/``) and runs
the test script that image carries; with no image it reuses the agent
environment and uploads ``tests/``. Harbor 0.22 instead used the step's tests
directory whenever it existed. Staging that kept the 0.22 rule would skip the
Compose model Harbor 0.24 actually starts, and would accept a task whose
verifier image has no test script.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("harbor")

from tests.test_harbor_runtime_skill_isolation import _write_minimal_native_task, _write_projection_fixture

from skillevaluator.tier3.harbor.adapter import stage_native_harbor_tasks

_SEPARATE_STEP = '\n[[steps]]\nname = "step-one"\n\n[steps.verifier]\nenvironment_mode = "separate"\n'


def _native_task(tmp_path: Path) -> tuple[Path, Path]:
    _, target, _, _ = _write_projection_fixture(tmp_path)
    _write_minimal_native_task(target)
    native_task = target / "evals" / "harbor" / "case-001"
    task_toml = native_task / "task.toml"
    task_toml.write_text(task_toml.read_text(encoding="utf-8") + _SEPARATE_STEP, encoding="utf-8")
    (native_task / "steps" / "step-one").mkdir(parents=True)
    (native_task / "steps" / "step-one" / "instruction.md").write_text("Run step one.\n", encoding="utf-8")
    return target, native_task


def test_task_level_verifier_compose_behind_a_plain_step_tests_dir_is_still_checked(tmp_path: Path) -> None:
    target, native_task = _native_task(tmp_path)
    # The step's tests/ defines no image, so Harbor 0.24 builds the verifier from the task's tests/.
    step_tests = native_task / "steps" / "step-one" / "tests"
    step_tests.mkdir()
    (step_tests / "test.sh").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    tests_dir = native_task / "tests"
    tests_dir.mkdir()
    (tests_dir / "test.sh").write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    (tests_dir / "docker-compose.yaml").write_text(
        "services:\n  main:\n    depends_on: [helper]\n  helper:\n    image: busybox:${DOCKER_AUTH_CONFIG}\n",
        encoding="utf-8",
    )

    output_dir = tmp_path / "staged"
    with pytest.raises(ValueError, match="undeclared interpolation variables"):
        stage_native_harbor_tasks(target, output_dir, grading_mode="custom_only")
    assert not output_dir.exists()


def test_plain_step_tests_dir_does_not_hide_the_task_test_script(tmp_path: Path) -> None:
    target, native_task = _native_task(tmp_path)
    # The step's tests/ only carries data and defines no image, so Harbor 0.24 reuses the
    # agent environment, uploads tests/ plus the step's tests/, and runs the task test script.
    step_tests = native_task / "steps" / "step-one" / "tests"
    step_tests.mkdir()
    (step_tests / "expected.json").write_text("{}\n", encoding="utf-8")

    [staged] = stage_native_harbor_tasks(target, tmp_path / "staged", grading_mode="custom_only")

    assert (staged / "tests" / "test.sh").is_file()
    assert not (staged / "steps" / "step-one" / "tests" / "test.sh").exists()
