# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SkillEvaluator's environment surface over the pinned Harbor release."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from skillevaluator.tier3.harbor import runner
from skillevaluator.tier3.harbor.runner import build_harbor_run_command


def _environment_kwargs(command: list[str]) -> list[str]:
    return [command[index + 1] for index, value in enumerate(command) if value == "--ek"]


def test_wandb_runs_as_harbor_cwsandbox_with_wandb_auth() -> None:
    command = build_harbor_run_command(
        dataset_path="/tmp/dataset",
        agent="opencode",
        job_name="wandb",
        env_mode="wandb",
        environment_kwargs={"max_lifetime_seconds": 600},
    )

    assert command[command.index("--env") + 1] == "cwsandbox"
    # SkillEvaluator's alias kwarg comes after every operator kwarg.
    assert _environment_kwargs(command) == ["max_lifetime_seconds=600", f"auth={json.dumps('wandb')}"]


@pytest.mark.parametrize("env_mode", ["cwsandbox", "wandb"])
def test_operators_cannot_choose_cwsandbox_auth(env_mode: str) -> None:
    with pytest.raises(ValueError, match="auth"):
        build_harbor_run_command(
            dataset_path="/tmp/dataset",
            agent="opencode",
            job_name="auth",
            env_mode=env_mode,
            environment_kwargs={"auth": "wandb"},
        )


@pytest.mark.parametrize("env_mode", ["daytona", "modal", "tensorlake", "beam"])
@pytest.mark.parametrize("name", ["stream", "enable_environment_dir_upload"])
def test_harbor_stream_and_upload_controls_are_reserved(env_mode: str, name: str) -> None:
    with pytest.raises(ValueError, match="reserved for Harbor runtime policy"):
        build_harbor_run_command(
            dataset_path="/tmp/dataset",
            agent="opencode",
            job_name="reserved",
            env_mode=env_mode,
            environment_kwargs={name: False},
        )


def test_backend_kwargs_track_the_pinned_harbor_release() -> None:
    with pytest.raises(ValueError, match=r"Harbor 0\.24\.0 environment 'modal'.*modal_sandbox_v2"):
        build_harbor_run_command(
            dataset_path="/tmp/dataset",
            agent="opencode",
            job_name="modal",
            env_mode="modal",
            environment_kwargs={"modal_sandbox_v2": True},
        )
    hyperbrowser = build_harbor_run_command(
        dataset_path="/tmp/dataset",
        agent="opencode",
        job_name="hyperbrowser",
        env_mode="hyperbrowser",
        environment_kwargs={"builder_cpus": 4},
    )
    assert "builder_cpus=4" in _environment_kwargs(hyperbrowser)
    tensorlake = build_harbor_run_command(
        dataset_path="/tmp/dataset",
        agent="opencode",
        job_name="tensorlake",
        env_mode="tensorlake",
        environment_kwargs={"dind_image": "docker:dind"},
    )
    assert f"dind_image={json.dumps('docker:dind')}" in _environment_kwargs(tensorlake)


@pytest.mark.parametrize(
    ("env_mode", "name", "value"),
    [
        ("runta", "mode", "direct"),
        ("mosaic", "volume", "shared-cache"),
        ("mosaic", "persist", True),
        ("mosaic", "enable_ssh", True),
        ("mosaic", "build_args", {"BASE_IMAGE": "python:3.12"}),
        ("mosaic", "build_target", "builder"),
    ],
)
def test_runta_and_mosaic_isolation_controls_are_reserved(env_mode: str, name: str, value: object) -> None:
    with pytest.raises(ValueError, match=rf"reserved for Harbor runtime policy: {name}"):
        build_harbor_run_command(
            dataset_path="/tmp/dataset",
            agent="opencode",
            job_name="reserved",
            env_mode=env_mode,
            environment_kwargs={name: value},
        )


@pytest.mark.parametrize(("env_mode", "name", "value"), [("runta", "token", "rt-123456"), ("mosaic", "secrets", ["s"])])
def test_runta_and_mosaic_credentials_stay_in_the_host_environment(env_mode: str, name: str, value: object) -> None:
    with pytest.raises(ValueError, match="secret-bearing"):
        build_harbor_run_command(
            dataset_path="/tmp/dataset",
            agent="opencode",
            job_name="credentials",
            env_mode=env_mode,
            environment_kwargs={name: value},
        )


def test_runta_and_mosaic_forward_operational_kwargs() -> None:
    runta = build_harbor_run_command(
        dataset_path="/tmp/dataset",
        agent="opencode",
        job_name="runta",
        env_mode="runta",
        environment_kwargs={"endpoint": "https://runta.example", "startup_timeout_sec": 300},
    )
    assert runta[runta.index("--env") + 1] == "runta"
    assert _environment_kwargs(runta) == [f"endpoint={json.dumps('https://runta.example')}", "startup_timeout_sec=300"]

    mosaic = build_harbor_run_command(
        dataset_path="/tmp/dataset",
        agent="opencode",
        job_name="mosaic",
        env_mode="mosaic",
        environment_kwargs={"metadata": {"team": "evals"}, "replicas": 2, "ttl_seconds": 7200},
    )
    assert mosaic[mosaic.index("--env") + 1] == "mosaic"
    assert _environment_kwargs(mosaic) == [
        f"metadata={json.dumps({'team': 'evals'}, separators=(',', ':'))}",
        "replicas=2",
        "ttl_seconds=7200",
    ]


@pytest.mark.parametrize("env_mode", ["prime", "smol"])
def test_harbor_backends_without_task_projection_stay_unexposed(env_mode: str) -> None:
    assert env_mode not in runner.HARBOR_ENV_MODES
    assert "Unsupported Harbor environment" in runner._check_prerequisites(env_mode=env_mode, agents=["opencode"])[0]


def test_cwsandbox_prerequisites_require_sdk_and_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner.importlib.util, "find_spec", lambda name: None if name == "cwsandbox" else object())
    assert "harbor[cwsandbox]==0.24.0" in runner._cwsandbox_prerequisite_errors("cwsandbox")[0]

    monkeypatch.setattr(runner.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.delenv("CWSANDBOX_API_KEY", raising=False)
    assert runner._cwsandbox_prerequisite_errors("cwsandbox") == [
        "Harbor environment 'cwsandbox' requires CWSANDBOX_API_KEY."
    ]
    monkeypatch.setenv("CWSANDBOX_API_KEY", "cw-key-123456")
    assert runner._cwsandbox_prerequisite_errors("cwsandbox") == []
    assert runner._cwsandbox_prerequisite_errors("daytona") == []


def test_wandb_prerequisites_accept_api_key_or_netrc(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(runner.importlib.util, "find_spec", lambda name: None if name == "wandb" else object())
    assert "wandb" in runner._cwsandbox_prerequisite_errors("wandb")[0]

    monkeypatch.setattr(runner.importlib.util, "find_spec", lambda _name: object())
    monkeypatch.setattr(Path, "home", classmethod(lambda _cls: tmp_path))
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    monkeypatch.delenv("NETRC", raising=False)
    assert "WANDB_API_KEY" in runner._cwsandbox_prerequisite_errors("wandb")[0]
    (tmp_path / ".netrc").write_text("machine api.wandb.ai\n", encoding="utf-8")
    assert runner._cwsandbox_prerequisite_errors("wandb") == []


def test_wandb_preflight_uses_the_cwsandbox_environment_type(monkeypatch: pytest.MonkeyPatch) -> None:
    from harbor.environments.factory import EnvironmentFactory
    from harbor.models.environment_type import EnvironmentType

    preflights: list[object] = []
    monkeypatch.setattr(runner, "_cwsandbox_prerequisite_errors", lambda _env_mode: [])
    monkeypatch.setattr(
        EnvironmentFactory, "run_preflight", lambda environment_type, **_kwargs: preflights.append(environment_type)
    )

    assert runner._check_prerequisites(env_mode="wandb", agents=["opencode"]) == []
    assert preflights == [EnvironmentType.CWSANDBOX]


def test_pinned_harbor_loads_env_local_from_its_working_directory() -> None:
    """Document why every Harbor launch uses an evaluator-owned working directory."""
    from importlib.metadata import version

    from harbor.cli import jobs

    assert ".env.local" in inspect.getsource(jobs)
    major, minor = (int(part) for part in version("python-dotenv").split(".")[:2])
    assert (major, minor) >= (1, 2)
