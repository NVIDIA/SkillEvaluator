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
from skillevaluator.tier3.harbor.secure_docker_environment import NVIDIA_BUILD_STDIN_SENTINEL


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


def test_kata_runs_the_hardened_kata_backend() -> None:
    from skillevaluator.tier3.harbor.secure_docker_environment import SECURE_KATA_ENV_IMPORT_PATH

    command = build_harbor_run_command(
        dataset_path="/tmp/dataset",
        agent="codex",
        job_name="kata",
        env_mode="kata",
        environment_kwargs={"kata_runtime": "kata-clh"},
    )

    assert command[command.index("--env") + 1] == SECURE_KATA_ENV_IMPORT_PATH
    assert command[command.index("--agent") + 1] == "codex"
    assert command[command.index("--ak") + 1] == f"reasoning_effort={json.dumps('high')}"
    assert _environment_kwargs(command) == [f"kata_runtime={json.dumps('kata-clh')}"]


@pytest.mark.parametrize(
    ("environment_kwargs", "message"),
    [
        ({"keep_containers": True}, "reserved for Harbor runtime policy"),
        ({"privileged": True}, "does not accept environment kwarg"),
        ({"kata_runtime": "runc"}, "kata_runtime"),
        ({"kata_runtime": "kata --privileged"}, "kata_runtime"),
        ({"kata_runtime": 7}, "kata_runtime"),
        ({"kata_dns": "1.1.1.1\noptions ndots:15"}, "kata_dns"),
        ({"kata_dns": ["dns.example"]}, "kata_dns"),
        ({"kata_dns": [" 1.1.1.1"]}, "kata_dns"),
        ({"kata_dns": {"server": "1.1.1.1"}}, "kata_dns"),
    ],
)
def test_kata_kwargs_keep_the_microvm_boundary(environment_kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        build_harbor_run_command(
            dataset_path="/tmp/dataset",
            agent="opencode",
            job_name="kata",
            env_mode="kata",
            environment_kwargs=environment_kwargs,
        )


@pytest.mark.parametrize("dns", ["1.1.1.1, 2606:4700:4700::1111", ["9.9.9.9"], []])
def test_kata_accepts_literal_dns_servers(dns: object) -> None:
    command = build_harbor_run_command(
        dataset_path="/tmp/dataset",
        agent="opencode",
        job_name="kata",
        env_mode="kata",
        environment_kwargs={"kata_dns": dns},
    )

    assert _environment_kwargs(command) == [f"kata_dns={json.dumps(dns, separators=(',', ':'))}"]


def test_docker_still_rejects_environment_kwargs() -> None:
    with pytest.raises(ValueError, match="not supported for SkillEvaluator Docker mode"):
        build_harbor_run_command(
            dataset_path="/tmp/dataset",
            agent="opencode",
            job_name="docker",
            env_mode="docker",
            environment_kwargs={"kata_runtime": "kata"},
        )


@pytest.mark.parametrize(("registered", "ready"), [(["kata", "kata-clh"], True), (["kata-qemu"], False)])
def test_kata_prerequisites_require_the_selected_runtime(
    monkeypatch: pytest.MonkeyPatch,
    registered: list[str],
    ready: bool,
) -> None:
    from harbor.environments.factory import EnvironmentFactory
    from harbor.environments.kata import KataEnvironment
    from harbor.models.environment_type import EnvironmentType

    preflights: list[object] = []
    monkeypatch.setattr(
        EnvironmentFactory, "run_preflight", lambda environment_type, **_kwargs: preflights.append(environment_type)
    )
    monkeypatch.setattr(KataEnvironment, "_registered_kata_runtimes", classmethod(lambda _cls: registered))
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *_args, **_kwargs: runner.subprocess.CompletedProcess(
            [], 0, stdout="Docker Compose version v2", stderr=""
        ),
    )

    errors = runner._check_prerequisites(
        env_mode="kata", agents=["opencode"], environment_kwargs={"kata_runtime": "kata-clh"}
    )

    assert preflights == [EnvironmentType.KATA]
    if ready:
        assert errors == []
    else:
        assert errors == ["Harbor environment 'kata' requires Docker to register the Kata runtime 'kata-clh'."]


def test_kata_uses_the_docker_nvidia_build_handoff_and_host_environment() -> None:
    assert runner._HARBOR_ENV_MODE_VARS["kata"] == runner._HARBOR_ENV_MODE_VARS["docker"]
    handoff = runner._nvidia_build_key_handoff(
        {"NVIDIA_API_KEY": "nvapi-kata-handoff", "SKILL_EVAL_LLM_PROVIDER": "nv_build"},
        env_mode="kata",
    )
    assert handoff.subprocess_env["NVIDIA_API_KEY"] == NVIDIA_BUILD_STDIN_SENTINEL
    assert handoff.stdin_text == "nvapi-kata-handoff"


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
