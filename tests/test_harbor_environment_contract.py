# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SkillEvaluator's environment surface over the pinned Harbor release."""

from __future__ import annotations

import inspect
import json
import sys
import types
from pathlib import Path

import pytest

from skillevaluator.tier3.harbor import DOCKER_AGENT_IMPORT_PATHS, runner
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
    # Kata runs the Docker agent wrappers, which skip the redundant Node/npm bootstrap.
    assert command[command.index("--agent") + 1] == DOCKER_AGENT_IMPORT_PATHS["codex"]
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
        ({"kata_dns": "fe80::1%eth0\nsearch attacker.example"}, "kata_dns"),
        ({"kata_dns": ["fe80::1%eth0\noptions ndots:15"]}, "kata_dns"),
        ({"kata_dns": ["fe80::1%eth0"]}, "kata_dns"),
        ({"kata_dns": ""}, "at least one IP address"),
        ({"kata_dns": ","}, "at least one IP address"),
        ({"kata_dns": []}, "at least one IP address"),
        ({"kata_dns": None}, "kata_dns"),
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


@pytest.mark.parametrize("dns", ["1.1.1.1, 2606:4700:4700::1111", ["9.9.9.9"], ["10.0.0.2", "::1"]])
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


def test_kata_default_runtime_must_be_registered(monkeypatch: pytest.MonkeyPatch) -> None:
    from harbor.environments.factory import EnvironmentFactory
    from harbor.environments.kata import KataEnvironment

    monkeypatch.setattr(EnvironmentFactory, "run_preflight", lambda _environment_type, **_kwargs: None)
    monkeypatch.setattr(KataEnvironment, "_registered_kata_runtimes", classmethod(lambda _cls: ["kata-clh"]))
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *_args, **_kwargs: runner.subprocess.CompletedProcess(
            [], 0, stdout="Docker Compose version v2", stderr=""
        ),
    )

    assert runner._check_prerequisites(env_mode="kata", agents=["opencode"]) == [
        "Harbor environment 'kata' requires Docker to register the Kata runtime 'kata'."
    ]


def test_kata_rejects_compose_sidecars_before_harbor_starts(tmp_path: Path) -> None:
    def task(name: str, compose: str | None) -> Path:
        environment = tmp_path / name / "environment"
        environment.mkdir(parents=True)
        if compose is not None:
            (environment / "docker-compose.yaml").write_text(compose, encoding="utf-8")
        return tmp_path / name

    dockerfile_only = task("dockerfile-only", None)
    main_only = task("main-only", "services:\n  main:\n    environment:\n      MODE: test\n")
    sidecar = task("with-db", "services:\n  main: {}\n  db:\n    image: postgres:16\n")

    assert runner._staged_task_environment_error("kata", [dockerfile_only, main_only]) is None
    error = runner._staged_task_environment_error("kata", [dockerfile_only, sidecar])
    assert error is not None
    assert "task 'with-db'" in error
    assert "(db)" in error
    assert runner._staged_task_environment_error("docker", [sidecar]) is None


def test_kata_runs_are_described_as_microvm_attempts() -> None:
    from skillevaluator.reporting.benchmark import _environment_note

    assert _environment_note("kata") == "Each task attempt ran in its own Kata Containers microVM."
    assert _environment_note("docker") == "Each task attempt ran in its own isolated Docker container."


def test_kata_uses_the_docker_nvidia_build_handoff_and_host_environment() -> None:
    assert runner._HARBOR_ENV_MODE_VARS["kata"] == runner._HARBOR_ENV_MODE_VARS["docker"]
    handoff = runner._nvidia_build_key_handoff(
        {"NVIDIA_API_KEY": "nvapi-kata-handoff", "SKILL_EVAL_LLM_PROVIDER": "nv_build"},
        env_mode="kata",
    )
    assert handoff.subprocess_env["NVIDIA_API_KEY"] == NVIDIA_BUILD_STDIN_SENTINEL
    assert handoff.stdin_text == "nvapi-kata-handoff"


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


def test_runta_and_mosaic_config_paths_survive_the_evaluator_owned_launch_directory(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.chdir(tmp_path)

    launch_env = runner._harbor_launch_environment(
        {
            "RUNTA_CONFIG": "runta/config.toml",
            "MOSAIC_CONFIG": "mosaic/config.json",
            "MAR_CONFIG": "~/.config/mosaic-sandbox/config.json",
            "MOSAIC_API_URL": "https://sandbox.example",
        }
    )

    assert launch_env["RUNTA_CONFIG"] == str(tmp_path / "runta" / "config.toml")
    assert launch_env["MOSAIC_CONFIG"] == str(tmp_path / "mosaic" / "config.json")
    assert launch_env["MAR_CONFIG"] == "~/.config/mosaic-sandbox/config.json"
    assert launch_env["MOSAIC_API_URL"] == "https://sandbox.example"


@pytest.mark.parametrize(
    ("env_mode", "name"),
    [
        ("mosaic", "MAR_ENDPOINT"),
        ("mosaic", "MAR_CONFIG"),
        ("mosaic", "MAR_API_TOKEN"),
        ("mosaic", "MOSAIC_CONFIG"),
        ("mosaic", "MOSAIC_API_URL"),
        ("runta", "RUNTA_ENDPOINT"),
        ("runta", "RUNTA_CONFIG"),
    ],
)
def test_skill_runtime_env_cannot_redirect_runta_or_mosaic_credentials(env_mode: str, name: str) -> None:
    resolved, errors = runner._resolve_runtime_env({name: "https://attacker.example"}, env_mode=env_mode)

    assert resolved == {}
    assert errors == [f"harbor.runtime_env.{name} controls the host process and is not allowed"]


def test_mosaic_allowlist_covers_the_sdk_credential_names() -> None:
    credentials = pytest.importorskip("mosaic_sandbox.credentials")

    for name in ("MOSAIC_API_TOKEN", "MOSAIC_API_URL", "MOSAIC_CONFIG"):
        assert name in runner._HARBOR_ENV_MODE_VARS["mosaic"]
        assert credentials.LEGACY_ENV_NAMES[name] in runner._HARBOR_ENV_MODE_VARS["mosaic"]


def test_runta_and_mosaic_credentials_do_not_cross_backends(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        runner.os,
        "environ",
        {"PATH": "/usr/bin", "HOME": "/home/test", "RUNTA_TOKEN": "runta-value", "MOSAIC_API_TOKEN": "mosaic-value"},
    )

    runta = runner._selected_host_environment(runner._HARBOR_ENV_MODE_VARS["runta"], runner.os.environ)
    mosaic = runner._selected_host_environment(runner._HARBOR_ENV_MODE_VARS["mosaic"], runner.os.environ)

    assert runta == {"RUNTA_TOKEN": "runta-value"}
    assert mosaic == {"MOSAIC_API_TOKEN": "mosaic-value"}


@pytest.mark.parametrize(
    ("env_mode", "name"), [("runta", "RUNTA_CONFIG"), ("mosaic", "MOSAIC_CONFIG"), ("mosaic", "MAR_CONFIG")]
)
def test_runta_and_mosaic_config_files_must_exist(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    env_mode: str,
    name: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setitem(sys.modules, "mosaic_sandbox.credentials", None)
    monkeypatch.setenv(name, "missing/config.toml")

    assert runner._backend_config_prerequisite_errors(env_mode) == [
        f"Harbor environment '{env_mode}' requires {name} to name an existing regular file."
    ]

    (tmp_path / "missing").mkdir()
    (tmp_path / "missing" / "config.toml").write_text("token = 'x'\n", encoding="utf-8")
    assert runner._backend_config_prerequisite_errors(env_mode) == []


@pytest.mark.parametrize(("source", "rejected"), [("e2b_environment", True), ("environment", False), ("config", False)])
def test_mosaic_rejects_a_token_only_in_e2b_api_key(
    monkeypatch: pytest.MonkeyPatch, source: str, rejected: bool
) -> None:
    fake = types.ModuleType("mosaic_sandbox.credentials")
    fake.resolve_token = lambda: ("msk_live_value", source)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mosaic_sandbox.credentials", fake)
    for name in ("MOSAIC_CONFIG", "MAR_CONFIG"):
        monkeypatch.delenv(name, raising=False)

    errors = runner._backend_config_prerequisite_errors("mosaic")

    assert bool(errors) is rejected
    if rejected:
        assert "E2B_API_KEY" in errors[0]


def test_mosaic_rejects_compose_tasks_before_harbor_starts(tmp_path: Path) -> None:
    single = tmp_path / "single"
    (single / "environment").mkdir(parents=True)
    compose = tmp_path / "compose"
    (compose / "environment").mkdir(parents=True)
    (compose / "environment" / "docker-compose.yaml").write_text("services: {}\n", encoding="utf-8")

    assert runner._staged_task_environment_error("mosaic", [single]) is None
    assert "cannot run Docker Compose task 'compose'" in runner._staged_task_environment_error(
        "mosaic", [single, compose]
    )
    assert runner._staged_task_environment_error("runta", [compose]) is None


def test_missing_extra_message_keeps_harbor_diagnosis_without_unpinned_install_hints() -> None:
    from harbor.utils.optional_import import MissingExtraError

    summary = runner._harbor_missing_dependency_summary(MissingExtraError(package="mosaic-sandbox", extra="mosaic"))

    assert summary == "The 'mosaic-sandbox' package is required but not installed"
    assert runner._harbor_missing_dependency_summary(ImportError("No module named 'runta'")) == (
        "No module named 'runta'"
    )


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
