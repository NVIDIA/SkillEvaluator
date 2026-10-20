# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Custom execution backends contributed by installed packages through entry points."""

from __future__ import annotations

import json
import subprocess
import sys
import types
from collections.abc import Iterator
from importlib.metadata import EntryPoint
from pathlib import Path

import pytest

from skillevaluator import tier3_environments
from skillevaluator.tier3.harbor import runner
from skillevaluator.tier3.harbor.runner import build_harbor_run_command
from skillevaluator.tier3_environments import (
    DISABLE_CUSTOM_BACKENDS_ENV,
    ENV_MODE_ENTRY_POINT_GROUP,
    EnvironmentBackendError,
    EnvironmentBackendSpec,
)

_MODULE = "skillevaluator_test_backend"
_ENV_IMPORT_PATH = f"{_MODULE}:AcmeEnvironment"
_AGENT_IMPORT_PATH = f"{_MODULE}:AcmeClaudeCode"


class _AcmeEnvironment:
    preflight_calls = 0
    preflight_error: Exception | None = None

    @classmethod
    def preflight(cls) -> None:
        cls.preflight_calls += 1
        if cls.preflight_error is not None:
            raise cls.preflight_error


def _spec(**overrides: object) -> EnvironmentBackendSpec:
    options: dict[str, object] = {
        "import_path": _ENV_IMPORT_PATH,
        "environment_kwargs": frozenset({"endpoint", "profile", "lease_seconds", "pool"}),
        "required_environment_kwargs": ("endpoint",),
        "reserved_environment_kwargs": frozenset({"pool"}),
        "host_env_vars": frozenset({"ACME_SANDBOX_TOKEN", "ACME_SANDBOX_URL"}),
        "agent_import_paths": {"claude-code": _AGENT_IMPORT_PATH},
        "prerequisites": f"{_MODULE}:check",
        "install_hint": "Install 'skillevaluator-acme'.",
    }
    options.update(overrides)
    return EnvironmentBackendSpec(**options)  # type: ignore[arg-type]


@pytest.fixture
def backend_module(monkeypatch: pytest.MonkeyPatch) -> Iterator[types.ModuleType]:
    module = types.ModuleType(_MODULE)
    module.AcmeEnvironment = _AcmeEnvironment  # type: ignore[attr-defined]
    module.AcmeClaudeCode = object  # type: ignore[attr-defined]
    module.SPEC = _spec()  # type: ignore[attr-defined]
    module.check = lambda kwargs: [] if kwargs.get("endpoint") != "down" else ["acme endpoint is down"]  # type: ignore[attr-defined]
    module.make_spec = _spec  # type: ignore[attr-defined]
    module.broken = 42  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, _MODULE, module)
    monkeypatch.delenv(DISABLE_CUSTOM_BACKENDS_ENV, raising=False)
    _AcmeEnvironment.preflight_calls = 0
    _AcmeEnvironment.preflight_error = None
    yield module
    tier3_environments.clear_custom_backend_cache()


def _install(monkeypatch: pytest.MonkeyPatch, **entries: str) -> None:
    points = [EntryPoint(name=name, value=value, group=ENV_MODE_ENTRY_POINT_GROUP) for name, value in entries.items()]
    monkeypatch.setattr(
        tier3_environments, "entry_points", lambda *, group: points if group == ENV_MODE_ENTRY_POINT_GROUP else []
    )
    tier3_environments.clear_custom_backend_cache()


@pytest.fixture
def acme(backend_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch) -> types.ModuleType:
    _install(monkeypatch, acme=f"{_MODULE}:SPEC")
    return backend_module


def test_installed_backends_extend_the_env_mode_list(acme: types.ModuleType) -> None:
    assert tier3_environments.custom_backend_names() == ("acme",)
    assert tier3_environments.all_env_modes()[-1] == "acme"
    assert tier3_environments.is_env_mode("acme")
    assert not tier3_environments.is_env_mode("acme-unknown")
    assert tier3_environments.load_custom_backend("docker") is None


def test_backends_cannot_replace_built_in_modes_or_use_invalid_names(
    backend_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch, docker=f"{_MODULE}:SPEC", local=f"{_MODULE}:SPEC", Bad_Name=f"{_MODULE}:SPEC")

    assert tier3_environments.custom_backend_names() == ()
    command = build_harbor_run_command(dataset_path="/tmp/dataset", agent="opencode", job_name="d", env_mode="docker")
    assert command[command.index("--env") + 1] != _ENV_IMPORT_PATH


def test_backends_can_be_disabled(acme: types.ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(DISABLE_CUSTOM_BACKENDS_ENV, "1")

    assert tier3_environments.custom_backend_names() == ()
    assert "Unsupported Harbor environment" in runner._check_prerequisites(env_mode="acme")[0]


def test_listing_backends_does_not_import_them(
    backend_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch, acme="skillevaluator_missing_backend_module:SPEC")

    assert tier3_environments.all_env_modes()[-1] == "acme"
    with pytest.raises(EnvironmentBackendError, match="could not be loaded"):
        tier3_environments.load_custom_backend("acme")


def test_entry_point_may_name_a_factory(backend_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    _install(monkeypatch, acme=f"{_MODULE}:make_spec", junk=f"{_MODULE}:broken")

    assert tier3_environments.load_custom_backend("acme") == _spec()
    with pytest.raises(EnvironmentBackendError, match="must provide an EnvironmentBackendSpec"):
        tier3_environments.load_custom_backend("junk")


def test_backend_mode_dispatches_through_harbor_import_paths(acme: types.ModuleType) -> None:
    command = build_harbor_run_command(
        dataset_path="/tmp/dataset",
        agent="claude-code",
        job_name="acme",
        env_mode="acme",
        environment_kwargs={"endpoint": "https://sandbox.example", "lease_seconds": 600},
    )

    assert command[command.index("--env") + 1] == _ENV_IMPORT_PATH
    assert command[command.index("--agent") + 1] == _AGENT_IMPORT_PATH
    ek = [command[index + 1] for index, value in enumerate(command) if value == "--ek"]
    assert ek == [f"endpoint={json.dumps('https://sandbox.example')}", "lease_seconds=600"]

    stock = build_harbor_run_command(dataset_path="/tmp/dataset", agent="opencode", job_name="acme", env_mode="acme")
    assert stock[stock.index("--agent") + 1] == "opencode"


def test_backend_agent_wrappers_are_the_only_extra_agent_paths_allowed(acme: types.ModuleType) -> None:
    command = build_harbor_run_command(
        dataset_path="/tmp/dataset",
        agent="claude-code",
        job_name="acme",
        env_mode="acme",
        agent_import_path=_AGENT_IMPORT_PATH,
    )
    assert command[command.index("--agent") + 1] == _AGENT_IMPORT_PATH

    with pytest.raises(ValueError, match="agent_import_path is supported only"):
        build_harbor_run_command(
            dataset_path="/tmp/dataset",
            agent="claude-code",
            job_name="acme",
            env_mode="acme",
            agent_import_path="attacker.module:Agent",
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"unknown": "x"}, "does not accept environment kwarg(s): unknown"),
        ({"pool": "shared"}, "reserved for Harbor runtime policy: pool"),
        ({"network_policy": "open"}, "reserved for Harbor runtime policy: network_policy"),
        ({"session_id": "s"}, "reserved for Harbor runtime policy: session_id"),
    ],
)
def test_backend_kwargs_follow_evaluator_policy(acme: types.ModuleType, kwargs: dict[str, str], message: str) -> None:
    with pytest.raises(ValueError, match=message.replace("(", r"\(").replace(")", r"\)")):
        build_harbor_run_command(
            dataset_path="/tmp/dataset",
            agent="opencode",
            job_name="acme",
            env_mode="acme",
            environment_kwargs=kwargs,
        )


def test_backend_kwargs_cannot_carry_secrets(acme: types.ModuleType) -> None:
    with pytest.raises(ValueError, match="secret-bearing"):
        build_harbor_run_command(
            dataset_path="/tmp/dataset",
            agent="opencode",
            job_name="acme",
            env_mode="acme",
            environment_kwargs={"api_token": "rt-123456"},
        )


def test_backend_prerequisites_run_required_kwargs_hook_and_harbor_preflight(acme: types.ModuleType) -> None:
    assert runner._check_prerequisites(env_mode="acme", environment_kwargs={}) == [
        "Harbor environment 'acme' requires non-empty string --environment-kwarg for: endpoint"
    ]
    assert _AcmeEnvironment.preflight_calls == 0

    assert runner._check_prerequisites(env_mode="acme", environment_kwargs={"endpoint": "down"}) == [
        "acme endpoint is down"
    ]
    assert _AcmeEnvironment.preflight_calls == 0

    assert runner._check_prerequisites(env_mode="acme", environment_kwargs={"endpoint": "https://up"}) == []
    assert _AcmeEnvironment.preflight_calls == 1

    _AcmeEnvironment.preflight_error = RuntimeError("ACME_SANDBOX_URL is not set")
    assert runner._check_prerequisites(env_mode="acme", environment_kwargs={"endpoint": "https://up"}) == [
        "Custom execution backend 'acme' is not ready: ACME_SANDBOX_URL is not set"
    ]


def test_backend_load_failure_is_a_prerequisite_error_with_install_hint(
    backend_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    _install(monkeypatch, acme="skillevaluator_missing_backend_module:SPEC")

    [error] = runner._check_prerequisites(env_mode="acme")
    assert "could not be loaded" in error
    assert "Reinstall the package that provides custom execution backend 'acme'" in error


def test_backend_host_variables_reach_harbor_and_are_protected_from_skills(acme: types.ModuleType) -> None:
    assert runner._env_mode_host_env_vars("acme") == frozenset({"ACME_SANDBOX_TOKEN", "ACME_SANDBOX_URL"})
    assert runner._env_mode_host_env_vars("daytona") == runner._HARBOR_ENV_MODE_VARS["daytona"]

    resolved, errors = runner._resolve_runtime_env({"ACME_SANDBOX_URL": "https://attacker.example"}, env_mode="acme")
    assert resolved == {}
    assert errors == ["harbor.runtime_env.ACME_SANDBOX_URL controls the host process and is not allowed"]


@pytest.mark.parametrize(
    "name",
    [
        "ANTHROPIC_API_KEY",
        "ANTHROPIC_AUTH_TOKEN",
        "CLAUDE_CODE_OAUTH_TOKEN",
        "CLAUDE_FORCE_OAUTH",
        "CODEX_AUTH_JSON_PATH",
        "CODEX_FORCE_AUTH_JSON",
        "OPENAI_BASE_URL",
        "OPENCODE_CONFIG",
        "GITHUB_TOKEN",
        "GH_TOKEN",
        "HF_TOKEN",
        "NVIDIA_API_KEY",
        "DAYTONA_API_KEY",
        "MODAL_TOKEN_SECRET",
        "RUNTA_TOKEN",
        "AWS_SECRET_ACCESS_KEY",
        "LD_PRELOAD",
        "SKILLEVALUATOR_X",
    ],
)
def test_backends_cannot_forward_evaluator_owned_variables(
    backend_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    backend_module.SPEC = _spec(host_env_vars=frozenset({"ACME_SANDBOX_URL", name}))  # type: ignore[attr-defined]
    _install(monkeypatch, acme=f"{_MODULE}:SPEC")

    with pytest.raises(EnvironmentBackendError, match=f"may not forward evaluator-owned variable.*{name}"):
        build_harbor_run_command(dataset_path="/tmp/dataset", agent="opencode", job_name="acme", env_mode="acme")
    assert runner._env_mode_host_env_vars("acme") == frozenset()
    assert "may not forward" in runner._check_prerequisites(env_mode="acme")[0]


def test_backend_compose_support_gates_staged_tasks(
    acme: types.ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    task = tmp_path / "task-a"
    (task / "environment").mkdir(parents=True)
    (task / "environment" / "Dockerfile").write_text("FROM python:3.12-slim\n", encoding="utf-8")
    assert runner._staged_task_environment_error("acme", [task]) is None

    (task / "environment" / "docker-compose.yaml").write_text("services: {}\n", encoding="utf-8")
    assert "cannot run Docker Compose task 'task-a'" in str(runner._staged_task_environment_error("acme", [task]))

    acme.SPEC = _spec(supports_docker_compose=True)  # type: ignore[attr-defined]
    _install(monkeypatch, acme=f"{_MODULE}:SPEC")
    assert runner._staged_task_environment_error("acme", [task]) is None


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"import_path": "not-an-import-path"}, "import_path"),
        ({"prerequisites": "nope"}, "prerequisites"),
        ({"environment_kwargs": "endpoint"}, "environment_kwargs"),
        ({"required_environment_kwargs": ("region",)}, "must also be accepted: region"),
        ({"host_env_vars": frozenset({"BAD-NAME"})}, "host_env_vars"),
        ({"host_env_vars": frozenset({"acme_sandbox_url"})}, "upper-case"),
        ({"host_env_vars": frozenset({"Acme_Sandbox_Url"})}, "upper-case"),
        ({"host_path_env_vars": frozenset({"ACME_CONFIG"})}, "host_path_env_vars must also be host_env_vars"),
        ({"host_path_environment_kwargs": frozenset({"kubeconfig"})}, "must also be environment_kwargs"),
        ({"agent_import_paths": {"claude-code": "nope"}}, "agent_import_paths"),
    ],
)
def test_backend_descriptors_are_validated(overrides: dict[str, object], message: str) -> None:
    with pytest.raises(EnvironmentBackendError, match=message):
        _spec(**overrides)


def test_cli_offers_installed_custom_backends(tmp_path: Path) -> None:
    dist = tmp_path / "skillevaluator_acme_backend-0.0.0.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text("Metadata-Version: 2.1\nName: skillevaluator-acme-backend\nVersion: 0.0.0\n")
    (dist / "entry_points.txt").write_text(f"[{ENV_MODE_ENTRY_POINT_GROUP}]\nacme = {_MODULE}:SPEC\n")
    code = (
        "import sys; sys.path.insert(0, sys.argv[1]);"
        "from skillevaluator.cli import ENV_MODE_CHOICE; print(ENV_MODE_CHOICE.choices[-1])"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, str(tmp_path)], capture_output=True, text=True, check=True, timeout=120
    )

    assert result.stdout.strip() == "acme"


def test_backends_may_forward_kubeconfig_and_their_own_backend_settings(
    backend_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    host = frozenset({"KUBECONFIG", "ACME_SANDBOX_URL", "ACME_SANDBOX_TOKEN", "HTTPS_PROXY"})
    backend_module.SPEC = _spec(host_env_vars=host)  # type: ignore[attr-defined]
    _install(monkeypatch, acme=f"{_MODULE}:SPEC")

    assert runner._env_mode_host_env_vars("acme") == host


def test_backend_path_settings_are_anchored_before_harbor_changes_directory(
    backend_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from skillevaluator.provider_config import ProviderConfig

    backend_module.SPEC = _spec(  # type: ignore[attr-defined]
        environment_kwargs=frozenset({"endpoint", "ca_file"}),
        host_env_vars=frozenset({"ACME_CONFIG", "ACME_SANDBOX_URL"}),
        host_path_env_vars=frozenset({"ACME_CONFIG"}),
        host_path_environment_kwargs=frozenset({"ca_file"}),
    )
    _install(monkeypatch, acme=f"{_MODULE}:SPEC")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ACME_CONFIG", "acme/config.yaml")
    monkeypatch.setenv("ACME_SANDBOX_URL", "https://sandbox.example")

    environment = runner._harbor_subprocess_environment(
        env_mode="acme",
        provider=ProviderConfig(
            provider="anthropic", model="claude", api_key="sk-ant", base_url=None, litellm_model="m", region=None
        ),
        configured_runtime_env={},
        provider_env={},
    )
    assert environment["ACME_CONFIG"] == str(tmp_path / "acme" / "config.yaml")
    assert environment["ACME_SANDBOX_URL"] == "https://sandbox.example"

    command = build_harbor_run_command(
        dataset_path="/tmp/dataset",
        agent="opencode",
        job_name="acme",
        env_mode="acme",
        environment_kwargs={"endpoint": "https://x", "ca_file": "certs/ca.pem"},
    )
    ek = [command[index + 1] for index, value in enumerate(command) if value == "--ek"]
    assert f"ca_file={json.dumps(str(tmp_path / 'certs' / 'ca.pem'))}" in ek


def test_install_hint_is_shown_when_the_environment_dependencies_are_missing(
    backend_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_import = runner.importlib.import_module

    def import_module(name: str, *args: object, **kwargs: object) -> object:
        if name == "skillevaluator_test_backend_env":
            raise ModuleNotFoundError("No module named 'acme_sdk'", name="acme_sdk")
        return real_import(name, *args, **kwargs)

    backend_module.SPEC = _spec(import_path="skillevaluator_test_backend_env:AcmeEnvironment")  # type: ignore[attr-defined]
    _install(monkeypatch, acme=f"{_MODULE}:SPEC")
    monkeypatch.setattr(runner.importlib, "import_module", import_module)

    [error] = runner._check_prerequisites(env_mode="acme", environment_kwargs={"endpoint": "https://up"})
    assert "needs optional dependencies: No module named 'acme_sdk'" in error
    assert "Install 'skillevaluator-acme'." in error
    assert _AcmeEnvironment.preflight_calls == 0


def test_install_hint_is_shown_when_harbor_wraps_the_import_failure(
    acme: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    def _wrapped(_cls: object, _type: object, import_path: str | None = None) -> None:
        try:
            raise ModuleNotFoundError("No module named 'acme_sdk'")
        except ModuleNotFoundError as exc:
            raise ValueError(f"Failed to import module '{import_path}'") from exc

    from harbor.environments.factory import EnvironmentFactory

    monkeypatch.setattr(EnvironmentFactory, "run_preflight", classmethod(_wrapped))
    [error] = runner._check_prerequisites(env_mode="acme", environment_kwargs={"endpoint": "https://up"})
    assert "needs optional dependencies: No module named 'acme_sdk'" in error
    assert "Install 'skillevaluator-acme'." in error


def _write_dist(root: Path, name: str, value: str) -> None:
    dist = root / f"{name.replace('-', '_')}-1.0.dist-info"
    dist.mkdir(parents=True)
    (dist / "METADATA").write_text(f"Metadata-Version: 2.1\nName: {name}\nVersion: 1.0\n")
    (dist / "entry_points.txt").write_text(f"[{ENV_MODE_ENTRY_POINT_GROUP}]\nshared = {value}\n")


def test_a_mode_name_claimed_by_two_packages_is_ambiguous(
    backend_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _write_dist(tmp_path, "aaa-backend", f"{_MODULE}:SPEC")
    _write_dist(tmp_path, "zzz-backend", f"{_MODULE}:make_spec")
    monkeypatch.syspath_prepend(str(tmp_path))
    tier3_environments.clear_custom_backend_cache()

    assert "shared" in tier3_environments.custom_backend_names()
    with pytest.raises(
        EnvironmentBackendError, match=r"registered by more than one package: aaa-backend 1\.0, zzz-backend 1\.0"
    ):
        tier3_environments.load_custom_backend("shared")
    [error] = runner._check_prerequisites(env_mode="shared")
    assert "aaa-backend 1.0, zzz-backend 1.0" in error


def test_public_benchmark_cards_do_not_name_custom_backends(acme: types.ModuleType) -> None:
    from skillevaluator.reporting import benchmark

    assert benchmark._publication_safe_environment("acme") == "Isolated sandbox"
    assert benchmark._publication_safe_environment("docker") == "docker"


def test_identical_targets_from_two_packages_are_still_ambiguous(
    backend_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _write_dist(tmp_path, "aaa-backend", f"{_MODULE}:SPEC")
    _write_dist(tmp_path, "zzz-backend", f"{_MODULE}:SPEC")
    monkeypatch.syspath_prepend(str(tmp_path))
    tier3_environments.clear_custom_backend_cache()

    with pytest.raises(EnvironmentBackendError, match=r"aaa-backend 1\.0, zzz-backend 1\.0"):
        tier3_environments.load_custom_backend("shared")


def test_one_package_seen_twice_on_the_path_is_one_claim(
    backend_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _write_dist(tmp_path / "first", "aaa-backend", f"{_MODULE}:SPEC")
    _write_dist(tmp_path / "second", "aaa-backend", f"{_MODULE}:SPEC")
    monkeypatch.syspath_prepend(str(tmp_path / "first"))
    monkeypatch.syspath_prepend(str(tmp_path / "second"))
    tier3_environments.clear_custom_backend_cache()

    assert tier3_environments.load_custom_backend("shared") == _spec()


def test_backends_may_accept_allow_workload_identity_and_receive_it(
    backend_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    backend_module.SPEC = _spec(environment_kwargs=frozenset({"endpoint", "allow_workload_identity"}))  # type: ignore[attr-defined]
    _install(monkeypatch, acme=f"{_MODULE}:SPEC")

    command = build_harbor_run_command(
        dataset_path="/tmp/dataset",
        agent="opencode",
        job_name="acme",
        env_mode="acme",
        environment_kwargs={"allow_workload_identity": True},
    )
    assert "allow_workload_identity=true" in [command[i + 1] for i, value in enumerate(command) if value == "--ek"]


_SYNTHETIC_TOKEN = "acme-synthetic-token-7f3c9e1a"


def test_backend_load_failures_are_redacted_everywhere_they_surface(
    backend_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken_factory() -> EnvironmentBackendSpec:
        raise RuntimeError(f"cannot reach https://ci:{_SYNTHETIC_TOKEN}@sandbox.example/v1 with {_SYNTHETIC_TOKEN}")

    backend_module.broken_factory = broken_factory  # type: ignore[attr-defined]
    monkeypatch.setenv("ACME_SANDBOX_TOKEN", _SYNTHETIC_TOKEN)
    _install(monkeypatch, acme=f"{_MODULE}:broken_factory")

    [doctor_error] = runner._check_prerequisites(env_mode="acme")
    policy_error = runner._environment_kwarg_policy_error("acme", {"endpoint": "https://x"})
    with pytest.raises(EnvironmentBackendError) as raised:
        build_harbor_run_command(dataset_path="/tmp/dataset", agent="opencode", job_name="acme", env_mode="acme")

    for message in (doctor_error, str(policy_error), str(raised.value)):
        assert "could not be loaded" in message
        assert _SYNTHETIC_TOKEN not in message
        assert "<redacted>" in message


def _codex_wrappers(module: types.ModuleType) -> None:
    from harbor.agents.installed.codex import Codex

    from skillevaluator.tier3.harbor.local_agents import SkillEvaluatorGatewayCodex

    class PrebuiltGatewayCodex(SkillEvaluatorGatewayCodex):
        """Keeps gateway routing; adapts setup for a non-root backend."""

    class PrebuiltCodex(Codex):
        """Adapts setup but drops the gateway routing."""

    module.PrebuiltGatewayCodex = PrebuiltGatewayCodex  # type: ignore[attr-defined]
    module.PrebuiltCodex = PrebuiltCodex  # type: ignore[attr-defined]


def _provider(name: str) -> object:
    from skillevaluator.provider_config import ProviderConfig

    return ProviderConfig(
        provider=name, model="m", api_key="k", base_url="https://gateway.example/v1", litellm_model="m", region=None
    )


@pytest.mark.parametrize("provider", ["openai", "openai-compatible"])
def test_runtime_keeps_a_backend_codex_wrapper_that_extends_the_provider_wrapper(
    backend_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch, provider: str
) -> None:
    _codex_wrappers(backend_module)
    wrapper = f"{_MODULE}:PrebuiltGatewayCodex"
    backend_module.SPEC = _spec(agent_import_paths={"codex": wrapper})  # type: ignore[attr-defined]
    _install(monkeypatch, acme=f"{_MODULE}:SPEC")

    selected = runner._agent_import_path(_provider(provider), "codex", "acme")
    assert selected == wrapper
    command = build_harbor_run_command(
        dataset_path="/tmp/dataset", agent="codex", job_name="acme", env_mode="acme", agent_import_path=selected
    )
    assert command[command.index("--agent") + 1] == wrapper


def test_runtime_refuses_a_backend_codex_wrapper_that_drops_gateway_routing(
    backend_module: types.ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    _codex_wrappers(backend_module)
    backend_module.SPEC = _spec(agent_import_paths={"codex": f"{_MODULE}:PrebuiltCodex"})  # type: ignore[attr-defined]
    _install(monkeypatch, acme=f"{_MODULE}:SPEC")

    assert runner._agent_import_path(_provider("openai"), "codex", "acme") == f"{_MODULE}:PrebuiltCodex"
    with pytest.raises(
        ValueError, match=r"requires skillevaluator\.tier3\.harbor\.local_agents:SkillEvaluatorGatewayCodex"
    ):
        runner._agent_import_path(_provider("openai-compatible"), "codex", "acme")


def test_runtime_uses_the_provider_wrapper_when_the_backend_names_none(acme: types.ModuleType) -> None:
    assert (
        runner._agent_import_path(_provider("openai-compatible"), "codex", "acme") == runner._GATEWAY_CODEX_IMPORT_PATH
    )
    assert runner._agent_import_path(_provider("openai"), "codex", "acme") is None
