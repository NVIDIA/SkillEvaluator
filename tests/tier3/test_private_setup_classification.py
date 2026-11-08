# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Private setup classification retains Harbor's routing and task failure types."""

from __future__ import annotations

import asyncio
import os
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from harbor.agents.installed.base import BaseInstalledAgent, NetworkConnectionError
from harbor.agents.installed.opencode import OpenCode
from harbor.utils.import_path import import_symbol

from skillevaluator.tier3.harbor import CONTAINER_AGENT_IMPORT_PATHS, DOCKER_AGENT_IMPORT_PATHS, runner
from skillevaluator.tier3.harbor.local_agents import AgentSetupNetworkError, SetupNetworkErrorAgent
from skillevaluator.tier3.plugin_native import (
    NATIVE_AGENT_IMPORT_PATHS,
    PRIVATE_AGENT_IMPORT_PATHS,
    native_agent_import_path,
    private_agent_import_path,
)


@pytest.mark.parametrize(("key", "path"), list(PRIVATE_AGENT_IMPORT_PATHS.items()))
def test_private_wrapper_retains_base_methods_and_has_no_native_launcher(key: tuple, path: str) -> None:
    cls = import_symbol(path)
    base = cls.__bases__[1]
    assert cls.__bases__[0] is SetupNetworkErrorAgent
    if key[1] is not None:
        assert issubclass(cls, import_symbol(key[1]))
    else:
        module = key[0].replace("-", "_")
        stock_name = {"claude-code": "ClaudeCode", "codex": "Codex", "opencode": "OpenCode", "hermes": "Hermes"}[key[0]]
        assert base is import_symbol(f"harbor.agents.installed.{module}:{stock_name}")
    for method in ("install", "run", "exec_as_agent"):
        assert getattr(cls, method) is getattr(base, method)
    assert not hasattr(cls, "skilleval_native_command")
    assert private_agent_import_path(*key) == path
    assert private_agent_import_path(key[0], path) == path


@pytest.mark.parametrize(("key", "path"), list(NATIVE_AGENT_IMPORT_PATHS.items()))
def test_native_wrapper_is_not_wrapped_twice(key: tuple, path: str) -> None:
    assert private_agent_import_path(key[0], path) == path
    assert issubclass(import_symbol(path), SetupNetworkErrorAgent)


class _NativeHooks:
    """The private runtime's agent routing, with every agent loading the plugin natively."""

    def with_agent_import_path(self, agent: str, base_import_path: str | None) -> str:
        return native_agent_import_path(agent, base_import_path)

    def baseline_agent_import_path(self, agent: str, base_import_path: str | None) -> str | None:
        return private_agent_import_path(agent, base_import_path)


@pytest.mark.parametrize(
    ("env_mode", "default_paths", "baseline", "with_arm"),
    [
        ("docker", DOCKER_AGENT_IMPORT_PATHS, "PrivateInstalledOpenCode", "NativeInstalledOpenCode"),
        ("gke", CONTAINER_AGENT_IMPORT_PATHS, "PrivateOpenCode", "NativeOpenCode"),
    ],
)
def test_opencode_plugin_arms_share_the_environment_default_agent(
    monkeypatch: pytest.MonkeyPatch,
    env_mode: str,
    default_paths: dict[str, str],
    baseline: str,
    with_arm: str,
) -> None:
    # Both plugin arms must set OpenCode up as an ordinary run in that
    # environment does: through its default import path, else Harbor's stock agent.
    monkeypatch.setattr(runner, "_private_runtime_hooks", _NativeHooks)
    default_path = default_paths.get("opencode")
    default_agent = import_symbol(default_path) if default_path is not None else OpenCode

    baseline_path, with_path = runner._private_agent_paths(["opencode"], {}, env_mode)["opencode"]

    assert baseline_path == f"skillevaluator.tier3.harbor.native_agents:{baseline}"
    assert with_path == f"skillevaluator.tier3.harbor.native_agents:{with_arm}"
    for path in (baseline_path, with_path):
        cls = import_symbol(path)
        assert issubclass(cls, default_agent)
        assert cls.install is default_agent.install
        assert cls.ensure_system_dependencies is default_agent.ensure_system_dependencies


def test_unknown_private_wrapper_does_not_silently_replace_custom_routing() -> None:
    assert private_agent_import_path("codex", "custom.module:MyAgent") == "custom.module:MyAgent"


class _ProcessEnvironment:
    """Run fixture commands in a temporary directory, retaining actual exit/output."""

    def __init__(self, root: Path):
        self.root = root

    async def exec(self, command: str, **_kwargs):
        command = command.replace("/installed-agent", str(self.root / "installed-agent"))
        completed = subprocess.run(
            ["bash", "-c", command],
            cwd=self.root,
            env={"PATH": os.environ["PATH"], "HOME": str(self.root)},
            capture_output=True,
            text=True,
            timeout=15,
        )
        return SimpleNamespace(return_code=completed.returncode, stdout=completed.stdout, stderr=completed.stderr)


@pytest.mark.parametrize("path", sorted(set(PRIVATE_AGENT_IMPORT_PATHS.values())))
def test_process_diagnostic_is_setup_only_even_with_same_failure_after_dispatch(
    tmp_path: Path, monkeypatch, path: str
) -> None:
    cls = import_symbol(path)
    fixture = tmp_path / "installer.py"
    fixture.write_text("import sys\nprint('curl: (6) Could not resolve host: fixture', file=sys.stderr)\nsys.exit(6)\n")
    command = f"{shlex.quote(sys.executable)} {shlex.quote(str(fixture))}"

    async def fixture_install(self, environment):
        await BaseInstalledAgent.exec_as_agent(self, environment, command)

    # Replace only the external package download; the real Harbor setup,
    # subprocess output classifier, and private mixin run unchanged.
    monkeypatch.setattr(cls, "install", fixture_install)
    agent = cls(logs_dir=tmp_path / "logs", model_name="openai/fixture")
    environment = _ProcessEnvironment(tmp_path)
    with pytest.raises(AgentSetupNetworkError) as setup_error:
        asyncio.run(agent.setup(environment))
    assert isinstance(setup_error.value.__cause__, NetworkConnectionError)
    with pytest.raises(NetworkConnectionError):
        asyncio.run(BaseInstalledAgent.exec_as_agent(agent, environment, command))
    assert (tmp_path / "logs/setup").is_dir()
