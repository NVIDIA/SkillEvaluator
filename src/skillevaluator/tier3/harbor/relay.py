# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Experimental: record Tier 3 agent runs with NeMo Relay inside the trial container.

Set ``SKILLEVALUATOR_NEMO_RELAY_BINARY`` to a host path of a Linux ``nemo-relay``
CLI build. Claude Code and Codex launches are wrapped with ``nemo-relay claude``
/ ``nemo-relay codex``; Hermes uses its built-in Relay runtime and only receives
a ``plugins.toml``. Relay writes ATOF events and ATIF trajectories under
``/logs/agent/relay``, next to Harbor's own ``trajectory.json``. OpenCode has no
Relay integration and runs unchanged.
"""

from __future__ import annotations

import os
import shlex
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from harbor.models.trial.paths import EnvironmentPaths

if TYPE_CHECKING:
    from harbor.environments.base import BaseEnvironment

RELAY_BINARY_ENV = "SKILLEVALUATOR_NEMO_RELAY_BINARY"
RELAY_REMOTE_DIR = PurePosixPath(EnvironmentPaths.agent_dir / "relay")
_RELAY_REMOTE_BIN = PurePosixPath("/usr/local/bin/nemo-relay")
_CLAUDE_LAUNCH = "claude --verbose --output-format=stream-json "

_CODEX_LAUNCH = "codex exec "
_CONFIG_TOML = '[agents.claude]\ncommand = "claude"\n\n[agents.codex]\ncommand = "codex"\n'


def _plugins_toml(agent_name: str) -> str:
    return f"""version = 1

[[components]]
kind = "observability"
enabled = true

[components.config]
version = 4
enable_full_payloads = false

[components.config.atof]
enabled = true

[[components.config.atof.sinks]]
type = "file"
output_directory = "{RELAY_REMOTE_DIR.as_posix()}"
filename = "events.jsonl"
mode = "overwrite"

[components.config.atif]
enabled = true
agent_name = "{agent_name}"
output_directory = "{RELAY_REMOTE_DIR.as_posix()}"
filename_template = "trajectory-{{session_id}}.json"
"""


HERMES_PLUGINS_ENV = "HERMES_NEMO_RELAY_PLUGINS_TOML"


def relay_binary() -> Path | None:
    raw = os.environ.get(RELAY_BINARY_ENV, "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser()
    if not path.is_file():
        raise RuntimeError(f"{RELAY_BINARY_ENV} does not point to a file: {path}")
    return path


def is_claude_launch(command: str) -> bool:
    return _CLAUDE_LAUNCH in command


async def write_relay_config(environment: BaseEnvironment, agent_name: str) -> str:
    """Write Relay's plugins.toml into the container; native Relay hosts (Hermes) only need this."""
    relay_dir = shlex.quote(RELAY_REMOTE_DIR.as_posix())
    plugins = f"{RELAY_REMOTE_DIR.as_posix()}/plugins.toml"
    result = await environment.exec(
        command=(
            f"mkdir -p {relay_dir} && chmod 0777 {relay_dir} && "
            f"printf '%s' {shlex.quote(_plugins_toml(agent_name))} > {shlex.quote(plugins)} && chmod 0644 {shlex.quote(plugins)}"
        ),
        user="root",
    )
    if result.return_code != 0:
        raise RuntimeError(f"NeMo Relay config write failed: {(result.stderr or result.stdout or '').strip()[-300:]}")
    return plugins


async def install_relay(environment: BaseEnvironment, binary: Path, agent_name: str = "claude-code") -> None:
    staging = "/tmp/nemo-relay-upload"
    await environment.upload_file(binary, staging)
    relay_dir = shlex.quote(RELAY_REMOTE_DIR.as_posix())
    result = await environment.exec(
        command=(
            f"install -m 0755 {staging} {_RELAY_REMOTE_BIN.as_posix()} && rm -f {staging} && "
            f"mkdir -p {relay_dir} && chmod 0777 {relay_dir} && "
            f"printf '%s' {shlex.quote(_CONFIG_TOML)} > {relay_dir}/config.toml && "
            f"printf '%s' {shlex.quote(_plugins_toml(agent_name))} > {relay_dir}/plugins.toml && "
            f"{_RELAY_REMOTE_BIN.as_posix()} --version"
        ),
        user="root",
    )
    if result.return_code != 0:
        raise RuntimeError(f"NeMo Relay install failed: {(result.stderr or result.stdout or '').strip()[-300:]}")


def wrap_claude_launch(command: str) -> str:
    config = (RELAY_REMOTE_DIR / "config.toml").as_posix()
    return command.replace(
        _CLAUDE_LAUNCH,
        f"{_RELAY_REMOTE_BIN.as_posix()} --config {config} claude -- --verbose --output-format=stream-json ",
        1,
    )


def is_codex_launch(command: str) -> bool:
    return _CODEX_LAUNCH in command and "nemo-relay" not in command


def wrap_codex_launch(command: str) -> str:
    config = (RELAY_REMOTE_DIR / "config.toml").as_posix()
    return command.replace(
        _CODEX_LAUNCH,
        f"{_RELAY_REMOTE_BIN.as_posix()} --config {config} codex -- exec ",
        1,
    )


def relay_env(env: dict[str, str] | None) -> dict[str, str]:
    """Point Relay's Anthropic upstream at the endpoint Claude Code would have used."""
    routed = dict(env or {})
    base_url = routed.get("ANTHROPIC_BASE_URL") or os.environ.get("ANTHROPIC_BASE_URL")
    if base_url:
        routed["NEMO_RELAY_ANTHROPIC_BASE_URL"] = base_url
    return routed
