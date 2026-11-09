#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: T201 — command-line helper diagnostics
"""Prepare only native agent runtimes, then doctor the CI-selected environment."""

from __future__ import annotations

import os
import subprocess
import sys

from skillevaluator.tier3.commands import resolve_agents
from skillevaluator.tier3.harbor.local_runtime import managed_install_argv, managed_runtime_prefix


def install_agent(agent: str) -> None:
    argv = managed_install_argv(agent)
    if agent == "opencode":
        # Unlike interactive local setup, CI never resolves an unpinned npm package.
        # Docker runs install the same release through Harbor's agent kwargs.
        from skillevaluator.tier3.harbor.runner import OPENCODE_VERSION

        argv[-1] = f"opencode-ai@{OPENCODE_VERSION}"
    subprocess.run(argv, check=True, timeout=600)
    print(f"Installed {agent} in {managed_runtime_prefix(agent)}")


def main() -> int:
    mode = os.environ.get("SKILL_EVAL_ENV_MODE", "docker")
    if mode not in {"docker", "local"}:
        print("Three-tier CI supports docker and local environments only.")
        return 2
    try:
        agents = resolve_agents(
            os.environ.get("SKILL_EVAL_AGENTS") or None, provider=os.environ.get("SKILL_EVAL_PROVIDER", "openai")
        )
        if mode == "local":
            for agent in agents:
                install_agent(agent)
        doctor_env = dict(os.environ)
        doctor_env["SKILL_EVAL_LLM_PROVIDER"] = os.environ.get("SKILL_EVAL_PROVIDER", "openai")
        doctor_env["PYTHON_DOTENV_DISABLED"] = "1"
        if model := os.environ.get("SKILL_EVAL_MODEL"):
            doctor_env["SKILL_EVAL_LLM_MODEL"] = model
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                "-m",
                "skillevaluator.cli",
                "doctor",
                "--env-mode",
                mode,
                "--agents",
                ",".join(agents),
            ],
            check=False,
            env=doctor_env,
            timeout=180,
        )
        return result.returncode
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        # Do not print subprocess output or provider credentials.
        print(f"Native CI preflight needs action: {type(exc).__name__}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
