# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prove the base install runs Tier 1 without Harbor or LLM extras.

A subprocess installs an import blocker for every extras-only top-level module,
then imports the CLI and runs static validation. This isolates from any heavy
modules other tests may have already imported into the parent interpreter.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FIXTURE = REPO / "tests" / "fixtures" / "skills" / "simple"

_SUBPROCESS = r"""
import os
import sys

BLOCKED = {
    "anthropic", "boto3", "botocore", "harbor", "openai", "litellm",
    "langgraph", "langgraph_checkpoint", "opentelemetry",
}


class _Blocker:
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in BLOCKED:
            raise ModuleNotFoundError(f"blocked extras-only module: {name}")
        return None


sys.meta_path.insert(0, _Blocker())

from click.testing import CliRunner

from skillevaluator.cli import cli

help_result = CliRunner().invoke(cli, ["--help"])
assert help_result.exit_code == 0, help_result.output

for name in (
    "SKILL_EVAL_LLM_PROVIDER", "SKILL_EVAL_LLM_MODEL",
    "SKILL_EVAL_LLM_API_KEY", "SKILL_EVAL_LLM_BASE_URL",
    "NVIDIA_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
):
    os.environ.pop(name, None)

models_result = CliRunner().invoke(cli, ["models"])
assert models_result.exit_code == 1, models_result.output
assert "No provider is configured" in models_result.output, models_result.output

validate_result = CliRunner().invoke(
    cli, ["validate", FIXTURE, "--no-llm", "--checks", "schema,quality,lint"]
)
assert validate_result.exit_code == 0, validate_result.output

leaked = sorted(m for m in BLOCKED if m in sys.modules)
assert not leaked, f"base path imported extras-only modules: {leaked}"
print("BASE_OK")
"""


def test_base_install_runs_tier1_without_extras() -> None:
    code = _SUBPROCESS.replace("FIXTURE", repr(str(FIXTURE)))
    env = dict(os.environ, PYTHONPATH=str(REPO / "src"))
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env=env,
    )
    assert proc.returncode == 0, f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
    assert "BASE_OK" in proc.stdout
