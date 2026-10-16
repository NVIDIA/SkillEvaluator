# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launch the optional dashboard without importing UI dependencies into the CLI."""

from __future__ import annotations

import importlib.util
import os
import subprocess
import sys
from pathlib import Path

import click

# Streamlit runs from this directory, so the caller's .streamlit/ settings and
# modules cannot replace the dashboard's own.
_APP = Path(__file__).with_name("dashboard_ui") / "skillevaluator_dashboard.py"


def launch_dashboard(paths: tuple[Path, ...], *, port: int | None = None, no_browser: bool = False) -> int:
    """Run Streamlit on loopback with explicit paths and usage telemetry disabled."""
    if importlib.util.find_spec("streamlit") is None:
        raise click.ClickException(
            "Install the dashboard extra: uv tool install --python 3.13 --reinstall "
            '"skillevaluator[all,dashboard] @ git+https://github.com/NVIDIA/SkillEvaluator.git" '
            "(from a checkout: uv run --extra dashboard skillevaluator dashboard, "
            'or python -m pip install -e ".[dashboard]")'
        )
    # -P keeps the working directory off sys.path. An explicit headless=false
    # would trigger the first-run email prompt and override Streamlit's
    # display detection, so the flag is only ever set to true.
    command = [
        sys.executable,
        "-P",
        "-m",
        "streamlit",
        "run",
        str(_APP),
        "--server.address=127.0.0.1",
        # A port on the command line counts as manually set, and Streamlit then
        # exits instead of moving to the next free port, so pass only a chosen one.
        *([f"--server.port={port}"] if port is not None else []),
        "--server.maxUploadSize=16",
        "--server.showEmailPrompt=false",
        "--server.fileWatcherType=none",
        "--server.enableCORS=true",
        "--server.enableXsrfProtection=true",
        "--client.toolbarMode=viewer",
        *(["--server.headless=true"] if no_browser else []),
        "--browser.gatherUsageStats=false",
        "--theme.primaryColor=#76b900",
        "--",
        *(str(path.resolve()) for path in paths),
    ]
    env = {**os.environ, "STREAMLIT_SERVER_ALLOWED_HOSTS": "127.0.0.1 localhost"}
    try:
        code = subprocess.call(command, cwd=str(_APP.parent), env=env)
    except KeyboardInterrupt:
        return 0
    if port is not None and code > 0:
        click.echo(
            f"Streamlit exited with status {code}. If port {port} is in use, choose another with --port.", err=True
        )
    return code
