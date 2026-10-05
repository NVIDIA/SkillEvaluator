# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Launch the optional dashboard without importing UI dependencies into the CLI."""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import click


def launch_dashboard(paths: tuple[Path, ...], *, port: int = 8501, no_browser: bool = False) -> int:
    """Run Streamlit on loopback with explicit paths and usage telemetry disabled."""
    if importlib.util.find_spec("streamlit") is None:
        raise click.ClickException('Install the dashboard extra: pip install "skillevaluator[dashboard]"')
    app = Path(__file__).with_name("dashboard_app.py")
    command = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app),
        "--server.address=127.0.0.1",
        f"--server.port={port}",
        f"--server.headless={str(no_browser).lower()}",
        "--browser.gatherUsageStats=false",
        "--theme.primaryColor=#76b900",
        "--",
        *(str(path.resolve()) for path in paths),
    ]
    try:
        return subprocess.call(command)
    except KeyboardInterrupt:
        return 0
