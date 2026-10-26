# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Lazy launcher: the base CLI never imports web-server dependencies."""

from __future__ import annotations

import threading
import webbrowser
from pathlib import Path

import click


def launch(port: int, data_dir: Path | None, no_browser: bool) -> None:
    try:
        import uvicorn

        from skillevaluator.studio.app import create_app
    except ImportError as exc:
        raise click.ClickException('Install the local UI first: pip install "skillevaluator[ui]"') from exc
    url = f"http://127.0.0.1:{port}"
    click.echo(f"SkillEvaluator Studio: {url}")
    click.echo("Press Ctrl+C to stop. Skills and reports stay in your local workspace.")
    app = create_app(data_dir)
    if not no_browser:
        threading.Timer(0.9, lambda: webbrowser.open(url)).start()
    uvicorn.run(app, host="127.0.0.1", port=port, access_log=False, log_level="warning", server_header=False)
