# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CLI output remains available when stdout cannot encode a report glyph."""

from __future__ import annotations

from io import BytesIO, TextIOWrapper

from rich.console import Console

from skillevaluator import cli as cli_module


def test_cli_replaces_characters_not_supported_by_stdout_encoding(monkeypatch) -> None:
    output = BytesIO()
    stdout = TextIOWrapper(output, encoding="cp936", errors="strict")
    monkeypatch.setattr(cli_module.sys, "stdout", stdout)
    monkeypatch.setattr(cli_module, "setup_logging", lambda **_kwargs: None)

    cli_module.cli.callback(verbose=False)
    Console(file=stdout, force_terminal=False).print("  • Errors: 1")
    stdout.flush()

    assert stdout.errors == "replace"
    assert output.getvalue().decode("cp936").startswith("  ? Errors: 1")
