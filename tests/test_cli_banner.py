# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The welcome banner must not change the CLI's output contracts."""

import os
import shutil
import sys

import click
import pytest
from click.testing import CliRunner

from skillevaluator import __version__
from skillevaluator.cli import cli
from skillevaluator.cli_help import render_help


@pytest.fixture
def terminal(monkeypatch):
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(sys.stderr, "isatty", lambda: True)
    monkeypatch.setattr(shutil, "get_terminal_size", lambda: os.terminal_size((120, 40)))
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.setenv("COLORTERM", "truecolor")
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.delenv("FORCE_COLOR", raising=False)


def root_help(width=80, *, color=False):
    with cli.make_context("skillevaluator", [], resilient_parsing=True, color=color) as ctx:
        return render_help(cli, ctx, width=width)


@pytest.mark.parametrize("terminal_width", [115, 120])
def test_interactive_root_help_has_wordmark_and_installed_version(terminal, monkeypatch, terminal_width):
    monkeypatch.setattr(shutil, "get_terminal_size", lambda: os.terminal_size((terminal_width, 40)))
    output = root_help()
    banner, help_text = output.split("Usage:", 1)

    assert len([line for line in banner.splitlines() if "█" in line]) == 5
    assert "╚══════╝" in banner
    assert f"v{__version__}" in banner
    assert "Static checks · Deduplication · Live evaluation" in banner
    assert "[OPTIONS] COMMAND [ARGS]..." in help_text
    assert all(len(line) <= terminal_width for line in banner.splitlines())
    assert all(len(line) <= 80 for line in help_text.splitlines())


@pytest.mark.parametrize("formatter_width,terminal_width", [(78, 40), (78, 80), (78, 114)])
def test_narrow_help_uses_compact_banner(terminal, monkeypatch, formatter_width, terminal_width):
    monkeypatch.setattr(shutil, "get_terminal_size", lambda: os.terminal_size((terminal_width, 40)))
    banner = root_help(formatter_width).split("Usage:", 1)[0]

    assert "█" not in banner
    assert "SKILLEVALUATOR" in banner
    assert f"v{__version__}" in banner
    assert all(len(line) <= terminal_width for line in banner.splitlines())


@pytest.mark.parametrize("redirected", ["stdout", "stderr"])
def test_redirected_stream_suppresses_banner_even_with_forced_color(terminal, monkeypatch, redirected):
    monkeypatch.setattr(getattr(sys, redirected), "isatty", lambda: False)
    monkeypatch.setenv("FORCE_COLOR", "1")

    assert root_help(color=None).startswith("\x1b[1mUsage:")
    assert "█" not in root_help(color=None)


def test_banner_respects_terminal_color_capabilities(terminal, monkeypatch):
    assert "\x1b[1;38;2;118;185;0m" in root_help(color=None)
    monkeypatch.setenv("NO_COLOR", "1")
    plain = root_help(color=None)
    assert "\x1b[" not in plain
    assert "█" in plain
    for term in ("dumb", "unknown"):
        monkeypatch.setenv("TERM", term)
        assert root_help().startswith("Usage:")


def test_nested_help_does_not_show_banner(terminal):
    parent = click.Context(cli, info_name="skillevaluator")
    tier = cli.get_command(parent, "tier1")
    ctx = click.Context(tier, info_name="tier1", parent=parent, color=False)

    assert render_help(tier, ctx).startswith("Usage: skillevaluator tier1")


@pytest.mark.parametrize(
    "args,exit_code,stream",
    [([], 2, "stderr"), (["--help"], 0, "stdout"), (["-h"], 0, "stdout"), (["--version"], 0, "stdout")],
)
def test_noninteractive_invocations_preserve_streams_and_exit_codes(args, exit_code, stream):
    result = CliRunner().invoke(cli, args)

    assert result.exit_code == exit_code
    assert getattr(result, stream)
    assert not getattr(result, "stderr" if stream == "stdout" else "stdout")
    assert "█" not in result.output
