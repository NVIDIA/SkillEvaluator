# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Probe grants are validated before either public plugin command starts work."""

import importlib

import click
import pytest
from click.testing import CliRunner

from skillevaluator.cli import cli


def _replace_command_entry(monkeypatch, command, callback):
    if command == ["validate"]:
        monkeypatch.setattr(importlib.import_module("skillevaluator.cli"), "_record_validate_json_report", callback)
    else:
        monkeypatch.setattr(importlib.import_module("skillevaluator.cli_plugin"), "_report_output_location", callback)


@pytest.mark.parametrize("command", [["validate"], ["tier3", "evaluate-plugin"]])
@pytest.mark.parametrize(
    "grants",
    [
        ("",),
        ("=host.example",),
        ("TOKEN=",),
        ("TOKEN=host.example:443",),
        ("TOKEN@",),
        ("TOKEN@server\nname",),
        ("A" * 513,),
        tuple(f"TOKEN_{index}" for index in range(65)),
    ],
    ids=[
        "empty",
        "missing-name",
        "missing-host",
        "host-with-port",
        "missing-server",
        "control",
        "oversized",
        "too-many",
    ],
)
def test_probe_grants_fail_as_usage_errors_before_command_work(tmp_path, monkeypatch, command, grants):
    def forbidden(*_args, **_kwargs):
        pytest.fail("Invalid probe grants must be rejected before command work")

    _replace_command_entry(monkeypatch, command, forbidden)
    args = [*command, str(tmp_path), "--probe-mcp"]
    for grant in grants:
        args.extend(("--probe-mcp-env", grant))

    result = CliRunner().invoke(cli, args)

    assert result.exit_code == 2, (result.output, result.exception)
    assert "Invalid value for '--probe-mcp-env'" in result.output
    assert "Traceback" not in result.output


@pytest.mark.parametrize("command", [["validate"], ["tier3", "evaluate-plugin"]])
@pytest.mark.parametrize(
    "grants",
    [
        ("TOKEN", "TOKEN=host.example", "TOKEN@server-name"),
        ("A" * 512,),
        tuple(f"TOKEN_{index}" for index in range(64)),
    ],
    ids=["supported-scopes", "maximum-length", "maximum-count"],
)
def test_probe_grants_accept_supported_scopes_and_bounds(tmp_path, monkeypatch, command, grants):
    seen = []

    def capture(*_args, **_kwargs):
        seen.append(click.get_current_context().params["probe_mcp_env"])
        raise click.exceptions.Exit(0)

    _replace_command_entry(monkeypatch, command, capture)
    args = [*command, str(tmp_path), "--probe-mcp"]
    for grant in grants:
        args.extend(("--probe-mcp-env", grant))

    result = CliRunner().invoke(cli, args)

    assert result.exit_code == 0, (result.output, result.exception)
    assert seen == [grants]
