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


def _passing_plugin(tmp_path):
    source = tmp_path / "demo"
    manifest = source / ".claude-plugin" / "plugin.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text('{"name": "demo"}', encoding="utf-8")
    return source


@pytest.mark.parametrize("probe", [(), ("--no-probe-mcp",)], ids=["absent", "disabled"])
def test_validate_refuses_grants_without_probes_before_any_tier(tmp_path, probe):
    source = _passing_plugin(tmp_path)
    args = ["validate", str(source), "--tiers", "1", "--skip", "security", "-r", "json"]
    # The plugin passes Tier 1, so only the grant decides the exit code.
    passing = CliRunner().invoke(cli, [*args, "-o", str(tmp_path / "passing")])
    assert passing.exit_code == 0, (passing.output, passing.exception)

    result = CliRunner().invoke(cli, [*args, *probe, "--probe-mcp-env", "TOKEN", "-o", str(tmp_path / "refused")])

    assert result.exit_code == 2, (result.output, result.exception)
    assert "--probe-mcp-env requires --probe-mcp" in result.output
    assert not (tmp_path / "refused").exists()


@pytest.mark.parametrize("probe", [(), ("--no-probe-mcp",)], ids=["absent", "disabled"])
def test_evaluate_plugin_refuses_grants_without_probes_before_running(tmp_path, monkeypatch, probe):
    from skillevaluator.evaluation.service import EvaluationService

    source = _passing_plugin(tmp_path)
    runs = []

    def execute(self, options, *, plugin_options=None, progress_reporter=None):
        runs.append(plugin_options)
        return {"execution_status": "succeeded"}

    monkeypatch.setattr(EvaluationService, "evaluate_plugin", execute)
    args = ["tier3", "evaluate-plugin", str(source), "--agents", "opencode", "--progress", "off", "-r", "json"]

    result = CliRunner().invoke(cli, [*args, *probe, "--probe-mcp-env", "TOKEN", "-o", str(tmp_path / "refused")])

    assert result.exit_code == 2, (result.output, result.exception)
    assert "--probe-mcp-env requires --probe-mcp" in result.output
    assert runs == []
    assert not (tmp_path / "refused").exists()
