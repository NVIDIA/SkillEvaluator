# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Kitchen-sink-shaped runtime-only plugins must not silently pass without reports."""

from __future__ import annotations

import json
from pathlib import Path

from click.testing import CliRunner

from skillevaluator.cli import cli
from skillevaluator.evaluation import EvaluationService
from skillevaluator.tier3.plugin_eval import prepare_plugin_eval_package


def _runtime_plugin(root: Path) -> Path:
    root.mkdir()
    (root / "openclaw.plugin.json").write_text(
        json.dumps(
            {
                "id": "kitchen-sink-shaped-runtime-only",
                "name": "Runtime fixture",
                "configSchema": {},
                "contracts": {"tools": ["runtime_tool"], "imageGenerationProviders": ["runtime-image"]},
            }
        ),
        encoding="utf-8",
    )
    (root / "package.json").write_text(json.dumps({"openclaw": {"extensions": ["./index.js"]}}), encoding="utf-8")
    (root / "index.js").write_text("throw new Error('never execute this in a portable evaluation');", encoding="utf-8")
    return root


def test_runtime_only_package_records_an_incomplete_skip(tmp_path: Path) -> None:
    package = prepare_plugin_eval_package(_runtime_plugin(tmp_path / "plugin"), stage_root=tmp_path / "stage")
    assert package.skipped
    assert package.incomplete_skip
    provenance = package.provenance()
    assert provenance["partial"]
    assert "OpenClaw runtime" in provenance["execution_incomplete"]
    assert any(
        row["name"] == "OpenClaw runtime" and row["state"] == "unsupported"
        for row in provenance["component_coverage"]["components"]
    )


def test_runtime_only_cli_writes_all_requested_reports_without_running_an_agent(tmp_path: Path, monkeypatch) -> None:
    root = _runtime_plugin(tmp_path / "plugin")
    reports = tmp_path / "reports"

    def must_not_run(*_args, **_kwargs):
        raise AssertionError("An unexecutable native runtime must not start an agent")

    monkeypatch.setattr(EvaluationService, "evaluate", must_not_run)
    invoked = CliRunner().invoke(
        cli, ["tier3", "evaluate-plugin", str(root), "-r", "json,html,markdown,sarif", "-o", str(reports)]
    )
    assert invoked.exit_code == 1, invoked.output
    assert "INCOMPLETE" in invoked.output
    assert "OpenClaw runtime" in invoked.output
    files = list(reports.iterdir())
    assert len(files) == 4
    output = next(p for p in files if p.suffix == ".json" and not p.name.endswith(".sarif.json"))
    report = json.loads(output.read_text())
    assert report["overall_status"] == "incomplete"
    assert report["overall_passed"] is False
    serialized = json.dumps(report)
    assert "OpenClaw runtime" in serialized
    assert "unsupported" in serialized
