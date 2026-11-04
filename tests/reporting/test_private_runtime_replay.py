# SPDX-License-Identifier: Apache-2.0
"""Retained reports replay without a source tree or runtime dependencies."""

from __future__ import annotations

import importlib
import json
import os

import pytest
from test_private_runtime_projection import api, sample


def io_api():
    return importlib.import_module("skillevaluator.reporting.plugin_runtime_io")


def test_snapshot_round_trip_is_self_contained(tmp_path):
    report = api().project_plugin_runtime(sample(), provenance={"plugin": "fixture"})
    io_api().save_plugin_runtime_report(report, tmp_path)
    assert io_api().load_plugin_runtime_report(tmp_path).payload() == report.payload()
    assert sorted(path.name for path in tmp_path.iterdir()) == ["plugin_runtime_report.json"]


def test_missing_older_snapshot_has_explicit_unavailable_error(tmp_path):
    with pytest.raises(ValueError, match="unavailable"):
        io_api().load_plugin_runtime_report(tmp_path)


@pytest.mark.parametrize("kind", ["root_link", "file_link", "hardlink", "directory"])
def test_replay_refuses_unsafe_selected_paths(tmp_path, kind):
    real = tmp_path / "real"
    real.mkdir()
    report = api().project_plugin_runtime(sample())
    io_api().save_plugin_runtime_report(report, real)
    target = real / "plugin_runtime_report.json"
    root = real
    if kind == "root_link":
        root = tmp_path / "linked"
        root.symlink_to(real, target_is_directory=True)
    elif kind == "file_link":
        target.rename(real / "original.json")
        target.symlink_to(real / "original.json")
    elif kind == "hardlink":
        os.link(target, real / "alias.json")
    else:
        target.unlink()
        target.mkdir()
    with pytest.raises(ValueError):
        io_api().load_plugin_runtime_report(root)


@pytest.mark.parametrize(
    "change", ["version", "columns", "row_width", "status", "status_shape", "status_object", "unknown_section"]
)
def test_replay_refuses_incompatible_snapshot_schema(tmp_path, change):
    payload = api().project_plugin_runtime(sample()).payload()
    if change == "version":
        payload["schema_version"] = 999
    elif change == "columns":
        payload["sections"][0]["columns"] = ["unexpected"]
    elif change == "row_width":
        payload["sections"][0]["rows"][0].append("extra")
    elif change == "status":
        payload["status"] = "passed"
    elif change == "status_shape":
        payload["status"] = []
    elif change == "status_object":
        payload["status"] = {}
    else:
        payload["sections"][0]["key"] = "unknown"
    (tmp_path / "plugin_runtime_report.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="unavailable"):
        io_api().load_plugin_runtime_report(tmp_path)


@pytest.mark.parametrize(
    "content", ['{"schema_version": NaN}', "[" * 101, '"unterminated', '"' + "x" * (256 * 1024) + '"']
)
def test_replay_rejects_invalid_or_unbounded_json(tmp_path, content):
    (tmp_path / "plugin_runtime_report.json").write_text(content)
    with pytest.raises(ValueError):
        io_api().load_plugin_runtime_report(tmp_path)


@pytest.mark.parametrize("value", [True, 2**100, {"unexpected": "nested"}])
def test_replay_refuses_cells_outside_the_display_schema(tmp_path, value):
    payload = api().project_plugin_runtime(sample()).payload()
    payload["sections"][0]["rows"][0][1] = value
    (tmp_path / "plugin_runtime_report.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        io_api().load_plugin_runtime_report(tmp_path)


def test_replay_refuses_unrecognized_section_fields(tmp_path):
    payload = api().project_plugin_runtime(sample()).payload()
    payload["sections"][0]["unexpected"] = "unchecked data"
    (tmp_path / "plugin_runtime_report.json").write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        io_api().load_plugin_runtime_report(tmp_path)
