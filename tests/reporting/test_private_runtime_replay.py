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


def test_saved_snapshot_is_the_live_report_for_multiline_diagnostics(tmp_path):
    # Runtime observer errors keep line breaks; redaction must not differ between
    # the live report and the file that offline replay reads.
    result = sample()
    result["runtime_evidence"]["errors"] = [
        "claude-code/with_skill: RuntimeError: census read failed\napi_key: placeholder\n"
        "rerun with --keep-jobs to inspect trials"
    ]
    report = api().project_plugin_runtime(result, provenance={"plugin": "fixture"})
    io_api().save_plugin_runtime_report(report, tmp_path)
    saved = (tmp_path / "plugin_runtime_report.json").read_text(encoding="utf-8")
    assert json.loads(saved) == report.payload()
    assert io_api().load_plugin_runtime_report(tmp_path).payload() == report.payload()
    assert "census read failed" in saved and "placeholder" not in saved


def _bounded_rows_payload():
    payload = api().project_plugin_runtime(sample()).payload()
    remaining = 257
    for section in payload["sections"]:
        count = min(64, remaining)
        remaining -= count
        section["rows"] = [["x"] * len(section["columns"]) for _row in range(count)]
        section["total"], section["omitted"] = count, 0
    return payload


@pytest.mark.parametrize(
    ("change", "reason"),
    [
        ("completeness_mismatch", "Inconsistent runtime report completeness"),
        ("omitted_negative", "Invalid runtime report omission counts"),
        ("total_mismatch", "Invalid runtime report omission counts"),
        ("extra_field", "Unexpected runtime report fields"),
        ("truncated_shape", "Invalid runtime report truncated"),
        ("row_bound", "Runtime report row bound exceeded"),
    ],
)
def test_replay_refuses_snapshots_that_misstate_completeness_or_omissions(tmp_path, change, reason):
    payload = _bounded_rows_payload() if change == "row_bound" else api().project_plugin_runtime(sample()).payload()
    run = payload["sections"][0]
    if change == "completeness_mismatch":
        payload["status"] = "complete"
    elif change == "omitted_negative":
        run["omitted"], run["total"] = -1, len(run["rows"]) - 1
    elif change == "total_mismatch":
        run["total"] += 1
    elif change == "extra_field":
        payload["unexpected"] = "x"
    elif change == "truncated_shape":
        payload["truncated"] = "yes"
    (tmp_path / "plugin_runtime_report.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match=f"unavailable: {reason}"):
        io_api().load_plugin_runtime_report(tmp_path)


@pytest.mark.parametrize(
    "content",
    ['{"schema_version": NaN}', "[" * 101, '"unterminated', '"' + "x" * (256 * 1024) + '"'],
    ids=["non-finite", "too-deep", "malformed", "oversized"],
)
def test_replay_rejects_invalid_or_unbounded_json(tmp_path, content):
    (tmp_path / "plugin_runtime_report.json").write_text(content)
    with pytest.raises(ValueError):
        io_api().load_plugin_runtime_report(tmp_path)


def test_replay_rejects_snapshot_that_exceeds_byte_bound_after_sanitization(tmp_path):
    payload = api().project_plugin_runtime(sample()).payload()
    for section in payload["sections"]:
        section["rows"] = []
        section["total"] = section["omitted"] = 0
        if section["key"] in {"comparisons", "context", "cost", "loading"}:
            section["rows"] = [["\u202e" * 40 for _column in section["columns"]] for _row in range(64)]
            section["total"] = 64
    encoded = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    assert len(encoded) < 256 * 1024
    (tmp_path / "plugin_runtime_report.json").write_bytes(encoded)

    with pytest.raises(ValueError, match=r"unavailable.*publication bound"):
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
