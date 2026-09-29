# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin provenance survives re-rendering a run from its run directory."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import pytest
from _plugin_fixtures import provenance, write_run_dir

from skillevaluator.evaluation import tier3_report
from skillevaluator.evaluation.tier3_report import (
    _MAX_PLUGIN_PROVENANCE_BYTES,
    _read_plugin_provenance,
    agent_eval_result_from_directory,
)

posix_only = pytest.mark.skipif(sys.platform == "win32", reason="POSIX link semantics")


@pytest.fixture(autouse=True)
def _no_llm_insights(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "skillevaluator.evaluation.insights_judge.build_insights",
        lambda *_args, **_kwargs: {"conclusions": [], "recommendations": []},
    )


def _run(tmp_path: Path, **kwargs) -> tuple[Path, Path]:
    plugin_dir = tmp_path / "demo-plugin"
    plugin_dir.mkdir(exist_ok=True)
    return plugin_dir, write_run_dir(tmp_path / "results" / "20260101_000000", **kwargs)


def test_rerender_reads_sidecar_and_keeps_incomplete_status(tmp_path: Path) -> None:
    plugin_dir, run_dir = _run(tmp_path, sidecar=provenance(partial=True))

    result = agent_eval_result_from_directory(plugin_dir, run_dir, use_llm_judge=False)

    assert result is not None
    payload = result.metadata["agent_eval"]
    assert payload["plugin_provenance"]["partial"] is True
    assert payload["plugin_provenance"]["component_coverage"]["not_evaluated"] == 2
    assert result.passed is False
    assert result.metadata["skip_reason"].startswith("INCOMPLETE:")
    assert payload["conclusions"][0]["title"] == "Evaluation INCOMPLETE - unresolved dependencies"


def test_explicit_provenance_wins_over_sidecar(tmp_path: Path) -> None:
    plugin_dir, run_dir = _run(tmp_path, sidecar=provenance(partial=True))
    explicit = provenance(partial=False)

    result = agent_eval_result_from_directory(plugin_dir, run_dir, plugin_provenance=explicit, use_llm_judge=False)

    assert result is not None
    assert result.metadata["agent_eval"]["plugin_provenance"]["partial"] is False
    assert result.passed is True


def test_run_without_sidecar_renders_as_before(tmp_path: Path) -> None:
    plugin_dir, run_dir = _run(tmp_path, signals=False)

    result = agent_eval_result_from_directory(plugin_dir, run_dir, use_llm_judge=False)

    assert result is not None
    assert "plugin_provenance" not in result.metadata["agent_eval"]
    assert _read_plugin_provenance(run_dir) == {}


@posix_only
def test_symlinked_sidecar_is_ignored(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    plugin_dir, run_dir = _run(tmp_path)
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps(provenance(partial=True)), encoding="utf-8")
    (run_dir / "plugin_provenance.json").symlink_to(outside)

    with caplog.at_level(logging.WARNING, logger=tier3_report.__name__):
        assert _read_plugin_provenance(run_dir) == {}
        result = agent_eval_result_from_directory(plugin_dir, run_dir, use_llm_judge=False)

    assert result is not None
    assert "plugin_provenance" not in result.metadata["agent_eval"]
    assert "symlink" in caplog.text


@posix_only
def test_hardlinked_sidecar_is_rejected(tmp_path: Path) -> None:
    _plugin_dir, run_dir = _run(tmp_path)
    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps(provenance(partial=True)), encoding="utf-8")
    os.link(outside, run_dir / "plugin_provenance.json")

    assert _read_plugin_provenance(run_dir) == {}


def test_directory_sidecar_is_rejected(tmp_path: Path) -> None:
    _plugin_dir, run_dir = _run(tmp_path)
    (run_dir / "plugin_provenance.json").mkdir()

    assert _read_plugin_provenance(run_dir) == {}


def test_oversize_sidecar_is_ignored(tmp_path: Path) -> None:
    _plugin_dir, run_dir = _run(tmp_path)
    record = provenance(partial=True)
    record["padding"] = "x" * (_MAX_PLUGIN_PROVENANCE_BYTES + 1)
    (run_dir / "plugin_provenance.json").write_text(json.dumps(record), encoding="utf-8")

    assert _read_plugin_provenance(run_dir) == {}


@pytest.mark.parametrize(
    "raw",
    [
        b"{not json",
        b"[1, 2, 3]",
        b'"a string"',
        b"\xff\xfe\x00 not utf-8",
        b"[" * 5000 + b"]" * 5000,
    ],
)
def test_malformed_sidecar_is_ignored(tmp_path: Path, raw: bytes) -> None:
    _plugin_dir, run_dir = _run(tmp_path)
    (run_dir / "plugin_provenance.json").write_bytes(raw)

    assert _read_plugin_provenance(run_dir) == {}


def test_mistyped_sidecar_fields_fail_closed(tmp_path: Path) -> None:
    _plugin_dir, run_dir = _run(tmp_path)
    (run_dir / "plugin_provenance.json").write_text(
        json.dumps(
            {
                "plugin_name": ["not", "a", "string"],
                "evaluated_member_skills": ["loader", 7],
                "unresolved_skill_refs": "github::org/repo::skills::remote",
                "dataset_case_count": -3,
                "partial": "no",
                "component_coverage": ["not", "a", "mapping"],
            }
        ),
        encoding="utf-8",
    )

    loaded = _read_plugin_provenance(run_dir)

    assert loaded["partial"] is True
    assert loaded["evaluated_member_skills"] == ["loader"]
    assert "plugin_name" not in loaded
    assert "unresolved_skill_refs" not in loaded
    assert "dataset_case_count" not in loaded
    assert "component_coverage" not in loaded


def test_sidecar_partial_flag_cannot_hide_deferred_components(tmp_path: Path) -> None:
    _plugin_dir, run_dir = _run(tmp_path)
    record = provenance(partial=True)
    record["partial"] = False
    (run_dir / "plugin_provenance.json").write_text(json.dumps(record), encoding="utf-8")

    assert _read_plugin_provenance(run_dir)["partial"] is True
