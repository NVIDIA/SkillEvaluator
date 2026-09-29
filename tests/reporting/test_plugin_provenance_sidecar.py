# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Plugin provenance survives re-rendering a run, and the delivered report shows it."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from _plugin_fixtures import element_text, provenance, write_run_dir
from click.testing import CliRunner

from skillevaluator import cli as cli_module
from skillevaluator.evaluation import EvaluationService, tier3_report
from skillevaluator.evaluation.tier3_report import (
    _MAX_PLUGIN_PROVENANCE_BYTES,
    _read_plugin_provenance,
    agent_eval_result_from_directory,
    refresh_plugin_run_report,
    render_agent_eval_html_report,
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


def test_standalone_render_round_trip_preserves_incomplete_and_coverage(tmp_path: Path) -> None:
    plugin_dir, run_dir = _run(tmp_path, sidecar=provenance(partial=True))

    report = render_agent_eval_html_report(plugin_dir, run_dir, use_llm_judge=False)
    html = report.read_text(encoding="utf-8")

    assert "Plugin under validation" in html
    assert element_text(html, "tier3-plugin-incomplete") is not None
    assert "INCOMPLETE" in (element_text(html, "tier3-plugin-incomplete") or "")
    assert '<span class="tier-card-verdict">INCOMPLETE</span>' in html
    coverage = element_text(html, "tier3-plugin-coverage") or ""
    assert "2 components not evaluated" in coverage
    completeness = element_text(html, "tier3-plugin-completeness") or ""
    assert "github::org/repo::skills::remote" in completeness


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


def test_refresh_replaces_the_runner_report_with_provenance(tmp_path: Path) -> None:
    plugin_dir, run_dir = _run(tmp_path, sidecar=provenance(partial=True))
    (run_dir / "report.html").write_text("<html>runner report without provenance</html>", encoding="utf-8")

    refreshed = refresh_plugin_run_report(plugin_dir, run_dir, use_llm_judge=False)

    assert refreshed == run_dir.resolve() / "report.html"
    html = refreshed.read_text(encoding="utf-8")
    assert element_text(html, "tier3-plugin-incomplete") is not None
    assert element_text(html, "tier3-plugin-coverage") is not None


def test_refresh_is_best_effort(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    missing = tmp_path / "missing-run"

    with caplog.at_level(logging.WARNING, logger=tier3_report.__name__):
        assert refresh_plugin_run_report(tmp_path, missing, use_llm_judge=False) is None


def _prepared_package(tmp_path: Path, record: dict) -> SimpleNamespace:
    package_path = tmp_path / "package"
    package_path.mkdir(exist_ok=True)
    return SimpleNamespace(
        skipped=False,
        skip_reason=None,
        package_path=package_path,
        include_skills=(),
        unresolved_skill_refs=tuple(record["unresolved_skill_refs"]),
        unresolved_rule_refs=(),
        unresolved_mcp_servers=tuple(record["provider_only_mcp_servers"]),
        integration_evidence_error=lambda: None,
        provenance=lambda: dict(record),
    )


def test_evaluate_plugin_delivers_report_with_provenance(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    plugin_dir = tmp_path / "demo-plugin"
    plugin_dir.mkdir()
    run_dir = tmp_path / "results" / "demo-plugin" / "20260101_000000"
    record = provenance(partial=True)

    def evaluate(_self, _options, **_kwargs) -> dict:
        # The runner writes its report before the CLI persists the sidecar.
        write_run_dir(run_dir)
        (run_dir / "report.html").write_text("<html>runner report without provenance</html>", encoding="utf-8")
        return {"run_dir": str(run_dir), "execution_status": "succeeded"}

    monkeypatch.setattr(
        "skillevaluator.tier3.plugin_eval.prepare_plugin_eval_package",
        lambda *_a, **_k: _prepared_package(tmp_path, record),
    )
    monkeypatch.setattr(EvaluationService, "evaluate", evaluate)
    monkeypatch.setattr(EvaluationService, "failure_reason", staticmethod(lambda _result: None))
    monkeypatch.setattr("skillevaluator.tier3.result_display.render_evaluation_result", lambda *_a, **_k: None)

    outcome = CliRunner().invoke(
        cli_module.cli,
        ["tier3", "evaluate-plugin", str(plugin_dir), "--lift-mode", "effectiveness", "--progress", "off"],
    )

    # A partial run is still reported as INCOMPLETE on the command line ...
    assert outcome.exit_code != 0
    assert "INCOMPLETE" in outcome.output
    # ... and the delivered report now carries the provenance, coverage and status.
    assert json.loads((run_dir / "plugin_provenance.json").read_text(encoding="utf-8"))["partial"] is True
    html = (run_dir / "report.html").read_text(encoding="utf-8")
    assert "runner report without provenance" not in html
    assert element_text(html, "tier3-plugin-incomplete") is not None
    assert "2 components not evaluated" in (element_text(html, "tier3-plugin-coverage") or "")


def test_validate_plugin_path_refreshes_the_run_report(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    plugin_dir = tmp_path / "demo-plugin"
    plugin_dir.mkdir()
    run_dir = tmp_path / "results" / "20260101_000000"
    record = provenance(partial=True)

    def evaluate(_self, _options, **_kwargs) -> dict:
        write_run_dir(run_dir)
        (run_dir / "report.html").write_text("<html>runner report without provenance</html>", encoding="utf-8")
        return {"run_dir": str(run_dir), "execution_status": "succeeded"}

    def from_run(skill_path, **kwargs):
        return agent_eval_result_from_directory(
            skill_path,
            run_dir,
            plugin_provenance=kwargs.get("plugin_provenance"),
            use_llm_judge=False,
        )

    monkeypatch.setattr(
        "skillevaluator.tier3.plugin_eval.prepare_plugin_eval_package",
        lambda *_a, **_k: _prepared_package(tmp_path, record),
    )
    monkeypatch.setattr(EvaluationService, "evaluate", evaluate)
    monkeypatch.setattr(EvaluationService, "failure_reason", staticmethod(lambda _result: None))
    monkeypatch.setattr("skillevaluator.evaluation.tier3_report.agent_eval_result_from_run", from_run)

    result = cli_module._run_agent_eval_or_skip(
        plugin_dir,
        agents="codex",
        env_mode="docker",
        skip_baseline=False,
        n_concurrent=1,
        max_agents=1,
        kind="plugin",
        lift_mode="effectiveness",
    )

    assert result.metadata["skip_reason"].startswith("INCOMPLETE:")
    html = (run_dir / "report.html").read_text(encoding="utf-8")
    assert element_text(html, "tier3-plugin-incomplete") is not None
